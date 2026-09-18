"""GRU memory, episode masking and actual PPO updates, without Isaac Sim."""
from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path
import sys
import tempfile
import types
import unittest

import torch
from gymnasium.spaces import Box
from skrl.memories.torch import RandomMemory

ROOT = Path(__file__).resolve().parents[1]
EMMA = ROOT / "source/emma_rl/emma_rl/tasks/direct/emma"
package = types.ModuleType("_emma_gru_test")
package.__path__ = [str(EMMA / "emma_obs_nav")]
sys.modules[package.__name__] = package
network = importlib.import_module(package.__name__ + ".emma_obs_nav_gru_network")
spec = importlib.util.spec_from_file_location("emma_recurrent_runner", EMMA / "utils/recurrent_runner.py")
integration = importlib.util.module_from_spec(spec)
spec.loader.exec_module(integration)


def model(num_envs=3, length=8):
    torch.manual_seed(18)
    m = network.EmmaNavGRU(
        Box(-10, 10, (134,)), None, Box(-1, 1, (2,)),
        {"policy": {"min_concentration": 1., "initial_concentration": 2., "action_eps": 1e-6},
         "value": {"clip_actions": False},
         "recurrent": {"hidden_size": 16, "num_layers": 1, "sequence_length": length, "num_envs": num_envs}},
        "cpu",
    )
    m.init_state_dict(role="policy")
    m.init_state_dict(role="value")
    return m


def agent(num_envs=3, length=4):
    m = model(num_envs, length)
    a = integration.EmmaRecurrentPPO(
        models={"policy": m, "value": m},
        memory=RandomMemory(memory_size=8, num_envs=num_envs, device="cpu"),
        observation_space=m.observation_space, state_space=None, action_space=m.action_space,
        device="cpu", cfg={"rollouts": 8, "mini_batches": 2, "learning_epochs": 1,
                           "time_limit_bootstrap": False,
                           "experiment": {"write_interval": 0, "checkpoint_interval": 0}},
    )
    a.init()
    return a


class GRUPolicyTests(unittest.TestCase):
    def test_history_changes_policy_and_value_for_same_current_observation(self):
        m = model()
        obs = torch.randn(3, 134)
        _, history = m.compute({"observations": torch.randn(3, 134)}, "policy")
        _, empty = m.act({"observations": obs}, "policy")
        _, remembered = m.act({"observations": obs, "rnn": history['rnn']}, "policy")
        self.assertFalse(torch.allclose(empty['mean_actions'], remembered['mean_actions'], atol=1e-8))
        v0, _ = m.compute({"observations": obs}, "value")
        v1, _ = m.compute({"observations": obs, "rnn": history['rnn']}, "value")
        self.assertFalse(torch.allclose(v0, v1))
        self.assertEqual(remembered['rnn'][0].shape, (1, 3, 16))

    def test_sequence_training_matches_step_inference_with_mixed_episode_ends(self):
        m = model()
        batch, length = 3, 8
        obs = torch.randn(batch, length, 134)
        term = torch.zeros(batch, length, 1, dtype=torch.bool)
        trunc = torch.zeros_like(term)
        term[0, 2] = True
        trunc[1, 4] = True
        term[2, -1] = True
        hidden = torch.randn(1, batch, 16)
        states, alphas, values = [], [], []
        for t in range(length):
            states.append(hidden.clone())
            inputs = {"observations": obs[:, t], "rnn": [hidden]}
            alpha, output = m.compute(inputs, "policy")
            value, _ = m.compute(inputs, "value")
            alphas.append(alpha)
            values.append(value)
            hidden = output['rnn'][0] * (~(term[:, t] | trunc[:, t])).float()[None]
        sequence_inputs = {"observations": obs.reshape(-1, 134),
                           "rnn": [torch.stack(states, dim=2).reshape(1, -1, 16)],
                           "terminated": term.reshape(-1, 1), "truncated": trunc.reshape(-1, 1)}
        alpha, output = m.compute(sequence_inputs, "policy")
        value, _ = m.compute(sequence_inputs, "value")
        torch.testing.assert_close(alpha.reshape(batch, length, 2), torch.stack(alphas, dim=1))
        torch.testing.assert_close(value.reshape(batch, length, 1), torch.stack(values, dim=1))
        torch.testing.assert_close(output['rnn'][0], hidden)

    def test_gradient_reaches_history_but_stops_at_episode_boundary(self):
        m = model(num_envs=1)
        obs = torch.randn(8, 134, requires_grad=True)
        term = torch.zeros(8, 1, dtype=torch.bool)
        term[2] = True
        values, _ = m.compute({"observations": obs, "rnn": [torch.zeros(1, 8, 16)],
                               "terminated": term}, "value")
        values[-1].backward()
        self.assertEqual(obs.grad[:3].abs().sum().item(), 0)
        self.assertGreater(obs.grad[3:7].abs().sum().item(), 0)
        self.assertGreater(m.gru.weight_hh_l0.grad.abs().sum().item(), 0)

    def test_value_bootstrap_does_not_reuse_policy_features(self):
        m = model()
        first, second = torch.randn(3, 134), torch.randn(3, 134)
        expected, _ = m.compute({"observations": second}, "value")
        m.act({"observations": first}, "policy")
        actual, _ = m.compute({"observations": second}, "value")
        torch.testing.assert_close(actual, expected)

    def test_log_prob_replay_and_action_bounds(self):
        m = model()
        inputs = {"observations": torch.randn(3, 134), "rnn": [torch.randn(1, 3, 16)]}
        actions, outputs = m.act(inputs, "policy")
        _, replay = m.act({**inputs, "taken_actions": actions}, "policy")
        torch.testing.assert_close(outputs['log_prob'], replay['log_prob'])
        self.assertTrue(((actions > -1) & (actions < 1)).all())
        self.assertTrue(torch.isfinite(m.get_entropy()).all())

    def transition(self, a, obs, terminated, truncated, step):
        with torch.no_grad():
            actions, outputs = a.act(obs, None, timestep=step, timesteps=16)
            next_hidden = outputs['rnn'][0].clone()
            a.record_transition(
                observations=obs, states=None, actions=actions, rewards=torch.ones(3, 1),
                next_observations=torch.randn_like(obs), next_states=None,
                terminated=terminated, truncated=truncated, infos={}, timestep=step, timesteps=16)
        return next_hidden

    def test_playback_commits_memory_and_resets_only_finished_environments(self):
        a = agent()
        a.enable_training_mode(False, apply_to_models=True)
        obs = torch.randn(3, 134)
        term = torch.tensor([[True], [False], [False]])
        trunc = torch.tensor([[False], [True], [False]])
        next_hidden = self.transition(a, obs, term, trunc, 0)
        h = a._rnn_initial_states['policy'][0]
        self.assertEqual(h[:, :2].abs().sum().item(), 0)
        torch.testing.assert_close(h[:, 2], next_hidden[:, 2])
        self.assertGreater(h[:, 2].abs().sum().item(), 0)
        self.assertEqual(len(a.memory), 0)

    def test_two_ppo_updates_keep_memory_and_checkpoint_weights(self):
        a = agent()
        a.enable_training_mode(True)
        before = a.policy.gru.weight_hh_l0.detach().clone()
        obs = torch.randn(3, 134)
        zeros = torch.zeros(3, 1, dtype=torch.bool)
        for step in range(16):
            term, trunc = zeros.clone(), zeros.clone()
            if step % 5 == 2:
                trunc[0] = True
            if step % 7 == 3:
                term[1] = True
            self.transition(a, obs, term, trunc, step)
            if trunc[0]:
                self.assertTrue(a.memory.get_tensor_by_name('terminated')[step % 8, 0])
            hidden = a._rnn_initial_states['policy'][0].clone()
            a.post_interaction(timestep=step, timesteps=16)
            torch.testing.assert_close(a._rnn_initial_states['policy'][0], hidden)
            obs = torch.randn_like(obs)
        self.assertFalse(torch.equal(before, a.policy.gru.weight_hh_l0))
        self.assertTrue(all(torch.isfinite(p).all() for p in a.policy.parameters()))
        with tempfile.TemporaryDirectory() as d:
            path = str(Path(d) / 'agent.pt')
            a.save(path)
            restored = agent()
            restored.load(path)
            for p, q in zip(a.policy.parameters(), restored.policy.parameters()):
                torch.testing.assert_close(p, q)
            self.assertEqual(restored._rnn_initial_states['policy'][0].abs().sum().item(), 0)

    def test_invalid_sequence_rejected(self):
        m = model()
        with self.assertRaises(ValueError):
            m.compute({'observations': torch.zeros(7, 134), 'rnn': [torch.zeros(1, 7, 16)],
                       'terminated': torch.zeros(7, 1, dtype=torch.bool)}, 'policy')

    def test_minibatches_preserve_environment_and_time_order(self):
        memory = RandomMemory(memory_size=8, num_envs=3, device='cpu')
        memory.create_tensor(name='index', size=1)
        for step in range(8):
            memory.add_samples(index=(torch.arange(3)*1000+step).float()[:, None])
        seen = []
        for batch, in memory.sample_all(names=['index'], mini_batches=2, sequence_length=4):
            sequences = batch.reshape(-1, 4)
            torch.testing.assert_close(sequences[:, 1:]-sequences[:, :-1], torch.ones_like(sequences[:, 1:]))
            self.assertTrue(((sequences // 1000) == (sequences[:, :1] // 1000)).all())
            seen.extend(batch.flatten().tolist())
        self.assertEqual(len(set(seen)), 24)

    def test_runner_rejects_split_sequences(self):
        runner = integration.EmmaRunner.__new__(integration.EmmaRunner)
        env = types.SimpleNamespace(num_envs=3)
        m = model(length=8)
        with self.assertRaisesRegex(ValueError, 'complete sequences'):
            runner._generate_agent(env, {'agent': {'class': 'PPO', 'rollouts': 8, 'mini_batches': 4}},
                                   {'agent': {'policy': m, 'value': m}})

    def test_evaluation_restores_saved_architecture_for_single_environment(self):
        import yaml
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root/'params').mkdir()
            saved_models = {'name': 'EmmaNavGRU', 'recurrent': {'hidden_size': 64, 'sequence_length': 32}}
            (root/'params/agent.yaml').write_text(yaml.safe_dump({'models': saved_models}))
            cfg = {'models': {'name': 'EmmaNavGRU'}, 'agent': {'mini_batches': 128}, 'memory': {}}
            integration.prepare_recurrent_evaluation(cfg, root/'checkpoints/agent.pt')
            self.assertEqual(cfg['models'], saved_models)
            self.assertEqual((cfg['agent']['mini_batches'], cfg['agent']['rollouts']), (1, 32))

    def test_mlp_control_uses_same_timeout_boundary_and_still_updates(self):
        m = network.EmmaNavShared(
            Box(-10, 10, (134,)), None, Box(-1, 1, (2,)),
            {'policy': {'min_concentration': 1., 'initial_concentration': 2., 'action_eps': 1e-6},
             'value': {'clip_actions': False}}, 'cpu')
        m.init_state_dict(role='policy')
        m.init_state_dict(role='value')
        a = integration.EmmaPPO(
            models={'policy': m, 'value': m},
            memory=RandomMemory(memory_size=4, num_envs=3, device='cpu'),
            observation_space=m.observation_space, state_space=None, action_space=m.action_space,
            device='cpu', cfg={'rollouts': 4, 'mini_batches': 2, 'learning_epochs': 1,
                               'time_limit_bootstrap': False,
                               'experiment': {'write_interval': 0, 'checkpoint_interval': 0}})
        a.init()
        a.enable_training_mode(True)
        before = m.net[0].weight.detach().clone()
        obs = torch.randn(3, 134)
        for step in range(8):
            with torch.no_grad():
                actions, _ = a.act(obs, None, timestep=step, timesteps=8)
                term = torch.zeros(3, 1, dtype=torch.bool)
                trunc = torch.tensor([[True], [False], [False]])
                a.record_transition(
                    observations=obs, states=None, actions=actions, rewards=torch.randn(3, 1),
                    next_observations=torch.randn_like(obs), next_states=None,
                    terminated=term, truncated=trunc, infos={}, timestep=step, timesteps=8)
                self.assertTrue(a.memory.get_tensor_by_name('terminated')[step % 4, 0])
                self.assertFalse(a.memory.get_tensor_by_name('terminated')[step % 4, 1])
            a.post_interaction(timestep=step, timesteps=8)
            obs = torch.randn_like(obs)
        self.assertFalse(torch.equal(before, m.net[0].weight))


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()

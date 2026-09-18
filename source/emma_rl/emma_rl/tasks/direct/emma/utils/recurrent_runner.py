"""Use skrl's recurrent PPO when an Emma model declares recurrent state."""
from __future__ import annotations

from skrl.agents.torch.ppo import PPO, PPO_RNN
from skrl.utils.runner.torch import Runner


def prepare_recurrent_evaluation(cfg, checkpoint):
    """Restore the saved GRU architecture and use one complete evaluation sequence."""
    if cfg["models"].get("name") != "EmmaNavGRU":
        return
    from pathlib import Path
    import copy
    import yaml
    saved_config = Path(checkpoint).resolve().parent.parent / "params" / "agent.yaml"
    if saved_config.is_file():
        with saved_config.open(encoding="utf-8") as stream:
            saved = yaml.safe_load(stream)
        if saved["models"].get("name") != "EmmaNavGRU":
            raise ValueError("This checkpoint is not a GRU policy; select its original task")
        cfg["models"] = copy.deepcopy(saved["models"])
    # No PPO updates run in evaluation. This also allows num_envs=1 without
    # splitting its short history into the training configuration's minibatches.
    cfg["agent"]["rollouts"] = cfg["models"]["recurrent"]["sequence_length"]
    cfg["agent"]["mini_batches"] = 1
    cfg["memory"]["memory_size"] = -1


class _EpisodeBoundary:
    def record_transition(self, *, terminated, truncated, **kwargs):
        # In skrl 2.0 PPO/PPO_RNN, GAE masks only `terminated`. Without timeout
        # bootstrapping, a timeout must also stop GAE, or rewards from the new
        # auto-reset episode leak into the preceding episode's advantages.
        if not self.cfg.time_limit_bootstrap:
            terminated = terminated | truncated
        return super().record_transition(terminated=terminated, truncated=truncated, **kwargs)


class EmmaPPO(_EpisodeBoundary, PPO):
    """Feedforward control with the same episode-boundary semantics as GRU."""


class EmmaRecurrentPPO(_EpisodeBoundary, PPO_RNN):
    """Recurrent PPO with independent episode returns."""


class EmmaRunner(Runner):
    def _generate_agent(self, env, cfg, models):
        recurrent = models["agent"]["policy"].get_specification().get("rnn") if "agent" in models else None
        self._emma_recurrent = bool(recurrent)
        if recurrent:
            import skrl
            from packaging.version import Version
            if Version(skrl.__version__) < Version("2.0.0"):
                raise RuntimeError("The Emma GRU baseline requires skrl >= 2.0.0")
            if cfg["agent"]["class"].lower() != "ppo":
                raise ValueError("The Emma GRU baseline supports PPO only")
            if cfg["agent"].get("time_limit_bootstrap", False):
                raise ValueError("The GRU baseline uses time_limit_bootstrap=False for auto-reset episodes")
            length = recurrent["sequence_length"]
            rollouts = cfg["agent"]["rollouts"]
            batches = cfg["agent"]["mini_batches"]
            samples = env.num_envs * rollouts
            if rollouts % length or batches < 1 or samples % batches or (samples // batches) % length:
                raise ValueError(
                    "GRU requires rollouts divisible by sequence_length and equal minibatches "
                    "of complete sequences. Reduce mini_batches for small num_envs. "
                    f"Got num_envs={env.num_envs}, rollouts={rollouts}, "
                    f"mini_batches={batches}, sequence_length={length}")
            if cfg.get("memory", {}).get("memory_size", -1) not in (-1, rollouts):
                raise ValueError("GRU memory_size must be -1 or equal to rollouts")
            skrl.logger.info(
                f"Emma PPO+GRU: PPO_RNN, sequence_length={length}, "
                f"hidden_state={recurrent['sizes']}, mini_batches={batches}")
        return super()._generate_agent(env, cfg, models)

    def _component(self, name):
        if name.lower() == "ppo":
            if getattr(self, "_emma_recurrent", False):
                return EmmaRecurrentPPO
            import skrl
            from packaging.version import Version
            if Version(skrl.__version__) >= Version("2.0.0"):
                return EmmaPPO
        return super()._component(name)

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""
Script to play a checkpoint of an RL agent from skrl.

Visit the skrl documentation (https://skrl.readthedocs.io) to see the examples structured in
a more user-friendly way.
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import csv

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Play a checkpoint of an RL agent from skrl.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")
parser.add_argument(
    "--seed", type=int, default=None,
    help="Random seed for playback and map generation; omitted means a fresh seed each launch.",
)
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument(
    "--ml_framework",
    type=str,
    default="torch",
    choices=["torch", "jax", "jax-numpy"],
    help="The ML framework used for training the skrl agent.",
)
parser.add_argument(
    "--algorithm",
    type=str,
    default="PPO",
    choices=["PPO"],
    help="The maintained training algorithm.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument(
    "--policy-trace",
    "--policy_trace",
    dest="policy_trace",
    type=str,
    default=None,
    help="Save raw observations and deterministic PT actions to an .npz trace.",
)
parser.add_argument(
    "--policy-trace-steps",
    "--policy_trace_steps",
    dest="policy_trace_steps",
    type=int,
    default=4000,
    help="Number of control steps to record when --policy-trace is set (default: 4000).",
)
parser.add_argument(
    "--policy-trace-env",
    "--policy_trace_env",
    dest="policy_trace_env",
    type=int,
    default=0,
    help="Environment index to record when --policy-trace is set (default: 0).",
)
parser.add_argument(
    "--action-plot-episodes",
    "--action_plot_episodes",
    dest="action_plot_episodes",
    type=int,
    default=-1,
    help=(
        "Save one action/command/response PNG and CSV for each episode "
        "(-1 records all episodes, default: -1; 0 disables; N>0 limits "
        "recording to the first N episodes)."
    ),
)
parser.add_argument(
    "--fixed-wheel-cmd",
    "--fixed_wheel_cmd",
    dest="fixed_wheel_cmd",
    type=float,
    nargs=2,
    metavar=("LEFT_RADPS", "RIGHT_RADPS"),
    default=None,
    help=(
        "Bypass policy inference and send constant left/right wheel velocity "
        "commands in rad/s through the normal env.step path."
    ),
)
parser.add_argument(
    "--fixed-wheel-steps",
    "--fixed_wheel_steps",
    dest="fixed_wheel_steps",
    type=int,
    default=150,
    help="Number of env.step calls for --fixed-wheel-cmd (default: 150).",
)
parser.add_argument(
    "--lidar-forward-half-fov-deg",
    "--lidar_forward_half_fov_deg",
    dest="lidar_forward_half_fov_deg",
    type=float,
    default=180.0,
    help=(
        "Policy-observation LiDAR half-FOV around robot forward in degrees "
        "(default: 180, matching the real 380-degree sensor; use 180 for 360 degrees)."
    ),
)
parser.add_argument(
    "--lidar-blind-fill-clearance",
    "--lidar_blind_fill_clearance",
    dest="lidar_blind_fill_clearance",
    type=float,
    default=1.0,
    help=(
        "Clearance in metres used to fill LiDAR sectors outside the play FOV "
        "(default: 1.0)."
    ),
)
parser.add_argument(
    "--debug-lidar-fov-fill",
    "--debug_lidar_fov_fill",
    dest="debug_lidar_fov_fill",
    action="store_true",
    help=(
        "Print a one-shot comparison of the LiDAR clearance before/after "
        "partial-FOV filling at representative body-frame angles."
    ),
)
parser.add_argument(
    "--no-motion-acceleration-limits",
    "--no_motion_acceleration_limits",
    dest="no_motion_acceleration_limits",
    action="store_true",
    help=(
        "Disable the body-twist command rate limit that the env config enables by "
        "default. Use this to reproduce checkpoints trained before the limit "
        "existed, or to A/B the limit on the same --seed."
    ),
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import hashlib
import numpy as np
import os
from datetime import datetime
from pathlib import Path
import time
import torch

import skrl
from skrl.utils import set_seed
from packaging import version

# check for minimum supported skrl version
SKRL_VERSION = "1.4.2"
if version.parse(skrl.__version__) < version.parse(SKRL_VERSION):
    skrl.logger.error(
        f"Unsupported skrl version: {skrl.__version__}. "
        f"Install supported version using 'pip install skrl>={SKRL_VERSION}'"
    )
    exit()

if args_cli.ml_framework.startswith("torch"):
    from skrl.utils.runner.torch import Runner
elif args_cli.ml_framework.startswith("jax"):
    from skrl.utils.runner.jax import Runner

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

from isaaclab_rl.skrl import SkrlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path, load_cfg_from_registry, parse_env_cfg

import emma_rl.tasks  # noqa: F401
from emma_rl.tasks.direct.emma.utils import generate_custom_model
if args_cli.ml_framework.startswith("torch"):
    from emma_rl.tasks.direct.emma.utils.recurrent_runner import (
        EmmaRunner as Runner, prepare_recurrent_evaluation,
    )

# config shortcuts
algorithm = args_cli.algorithm.lower()


def _sha256(path: str | os.PathLike) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class PolicyTraceRecorder:
    """Collect scaler-input observations and the matching deterministic actions."""

    OBSERVATION_STAGE = "raw_before_running_standard_scaler"

    def __init__(self, path: str, target_steps: int, env_index: int, checkpoint: str, task: str):
        if target_steps <= 0:
            raise ValueError("--policy-trace-steps must be positive")
        if env_index < 0:
            raise ValueError("--policy-trace-env must be non-negative")
        self.path = Path(path).expanduser().resolve()
        self.target_steps = target_steps
        self.env_index = env_index
        self.checkpoint = os.path.abspath(checkpoint)
        self.task = task
        self.observations: list[np.ndarray] = []
        self.actions: list[np.ndarray] = []
        self.episode_ids: list[int] = []
        self.episode_steps: list[int] = []
        self.terminated: list[bool] = []
        self.truncated: list[bool] = []
        self.episode_id = 0
        self.episode_step = 0

    @property
    def complete(self) -> bool:
        return len(self.observations) >= self.target_steps

    def append(
        self,
        observation: torch.Tensor,
        action: torch.Tensor,
        terminated: bool,
        truncated: bool,
    ) -> None:
        if self.complete:
            return
        raw_observation = observation.detach().to(device="cpu", dtype=torch.float32).numpy().copy()
        mean_action = action.detach().to(device="cpu", dtype=torch.float32).numpy().copy()
        if raw_observation.ndim != 1:
            raise ValueError(
                f"A traced observation must be 1-D, got shape {raw_observation.shape}"
            )
        if mean_action.ndim != 1:
            raise ValueError(f"A traced action must be 1-D, got shape {mean_action.shape}")
        if not np.isfinite(raw_observation).all():
            raise ValueError("Raw policy observation contains NaN or inf")
        if not np.isfinite(mean_action).all():
            raise ValueError("Deterministic policy action contains NaN or inf")

        self.observations.append(raw_observation)
        self.actions.append(mean_action)
        self.episode_ids.append(self.episode_id)
        self.episode_steps.append(self.episode_step)
        self.terminated.append(terminated)
        self.truncated.append(truncated)
        if terminated or truncated:
            self.episode_id += 1
            self.episode_step = 0
        else:
            self.episode_step += 1

    def save(self) -> None:
        if not self.observations:
            print("[WARN] Policy trace requested, but no frames were collected")
            return
        observations = np.stack(self.observations).astype(np.float32, copy=False)
        actions = np.stack(self.actions).astype(np.float32, copy=False)
        # Legacy 130/134 layouts are kept so a trace recorded by an older checkout
        # still validates; the current env emits 136 (134 + 2 previous actions).
        if observations.ndim != 2 or observations.shape[1] not in (130, 134, 136):
            raise ValueError(
                "Expected traced observations with shape (N, 130), (N, 134) or "
                f"(N, 136), got {observations.shape}"
            )
        if actions.shape[1:] != (2,):
            raise ValueError(f"Expected traced actions with shape (N, 2), got {actions.shape}")

        self.path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            self.path,
            observations=observations,
            pt_actions=actions,
            episode_ids=np.asarray(self.episode_ids, dtype=np.int64),
            episode_steps=np.asarray(self.episode_steps, dtype=np.int64),
            terminated=np.asarray(self.terminated, dtype=np.bool_),
            truncated=np.asarray(self.truncated, dtype=np.bool_),
            checkpoint_path=np.asarray(self.checkpoint),
            checkpoint_sha256=np.asarray(_sha256(self.checkpoint)),
            observation_stage=np.asarray(self.OBSERVATION_STAGE),
            task=np.asarray(self.task),
        )
        print(
            f"[INFO] Saved policy trace: {self.path} "
            f"(frames={len(observations)}, episodes={len(np.unique(self.episode_ids))})"
        )


class EpisodeActionPlotRecorder:
    """记录单环境 episode 的目标动作、可执行命令与仿真实际响应。"""

    CSV_COLUMNS = (
        "step",
        "command_time_s",
        "measurement_time_s",
        "policy_action_linear",
        "policy_action_angular",
        "applied_action_linear",
        "applied_action_angular",
        "requested_vx_mps",
        "requested_wz_radps",
        "projected_vx_mps",
        "projected_wz_radps",
        "safe_vx_mps",
        "safe_wz_radps",
        "actual_vx_mps",
        "actual_wz_radps",
        "wheel_projection_scale",
        "wheel_acceleration_scale",
        "wheel_left_cmd_radps",
        "wheel_right_cmd_radps",
        "wheel_left_actual_radps",
        "wheel_right_actual_radps",
        "terminated",
        "truncated",
    )
    # Derived from CSV_COLUMNS so the plot cannot drift out of step with the
    # written schema when a column is added.
    CSV_INDEX = {name: index for index, name in enumerate(CSV_COLUMNS)}

    def __init__(self, output_dir: str, step_dt: float, max_episodes: int):
        if step_dt <= 0:
            raise ValueError("action plot step_dt must be positive")
        if max_episodes < -1:
            raise ValueError(
                "--action-plot-episodes must be -1, 0, or a positive integer"
            )
        self.output_dir = Path(output_dir).resolve()
        self.step_dt = step_dt
        self.max_episodes = None if max_episodes == -1 else max_episodes
        self.saved_episode_count = 0
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.episode_index = self._next_episode_index()
        self.records: list[list[float]] = []

    @property
    def complete(self) -> bool:
        return (
            self.max_episodes is not None
            and self.saved_episode_count >= self.max_episodes
        )

    def _next_episode_index(self) -> int:
        """从已有诊断文件之后继续编号，避免多次 play 覆盖历史记录。"""

        prefix = "emma_action_episode_"
        existing_indices: list[int] = []
        for path in self.output_dir.glob(f"{prefix}*.*"):
            if path.suffix.lower() not in (".csv", ".png"):
                continue
            suffix = path.stem[len(prefix):]
            if suffix.isdigit():
                existing_indices.append(int(suffix))
        return max(existing_indices, default=-1) + 1

    def append(
        self,
        action: torch.Tensor,
        base_env,
        *,
        terminated: bool,
        truncated: bool,
    ) -> None:
        if self.complete:
            return
        if action.ndim != 1 or action.numel() != 2:
            raise ValueError(f"Action plot expects one 2-D action, got {tuple(action.shape)}")

        action_cpu = action.detach().to(device="cpu", dtype=torch.float32)
        applied_action = base_env.actions[0].detach().to(
            device="cpu", dtype=torch.float32
        )
        wheel_command = base_env._last_wheel_command[0].detach().to(
            device="cpu", dtype=torch.float32
        )
        wheel_actual = base_env._last_wheel_actual[0].detach().to(
            device="cpu", dtype=torch.float32
        )
        requested_twist = base_env._requested_body_twist_cmd[0].detach().to(
            device="cpu", dtype=torch.float32
        )
        projected_twist = base_env._projected_body_twist_cmd[0].detach().to(
            device="cpu", dtype=torch.float32
        )
        safe_twist = base_env._safe_body_twist_cmd[0].detach().to(
            device="cpu", dtype=torch.float32
        )
        projection_scale = float(base_env._wheel_projection_scale[0].item())
        acceleration_scale = float(base_env._wheel_acceleration_scale[0].item())
        body_speed = float(base_env._last_forward_speed[0].item())
        body_angular_speed = float(base_env._last_angular_speed[0].item())
        step = len(self.records)

        self.records.append([
            float(step),
            step * self.step_dt,
            (step + 1) * self.step_dt,
            float(action_cpu[0]),
            float(action_cpu[1]),
            float(applied_action[0]),
            float(applied_action[1]),
            float(requested_twist[0]),
            float(requested_twist[1]),
            float(projected_twist[0]),
            float(projected_twist[1]),
            float(safe_twist[0]),
            float(safe_twist[1]),
            body_speed,
            body_angular_speed,
            projection_scale,
            acceleration_scale,
            float(wheel_command[0]),
            float(wheel_command[1]),
            float(wheel_actual[0]),
            float(wheel_actual[1]),
            float(terminated),
            float(truncated),
        ])

        if terminated or truncated:
            reason = "terminated" if terminated else "timeout"
            self._save(reason)

    def save_pending(self) -> None:
        if self.records and not self.complete:
            self._save("interrupted")

    def _save(self, reason: str) -> None:
        if not self.records:
            return
        data = np.asarray(self.records, dtype=np.float64)
        col = self.CSV_INDEX
        # Isaac Lab 可能在 env.step 返回前已经重置终止环境，因此终止行仍保留
        # 在 CSV 中供识别，但不画入响应曲线，避免把重置后的零命令误判成制动响应。
        valid_plot_rows = (data[:, col["terminated"]] < 0.5) & (
            data[:, col["truncated"]] < 0.5
        )
        plot_data = data[valid_plot_rows] if valid_plot_rows.any() else data
        command_time = plot_data[:, col["command_time_s"]]
        measurement_time = plot_data[:, col["measurement_time_s"]]

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(4, 1, figsize=(14, 13), sharex=True)
        axes[0].plot(
            command_time,
            plot_data[:, col["policy_action_linear"]],
            label="policy action linear",
            alpha=0.65,
            linestyle="--",
        )
        axes[0].plot(
            command_time,
            plot_data[:, col["policy_action_angular"]],
            label="policy action angular",
            alpha=0.65,
            linestyle="--",
        )
        axes[0].plot(
            command_time,
            plot_data[:, col["applied_action_linear"]],
            label="applied action linear",
            alpha=0.9,
        )
        axes[0].plot(
            command_time,
            plot_data[:, col["applied_action_angular"]],
            label="applied action angular",
            alpha=0.9,
        )
        axes[0].set_ylabel("normalized action [-1, 1]")
        axes[0].set_ylim(-1.05, 1.05)
        axes[0].legend()
        axes[0].grid(True)

        axes[1].plot(
            command_time,
            plot_data[:, col["requested_vx_mps"]],
            label="requested vx",
            alpha=0.7,
        )
        axes[1].plot(
            command_time,
            plot_data[:, col["projected_vx_mps"]],
            label="projected vx",
            alpha=0.8,
        )
        axes[1].plot(
            command_time,
            plot_data[:, col["safe_vx_mps"]],
            label="safe vx",
            alpha=0.9,
        )
        axes[1].plot(
            measurement_time,
            plot_data[:, col["actual_vx_mps"]],
            label="actual vx",
            linewidth=1.1,
        )
        axes[1].set_ylabel("linear velocity [m/s]")
        axes[1].legend()
        axes[1].grid(True)

        axes[2].plot(
            command_time,
            plot_data[:, col["requested_wz_radps"]],
            label="requested wz",
            alpha=0.7,
        )
        axes[2].plot(
            command_time,
            plot_data[:, col["projected_wz_radps"]],
            label="projected wz",
            alpha=0.8,
        )
        axes[2].plot(
            command_time,
            plot_data[:, col["safe_wz_radps"]],
            label="safe wz",
            alpha=0.9,
        )
        axes[2].plot(
            measurement_time,
            plot_data[:, col["actual_wz_radps"]],
            label="actual wz",
            linewidth=1.1,
        )
        axes[2].set_ylabel("angular velocity [rad/s]")
        axes[2].legend()
        axes[2].grid(True)

        axes[3].plot(
            command_time,
            plot_data[:, col["wheel_left_cmd_radps"]],
            label="wheel left cmd",
            linewidth=0.8,
        )
        axes[3].plot(
            command_time,
            plot_data[:, col["wheel_right_cmd_radps"]],
            label="wheel right cmd",
            linewidth=0.8,
        )
        axes[3].plot(
            measurement_time,
            plot_data[:, col["wheel_left_actual_radps"]],
            label="wheel left actual",
            linewidth=0.8,
        )
        axes[3].plot(
            measurement_time,
            plot_data[:, col["wheel_right_actual_radps"]],
            label="wheel right actual",
            linewidth=0.8,
        )
        axes[3].set_ylabel("wheel velocity [rad/s]")
        axes[3].legend()
        axes[3].grid(True)

        axes[3].set_xlabel("time [s]")

        fig.suptitle(
            f"Emma action monitor — episode {self.episode_index}, "
            f"{reason}, {len(data)} steps"
        )
        image_path = self.output_dir / (
            f"emma_action_episode_{self.episode_index:03d}.png"
        )
        csv_path = self.output_dir / (
            f"emma_action_episode_{self.episode_index:03d}.csv"
        )
        fig.savefig(image_path, dpi=120, bbox_inches="tight")
        plt.close(fig)

        with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(self.CSV_COLUMNS)
            writer.writerows(self.records)

        print(
            f"[RECORD] Saved action monitor: {len(data)} steps -> "
            f"{image_path}, {csv_path}"
        )

        self.records.clear()
        self.episode_index += 1
        self.saved_episode_count += 1


def main():
    """Play with skrl agent."""
    # configure the ML framework into the global skrl variable
    if args_cli.ml_framework.startswith("jax"):
        skrl.config.jax.backend = "jax" if args_cli.ml_framework == "jax" else "numpy"

    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    # Seed skrl BEFORE constructing its wrapper: it caches config.torch/jax.key
    # and passes that key to the environment's first reset (otherwise seed=0).
    play_seed = set_seed(args_cli.seed)
    env_cfg.seed = play_seed
    if hasattr(env_cfg, "curriculum"):
        env_cfg.curriculum.enabled = False  # Fixed full-task evaluation, including multi-env play.
    print(f"[INFO] Playback/map seed: {play_seed} (reproduce with --seed {play_seed})", flush=True)
    # play 始终写入部署 LiDAR 参数，避免配置文件版本不一致时静默跳过 mask。
    env_cfg.lidar_observation_forward_half_fov_deg = (
        args_cli.lidar_forward_half_fov_deg
    )
    env_cfg.lidar_observation_blind_fill_clearance = (
        args_cli.lidar_blind_fill_clearance
    )

    env_cfg.collision_clearance_threshold = 0.001

    env_cfg.debug_lidar_fov_fill = args_cli.debug_lidar_fov_fill
    print(
        "[PLAY LIDAR] policy observation: "
        f"forward ±{args_cli.lidar_forward_half_fov_deg:.1f} deg, "
        f"blind-sector clearance={args_cli.lidar_blind_fill_clearance:.3f} m",
        flush=True,
    )
    fixed_wheel_mode = args_cli.fixed_wheel_cmd is not None
    runner = None
    resume_path = None
    if fixed_wheel_mode:
        if args_cli.num_envs != 1:
            raise ValueError("--fixed-wheel-cmd requires --num_envs 1")
        if not args_cli.ml_framework.startswith("torch"):
            raise ValueError("--fixed-wheel-cmd currently requires the torch framework")
        if args_cli.fixed_wheel_steps <= 0:
            raise ValueError("--fixed-wheel-steps must be positive")
        wheel_limit = float(env_cfg.wheel_max_velocity)
        if any(abs(command) > wheel_limit for command in args_cli.fixed_wheel_cmd):
            raise ValueError(
                f"fixed wheel commands must stay within ±{wheel_limit:.3f} rad/s"
            )
        env_cfg.action_control_mode = "wheel_velocity"
        env_cfg.safety_control.enabled = False
        env_cfg.domain_rand.enable_dynamics_noise = False
        env_cfg.domain_rand.enable_action_noise = False
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_dir = os.path.abspath(
            os.path.join("logs", "wheel_response", timestamp)
        )
        print(
            "[FIXED WHEEL] Using normal play env.step path: "
            f"cmd=({args_cli.fixed_wheel_cmd[0]:.3f}, "
            f"{args_cli.fixed_wheel_cmd[1]:.3f}) rad/s, "
            f"steps={args_cli.fixed_wheel_steps}",
            flush=True,
        )
    else:
        try:
            experiment_cfg = load_cfg_from_registry(
                args_cli.task, f"skrl_{algorithm}_cfg_entry_point"
            )
        except ValueError:
            experiment_cfg = load_cfg_from_registry(
                args_cli.task, "skrl_cfg_entry_point"
            )

        # Runner reseeds Python/NumPy/Torch before the first reset, where the map
        # is sampled. Keep it aligned with the environment instead of YAML seed=42.
        experiment_cfg["seed"] = play_seed

        # specify directory for logging experiments (load checkpoint)
        log_root_path = os.path.join(
            "logs", "skrl", experiment_cfg["agent"]["experiment"]["directory"]
        )
        log_root_path = os.path.abspath(log_root_path)
        print(f"[INFO] Loading experiment from directory: {log_root_path}")
        if args_cli.use_pretrained_checkpoint:
            resume_path = get_published_pretrained_checkpoint("skrl", args_cli.task)
            if not resume_path:
                print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
                return
        elif args_cli.checkpoint:
            resume_path = os.path.abspath(args_cli.checkpoint)
        else:
            resume_path = get_checkpoint_path(
                log_root_path,
                run_dir=f".*_{algorithm}_{args_cli.ml_framework}",
                other_dirs=["checkpoints"],
            )
        log_dir = os.path.dirname(os.path.dirname(resume_path))
        if experiment_cfg["models"].get("name") == "EmmaNavGRU":
            if not args_cli.ml_framework.startswith("torch"):
                raise ValueError("The Emma PPO+GRU baseline requires --ml_framework torch")
            prepare_recurrent_evaluation(experiment_cfg, resume_path)

    # Resolve the command rate limit before the env is built: it is written into
    # env_cfg, so it must be final before gym.make. The env config enables it by
    # default; --fixed-wheel-cmd characterises the raw actuator, so it stays
    # unlimited regardless (the same convention as probe_emma_drive.py).
    if fixed_wheel_mode or args_cli.no_motion_acceleration_limits:
        env_cfg.enable_motion_acceleration_limits = False
    if getattr(env_cfg, "enable_motion_acceleration_limits", False):
        _step_dt = env_cfg.sim.dt * env_cfg.decimation
        _linear = env_cfg.max_linear_acceleration * env_cfg.linear_acceleration_command_scale
        _angular = env_cfg.max_angular_acceleration * env_cfg.angular_acceleration_command_scale
        print(
            f"[PLAY RATE LIMIT] body-twist command bounded to {_linear:.3f} m/s^2 and "
            f"{_angular:.3f} rad/s^2 = {_linear * _step_dt:.4f} m/s and "
            f"{_angular * _step_dt:.4f} rad/s per {_step_dt * 1000.0:.0f} ms step",
            flush=True,
        )
    else:
        print(
            "[PLAY RATE LIMIT] disabled: the policy command reaches the wheels "
            "unfiltered, which is how checkpoints trained before the limiter behave.",
            flush=True,
        )

    # create isaac environment
    print("[INFO] Creating Isaac environment...", flush=True)
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    print("[INFO] gym.make returned", flush=True)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv) and algorithm in ["ppo"]:
        env = multi_agent_to_single_agent(env)
    base_env = env.unwrapped

    # get environment (step) dt for real-time evaluation
    try:
        dt = env.step_dt
    except AttributeError:
        dt = env.unwrapped.step_dt

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for skrl
    env = SkrlVecEnvWrapper(env, ml_framework=args_cli.ml_framework)  # same as: `wrap_env(env, wrapper="auto")`

    if not fixed_wheel_mode:
        # configure and instantiate the skrl runner
        experiment_cfg["trainer"]["close_environment_at_exit"] = False
        experiment_cfg["agent"]["experiment"]["write_interval"] = 0
        experiment_cfg["agent"]["experiment"]["checkpoint_interval"] = 0
        if experiment_cfg["models"]["custom"]:
            Runner._generate_models = generate_custom_model
        del experiment_cfg["models"]["custom"]
        runner = Runner(env, experiment_cfg)
        if hasattr(base_env, "_curriculum"):
            runner.agent.checkpoint_modules["curriculum"] = base_env._curriculum

        print(f"[INFO] Loading model checkpoint from: {resume_path}")
        runner.agent.load(resume_path)
        runner.agent.enable_training_mode(False, apply_to_models=True)

    # reset environment
    print("[INFO] Resetting environment...", flush=True)
    obs, _ = env.reset()
    print("[INFO] env.reset returned", flush=True)
    fixed_actions = None
    if fixed_wheel_mode:
        fixed_actions = torch.tensor(
            [args_cli.fixed_wheel_cmd],
            device=base_env.device,
            dtype=torch.float32,
        ) / float(base_env.cfg.wheel_max_velocity)
        print(
            f"[FIXED WHEEL] normalized action={fixed_actions[0].tolist()}",
            flush=True,
        )
    trace_recorder = None
    action_plot_recorder = None
    if args_cli.action_plot_episodes != 0:
        action_plot_fields = (
            "actions",
            "_requested_body_twist_cmd",
            "_projected_body_twist_cmd",
            "_safe_body_twist_cmd",
            "_wheel_projection_scale",
            "_wheel_acceleration_scale",
            "_last_wheel_command",
            "_last_wheel_actual",
            "_last_forward_speed",
            "_last_angular_speed",
        )
        if not args_cli.ml_framework.startswith("torch"):
            print("[WARN] Action episode diagnostics currently require the torch framework")
        elif args_cli.num_envs != 1:
            print(
                "[WARN] Action episode diagnostics require --num_envs 1; "
                "recording is disabled"
            )
        elif not all(hasattr(base_env, name) for name in action_plot_fields):
            print(
                "[WARN] This task does not expose Emma action diagnostics; "
                "plot recording is disabled"
            )
        else:
            action_plot_recorder = EpisodeActionPlotRecorder(
                os.path.join(log_dir, "play_diagnostics"),
                dt,
                args_cli.action_plot_episodes,
            )
            episode_limit = (
                "all"
                if args_cli.action_plot_episodes == -1
                else str(args_cli.action_plot_episodes)
            )
            print(
                f"[INFO] Recording {episode_limit} action diagnostic episode(s) "
                f"to {action_plot_recorder.output_dir}"
            )
    if args_cli.policy_trace:
        if fixed_wheel_mode:
            raise ValueError("--policy-trace cannot be combined with --fixed-wheel-cmd")
        if not args_cli.ml_framework.startswith("torch"):
            raise ValueError("Policy trace capture currently supports only the torch framework")
        if args_cli.video:
            raise ValueError("Policy trace capture cannot be combined with --video")
        if args_cli.num_envs != 1:
            raise ValueError(
                "Policy trace capture currently requires --num_envs 1 so episode "
                "boundaries remain unambiguous"
            )
        if args_cli.policy_trace_env >= args_cli.num_envs:
            raise ValueError(
                f"--policy-trace-env={args_cli.policy_trace_env} is outside "
                f"num_envs={args_cli.num_envs}"
            )
        if isinstance(obs, dict):
            raise TypeError("Policy trace capture expects a single-agent tensor observation")
        trace_recorder = PolicyTraceRecorder(
            args_cli.policy_trace,
            args_cli.policy_trace_steps,
            args_cli.policy_trace_env,
            resume_path,
            args_cli.task or "",
        )
        print(
            f"[INFO] Recording policy trace: path={trace_recorder.path}, "
            f"steps={trace_recorder.target_steps}, env={trace_recorder.env_index}"
        )
    timestep = 0
    try:
        # simulate environment
        while simulation_app.is_running():
            start_time = time.time()

            # run everything in inference mode
            with torch.inference_mode():
                # Keep this snapshot before agent.act: the skrl observation
                # preprocessor is applied inside the agent.
                trace_observation = None
                if trace_recorder is not None:
                    trace_observation = obs[trace_recorder.env_index].clone()

                if fixed_wheel_mode:
                    actions = fixed_actions
                else:
                    # agent stepping
                    outputs = runner.agent.act(
                        obs, None, timestep=timestep, timesteps=timestep
                    )
                    # - multi-agent (deterministic) actions
                    if hasattr(env, "possible_agents"):
                        actions = {
                            a: outputs[-1][a].get(
                                "mean_actions", outputs[0][a]
                            )
                            for a in env.possible_agents
                        }
                    # - single-agent (deterministic) actions
                    else:
                        actions = outputs[-1].get("mean_actions", outputs[0])
                # env stepping
                next_obs, rewards, terminated, truncated, infos = env.step(actions)
                if not fixed_wheel_mode and getattr(runner.agent, "_rnn", False):
                    # act() only computes the next hidden state in skrl. This
                    # call commits it and clears just the auto-reset episodes.
                    runner.agent.record_transition(
                        observations=obs, states=None, actions=actions, rewards=rewards,
                        next_observations=next_obs, next_states=None,
                        terminated=terminated, truncated=truncated, infos=infos,
                        timestep=timestep, timesteps=timestep,
                    )

                if action_plot_recorder is not None and not action_plot_recorder.complete:
                    action_plot_recorder.append(
                        actions[0],
                        base_env,
                        terminated=bool(terminated[0].item()),
                        truncated=bool(truncated[0].item()),
                    )

                if trace_recorder is not None:
                    assert trace_observation is not None
                    done_terminated = bool(terminated[trace_recorder.env_index].item())
                    done_truncated = bool(truncated[trace_recorder.env_index].item())
                    trace_recorder.append(
                        trace_observation,
                        actions[trace_recorder.env_index],
                        done_terminated,
                        done_truncated,
                    )

                obs = next_obs
                if terminated.any() or truncated.any():
                    obs, _ = env.reset()

            if trace_recorder is not None and trace_recorder.complete:
                break
            timestep += 1
            if fixed_wheel_mode and timestep >= args_cli.fixed_wheel_steps:
                print(
                    f"[FIXED WHEEL] completed {timestep} env.step calls",
                    flush=True,
                )
                break
            if args_cli.video and timestep == args_cli.video_length:
                break

            # time delay for real-time evaluation
            sleep_time = dt - (time.time() - start_time)
            if args_cli.real_time and sleep_time > 0:
                time.sleep(sleep_time)
    finally:
        try:
            if trace_recorder is not None:
                trace_recorder.save()
            if action_plot_recorder is not None:
                action_plot_recorder.save_pending()
        finally:
            # close the simulator even if trace serialization detects invalid data
            env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()

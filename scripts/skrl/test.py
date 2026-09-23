#!/usr/bin/env python3
"""Evaluate an skrl checkpoint in the three documented obstacle scenarios."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any

from isaaclab.app import AppLauncher

from evaluation_utils import (
    minimum_passing_width,
    straight_returned_to_centerline,
    summarize_width,
    termination_reason_name,
    width_sequence,
)


SCENARIOS = ("ring", "corridor", "straight_obstacle")
SCENARIO_LABELS = {
    "ring": "回型路径",
    "corridor": "S 型路径",
    "straight_obstacle": "直行绕障",
}
DEFAULT_WIDTHS = {
    "ring": (1.40, 2.50, 0.10),
    "corridor": (0.70, 2.50, 0.10),
    "straight_obstacle": (0.70, 1.50, 0.10),
}
EPISODE_FIELDS = (
    "scenario",
    "scenario_label",
    "passage_width_m",
    "trial",
    "seed",
    "success",
    "clean_success",
    "termination_reason",
    "termination_reason_code",
    "steps",
    "elapsed_time_s",
    "path_length_m",
    "initial_goal_distance_m",
    "final_goal_distance_m",
    "minimum_clearance_m",
    "mean_forward_speed_mps",
    "max_forward_speed_mps",
    "mean_abs_angular_speed_radps",
    "max_abs_angular_speed_radps",
    "collision",
    "collision_terminal",
    "collision_events",
    "goal_reached",
    "timeout",
    "fell",
    "drifted",
    "maximum_lateral_deviation_m",
    "post_obstacle_final_centerline_error_m",
    "returned_to_centerline",
    "load_kg",
    "mapping_mode",
    "obstacle_type",
    "obstacle_length_m",
    "obstacle_width_m",
    "obstacle_height_m",
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Scan passage widths for 回型路径, S 型路径, and 直行绕障 with "
            "deterministic skrl PPO inference."
        )
    )
    parser.add_argument("--task", default="Emma-RL-ObsNav-Direct-v0")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--use_pretrained_checkpoint", action="store_true")
    parser.add_argument("--algorithm", default="PPO", choices=["PPO"])
    parser.add_argument(
        "--ml_framework",
        default="torch",
        choices=["torch"],
        help="Evaluation currently supports deterministic torch inference.",
    )
    parser.add_argument("--disable_fabric", action="store_true")
    parser.add_argument(
        "--scenarios",
        nargs="+",
        choices=SCENARIOS,
        default=list(SCENARIOS),
    )
    parser.add_argument("--width-min", type=float, default=None)
    parser.add_argument("--width-max", type=float, default=None)
    parser.add_argument("--width-step", type=float, default=None)
    for scenario in SCENARIOS:
        cli_name = scenario.replace("_", "-")
        parser.add_argument(f"--{cli_name}-width-min", type=float, default=None)
        parser.add_argument(f"--{cli_name}-width-max", type=float, default=None)
        parser.add_argument(f"--{cli_name}-width-step", type=float, default=None)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--required-consecutive-successes", type=int, default=3)
    parser.add_argument("--eval-seed", type=int, default=42)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--load-kg", type=float, default=0.0)
    parser.add_argument(
        "--mapping-mode",
        choices=("lidar", "visual", "simulation"),
        default="simulation",
    )
    parser.add_argument("--lidar-forward-half-fov-deg", type=float, default=110.0)
    parser.add_argument("--lidar-blind-fill-clearance", type=float, default=1.0)
    parser.add_argument("--straight-return-tolerance", type=float, default=0.35)
    parser.add_argument("--real-time", action="store_true")
    parser.add_argument("--num_envs", type=int, default=1, help=argparse.SUPPRESS)
    parser.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--_worker-scenario", choices=SCENARIOS, help=argparse.SUPPRESS)
    parser.add_argument("--_worker-width", type=float, help=argparse.SUPPRESS)
    parser.add_argument("--_worker-output", help=argparse.SUPPRESS)
    parser.add_argument(
        "--no-motion-acceleration-limits",
        "--no_motion_acceleration_limits",
        dest="no_motion_acceleration_limits",
        action="store_true",
        help=(
            "Disable the body-twist command rate limit that the env config enables "
            "by default, to score a checkpoint against the unlimited actuator it "
            "was trained on. Forwarded to every scenario worker."
        ),
    )
    AppLauncher.add_app_launcher_args(parser)
    return parser


def _scenario_widths(args: argparse.Namespace, scenario: str) -> list[float]:
    defaults = DEFAULT_WIDTHS[scenario]
    prefix = scenario.replace("_", "_")
    minimum = getattr(args, f"{prefix}_width_min")
    maximum = getattr(args, f"{prefix}_width_max")
    step = getattr(args, f"{prefix}_width_step")
    return width_sequence(
        args.width_min if minimum is None and args.width_min is not None else (
            defaults[0] if minimum is None else minimum
        ),
        args.width_max if maximum is None and args.width_max is not None else (
            defaults[1] if maximum is None else maximum
        ),
        args.width_step if step is None and args.width_step is not None else (
            defaults[2] if step is None else step
        ),
    )


def _validate_args(args: argparse.Namespace) -> None:
    if args.trials < args.required_consecutive_successes:
        raise ValueError(
            "--trials must be at least --required-consecutive-successes"
        )
    if args.required_consecutive_successes < 3:
        raise ValueError("the test document requires at least 3 consecutive successes")
    if args.num_envs != 1:
        raise ValueError("evaluation uses exactly one environment per worker")
    if not math.isfinite(args.load_kg) or args.load_kg < 0.0:
        raise ValueError("--load-kg must be finite and non-negative")
    if (
        not math.isfinite(args.straight_return_tolerance)
        or args.straight_return_tolerance < 0.0
    ):
        raise ValueError("--straight-return-tolerance must be finite and non-negative")
    if (
        not math.isfinite(args.lidar_forward_half_fov_deg)
        or not 0.0 < args.lidar_forward_half_fov_deg <= 180.0
    ):
        raise ValueError("--lidar-forward-half-fov-deg must be in (0, 180]")
    if (
        not math.isfinite(args.lidar_blind_fill_clearance)
        or args.lidar_blind_fill_clearance < 0.0
    ):
        raise ValueError("--lidar-blind-fill-clearance must be finite and non-negative")
    for scenario in args.scenarios:
        _scenario_widths(args, scenario)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    os.replace(temporary, path)


def _sha256(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _worker_command(
    original_args: list[str],
    scenario: str,
    width: float,
    output_path: Path,
) -> list[str]:
    filtered_args: list[str] = []
    skip_next = False
    hidden_values = {"--_worker-scenario", "--_worker-width", "--_worker-output"}
    for argument in original_args:
        if skip_next:
            skip_next = False
            continue
        if argument in hidden_values:
            skip_next = True
            continue
        if argument == "--_worker":
            continue
        filtered_args.append(argument)
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        *filtered_args,
        "--_worker",
        "--_worker-scenario",
        scenario,
        "--_worker-width",
        format(width, ".12g"),
        "--_worker-output",
        str(output_path),
    ]


def _controller(args: argparse.Namespace, original_args: list[str]) -> int:
    _validate_args(args)
    if args.checkpoint:
        args.checkpoint = str(Path(args.checkpoint).expanduser().resolve())
        original_args = [
            argument
            for index, argument in enumerate(original_args)
            if argument != "--checkpoint"
            and not (
                index > 0 and original_args[index - 1] == "--checkpoint"
            )
        ]
        original_args.extend(("--checkpoint", args.checkpoint))
    if args.output_dir:
        output_dir = Path(args.output_dir).expanduser().resolve()
    else:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_dir = Path("logs") / "skrl" / "evaluation" / timestamp
        output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    episodes: list[dict[str, Any]] = []
    widths_summary: list[dict[str, Any]] = []
    worker_metadata: dict[str, Any] | None = None
    with tempfile.TemporaryDirectory(prefix="emma_eval_") as temp_dir:
        for scenario in args.scenarios:
            for width_index, width in enumerate(_scenario_widths(args, scenario)):
                print(
                    f"[SCAN] {SCENARIO_LABELS[scenario]} ({scenario}) "
                    f"width={width:.3f} m",
                    flush=True,
                )
                worker_output = Path(temp_dir) / f"{scenario}_{width_index}.json"
                completed = subprocess.run(
                    _worker_command(original_args, scenario, width, worker_output),
                    check=False,
                )
                if completed.returncode != 0 or not worker_output.is_file():
                    raise RuntimeError(
                        f"evaluation worker failed for {scenario} width={width:.3f}m "
                        f"with exit code {completed.returncode}"
                    )
                with worker_output.open(encoding="utf-8") as stream:
                    result = json.load(stream)
                if worker_metadata is None:
                    worker_metadata = result["metadata"]
                elif result["metadata"]["checkpoint_sha256"] != worker_metadata["checkpoint_sha256"]:
                    raise RuntimeError("workers resolved different checkpoints during one scan")
                width_episodes = result["episodes"]
                episodes.extend(width_episodes)
                summary = summarize_width(
                    scenario,
                    width,
                    width_episodes,
                    args.required_consecutive_successes,
                )
                widths_summary.append(summary)
                print(
                    f"[WIDTH] success={summary['success_count']}/{summary['trials']} "
                    f"streak={summary['longest_consecutive_successes']} "
                    f"passed={'YES' if summary['passed'] else 'NO'}",
                    flush=True,
                )

    scenario_summaries = []
    for scenario in args.scenarios:
        rows = [row for row in widths_summary if row["scenario"] == scenario]
        scenario_summaries.append(
            {
                "scenario": scenario,
                "scenario_label": SCENARIO_LABELS[scenario],
                "minimum_passing_width_m": minimum_passing_width(rows),
                "tested_widths_m": [row["passage_width_m"] for row in rows],
                "passed_widths_m": [
                    row["passage_width_m"] for row in rows if row["passed"]
                ],
            }
        )

    csv_path = output_dir / "episodes.csv"
    temporary_csv = csv_path.with_suffix(".csv.tmp")
    with temporary_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=EPISODE_FIELDS)
        writer.writeheader()
        writer.writerows({field: row.get(field) for field in EPISODE_FIELDS} for row in episodes)
    os.replace(temporary_csv, csv_path)

    payload = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "metadata": worker_metadata,
        "configuration": {
            "task": args.task,
            "scenarios": list(args.scenarios),
            "trials_per_width": args.trials,
            "required_consecutive_successes": args.required_consecutive_successes,
            "base_seed": args.eval_seed,
            "load_kg": args.load_kg,
            "load_is_metadata_only": True,
            "mapping_mode": args.mapping_mode,
            "straight_return_tolerance_m": args.straight_return_tolerance,
            "widths_m": {
                scenario: _scenario_widths(args, scenario)
                for scenario in args.scenarios
            },
        },
        "scenarios": scenario_summaries,
        "width_results": widths_summary,
        "episodes": episodes,
    }
    _atomic_json(output_dir / "summary.json", payload)

    print("\n=== Evaluation summary ===")
    for summary in scenario_summaries:
        minimum = summary["minimum_passing_width_m"]
        minimum_text = "not passed" if minimum is None else f"{minimum:.3f} m"
        print(f"{summary['scenario_label']:<12} minimum passing width: {minimum_text}")
    print(f"CSV : {csv_path}")
    print(f"JSON: {output_dir / 'summary.json'}")
    return 0


def _tensor_scalar(values: Any, index: int = 0) -> Any:
    value = values[index]
    if hasattr(value, "item"):
        return value.item()
    return value


def _worker(args: argparse.Namespace) -> int:
    if args._worker_scenario is None or args._worker_width is None or not args._worker_output:
        raise ValueError("worker scenario, width, and output are required")
    args.num_envs = 1
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app
    env = None
    try:
        import time

        import gymnasium as gym
        import torch
        import skrl
        from packaging import version

        if version.parse(skrl.__version__) < version.parse("1.4.2"):
            raise RuntimeError(f"skrl >= 1.4.2 is required, found {skrl.__version__}")
        from skrl.utils.runner.torch import Runner
        from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
        from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
        from isaaclab_tasks.utils import (
            get_checkpoint_path,
            load_cfg_from_registry,
            parse_env_cfg,
        )
        from isaaclab_rl.skrl import SkrlVecEnvWrapper
        import isaaclab_tasks  # noqa: F401
        import emma_rl.tasks  # noqa: F401
        from emma_rl.tasks.direct.emma.utils import generate_custom_model
        from emma_rl.tasks.direct.emma.utils.recurrent_runner import (
            EmmaRunner as Runner, prepare_recurrent_evaluation,
        )

        algorithm = args.algorithm.lower()
        try:
            experiment_cfg = load_cfg_from_registry(
                args.task, f"skrl_{algorithm}_cfg_entry_point"
            )
        except ValueError:
            experiment_cfg = load_cfg_from_registry(args.task, "skrl_cfg_entry_point")
        if args.use_pretrained_checkpoint:
            checkpoint = get_published_pretrained_checkpoint("skrl", args.task)
            if not checkpoint:
                raise FileNotFoundError("no published checkpoint is available for this task")
        elif args.checkpoint:
            checkpoint = os.path.abspath(os.path.expanduser(args.checkpoint))
        else:
            log_root = os.path.abspath(
                os.path.join(
                    "logs",
                    "skrl",
                    experiment_cfg["agent"]["experiment"]["directory"],
                )
            )
            checkpoint = get_checkpoint_path(
                log_root,
                run_dir=f".*_{algorithm}_{args.ml_framework}",
                other_dirs=["checkpoints"],
            )
        if not os.path.isfile(checkpoint):
            raise FileNotFoundError(f"checkpoint does not exist: {checkpoint}")
        prepare_recurrent_evaluation(experiment_cfg, checkpoint)

        env_cfg = parse_env_cfg(
            args.task,
            device=args.device,
            num_envs=1,
            use_fabric=not args.disable_fabric,
        )
        required = ("world_mode", "domain_rand", "lidar_sensor_mode")
        if not all(hasattr(env_cfg, name) for name in required):
            raise TypeError("the three-scene evaluator requires Emma ObsNav configuration")
        scenario = args._worker_scenario
        width = float(args._worker_width)
        env_cfg.world_mode = scenario
        env_cfg.lidar_observation_forward_half_fov_deg = args.lidar_forward_half_fov_deg
        env_cfg.lidar_observation_blind_fill_clearance = args.lidar_blind_fill_clearance
        env_cfg.domain_rand.enable_lidar_noise = False
        env_cfg.domain_rand.enable_lidar_dropout = False
        env_cfg.domain_rand.enable_dynamics_noise = False
        env_cfg.domain_rand.enable_action_noise = False
        env_cfg.domain_rand.enable_init_noise = False
        env_cfg.domain_rand.action_delay_max_steps = 0
        env_cfg.lidar_sensor_mode = "synchronous"
        env_cfg.lidar_latency_range_s = (0.0, 0.0)
        env_cfg.lidar_jitter_max_steps = 0
        env_cfg.lidar_scan_dropout_prob = 0.0
        env_cfg.curriculum.enabled = False
        # The rate limit decides whether the evaluated angular command is one the
        # real base could execute, so the scan must say which actuator it scored.
        if args.no_motion_acceleration_limits:
            env_cfg.enable_motion_acceleration_limits = False
        print(
            f"[TEST RATE LIMIT] "
            f"{'enabled' if env_cfg.enable_motion_acceleration_limits else 'disabled'} "
            f"for {scenario} width={width:.3f}m",
            flush=True,
        )
        if scenario == "ring":
            env_cfg.obs_world.corridor_width = width
            env_cfg.obs_world.num_obstacles = 0
        elif scenario == "corridor":
            env_cfg.corridor_world.gate_width = width
            env_cfg.corridor_world.mirror_across_envs = False
            env_cfg.corridor_world.spawn_jitter_x = 0.0
            env_cfg.corridor_world.spawn_jitter_y = 0.0
        else:
            env_cfg.straight_obstacle_world.passage_width = width

        torch.manual_seed(args.eval_seed)
        env_cfg.seed = args.eval_seed
        env = gym.make(
            args.task,
            cfg=env_cfg,
        )
        if isinstance(env.unwrapped, DirectMARLEnv):
            env = multi_agent_to_single_agent(env)
        base_env = env.unwrapped
        if not hasattr(base_env, "_termination_reason_window"):
            raise TypeError(
                "the three-scene evaluator requires the maintained Emma ObsNav environment"
            )
        step_dt = float(base_env.step_dt)
        wrapped_env = SkrlVecEnvWrapper(env, ml_framework=args.ml_framework)
        env = wrapped_env

        experiment_cfg["trainer"]["close_environment_at_exit"] = False
        experiment_cfg["agent"]["experiment"]["write_interval"] = 0
        experiment_cfg["agent"]["experiment"]["checkpoint_interval"] = 0
        if experiment_cfg["models"]["custom"]:
            Runner._generate_models = generate_custom_model
        del experiment_cfg["models"]["custom"]
        runner = Runner(env, experiment_cfg)
        runner.agent.checkpoint_modules["curriculum"] = base_env._curriculum
        runner.agent.load(checkpoint)
        runner.agent.enable_training_mode(False, apply_to_models=True)

        episodes: list[dict[str, Any]] = []
        timestep = 0
        for trial in range(1, args.trials + 1):
            trial_seed = args.eval_seed + trial - 1
            torch.manual_seed(trial_seed)
            if hasattr(base_env, "seed"):
                base_env.seed(trial_seed)
            obs, _ = env.reset()
            start_xy = (
                base_env.robot.data.root_pos_w[0, :2]
                - base_env.scene.env_origins[0, :2]
            ).detach().cpu().tolist()
            goal_xy = (
                base_env.goal_pos_xy[0]
                - base_env.scene.env_origins[0, :2]
            ).detach().cpu().tolist()
            positions = [(float(start_xy[0]), float(start_xy[1]))]
            initial_goal_distance = math.dist(start_xy, goal_xy)
            clearances: list[float] = []
            forward_speeds: list[float] = []
            angular_speeds: list[float] = []
            terminal: dict[str, Any] | None = None

            while simulation_app.is_running():
                started = time.time()
                with torch.inference_mode():
                    outputs = runner.agent.act(
                        obs, None, timestep=timestep, timesteps=timestep
                    )
                    actions = outputs[-1].get("mean_actions", outputs[0])
                    next_obs, rewards, terminated, truncated, info = env.step(actions)
                    if getattr(runner.agent, "_rnn", False):
                        runner.agent.record_transition(
                            observations=obs, states=None, actions=actions, rewards=rewards,
                            next_observations=next_obs, next_states=None,
                            terminated=terminated, truncated=truncated, infos=info,
                            timestep=timestep, timesteps=timestep,
                        )
                    obs = next_obs
                evaluation = info.get("evaluation") if isinstance(info, dict) else None
                if not isinstance(evaluation, dict):
                    raise RuntimeError(
                        "environment did not return extras['evaluation']; reinstall local package"
                    )
                position = evaluation["position_xy"][0].detach().cpu().tolist()
                positions.append((float(position[0]), float(position[1])))
                clearances.append(float(_tensor_scalar(evaluation["minimum_clearance"])))
                forward_speeds.append(float(_tensor_scalar(evaluation["forward_speed"])))
                angular_speeds.append(abs(float(_tensor_scalar(evaluation["angular_speed"]))))
                done = bool(terminated[0].item() or truncated[0].item())
                if done:
                    terminal = {
                        key: _tensor_scalar(value)
                        for key, value in evaluation.items()
                        if key not in ("position_xy", "goal_position_xy")
                    }
                    break
                timestep += 1
                if args.real_time:
                    delay = step_dt - (time.time() - started)
                    if delay > 0.0:
                        time.sleep(delay)
            if terminal is None:
                raise RuntimeError("simulation stopped before the trial ended")

            reason_code = int(terminal["termination_reason"])
            reason = termination_reason_name(reason_code)
            returned: bool | None = None
            final_center_error: float | None = None
            if scenario == "straight_obstacle":
                direction = 1.0 if goal_xy[0] > start_xy[0] else -1.0
                returned, final_center_error = straight_returned_to_centerline(
                    positions,
                    travel_direction=direction,
                    obstacle_half_length=(
                        env_cfg.straight_obstacle_world.obstacle_length / 2.0
                    ),
                    robot_half_length=env_cfg.robot_half_length,
                    tolerance=args.straight_return_tolerance,
                )
            success = reason_code == 5 and (returned is not False)
            path_length = sum(
                math.dist(previous, current)
                for previous, current in zip(positions, positions[1:])
            )
            obstacle_cfg = env_cfg.straight_obstacle_world
            episodes.append(
                {
                    "scenario": scenario,
                    "scenario_label": SCENARIO_LABELS[scenario],
                    "passage_width_m": width,
                    "trial": trial,
                    "seed": trial_seed,
                    "success": success,
                    "clean_success": success and not bool(terminal["collision"]),
                    "termination_reason": reason,
                    "termination_reason_code": reason_code,
                    "steps": int(terminal["episode_steps"]),
                    "elapsed_time_s": int(terminal["episode_steps"]) * step_dt,
                    "path_length_m": path_length,
                    "initial_goal_distance_m": initial_goal_distance,
                    "final_goal_distance_m": float(terminal["goal_distance"]),
                    "minimum_clearance_m": min(clearances),
                    "mean_forward_speed_mps": sum(forward_speeds) / len(forward_speeds),
                    "max_forward_speed_mps": max(forward_speeds),
                    "mean_abs_angular_speed_radps": sum(angular_speeds) / len(angular_speeds),
                    "max_abs_angular_speed_radps": max(angular_speeds),
                    "collision": bool(terminal["collision"]),
                    "collision_terminal": bool(
                        terminal.get("collision_terminal", False)
                    ),
                    "collision_events": int(
                        terminal.get("collision_event_count", 0)
                    ),
                    "goal_reached": bool(terminal["goal_reached"]),
                    "timeout": bool(terminal["timeout"]),
                    "fell": bool(terminal["fell"]),
                    "drifted": bool(terminal["drifted"]),
                    "maximum_lateral_deviation_m": (
                        max(abs(y) for _, y in positions)
                        if scenario == "straight_obstacle"
                        else None
                    ),
                    "post_obstacle_final_centerline_error_m": final_center_error,
                    "returned_to_centerline": returned,
                    "load_kg": args.load_kg,
                    "mapping_mode": args.mapping_mode,
                    "obstacle_type": (
                        obstacle_cfg.obstacle_type
                        if scenario == "straight_obstacle"
                        else None
                    ),
                    "obstacle_length_m": (
                        obstacle_cfg.obstacle_length
                        if scenario == "straight_obstacle"
                        else None
                    ),
                    "obstacle_width_m": (
                        obstacle_cfg.obstacle_width
                        if scenario == "straight_obstacle"
                        else None
                    ),
                    "obstacle_height_m": (
                        obstacle_cfg.obstacle_height
                        if scenario == "straight_obstacle"
                        else None
                    ),
                }
            )
            print(
                f"[TRIAL] {trial}/{args.trials} seed={trial_seed} "
                f"reason={reason} success={'YES' if success else 'NO'}",
                flush=True,
            )
            timestep += 1

        payload = {
            "metadata": {
                "task": args.task,
                "checkpoint": os.path.abspath(checkpoint),
                "checkpoint_sha256": _sha256(checkpoint),
                "device": str(base_env.device),
                "control_step_dt_s": step_dt,
                "deterministic_actions": True,
                "domain_randomization_enabled": False,
                "load_is_metadata_only": True,
            },
            "episodes": episodes,
        }
        _atomic_json(Path(args._worker_output), payload)
        return 0
    finally:
        try:
            if env is not None:
                env.close()
        finally:
            simulation_app.close()


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    if args._worker:
        return _worker(args)
    return _controller(args, sys.argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())

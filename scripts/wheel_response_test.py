"""Observation-free sinusoidal action calibration in one Isaac Lab environment.

Uses the same profile, kinematics, CSV schema, lag estimator and plotting code
as the 159 real-robot adapter. The policy, observations, rewards and episode
termination are bypassed; only the action-output side and physics are stepped.
"""
from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", type=str, default="Emma-RL-ObsNav-Direct-v0")
parser.add_argument("--linear_accelerations", default="0.3,0.6,0.9")
parser.add_argument("--angular_accelerations", default="0.6,1.2,1.88")
parser.add_argument("--hold_seconds", type=float, default=.5)
parser.add_argument("--settle_seconds", type=float, default=.5)
parser.add_argument("--output_dir", type=str, default=None)
parser.add_argument("--disable_fabric", action="store_true", default=False)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


from datetime import datetime
import json
from pathlib import Path

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg
import isaaclab.utils.math as math_utils

import emma_rl.tasks  # noqa: F401

from emma_action_calibration import (
    action_components, action_to_motion, build_runs, parse_positive_csv,
    stream_row, write_artifacts,
)


def _step_physics_only(base_env, action: torch.Tensor) -> None:
    """Advance one control period without observation/reward/termination work."""
    base_env._pre_physics_step(action)
    for _ in range(int(base_env.cfg.decimation)):
        if hasattr(base_env, "_sim_step_counter"):
            base_env._sim_step_counter += 1
        base_env._apply_action()
        base_env.scene.write_data_to_sim()
        base_env.sim.step(render=False)
        base_env.scene.update(dt=base_env.physics_dt)


def _actual_body_motion(base_env) -> tuple[torch.Tensor, torch.Tensor]:
    robot = base_env.robot
    forward_w = math_utils.quat_apply(
        robot.data.root_quat_w, robot.data.FORWARD_VEC_B
    )
    forward_w = torch.nn.functional.normalize(forward_w[:, :2], dim=-1)
    forward_speed = torch.sum(
        robot.data.root_com_lin_vel_w[:, :2] * forward_w, dim=-1
    )
    return forward_speed, robot.data.root_ang_vel_b[:, 2]


def _metadata(base_env, step_dt):
    cfg = base_env.cfg
    return {
        "backend": "isaac_lab_physics",
        "task": args_cli.task,
        "device": str(base_env.device),
        "sim_dt_s": float(cfg.sim.dt),
        "decimation": int(cfg.decimation),
        "control_step_dt_s": step_dt,
        "wheel_radius_m": float(cfg.wheel_radius),
        "wheel_base_m": float(cfg.wheel_base),
        "wheel_max_velocity_radps": float(cfg.wheel_max_velocity),
        "max_linear_velocity_mps": float(cfg.max_linear_velocity),
        "max_angular_velocity_radps": float(cfg.max_angular_velocity),
        "action_control_mode": str(cfg.action_control_mode),
        "motion_acceleration_limiter": bool(cfg.enable_motion_acceleration_limits),
        "step_mode": "physics_only_no_observation_reward_or_done",
    }


def main() -> None:
    if args_cli.hold_seconds < 0. or args_cli.settle_seconds < 0.:
        raise ValueError("hold and settle durations must be non-negative")
    linear = parse_positive_csv(args_cli.linear_accelerations, "linear accelerations")
    angular = parse_positive_csv(args_cli.angular_accelerations, "angular accelerations")

    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=1,
        use_fabric=not args_cli.disable_fabric,
    )
    env_cfg.action_control_mode = "body_twist"
    # The generated profile itself has the selected acceleration. A second
    # limiter would make the commanded curve differ between backends.
    env_cfg.enable_motion_acceleration_limits = False
    env_cfg.safety_control.enabled = False
    env_cfg.domain_rand.enable_lidar_noise = False
    env_cfg.domain_rand.enable_lidar_dropout = False
    env_cfg.domain_rand.enable_dynamics_noise = False
    env_cfg.domain_rand.enable_action_noise = False
    env_cfg.obs_world.num_obstacles = 0
    env_cfg.random_obstacle_world.num_obstacles = 0
    env_cfg.random_obstacle_world.structured_layout = False
    env_cfg.curriculum.enabled = False
    env_cfg.lidar_sensor_mode = "synchronous"
    env_cfg.lidar_latency_range_s = (0.0, 0.0)
    env_cfg.lidar_jitter_max_steps = 0
    env_cfg.lidar_scan_dropout_prob = 0.0

    env = gym.make(args_cli.task, cfg=env_cfg)
    base_env = env.unwrapped
    device = base_env.device
    dt = float(base_env.step_dt)
    radius = float(base_env.cfg.wheel_radius)
    wheel_base = float(base_env.cfg.wheel_base)
    wheel_limit = float(base_env.cfg.wheel_max_velocity)
    max_linear = float(base_env.cfg.max_linear_velocity)
    max_angular = float(base_env.cfg.max_angular_velocity)
    runs = build_runs(linear, angular, dt, args_cli.hold_seconds,
                      max_linear, max_angular)
    metadata = _metadata(base_env, dt)
    settle_steps = int(round(args_cli.settle_seconds / dt))
    requested, commanded, feedback = [], [], []
    try:
        # Reset once. Every signed profile has zero integrated displacement (or
        # yaw), and a zero-action settle separates runs. Repeated DirectRLEnv
        # resets after physics-only stepping can terminate Isaac Sim because
        # the normal observation/reward lifecycle was deliberately bypassed.
        env.reset()
        for run in runs:
            print("[CALIBRATION] %s: settling" % run["run_id"], flush=True)
            zero = torch.zeros((1, 2), device=device)
            with torch.inference_mode():
                for _ in range(settle_steps):
                    _step_physics_only(base_env, zero)
                for sequence, item in enumerate(run["samples"]):
                    value, segment = item
                    action = action_components(run["axis"], value)
                    target = action_to_motion(action[0], action[1], max_linear,
                                              max_angular, radius, wheel_base,
                                              wheel_limit)
                    command_time = sequence * dt
                    requested.append(stream_row(
                        run, sequence, segment, command_time, target,
                        action=action, message_stamp_s=command_time,
                        receipt_time_s=command_time))
                    tensor = torch.tensor([action], device=device, dtype=torch.float32)
                    _step_physics_only(base_env, tensor)
                    wheel_command = base_env._wheel_vel_cmd[0]
                    command_motion = {
                        "left_radps": float(wheel_command[0].item()),
                        "right_radps": float(wheel_command[1].item()),
                    }
                    command_motion["vx_mps"] = .5 * radius * (
                        command_motion["left_radps"] + command_motion["right_radps"])
                    command_motion["wz_radps"] = radius * (
                        command_motion["right_radps"] - command_motion["left_radps"]) / wheel_base
                    commanded.append(stream_row(
                        run, sequence, segment, command_time, command_motion,
                        action=action, message_stamp_s=command_time,
                        receipt_time_s=command_time))
                    left_id = int(base_env.left_ids[0])
                    right_id = int(base_env.right_ids[0])
                    forward, angular_speed = _actual_body_motion(base_env)
                    actual = {
                        "left_radps": float(base_env.robot.data.joint_vel[0, left_id].item()),
                        "right_radps": float(base_env.robot.data.joint_vel[0, right_id].item()),
                        "vx_mps": float(forward[0].item()),
                        "wz_radps": float(angular_speed[0].item()),
                    }
                    measurement_time = (sequence + 1) * dt
                    feedback.append(stream_row(
                        run, sequence, "", measurement_time, actual,
                        message_stamp_s=measurement_time,
                        receipt_time_s=measurement_time))
            print("[CALIBRATION] %s: complete" % run["run_id"], flush=True)
    finally:
        env.close()

    if args_cli.output_dir:
        output = Path(args_cli.output_dir).expanduser().resolve()
    else:
        output = (Path(__file__).resolve().parents[1] / "logs" /
                  "action_calibration" /
                  datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    result = write_artifacts(str(output), metadata, runs,
                             requested, commanded, feedback)
    print("[CALIBRATION] " + json.dumps(result, ensure_ascii=False), flush=True)
    print("[CALIBRATION] results: %s" % output, flush=True)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()

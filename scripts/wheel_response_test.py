"""单环境恒定轮速响应测试。

该脚本绕过策略网络，使用 wheel_velocity action 直接下发左右轮目标，
用于区分策略抖动与 USD 执行器/PhysX 接触造成的轮速跟踪问题。
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Test Emma wheel velocity response in one environment.")
parser.add_argument("--task", type=str, default="Emma-RL-ObsNav-Direct-v0")
parser.add_argument("--command_seconds", type=float, default=3.0)
parser.add_argument("--settle_seconds", type=float, default=0.5)
parser.add_argument("--output_dir", type=str, default=None)
parser.add_argument("--disable_fabric", action="store_true", default=False)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import csv
from datetime import datetime
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg
import isaaclab.utils.math as math_utils

import emma_rl.tasks  # noqa: F401


SCENARIOS = (
    ("straight_3_3", 3.0, 3.0),
    ("spin_-3_3", -3.0, 3.0),
    ("curve_2_5", 2.0, 5.0),
)


def _diagnostic_metadata(base_env, step_dt: float) -> dict[str, object]:
    """收集影响轮速跟踪的主要配置，随测试结果一起保存。"""

    cfg = base_env.cfg
    articulation_props = cfg.robot.spawn.articulation_props
    actuators = {}
    for name, actuator in cfg.robot.actuators.items():
        actuators[name] = {
            "joint_names_expr": list(actuator.joint_names_expr),
            "velocity_limit": getattr(actuator, "velocity_limit", None),
            "velocity_limit_sim": getattr(actuator, "velocity_limit_sim", None),
            "effort_limit": getattr(actuator, "effort_limit", None),
            "effort_limit_sim": getattr(actuator, "effort_limit_sim", None),
            "stiffness": getattr(actuator, "stiffness", None),
            "damping": getattr(actuator, "damping", None),
        }
    return {
        "task": args_cli.task,
        "device": str(base_env.device),
        "sim_dt_s": float(cfg.sim.dt),
        "decimation": int(cfg.decimation),
        "control_step_dt_s": step_dt,
        "wheel_radius_m": float(cfg.wheel_radius),
        "wheel_base_m": float(cfg.wheel_base),
        "wheel_max_velocity_radps": float(cfg.wheel_max_velocity),
        "ground_static_friction": float(cfg.ground_static_friction),
        "ground_dynamic_friction": float(cfg.ground_dynamic_friction),
        "solver_position_iteration_count": int(
            articulation_props.solver_position_iteration_count
        ),
        "solver_velocity_iteration_count": int(
            articulation_props.solver_velocity_iteration_count
        ),
        "action_control_mode": str(cfg.action_control_mode),
        "step_mode": "physics_only",
        "active_obstacles": int(cfg.random_obstacle_world.num_obstacles if cfg.world_mode == "random_obstacles" else cfg.obs_world.num_obstacles),
        "safety_enabled": bool(cfg.safety_control.enabled),
        "domain_randomization": {
            "lidar_noise": bool(cfg.domain_rand.enable_lidar_noise),
            "lidar_dropout": bool(cfg.domain_rand.enable_lidar_dropout),
            "dynamics_noise": bool(cfg.domain_rand.enable_dynamics_noise),
            "action_noise": bool(cfg.domain_rand.enable_action_noise),
        },
        "actuators": actuators,
    }


def _percentile_abs_delta(values: np.ndarray, percentile: float = 95.0) -> float:
    if values.size < 2:
        return 0.0
    return float(np.percentile(np.abs(np.diff(values)), percentile))


def _mean_abs_delta(values: np.ndarray) -> float:
    if values.size < 2:
        return 0.0
    return float(np.mean(np.abs(np.diff(values))))


def _step_physics_only(base_env, action: torch.Tensor) -> None:
    """推进一个控制周期，但不计算 LiDAR、奖励、观测和终止条件。"""

    base_env._pre_physics_step(action)
    for _ in range(int(base_env.cfg.decimation)):
        if hasattr(base_env, "_sim_step_counter"):
            base_env._sim_step_counter += 1
        base_env._apply_action()
        base_env.scene.write_data_to_sim()
        base_env.sim.step(render=False)
        base_env.scene.update(dt=base_env.physics_dt)


def _actual_body_motion(base_env) -> tuple[torch.Tensor, torch.Tensor]:
    """使用与 ObsNav 奖励一致的定义读取车体前向速度和角速度。"""

    robot = base_env.robot
    forward_w = math_utils.quat_apply(
        robot.data.root_quat_w, robot.data.FORWARD_VEC_B
    )
    forward_w = torch.nn.functional.normalize(forward_w[:, :2], dim=-1)
    forward_speed = torch.sum(
        robot.data.root_com_lin_vel_w[:, :2] * forward_w, dim=-1
    )
    angular_speed = robot.data.root_ang_vel_b[:, 2]
    return forward_speed, angular_speed


def _scenario_summary(rows: list[dict[str, float | str]]) -> dict[str, float]:
    def values(name: str) -> np.ndarray:
        return np.asarray([float(row[name]) for row in rows], dtype=np.float64)

    left_cmd = values("wheel_left_cmd_radps")
    right_cmd = values("wheel_right_cmd_radps")
    left_actual = values("wheel_left_actual_radps")
    right_actual = values("wheel_right_actual_radps")
    expected_vx = values("expected_vx_mps")
    expected_wz = values("expected_wz_radps")
    actual_vx = values("actual_vx_mps")
    actual_wz = values("actual_wz_radps")
    return {
        "samples": len(rows),
        "left_mae_radps": float(np.mean(np.abs(left_actual - left_cmd))),
        "right_mae_radps": float(np.mean(np.abs(right_actual - right_cmd))),
        "left_rmse_radps": float(np.sqrt(np.mean((left_actual - left_cmd) ** 2))),
        "right_rmse_radps": float(np.sqrt(np.mean((right_actual - right_cmd) ** 2))),
        "left_actual_delta_mean_radps": _mean_abs_delta(left_actual),
        "right_actual_delta_mean_radps": _mean_abs_delta(right_actual),
        "left_actual_delta_p95_radps": _percentile_abs_delta(left_actual),
        "right_actual_delta_p95_radps": _percentile_abs_delta(right_actual),
        "vx_mae_mps": float(np.mean(np.abs(actual_vx - expected_vx))),
        "wz_mae_radps": float(np.mean(np.abs(actual_wz - expected_wz))),
    }


def _save_plot(
    records: list[dict[str, float | str]],
    output_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, len(SCENARIOS), figsize=(18, 10), squeeze=False)
    for column, (scenario, _, _) in enumerate(SCENARIOS):
        rows = [row for row in records if row["scenario"] == scenario]
        command_time_s = np.asarray(
            [float(row["command_time_s"]) for row in rows]
        )
        measurement_time_s = np.asarray(
            [float(row["measurement_time_s"]) for row in rows]
        )

        axes[0, column].plot(
            command_time_s,
            [float(row["wheel_left_cmd_radps"]) for row in rows],
            label="left cmd",
        )
        axes[0, column].plot(
            command_time_s,
            [float(row["wheel_right_cmd_radps"]) for row in rows],
            label="right cmd",
        )
        axes[0, column].plot(
            measurement_time_s,
            [float(row["wheel_left_actual_radps"]) for row in rows],
            label="left actual",
            alpha=0.8,
        )
        axes[0, column].plot(
            measurement_time_s,
            [float(row["wheel_right_actual_radps"]) for row in rows],
            label="right actual",
            alpha=0.8,
        )
        axes[0, column].set_title(scenario)
        axes[0, column].set_ylabel("wheel [rad/s]")
        axes[0, column].grid(True)
        axes[0, column].legend(fontsize=8)

        axes[1, column].plot(
            command_time_s,
            [float(row["expected_vx_mps"]) for row in rows],
            label="expected vx",
        )
        axes[1, column].plot(
            measurement_time_s,
            [float(row["actual_vx_mps"]) for row in rows],
            label="actual vx",
        )
        axes[1, column].set_ylabel("vx [m/s]")
        axes[1, column].grid(True)
        axes[1, column].legend(fontsize=8)

        axes[2, column].plot(
            command_time_s,
            [float(row["expected_wz_radps"]) for row in rows],
            label="expected wz",
        )
        axes[2, column].plot(
            measurement_time_s,
            [float(row["actual_wz_radps"]) for row in rows],
            label="actual wz",
        )
        axes[2, column].set_xlabel("time [s]")
        axes[2, column].set_ylabel("wz [rad/s]")
        axes[2, column].grid(True)
        axes[2, column].legend(fontsize=8)

    fig.suptitle("Emma constant wheel velocity response")
    fig.savefig(output_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    if args_cli.command_seconds <= 0.0 or args_cli.settle_seconds < 0.0:
        raise ValueError("command_seconds must be positive and settle_seconds non-negative")

    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=1,
        use_fabric=not args_cli.disable_fabric,
    )
    env_cfg.action_control_mode = "wheel_velocity"
    env_cfg.safety_control.enabled = False
    env_cfg.domain_rand.enable_lidar_noise = False
    env_cfg.domain_rand.enable_lidar_dropout = False
    env_cfg.domain_rand.enable_dynamics_noise = False
    env_cfg.domain_rand.enable_action_noise = False
    # 轮速响应不测试避障，移除活动障碍，避免接触力污染执行器跟踪结果。
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
    step_dt = float(base_env.step_dt)
    metadata = _diagnostic_metadata(base_env, step_dt)
    wheel_limit = float(base_env.cfg.wheel_max_velocity)
    wheel_radius = float(base_env.cfg.wheel_radius)
    wheel_base = float(base_env.cfg.wheel_base)
    command_steps = max(1, round(args_cli.command_seconds / step_dt))
    settle_steps = max(0, round(args_cli.settle_seconds / step_dt))
    progress_interval = max(1, round(1.0 / step_dt))

    if args_cli.output_dir:
        output_dir = Path(args_cli.output_dir).expanduser().resolve()
    else:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_dir = (
            Path(__file__).resolve().parents[1]
            / "logs"
            / "wheel_response"
            / timestamp
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, float | str]] = []
    try:
        for scenario, desired_left, desired_right in SCENARIOS:
            print(f"[TEST] {scenario}: resetting environment...", flush=True)
            env.reset()
            print(
                f"[TEST] {scenario}: command=({desired_left:.1f}, "
                f"{desired_right:.1f}) rad/s, settle={settle_steps} steps, "
                f"record={command_steps} steps",
                flush=True,
            )
            zero_action = torch.zeros((1, 2), device=device)
            with torch.inference_mode():
                for _ in range(settle_steps):
                    _step_physics_only(base_env, zero_action)

                action = torch.tensor(
                    [[desired_left / wheel_limit, desired_right / wheel_limit]],
                    device=device,
                    dtype=torch.float32,
                )
                for step in range(command_steps):
                    _step_physics_only(base_env, action)
                    wheel_cmd = base_env._wheel_vel_cmd[0]
                    left_id = int(base_env.left_ids[0])
                    right_id = int(base_env.right_ids[0])
                    left_actual = base_env.robot.data.joint_vel[0, left_id]
                    right_actual = base_env.robot.data.joint_vel[0, right_id]
                    forward_speed, angular_speed = _actual_body_motion(base_env)
                    expected_vx = 0.5 * wheel_radius * (
                        wheel_cmd[0] + wheel_cmd[1]
                    )
                    expected_wz = wheel_radius * (
                        wheel_cmd[1] - wheel_cmd[0]
                    ) / wheel_base
                    records.append(
                        {
                            "scenario": scenario,
                            "step": step,
                            "command_time_s": step * step_dt,
                            "measurement_time_s": (step + 1) * step_dt,
                            "desired_left_radps": desired_left,
                            "desired_right_radps": desired_right,
                            "wheel_left_cmd_radps": float(wheel_cmd[0].item()),
                            "wheel_right_cmd_radps": float(wheel_cmd[1].item()),
                            "wheel_left_actual_radps": float(left_actual.item()),
                            "wheel_right_actual_radps": float(right_actual.item()),
                            "expected_vx_mps": float(expected_vx.item()),
                            "expected_wz_radps": float(expected_wz.item()),
                            "actual_vx_mps": float(forward_speed[0].item()),
                            "actual_wz_radps": float(angular_speed[0].item()),
                            "projection_scale": float(
                                base_env._wheel_projection_scale[0].item()
                            ),
                        }
                    )
                    if (step + 1) % progress_interval == 0:
                        print(
                            f"[TEST] {scenario}: {step + 1}/{command_steps} "
                            "control steps",
                            flush=True,
                        )
            print(f"[TEST] {scenario}: completed", flush=True)
    finally:
        env.close()

    if not records:
        raise RuntimeError("No wheel response samples were recorded")

    csv_path = output_dir / "wheel_response.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)

    scenario_summaries = {
        scenario: _scenario_summary(
            [row for row in records if row["scenario"] == scenario]
        )
        for scenario, _, _ in SCENARIOS
    }
    summaries = {
        "metadata": metadata,
        "scenarios": scenario_summaries,
    }
    summary_path = output_dir / "wheel_response_summary.json"
    summary_path.write_text(
        json.dumps(summaries, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    plot_path = output_dir / "wheel_response.png"
    _save_plot(records, plot_path)

    print("[CONFIG] " + json.dumps(metadata, ensure_ascii=False))
    for scenario, summary in scenario_summaries.items():
        print(
            f"[RESULT] {scenario}: "
            f"wheel MAE L/R={summary['left_mae_radps']:.3f}/"
            f"{summary['right_mae_radps']:.3f} rad/s, "
            f"actual Δmean L/R={summary['left_actual_delta_mean_radps']:.3f}/"
            f"{summary['right_actual_delta_mean_radps']:.3f} rad/s, "
            f"actual Δ P95 L/R={summary['left_actual_delta_p95_radps']:.3f}/"
            f"{summary['right_actual_delta_p95_radps']:.3f} rad/s, "
            f"vx/wz MAE={summary['vx_mae_mps']:.3f}/"
            f"{summary['wz_mae_radps']:.3f}"
        )
    print(f"[RESULT] CSV: {csv_path}")
    print(f"[RESULT] summary: {summary_path}")
    print(f"[RESULT] plot: {plot_path}")


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()

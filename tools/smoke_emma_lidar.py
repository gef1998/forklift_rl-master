#!/usr/bin/env python3
"""Emma LiDAR 观测测试 — 创建 env + reset + step, 打印 LiDAR 距离.

用法:
  $ISAAC_PYTHON tools/smoke_emma_lidar.py --num_envs 1 --headless
"""

from __future__ import annotations
import argparse, math
import torch

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--steps", type=int, default=100)
parser.add_argument(
    "--action_control_mode",
    choices=["body_twist", "wheel_velocity"],
    default="body_twist",
)
parser.add_argument("--lidar_mode", choices=["synchronous", "fixed", "random"], default="fixed")
parser.add_argument("--lidar_frequency", type=float, default=20.0)
parser.add_argument("--lidar_latency", type=float, default=0.0, help="Fixed latency in seconds")
parser.add_argument("--lidar_jitter_steps", type=int, default=0)
parser.add_argument("--lidar_scan_dropout", type=float, default=0.0)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from emma_rl.tasks.direct.emma.emma_obs_nav.emma_obs_nav_env import EmmaObsNavEnv
from emma_rl.tasks.direct.emma.emma_obs_nav.emma_obs_nav_env_cfg import EmmaObsNavEnvCfg


def main():
    num_envs = args_cli.num_envs
    num_steps = args_cli.steps

    print("=" * 60)
    print("  Emma LiDAR Test")
    print(f"  num_envs={num_envs}, steps={num_steps}")
    print("=" * 60)

    cfg = EmmaObsNavEnvCfg()
    cfg.scene.num_envs = num_envs
    cfg.terrain_num_envs = num_envs
    cfg.action_control_mode = args_cli.action_control_mode
    cfg.lidar.debug_vis = False
    cfg.lidar_sensor_mode = args_cli.lidar_mode
    cfg.lidar_fixed_frequency_hz = args_cli.lidar_frequency
    cfg.lidar_latency_range_s = (args_cli.lidar_latency, args_cli.lidar_latency)
    cfg.lidar_jitter_max_steps = args_cli.lidar_jitter_steps
    cfg.lidar_scan_dropout_prob = args_cli.lidar_scan_dropout

    print(f"\n[Init] Creating Emma env ({num_envs} envs)...")
    env = EmmaObsNavEnv(cfg)

    obs_dict, _ = env.reset()
    obs = obs_dict["policy"]
    print(f"  obs shape: {obs.shape}  (expected dim={cfg.single_obs_dim})")
    print(f"  LiDAR initialized: {env._lidar.is_initialized}")
    print(f"  action control: {cfg.action_control_mode}")
    print(f"  sensor mode: {cfg.lidar_sensor_mode}; "
          f"period range={env._sensor_period.min().item():.3f}..{env._sensor_period.max().item():.3f}s")

    # ---- Collision diagnostic ----
    from pxr import UsdPhysics
    import omni.usd
    stage = omni.usd.get_context().get_stage()
    print("\n=== Cylinder Collision Check ===")
    for obs_i in range(3):
        path = f"/World/envs/env_0/Obstacle_{obs_i}"
        prim = stage.GetPrimAtPath(path)
        if not prim.IsValid():
            print(f"  [{obs_i}] {path} — NOT FOUND")
            continue
        xform = prim.GetAttribute("xformOp:translate")
        pos = list(xform.Get()) if xform else "NO ATTR"
        ca = UsdPhysics.CollisionAPI(prim)
        has_ca = bool(ca)
        ce = ca.GetCollisionEnabledAttr().Get() if has_ca else "N/A"
        print(f"  [{obs_i}] pos=({pos[0]:.1f},{pos[1]:.1f},{pos[2]:.1f})  collision={has_ca}  enabled={ce}")
    print("=" * 30)

    # LiDAR diagnostic
    lidar = env._lidar
    if lidar.data.ray_hits_w is not None:
        hits = lidar.data.ray_hits_w
        pos = lidar.data.pos_w
        dist_2d = torch.norm(hits[..., :2] - pos[:, None, :2], dim=-1).clamp(max=cfg.lidar_max_distance)

        # Directional check: 打印 4 个方位的 distance / rect_proj / clearance
        rect = env._rect_projection()
        clearance_raw = dist_2d - rect
        # 1440 rays @ 0.25°: 0/360/720/1080 correspond to cardinal directions.
        dirs = {"front(0°)": 0, "left(90°)": 360, "rear(180°)": 720, "right(270°)": 1080}
        print("\n=== Directional Clearance ===")
        for label, idx in dirs.items():
            d_val = dist_2d[0, idx].item()
            r_val = rect[0, idx].item()
            c_val = clearance_raw[0, idx].item()
            print(f"  {label}: dist={d_val:.2f}  rect_proj={r_val:.2f}  clearance={c_val:.2f}")
        print("  → front clearance should be large (open space ahead), rear may be small")

        print(f"\n=== LiDAR Stats (after reset) ===")
        for env_i in range(min(num_envs, 4)):
            d = dist_2d[env_i]
            print(f"  env_{env_i}: min={d.min():.3f} max={d.max():.3f} mean={d.mean():.3f}")
            print(f"           hits<1m={(d<1).sum().item()} hits<2m={(d<2).sum().item()} "
                  f"hits<5m={(d<5).sum().item()} hits<8m={(d<8).sum().item()}")
            # Sector analysis
            n = cfg.num_lidar_rays
            front = d[:n//4].mean().item()
            right = d[n//4:n//2].mean().item()
            rear = d[n//2:3*n//4].mean().item()
            left = d[3*n//4:].mean().item()
            print(f"           front={front:.2f} right={right:.2f} rear={rear:.2f} left={left:.2f}")

        print(f"\n  LiDAR data.pos_w[0]  = ({pos[0,0]:.2f}, {pos[0,1]:.2f}, {pos[0,2]:.2f})  # prim 位置, 不含 offset")
        lidar_actual_pos = env._get_lidar_sensor_pos_w()
        print(f"  LiDAR actual pos[0]  = ({lidar_actual_pos[0,0]:.2f}, {lidar_actual_pos[0,1]:.2f}, "
              f"{lidar_actual_pos[0,2]:.2f})  # 含 offset (0.38, 0, 0.12)")
        print(f"  Robot root[0]        = ({env.robot.data.root_pos_w[0,0]:.2f}, "
              f"{env.robot.data.root_pos_w[0,1]:.2f}, {env.robot.data.root_pos_w[0,2]:.2f})")

    # Step loop
    print(f"\n=== Step loop ({num_steps} steps) ===")
    prev_mean = None

    for step in range(1, num_steps + 1):
        actions = 2.0 * torch.rand(
            num_envs, cfg.action_space, device=env.device) - 1.0
        obs_dict, reward, terminated, truncated, _ = env.step(actions)

        if lidar.data.ray_hits_w is not None:
            dist = torch.norm(
                lidar.data.ray_hits_w[..., :2] - lidar.data.pos_w[:, None, :2], dim=-1
            ).clamp(max=cfg.lidar_max_distance)
        else:
            dist = torch.zeros(num_envs, cfg.num_lidar_rays)

        if step % 10 == 0 or step <= 3:
            n_term = terminated.sum().item()
            generated = env._last_sensor_generation_mask.sum().item()
            delivered = env._last_sensor_delivery_mask.sum().item()
            safety_min = env._safety_clearance.amin(dim=-1).mean().item()
            collisions = env._collision_mask.sum().item()
            print(f"  [{step:4d}] reward={reward.mean():+.3f} term={n_term}/{num_envs} | "
                  f"LiDAR min={dist.min():.3f} mean={dist.mean():.3f} max={dist.max():.3f} | "
                  f"gen={generated} deliver={delivered} "
                  f"AoI={env._lidar_aoi.mean().item() * 1000:.1f}ms "
                  f"safety_min={safety_min:.3f}m collision={collisions}")
            if hasattr(env, "_last_approach_gate"):
                print(
                    f"          approach_gate={env._last_approach_gate.mean().item():.3f} "
                    f"approach={env._last_approach_reward.mean().item():+.3f} "
                    f"repulse={env._last_repulse_reward.mean().item():+.3f}"
                )

            # Check LiDAR changes
            curr_mean = dist.mean(dim=-1)
            if prev_mean is not None:
                delta = (curr_mean - prev_mean).abs().max().item()
                if delta > 0.01:
                    print(f"          LiDAR changing: |Δmean|={delta:.4f}m ✓")
                elif step > 10:
                    print(f"          [WARN] LiDAR mean static (robot stuck or all rays at max_range)")
            prev_mean = curr_mean.clone()

    print(f"\n[OK] Test complete")
    simulation_app.close()


if __name__ == "__main__":
    main()

"""Preview actual dedicated-detour reference paths without loading a policy.

Headless exports PNGs and exits; a window shows the last selected level until
closed. No actions are issued: this is a layout/path preview, not a controller.
"""
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--task', default='Emma-RL-ObsNav-GRU-Hybrid180-Direct-v0')
parser.add_argument('--num_envs', type=int, default=1)
parser.add_argument('--env_ids', type=int, nargs='+', default=None,
                    help='Dedicated env IDs to show; default first dedicated env.')
parser.add_argument('--level', type=int, choices=range(6), default=5)
parser.add_argument('--all_levels', action='store_true')
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--output_dir', default='logs/reference_paths')
parser.add_argument('--min_turning_radius', type=float, default=None,
                    help='Override moving reference radius in metres (config default: 0.50).')
parser.add_argument('--no_rotation_fallback', action='store_true',
                    help='Require planning without in-place rotations; failures remain visible.')
parser.add_argument('--frames', type=int, default=0,
                    help='GUI frames before exit; 0 waits for window close. Headless renders 12 frames.')
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.num_envs < 1 or args.frames < 0:
    parser.error('num_envs must be positive and frames nonnegative')
if args.min_turning_radius is not None and not (0 < args.min_turning_radius < float('inf')):
    parser.error('min_turning_radius must be finite and positive')
launcher = AppLauncher(args)
app = launcher.app


def main():
    import gymnasium as gym
    import numpy as np
    import torch
    import omni.usd
    from pxr import UsdGeom, UsdPhysics
    import isaaclab_tasks  # noqa: F401
    import emma_rl.tasks  # noqa: F401
    from isaaclab_tasks.utils import parse_env_cfg

    cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    cfg.seed = args.seed
    cfg.world_mode = 'random_obstacles'
    cfg.curriculum.enabled = False
    cfg.curriculum.evaluation_level = 0 if args.all_levels else args.level
    cfg.random_obstacle_world.dedicated_detour_env_count = max(1, cfg.random_obstacle_world.dedicated_detour_env_count)
    cfg.reference_path_reward_enabled = True
    cfg.reference_path_backend = 'hybrid_astar'
    if args.min_turning_radius is not None:
        cfg.reference_planner_min_turning_radius = args.min_turning_radius
    if args.no_rotation_fallback:
        cfg.reference_planner_inplace_rotation_fallback = False
    cfg.reference_path_visualize = True
    cfg.reference_path_visualize_env_ids = tuple(args.env_ids or ())
    cfg.reference_path_export_png = True
    cfg.reference_path_export_dir = args.output_dir
    cfg.reference_path_export_max_per_env = 0
    # Small preview does not need the training-scale GPU buffer reservations.
    if args.num_envs <= 8:
        cfg.sim.physx.gpu_found_lost_pairs_capacity = 2**20
        cfg.sim.physx.gpu_total_aggregate_pairs_capacity = 2**20
        cfg.sim.physx.gpu_collision_stack_size = 2**24
    env = gym.make(args.task, cfg=cfg)
    try:
        base = env.unwrapped
        levels = range(6) if args.all_levels else (args.level,)
        for level in levels:
            base._curriculum.level = level
            with torch.inference_mode():
                obs, _ = env.reset()
            assert torch.isfinite(obs['policy']).all()
        selected = cfg.reference_path_visualize_env_ids or tuple(sorted(base._world_manager.dedicated_detour_layouts)[:1])
        stage = omni.usd.get_context().get_stage()
        for env_id in selected:
            root = stage.GetPrimAtPath(f'/World/ReferencePathDebug/env_{env_id}')
            assert root, f'No reference overlay for env {env_id}'
            transform = UsdGeom.Xformable(root).ComputeLocalToWorldTransform(0)
            np.testing.assert_allclose(np.asarray(transform.ExtractTranslation()), base.scene.env_origins[env_id].cpu().numpy())
            for child in root.GetChildren():
                assert not child.HasAPI(UsdPhysics.CollisionAPI)
        origin = base.scene.env_origins[selected[0]].cpu().numpy()
        base.sim.set_camera_view(origin + np.array([0., -.01, 14.]), origin)
        print(f'[ReferencePreview] envs={selected}, output={base._reference_visual_output.resolve()}', flush=True)
        frames = args.frames or (12 if args.headless else 0)
        frame = 0
        while app.is_running() and (not frames or frame < frames):
            base.sim.render()
            frame += 1
        print('[ReferencePreview] complete', flush=True)
    finally:
        env.close()


try:
    main()
finally:
    app.close()

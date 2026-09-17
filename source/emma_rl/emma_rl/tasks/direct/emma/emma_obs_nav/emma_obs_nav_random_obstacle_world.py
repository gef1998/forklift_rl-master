"""Isaac Sim 实现：随机连通内墙与圆柱/方柱联合生成场景。"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

import numpy as np
import torch
import trimesh

import isaaclab.sim as sim_utils
from isaaclab.utils.warp import convert_to_warp_mesh

from .structured_random_geometry import sample_connected_layout

from .random_obstacle_geometry import (
    alternating_obstacle_types,
    outer_wall_specs,
    sample_random_obstacle_positions,
    sample_safe_spawn_pairs,
)


@dataclass
class RandomObstacleWorldConfig:
    """随机混合障碍课程场景的几何和安全采样配置。"""

    arena_size: float = 10.0
    wall_thickness: float = 0.15
    wall_height: float = 1.0
    num_obstacles: int = 12

    radius_min: float = 0.20
    radius_max: float = 0.30
    cube_size_min: float = 0.30
    cube_size_max: float = 0.50
    obstacle_height_min: float = 1.0
    obstacle_height_max: float = 1.2

    # 几何边缘间距，不是中心距。方柱按任意旋转下的外接圆检查。
    min_obstacle_spacing: float = 1.50
    obstacle_wall_clearance: float = 0.30

    # Emma 半对角线约 0.50m，再保留约 0.25m 出生余量。
    emma_spawn_clearance_radius: float = 0.75
    # Goal 成功半径 0.30m，再保留约 0.30m 障碍/外墙余量。
    goal_spawn_clearance_radius: float = 0.60
    # 配置略高于需求，避免浮点边界把“等于5m”当作合法。
    goal_min_distance: float = 5.20
    obstacle_candidate_count: int = 32
    spawn_candidate_count: int = 96

    # One procedural distribution: connected wall graph + independently sampled clutter.
    structured_layout: bool = True
    grid_size: int = 3
    interior_wall_thickness: float = 0.15
    wall_jitter: float = 0.20
    loop_probability_range: tuple[float, float] = (0.0, 0.65)
    clutter_density_range: tuple[float, float] = (0.35, 1.0)
    navigation_clearance: float = 0.60
    connectivity_grid_resolution: float = 0.15
    spawn_bank_size: int = 128
    curriculum_distance_ranges: tuple = ((1.5, 3.0), (3.0, 5.2))
    spawn_heading_half_angles_deg: tuple = (30.0, 90.0, 180.0)

    @property
    def num_wall_slots(self) -> int:
        return 2 * self.grid_size * (self.grid_size - 1) if self.structured_layout else 0

    @property
    def interior_wall_length(self) -> float:
        return (self.arena_size - 2 * self.wall_thickness) / self.grid_size

    _obstacle_types: list[str] = field(default_factory=list)
    _obstacle_radii: list[float] = field(default_factory=list)
    _obstacle_cube_sizes: list[tuple[float, float]] = field(default_factory=list)
    _obstacle_bound_radii: list[float] = field(default_factory=list)
    _obstacle_heights: list[float] = field(default_factory=list)


def _ensure_collision_api(prim_path: str) -> None:
    from pxr import UsdPhysics
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(prim_path)
    if prim.IsValid() and not UsdPhysics.CollisionAPI(prim):
        UsdPhysics.CollisionAPI.Apply(prim)


def spawn_random_obstacle_world(cfg: RandomObstacleWorldConfig) -> None:
    """在 env_0 创建四面外墙和圆柱/方柱模板，随后由场景克隆。"""

    if cfg.num_obstacles < 0:
        raise ValueError("random_obstacles num_obstacles must be positive")
    for name, cx, cy, sx, sy in outer_wall_specs(
        cfg.arena_size, cfg.wall_thickness
    ):
        wall_cfg = sim_utils.CuboidCfg(
            size=(sx, sy, cfg.wall_height),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True, kinematic_enabled=True
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.35, 0.35, 0.40)
            ),
        )
        path = f"/World/envs/env_0/RandomObstacleWorld_{name}"
        wall_cfg.func(path, wall_cfg, translation=(cx, cy, cfg.wall_height / 2.0))
        _ensure_collision_api(path)

    for index in range(cfg.num_wall_slots):
        wall_cfg = sim_utils.CuboidCfg(
            size=(cfg.interior_wall_length, cfg.interior_wall_thickness, cfg.wall_height),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True, kinematic_enabled=True
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.40, 0.45, 0.50)),
        )
        path = f"/World/envs/env_0/RandomInteriorWall_{index}"
        wall_cfg.func(path, wall_cfg, translation=(0.0, 0.0, -100.0))
        _ensure_collision_api(path)

    cfg._obstacle_types = []
    cfg._obstacle_radii = []
    cfg._obstacle_cube_sizes = []
    cfg._obstacle_bound_radii = []
    cfg._obstacle_heights = []
    for obstacle_index, obstacle_type in enumerate(
        alternating_obstacle_types(cfg.num_obstacles)
    ):
        height = random.uniform(cfg.obstacle_height_min, cfg.obstacle_height_max)
        radius = 0.0
        cube_size = (0.0, 0.0)
        common = dict(
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True, kinematic_enabled=True
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        )
        if obstacle_type == "cylinder":
            radius = random.uniform(cfg.radius_min, cfg.radius_max)
            bound_radius = radius
            spawn_cfg = sim_utils.CylinderCfg(
                radius=radius,
                height=height,
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.85, 0.20, 0.08)
                ),
                **common,
            )
        else:
            size_x = random.uniform(cfg.cube_size_min, cfg.cube_size_max)
            size_y = random.uniform(cfg.cube_size_min, cfg.cube_size_max)
            cube_size = (size_x, size_y)
            bound_radius = 0.5 * math.hypot(size_x, size_y)
            spawn_cfg = sim_utils.CuboidCfg(
                size=(size_x, size_y, height),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.15, 0.35, 0.85)
                ),
                **common,
            )

        cfg._obstacle_types.append(obstacle_type)
        cfg._obstacle_radii.append(radius)
        cfg._obstacle_cube_sizes.append(cube_size)
        cfg._obstacle_bound_radii.append(bound_radius)
        cfg._obstacle_heights.append(height)
        path = f"/World/envs/env_0/RandomObstacle_{obstacle_index}"
        # 模板在首次 reset 写入安全随机位置前先放到地下。
        spawn_cfg.func(path, spawn_cfg, translation=(0.0, 0.0, -100.0))
        _ensure_collision_api(path)


class RandomObstacleWorldManager:
    """管理随机障碍、出生点和自定义 LiDAR mesh。"""

    def __init__(
        self,
        num_envs: int,
        cfg: RandomObstacleWorldConfig,
        device: str,
    ):
        self.num_envs = num_envs
        self.cfg = cfg
        self.device = device
        self.num_obstacles = cfg.num_obstacles
        self.types = tuple(cfg._obstacle_types)
        self.radii = torch.tensor(cfg._obstacle_radii, device=device)
        self.cube_sizes = torch.tensor(cfg._obstacle_cube_sizes, device=device)
        self.bound_radii = torch.tensor(cfg._obstacle_bound_radii, device=device)
        self.heights = torch.tensor(cfg._obstacle_heights, device=device)
        if len(self.types) != self.num_obstacles:
            raise RuntimeError(
                "random-obstacle templates were not initialized"
            )
        self.positions = torch.zeros(
            num_envs, self.num_obstacles, 2, device=device
        )
        self.yaws = torch.zeros(num_envs, self.num_obstacles, device=device)

        self.active = torch.ones(num_envs, self.num_obstacles, dtype=torch.bool, device=device)
        self.wall_poses = torch.zeros(num_envs, cfg.num_wall_slots, 3, device=device)
        self.wall_active = torch.zeros(num_envs, cfg.num_wall_slots, dtype=torch.bool, device=device)
        self.spawn_pairs = None
        self.curriculum_spawn_pairs = None
        if len(cfg.spawn_heading_half_angles_deg) != 3 or any(
            not math.isfinite(a) or not 0 <= a <= 180 for a in cfg.spawn_heading_half_angles_deg
        ):
            raise ValueError("three spawn heading half-angles in [0, 180] are required")

    def sample_initial_layout(self) -> None:
        if self.cfg.structured_layout:
            # Draw seeds from Torch so the existing training seed also controls CPU geometry.
            seeds = torch.randint(0, 2**31 - 1, (self.num_envs,), device=self.device).cpu().tolist()
            cfg = self.cfg
            layouts = [sample_connected_layout(
                cfg._obstacle_bound_radii, seed=seed, arena_size=cfg.arena_size,
                wall_thickness=cfg.wall_thickness, grid_size=cfg.grid_size,
                interior_wall_thickness=cfg.interior_wall_thickness,
                wall_jitter=cfg.wall_jitter, loop_probability_range=cfg.loop_probability_range,
                clutter_density_range=cfg.clutter_density_range,
                min_obstacle_spacing=cfg.min_obstacle_spacing,
                obstacle_wall_clearance=cfg.obstacle_wall_clearance,
                navigation_clearance=cfg.navigation_clearance,
                grid_resolution=cfg.connectivity_grid_resolution,
                emma_clearance=cfg.emma_spawn_clearance_radius,
                goal_clearance=cfg.goal_spawn_clearance_radius,
                min_goal_distance=cfg.goal_min_distance, spawn_bank_size=cfg.spawn_bank_size,
                include_curriculum_banks=True,
                curriculum_distance_ranges=cfg.curriculum_distance_ranges,
            ) for seed in seeds]
            for name in ("positions", "active", "wall_poses", "wall_active"):
                target = getattr(self, name)
                target.copy_(torch.as_tensor(np.stack([layout[name] for layout in layouts]),
                                             device=self.device, dtype=target.dtype))
            self.spawn_pairs = torch.as_tensor(
                np.stack([layout["spawn_pairs"] for layout in layouts]),
                device=self.device, dtype=self.positions.dtype,
            )
            self.curriculum_spawn_pairs = torch.as_tensor(
                np.stack([layout["curriculum_pairs"] for layout in layouts]),
                device=self.device, dtype=self.positions.dtype,
            )
            self.yaws.uniform_(0, 2 * math.pi)
            return
        self.positions.copy_(
            sample_random_obstacle_positions(
                self.num_envs,
                self.bound_radii,
                arena_size=self.cfg.arena_size,
                wall_thickness=self.cfg.wall_thickness,
                min_edge_spacing=self.cfg.min_obstacle_spacing,
                obstacle_wall_clearance=self.cfg.obstacle_wall_clearance,
                candidate_pool_size=self.cfg.obstacle_candidate_count,
            )
        )
        self.yaws.copy_(
            torch.rand(
                self.num_envs, self.num_obstacles, device=self.device
            )
            * (2.0 * math.pi)
        )

    def write_poses_to_sim(
        self,
        scene,
        env_origins_xy: torch.Tensor,
    ) -> None:
        """将每个环境的随机障碍位置和方柱朝向写入 PhysX。"""

        for obstacle_index in range(self.num_obstacles):
            obstacle = scene.rigid_objects.get(
                f"random_obstacle_{obstacle_index}"
            )
            if obstacle is None or obstacle.data.default_root_state is None:
                raise RuntimeError(
                    f"missing random-obstacle asset {obstacle_index}"
                )
            state = obstacle.data.default_root_state.clone()
            state_view = state.view(self.num_envs, -1, 13)
            state_view[:, 0, 0:2] = (
                self.positions[:, obstacle_index] + env_origins_xy
            )
            state_view[:, 0, 2] = torch.where(
                self.active[:, obstacle_index], self.heights[obstacle_index] / 2.0, -100.0
            )
            state_view[:, 0, 3:7] = 0.0
            yaw = self.yaws[:, obstacle_index]
            state_view[:, 0, 3] = torch.cos(yaw / 2.0)
            state_view[:, 0, 6] = torch.sin(yaw / 2.0)
            state_view[:, 0, 7:13] = 0.0
            obstacle.write_root_state_to_sim(state)

        for index in range(self.cfg.num_wall_slots):
            wall = scene.rigid_objects[f"random_interior_wall_{index}"]
            state = wall.data.default_root_state.clone()
            view = state.view(self.num_envs, -1, 13)
            view[:, 0, :2] = self.wall_poses[:, index, :2] + env_origins_xy
            view[:, 0, 2] = torch.where(self.wall_active[:, index], self.cfg.wall_height / 2, -100.0)
            view[:, 0, 3:13] = 0
            yaw = self.wall_poses[:, index, 2]
            view[:, 0, 3] = torch.cos(yaw / 2)
            view[:, 0, 6] = torch.sin(yaw / 2)
            wall.write_root_state_to_sim(state)

    def sample_spawn_pairs(
        self, env_ids: torch.Tensor, task_levels: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.cfg.structured_layout:
            if self.spawn_pairs is None:
                raise RuntimeError("sample_initial_layout must precede spawn sampling")
            choice = torch.randint(self.cfg.spawn_bank_size, (env_ids.numel(),), device=self.device)
            if task_levels is None:
                task_levels = torch.full_like(env_ids, 2)
            # Levels 2 and 3 share full geometry; level 3 adds stronger noise.
            geometry_levels = task_levels.clamp(max=2)
            pairs = self.curriculum_spawn_pairs[env_ids, geometry_levels, choice]
            emma, goal = pairs[:, 0], pairs[:, 1]
            heading_half = torch.tensor(self.cfg.spawn_heading_half_angles_deg,
                                        device=self.device)[geometry_levels] * (math.pi / 180)
            direction = torch.atan2(goal[:, 1] - emma[:, 1], goal[:, 0] - emma[:, 0])
            yaw = direction + (2 * torch.rand(env_ids.numel(), device=self.device) - 1) * heading_half
            yaw = torch.remainder(yaw, 2 * math.pi)
            emma_region = (emma[:, 0] >= 0).long() + 2 * (emma[:, 1] < 0).long()
            goal_region = (goal[:, 0] >= 0).long() + 2 * (goal[:, 1] < 0).long()
            return emma, yaw, goal, emma_region, goal_region
        return sample_safe_spawn_pairs(
            self.positions[env_ids],
            self.bound_radii,
            arena_size=self.cfg.arena_size,
            wall_thickness=self.cfg.wall_thickness,
            emma_clearance_radius=self.cfg.emma_spawn_clearance_radius,
            goal_clearance_radius=self.cfg.goal_spawn_clearance_radius,
            min_goal_distance=self.cfg.goal_min_distance,
            candidate_pool_size=self.cfg.spawn_candidate_count,
        )

    def build_warp_mesh(self, env_origins: torch.Tensor, device: str):
        """构建地面、外墙及每环境随机混合障碍的 LiDAR mesh。"""

        meshes: list[trimesh.Trimesh] = []
        total_size = self.cfg.arena_size * max(int(math.sqrt(self.num_envs)), 1)
        ground = trimesh.creation.box(extents=[total_size, total_size, 0.02])
        ground.apply_translation([0.0, 0.0, -0.01])
        meshes.append(ground)
        walls = outer_wall_specs(self.cfg.arena_size, self.cfg.wall_thickness)

        for env_index in range(self.num_envs):
            origin_x = float(env_origins[env_index, 0].item())
            origin_y = float(env_origins[env_index, 1].item())
            for _, wall_x, wall_y, size_x, size_y in walls:
                wall = trimesh.creation.box(
                    extents=[size_x, size_y, self.cfg.wall_height]
                )
                wall.apply_translation(
                    [origin_x + wall_x, origin_y + wall_y, self.cfg.wall_height / 2.0]
                )
                meshes.append(wall)

            for index in range(self.cfg.num_wall_slots):
                if not self.wall_active[env_index, index].item():
                    continue
                x, y, yaw = self.wall_poses[env_index, index].cpu().tolist()
                wall = trimesh.creation.box(extents=[self.cfg.interior_wall_length,
                                                    self.cfg.interior_wall_thickness,
                                                    self.cfg.wall_height])
                wall.apply_transform(trimesh.transformations.rotation_matrix(yaw, [0, 0, 1]))
                wall.apply_translation([origin_x + x, origin_y + y, self.cfg.wall_height / 2])
                meshes.append(wall)

            for obstacle_index in range(self.num_obstacles):
                if not self.active[env_index, obstacle_index].item():
                    continue
                height = float(self.heights[obstacle_index].item())
                world_x = float(self.positions[env_index, obstacle_index, 0].item()) + origin_x
                world_y = float(self.positions[env_index, obstacle_index, 1].item()) + origin_y
                if self.types[obstacle_index] == "cylinder":
                    mesh = trimesh.creation.cylinder(
                        radius=float(self.radii[obstacle_index].item()),
                        height=height,
                        sections=24,
                    )
                else:
                    size_x = float(self.cube_sizes[obstacle_index, 0].item())
                    size_y = float(self.cube_sizes[obstacle_index, 1].item())
                    mesh = trimesh.creation.box(extents=[size_x, size_y, height])
                    yaw = float(self.yaws[env_index, obstacle_index].item())
                    mesh.apply_transform(
                        trimesh.transformations.rotation_matrix(
                            yaw, [0.0, 0.0, 1.0]
                        )
                    )
                mesh.apply_translation([world_x, world_y, height / 2.0])
                meshes.append(mesh)

        merged = trimesh.util.concatenate(meshes)
        return convert_to_warp_mesh(merged.vertices, merged.faces, device=device)

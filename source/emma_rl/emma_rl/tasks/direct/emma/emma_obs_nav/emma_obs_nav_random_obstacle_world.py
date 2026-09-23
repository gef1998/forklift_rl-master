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
from .straight_detour_geometry import sample_straight_detour_layout

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
    navigation_clearance: float = 0.52
    connectivity_grid_resolution: float = 0.15
    spawn_bank_size: int = 128
    max_layout_attempts: int = 64
    curriculum_distance_ranges: tuple = ((1.5, 3.0), (3.0, 5.2))
    spawn_heading_half_angles_deg: tuple = (30.0, 90.0, 180.0)
    narrow_passage_probability: float = 0.85
    narrow_passage_width_range: tuple[float, float] = (1.20, 1.50)
    max_narrow_passages: int = 2
    narrow_pair_fraction: float = 0.50
    dense_clutter_influence_radius: float = 1.25
    ultra_dense_clutter_influence_radius: float = 0.70
    min_connected_free_fraction: float = 0.60

    # Reserve the final N parallel environments for a sealed straight-line
    # detour task. Geometry is sampled once at startup so the static Warp mesh
    # and PhysX collision geometry remain identical throughout training.
    dedicated_detour_env_count: int = 128
    dedicated_detour_lane_count: int = 3
    # One explicit width range per curriculum level, easiest to hardest.
    # Every lane contains two level-specific gates plus one shared wide gate.
    dedicated_detour_curriculum_passage_width_ranges: tuple[
        tuple[float, float], ...
    ] = (
            (2.0, 2.5),
            (1.6, 2.2),
            (1.6, 1.9),
            (1.3, 1.7),
            (1.0, 1.5),
            (0.7, 1.2),
    )
    dedicated_detour_passage_width_samples_per_level: int = 3
    dedicated_detour_barrier_count_range: tuple[int, int] = (3, 3)
    dedicated_detour_barrier_y_span: float = 2.70
    dedicated_detour_extra_obstacle_count_range: tuple[int, int] = (3, 6)
    dedicated_detour_obstacle_wall_clearance: float = 0.10
    dedicated_detour_robot_half_width: float = 0.275
    dedicated_detour_robot_half_length: float = 0.425
    dedicated_detour_collision_clearance: float = 0.01
    dedicated_detour_spawn_y: float = 3.55
    dedicated_detour_spawn_lateral_jitter: float = 0.20
    dedicated_detour_heading_half_angle_deg: float = 15.0

    @property
    def num_graph_wall_slots(self) -> int:
        return 2 * self.grid_size * (self.grid_size - 1) if self.structured_layout else 0

    @property
    def num_wall_slots(self) -> int:
        if not self.structured_layout:
            return 0
        return self.base_num_wall_slots + self.dedicated_detour_barrier_slots

    @property
    def base_num_wall_slots(self) -> int:
        if not self.structured_layout:
            return 0
        return self.num_graph_wall_slots + 2 * self.max_narrow_passages

    @property
    def dedicated_detour_barrier_slots(self) -> int:
        if self.dedicated_detour_env_count <= 0:
            return 0
        return self.dedicated_detour_lane_count * (
            2 * self.dedicated_detour_passage_width_samples_per_level + 1
        )

    @property
    def interior_wall_length(self) -> float:
        return (self.arena_size - 2 * self.wall_thickness) / self.grid_size

    @property
    def narrow_gate_wall_length(self) -> float:
        return (self.interior_wall_length - self.narrow_passage_width_range[1]) / 2

    @property
    def dedicated_detour_passage_width_range(self) -> tuple[float, float]:
        """Overall geometry bounds derived from the explicit curriculum tiers."""
        ranges = self.dedicated_detour_curriculum_passage_width_ranges
        if len(ranges) != 2 * self.dedicated_detour_lane_count:
            raise ValueError(
                "dedicated detour requires two passage-width ranges per lane"
            )
        parsed = []
        for values in ranges:
            if len(values) != 2:
                raise ValueError(
                    "each dedicated passage-width range must have two values"
                )
            low, high = values
            if (not math.isfinite(low) or not math.isfinite(high)
                    or low <= 0.0 or low > high):
                raise ValueError("invalid dedicated passage-width range")
            parsed.append((low, high))
        if any(
            parsed[index][bound] < parsed[index + 1][bound]
            for index in range(len(parsed) - 1)
            for bound in (0, 1)
        ):
            raise ValueError(
                "dedicated passage-width ranges must run easiest to hardest"
            )
        return (
            min(low for low, _ in parsed),
            max(high for _, high in parsed),
        )

    @property
    def wall_slot_lengths(self) -> tuple[float, ...]:
        base = ((self.interior_wall_length,) * self.num_graph_wall_slots
                + (self.narrow_gate_wall_length,) * (2 * self.max_narrow_passages))
        if not self.dedicated_detour_barrier_slots:
            return base
        lane_pitch = (
            self.arena_size - 2 * self.wall_thickness
        ) / self.dedicated_detour_lane_count
        # Validate the explicit ranges before deriving the fixed wall assets.
        _ = self.dedicated_detour_passage_width_range
        level_ranges = self.dedicated_detour_curriculum_passage_width_ranges
        samples = self.dedicated_detour_passage_width_samples_per_level
        if not isinstance(samples, int) or samples < 1:
            raise ValueError(
                "dedicated passage-width samples per level must be positive"
            )
        if self.dedicated_detour_barrier_count_range != (3, 3):
            raise ValueError(
                "six-level dedicated curriculum requires exactly three barriers per lane"
            )
        patterns = []
        for lane in range(self.dedicated_detour_lane_count):
            lower_range = level_ranges[2 * lane]
            upper_range = level_ranges[2 * lane + 1]
            available_span = lane_pitch
            if 0 < lane < self.dedicated_detour_lane_count - 1:
                available_span -= self.interior_wall_thickness / 2
            def sampled_lengths(width_range):
                low, high = width_range
                if samples == 1:
                    widths = ((low + high) / 2,)
                else:
                    widths = tuple(
                        low + (high - low) * index / (samples - 1)
                        for index in range(samples)
                    )
                return tuple(available_span - width for width in widths)

            patterns.extend(sampled_lengths(lower_range))
            # The shared middle gate is wider than both level-specific gates.
            patterns.append(
                available_span - max(lower_range[1], upper_range[1])
            )
            patterns.extend(sampled_lengths(upper_range))
        return base + tuple(patterns)

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

    for index, wall_length in enumerate(cfg.wall_slot_lengths):
        wall_cfg = sim_utils.CuboidCfg(
            size=(wall_length, cfg.interior_wall_thickness, cfg.wall_height),
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
        if not isinstance(cfg.dedicated_detour_env_count, int) or cfg.dedicated_detour_env_count < 0:
            raise ValueError("dedicated_detour_env_count must be a non-negative integer")
        if cfg.dedicated_detour_env_count and not cfg.structured_layout:
            raise ValueError("dedicated detour environments require structured_layout")
        self.dedicated_detour_start = max(
            0, self.num_envs - min(cfg.dedicated_detour_env_count, self.num_envs)
        )
        self.dedicated_detour_layouts: dict[int, dict] = {}
        if not math.isfinite(cfg.dedicated_detour_heading_half_angle_deg) or not (
            0.0 <= cfg.dedicated_detour_heading_half_angle_deg <= 180.0
        ):
            raise ValueError("dedicated detour heading half-angle must be in [0, 180]")
        if len(cfg.spawn_heading_half_angles_deg) != 3 or any(
            not math.isfinite(a) or not 0 <= a <= 180 for a in cfg.spawn_heading_half_angles_deg
        ):
            raise ValueError("three spawn heading half-angles in [0, 180] are required")

    def sample_initial_layout(self) -> None:
        if self.cfg.structured_layout:
            # Draw seeds from Torch so the existing training seed also controls CPU geometry.
            seeds = torch.randint(0, 2**31 - 1, (self.num_envs,), device=self.device).cpu().tolist()
            cfg = self.cfg

            def sample_one(seed, density_range, max_attempts):
                return sample_connected_layout(
                    cfg._obstacle_bound_radii, seed=seed,
                    arena_size=cfg.arena_size,
                    wall_thickness=cfg.wall_thickness, grid_size=cfg.grid_size,
                    interior_wall_thickness=cfg.interior_wall_thickness,
                    wall_jitter=cfg.wall_jitter,
                    loop_probability_range=cfg.loop_probability_range,
                    clutter_density_range=density_range,
                    min_obstacle_spacing=cfg.min_obstacle_spacing,
                    obstacle_wall_clearance=cfg.obstacle_wall_clearance,
                    navigation_clearance=cfg.navigation_clearance,
                    grid_resolution=cfg.connectivity_grid_resolution,
                    emma_clearance=cfg.emma_spawn_clearance_radius,
                    goal_clearance=cfg.goal_spawn_clearance_radius,
                    min_goal_distance=cfg.goal_min_distance,
                    spawn_bank_size=cfg.spawn_bank_size,
                    max_layout_attempts=max_attempts,
                    include_curriculum_banks=True,
                    curriculum_distance_ranges=cfg.curriculum_distance_ranges,
                    narrow_passage_probability=cfg.narrow_passage_probability,
                    narrow_passage_width_range=cfg.narrow_passage_width_range,
                    max_narrow_passages=cfg.max_narrow_passages,
                    narrow_pair_fraction=cfg.narrow_pair_fraction,
                    dense_clutter_influence_radius=(
                        cfg.dense_clutter_influence_radius
                    ),
                    ultra_dense_clutter_influence_radius=(
                        cfg.ultra_dense_clutter_influence_radius
                    ),
                    min_connected_free_fraction=cfg.min_connected_free_fraction,
                )

            density_low, density_high = cfg.clutter_density_range
            fallback_ranges = (
                (max(0.0, density_low - 0.20),
                 max(0.0, density_high - 0.20)),
                (max(0.0, density_low - 0.35),
                 max(0.0, density_high - 0.35)),
            )
            layouts = []
            fallback_records = []
            for env_index, seed in enumerate(seeds):
                try:
                    layout = sample_one(
                        seed, cfg.clutter_density_range,
                        cfg.max_layout_attempts,
                    )
                except RuntimeError as initial_error:
                    layout = None
                    for fallback_level, density_range in enumerate(
                            fallback_ranges, start=1):
                        retry_seed = (
                            seed ^ (0x6A09E667 * fallback_level)
                        ) & 0x7FFFFFFF
                        try:
                            layout = sample_one(
                                retry_seed, density_range,
                                cfg.max_layout_attempts,
                            )
                            fallback_records.append(
                                (env_index, seed, fallback_level, density_range)
                            )
                            break
                        except RuntimeError:
                            continue
                    if layout is None:
                        raise RuntimeError(
                            "unable to generate structured layout after "
                            f"density fallbacks (env={env_index}, seed={seed}, "
                            f"configured_density={cfg.clutter_density_range})"
                        ) from initial_error
                layouts.append(layout)
                if (env_index + 1) % 512 == 0:
                    print(
                        "[RandomObstacleWorld] generated "
                        f"{env_index + 1}/{self.num_envs} structured layouts "
                        f"(fallbacks={len(fallback_records)})",
                        flush=True,
                    )
            if fallback_records:
                levels = {
                    level: sum(record[2] == level for record in fallback_records)
                    for level in range(1, len(fallback_ranges) + 1)
                }
                examples = ", ".join(
                    f"env={env},seed={seed},level={level}"
                    for env, seed, level, _ in fallback_records[:5]
                )
                print(
                    "[RandomObstacleWorld] recovered "
                    f"{len(fallback_records)}/{self.num_envs} layouts with "
                    f"density fallback levels={levels}; examples: {examples}"
                )
            for name in ("positions", "active"):
                target = getattr(self, name)
                target.copy_(torch.as_tensor(np.stack([layout[name] for layout in layouts]),
                                             device=self.device, dtype=target.dtype))
            base_slots = cfg.base_num_wall_slots
            self.wall_poses.zero_()
            self.wall_active.zero_()
            self.wall_poses[:, :base_slots].copy_(torch.as_tensor(
                np.stack([layout["wall_poses"] for layout in layouts]),
                device=self.device, dtype=self.wall_poses.dtype))
            self.wall_active[:, :base_slots].copy_(torch.as_tensor(
                np.stack([layout["wall_active"] for layout in layouts]),
                device=self.device, dtype=self.wall_active.dtype))
            self.spawn_pairs = torch.as_tensor(
                np.stack([layout["spawn_pairs"] for layout in layouts]),
                device=self.device, dtype=self.positions.dtype,
            )
            self.curriculum_spawn_pairs = torch.as_tensor(
                np.stack([layout["curriculum_pairs"] for layout in layouts]),
                device=self.device, dtype=self.positions.dtype,
            )
            self.yaws.uniform_(0, 2 * math.pi)
            self._install_dedicated_detour_layouts(seeds)
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

    def _install_dedicated_detour_layouts(self, seeds: list[int]) -> None:
        """Replace the last configured layouts with sealed detour corridors."""

        if self.cfg.dedicated_detour_env_count == 0:
            return
        wall_lengths = np.asarray(self.cfg.wall_slot_lengths, dtype=np.float64)
        for env_index in range(self.dedicated_detour_start, self.num_envs):
            layout = sample_straight_detour_layout(
                self.cfg._obstacle_bound_radii,
                wall_lengths,
                seed=seeds[env_index] ^ 0x5EED5EED,
                arena_size=self.cfg.arena_size,
                outer_wall_thickness=self.cfg.wall_thickness,
                wall_width=self.cfg.interior_wall_thickness,
                base_wall_slot_count=self.cfg.base_num_wall_slots,
                lane_count=self.cfg.dedicated_detour_lane_count,
                passage_width_range=self.cfg.dedicated_detour_passage_width_range,
                curriculum_passage_width_ranges=(
                    self.cfg.dedicated_detour_curriculum_passage_width_ranges
                ),
                passage_width_samples_per_level=(
                    self.cfg.dedicated_detour_passage_width_samples_per_level
                ),
                barrier_count_range=self.cfg.dedicated_detour_barrier_count_range,
                barrier_y_span=self.cfg.dedicated_detour_barrier_y_span,
                extra_obstacle_count_range=(
                    self.cfg.dedicated_detour_extra_obstacle_count_range
                ),
                obstacle_wall_clearance=(
                    self.cfg.dedicated_detour_obstacle_wall_clearance
                ),
                robot_half_width=self.cfg.dedicated_detour_robot_half_width,
                robot_half_length=self.cfg.dedicated_detour_robot_half_length,
                collision_clearance=(
                    self.cfg.dedicated_detour_collision_clearance
                ),
                min_obstacle_spacing=self.cfg.min_obstacle_spacing,
                navigation_clearance=self.cfg.navigation_clearance,
                emma_clearance=self.cfg.emma_spawn_clearance_radius,
                goal_clearance=self.cfg.goal_spawn_clearance_radius,
                min_goal_distance=self.cfg.goal_min_distance,
                spawn_bank_size=self.cfg.spawn_bank_size,
                spawn_y=self.cfg.dedicated_detour_spawn_y,
                spawn_lateral_jitter=(
                    self.cfg.dedicated_detour_spawn_lateral_jitter
                ),
                curriculum_levels=self.curriculum_spawn_pairs.shape[1],
            )
            for name in ("positions", "active", "yaws", "wall_poses", "wall_active"):
                target = self.yaws if name == "yaws" else getattr(self, name)
                target[env_index].copy_(torch.as_tensor(
                    layout[name], device=self.device, dtype=target.dtype))
            self.spawn_pairs[env_index].copy_(torch.as_tensor(
                layout["spawn_pairs"], device=self.device,
                dtype=self.spawn_pairs.dtype))
            self.curriculum_spawn_pairs[env_index].copy_(torch.as_tensor(
                layout["curriculum_pairs"], device=self.device,
                dtype=self.curriculum_spawn_pairs.dtype))
            self.dedicated_detour_layouts[env_index] = layout
            widths = ",".join(
                f"{value:.2f}" for lane in layout["passage_widths"]
                for value in lane
            )
            print(
                f"[DedicatedDetour] env={env_index} "
                f"lanes={self.cfg.dedicated_detour_lane_count} "
                f"barriers={layout['barrier_count']} "
                f"passages=[{widths}]m obstacles={int(layout['active'].sum())}"
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
            # Levels 2 and 3 share the original long-range bank; levels 4 and
            # 5 use increasingly local static-clutter density rankings.
            bank_levels = task_levels.clamp(max=5)
            pairs = self.curriculum_spawn_pairs[env_ids, bank_levels, choice]
            emma, goal = pairs[:, 0], pairs[:, 1]
            heading_levels = task_levels.clamp(max=2)
            heading_half = torch.tensor(self.cfg.spawn_heading_half_angles_deg,
                                        device=self.device)[heading_levels] * (math.pi / 180)
            dedicated = env_ids >= self.dedicated_detour_start
            if dedicated.any() and self.cfg.dedicated_detour_env_count > 0:
                dedicated_half = math.radians(
                    self.cfg.dedicated_detour_heading_half_angle_deg
                )
                heading_half = torch.where(
                    dedicated, torch.full_like(heading_half, dedicated_half),
                    heading_half,
                )
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

    def reference_routes(
        self,
        env_ids: torch.Tensor,
        task_levels: torch.Tensor,
        *,
        max_waypoints: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return padded local-frame routes for dedicated detour environments.

        General procedural layouts do not yet expose reference paths. Their rows
        remain invalid, so the standard goal-distance reward is retained.
        """
        count = env_ids.numel()
        routes = torch.zeros(
            count, max_waypoints, 2,
            device=self.device, dtype=self.positions.dtype,
        )
        point_counts = torch.zeros(
            count, device=self.device, dtype=torch.long,
        )
        valid = torch.zeros(count, device=self.device, dtype=torch.bool)
        if not self.dedicated_detour_layouts:
            return routes, point_counts, valid

        dedicated_rows = torch.where(
            env_ids >= self.dedicated_detour_start
        )[0]
        for row in dedicated_rows.cpu().tolist():
            env_index = int(env_ids[row].item())
            layout = self.dedicated_detour_layouts.get(env_index)
            if layout is None:
                continue
            level = min(
                max(int(task_levels[row].item()), 0),
                len(layout["curriculum_lane_ids"]) - 1,
            )
            lane = int(layout["curriculum_lane_ids"][level])
            route = layout["route_waypoints"][lane]
            route_count = int(route.shape[0])
            if route_count > max_waypoints:
                raise RuntimeError(
                    f"reference route needs {route_count} waypoints, "
                    f"buffer provides {max_waypoints}"
                )
            routes[row, :route_count] = torch.as_tensor(
                route, device=self.device, dtype=routes.dtype,
            )
            point_counts[row] = route_count
            valid[row] = True
        return routes, point_counts, valid

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
                wall = trimesh.creation.box(extents=[self.cfg.wall_slot_lengths[index],
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

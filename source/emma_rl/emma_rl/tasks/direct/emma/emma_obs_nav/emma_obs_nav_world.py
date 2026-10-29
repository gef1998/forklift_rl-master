"""Mixed-obstacle maze world used by the legacy ``ring`` runtime mode."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

import torch
import trimesh

import isaaclab.sim as sim_utils
from isaaclab.utils.warp import convert_to_warp_mesh

from .lidar_ground import make_lidar_ground

from .ring_geometry import (
    alternating_obstacle_types,
    maze_wall_specs,
    sample_maze_obstacle_positions,
    sample_maze_spawn_pairs,
    validate_maze_geometry,
)


@dataclass
class ObstacleWorldConfig:
    """Geometry and sampling configuration for the bounded local maze."""

    arena_size: float = 10.0
    num_obstacles: int = 20

    # Obstacles alternate between cylinders and cubes. A cube's planar bound
    # is half its diagonal, so wall, passage, and spacing checks are shape-safe.
    radius_min: float = 0.18
    radius_max: float = 0.30
    cube_size_min: float = 0.30
    cube_size_max: float = 0.40
    height_min: float = 1.0
    height_max: float = 1.2
    min_obs_spacing: float = 0.45

    wall_thickness: float = 0.15
    interior_wall_thickness: float = 0.15
    wall_height: float = 1.0
    corridor_width: float = 2.0
    wall_tip_offset: float = 1.0

    # 障碍物使用保守的平面外接圆；走廊内至少有一侧保留该净通道宽度。
    obstacle_wall_clearance: float = 0.05
    passage_side_clearance: float = 0.30
    min_passage_width: float = 1.15
    spawn_exclusion_radius: float = 1.20
    gate_exclusion_radius: float = 0.75
    sampling_candidate_count: int = 24

    # Emma and the goal occupy different obstacle-free corner corridor ends.
    spawn_jitter: float = 0.25
    goal_min_distance: float = 7.0

    _obstacle_types: list[str] = field(default_factory=list)
    _obstacle_radii: list[float] = field(default_factory=list)
    _obstacle_cube_sizes: list[tuple[float, float]] = field(default_factory=list)
    _obstacle_bound_radii: list[float] = field(default_factory=list)
    _obstacle_heights: list[float] = field(default_factory=list)


def _ensure_collision_api(prim_path: str) -> None:
    """Apply CollisionAPI when Isaac Sim does not add it automatically."""

    from pxr import UsdPhysics
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(prim_path)
    if prim.IsValid() and not UsdPhysics.CollisionAPI(prim):
        UsdPhysics.CollisionAPI.Apply(prim)


def _spawn_maze_walls(cfg: ObstacleWorldConfig) -> None:
    for name, cx, cy, sx, sy in maze_wall_specs(
        arena_size=cfg.arena_size,
        wall_thickness=cfg.wall_thickness,
        interior_wall_thickness=cfg.interior_wall_thickness,
        corridor_width=cfg.corridor_width,
        wall_tip_offset=cfg.wall_tip_offset,
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
        path = f"/World/envs/env_0/Maze_{name}"
        wall_cfg.func(
            path, wall_cfg, translation=(cx, cy, cfg.wall_height / 2.0)
        )
        _ensure_collision_api(path)


def spawn_obstacles(num_envs: int, cfg: ObstacleWorldConfig) -> None:
    """Spawn eight maze walls and mixed obstacle templates in ``env_0``."""

    del num_envs  # templates are cloned by InteractiveScene
    validate_maze_geometry(
        arena_size=cfg.arena_size,
        wall_thickness=cfg.wall_thickness,
        interior_wall_thickness=cfg.interior_wall_thickness,
        corridor_width=cfg.corridor_width,
        wall_tip_offset=cfg.wall_tip_offset,
        spawn_jitter=cfg.spawn_jitter,
        min_passage_width=cfg.min_passage_width,
    )
    _spawn_maze_walls(cfg)

    cfg._obstacle_types = []
    cfg._obstacle_radii = []
    cfg._obstacle_cube_sizes = []
    cfg._obstacle_bound_radii = []
    cfg._obstacle_heights = []

    obstacle_types = alternating_obstacle_types(cfg.num_obstacles)
    for obstacle_index, obstacle_type in enumerate(obstacle_types):
        height = random.uniform(cfg.height_min, cfg.height_max)
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

        path = f"/World/envs/env_0/Obstacle_{obstacle_index}"
        spawn_cfg.func(path, spawn_cfg, translation=(0.0, 0.0, height / 2.0))
        _ensure_collision_api(path)


class ObstacleWorldManager:
    """Manage maze layouts, simulator poses, spawns, and the LiDAR warp mesh."""

    def __init__(self, num_envs: int, cfg: ObstacleWorldConfig, device: str):
        self.cfg = cfg
        self.num_envs = num_envs
        self.device = device
        self.num_obstacles = cfg.num_obstacles

        self.types = tuple(cfg._obstacle_types)
        self.radii = torch.tensor(cfg._obstacle_radii, device=device)
        self.cube_sizes = torch.tensor(cfg._obstacle_cube_sizes, device=device)
        self.bound_radii = torch.tensor(cfg._obstacle_bound_radii, device=device)
        self.heights = torch.tensor(cfg._obstacle_heights, device=device)
        self.positions = torch.zeros(
            num_envs, self.num_obstacles, 2, device=device
        )
        self.yaws = torch.zeros(num_envs, self.num_obstacles, device=device)
        self.active_count = 0

    def sample_initial_layout(self, num_obstacles: int | None = None) -> None:
        count = self.num_obstacles if num_obstacles is None else int(num_obstacles)
        if not 0 <= count <= self.num_obstacles:
            raise ValueError("num_obstacles is outside the spawned template count")

        active_bounds = self.bound_radii[:count]
        sampled = sample_maze_obstacle_positions(
            self.num_envs,
            active_bounds,
            arena_size=self.cfg.arena_size,
            wall_thickness=self.cfg.wall_thickness,
            interior_wall_thickness=self.cfg.interior_wall_thickness,
            corridor_width=self.cfg.corridor_width,
            wall_tip_offset=self.cfg.wall_tip_offset,
            obstacle_wall_clearance=self.cfg.obstacle_wall_clearance,
            min_passage_width=self.cfg.min_passage_width,
            spawn_exclusion_radius=self.cfg.spawn_exclusion_radius,
            gate_exclusion_radius=self.cfg.gate_exclusion_radius,
            min_edge_spacing=self.cfg.min_obs_spacing,
            candidate_pool_size=self.cfg.sampling_candidate_count,
        )
        self.positions.zero_()
        self.yaws.zero_()
        self.positions[:, :count] = sampled
        self.yaws[:, :count] = torch.rand(
            self.num_envs, count, device=self.device
        ) * (2.0 * math.pi)
        self.active_count = count

    def sample_spawn_pairs(
        self, env_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample safe, sufficiently long corner-to-corner local tasks."""

        return sample_maze_spawn_pairs(
            len(env_ids),
            arena_size=self.cfg.arena_size,
            wall_thickness=self.cfg.wall_thickness,
            corridor_width=self.cfg.corridor_width,
            spawn_jitter=self.cfg.spawn_jitter,
            min_goal_distance=self.cfg.goal_min_distance,
            device=self.device,
        )

    def write_poses_to_sim(
        self, scene, num_envs: int, env_origins_xy: torch.Tensor
    ) -> None:
        """Write mixed obstacle positions and cube yaw to cloned prims."""

        for obstacle_index in range(self.num_obstacles):
            obstacle = scene.rigid_objects.get(f"obstacle_{obstacle_index}")
            if obstacle is None or obstacle.data.default_root_state is None:
                continue
            try:
                state = obstacle.data.default_root_state.clone()
                state_view = state.view(num_envs, -1, 13)
                active = obstacle_index < self.active_count
                state_view[:, 0, 0:2] = (
                    self.positions[:num_envs, obstacle_index]
                    + env_origins_xy[:num_envs]
                )
                state_view[:, 0, 3:7] = 0.0
                state_view[:, 0, 7:13] = 0.0
                if active:
                    yaw = self.yaws[:num_envs, obstacle_index]
                    state_view[:, 0, 2] = self.heights[obstacle_index] / 2.0
                    state_view[:, 0, 3] = torch.cos(yaw / 2.0)
                    state_view[:, 0, 6] = torch.sin(yaw / 2.0)
                else:
                    # Inactive templates are kept underground with an identity
                    # quaternion. Avoid passing the Python float 0.0 to
                    # torch.cos/torch.sin (the single-env crash at obstacle_24).
                    state_view[:, 0, 2] = -100.0
                    state_view[:, 0, 3] = 1.0
                obstacle.write_root_state_to_sim(state)
            except Exception:
                import logging

                logging.getLogger(__name__).exception(
                    "write_poses_to_sim failed for obstacle_%d", obstacle_index
                )
                raise

    def build_warp_mesh(self, env_origins: torch.Tensor, device: str):
        """Build ground, maze walls, cylinders, and oriented cubes for LiDAR."""

        meshes: list[trimesh.Trimesh] = []
        ground = make_lidar_ground(env_origins, self.cfg.arena_size)
        meshes.append(ground)

        wall_specs = maze_wall_specs(
            arena_size=self.cfg.arena_size,
            wall_thickness=self.cfg.wall_thickness,
            interior_wall_thickness=self.cfg.interior_wall_thickness,
            corridor_width=self.cfg.corridor_width,
            wall_tip_offset=self.cfg.wall_tip_offset,
        )

        for env_index in range(self.num_envs):
            origin_x = float(env_origins[env_index, 0].item())
            origin_y = float(env_origins[env_index, 1].item())
            for _, wall_x, wall_y, size_x, size_y in wall_specs:
                wall = trimesh.creation.box(
                    extents=[size_x, size_y, self.cfg.wall_height]
                )
                wall.apply_translation(
                    [origin_x + wall_x, origin_y + wall_y, self.cfg.wall_height / 2.0]
                )
                meshes.append(wall)

            for obstacle_index in range(self.active_count):
                height = float(self.heights[obstacle_index].item())
                world_x = (
                    float(self.positions[env_index, obstacle_index, 0].item())
                    + origin_x
                )
                world_y = (
                    float(self.positions[env_index, obstacle_index, 1].item())
                    + origin_y
                )
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
                        trimesh.transformations.rotation_matrix(yaw, [0.0, 0.0, 1.0])
                    )
                mesh.apply_translation([world_x, world_y, height / 2.0])
                meshes.append(mesh)

        merged = trimesh.util.concatenate(meshes)
        return convert_to_warp_mesh(merged.vertices, merged.faces, device=device)

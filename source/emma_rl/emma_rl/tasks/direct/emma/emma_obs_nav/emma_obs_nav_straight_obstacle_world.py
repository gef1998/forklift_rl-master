"""Isaac Sim world for the documented straight-line obstacle test."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import trimesh

import isaaclab.sim as sim_utils
from isaaclab.utils.warp import convert_to_warp_mesh

from .lidar_ground import make_lidar_ground

from .straight_obstacle_geometry import (
    sample_straight_obstacle_spawn_pairs,
    straight_obstacle_arena_width,
    straight_obstacle_specs,
    validate_straight_obstacle_geometry,
)


@dataclass
class StraightObstacleWorldConfig:
    """Geometry and spawn configuration for one static center obstacle."""

    arena_length: float = 12.0
    passage_width: float = 1.50
    wall_thickness: float = 0.15
    wall_height: float = 1.0
    obstacle_length: float = 1.0
    obstacle_width: float = 0.8
    obstacle_height: float = 1.0
    obstacle_type: str = "cuboid"
    spawn_wall_margin: float = 1.0
    robot_half_width: float = 0.267

    @property
    def arena_size(self) -> float:
        # InteractiveScene uses a square env spacing even though this world is
        # rectangular. Keep cloned environments separated by the larger side.
        return max(
            self.arena_length,
            straight_obstacle_arena_width(
                self.passage_width,
                self.obstacle_width,
                self.wall_thickness,
            ),
        )


def _ensure_collision_api(prim_path: str) -> None:
    from pxr import UsdPhysics
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(prim_path)
    if prim.IsValid() and not UsdPhysics.CollisionAPI(prim):
        UsdPhysics.CollisionAPI.Apply(prim)


def spawn_straight_obstacle_world(cfg: StraightObstacleWorldConfig) -> None:
    """Spawn four channel walls and one static cuboid in ``env_0``."""

    if cfg.obstacle_type != "cuboid":
        raise ValueError("straight_obstacle currently supports obstacle_type='cuboid'")
    validate_straight_obstacle_geometry(
        arena_length=cfg.arena_length,
        passage_width=cfg.passage_width,
        wall_thickness=cfg.wall_thickness,
        obstacle_length=cfg.obstacle_length,
        obstacle_width=cfg.obstacle_width,
        spawn_wall_margin=cfg.spawn_wall_margin,
        robot_half_width=cfg.robot_half_width,
    )
    for name, cx, cy, sx, sy, sz in straight_obstacle_specs(
        arena_length=cfg.arena_length,
        passage_width=cfg.passage_width,
        wall_thickness=cfg.wall_thickness,
        wall_height=cfg.wall_height,
        obstacle_length=cfg.obstacle_length,
        obstacle_width=cfg.obstacle_width,
        obstacle_height=cfg.obstacle_height,
    ):
        color = (0.85, 0.20, 0.08) if name == "CenterObstacle" else (0.35, 0.35, 0.40)
        shape_cfg = sim_utils.CuboidCfg(
            size=(sx, sy, sz),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                kinematic_enabled=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
        )
        path = f"/World/envs/env_0/StraightObstacleWorld_{name}"
        shape_cfg.func(path, shape_cfg, translation=(cx, cy, sz / 2.0))
        _ensure_collision_api(path)


class StraightObstacleWorldManager:
    """Provide spawn and LiDAR mesh interfaces used by ``EmmaObsNavEnv``."""

    def __init__(
        self,
        num_envs: int,
        cfg: StraightObstacleWorldConfig,
        device: str,
    ):
        self.num_envs = num_envs
        self.cfg = cfg
        self.device = device

    def sample_spawn_pairs(
        self,
        env_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return sample_straight_obstacle_spawn_pairs(
            len(env_ids),
            arena_length=self.cfg.arena_length,
            passage_width=self.cfg.passage_width,
            wall_thickness=self.cfg.wall_thickness,
            obstacle_length=self.cfg.obstacle_length,
            obstacle_width=self.cfg.obstacle_width,
            spawn_wall_margin=self.cfg.spawn_wall_margin,
            robot_half_width=self.cfg.robot_half_width,
            device=self.device,
        )

    def build_warp_mesh(self, env_origins: torch.Tensor, device: str):
        meshes: list[trimesh.Trimesh] = []
        ground = make_lidar_ground(env_origins, self.cfg.arena_size)
        meshes.append(ground)
        specs = straight_obstacle_specs(
            arena_length=self.cfg.arena_length,
            passage_width=self.cfg.passage_width,
            wall_thickness=self.cfg.wall_thickness,
            wall_height=self.cfg.wall_height,
            obstacle_length=self.cfg.obstacle_length,
            obstacle_width=self.cfg.obstacle_width,
            obstacle_height=self.cfg.obstacle_height,
        )
        for env_index in range(self.num_envs):
            origin_x = float(env_origins[env_index, 0].item())
            origin_y = float(env_origins[env_index, 1].item())
            for _, cx, cy, sx, sy, sz in specs:
                mesh = trimesh.creation.box(extents=[sx, sy, sz])
                mesh.apply_translation([origin_x + cx, origin_y + cy, sz / 2.0])
                meshes.append(mesh)
        merged = trimesh.util.concatenate(meshes)
        return convert_to_warp_mesh(merged.vertices, merged.faces, device=device)

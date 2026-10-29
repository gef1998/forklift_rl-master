"""Isaac Sim 场景实现：两面交错长墙组成的 S 形课程走廊。"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import trimesh

import isaaclab.sim as sim_utils
from isaaclab.utils.warp import convert_to_warp_mesh

from .lidar_ground import make_lidar_ground

from .corridor_geometry import (
    s_corridor_wall_specs,
    sample_s_corridor_spawn_pairs,
    validate_s_corridor_geometry,
)


@dataclass
class CorridorWorldConfig:
    """S 形走廊几何与出生点配置。"""

    arena_size: float = 10.0
    wall_thickness: float = 0.15
    interior_wall_thickness: float = 0.20
    wall_height: float = 1.0
    inner_wall_offset: float = 1.50
    gate_width: float = 2.50

    # 出生点位于上下端安全区；两端随机互换形成双向课程。
    spawn_wall_margin: float = 1.0
    spawn_jitter_x: float = 0.75
    spawn_jitter_y: float = 0.20
    goal_min_distance: float = 7.0
    # 偶数/奇数环境分别使用原布局和水平镜像，避免固定转向顺序偏置。
    mirror_across_envs: bool = True


def _ensure_collision_api(prim_path: str) -> None:
    from pxr import UsdPhysics
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(prim_path)
    if prim.IsValid() and not UsdPhysics.CollisionAPI(prim):
        UsdPhysics.CollisionAPI.Apply(prim)


def spawn_s_corridor(cfg: CorridorWorldConfig) -> None:
    """在 env_0 创建六面静态墙，随后由 InteractiveScene 克隆。"""

    validate_s_corridor_geometry(
        arena_size=cfg.arena_size,
        wall_thickness=cfg.wall_thickness,
        interior_wall_thickness=cfg.interior_wall_thickness,
        inner_wall_offset=cfg.inner_wall_offset,
        gate_width=cfg.gate_width,
        spawn_wall_margin=cfg.spawn_wall_margin,
        spawn_jitter_x=cfg.spawn_jitter_x,
        spawn_jitter_y=cfg.spawn_jitter_y,
        min_goal_distance=cfg.goal_min_distance,
    )
    for name, cx, cy, sx, sy in s_corridor_wall_specs(
        arena_size=cfg.arena_size,
        wall_thickness=cfg.wall_thickness,
        interior_wall_thickness=cfg.interior_wall_thickness,
        inner_wall_offset=cfg.inner_wall_offset,
        gate_width=cfg.gate_width,
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
        path = f"/World/envs/env_0/Corridor_{name}"
        wall_cfg.func(
            path, wall_cfg, translation=(cx, cy, cfg.wall_height / 2.0)
        )
        _ensure_collision_api(path)


class CorridorWorldManager:
    """提供与其他世界管理器一致的出生采样和 LiDAR mesh 接口。"""

    def __init__(self, num_envs: int, cfg: CorridorWorldConfig, device: str):
        self.num_envs = num_envs
        self.cfg = cfg
        self.device = device
        self.mirror_mask = (
            torch.arange(num_envs, device=device, dtype=torch.long) % 2 == 1
            if cfg.mirror_across_envs
            else torch.zeros(num_envs, device=device, dtype=torch.bool)
        )

    def write_barrier_poses_to_sim(
        self,
        scene,
        env_origins_xy: torch.Tensor,
    ) -> None:
        """把奇数环境的两面内墙水平镜像；外墙保持不变。"""

        half_gate = torch.full(
            (self.num_envs,),
            self.cfg.gate_width / 2.0,
            device=self.device,
        )
        local_x = {
            "corridor_lower_barrier": torch.where(
                self.mirror_mask,
                -half_gate,
                half_gate,
            ),
            "corridor_upper_barrier": torch.where(
                self.mirror_mask,
                half_gate,
                -half_gate,
            ),
        }
        local_y = {
            "corridor_lower_barrier": -self.cfg.inner_wall_offset,
            "corridor_upper_barrier": self.cfg.inner_wall_offset,
        }
        for asset_name in local_x:
            barrier = scene.rigid_objects.get(asset_name)
            if barrier is None or barrier.data.default_root_state is None:
                raise RuntimeError(f"missing corridor asset: {asset_name}")
            state = barrier.data.default_root_state.clone()
            state_view = state.view(self.num_envs, -1, 13)
            state_view[:, 0, 0] = (
                local_x[asset_name] + env_origins_xy[:, 0]
            )
            state_view[:, 0, 1] = (
                local_y[asset_name] + env_origins_xy[:, 1]
            )
            state_view[:, 0, 2] = self.cfg.wall_height / 2.0
            state_view[:, 0, 3:7] = 0.0
            state_view[:, 0, 3] = 1.0
            state_view[:, 0, 7:13] = 0.0
            barrier.write_root_state_to_sim(state)

    def sample_spawn_pairs(
        self, env_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return sample_s_corridor_spawn_pairs(
            len(env_ids),
            arena_size=self.cfg.arena_size,
            wall_thickness=self.cfg.wall_thickness,
            interior_wall_thickness=self.cfg.interior_wall_thickness,
            inner_wall_offset=self.cfg.inner_wall_offset,
            gate_width=self.cfg.gate_width,
            spawn_wall_margin=self.cfg.spawn_wall_margin,
            spawn_jitter_x=self.cfg.spawn_jitter_x,
            spawn_jitter_y=self.cfg.spawn_jitter_y,
            min_goal_distance=self.cfg.goal_min_distance,
            device=self.device,
        )

    def build_warp_mesh(self, env_origins: torch.Tensor, device: str):
        """构建地面和每个克隆环境的六面墙，用于自定义 LiDAR。"""

        meshes: list[trimesh.Trimesh] = []
        ground = make_lidar_ground(env_origins, self.cfg.arena_size)
        meshes.append(ground)

        for env_index in range(self.num_envs):
            origin_x = float(env_origins[env_index, 0].item())
            origin_y = float(env_origins[env_index, 1].item())
            wall_specs = s_corridor_wall_specs(
                arena_size=self.cfg.arena_size,
                wall_thickness=self.cfg.wall_thickness,
                interior_wall_thickness=self.cfg.interior_wall_thickness,
                inner_wall_offset=self.cfg.inner_wall_offset,
                gate_width=self.cfg.gate_width,
                mirrored=bool(self.mirror_mask[env_index].item()),
            )
            for _, wall_x, wall_y, size_x, size_y in wall_specs:
                wall = trimesh.creation.box(
                    extents=[size_x, size_y, self.cfg.wall_height]
                )
                wall.apply_translation(
                    [
                        origin_x + wall_x,
                        origin_y + wall_y,
                        self.cfg.wall_height / 2.0,
                    ]
                )
                meshes.append(wall)

        merged = trimesh.util.concatenate(meshes)
        return convert_to_warp_mesh(merged.vertices, merged.faces, device=device)

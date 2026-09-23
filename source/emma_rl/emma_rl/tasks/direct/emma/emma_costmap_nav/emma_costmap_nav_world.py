"""混合障碍物场景管理器 — 静态 Cube/Cylinder + 动态 Cylinder.

独立于 emma_obs_nav, 支持:
  - 静态长方体 (CuboidCfg): 模拟墙壁/货架, 任意 yaw
  - 静态圆柱 (CylinderCfg): 模拟柱子
  - 动态圆柱 (CylinderCfg): 模拟行人, kinematic 游走 + 边界反弹
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

import torch

import isaaclab.sim as sim_utils


# ============================================================================
# 1. NavWorldConfig
# ============================================================================
@dataclass
class NavWorldConfig:
    """混合障碍物场景配置 — 所有数量/尺寸均可调."""

    arena_size: float = 30.0

    # ---- 静态长方体 (Cube) ----
    num_static_cubes: int = 80
    cube_length_range: tuple = (2.0, 4.0)        # 长 (body X, 沿长边)
    cube_width_range: tuple = (0.4, 0.6)          # 宽
    cube_height_range: tuple = (2.0, 3.0)          # 高

    # ---- 静态圆柱 ----
    num_static_cylinders: int = 30
    static_cylinder_radius_range: tuple = (0.15, 0.35)
    static_cylinder_height_range: tuple = (2.0, 3.0)

    # ---- 动态圆柱 (行人) ----
    num_dynamic_cylinders: int = 10
    dynamic_radius_range: tuple = (0.15, 0.25)
    dynamic_height_range: tuple = (2.0, 3.0)
    dynamic_speed_range: tuple = (0.3, 0.8)       # m/s

    # ---- 布局约束 ----
    min_obs_spacing: float = 1.0                   # 障碍物边缘最小间距
    side_zone_width: float = 1.5                   # Emma/Goal 侧区宽度
    buffer_width: float = 0.5                      # 侧区与障碍区缓冲

    # ---- 内部存储 (spawn 时填充) ----
    _cube_dims: list = field(default_factory=list)          # [(l, w, h), ...]
    _cube_yaws: list = field(default_factory=list)          # [yaw_rad, ...]
    _static_cyl_radii: list = field(default_factory=list)   # [r, ...]
    _static_cyl_heights: list = field(default_factory=list) # [h, ...]
    _dynamic_cyl_radii: list = field(default_factory=list)  # [r, ...]
    _dynamic_cyl_heights: list = field(default_factory=list)# [h, ...]

    @property
    def num_total_static(self) -> int:
        return self.num_static_cubes + self.num_static_cylinders

    @property
    def num_total_obstacles(self) -> int:
        return self.num_total_static + self.num_dynamic_cylinders


# ============================================================================
# 2. Spawner — 在 env_0 创建模板 prims
# ============================================================================
def _apply_collision_api(prim_path: str):
    """Isaac Sim 4.5 兼容: 手动确保 CollisionAPI 挂载."""
    from pxr import UsdPhysics
    import omni.usd
    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(prim_path)
    if prim.IsValid() and not UsdPhysics.CollisionAPI(prim):
        UsdPhysics.CollisionAPI.Apply(prim)


def spawn_nav_obstacles(cfg: NavWorldConfig):
    """在 /World/envs/env_0/ 下创建所有障碍物模板.

    命名规则:
      - StaticCube_{i}   (i = 0..N_cubes-1)
      - StaticCyl_{j}     (j = 0..N_static_cyls-1)
      - DynamicCyl_{k}    (k = 0..N_dyn_cyls-1)
    """
    # ---- 静态长方体 ----
    cfg._cube_dims = []
    cfg._cube_yaws = []
    for i in range(cfg.num_static_cubes):
        length = random.uniform(*cfg.cube_length_range)
        width = random.uniform(*cfg.cube_width_range)
        height = random.uniform(*cfg.cube_height_range)
        yaw = random.uniform(0.0, math.pi)  # 世界 yaw
        cfg._cube_dims.append((length, width, height))
        cfg._cube_yaws.append(yaw)

        cube_cfg = sim_utils.CuboidCfg(
            size=(length, width, height),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                kinematic_enabled=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.45, 0.35, 0.25),  # 棕色 — 模拟货架/墙壁
            ),
        )
        prim_path = f"/World/envs/env_0/StaticCube_{i}"
        cube_cfg.func(prim_path, cube_cfg, translation=(0.0, 0.0, height / 2.0))
        _apply_collision_api(prim_path)

    # ---- 静态圆柱 ----
    cfg._static_cyl_radii = []
    cfg._static_cyl_heights = []
    for j in range(cfg.num_static_cylinders):
        r = random.uniform(*cfg.static_cylinder_radius_range)
        h = random.uniform(*cfg.static_cylinder_height_range)
        cfg._static_cyl_radii.append(r)
        cfg._static_cyl_heights.append(h)

        cyl_cfg = sim_utils.CylinderCfg(
            radius=r,
            height=h,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                kinematic_enabled=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.55, 0.45, 0.35),  # 浅棕色
            ),
        )
        prim_path = f"/World/envs/env_0/StaticCyl_{j}"
        cyl_cfg.func(prim_path, cyl_cfg, translation=(0.0, 0.0, h / 2.0))
        _apply_collision_api(prim_path)

    # ---- 动态圆柱 ----
    cfg._dynamic_cyl_radii = []
    cfg._dynamic_cyl_heights = []
    for k in range(cfg.num_dynamic_cylinders):
        r = random.uniform(*cfg.dynamic_radius_range)
        h = random.uniform(*cfg.dynamic_height_range)
        cfg._dynamic_cyl_radii.append(r)
        cfg._dynamic_cyl_heights.append(h)

        dyn_cfg = sim_utils.CylinderCfg(
            radius=r,
            height=h,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                kinematic_enabled=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.8, 0.3, 0.2),  # 红色 — 动态障碍物醒目
            ),
        )
        prim_path = f"/World/envs/env_0/DynamicCyl_{k}"
        dyn_cfg.func(prim_path, dyn_cfg, translation=(0.0, 0.0, h / 2.0))
        _apply_collision_api(prim_path)


# ============================================================================
# 3. NavWorldManager
# ============================================================================
class NavWorldManager:
    """管理所有障碍物的位置、Sim 写入和动态移动."""

    def __init__(self, num_envs: int, cfg: NavWorldConfig, device: str):
        self.num_envs = num_envs
        self.cfg = cfg
        self.device = device

        # ---- 静态障碍物属性张量 ----
        self.num_cubes = cfg.num_static_cubes
        self.num_static_cyls = cfg.num_static_cylinders
        self.num_dyn_cyls = cfg.num_dynamic_cylinders

        # Cube 半长/半宽 (不含高, costmap 只关心 XY 投影)
        if self.num_cubes > 0:
            self.cube_half_l = torch.tensor(
                [d[0] / 2.0 for d in cfg._cube_dims], device=device)  # (N_cubes,)
            self.cube_half_w = torch.tensor(
                [d[1] / 2.0 for d in cfg._cube_dims], device=device)  # (N_cubes,)
            self.cube_heights = torch.tensor(
                [d[2] for d in cfg._cube_dims], device=device)
            self.cube_yaws_w = torch.tensor(cfg._cube_yaws, device=device)  # (N_cubes,)
            # 矩形包围圆半径 (用于快速碰撞预检)
            self.cube_bounding_r = torch.sqrt(self.cube_half_l ** 2 + self.cube_half_w ** 2)
        else:
            self.cube_half_l = torch.zeros(0, device=device)
            self.cube_half_w = torch.zeros(0, device=device)
            self.cube_heights = torch.zeros(0, device=device)
            self.cube_yaws_w = torch.zeros(0, device=device)
            self.cube_bounding_r = torch.zeros(0, device=device)

        # 静态圆柱半径/高度
        if self.num_static_cyls > 0:
            self.static_cyl_radii = torch.tensor(cfg._static_cyl_radii, device=device)
            self.static_cyl_heights = torch.tensor(cfg._static_cyl_heights, device=device)
        else:
            self.static_cyl_radii = torch.zeros(0, device=device)
            self.static_cyl_heights = torch.zeros(0, device=device)

        # 动态圆柱半径/高度
        if self.num_dyn_cyls > 0:
            self.dyn_cyl_radii = torch.tensor(cfg._dynamic_cyl_radii, device=device)
            self.dyn_cyl_heights = torch.tensor(cfg._dynamic_cyl_heights, device=device)
        else:
            self.dyn_cyl_radii = torch.zeros(0, device=device)
            self.dyn_cyl_heights = torch.zeros(0, device=device)

        # ---- 位置张量 ----
        # 静态障碍物: 局部坐标 (相对于 env 中心)
        self.static_cube_pos = torch.zeros(num_envs, self.num_cubes, 2, device=device)
        self.static_cyl_pos = torch.zeros(num_envs, self.num_static_cyls, 2, device=device)
        # 动态障碍物: 世界坐标 (方便物理写入)
        self.dyn_cyl_pos_w = torch.zeros(num_envs, self.num_dyn_cyls, 2, device=device)
        self.dyn_cyl_vel = torch.zeros(num_envs, self.num_dyn_cyls, 2, device=device)

    # =====================================================================
    # Layout Sampling
    # =====================================================================
    def sample_initial_layout(self):
        """为所有 env 采样所有静态障碍物位置 (rejection sampling)."""
        for env_i in range(self.num_envs):
            eid = torch.tensor([env_i], device=self.device)
            # 分两阶段放置: 先 Cube (大), 再 Cylinder (小), 避免小障碍物占掉大障碍物的空间
            self._sample_cubes(eid)
            self._sample_static_cyls(eid)

        self._init_dynamic_positions()

    def _obstacle_zone_bounds(self):
        """返回障碍区边界 (局部坐标)."""
        cfg = self.cfg
        half = cfg.arena_size / 2.0
        x_min = -half + cfg.side_zone_width + cfg.buffer_width   # -13
        x_max = half - cfg.side_zone_width - cfg.buffer_width     # 13
        y_min = -half                                              # -15
        y_max = half                                               # 15
        return x_min, x_max, y_min, y_max

    def _sample_cubes(self, env_ids: torch.Tensor):
        """矩形障碍物 rejection sampling."""
        if self.num_cubes == 0:
            return
        n = len(env_ids)
        M = self.num_cubes
        x_min, x_max, y_min, y_max = self._obstacle_zone_bounds()
        new_pos = torch.zeros(n, M, 2, device=self.device)

        for i in range(M):
            br_i = self.cube_bounding_r[i].item()
            valid = torch.zeros(n, dtype=torch.bool, device=self.device)

            for _ in range(100):
                remaining = (~valid).nonzero(as_tuple=False).squeeze(-1)
                if len(remaining) == 0:
                    break
                cx = x_min + br_i + torch.rand(len(remaining), device=self.device) * (
                    x_max - x_min - 2 * br_i)
                cy = y_min + br_i + torch.rand(len(remaining), device=self.device) * (
                    y_max - y_min - 2 * br_i)
                cand = torch.stack([cx, cy], dim=-1)

                safe = torch.ones(len(remaining), dtype=torch.bool, device=self.device)
                if i > 0:
                    prev = new_pos[remaining, :i, :]
                    prev_r = self.cube_bounding_r[:i].unsqueeze(0).repeat(len(remaining), 1)
                    dist = torch.norm(cand.unsqueeze(1) - prev, dim=-1)
                    safe &= (dist > br_i + prev_r + self.cfg.min_obs_spacing).all(dim=-1)

                good = remaining[safe]
                new_pos[good, i] = cand[safe]
                valid[good] = True

            # fallback
            still_bad = (~valid).nonzero(as_tuple=False).squeeze(-1)
            if len(still_bad) > 0:
                new_pos[still_bad, i, 0] = x_min + br_i + torch.rand(
                    len(still_bad), device=self.device) * (x_max - x_min - 2 * br_i)
                new_pos[still_bad, i, 1] = y_min + br_i + torch.rand(
                    len(still_bad), device=self.device) * (y_max - y_min - 2 * br_i)

        self.static_cube_pos[env_ids] = new_pos

    def _sample_static_cyls(self, env_ids: torch.Tensor):
        """圆柱障碍物 rejection sampling (避开已有 Cube)."""
        if self.num_static_cyls == 0:
            return
        n = len(env_ids)
        M = self.num_static_cyls
        x_min, x_max, y_min, y_max = self._obstacle_zone_bounds()
        new_pos = torch.zeros(n, M, 2, device=self.device)

        for i in range(M):
            r_i = self.static_cyl_radii[i].item()
            valid = torch.zeros(n, dtype=torch.bool, device=self.device)

            for _ in range(100):
                remaining = (~valid).nonzero(as_tuple=False).squeeze(-1)
                if len(remaining) == 0:
                    break
                cx = x_min + r_i + torch.rand(len(remaining), device=self.device) * (
                    x_max - x_min - 2 * r_i)
                cy = y_min + r_i + torch.rand(len(remaining), device=self.device) * (
                    y_max - y_min - 2 * r_i)
                cand = torch.stack([cx, cy], dim=-1)

                safe = torch.ones(len(remaining), dtype=torch.bool, device=self.device)

                # 检查与已放置圆柱的间距
                if i > 0:
                    prev = new_pos[remaining, :i, :]
                    prev_r = self.static_cyl_radii[:i].unsqueeze(0).repeat(len(remaining), 1)
                    dist = torch.norm(cand.unsqueeze(1) - prev, dim=-1)
                    safe &= (dist > r_i + prev_r + self.cfg.min_obs_spacing).all(dim=-1)

                # 检查与 Cube 的间距 (用 Cube 包围圆)
                if self.num_cubes > 0:
                    cube_pos = self.static_cube_pos[env_ids][remaining]  # (n, N_cubes, 2)
                    dist_cube = torch.norm(
                        cand.unsqueeze(1) - cube_pos, dim=-1)  # (n, N_cubes)
                    safe &= (dist_cube > r_i + self.cube_bounding_r.unsqueeze(0) +
                             self.cfg.min_obs_spacing).all(dim=-1)

                good = remaining[safe]
                new_pos[good, i] = cand[safe]
                valid[good] = True

            still_bad = (~valid).nonzero(as_tuple=False).squeeze(-1)
            if len(still_bad) > 0:
                new_pos[still_bad, i, 0] = x_min + r_i + torch.rand(
                    len(still_bad), device=self.device) * (x_max - x_min - 2 * r_i)
                new_pos[still_bad, i, 1] = y_min + r_i + torch.rand(
                    len(still_bad), device=self.device) * (y_max - y_min - 2 * r_i)

        self.static_cyl_pos[env_ids] = new_pos

    def _init_dynamic_positions(self):
        """初始化动态障碍物: 在障碍区随机位置 + 随机速度."""
        if self.num_dyn_cyls == 0:
            return
        x_min, x_max, y_min, y_max = self._obstacle_zone_bounds()

        for i in range(self.num_dyn_cyls):
            r_i = self.dyn_cyl_radii[i].item()
            for env_i in range(self.num_envs):
                cx = x_min + r_i + random.random() * (x_max - x_min - 2 * r_i)
                cy = y_min + r_i + random.random() * (y_max - y_min - 2 * r_i)
                self.dyn_cyl_pos_w[env_i, i, 0] = cx
                self.dyn_cyl_pos_w[env_i, i, 1] = cy

                # 随机速度方向
                angle = random.random() * 2 * math.pi
                speed = random.uniform(*self.cfg.dynamic_speed_range)
                self.dyn_cyl_vel[env_i, i, 0] = math.cos(angle) * speed
                self.dyn_cyl_vel[env_i, i, 1] = math.sin(angle) * speed

    # =====================================================================
    # Dynamic Step
    # =====================================================================
    def step_dynamic(self, dt: float):
        """每 sim step 推进动态障碍物 (kinematic 游走 + 边界反弹)."""
        if self.num_dyn_cyls == 0:
            return
        x_min, x_max, y_min, y_max = self._obstacle_zone_bounds()

        # 位置更新
        self.dyn_cyl_pos_w += self.dyn_cyl_vel * dt

        # 边界反弹 (逐障碍物, 批量处理)
        for i in range(self.num_dyn_cyls):
            r_i = self.dyn_cyl_radii[i].item()
            # X 边界
            pos_x = self.dyn_cyl_pos_w[:, i, 0]
            bounce_x_low = pos_x < (x_min + r_i)
            bounce_x_high = pos_x > (x_max - r_i)
            self.dyn_cyl_vel[bounce_x_low | bounce_x_high, i, 0] *= -1.0
            self.dyn_cyl_pos_w[:, i, 0] = torch.clamp(pos_x, x_min + r_i, x_max - r_i)

            # Y 边界
            pos_y = self.dyn_cyl_pos_w[:, i, 1]
            bounce_y_low = pos_y < (y_min + r_i)
            bounce_y_high = pos_y > (y_max - r_i)
            self.dyn_cyl_vel[bounce_y_low | bounce_y_high, i, 1] *= -1.0
            self.dyn_cyl_pos_w[:, i, 1] = torch.clamp(pos_y, y_min + r_i, y_max - r_i)

    # =====================================================================
    # Sim Write
    # =====================================================================
    def _write_poses_batch(
        self, scene, num_envs: int, env_origins_xy: torch.Tensor,
        prim_names: list[str], positions: torch.Tensor,  # (N, M, 2) 局部坐标
        heights: torch.Tensor,                            # (M,)
        is_world_pos: bool = False,
    ):
        """批量写入障碍物位姿到 Sim.

        Args:
            scene: InteractiveScene.
            num_envs: 环境数量.
            env_origins_xy: (num_envs, 2) 各 env 的世界原点.
            prim_names: RigidObject scene key 列表.
            positions: (N, M, 2) 障碍物位置 (局部或世界).
            heights: (M,) 障碍物高度.
            is_world_pos: True 表示 positions 已经是世界坐标 (动态障碍物).
        """
        M = len(prim_names)
        for i in range(M):
            obs = scene.rigid_objects.get(prim_names[i])
            if obs is None:
                continue
            try:
                state = obs.data.default_root_state
                if state is None:
                    continue
                poses = state.clone()
                poses_view = poses.view(num_envs, -1, 13)
                hz = heights[i].item() / 2.0

                for env_i in range(num_envs):
                    if is_world_pos:
                        wx = positions[env_i, i, 0].item()
                        wy = positions[env_i, i, 1].item()
                    else:
                        wx = positions[env_i, i, 0].item() + env_origins_xy[env_i, 0].item()
                        wy = positions[env_i, i, 1].item() + env_origins_xy[env_i, 1].item()
                    poses_view[env_i, 0, 0] = wx
                    poses_view[env_i, 0, 1] = wy
                    poses_view[env_i, 0, 2] = hz
                    poses_view[env_i, 0, 3] = 1.0
                    poses_view[env_i, 0, 4:7] = 0.0
                obs.write_root_state_to_sim(poses)
            except Exception:
                import logging
                logging.getLogger(__name__).warning(
                    f"write_poses_batch: failed for {prim_names[i]}")

    def write_static_poses_to_sim(self, scene, num_envs: int, env_origins_xy: torch.Tensor):
        """写入所有静态障碍物位姿到 Sim."""
        # Cube prims
        if self.num_cubes > 0:
            cube_names = [f"StaticCube_{i}" for i in range(self.num_cubes)]
            self._write_poses_batch(scene, num_envs, env_origins_xy,
                                    cube_names, self.static_cube_pos, self.cube_heights)
        # Static Cylinder prims
        if self.num_static_cyls > 0:
            cyl_names = [f"StaticCyl_{i}" for i in range(self.num_static_cyls)]
            self._write_poses_batch(scene, num_envs, env_origins_xy,
                                    cyl_names, self.static_cyl_pos, self.static_cyl_heights)

    def write_dynamic_poses_to_sim(self, scene, num_envs: int, env_origins_xy: torch.Tensor):
        """写入动态障碍物位姿到 Sim (每 step 调用)."""
        if self.num_dyn_cyls > 0:
            dyn_names = [f"DynamicCyl_{k}" for k in range(self.num_dyn_cyls)]
            # 动态障碍物位置是局部坐标, 需转为世界坐标
            world_pos = self.dyn_cyl_pos_w.clone()
            # 加上 env origin
            world_pos = world_pos + env_origins_xy.unsqueeze(1)
            self._write_poses_batch(scene, num_envs, env_origins_xy,
                                    dyn_names, world_pos, self.dyn_cyl_heights,
                                    is_world_pos=True)

    # =====================================================================
    # World-frame centers for costmap (每 step 调用)
    # =====================================================================
    def get_static_world_centers(self, env_origins_xy: torch.Tensor):
        """返回静态障碍物的世界坐标.

        Returns:
            cube_centers_w: (N, N_cubes, 2) — Cube 世界 XY.
            cyl_centers_w:  (N, N_static_cyls, 2) — 静态圆柱世界 XY.
        """
        cube_w = self.static_cube_pos + env_origins_xy.unsqueeze(1) if self.num_cubes > 0 else None
        cyl_w = self.static_cyl_pos + env_origins_xy.unsqueeze(1) if self.num_static_cyls > 0 else None
        return cube_w, cyl_w

    def get_dynamic_world_centers(self, env_origins_xy: torch.Tensor):
        """返回动态障碍物的世界坐标.

        Returns:
            dyn_centers_w: (N, N_dyn_cyls, 2).
        """
        if self.num_dyn_cyls == 0:
            return None
        return self.dyn_cyl_pos_w + env_origins_xy.unsqueeze(1)

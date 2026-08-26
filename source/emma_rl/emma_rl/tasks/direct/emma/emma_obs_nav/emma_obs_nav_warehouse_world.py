"""Warehouse World Manager for Emma ObsNav.

默认仓库布局: 16×16m，4 边墙 + 3 列货架(每列 5 段) + 通道圆柱障碍物。
具体尺寸均由 :class:`WarehouseWorldConfig` 控制。
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

import torch
import trimesh

import isaaclab.sim as sim_utils
from isaaclab.utils.warp import convert_to_warp_mesh


# ===================================================================
# 1. Warehouse Config
# ===================================================================
@dataclass
class WarehouseWorldConfig:
    """仓库障碍物世界配置.

    布局说明:
      - arena 边长 arena_size, 原点为中心
      - 四周 wall_thickness 厚的墙壁
      - num_shelf_cols 列货架, 每列 shelves_per_col 段
      - 列之间 column_gap 宽通道, 货架段之间 shelf_row_gap_min~max 间隙
      - 货架 X 方向长 shelf_length, Y 方向宽 shelf_width
      - 圆柱均匀分布在 arena 内
    """

    arena_size: float = 16.0

    # 墙壁
    wall_thickness: float = 0.4
    wall_height_min: float = 0.8
    wall_height_max: float = 1.0

    # 货架列 — 列数 × 每列段数
    num_shelf_cols: int = 3
    shelves_per_col: int = 5
    shelf_length: float = 3.0                  # X 方向长度
    shelf_width: float = 1.0                   # Y 方向宽度 (跨越方向)
    shelf_height_min: float = 0.8
    shelf_height_max: float = 1.0
    column_gap: float = 2.2                    # 列间净通道宽度 (shelf X边到相邻shelf X边)
    shelf_row_gap_min: float = 2.2             # 货架段间最小间隙 (Y方向)
    shelf_row_gap_max: float = 2.4             # 货架段间最大间隙 (Y方向)

    # 圆柱障碍物
    cylinder_radius_min: float = 0.10
    cylinder_radius_max: float = 0.25
    cylinder_height_min: float = 0.8
    cylinder_height_max: float = 1.0
    cylinder_spawn_safety: float = 0.8          # 圆柱间最小间距 margin

    # Spawn 安全余量 (Emma 半对角 ≈ sqrt(0.42²+0.267²) ≈ 0.50)
    robot_spawn_margin: float = 0.55            # spawn 时的膨胀半径 (比运动 margin 小)
    emma_goal_min_dist: float = 5.0              # Emma-Goal 最小 spawn 距离
    emma_goal_max_dist: float = 10.0             # Emma-Goal 最大 spawn 距离
    goal_marker_radius: float = 0.2             # Goal 标记球半径 (用于 spawn 碰撞检测)

    # 每个 obstacle 的尺寸 (运行时填充)
    _wall_heights: list = field(default_factory=list)
    _shelf_configs_per_col: list = field(default_factory=list)  # [[{center_x, center_y, half_length, half_width, height}], ...]
    _cylinder_radii: list = field(default_factory=list)
    _cylinder_heights: list = field(default_factory=list)


# ===================================================================
# 2. Warehouse Spawner
# ===================================================================
def spawn_warehouse(num_envs: int, cfg: WarehouseWorldConfig):
    """创建墙壁 + 货架 + 圆柱模板 (env_0)."""

    arena_half = cfg.arena_size / 2.0
    wall_center = arena_half - cfg.wall_thickness / 2.0
    wall_inner = arena_half - cfg.wall_thickness

    prim_count = 0

    # ---- 墙壁 (4 条) ----
    cfg._wall_heights = []
    wall_configs = [
        ("TopWall",    (0.0,  wall_center), (cfg.arena_size, cfg.wall_thickness)),
        ("BottomWall", (0.0, -wall_center), (cfg.arena_size, cfg.wall_thickness)),
        ("LeftWall",   (-wall_center, 0.0), (cfg.wall_thickness, cfg.arena_size)),
        ("RightWall",  ( wall_center, 0.0), (cfg.wall_thickness, cfg.arena_size)),
    ]
    for name, (cx, cy), (sx, sy) in wall_configs:
        h = random.uniform(cfg.wall_height_min, cfg.wall_height_max)
        cfg._wall_heights.append(h)
        cuboid_cfg = sim_utils.CuboidCfg(
            size=(sx, sy, h),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True, kinematic_enabled=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.4, 0.4, 0.45),
            ),
        )
        path = f"/World/envs/env_0/Warehouse_{name}"
        cuboid_cfg.func(path, cuboid_cfg, translation=(cx, cy, h / 2.0))

        # Isaac Sim 4.5: ensure CollisionAPI
        from pxr import UsdPhysics
        import omni.usd
        stage = omni.usd.get_context().get_stage()
        prim = stage.GetPrimAtPath(path)
        if prim.IsValid() and not UsdPhysics.CollisionAPI(prim):
            UsdPhysics.CollisionAPI.Apply(prim)
        prim_count += 1

    # ---- 货架: num_shelf_cols 列 × shelves_per_col 段, 宽贴墙 ----
    num_shelf_cols = cfg.num_shelf_cols
    shelves_per_col = cfg.shelves_per_col
    shelf_len_x = cfg.shelf_length              # X 方向长度 (沿通道)
    shelf_wid_y = cfg.shelf_width               # Y 方向宽度 (跨越)
    channel_x = cfg.column_gap                  # 列间净通道宽度
    row_gap_min = cfg.shelf_row_gap_min         # 段间最小间隙
    row_gap_max = cfg.shelf_row_gap_max         # 段间最大间隙

    layout_width = num_shelf_cols * shelf_len_x + max(num_shelf_cols - 1, 0) * channel_x
    available_width = 2.0 * wall_inner
    if layout_width > available_width:
        raise ValueError(
            "warehouse shelf columns do not fit inside arena: "
            f"required={layout_width:.2f}m available={available_width:.2f}m"
        )

    # X layout: Col1 贴左墙, 后续列间隔 column_gap
    col1_center_x = -wall_inner + shelf_len_x / 2.0
    col_centers_x = [col1_center_x]
    for c in range(1, num_shelf_cols):
        prev_right = col_centers_x[-1] + shelf_len_x / 2.0
        col_centers_x.append(prev_right + channel_x + shelf_len_x / 2.0)

    # Y layout: 将整列货架在墙内居中；列间随机间隙使各列开口不完全对齐。
    cfg._shelf_configs_per_col = []

    for col_idx, cx in enumerate(col_centers_x):
        inter_gaps = [
            random.uniform(row_gap_min, row_gap_max)
            for _ in range(max(shelves_per_col - 1, 0))
        ]
        layout_height = shelves_per_col * shelf_wid_y + sum(inter_gaps)
        available_height = 2.0 * wall_inner
        if layout_height > available_height:
            raise ValueError(
                "warehouse shelf rows do not fit inside arena: "
                f"required={layout_height:.2f}m available={available_height:.2f}m"
            )
        edge_margin = (available_height - layout_height) / 2.0
        shelf_data = []
        current_y = wall_inner - edge_margin
        for s in range(shelves_per_col):
            center_y = current_y - shelf_wid_y / 2.0
            h = random.uniform(cfg.shelf_height_min, cfg.shelf_height_max)
            shelf_data.append({
                "center_x": cx,
                "center_y": center_y,
                "half_length": shelf_len_x / 2.0,
                "half_width": shelf_wid_y / 2.0,       # 0.5m in Y
                "height": h,
            })
            current_y -= shelf_wid_y
            if s < shelves_per_col - 1:
                current_y -= inter_gaps[s]

        cfg._shelf_configs_per_col.append(shelf_data)

        # Spawn shelf prims
        for s, sd in enumerate(shelf_data):
            h = sd["height"]
            cuboid_cfg = sim_utils.CuboidCfg(
                size=(shelf_len_x, shelf_wid_y, h),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    disable_gravity=True, kinematic_enabled=True,
                ),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.6, 0.55, 0.35),
                ),
            )
            path = f"/World/envs/env_0/Warehouse_Shelf_C{col_idx}_S{s}"
            cuboid_cfg.func(path, cuboid_cfg, translation=(sd["center_x"], sd["center_y"], h / 2.0))

            from pxr import UsdPhysics
            import omni.usd
            stage = omni.usd.get_context().get_stage()
            prim = stage.GetPrimAtPath(path)
            if prim.IsValid() and not UsdPhysics.CollisionAPI(prim):
                UsdPhysics.CollisionAPI.Apply(prim)
            prim_count += 1

    print(f"[Warehouse] spawned: 4 walls + {num_shelf_cols}×{shelves_per_col} shelves = {prim_count} prims")

    # ---- 圆柱: runtime 采样, 不在 spawn 阶段创建 ----
    # 圆柱通过 WarehouseWorldManager 的 sample/resample 动态管理
    cfg._cylinder_radii = []
    cfg._cylinder_heights = []


# ===================================================================
# 3. Warehouse World Manager
# ===================================================================
class WarehouseWorldManager:
    """管理仓库障碍物布局、圆柱注入和 warp mesh 构建."""

    def __init__(self, num_envs: int, cfg: WarehouseWorldConfig, device: str):
        self.cfg = cfg
        self.num_envs = num_envs
        self.device = device
        self.arena_half = cfg.arena_size / 2.0

        # 货架数据 (所有 env 共享, 不变化)
        self._shelf_data = cfg._shelf_configs_per_col  # list of list of dict

        # Build shelf boundary for channel checks (stored once)
        self._shelf_bounds = []  # list of (cx, cy, half_x, half_y)
        for col_shelves in self._shelf_data:
            for sd in col_shelves:
                self._shelf_bounds.append((
                    sd["center_x"], sd["center_y"],
                    sd["half_length"],     # X extent (2m)
                    sd["half_width"],      # Y extent (0.5m)
                ))

        # 圆柱 (通过 setup_cylinders 配置)
        self.num_cylinders = 0
        self.cylinder_radii_list = []
        self.cylinder_heights_list = []
        self.positions = None  # (N, M, 2)

    # ==================================================================
    # Cylinder management
    # ==================================================================
    def setup_cylinders(self, radii: list, heights: list):
        """Set cylinder parameters (called once at init)."""
        self.cylinder_radii_list = radii
        self.cylinder_heights_list = heights
        M = len(radii)
        self.num_cylinders = M
        self.radii = torch.tensor(radii, device=self.device)
        self.heights = torch.tensor(heights, device=self.device)
        self.positions = torch.zeros(self.num_envs, M, 2, device=self.device)

    def spawn_cylinder_prims(self):
        """Spawn cylinder prim templates in env_0."""
        for i in range(self.num_cylinders):
            r = self.cylinder_radii_list[i]
            h = self.cylinder_heights_list[i]
            cyl_cfg = sim_utils.CylinderCfg(
                radius=r, height=h,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    disable_gravity=True, kinematic_enabled=True,
                ),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.6, 0.2, 0.1),
                ),
            )
            path = f"/World/envs/env_0/Cylinder_{i}"
            cyl_cfg.func(path, cyl_cfg, translation=(0.0, 0.0, h / 2.0))
            from pxr import UsdPhysics
            import omni.usd
            stage = omni.usd.get_context().get_stage()
            prim = stage.GetPrimAtPath(path)
            if prim.IsValid() and not UsdPhysics.CollisionAPI(prim):
                UsdPhysics.CollisionAPI.Apply(prim)

    # ==================================================================
    # Cylinder placement (均匀 + 最小间距)
    # ==================================================================
    def sample_cylinders(self, env_ids, num_cylinders: int, ensure_per_channel: int = 1):
        """在货架外均匀放置圆柱，并满足圆柱之间的最小间距约束."""
        n = len(env_ids)
        M = num_cylinders
        wall_inner = self.arena_half - self.cfg.wall_thickness
        min_dist = self.cfg.cylinder_spawn_safety * 2 + 0.3  # 圆柱间最小间距
        M_total = self.num_cylinders
        new_pos = torch.full((n, M_total, 2), 9999.0, device=self.device)

        for env_i_idx, env_i in enumerate(env_ids.tolist()):
            placed = []
            for cyl_i in range(M):
                r_i = self.radii[cyl_i].item()
                for _ in range(200):
                    cx = -wall_inner + r_i + torch.rand(1, device=self.device).item() * \
                         (2 * wall_inner - 2 * r_i)
                    cy = -wall_inner + r_i + torch.rand(1, device=self.device).item() * \
                         (2 * wall_inner - 2 * r_i)
                    # 圆柱不能与货架几何重叠；保留少量数值余量。
                    shelf_clearance = 0.05
                    if any(
                        math.hypot(
                            max(abs(cx - scx) - shx, 0.0),
                            max(abs(cy - scy) - shy, 0.0),
                        ) < r_i + shelf_clearance
                        for scx, scy, shx, shy in self._shelf_bounds
                    ):
                        continue

                    # 检查与已放置圆柱的间距
                    ok = True
                    for px, py, pr in placed:
                        dist = ((cx - px)**2 + (cy - py)**2)**0.5
                        if dist < r_i + pr + min_dist:
                            ok = False
                            break
                    if ok:
                        new_pos[env_i_idx, cyl_i, 0] = cx
                        new_pos[env_i_idx, cyl_i, 1] = cy
                        placed.append((cx, cy, r_i))
                        break
                else:
                    raise RuntimeError(
                        "unable to place warehouse cylinders without overlap: "
                        f"env={env_i} placed={len(placed)}/{M}; "
                        "reduce phase*_num_cylinders or cylinder_spawn_safety"
                    )

        self.positions[env_ids] = new_pos
        arena_m2 = 4 * wall_inner * wall_inner
        print(f"[Cylinders] total={M}, arena={arena_m2:.0f}m², "
              f"density=1cyl/{arena_m2/M:.1f}m², min_dist={min_dist:.2f}m")

    # ==================================================================
    # Sim write
    # ==================================================================
    def write_cylinder_poses_to_sim(self, scene, num_envs: int, env_origins_xy: torch.Tensor):
        """Set cylinder positions via RigidObject."""
        for obs_i in range(self.num_cylinders):
            obs = scene.rigid_objects.get(f"cylinder_{obs_i}")
            if obs is None:
                continue
            try:
                state = obs.data.default_root_state
                if state is None:
                    continue
                poses = state.clone()
                poses_view = poses.view(num_envs, -1, 13)
                for env_i in range(num_envs):
                    wx = self.positions[env_i, obs_i, 0].item() + env_origins_xy[env_i, 0].item()
                    wy = self.positions[env_i, obs_i, 1].item() + env_origins_xy[env_i, 1].item()
                    wz = self.heights[obs_i].item() / 2.0
                    poses_view[env_i, 0, 0] = wx
                    poses_view[env_i, 0, 1] = wy
                    poses_view[env_i, 0, 2] = wz
                    poses_view[env_i, 0, 3] = 1.0
                    poses_view[env_i, 0, 4:7] = 0.0
                obs.write_root_state_to_sim(poses)
            except Exception:
                pass

    # ==================================================================
    # Spawn block system (数量由当前货架布局动态决定)
    # ==================================================================
    def _get_col_centers_x(self) -> list:
        """返回当前布局的列 X 中心."""
        cfg = self.cfg
        wall_inner = self.arena_half - cfg.wall_thickness
        shelf_len = cfg.shelf_length
        cols = [-wall_inner + shelf_len / 2.0]
        for c in range(1, cfg.num_shelf_cols):
            prev_right = cols[-1] + shelf_len / 2.0
            cols.append(prev_right + cfg.column_gap + shelf_len / 2.0)
        return cols

    def _build_spawn_blocks(self) -> list:
        """构建 spawn 区块列表.

        返回 list of dict:
          {id, type: "row"|"col", x_min, x_max, y_min, y_max, col_idx|col_gap_idx}

        区块类型:
          - "row": 货架段之间以及货架与墙之间的 Y 间隙
          - "col": 列间 X 通道，以及布局两侧仍足够宽的边缘通道
        """
        cfg = self.cfg
        wall_inner = self.arena_half - cfg.wall_thickness
        shelf_hx = cfg.shelf_length / 2.0
        shelf_hy = cfg.shelf_width / 2.0

        blocks = []
        block_id = 0

        # ---- Row-channel blocks: 每列的段间 Y 间隙 ----
        for col_idx, col_shelves in enumerate(self._shelf_data):
            cx = col_shelves[0]["center_x"]
            x_min = cx - shelf_hx
            x_max = cx + shelf_hx

            # 收集该列所有货架的 Y 区间
            shelf_ys = sorted([
                (sd["center_y"] - shelf_hy, sd["center_y"] + shelf_hy)
                for sd in col_shelves
            ], key=lambda x: x[0])

            # Y 间隙: 墙到第一个货架, 货架之间, 最后一个货架到墙
            gaps = []
            prev = -wall_inner
            for sy0, sy1 in shelf_ys:
                if sy0 - prev > 0.4:  # 最小 0.4m 可用间隙
                    gaps.append((prev, sy0))
                prev = max(prev, sy1)
            if wall_inner - prev > 0.4:
                gaps.append((prev, wall_inner))

            for gy0, gy1 in gaps:
                blocks.append({
                    "id": block_id,
                    "type": "row",
                    "x_min": x_min, "x_max": x_max,
                    "y_min": gy0, "y_max": gy1,
                    "col_idx": col_idx,
                })
                block_id += 1

        # ---- Col-channel blocks: 列间 X 通道 ----
        col_centers = self._get_col_centers_x()

        # 左侧 margin (墙→col0)
        x0 = -wall_inner
        x1 = col_centers[0] - shelf_hx
        if x1 - x0 > 0.8:  # 至少 0.8m 宽才可用
            blocks.append({
                "id": block_id, "type": "col",
                "x_min": x0, "x_max": x1,
                "y_min": -wall_inner, "y_max": wall_inner,
                "col_gap_idx": 0,
            })
            block_id += 1

        # 列间通道
        for i in range(len(col_centers) - 1):
            x0 = col_centers[i] + shelf_hx
            x1 = col_centers[i + 1] - shelf_hx
            if x1 - x0 > 0.8:
                blocks.append({
                    "id": block_id, "type": "col",
                    "x_min": x0, "x_max": x1,
                    "y_min": -wall_inner, "y_max": wall_inner,
                    "col_gap_idx": i + 1,
                })
                block_id += 1

        # 右侧 margin (col{last}→墙)
        x0 = col_centers[-1] + shelf_hx
        x1 = wall_inner
        if x1 - x0 > 0.8:
            blocks.append({
                "id": block_id, "type": "col",
                "x_min": x0, "x_max": x1,
                "y_min": -wall_inner, "y_max": wall_inner,
                "col_gap_idx": len(col_centers),
            })
            block_id += 1

        # ---- 过滤掉与货架 (其他列) 大量重叠的 row block ----
        # 如果一个 row block 的 >60% 面积被其他列货架覆盖, 则移除
        valid_blocks = []
        for blk in blocks:
            if blk["type"] == "col":
                valid_blocks.append(blk)
                continue
            # 检查与其他列货架的重叠
            blocked_area = 0.0
            total_area = (blk["x_max"] - blk["x_min"]) * (blk["y_max"] - blk["y_min"])
            for ci, col_shelves in enumerate(self._shelf_data):
                if ci == blk.get("col_idx", -1):
                    continue
                for sd in col_shelves:
                    # 重叠区域
                    ox0 = max(blk["x_min"], sd["center_x"] - sd["half_length"])
                    ox1 = min(blk["x_max"], sd["center_x"] + sd["half_length"])
                    oy0 = max(blk["y_min"], sd["center_y"] - sd["half_width"])
                    oy1 = min(blk["y_max"], sd["center_y"] + sd["half_width"])
                    if ox1 > ox0 and oy1 > oy0:
                        blocked_area += (ox1 - ox0) * (oy1 - oy0)
            if blocked_area < 0.6 * total_area:
                valid_blocks.append(blk)

        row_count = sum(1 for b in valid_blocks if b["type"] == "row")
        col_count = sum(1 for b in valid_blocks if b["type"] == "col")
        self._spawn_blocks = valid_blocks
        print(f"[SpawnBlocks] {len(valid_blocks)} valid ({row_count} row + {col_count} col)")
        return valid_blocks

    # ==================================================================
    # OBB-vs-AABB + Cylinder safety check (SAT + circle distance)
    # ==================================================================
    @staticmethod
    def _check_obb_safe(emma_cx: float, emma_cy: float, emma_yaw: float,
                        emma_hl: float, emma_hw: float,
                        shelf_bounds: list, wall_inner: float,
                        cylinders: list | None = None,
                        spawn_margin: float = 0.10) -> bool:
        """检查 Emma 的旋转矩形是否与所有货架/墙壁/圆柱分离.

        Args:
            emma_cx, emma_cy: Emma 中心位置 (world)
            emma_yaw: Emma heading (rad)
            emma_hl: Emma half_length (0.42m, body +X)
            emma_hw: Emma half_width  (0.267m, body +Y)
            spawn_margin: 障碍物膨胀余量 (m), 保证 LiDAR clearance > 0.05m
            shelf_bounds: [(cx, cy, half_x, half_y), ...] 货架 AABB
            wall_inner: 墙壁内边界
            cylinders: [(cx, cy, r), ...] 圆柱 (可选)

        Returns:
            True if position is safe
        """
        cos = math.cos(emma_yaw)
        sin = math.sin(emma_yaw)

        # Emma 的 4 个角 (body frame → world)
        corners_b = [(-emma_hl, -emma_hw), (-emma_hl, emma_hw),
                     (emma_hl, -emma_hw),  (emma_hl, emma_hw)]
        emma_corners_w = [(emma_cx + x * cos - y * sin,
                           emma_cy + x * sin + y * cos) for x, y in corners_b]

        # Emma 的轴 (用于 SAT 投影)
        emma_axes = [(cos, sin), (-sin, cos)]

        def _project_emma(ax, ay):
            """Emma 矩形在轴上的投影区间."""
            c_proj = emma_cx * ax + emma_cy * ay
            r = abs(emma_hl * (cos * ax + sin * ay)) + abs(emma_hw * (-sin * ax + cos * ay))
            return (c_proj - r, c_proj + r)

        # 墙壁检查 (加 LiDAR clearance margin)
        inner = wall_inner - spawn_margin
        for wx, wy in emma_corners_w:
            if abs(wx) >= inner or abs(wy) >= inner:
                return False

        # 对每个货架做 SAT (膨胀 spawn_margin 保证 LiDAR clearance > 0.05m)
        for (scx, scy, shx, shy) in shelf_bounds:
            shx_m = shx + spawn_margin
            shy_m = shy + spawn_margin

            # 快速 AABB rejection
            if (abs(emma_cx - scx) > shx_m + emma_hl + emma_hw or
                abs(emma_cy - scy) > shy_m + emma_hl + emma_hw):
                continue

            s_on_x = (scx - shx_m, scx + shx_m)
            e_on_x = _project_emma(1, 0)
            if e_on_x[1] <= s_on_x[0] or s_on_x[1] <= e_on_x[0]:
                continue

            s_on_y = (scy - shy_m, scy + shy_m)
            e_on_y = _project_emma(0, 1)
            if e_on_y[1] <= s_on_y[0] or s_on_y[1] <= e_on_y[0]:
                continue

            for ax, ay in emma_axes:
                s_proj = scx * ax + scy * ay
                s_r = abs(shx_m * ax) + abs(shy_m * ay)
                e0, e1 = _project_emma(ax, ay)
                if e1 <= s_proj - s_r or s_proj + s_r <= e0:
                    break
            else:
                return False

        # ---- 圆柱碰撞 (OBB vs circle, 膨胀 spawn_margin) ----
        if cylinders:
            for (ccx, ccy, cr) in cylinders:
                cr_m = cr + spawn_margin
                # 快速 rejection
                if abs(emma_cx - ccx) > emma_hl + emma_hw + cr_m:
                    continue
                if abs(emma_cy - ccy) > emma_hl + emma_hw + cr_m:
                    continue
                # 将圆柱中心变换到 Emma body frame
                dx = ccx - emma_cx
                dy = ccy - emma_cy
                bx = dx * cos + dy * sin
                by = -dx * sin + dy * cos
                # 最近点 (clamp 到矩形)
                cx_b = max(-emma_hl, min(emma_hl, bx))
                cy_b = max(-emma_hw, min(emma_hw, by))
                # 距离
                dist_sq = (bx - cx_b)**2 + (by - cy_b)**2
                if dist_sq < cr_m * cr_m:
                    return False

        return True

    # ==================================================================
    # Spawn pair sampling
    # ==================================================================
    def sample_spawn_pair(self, n_pairs: int, emma_hl: float, emma_hw: float,
                          rng_seed: int = 0,
                          cyl_positions: torch.Tensor = None,
                          cyl_radii: torch.Tensor = None,
                          num_active_cyl: int = 0,
                          goal_reach_dist: float = 0.3) -> dict:
        """为 n_pairs 个环境采样 Emma + Goal 位置.

        Args:
            n_pairs: 需要采样的环境数
            emma_hl: Emma half_length (0.42m)
            emma_hw: Emma half_width (0.267m)
            rng_seed: 随机种子
            cyl_positions: (n_pairs, M, 2) 每个环境的圆柱位置 (local frame)
            cyl_radii: (M,) 圆柱半径
            num_active_cyl: 实际使用的圆柱数 (前 num_active_cyl 个)
            goal_reach_dist: Goal 到达阈值 (ICR→Goal < 此距离)

        Returns:
            {emma_x, emma_y, emma_yaw, goal_x, goal_y, block_emma, block_goal}
        """
        cfg = self.cfg
        wall_inner = self.arena_half - cfg.wall_thickness

        # Lazy init spawn blocks
        if not hasattr(self, '_spawn_blocks') or not self._spawn_blocks:
            self._build_spawn_blocks()

        blocks = self._spawn_blocks
        if len(blocks) < 2:
            raise RuntimeError(f"Need >=2 spawn blocks, got {len(blocks)}")

        rng = random.Random(rng_seed)

        # 准备圆柱数据 (按 env)
        cyl_per_env = []
        if cyl_positions is not None and cyl_radii is not None and num_active_cyl > 0:
            for ei in range(n_pairs):
                cyl_list = []
                for ci in range(num_active_cyl):
                    cx = cyl_positions[ei, ci, 0].item()
                    cy = cyl_positions[ei, ci, 1].item()
                    cr = cyl_radii[ci].item()
                    if abs(cx) < 9990 and abs(cy) < 9990:  # 有效圆柱 (非 sentinel)
                        cyl_list.append((cx, cy, cr))
                cyl_per_env.append(cyl_list)
        else:
            cyl_per_env = [None] * n_pairs

        emma_xs, emma_ys, emma_yaws = [], [], []
        goal_xs, goal_ys = [], []
        blk_emma, blk_goal = [], []

        # 局部导航难度约束:
        #   Emma/Goal 欧氏距离限制在配置区间内；短时绕行可以远离目标，
        #   但任务不会退化成需要全局地图先验的长距离仓库导航。
        for pair_i in range(n_pairs):
            cylinders_i = cyl_per_env[pair_i]
            found = False
            for _ in range(500):
                # ---- 选择两个不同区块 ----
                bi, bj = rng.sample(range(len(blocks)), 2)
                blk_i = blocks[bi]
                blk_j = blocks[bj]

                # ---- 在 block_i 内采样 Emma 位置 ----
                for _ in range(50):
                    ex = rng.uniform(blk_i["x_min"], blk_i["x_max"])
                    ey = rng.uniform(blk_i["y_min"], blk_i["y_max"])
                    eyaw = rng.uniform(0, 2 * math.pi)

                    if not self._check_obb_safe(
                        ex, ey, eyaw, emma_hl, emma_hw,
                        self._shelf_bounds, wall_inner, cylinders_i,
                    ):
                        continue
                    break
                else:
                    continue

                # ---- 在 block_j 内采样 Goal 位置 ----
                for _ in range(50):
                    gx = rng.uniform(blk_j["x_min"], blk_j["x_max"])
                    gy = rng.uniform(blk_j["y_min"], blk_j["y_max"])

                    # 欧氏距离
                    dist_eg = math.sqrt((ex - gx)**2 + (ey - gy)**2)
                    if dist_eg < cfg.emma_goal_min_dist:
                        continue
                    if dist_eg > cfg.emma_goal_max_dist:
                        continue

                    # Goal 安全检测: 需要保证 Emma 到达时 (ICR 在 reach_dist 内)
                    # 身体不与障碍物重叠: safe_r = marker_r + reach_dist + half_diag
                    robot_half_diag = math.sqrt(emma_hl**2 + emma_hw**2)
                    goal_safe_r = cfg.goal_marker_radius + goal_reach_dist + robot_half_diag
                    safe_goal = True
                    if abs(gx) >= wall_inner - goal_safe_r or abs(gy) >= wall_inner - goal_safe_r:
                        safe_goal = False
                    if safe_goal:
                        for (scx, scy, shx, shy) in self._shelf_bounds:
                            if (abs(gx - scx) < shx + goal_safe_r and
                                abs(gy - scy) < shy + goal_safe_r):
                                safe_goal = False
                                break
                    if safe_goal and cylinders_i:
                        for (ccx, ccy, cr) in cylinders_i:
                            if math.sqrt((gx - ccx)**2 + (gy - ccy)**2) < cr + goal_safe_r:
                                safe_goal = False
                                break
                    if not safe_goal:
                        continue
                    break
                else:
                    continue

                emma_xs.append(ex)
                emma_ys.append(ey)
                emma_yaws.append(eyaw)
                goal_xs.append(gx)
                goal_ys.append(gy)
                blk_emma.append(bi)
                blk_goal.append(bj)
                found = True
                break

            if not found:
                raise RuntimeError(
                    "unable to sample a safe warehouse Emma/Goal pair within "
                    f"{cfg.emma_goal_min_dist:.1f}~{cfg.emma_goal_max_dist:.1f}m; "
                    "reduce obstacle density or relax the goal-distance range"
                )

        return {
            "emma_x": emma_xs, "emma_y": emma_ys, "emma_yaw": emma_yaws,
            "goal_x": goal_xs, "goal_y": goal_ys,
            "block_emma": blk_emma, "block_goal": blk_goal,
        }

    # ==================================================================
    # Collision check (geometric, independent of LiDAR)
    # ==================================================================
    def check_collision(self, robot_pos_w: torch.Tensor, robot_yaw: torch.Tensor,
                        env_origins: torch.Tensor = None) -> torch.Tensor:
        """几何碰撞检查 (不依赖 LiDAR).

        Args:
            robot_pos_w: (N, 2) robot world-frame XY positions
            robot_yaw: (N,) robot heading
            env_origins: (N, 2) env origin XY (用于转换到局部坐标)
        Returns:
            (N,) bool: True = collision with wall or shelf
        """
        cfg = self.cfg
        wall_inner = self.arena_half - cfg.wall_thickness

        # 转换到局部坐标 (shelf_bounds 和 wall_inner 都是局部坐标)
        if env_origins is not None:
            pos = robot_pos_w - env_origins
        else:
            pos = robot_pos_w

        n = pos.shape[0]
        device = pos.device

        # 墙壁碰撞 (矢量)
        collision = (pos[:, 0].abs() > wall_inner - 0.27) | \
                    (pos[:, 1].abs() > wall_inner - 0.27)

        # 货架碰撞: OBB-vs-AABB 投影到世界 X/Y 轴
        # Emma OBB 在 X/Y 轴上的投影半径取决于 yaw
        cos_yaw = torch.cos(robot_yaw)
        sin_yaw = torch.sin(robot_yaw)
        emma_hl = 0.42
        emma_hw = 0.267
        ext_x = emma_hl * cos_yaw.abs() + emma_hw * sin_yaw.abs()  # (N,)
        ext_y = emma_hl * sin_yaw.abs() + emma_hw * cos_yaw.abs()  # (N,)

        px = pos[:, 0]  # (N,)
        py = pos[:, 1]  # (N,)

        for (scx, scy, shx, shy) in self._shelf_bounds:
            in_x = (px - scx).abs() < (shx + ext_x)     # (N,)
            in_y = (py - scy).abs() < (shy + ext_y)     # (N,)
            collision |= (in_x & in_y)

        return collision

    # ==================================================================
    # Warp mesh
    # ==================================================================
    def build_warp_mesh(self, env_origins: torch.Tensor, device: str):
        """Build warp mesh: ground + walls + shelves + cylinders."""
        all_meshes = []

        total_size = self.cfg.arena_size * max(int(math.sqrt(self.num_envs)), 1)
        ground = trimesh.creation.box(extents=[total_size, total_size, 0.02])
        ground.apply_translation([0, 0, -0.01])
        all_meshes.append(ground)

        for env_i in range(self.num_envs):
            origin = env_origins[env_i, :2].cpu()

            # 墙壁
            for (_, (cx, cy), (sx, sy)), h in zip([
                ("TopWall", (0.0, self.arena_half - self.cfg.wall_thickness / 2.0),
                 (self.cfg.arena_size, self.cfg.wall_thickness)),
                ("BottomWall", (0.0, -self.arena_half + self.cfg.wall_thickness / 2.0),
                 (self.cfg.arena_size, self.cfg.wall_thickness)),
                ("LeftWall", (-self.arena_half + self.cfg.wall_thickness / 2.0, 0.0),
                 (self.cfg.wall_thickness, self.cfg.arena_size)),
                ("RightWall", (self.arena_half - self.cfg.wall_thickness / 2.0, 0.0),
                 (self.cfg.wall_thickness, self.cfg.arena_size)),
            ], self.cfg._wall_heights):
                wx = cx + origin[0].item()
                wy = cy + origin[1].item()
                box = trimesh.creation.box(extents=[sx, sy, h])
                box.apply_translation([wx, wy, h / 2.0])
                all_meshes.append(box)

            # 货架
            for col_shelves in self._shelf_data:
                for sd in col_shelves:
                    wx = sd["center_x"] + origin[0].item()
                    wy = sd["center_y"] + origin[1].item()
                    sx = sd["half_length"] * 2.0
                    sy = sd["half_width"] * 2.0
                    h = sd["height"]
                    box = trimesh.creation.box(extents=[sx, sy, h])
                    box.apply_translation([wx, wy, h / 2.0])
                    all_meshes.append(box)

            # 圆柱
            if self.positions is not None:
                for obs_i in range(self.num_cylinders):
                    # 当前 phase 未启用的模板使用 sentinel 隐藏，不应加入 LiDAR mesh。
                    if (self.positions[env_i, obs_i].abs() >= 9990.0).any():
                        continue
                    r = self.radii[obs_i].item()
                    h = self.heights[obs_i].item()
                    wx = self.positions[env_i, obs_i, 0].item() + origin[0].item()
                    wy = self.positions[env_i, obs_i, 1].item() + origin[1].item()
                    cyl = trimesh.creation.cylinder(radius=r, height=h, sections=32)
                    cyl.apply_translation([wx, wy, h / 2.0])
                    all_meshes.append(cyl)

        merged = trimesh.util.concatenate(all_meshes)
        wp = convert_to_warp_mesh(merged.vertices, merged.faces, device=device)
        if self.num_envs == 1 and self.positions is not None and self.num_cylinders > 0:
            p0 = self.positions[0, 0]
            print(f"[WarpMesh] envs={self.num_envs}, faces={merged.faces.shape[0]}, "
                  f"cyl0_xy=({p0[0]:.2f},{p0[1]:.2f}), "
                  f"cyl0_r={self.radii[0].item():.2f}")
        return wp

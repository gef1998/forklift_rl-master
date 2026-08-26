"""Costmap 生成器 — GT 障碍物投影到 BEV 栅格, 用于 RL 导航观测.

核心思路 (参考 docs/costmap_version.md):
  仿真中用 GT 位姿生成 costmap, 真实中用 3D 检测结果生成 costmap.
  RL 策略只认 costmap 这一种"世界语言", 以此桥接 Sim2Real.

输出: (N, C, H, W) 栅格张量, 占据=1.0, 自由=0.0.
"""

import torch
import torch.nn.functional as F


def _quat_rotate_inverse(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """纯 PyTorch 四元数逆旋转 (等价于 isaaclab quat_apply_inverse).

    对单位四元数, 逆旋转 = 共轭旋转.
    q: (..., 4) wxyz format.
    v: (..., 3) vector to rotate.
    Returns: (..., 3) rotated vector.
    """
    # 共轭: (w, -x, -y, -z)
    w, x, y, z = q[..., 0], -q[..., 1], -q[..., 2], -q[..., 3]
    vx, vy, vz = v[..., 0], v[..., 1], v[..., 2]
    # t = 2 * cross(q_xyz, v)
    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)
    # result = v + w * t + cross(q_xyz, t)
    rx = vx + w * tx + (y * tz - z * ty)
    ry = vy + w * ty + (z * tx - x * tz)
    rz = vz + w * tz + (x * ty - y * tx)
    return torch.stack([rx, ry, rz], dim=-1)


def _quat_rotate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """纯 PyTorch 四元数正向旋转 (等价于 isaaclab quat_apply).

    q: (..., 4) wxyz format. v: (..., 3).
    """
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    vx, vy, vz = v[..., 0], v[..., 1], v[..., 2]
    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)
    rx = vx + w * tx + (y * tz - z * ty)
    ry = vy + w * ty + (z * tx - x * tz)
    rz = vz + w * tz + (x * ty - y * tx)
    return torch.stack([rx, ry, rz], dim=-1)


def _world_to_body_2d(
    points_w: torch.Tensor,      # (..., 2) 或 (..., 3)
    robot_pos_w: torch.Tensor,   # (N, 3)
    robot_quat_w: torch.Tensor,  # (N, 4) wxyz
) -> torch.Tensor:
    """将世界坐标点转换到 Emma body frame (XY 平面).

    Body frame: 原点 = ICR, +X = 前进方向, +Y = 左侧.
    """
    # 确保是 3D 坐标
    if points_w.shape[-1] == 2:
        points_w = torch.cat([
            points_w,
            torch.zeros(*points_w.shape[:-1], 1, device=points_w.device),
        ], dim=-1)

    # 平移到 robot 坐标系
    rel = points_w - robot_pos_w.unsqueeze(-2)  # (..., N, M, 3)
    # 旋转到 body frame (apply inverse quaternion)
    local = _quat_rotate_inverse(
        robot_quat_w.unsqueeze(-2), rel
    )
    return local[..., :2]  # 只保留 XY


def _grid_coords(
    points_b: torch.Tensor,      # (..., 2) body frame XY
    half_grid: float,
    resolution: float,
    map_res: int,
) -> torch.Tensor:
    """Body frame 坐标 → 栅格行列索引 (float, 可用于插值)."""
    col = (points_b[..., 0] + half_grid) / resolution   # body +X → grid col
    row = (points_b[..., 1] + half_grid) / resolution   # body +Y → grid row (Y 轴翻转在可视化时处理)
    return torch.stack([row, col], dim=-1)               # (..., 2): [row, col]


def generate_static_costmap(
    robot_pos_w: torch.Tensor,         # (N, 3)
    robot_quat_w: torch.Tensor,        # (N, 4) wxyz
    circle_centers_w: torch.Tensor,    # (N, C_circ, 2) 世界 XY — 静态圆柱障碍物
    circle_radii: torch.Tensor,        # (C_circ,)
    rect_centers_w: torch.Tensor,      # (N, C_rect, 2) 世界 XY — 静态长方体障碍物
    rect_half_l: torch.Tensor,         # (C_rect,) 半长 (body +X 方向)
    rect_half_w: torch.Tensor,         # (C_rect,) 半宽
    rect_yaws_w: torch.Tensor,         # (C_rect,) yaw in world frame (rad)
    grid_size: float = 24.0,
    map_res: int = 128,
    inflation: float = 0.25,
    dropout_prob: float = 0.0,
) -> torch.Tensor:
    """生成静态障碍物 costmap (单通道).

    在 _reset_idx 中调用一次, 整个 episode 复用.
    障碍物属性不随 env 变化 (形状一致, 仅位置不同), 支持 GPU 批量生成.

    Args:
        robot_pos_w:   Emma ICR 世界位置, (N, 3).
        robot_quat_w:  Emma 姿态四元数 wxyz, (N, 4).
        circle_centers_w: 静态圆柱中心世界坐标, (N, C_circ, 2). 无效障碍物填 [inf, inf].
        circle_radii: 圆柱半径, (C_circ,).
        rect_centers_w: 静态长方体中心世界坐标, (N, C_rect, 2).
        rect_half_l: 长方体半长, (C_rect,).
        rect_half_w: 长方体半宽, (C_rect,).
        rect_yaws_w: 长方体世界 yaw, (C_rect,).
        grid_size: costmap 覆盖边长 (m), 默认 24m (±12m).
        map_res: 栅格分辨率, 默认 128.
        inflation: 膨胀半径 (m), 默认 0.25.
        dropout_prob: 随机丢帧概率 (sim2real), 默认 0.

    Returns:
        costmap: (N, 1, map_res, map_res), 占据=1.0, 自由=0.0.
    """
    N = robot_pos_w.shape[0]
    device = robot_pos_w.device
    half_grid = grid_size / 2.0
    resolution = grid_size / map_res

    # 初始化空栅格
    occupancy = torch.zeros(N, 1, map_res, map_res, device=device, dtype=torch.float32)

    # ---- 渲染圆形障碍物 (圆柱) ----
    n_circ = circle_radii.shape[0] if circle_radii is not None else 0
    if n_circ > 0:
        # 转换到 body frame
        circ_b = _world_to_body_2d(circle_centers_w, robot_pos_w, robot_quat_w)  # (N, C_circ, 2)
        # 栅格坐标
        circ_grid = _grid_coords(circ_b, half_grid, resolution, map_res)  # (N, C_circ, 2) [row, col]

        # 圆形有效半径 (栅格单元)
        circ_r_cells = (circle_radii + inflation) / resolution  # (C_circ,)

        # 逐障碍物批量渲染 (每批 8 个, 控制显存)
        batch_size = 8
        for start in range(0, n_circ, batch_size):
            end = min(start + batch_size, n_circ)
            centers_batch = circ_grid[:, start:end, :]  # (N, B, 2)
            radii_batch = circ_r_cells[start:end]        # (B,)

            # 生成栅格坐标网格
            rows = torch.arange(map_res, device=device, dtype=torch.float32)
            cols = torch.arange(map_res, device=device, dtype=torch.float32)
            grid_r, grid_c = torch.meshgrid(rows, cols, indexing='ij')  # (H, W) each

            # 每个障碍物: dist_cells = sqrt((row - c_row)^2 + (col - c_col)^2)
            d_row = grid_r.unsqueeze(0).unsqueeze(0) - centers_batch[..., 0].unsqueeze(-1).unsqueeze(-1)
            d_col = grid_c.unsqueeze(0).unsqueeze(0) - centers_batch[..., 1].unsqueeze(-1).unsqueeze(-1)
            dist = torch.sqrt(d_row ** 2 + d_col ** 2)  # (N, B, H, W)

            # 判断占据
            r = radii_batch.view(1, -1, 1, 1)
            hit = dist <= r  # (N, B, H, W)
            occupancy = torch.maximum(occupancy, hit.any(dim=1, keepdim=True).float())

    # ---- 渲染矩形障碍物 (长方体) ----
    n_rect = rect_half_l.shape[0] if rect_half_l is not None else 0
    if n_rect > 0:
        # 转换中心到 body frame
        rect_b = _world_to_body_2d(rect_centers_w, robot_pos_w, robot_quat_w)  # (N, C_rect, 2)
        rect_grid = _grid_coords(rect_b, half_grid, resolution, map_res)        # (N, C_rect, 2)

        # 世界 yaw → body frame yaw: 用 forward vector 算 robot yaw
        forward = _quat_rotate(
            robot_quat_w, torch.tensor([1., 0., 0.], device=device).repeat(N, 1))
        robot_yaw = torch.atan2(forward[:, 1], forward[:, 0])  # (N,)
        rect_yaws_b = rect_yaws_w.unsqueeze(0) - robot_yaw.unsqueeze(-1)  # (N, C_rect) body-frame yaw

        # 膨胀后的半长/半宽 (栅格单元)
        hl_cells = (rect_half_l + inflation) / resolution  # (C_rect,)
        hw_cells = (rect_half_w + inflation) / resolution  # (C_rect,)

        batch_size = 4  # 矩形比圆形更复杂, 减小批大小
        for start in range(0, n_rect, batch_size):
            end = min(start + batch_size, n_rect)
            centers_batch = rect_grid[:, start:end, :]  # (N, B, 2)
            yaws_batch = rect_yaws_b[:, start:end]       # (N, B)
            hl_batch = hl_cells[start:end]                # (B,)
            hw_batch = hw_cells[start:end]                # (B,)

            # 栅格坐标网格
            rows = torch.arange(map_res, device=device, dtype=torch.float32)
            cols = torch.arange(map_res, device=device, dtype=torch.float32)
            grid_r, grid_c = torch.meshgrid(rows, cols, indexing='ij')

            # 相对坐标 (row, col)
            dr = grid_r.unsqueeze(0).unsqueeze(0) - centers_batch[..., 0].unsqueeze(-1).unsqueeze(-1)  # (N,B,H,W)
            dc = grid_c.unsqueeze(0).unsqueeze(0) - centers_batch[..., 1].unsqueeze(-1).unsqueeze(-1)  # (N,B,H,W)

            # 旋转到矩形局部坐标系: 局部 x 沿矩形长边方向
            # 注意: body frame: +X=前进, +Y=左侧. 矩形 yaw 是长边方向 vs body +X.
            cos_a = torch.cos(-yaws_batch).view(N, -1, 1, 1)  # 反向旋转: 栅格 → 矩形局部
            sin_a = torch.sin(-yaws_batch).view(N, -1, 1, 1)
            # 栅格坐标: col(+X body), row(+Y body).  矩形局部 x=长边, y=宽边.
            local_x = dc * cos_a - dr * sin_a   # 矩形长边方向分量
            local_y = dc * sin_a + dr * cos_a   # 矩形宽边方向分量

            # 点是否在膨胀后的矩形内
            hl = hl_batch.view(1, -1, 1, 1)
            hw = hw_batch.view(1, -1, 1, 1)
            inside = (local_x.abs() <= hl) & (local_y.abs() <= hw)  # (N, B, H, W)

            occupancy = torch.maximum(occupancy, inside.any(dim=1, keepdim=True).float())

    # ---- 膨胀 (形态学 max-pool, 消除栅格化锯齿) ----
    # 计算膨胀核大小: inflation / resolution, 至少 3
    kernel_cells = max(3, int(inflation / resolution) * 2 + 1)
    if kernel_cells % 2 == 0:
        kernel_cells += 1  # 确保奇数
    if kernel_cells >= 3:
        padding = kernel_cells // 2
        occupancy = F.max_pool2d(occupancy, kernel_size=kernel_cells, stride=1, padding=padding)

    # ---- Dropout: 随机丢弃障碍物占据, 模拟真实感知漏检 ----
    if dropout_prob > 0 and not torch.is_inference_mode_enabled():
        mask = torch.rand_like(occupancy) >= dropout_prob
        occupancy = occupancy * mask

    return occupancy


def generate_dynamic_costmap(
    robot_pos_w: torch.Tensor,         # (N, 3)
    robot_quat_w: torch.Tensor,        # (N, 4) wxyz
    circle_centers_w: torch.Tensor,    # (N, C_dyn, 2) 世界 XY — 动态圆柱障碍物
    circle_radii: torch.Tensor,        # (C_dyn,)
    grid_size: float = 24.0,
    map_res: int = 128,
    inflation: float = 0.25,
    dropout_prob: float = 0.0,
) -> torch.Tensor:
    """生成动态障碍物 costmap (单通道).

    每 step 调用, 障碍物位置随 _step_dynamic_obstacles() 更新.
    动态障碍物数量少 (~10), 开销可控.

    Args:
        robot_pos_w:  Emma ICR 世界位置, (N, 3).
        robot_quat_w: Emma 姿态四元数 wxyz, (N, 4).
        circle_centers_w: 动态圆柱中心世界坐标, (N, C_dyn, 2).
        circle_radii: 圆柱半径, (C_dyn,).
        grid_size: costmap 覆盖边长 (m).
        map_res: 栅格分辨率.
        inflation: 膨胀半径 (m).
        dropout_prob: 随机丢帧概率.

    Returns:
        costmap: (N, 1, map_res, map_res), 占据=1.0, 自由=0.0.
    """
    # 动态障碍物都是圆形, 直接复用 generate_static_costmap
    return generate_static_costmap(
        robot_pos_w=robot_pos_w,
        robot_quat_w=robot_quat_w,
        circle_centers_w=circle_centers_w,
        circle_radii=circle_radii,
        rect_centers_w=None,
        rect_half_l=None,
        rect_half_w=None,
        rect_yaws_w=None,
        grid_size=grid_size,
        map_res=map_res,
        inflation=inflation,
        dropout_prob=dropout_prob,
    )


def add_ego_footprint(
    costmap: torch.Tensor,                # (N, 1, H, W) 或 (N, H, W)
    robot_half_length: float,
    robot_half_width: float,
    forward_speed: torch.Tensor,          # (N,) body-frame 前进速度 (m/s)
    latency: float,                       # 盲开时窗 (s), 如 0.1s for 10Hz
    map_res: int = 64,
    grid_size: float = 12.0,
) -> torch.Tensor:
    """在 costmap 上叠加自车占位 (ego-footprint).

    用于模拟真实 LiDAR 更新间隙的盲开保护:
      - 车身矩形占据 → 策略不会"穿过"自己
      - 速度方向膨胀带 → 策略不会冲向盲开区域的障碍物

    Args:
        costmap: 待叠加的 costmap, (N, 1, H, W) 或 (N, H, W).
        robot_half_length: 车身半长 (m), body +X 方向.
        robot_half_width:  车身半宽 (m), body +Y 方向.
        forward_speed: (N,) body-frame 前进速度, 正=前进.
        latency: 观测盲开时窗 (s).
        map_res: 栅格分辨率.
        grid_size: costmap 覆盖边长 (m).

    Returns:
        costmap with ego-footprint, same shape as input.
    """
    if costmap.dim() == 3:
        costmap = costmap.unsqueeze(1)  # (N, H, W) → (N, 1, H, W)
    N, C, H, W = costmap.shape
    device = costmap.device
    resolution = grid_size / map_res
    center = H / 2.0  # Emma 在 costmap 中心

    # 车身半长/半宽 (栅格单元)
    hl_cells = robot_half_length / resolution
    hw_cells = robot_half_width / resolution

    # 速度方向膨胀: forward_speed * latency / resolution (栅格单元)
    inflate_cells = (forward_speed.clamp(min=0.0) * latency / resolution).view(N, 1, 1)

    # 生成栅格坐标 (只做一次, 动态计算膨胀)
    rows = torch.arange(H, device=device, dtype=torch.float32)
    cols = torch.arange(W, device=device, dtype=torch.float32)
    grid_r, grid_c = torch.meshgrid(rows, cols, indexing='ij')  # (H, W)

    dr = (grid_r - center).abs().unsqueeze(0)  # (1, H, W), 行偏移 (body Y)
    dc = (grid_c - center).unsqueeze(0)         # (1, H, W), 列偏移 (body X)

    # 车身矩形: 仅在 -hl..hl (body X) 和 -hw..hw (body Y) 范围内
    # 速度膨胀沿 body +X 方向延长 hl
    body_x_front = hl_cells + inflate_cells  # (N, 1, 1) 前方膨胀
    body_x_rear = hl_cells + 0.5 * inflate_cells  # 后方也有小幅膨胀

    inside_x = (dc <= body_x_front) & (dc >= -body_x_rear)  # (N, H, W)
    inside_y = dr <= hw_cells  # (N, H, W), 已取 abs

    ego_mask = (inside_x & inside_y).unsqueeze(1)  # (N, 1, H, W)

    # 叠加: ego 占据区域标记为 1.0
    costmap = torch.maximum(costmap, ego_mask.float())
    return costmap


def visualize_costmap(costmap: torch.Tensor, save_path: str = "costmap.png", env_idx: int = 0):
    """导出 costmap 为 PNG 用于调试 (纯 numpy/PIL, 无需 cv2).

    Args:
        costmap: (N, C, H, W) 或 (C, H, W) 或 (H, W).
        save_path: 输出路径.
        env_idx: 要可视化的 env 索引.
    """
    import numpy as np

    if costmap.dim() == 4:
        cmap = costmap[env_idx].detach().cpu().numpy()  # (C, H, W)
    elif costmap.dim() == 3:
        cmap = costmap.detach().cpu().numpy()
    else:
        cmap = costmap.detach().cpu().numpy()[np.newaxis, ...]

    C = cmap.shape[0]
    imgs = []
    for c in range(C):
        ch = cmap[c]
        img = (ch * 255).astype(np.uint8)
        img = np.flipud(img)  # body +Y → 图像上方
        imgs.append(img)

    out = imgs[0] if len(imgs) == 1 else np.hstack(imgs)

    # 使用 PIL 保存 (跨平台, 无需 cv2)
    from PIL import Image
    Image.fromarray(out, mode='L').save(save_path)
    return out

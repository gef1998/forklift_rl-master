"""Emma Costmap Navigation 环境 — 基于 costmap 观测的二轮差速避障导航.

与 emma_obs_nav 的核心差异:
  1. 观测: costmap (2×128×128) + robot_state (8), 不再使用 LiDAR RayCaster
  2. 场景: 静态 Cube + Cylinder 混合 + 动态游走 Cylinder
  3. 避障奖励: 基于 costmap 射线采样, 替代 VO cone_half 计算
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject, RigidObjectCfg
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.shapes.shapes_cfg import CuboidCfg, CylinderCfg, SphereCfg

from .costmap_generator_nav import (
    generate_static_costmap, generate_dynamic_costmap, add_ego_footprint)
from .emma_costmap_nav_world import NavWorldManager, NavWorldConfig, spawn_nav_obstacles
from .emma_costmap_nav_env_cfg import EmmaCostmapNavEnvCfg


# ============================================================================
# 辅助函数
# ============================================================================
def _rotate_world_to_body_2d(vec_w: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
    """世界方向向量 → body frame (只用 yaw). 这是纯旋转, 与 _world_to_body_2d 不同."""
    cos_yaw = torch.cos(yaw)
    sin_yaw = torch.sin(yaw)
    x_b = vec_w[..., 0] * cos_yaw + vec_w[..., 1] * sin_yaw
    y_b = -vec_w[..., 0] * sin_yaw + vec_w[..., 1] * cos_yaw
    return torch.stack([x_b, y_b], dim=-1)


# ============================================================================
# EmmaCostmapNavEnv
# ============================================================================
class EmmaCostmapNavEnv(DirectRLEnv):
    """Emma AMR costmap 避障导航环境."""

    cfg: EmmaCostmapNavEnvCfg

    def __init__(self, cfg, render_mode: str | None = None, **kwargs):
        # ---- 状态变量 (必须在 super().__init__ 之前, 因为 _setup_scene 中会用到) ----
        self._world = None
        self._goal_marker = None
        self._static_costmap = None        # 缓存: 静态层 (N, 1, H, W)
        self._active_num_dynamic = 0       # 当前 phase 的活跃动态障碍物数

        super().__init__(cfg, render_mode, **kwargs)

        # ---- 场景管理器 ----
        self._world = NavWorldManager(self.num_envs, cfg.nav_world, self.device)
        # ---- 初始 phase 的动态障碍物数量
        self._active_num_dynamic = getattr(cfg, f"phase{cfg.training_phase}_num_dynamic")

        # ---- Costmap 频率控制 (模拟真实 LiDAR 10Hz 更新) ----
        self._costmap_step_counter = 0
        # 控制频率 = 1 / (sim.dt * decimation) = 1 / 0.02 = 50Hz
        control_freq = 1.0 / (cfg.sim.dt * cfg.decimation)
        self._costmap_update_interval = max(1, int(control_freq / cfg.costmap_update_freq))
        # 缓存动态层 (更新间隙复用上一帧)
        self._cached_dynamic_costmap = None
        # costmap 盲开时窗 = 1 / update_freq
        self._costmap_latency = 1.0 / cfg.costmap_update_freq

        # ---- 获取 joint 索引 ----
        joint_names = self.robot.data.joint_names
        self._wheel_left_idx = joint_names.index("wheel_left")
        self._wheel_right_idx = joint_names.index("wheel_right")

        # ---- Goal 状态 ----
        self.goal_pos_xy = torch.zeros(self.num_envs, 2, device=self.device)

        # ---- 跟踪状态 ----
        self._prev_goal_dist = torch.zeros(self.num_envs, device=self.device)
        self._prev_actions = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device)
        self._stuck_counter = torch.zeros(self.num_envs, device=self.device)
        self._stuck_triggered = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.progress_ema = torch.zeros(self.num_envs, device=self.device)

        # ---- 观测缓存 ----
        self._single_obs_dim = self.cfg.costmap_channels * self.cfg.costmap_resolution ** 2 + self.cfg.robot_state_dim
        self.obs_buffer = torch.zeros(
            self.num_envs, 1, self._single_obs_dim, device=self.device
        )

        # ---- 终止统计 ----
        self._term_reasons = {
            "timeout": 0, "fell": 0, "drifted": 0,
            "obstacle": 0, "goal_reached": 0,
        }

        # ---- DR 状态 buffer (episode 级) ----
        self._dr_mass_scale = torch.ones(self.num_envs, device=self.device)
        self._dr_wheel_r_scale = torch.ones(self.num_envs, device=self.device)
        self._dr_com_offset = torch.zeros(self.num_envs, 2, device=self.device)
        self._action_delay_buf = torch.zeros(
            self.num_envs, self.cfg.domain_rand.action_delay_max_steps + 2,
            self.cfg.action_space, device=self.device,
        )
        self._action_delay_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._dr_action_delay = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        # ---- 诊断 ----
        print(f"[EmmaCostmapNavEnv] num_envs={self.num_envs}, "
              f"obs_flat_dim={self._single_obs_dim}, "
              f"costmap=({self.cfg.costmap_channels},{self.cfg.costmap_resolution},{self.cfg.costmap_resolution})")

    # =====================================================================
    # Scene Setup
    # =====================================================================
    def _setup_scene(self):
        """搭建场景: Ground + 障碍物模板 + Emma + GoalMarker + 灯光."""
        cfg = self.cfg

        # Ground
        cfg_ground = sim_utils.GroundPlaneCfg()
        cfg_ground.func("/World/groundPlane", cfg_ground)

        # 障碍物模板 (在 env_0 中创建)
        spawn_nav_obstacles(cfg.nav_world)

        # 环境克隆 (从 env_0 复制到所有 env)
        self.scene.clone_environments(copy_from_source=False)

        # Emma Robot
        self.robot = Articulation(cfg.robot_cfg)
        self.scene.articulations["robot"] = self.robot

        # Goal Marker (绿色小球, kinematic 无碰撞)
        goal_cfg = SphereCfg(
            radius=cfg.goal_marker_radius,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
        )
        goal_cfg.func("/World/envs/env_.*/GoalMarker", goal_cfg,
                      translation=(0.0, 0.0, cfg.goal_marker_radius))
        goal_rigid_cfg = RigidObjectCfg(
            prim_path="/World/envs/env_.*/GoalMarker",
            spawn=None,
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0, 0, -100)),  # 初始隐藏
        )
        self.scene.rigid_objects["GoalMarker"] = RigidObject(goal_rigid_cfg)

        # 注册障碍物 RigidObject (每个 cube/cylinder 独立索引)
        for i in range(cfg.nav_world.num_static_cubes):
            name = f"StaticCube_{i}"
            rigid_cfg = RigidObjectCfg(
                prim_path=f"/World/envs/env_.*/StaticCube_{i}", spawn=None,
            )
            self.scene.rigid_objects[name] = RigidObject(rigid_cfg)

        for j in range(cfg.nav_world.num_static_cylinders):
            name = f"StaticCyl_{j}"
            rigid_cfg = RigidObjectCfg(
                prim_path=f"/World/envs/env_.*/StaticCyl_{j}", spawn=None,
            )
            self.scene.rigid_objects[name] = RigidObject(rigid_cfg)

        for k in range(cfg.nav_world.num_dynamic_cylinders):
            name = f"DynamicCyl_{k}"
            rigid_cfg = RigidObjectCfg(
                prim_path=f"/World/envs/env_.*/DynamicCyl_{k}", spawn=None,
            )
            self.scene.rigid_objects[name] = RigidObject(rigid_cfg)

        # DomeLight
        light_cfg = sim_utils.DomeLightCfg(intensity=1000.0)
        light_cfg.func("/World/DomeLight", light_cfg)

    # =====================================================================
    # Pre-physics Step
    # =====================================================================
    def _pre_physics_step(self, actions: torch.Tensor):
        """动作处理: 延迟 → 噪声 → 运动学 → 关节速度."""
        n = actions.shape[0]
        cfg = self.cfg
        dr = cfg.domain_rand
        ns = self._get_noise_scale()

        # ---- Action delay (ring buffer) ----
        if dr.enable_action_noise and dr.action_delay_max_steps > 0:
            delay_max = max(1, int(dr.action_delay_max_steps * ns))
            self._dr_action_delay[:n] = torch.randint(0, delay_max + 1, (n,), device=self.device)

            idx = self._action_delay_idx[:n].clamp(min=0)
            self._action_delay_buf[torch.arange(n, device=self.device), idx] = actions
            self._action_delay_idx[:n] += 1

            delay = self._dr_action_delay[:n].clamp(min=0)
            read_idx = (self._action_delay_idx[:n] - 1 - delay).clamp(min=0)
            actions_delayed = self._action_delay_buf[
                torch.arange(n, device=self.device), read_idx]
        else:
            actions_delayed = actions

        # ---- Action noise ----
        if dr.enable_action_noise and ns > 0:
            noise = torch.randn_like(actions_delayed) * dr.action_noise_std * ns
            actions_noisy = (actions_delayed + noise).clamp(0.0, 1.0)
        else:
            actions_noisy = actions_delayed

        # ---- 反归一化: [0,1] → [v_lin, v_ang] ----
        v_lin = (actions_noisy[:, 0] * 2.0 - 1.0) * cfg.max_linear_velocity
        v_ang = (actions_noisy[:, 1] * 2.0 - 1.0) * cfg.max_angular_velocity

        # ---- 差速运动学 ----
        half_base = cfg.wheel_base / 2.0
        effective_r = cfg.wheel_radius * self._dr_wheel_r_scale[:n]

        w_left = (v_lin - v_ang * half_base) / effective_r
        w_right = (v_lin + v_ang * half_base) / effective_r

        # 轮速滑移噪声
        if dr.enable_dynamics_noise and dr.wheel_slip_noise_std > 0 and ns > 0:
            slip = torch.randn(n, device=self.device) * dr.wheel_slip_noise_std * ns
            w_left = w_left * (1.0 + slip)
            w_right = w_right * (1.0 + slip)

        # ---- 写关节速度 (DirectRLEnv 要求全量) ----
        joint_vels = torch.zeros(self.num_envs, len(cfg.dof_names), device=self.device)
        joint_vels[:n, self._wheel_left_idx] = w_left.clamp(-cfg.wheel_max_velocity, cfg.wheel_max_velocity)
        joint_vels[:n, self._wheel_right_idx] = w_right.clamp(-cfg.wheel_max_velocity, cfg.wheel_max_velocity)
        self._joint_vel_cmd = joint_vels

        # ---- 动态障碍物推进 (仅在 phase 有活跃动态障碍物时) ----
        if self._active_num_dynamic > 0:
            self._world.step_dynamic(self.physics_dt)
            self._world.write_dynamic_poses_to_sim(
                self.scene, self.num_envs, self.scene.env_origins[:, :2])

    def _apply_action(self):
        """每 physics substep: 将 _joint_vel_cmd 写入 robot 关节."""
        if not hasattr(self, '_joint_vel_cmd') or self._joint_vel_cmd is None:
            return
        self.robot.set_joint_velocity_target(
            self._joint_vel_cmd[:, self._wheel_left_idx:self._wheel_left_idx+1],
            joint_ids=[self._wheel_left_idx],
        )
        self.robot.set_joint_velocity_target(
            self._joint_vel_cmd[:, self._wheel_right_idx:self._wheel_right_idx+1],
            joint_ids=[self._wheel_right_idx],
        )

    # =====================================================================
    # Observations
    # =====================================================================
    def _get_observations(self) -> dict:
        """返回 costmap + robot_state 观测 (含频率门控 + 自车占位).

        Costmap 按真实 LiDAR 频率 (10Hz) 更新, 模拟感知滞后.
        每帧叠加 ego-footprint 填补盲开窗口.
        """
        cfg = self.cfg

        # 频率门控: 仅在更新周期到来时重新生成 costmap
        do_update = (self._costmap_step_counter % self._costmap_update_interval == 0)
        if do_update:
            self._static_costmap = self._generate_static_layer()
            self._cached_dynamic_costmap = self._generate_dynamic_layer()
        self._costmap_step_counter += 1

        # 使用缓存的 costmap (首帧时 _static_costmap 可能为 None, 按需生成)
        static_layer = self._static_costmap
        if static_layer is None:
            static_layer = self._generate_static_layer()
            self._static_costmap = static_layer
        dyn_layer = self._cached_dynamic_costmap
        if dyn_layer is None:
            dyn_layer = torch.zeros(
                self.num_envs, 1, cfg.costmap_resolution, cfg.costmap_resolution,
                device=self.device)

        # 叠加自车占位 (ego-footprint): 防止盲开期间冲入未观测区域
        if cfg.costmap_ego_footprint:
            forward_speed = torch.sum(
                self.robot.data.root_com_lin_vel_w[:, :2] *
                torch.nn.functional.normalize(
                    math_utils.quat_apply(
                        self.robot.data.root_quat_w,
                        self.robot.data.FORWARD_VEC_B,
                    )[:, :2], dim=-1),
                dim=-1,
            )  # (N,) body-frame 前进速度
            static_layer = add_ego_footprint(
                static_layer,
                robot_half_length=cfg.robot_half_length,
                robot_half_width=cfg.robot_half_width,
                forward_speed=forward_speed,
                latency=self._costmap_latency,
                map_res=cfg.costmap_resolution,
                grid_size=cfg.costmap_grid_size,
            )

        costmap = torch.cat([static_layer, dyn_layer], dim=1)  # (N, 2, H, W)

        # Robot state (body frame)
        robot_state = self._compute_robot_state()               # (N, 8)

        # 展平为 1D 观测 (网络内部按 _costmap_flat_dim 拆分)
        costmap_flat = costmap.reshape(self.num_envs, -1)      # (N, 2*H*W)
        obs_flat = torch.cat([costmap_flat, robot_state], dim=-1)  # (N, 8200)
        return {"policy": obs_flat}

    def _generate_static_layer(self) -> torch.Tensor:
        """生成静态障碍物 costmap 层 (仅在 _reset_idx 中调用一次)."""
        env_origins = self.scene.env_origins
        origins_xy = env_origins[:, :2]

        cube_w, cyl_w = self._world.get_static_world_centers(origins_xy)
        # 合并圆柱 + 静态圆柱 (如果有的话)
        circle_centers_list = []
        circle_radii_list = []
        if cyl_w is not None:
            circle_centers_list.append(cyl_w)
            circle_radii_list.append(self._world.static_cyl_radii)

        circle_w = (torch.cat(circle_centers_list, dim=1) if len(circle_centers_list) > 0
                     else torch.zeros(self.num_envs, 0, 2, device=self.device))
        circle_r = (torch.cat(circle_radii_list, dim=0) if len(circle_radii_list) > 0
                     else torch.zeros(0, device=self.device))

        dr = self.cfg.domain_rand
        dropout = dr.costmap_dropout_prob if dr.enable_costmap_dropout else 0.0

        return generate_static_costmap(
            robot_pos_w=self.robot.data.root_pos_w,
            robot_quat_w=self.robot.data.root_quat_w,
            circle_centers_w=circle_w,
            circle_radii=circle_r,
            rect_centers_w=cube_w,
            rect_half_l=self._world.cube_half_l,
            rect_half_w=self._world.cube_half_w,
            rect_yaws_w=self._world.cube_yaws_w,
            grid_size=self.cfg.costmap_grid_size,
            map_res=self.cfg.costmap_resolution,
            inflation=self.cfg.costmap_inflation,
            dropout_prob=dropout * self._get_noise_scale(),
        )

    def _generate_dynamic_layer(self) -> torch.Tensor:
        """生成动态障碍物 costmap 层 (每步调用, 障碍物在移动)."""
        if self._active_num_dynamic == 0:
            return torch.zeros(self.num_envs, 1, self.cfg.costmap_resolution,
                               self.cfg.costmap_resolution, device=self.device)

        env_origins = self.scene.env_origins
        origins_xy = env_origins[:, :2]
        dyn_w = self._world.get_dynamic_world_centers(origins_xy)

        dr = self.cfg.domain_rand
        dropout = dr.costmap_dropout_prob if dr.enable_costmap_dropout else 0.0

        return generate_dynamic_costmap(
            robot_pos_w=self.robot.data.root_pos_w,
            robot_quat_w=self.robot.data.root_quat_w,
            circle_centers_w=dyn_w,
            circle_radii=self._world.dyn_cyl_radii,
            grid_size=self.cfg.costmap_grid_size,
            map_res=self.cfg.costmap_resolution,
            inflation=self.cfg.costmap_inflation,
            dropout_prob=dropout * self._get_noise_scale(),
        )

    def _compute_robot_state(self) -> torch.Tensor:
        """计算 robot state 观测 (8 维, body frame)."""
        robot = self.robot
        cfg = self.cfg

        # Goal 方向 + 距离
        ICR_to_goal_w = self.goal_pos_xy - robot.data.root_pos_w[:, :2]
        goal_dist = torch.linalg.norm(ICR_to_goal_w, dim=1)
        goal_dir_w = torch.nn.functional.normalize(ICR_to_goal_w, dim=-1)

        forward_xy = math_utils.quat_apply(
            robot.data.root_quat_w, robot.data.FORWARD_VEC_B)[:, :2]
        forward_xy = torch.nn.functional.normalize(forward_xy, dim=-1)

        yaw = torch.atan2(forward_xy[:, 1], forward_xy[:, 0])
        goal_dir_b = _rotate_world_to_body_2d(goal_dir_w, yaw)  # (N, 2)

        goal_dist_norm = (goal_dist / cfg.goal_max_dist).clamp(0.0, 1.0).unsqueeze(-1)

        # Body frame 速度
        vel_w = robot.data.root_com_lin_vel_w[:, :2]
        vel_b = _rotate_world_to_body_2d(vel_w, yaw)  # (N, 2)
        vx_norm = vel_b[:, 0:1] / cfg.lin_vel_scale
        vy_norm = vel_b[:, 1:2] / cfg.lin_vel_scale

        # 角速度
        ang_vel_z = robot.data.root_ang_vel_b[:, 2:3]
        ang_vel_norm = ang_vel_z / cfg.ang_vel_scale
        # 前一步动作
        prev_act = self._prev_actions[:, :2].clone()

        return torch.cat([
            goal_dir_b,         # 2
            goal_dist_norm,     # 1
            vx_norm, vy_norm,   # 2
            ang_vel_norm,       # 1
            prev_act,           # 2
        ], dim=-1)  # (N, 8)

    # =====================================================================
    # Reward
    # =====================================================================
    def _get_rewards(self) -> torch.Tensor:
        phase = self._get_phase_params()
        nav_sum, obs_collision, goal_reached = self._compute_nav_rewards(phase)

        # 终端奖励
        goal_bonus = goal_reached.float() * self.cfg.goal_reach_bonus
        collision_penalty = obs_collision.float() * self.cfg.obstacle_collision_penalty

        rewards = nav_sum + goal_bonus + collision_penalty

        # 把碰撞/到达写入 dones 标记 (在 _get_dones 中检查)
        self._obs_collision = obs_collision
        self._goal_reached = goal_reached

        return rewards

    def _compute_nav_rewards(self, phase: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """基于 costmap 的导航奖励."""
        p = phase["phase"]
        pfx = f"phase{p}"
        cfg = self.cfg
        robot = self.robot

        # ---- 基础量 ----
        ICR_pos_w = robot.data.root_pos_w
        ICR_vel = robot.data.root_com_lin_vel_w[:, :2]

        forwards = math_utils.quat_apply(robot.data.root_quat_w, robot.data.FORWARD_VEC_B)
        forward_xy = torch.nn.functional.normalize(forwards[:, :2], dim=-1)
        ang_vel_z = robot.data.root_ang_vel_b[:, 2]

        speed_mag = torch.norm(ICR_vel, dim=-1)
        forward_speed = torch.sum(ICR_vel * forward_xy, dim=-1)
        vel_max = cfg.max_linear_velocity
        speed_gate = torch.clamp(speed_mag / vel_max, min=0.3, max=3.0)

        # ---- 目标量 ----
        ICR_to_goal_w = self.goal_pos_xy - ICR_pos_w[:, :2]
        goal_dist = torch.linalg.norm(ICR_to_goal_w, dim=1)
        goal_dir_w = torch.nn.functional.normalize(ICR_to_goal_w, dim=-1)
        dot_forward_goal = torch.sum(forward_xy * goal_dir_w, dim=-1)

        progress = self._prev_goal_dist - goal_dist
        self.progress_ema = 0.9 * self.progress_ema + 0.1 * progress

        # ---- 1. r_approach — 靠近目标 ----
        r_approach = progress.clamp(min=0.0) * getattr(cfg, f"{pfx}_nav_approach_scale") * 50

        # ---- 2. r_heading — 朝向目标 ----
        r_heading = dot_forward_goal * forward_speed.clamp(min=-1.0) * getattr(cfg, f"{pfx}_nav_heading_scale")

        # ---- 3. r_speed — 前进速度 ----
        r_speed = forward_speed.clamp(min=0.0) * dot_forward_goal.clamp(min=0.0) * getattr(cfg, f"{pfx}_nav_vel_dir_scale")

        # ---- 4. r_costmap_avoid — 基于 costmap 射线采样的避障 ----
        avoid_scale = getattr(cfg, f"{pfx}_nav_costmap_avoid_scale")
        if avoid_scale > 0:
            r_costmap_avoid, obs_collision = self._costmap_avoidance_reward(
                goal_dir_w, forward_xy, speed_gate, avoid_scale)
        else:
            r_costmap_avoid = torch.zeros_like(goal_dist)
            obs_collision = torch.zeros_like(goal_dist, dtype=torch.bool)

        # ---- 5. 惩罚项 ----
        p_backtrack = -torch.clamp(-progress, min=0.0).pow(2) * getattr(cfg, f"{pfx}_nav_backtrack_scale")
        p_spin = -(ang_vel_z.pow(2)) * getattr(cfg, f"{pfx}_nav_spin_scale")

        under_speed_scale = getattr(cfg, f"{pfx}_nav_under_speed_scale")
        if under_speed_scale > 0:
            speed_ratio = forward_speed / 0.5
            p_under_speed = -torch.relu(0.5 - speed_ratio).pow(2) * under_speed_scale
        else:
            p_under_speed = torch.zeros_like(goal_dist)

        stuck_scale = getattr(cfg, f"{pfx}_nav_stuck_scale")
        if stuck_scale > 0:
            self._stuck_counter += (speed_mag < 0.1).float()
            self._stuck_counter[speed_mag >= 0.1] = 0
            self._stuck_triggered = self._stuck_counter > 40
            p_stuck = -torch.clamp(self._stuck_counter - 40, min=0.0) * stuck_scale
        else:
            p_stuck = torch.zeros_like(goal_dist)

        # 动作平滑
        p_smooth = -torch.sum((self.actions - self._prev_actions).pow(2), dim=-1) * cfg.action_smooth_scale

        # 时间惩罚
        r_time = torch.full_like(goal_dist, cfg.time_penalty)

        nav_sum = (
            r_approach + r_heading + r_speed +
            r_costmap_avoid +
            p_backtrack + p_spin + p_under_speed + p_stuck +
            p_smooth + r_time
        )

        goal_reached = goal_dist < cfg.goal_reach_dist

        return nav_sum, obs_collision, goal_reached

    def _costmap_avoidance_reward(
        self, goal_dir_w, forward_xy, speed_gate, scale
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """在 costmap 上沿 goal 方向扇区采样射线, 计算避障奖励.

        思路: 在 goal 方向 ±45° 内均匀采样 N 条射线, 每条射线在 costmap 上取 max 代价.
              选择最安全 (代价最低) 的射线, 鼓励速度方向与之对齐.

        Returns:
            r_avoid: (N,) 避障奖励.
            collision: (N,) bool 是否过于靠近障碍物.
        """
        cfg = self.cfg
        N = self.num_envs
        n_rays = cfg.nav_costmap_avoid_rays
        fov_half = math.radians(cfg.nav_costmap_avoid_fov_deg / 2.0)

        # 使用缓存的 costmap (避免重复生成), 取 max over channels
        dyn_layer = self._cached_dynamic_costmap
        if dyn_layer is None:
            dyn_layer = torch.zeros(N, 1, H, W, device=self.device)
        costmap = torch.cat([self._static_costmap, dyn_layer], dim=1)  # (N, 2, H, W)
        costmap_max = costmap.max(dim=1, keepdim=True)[0]  # (N, 1, H, W)

        # Goal 方向角 (body frame) — 处理到达目标时 goal_dir 为 NaN 的边界情况
        goal_dir_w_safe = torch.nan_to_num(goal_dir_w, nan=0.0)
        yaw = torch.atan2(forward_xy[:, 1], forward_xy[:, 0])
        goal_dir_b = _rotate_world_to_body_2d(goal_dir_w_safe, yaw)
        goal_angle_b = torch.atan2(goal_dir_b[:, 1], goal_dir_b[:, 0])  # (N,)

        # 采样射线角度: goal_angle ± fov_half 均匀分布
        angles = torch.linspace(-fov_half, fov_half, n_rays, device=self.device)  # (n_rays,)
        ray_angles = goal_angle_b.unsqueeze(1) + angles.unsqueeze(0)  # (N, n_rays)

        # 在 costmap 上沿每条射线采样: 10 个等距 checkpoints
        H = W = cfg.costmap_resolution
        half_grid = cfg.costmap_grid_size / 2.0
        res = cfg.costmap_grid_size / cfg.costmap_resolution
        center = H / 2.0  # Emma 在 costmap 中心

        ray_costs = torch.zeros(N, n_rays, device=self.device)
        n_checks = 10
        for d in range(1, n_checks + 1):
            dist = d * (cfg.costmap_grid_size / 2.0) / n_checks  # 最大采样到 costmap 边缘
            # body frame 偏移
            dx = dist * torch.cos(ray_angles)  # body X
            dy = dist * torch.sin(ray_angles)  # body Y
            # 栅格坐标
            col = (dx / res + center).long().clamp(0, W - 1)
            row = (dy / res + center).long().clamp(0, H - 1)

            # Gather costmap 值 (防御性 clamp 防浮点边界越界)
            row = row.clamp(0, H - 1)
            col = col.clamp(0, W - 1)
            env_idx = torch.arange(N, device=self.device).unsqueeze(1).expand(-1, n_rays)
            check_cost = costmap_max[env_idx, 0, row, col]  # (N, n_rays)
            ray_costs = torch.maximum(ray_costs, check_cost)

        # 每条射线的最小代价 (最安全路径)
        min_cost, best_idx = ray_costs.min(dim=1)  # (N,), (N,)

        # 碰撞检测: 最近距离方向的代价 > 阈值
        closest_cost = ray_costs[
            torch.arange(N, device=self.device),
            ray_costs.argmax(dim=1) if n_rays > 0 else torch.zeros(N, dtype=torch.long, device=self.device)
        ] if n_rays > 0 else torch.zeros(N, device=self.device)
        collision = min_cost > cfg.nav_costmap_avoid_cost_threshold

        # 避障奖励: 路径越安全, 奖励越高
        r_avoid = (1.0 - min_cost) * scale * speed_gate

        return r_avoid, collision

    # =====================================================================
    # Reset
    # =====================================================================
    def _reset_idx(self, env_ids: torch.Tensor):
        """重置指定 env: 布局 → Emma pose → Goal → 静态 costmap → DR."""
        super()._reset_idx(env_ids)

        # 检测 phase 变化 (首次 reset 也触发初始化)
        phase = self._get_phase_params()
        current_phase = phase["phase"]
        is_first = not hasattr(self, '_active_phase')
        if is_first:
            self._active_phase = current_phase - 1  # 强制触发 _on_phase_change
        if current_phase != self._active_phase:
            self._active_phase = current_phase
            # Phase 变化 → 重新采样布局
            self._on_phase_change(current_phase)

        # 采样 DR (episode 级)
        self._sample_domain_rand(env_ids)

        # Emma 初始位姿 (右侧区的随机位置 + 朝左)
        self._spawn_emma(env_ids)

        # Goal 位姿 (左侧区)
        self._spawn_goal(env_ids)

        # 重置跟踪状态
        self._prev_goal_dist[env_ids] = torch.linalg.norm(
            self.goal_pos_xy[env_ids] - self.robot.data.root_pos_w[env_ids, :2], dim=1)
        self._prev_actions[env_ids] = 0.5  # 中性动作
        self._stuck_counter[env_ids] = 0
        self._stuck_triggered[env_ids] = False
        self.progress_ema[env_ids] = 0.0
        self._action_delay_buf[env_ids] = 0.5
        self._action_delay_idx[env_ids] = 0

        # 应用质量 DR (与 emma_obs_nav 完全一致)
        if self.cfg.domain_rand.enable_dynamics_noise:
            try:
                masses = self.robot.root_physx_view.get_masses().clone()
                default_mass = self.robot.data.default_mass[env_ids, 0].clone()
                masses[env_ids, 0] = default_mass * self._dr_mass_scale[env_ids]
                self.robot.root_physx_view.set_masses(masses, env_ids.cpu())
            except Exception:
                pass  # PhysX view may not be available in play mode

        # 静态 costmap (重置时重新生成, 同时清空动态缓存触发首帧更新)
        self._static_costmap = self._generate_static_layer()
        self._cached_dynamic_costmap = None
        self._costmap_step_counter = 0

        # 初始观测填充
        self.obs_buffer[env_ids] = 0.0

    def _spawn_emma(self, env_ids: torch.Tensor):
        """将 Emma 放置在右侧区 (x∈[13.5, 14.38], y∈[-14.38, 14.38]), 朝向 -X (向左)."""
        cfg = self.cfg
        dr = cfg.domain_rand
        ns = self._get_noise_scale()
        n = len(env_ids)

        half = cfg.terrain_sub_size / 2.0  # 15
        zone_x_min = half - cfg.nav_world.side_zone_width  # 13.5
        zone_x_max = half - cfg.nav_world.buffer_width       # 14.5, 但要留 robot 半长
        margin = cfg.robot_half_length + 0.1
        zone_y_min = -half + margin
        zone_y_max = half - margin

        x = zone_x_min + margin + torch.rand(n, device=self.device) * (zone_x_max - zone_x_min - 2 * margin)
        y = zone_y_min + torch.rand(n, device=self.device) * (zone_y_max - zone_y_min)

        # 朝向: 面向 -X (aiming at goal zone), 即 yaw = π
        yaw = torch.full((n,), math.pi, device=self.device)

        # Init noise
        if dr.enable_init_noise and ns > 0:
            x += (torch.rand(n, device=self.device) * 2 - 1) * dr.init_pos_noise_xy * ns
            y += (torch.rand(n, device=self.device) * 2 - 1) * dr.init_pos_noise_xy * ns
            yaw += (torch.rand(n, device=self.device) * 2 - 1) * math.radians(dr.init_yaw_noise_deg) * ns

        # 写入 robot pose
        root_pose = self.robot.data.root_state_w[env_ids].clone()
        root_pose[:, 0] = x + self.scene.env_origins[env_ids, 0]
        root_pose[:, 1] = y + self.scene.env_origins[env_ids, 1]
        root_pose[:, 2] = 0.045
        # Quaternion from yaw
        root_pose[:, 3] = math.cos(yaw[0].item() / 2.0)  # w — 这里每个 env 有独立 yaw
        root_pose[:, 6] = 0.0  # z
        # 使用循环处理不同的 yaw
        for i, eid in enumerate(env_ids):
            half_yaw = yaw[i].item() / 2.0
            root_pose[i, 3] = math.cos(half_yaw)
            root_pose[i, 4] = 0.0
            root_pose[i, 5] = 0.0
            root_pose[i, 6] = math.sin(half_yaw)

        root_pose[:, 7:] = 0.0  # 线速度/角速度为 0
        self.robot.write_root_state_to_sim(root_pose, env_ids=env_ids)

    def _spawn_goal(self, env_ids: torch.Tensor):
        """将 Goal 放置在左侧区 (x∈[-14.38, -13.5], y∈[-14.38, 14.38])."""
        cfg = self.cfg
        n = len(env_ids)

        half = cfg.terrain_sub_size / 2.0
        zone_x_min = -half + cfg.nav_world.buffer_width
        zone_x_max = -half + cfg.nav_world.side_zone_width
        margin = cfg.goal_marker_radius + 0.1
        zone_y_min = -half + margin
        zone_y_max = half - margin

        gx = zone_x_min + margin + torch.rand(n, device=self.device) * (zone_x_max - zone_x_min - 2 * margin)
        gy = zone_y_min + torch.rand(n, device=self.device) * (zone_y_max - zone_y_min)

        self.goal_pos_xy[env_ids, 0] = gx + self.scene.env_origins[env_ids, 0]
        self.goal_pos_xy[env_ids, 1] = gy + self.scene.env_origins[env_ids, 1]

        # 更新 Goal Marker 位姿
        marker = self.scene.rigid_objects["GoalMarker"]
        if marker is not None:
            state = marker.data.default_root_state.clone()
            state_view = state.view(self.num_envs, -1, 13)
            for i, eid in enumerate(env_ids):
                state_view[eid, 0, 0] = self.goal_pos_xy[eid, 0].item()
                state_view[eid, 0, 1] = self.goal_pos_xy[eid, 1].item()
                state_view[eid, 0, 2] = cfg.goal_marker_radius
                state_view[eid, 0, 3] = 1.0
            marker.write_root_state_to_sim(state)

    # =====================================================================
    # Phase & DR
    # =====================================================================
    def _get_phase_params(self) -> dict:
        """返回当前 phase 的参数."""
        cfg = self.cfg
        p = cfg.training_phase
        pfx = f"phase{p}"
        return {
            "phase": p,
            "obs_spacing": getattr(cfg, f"{pfx}_obs_spacing"),
            "num_dynamic": getattr(cfg, f"{pfx}_num_dynamic"),
        }

    def _on_phase_change(self, new_phase: int):
        """Phase 变化时回调: 更新障碍物密度和动态障碍物数量."""
        phase_params = self._get_phase_params()
        self._active_num_dynamic = phase_params["num_dynamic"]

        # 更新障碍物间距 → 重新采样布局
        self.cfg.nav_world.min_obs_spacing = phase_params["obs_spacing"]
        self._world.sample_initial_layout()
        self._world.write_static_poses_to_sim(
            self.scene, self.num_envs, self.scene.env_origins[:, :2])
        if self._active_num_dynamic > 0:
            self._world.write_dynamic_poses_to_sim(
                self.scene, self.num_envs, self.scene.env_origins[:, :2])

        print(f"[EmmaCostmapNavEnv] Phase → {new_phase}, "
              f"spacing={phase_params['obs_spacing']:.2f}, "
              f"num_dynamic={self._active_num_dynamic}")

    def _get_noise_scale(self) -> float:
        """噪声课程: phase 内线性 ramp."""
        cfg = self.cfg
        p = cfg.training_phase
        noise_start = getattr(cfg, f"phase{p}_noise_start")
        noise_end = getattr(cfg, f"phase{p}_noise_end")
        phase_start_step = (p - 1) * cfg.phase_steps
        steps_in_phase = self.common_step_counter - phase_start_step
        progress = min(1.0, max(0.0, steps_in_phase / cfg.phase_steps))
        return noise_start + (noise_end - noise_start) * progress

    def _sample_domain_rand(self, env_ids: torch.Tensor):
        """采样 episode 级 DR 参数."""
        dr = self.cfg.domain_rand
        ns = self._get_noise_scale()
        n = len(env_ids)

        if dr.enable_dynamics_noise:
            m_min, m_max = dr.mass_scale_range
            self._dr_mass_scale[env_ids] = 1.0 + (torch.rand(n, device=self.device) * (m_max - m_min) + m_min - 1.0) * ns
            r_min, r_max = dr.wheel_radius_scale_range
            self._dr_wheel_r_scale[env_ids] = 1.0 + (torch.rand(n, device=self.device) * (r_max - r_min) + r_min - 1.0) * ns
            c_min, c_max = dr.com_offset_xy_range
            self._dr_com_offset[env_ids] = (torch.rand(n, 2, device=self.device) * (c_max - c_min) + c_min) * ns

        if dr.enable_action_noise:
            delay_max = max(1, int(dr.action_delay_max_steps * ns))
            self._dr_action_delay[env_ids] = torch.randint(0, delay_max + 1, (n,), device=self.device)

    # =====================================================================
    # Termination
    # =====================================================================
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """判断终止条件."""
        timeout = self.episode_length_buf >= self.max_episode_length - 1
        fell = self.robot.data.root_pos_w[:, 2] < self.cfg.fall_reset_height

        half = self.cfg.terrain_sub_size / 2.0
        drifted_x = (self.robot.data.root_pos_w[:, 0] - self.scene.env_origins[:, 0]).abs() > half - self.cfg.robot_half_length
        drifted_y = (self.robot.data.root_pos_w[:, 1] - self.scene.env_origins[:, 1]).abs() > half - self.cfg.robot_half_length
        drifted = drifted_x | drifted_y

        obstacle = self._obs_collision if hasattr(self, '_obs_collision') else torch.zeros_like(timeout)
        goal_reached = self._goal_reached if hasattr(self, '_goal_reached') else torch.zeros_like(timeout)

        return timeout | fell | drifted | obstacle | goal_reached, torch.zeros_like(timeout)

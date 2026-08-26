"""Emma AMR 二轮差速避障导航环境 (pure navigation, no insertion).

场景架构:
  /World/groundPlane               ← GroundPlane 平地
  /World/envs/env_.*/Robot         ← Articulation（Emma AMR）
  /World/envs/env_.*/GoalMarker    ← 目标点标记球
  /World/envs/env_.*/Obstacles/    ← Cylinder Mesh RigidObjects

坐标系 (Emma body frame):
  +X = 前进方向, +Y = 左侧, +Z = 上
  不需要 fork-forward 转换, 直接使用 root body frame.

二轮差速驱动:
  Beta u ∈ (0,1)² → action = 2u-1 ∈ (-1,1)²
  action_control_mode="body_twist":     action=(v_lin, v_ang)
  action_control_mode="wheel_velocity": action=(ω_L, ω_R)
  两种模式最终都生成满足物理上限的左右轮速度目标。
"""

from __future__ import annotations

from collections import Counter, deque
import math
import random
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject, RigidObjectCfg
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors.ray_caster import RayCaster
import isaaclab.utils.math as math_utils

from .action_control import (
    actions_to_raw_wheel_velocities,
    differential_drive_kinematic_limits,
    normalize_action_control_mode,
    project_wheel_velocities_to_motion_limits,
    wheel_velocities_to_body_twist,
)
from .async_lidar import (
    advance_sensor_schedule,
    bpearl_planar_ray_angles,
    compute_aoi,
    fill_clearance_outside_forward_fov,
    min_pool_clearance_rays,
    sample_sensor_periods,
    sensor_due_mask,
    sinusoidal_aoi_encoding,
    warp_cached_ranges_and_clearances,
)
from .emma_obs_nav_env_cfg import EmmaObsNavEnvCfg
from .emma_obs_nav_world import ObstacleWorldManager, spawn_obstacles
from .navigation_rewards import (
    action_jitter_penalty,
    bounded_smoothstep_blockage,
    closing_repulse_penalty,
    goal_aware_obstacle_relevance,
    step_idle_penalty,
    step_progress_reward,
    step_safe_forward_reward,
    step_turn_direction_reward,
    vfh_opening_quality,
    vfh_passage_mask,
    vfh_reference_angle,
)
from .safety_control import apply_safety_control


class EmmaObsNavEnv(DirectRLEnv):
    """Emma AMR 纯避障导航环境.

    Random-obstacles 模式用于先学习基础避障与长距离到达。
    Ring 模式保留旧任务名，但场景是四角出生、错位内墙和混合障碍组成的小型迷宫。
    Warehouse 模式保留原有区块感知 spawn 逻辑。
    """

    cfg: EmmaObsNavEnvCfg

    def __init__(self, cfg: EmmaObsNavEnvCfg, render_mode: str | None = None, **kwargs):
        # 必须在 super().__init__() 之前设置 (因为 _setup_scene 在 super 内调用)
        self._world_mode = cfg.world_mode.strip().lower()
        if self._world_mode not in ("random_obstacles", "ring", "warehouse"):
            raise ValueError(
                "world_mode must be 'random_obstacles', 'ring', or 'warehouse'"
            )
        self._warehouse_cfg = cfg.warehouse_world if self._world_mode == "warehouse" else None
        goal_distance_normalization = cfg.goal_distance_normalization.strip().lower()
        if goal_distance_normalization not in ("initial", "lidar_max"):
            raise ValueError(
                "goal_distance_normalization must be 'initial' or 'lidar_max'"
            )
        cfg.goal_distance_normalization = goal_distance_normalization
        self._action_control_mode = normalize_action_control_mode(cfg.action_control_mode)
        kinematic_vmax, kinematic_wmax = differential_drive_kinematic_limits(
            cfg.wheel_max_velocity,
            cfg.wheel_radius,
            cfg.wheel_base,
        )
        if cfg.sync_body_velocity_limits_to_wheels:
            # Hydra 可能只覆盖轮径、轮距或轮速上限；运行时重新计算，保证
            # action 映射、轮速投影和观测归一化始终使用同一套运动学参数。
            cfg.max_linear_velocity = kinematic_vmax
            cfg.max_angular_velocity = kinematic_wmax
        elif (
            cfg.max_linear_velocity > kinematic_vmax + 1.0e-6
            or cfg.max_angular_velocity > kinematic_wmax + 1.0e-6
        ):
            raise ValueError(
                "configured body velocity limit exceeds the differential-drive "
                "wheel-speed envelope"
            )
        # Keep each world's size independent: ring and warehouse can use
        # different arena sizes without invalidating either layout.
        selected_world_cfg = {
            "random_obstacles": cfg.random_obstacle_world,
            "ring": cfg.obs_world,
            "warehouse": cfg.warehouse_world,
        }[self._world_mode]
        cfg.terrain_sub_size = selected_world_cfg.arena_size
        cfg.scene.env_spacing = selected_world_cfg.arena_size
        if self._world_mode == "ring":
            required_safe_passage = (
                2.0 * cfg.robot_half_width
                + 2.0 * cfg.obs_world.passage_side_clearance
            )
            if cfg.obs_world.min_passage_width < required_safe_passage:
                raise ValueError(
                    "ring min_passage_width must include Emma width and "
                    "the configured side-clearance margins"
                )
            robot_half_diagonal = math.hypot(
                cfg.robot_half_length, cfg.robot_half_width
            )
            corner_wall_margin = (
                cfg.obs_world.corridor_width / 2.0
                - cfg.obs_world.spawn_jitter
                - robot_half_diagonal
            )
            if corner_wall_margin <= cfg.obs_world.obstacle_wall_clearance:
                raise ValueError(
                    "ring corner spawn does not leave enough arbitrary-yaw wall clearance"
                )
        elif self._world_mode == "random_obstacles":
            random_world_cfg = cfg.random_obstacle_world
            robot_half_diagonal = math.hypot(
                cfg.robot_half_length, cfg.robot_half_width
            )
            if random_world_cfg.num_obstacles <= 0:
                raise ValueError(
                    "random_obstacles num_obstacles must be positive"
                )
            if random_world_cfg.min_obstacle_spacing < 1.50:
                raise ValueError(
                    "random_obstacles edge spacing must be at least 1.50m"
                )
            if (
                random_world_cfg.emma_spawn_clearance_radius
                < robot_half_diagonal
            ):
                raise ValueError(
                    "random_obstacles Emma spawn clearance must cover the "
                    "robot half diagonal"
                )
            if random_world_cfg.goal_min_distance <= 5.0:
                raise ValueError(
                    "random_obstacles Emma-to-goal distance must be strictly "
                    "greater than 5m"
                )

        # Wheel actuator and body-twist limits are independent deployment
        # constraints. Both are projected after delay/noise without changing
        # path curvature.
        if (
            cfg.wheel_max_velocity <= 0
            or cfg.max_linear_velocity <= 0
            or cfg.max_angular_velocity <= 0
        ):
            raise ValueError("wheel and body velocity limits must be positive")
        for actuator_name in ("wheel_left_actuator", "wheel_right_actuator"):
            actuator_cfg = cfg.robot_cfg.actuators[actuator_name]
            actuator_cfg.velocity_limit = cfg.wheel_max_velocity
            actuator_cfg.velocity_limit_sim = cfg.wheel_max_velocity
        # Keep the RayCaster and all range/clearance normalization on one
        # canonical distance. A Hydra override of lidar_max_distance must not
        # leave the nested sensor configuration at its class-definition value.
        cfg.lidar.max_distance = cfg.lidar_max_distance
        super().__init__(cfg, render_mode, **kwargs)

        safety_state = "enabled" if self.cfg.safety_control.enabled else "DISABLED (ablation)"
        print(f"[Env] num_envs={self.num_envs}, obs_dim={self.cfg.single_obs_dim}"
              f"×{self.cfg.frame_stack}={self.cfg.observation_space}, "
              f"action_control={self._action_control_mode}, safety={safety_state}")
        print(
            f"[Drive] wheel_radius={self.cfg.wheel_radius:.4f}m "
            f"track_width={self.cfg.wheel_base:.4f}m "
            f"wheel_limit={self.cfg.wheel_max_velocity:.3f}rad/s "
            f"body_limits=({self.cfg.max_linear_velocity:.3f}m/s, "
            f"{self.cfg.max_angular_velocity:.3f}rad/s) "
            f"sync={self.cfg.sync_body_velocity_limits_to_wheels}"
        )
        safety_cfg = self.cfg.safety_control
        max_reaction_time = (
            safety_cfg.controller_latency_s
            + safety_cfg.actuator_brake_build_up_s
            + self.cfg.domain_rand.action_delay_max_steps * self.step_dt
        )
        vmax = self.cfg.max_linear_velocity
        umax = safety_cfg.possible_obstacle_speed
        max_aoi = safety_cfg.max_lidar_aoi_s
        max_braking_boundary = (
            safety_cfg.static_margin
            + umax * max_aoi
            + (vmax + umax) * max_reaction_time
            + vmax * vmax / (2.0 * safety_cfg.braking_deceleration)
            + umax * vmax / safety_cfg.braking_deceleration
        )
        max_slow_boundary = (
            safety_cfg.static_margin
            + umax * max_aoi
            + (vmax + umax) * safety_cfg.ttc_slow_s
        )
        print(
            f"[Safety] vmax={vmax:.3f}m/s obstacle_speed={umax:.3f}m/s "
            f"slow_start@vmax={max_slow_boundary:.3f}m "
            f"braking_boundary@vmax={max_braking_boundary:.3f}m"
        )

        # ---- Joint indices (wheel_left=0, wheel_right=1, lift_joint=2 unused) ----
        jn = list(self.cfg.dof_names)
        self.left_ids, _ = self.robot.find_joints(jn[0])   # wheel_left
        self.right_ids, _ = self.robot.find_joints(jn[1])  # wheel_right

        # ---- Obstacle World Manager ----
        if self._world_mode == "warehouse":
            from .emma_obs_nav_warehouse_world import WarehouseWorldManager
            self._world_manager = WarehouseWorldManager(
                num_envs=self.num_envs,
                cfg=self.cfg.warehouse_world,
                device=self.device,
            )
            self._world_manager.setup_cylinders(
                self._warehouse_radii, self._warehouse_heights)
        elif self._world_mode == "random_obstacles":
            from .emma_obs_nav_random_obstacle_world import (
                RandomObstacleWorldManager,
            )

            self._world_manager = RandomObstacleWorldManager(
                num_envs=self.num_envs,
                cfg=self.cfg.random_obstacle_world,
                device=self.device,
            )
        else:
            self._world_manager = ObstacleWorldManager(
                num_envs=self.num_envs,
                cfg=self.cfg.obs_world,
                device=self.device,
            )

        # ---- Phase tracking (detect phase change → resample layouts) ----
        self._active_phase: int = self.cfg.training_phase
        self._warp_mesh_dirty: bool = False
        self._lidar_mesh_ready: bool = False

        # ---- LiDAR ----
        self._lidar = RayCaster(self.cfg.lidar)

        # ---- Observation buffer ----
        self.obs_buffer = torch.zeros(
            self.num_envs, self.cfg.frame_stack, self.cfg.single_obs_dim,
            device=self.device,
        )

        # ---- Goal tracking ----
        self.goal_pos_xy = torch.zeros((self.num_envs, 2), device=self.device)
        self._prev_goal_dist = torch.zeros((self.num_envs,), device=self.device)
        self._init_goal_dist = torch.ones((self.num_envs,), device=self.device)  # 初始距离, 用于归一化
        # ---- 策略原始动作与上一控制步动作 ----
        self._policy_actions = torch.zeros_like(self.actions)
        self._prev_policy_actions = torch.zeros_like(self.actions)
        self._last_forward_speed = torch.zeros(self.num_envs, device=self.device)
        self._last_angular_speed = torch.zeros(self.num_envs, device=self.device)
        self._last_wheel_command = torch.zeros_like(self.actions)
        self._last_wheel_actual = torch.zeros_like(self.actions)
        self._reference_angle_state = torch.zeros(self.num_envs, device=self.device)
        self._vfh_blockage_state = torch.zeros(self.num_envs, device=self.device)
        self._reference_angle_valid = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        # 预先创建 VFH 诊断量，使 reset 后、第一次 step 前也能启用 play 记录器。
        self._last_reference_angle = torch.zeros(self.num_envs, device=self.device)
        self._last_reference_delta = torch.zeros(self.num_envs, device=self.device)
        self._last_vfh_free_fraction = torch.ones(self.num_envs, device=self.device)
        self._last_vfh_goal_histogram = torch.zeros(self.num_envs, device=self.device)
        self._last_vfh_goal_blocked = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._last_vfh_blockage_score = torch.zeros(
            self.num_envs, device=self.device
        )
        self._last_vfh_opening_width = torch.zeros(
            self.num_envs, device=self.device
        )
        self._last_vfh_opening_depth = torch.zeros(
            self.num_envs, device=self.device
        )
        self._last_vfh_passable_fraction = torch.ones(
            self.num_envs, device=self.device
        )
        self._last_vfh_passage_fallback = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._last_vfh_kept_previous = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        # ---- Action mapping diagnostics ----
        self._raw_wheel_vel_cmd = torch.zeros_like(self.actions)
        self._feasible_wheel_vel_cmd = torch.zeros_like(self.actions)
        self._wheel_vel_cmd = torch.zeros_like(self.actions)
        self._wheel_projection_scale = torch.ones(self.num_envs, device=self.device)
        self._requested_body_twist_cmd = torch.zeros_like(self.actions)
        self._projected_body_twist_cmd = torch.zeros_like(self.actions)
        self._safe_body_twist_cmd = torch.zeros_like(self.actions)
        self._safety_curve_scale = torch.ones(self.num_envs, device=self.device)
        self._safety_brake_scale = torch.ones(self.num_envs, device=self.device)
        self._safety_total_scale = torch.ones(self.num_envs, device=self.device)
        self._safety_safe_speed = torch.zeros(self.num_envs, device=self.device)
        self._safety_min_ttc = torch.full(
            (self.num_envs,), float("inf"), device=self.device)
        self._safety_travel_clearance = torch.full(
            (self.num_envs,), self.cfg.lidar_max_distance, device=self.device)
        self._safety_d_now_lower = torch.full(
            (self.num_envs,), self.cfg.lidar_max_distance, device=self.device)
        self._safety_intervention_amount = torch.zeros(
            self.num_envs, device=self.device)
        self._safety_intervened = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device)
        self._safety_emergency_stopped = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device)

        # ---- Termination mask ----
        self._terminated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._term_reason = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        # 最近 2048 个结束回合的终止原因，用统一 extras 日志写入训练 TensorBoard。
        self._termination_reason_window: deque[int] = deque(maxlen=2048)
        self._termination_rates: dict[str, float] = {
            "goal": 0.0,
            "collision": 0.0,
            "drift": 0.0,
            "timeout": 0.0,
            "fell": 0.0,
            "other": 0.0,
        }

        # ---- LiDAR 射线角度预计算 ----
        # Bpearl 先生成 h∈[-FOV/2,+FOV/2)，随后对三维方向整体取负。
        # 因而平面物理方位是 h+180°；360° 单线配置的 raw_range[0]
        # 正好指向 body +X。这里必须使用物理方位，不能直接把 h 当射线角。
        res = self.cfg.lidar.pattern_cfg.horizontal_res                 # deg
        self._lidar_ray_angles = bpearl_planar_ray_angles(
            self.cfg.num_lidar_rays,
            horizontal_fov_deg=self.cfg.lidar.pattern_cfg.horizontal_fov,
            horizontal_resolution_deg=res,
            device=self.device,
        ).unsqueeze(0).repeat(self.num_envs, 1)  # (N, num_rays)

        # 1440 条物理射线从正前方开始，reshape 后每 12 条组成 3° 扇区，
        # 因此标准 120 维顺序是 index 0≈+1.5°，index 60≈-178.5°。
        sector_step = 360.0 / self.cfg.lidar_output_sectors
        _sector_centers_deg = torch.arange(
            self.cfg.lidar_output_sectors, device=self.device
        ) * sector_step + sector_step / 2.0
        _sector_centers_rad = torch.deg2rad(_sector_centers_deg)
        self._lidar_sector_angles = torch.atan2(
            torch.sin(_sector_centers_rad), torch.cos(_sector_centers_rad)
        ).unsqueeze(0).repeat(self.num_envs, 1)  # (N, 120)

        # ================================================================
        # Domain Randomization state buffers
        # ================================================================
        dr = self.cfg.domain_rand
        self._dr_mass_scale = torch.ones(self.num_envs, device=self.device)
        self._dr_wheel_r_scale = torch.ones(self.num_envs, device=self.device)
        self._dr_com_offset = torch.zeros(self.num_envs, 2, device=self.device)

        # Action delay ring buffer
        self._action_delay_buf = torch.zeros(
            self.num_envs, dr.action_delay_max_steps + 2, self.cfg.action_space,
            device=self.device)
        self._action_delay_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._dr_action_delay = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        # Asynchronous LiDAR state. The raw scan cache is immutable between
        # deliveries; policy-time ranges are always warped from this cache.
        sectors = self.cfg.lidar_output_sectors
        queue_size = self.cfg.lidar_pending_queue_size
        if queue_size < 1:
            raise ValueError("lidar_pending_queue_size must be at least 1")
        if self.cfg.num_lidar_rays != sectors * self.cfg.lidar_rays_per_sector:
            raise ValueError("num_lidar_rays must equal lidar_output_sectors * lidar_rays_per_sector")
        if not 0.0 <= self.cfg.lidar_scan_dropout_prob <= 1.0:
            raise ValueError("lidar_scan_dropout_prob must be in [0, 1]")
        if self.cfg.lidar_jitter_max_steps < 0:
            raise ValueError("lidar_jitter_max_steps must be non-negative")
        if not (
            math.isfinite(self.cfg.lidar_observation_forward_half_fov_deg)
            and 0.0 < self.cfg.lidar_observation_forward_half_fov_deg <= 180.0
        ):
            raise ValueError(
                "lidar_observation_forward_half_fov_deg must be in (0, 180]"
            )
        if not (
            math.isfinite(self.cfg.lidar_observation_blind_fill_clearance)
            and 0.0
            <= self.cfg.lidar_observation_blind_fill_clearance
            <= self.cfg.lidar_max_distance
        ):
            raise ValueError(
                "lidar_observation_blind_fill_clearance must be in "
                "[0, lidar_max_distance]"
            )
        if self.cfg.collision_clearance_threshold < 0:
            raise ValueError("collision_clearance_threshold must be non-negative")
        positive_reward_parameters = {
            "nav_progress_scale_per_step": self.cfg.nav_progress_scale_per_step,
            "nav_turn_scale_per_step": self.cfg.nav_turn_scale_per_step,
            "nav_turn_alignment_deadband_deg":
                self.cfg.nav_turn_alignment_deadband_deg,
            "nav_safe_forward_scale_per_step":
                self.cfg.nav_safe_forward_scale_per_step,
            "nav_safe_forward_full_reward_angle_deg":
                self.cfg.nav_safe_forward_full_reward_angle_deg,
            "nav_safe_forward_cutoff_angle_deg":
                self.cfg.nav_safe_forward_cutoff_angle_deg,
            "nav_idle_scale_per_step": self.cfg.nav_idle_scale_per_step,
            "nav_repulse_start_clearance": self.cfg.nav_repulse_start_clearance,
            "nav_repulse_scale_per_step": self.cfg.nav_repulse_scale_per_step,
            "nav_repulse_lateral_weight_floor":
                self.cfg.nav_repulse_lateral_weight_floor,
            "nav_repulse_rotation_weight": self.cfg.nav_repulse_rotation_weight,
            "nav_repulse_motion_floor": self.cfg.nav_repulse_motion_floor,
            "nav_backward_scale_per_step": self.cfg.nav_backward_scale_per_step,
            "action_linear_jitter_scale": self.cfg.action_linear_jitter_scale,
            "action_angular_jitter_scale": self.cfg.action_angular_jitter_scale,
            "nav_vfh_route_lookahead": self.cfg.nav_vfh_route_lookahead,
            "nav_vfh_route_transition": self.cfg.nav_vfh_route_transition,
            "nav_vfh_blocked_threshold": self.cfg.nav_vfh_blocked_threshold,
            "nav_vfh_goal_cost": self.cfg.nav_vfh_goal_cost,
            "nav_vfh_obstacle_cost": self.cfg.nav_vfh_obstacle_cost,
            "nav_vfh_blocked_goal_cost": self.cfg.nav_vfh_blocked_goal_cost,
            "nav_vfh_blocked_obstacle_cost":
                self.cfg.nav_vfh_blocked_obstacle_cost,
            "nav_vfh_continuity_cost": self.cfg.nav_vfh_continuity_cost,
            "nav_vfh_switch_margin": self.cfg.nav_vfh_switch_margin,
            "nav_vfh_max_reference_rate_deg_s":
                self.cfg.nav_vfh_max_reference_rate_deg_s,
            "nav_vfh_valley_half_width_deg":
                self.cfg.nav_vfh_valley_half_width_deg,
            "nav_vfh_valley_width_cost": self.cfg.nav_vfh_valley_width_cost,
            "nav_vfh_valley_depth_cost": self.cfg.nav_vfh_valley_depth_cost,
            "nav_vfh_valley_keep_half_angle_deg":
                self.cfg.nav_vfh_valley_keep_half_angle_deg,
            "nav_vfh_blockage_time_constant_s":
                self.cfg.nav_vfh_blockage_time_constant_s,
            "nav_vfh_min_passage_width":
                self.cfg.nav_vfh_min_passage_width,
            "nav_vfh_passage_lookahead":
                self.cfg.nav_vfh_passage_lookahead,
        }
        for name, value in positive_reward_parameters.items():
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.cfg.nav_repulse_start_clearance <= 0.0:
            raise ValueError("nav_repulse_start_clearance must be positive")
        if not (
            self.cfg.ground_static_friction
            >= self.cfg.ground_dynamic_friction
            >= 0.0
        ):
            raise ValueError(
                "ground friction must satisfy static >= dynamic >= 0"
            )
        if self.cfg.nav_turn_alignment_deadband_deg >= 90.0:
            raise ValueError("nav_turn_alignment_deadband_deg must be less than 90")
        if not (
            self.cfg.nav_safe_forward_full_reward_angle_deg
            < self.cfg.nav_safe_forward_cutoff_angle_deg
            <= 180.0
        ):
            raise ValueError(
                "safe-forward angles must satisfy "
                "0 <= full_reward < cutoff <= 180 degrees"
            )
        if self.cfg.nav_repulse_lateral_weight_floor > 1.0:
            raise ValueError("nav_repulse_lateral_weight_floor must not exceed 1")
        if self.cfg.nav_repulse_motion_floor > 1.0:
            raise ValueError("nav_repulse_motion_floor must not exceed 1")
        if self.cfg.nav_vfh_smoothing_radius_bins < 0:
            raise ValueError("nav_vfh_smoothing_radius_bins must be non-negative")
        if self.cfg.nav_vfh_route_lookahead <= 0.0:
            raise ValueError("nav_vfh_route_lookahead must be positive")
        if not 0.0 < self.cfg.nav_vfh_route_transition < self.cfg.nav_vfh_route_lookahead:
            raise ValueError(
                "nav_vfh_route_transition must be in (0, route_lookahead)"
            )
        if not 0.0 < self.cfg.nav_vfh_valley_half_width_deg < 180.0:
            raise ValueError("nav_vfh_valley_half_width_deg must be in (0, 180)")
        if not 0.0 < self.cfg.nav_vfh_valley_keep_half_angle_deg <= 180.0:
            raise ValueError(
                "nav_vfh_valley_keep_half_angle_deg must be in (0, 180]"
            )
        if self.cfg.nav_vfh_blockage_time_constant_s <= 0.0:
            raise ValueError("nav_vfh_blockage_time_constant_s must be positive")
        if not 0.0 < self.cfg.nav_vfh_blocked_threshold < 1.0:
            raise ValueError("nav_vfh_blocked_threshold must be in (0, 1)")
        if self.cfg.nav_vfh_min_passage_width <= 0.0:
            raise ValueError("nav_vfh_min_passage_width must be positive")
        if self.cfg.nav_vfh_passage_lookahead <= 0.0:
            raise ValueError("nav_vfh_passage_lookahead must be positive")
        self._cached_scan_ranges = torch.full(
            (self.num_envs, sectors), self.cfg.lidar_max_distance, device=self.device)
        self._cached_scan_angles = self._lidar_sector_angles.clone()
        # Shield cache follows the same physical timing but is independent of
        # policy-only sector dropout. It therefore cannot interpret training
        # dropout as free space.
        self._cached_shield_scan_ranges = torch.full(
            (self.num_envs, sectors), self.cfg.lidar_max_distance, device=self.device)
        self._cached_shield_scan_angles = self._lidar_sector_angles.clone()
        self._cached_sensor_pose = torch.zeros(self.num_envs, 3, device=self.device)
        self._cached_timestamp = torch.zeros(self.num_envs, device=self.device)
        self._has_cached_scan = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        self._sensor_period = torch.full((self.num_envs,), self.step_dt, device=self.device)
        self._sensor_latency = torch.zeros(self.num_envs, device=self.device)
        self._next_sensor_timestamp = torch.zeros(self.num_envs, device=self.device)
        self._sensor_schedule_initialized = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device)
        self._last_sensor_generation_mask = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device)
        self._last_sensor_delivery_mask = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device)
        self._lidar_aoi = torch.zeros(self.num_envs, device=self.device)

        self._pending_scan_ranges = torch.full(
            (self.num_envs, queue_size, sectors), self.cfg.lidar_max_distance, device=self.device)
        self._pending_scan_angles = self._lidar_sector_angles.unsqueeze(1).repeat(
            1, queue_size, 1)
        self._pending_shield_scan_ranges = torch.full(
            (self.num_envs, queue_size, sectors), self.cfg.lidar_max_distance,
            device=self.device)
        self._pending_shield_scan_angles = self._lidar_sector_angles.unsqueeze(1).repeat(
            1, queue_size, 1)
        self._pending_sensor_pose = torch.zeros(
            self.num_envs, queue_size, 3, device=self.device)
        self._pending_generation_timestamp = torch.zeros(
            self.num_envs, queue_size, device=self.device)
        self._pending_delivery_timestamp = torch.full(
            (self.num_envs, queue_size), float("inf"), device=self.device)
        self._pending_valid = torch.zeros(
            self.num_envs, queue_size, dtype=torch.bool, device=self.device)
        self._pending_write_index = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device)

        # Derived current-frame buffers consumed by the reward function.
        self._cached_clearance = torch.full(
            (self.num_envs, sectors), self.cfg.lidar_max_distance,
            device=self.device)
        self._cached_dist = torch.full(
            (self.num_envs, sectors), self.cfg.lidar_max_distance,
            device=self.device)
        self._shield_clearance = torch.full(
            (self.num_envs, sectors), self.cfg.lidar_max_distance,
            device=self.device)
        self._shield_aoi = torch.zeros(self.num_envs, device=self.device)
        self._safety_clearance = torch.full(
            (self.num_envs, sectors), self.cfg.lidar_max_distance,
            device=self.device)
        self._collision_mask = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device)
        self._latest_clean_raw_ranges = torch.full(
            (self.num_envs, self.cfg.num_lidar_rays),
            self.cfg.lidar_max_distance,
            device=self.device,
        )
        self._latest_clean_valid = torch.zeros(
            self.num_envs,
            self.cfg.num_lidar_rays,
            dtype=torch.bool,
            device=self.device,
        )
        self._lidar_sample_step = -1
        rect_projection = self._rect_projection().view(
            self.num_envs, sectors, self.cfg.lidar_rays_per_sector)
        self._ray_rect_projection = rect_projection.reshape(
            self.num_envs, self.cfg.num_lidar_rays)
        self._sector_rect_projection = rect_projection[
            :, :, self.cfg.lidar_rays_per_sector // 2]

    # ==================================================================
    # Scene setup
    # ==================================================================
    def _setup_scene(self):
        """构建场景: 平地 + 障碍物 + Emma + 目标标记."""
        # Ground
        sim_utils.spawn_ground_plane("/World/groundPlane", sim_utils.GroundPlaneCfg(
            color=(0.1, 0.1, 0.1),
        ))

        # 给地面加高摩擦力物理材质 (修复轮子打滑)
        from pxr import UsdPhysics, UsdShade, Sdf
        import omni.usd
        stage = omni.usd.get_context().get_stage()
        mat_path = "/World/groundPlane/physxMaterial"
        if not stage.GetPrimAtPath(mat_path):
            UsdShade.Material.Define(stage, mat_path)
        physx_mat = UsdPhysics.MaterialAPI.Apply(stage.GetPrimAtPath(mat_path))
        physx_mat.CreateStaticFrictionAttr().Set(self.cfg.ground_static_friction)
        physx_mat.CreateDynamicFrictionAttr().Set(self.cfg.ground_dynamic_friction)
        physx_mat.CreateRestitutionAttr().Set(0.0)
        # 绑定到 groundPlane
        try:
            UsdPhysics.MaterialAPI.Apply(stage.GetPrimAtPath("/World/groundPlane"))
        except Exception:
            pass  # 可能已有
        mat_binding = UsdShade.MaterialBindingAPI.Apply(
            stage.GetPrimAtPath("/World/groundPlane"))
        mat_binding.GetDirectBindingRel().SetTargets([Sdf.Path(mat_path)])
        print(
            "[Ground] physics material applied: "
            f"friction={self.cfg.ground_static_friction}/"
            f"{self.cfg.ground_dynamic_friction}"
        )

        if self._world_mode == "warehouse":
            # 仓库: 墙壁 + 货架
            from .emma_obs_nav_warehouse_world import spawn_warehouse
            spawn_warehouse(self.num_envs, self._warehouse_cfg)
            # 圆柱模板 (动态数量)
            num_cyl = self.cfg.phase3_num_cylinders
            rng = self._warehouse_cfg
            radii = [random.uniform(rng.cylinder_radius_min, rng.cylinder_radius_max)
                     for _ in range(num_cyl)]
            heights = [random.uniform(rng.cylinder_height_min + 0.4, rng.cylinder_height_max + 0.4)
                       for _ in range(num_cyl)]  # 高出货架 0.4m
            self._warehouse_radii = radii
            self._warehouse_heights = heights
            # spawn cylinder prims inline (亮红色, 便于视觉区分)
            for i in range(num_cyl):
                r, h = radii[i], heights[i]
                cyl_cfg = sim_utils.CylinderCfg(
                    radius=r, height=h,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        disable_gravity=True, kinematic_enabled=True),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(1.0, 0.15, 0.05)),  # 亮红色
                )
                path = f"/World/envs/env_0/Cylinder_{i}"
                cyl_cfg.func(path, cyl_cfg, translation=(0.0, 0.0, h / 2.0))
                from pxr import UsdPhysics
                import omni.usd
                stage = omni.usd.get_context().get_stage()
                prim = stage.GetPrimAtPath(path)
                if prim.IsValid() and not UsdPhysics.CollisionAPI(prim):
                    UsdPhysics.CollisionAPI.Apply(prim)
            obstacle_count = 4 + self._warehouse_cfg.num_shelf_cols * \
                             self._warehouse_cfg.shelves_per_col + num_cyl
        elif self._world_mode == "random_obstacles":
            # 随机障碍课程: 四面外墙 + 每环境独立的圆柱/方柱布局。
            from .emma_obs_nav_random_obstacle_world import (
                spawn_random_obstacle_world,
            )

            spawn_random_obstacle_world(self.cfg.random_obstacle_world)
            obstacle_count = 4 + self.cfg.random_obstacle_world.num_obstacles
        else:
            # 混合圆柱/方块局部迷宫 + 外墙/错位内墙
            spawn_obstacles(self.num_envs, self.cfg.obs_world)
            obstacle_count = self.cfg.obs_world.num_obstacles + 8

        self.robot = Articulation(self.cfg.robot_cfg)

        goal_marker_cfg = RigidObjectCfg(
            prim_path="/World/envs/env_.*/GoalMarker",
            spawn=sim_utils.SphereCfg(
                radius=self.cfg.goal_marker_radius,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0, 0, -100)),
        )
        self.goal_marker = RigidObject(goal_marker_cfg)

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        self.scene.clone_environments(copy_from_source=False)

        self.scene.articulations["robot"] = self.robot

        if self._world_mode == "warehouse":
            # 圆柱 RigidObject 注册
            for i in range(len(self._warehouse_radii)):
                name = f"cylinder_{i}"
                obs_cfg = RigidObjectCfg(
                    prim_path=f"/World/envs/env_.*/Cylinder_{i}",
                    spawn=None,
                )
                self.scene.rigid_objects[name] = RigidObject(obs_cfg)
        elif self._world_mode == "random_obstacles":
            for i in range(self.cfg.random_obstacle_world.num_obstacles):
                obstacle_cfg = RigidObjectCfg(
                    prim_path=f"/World/envs/env_.*/RandomObstacle_{i}",
                    spawn=None,
                )
                self.scene.rigid_objects[f"random_obstacle_{i}"] = (
                    RigidObject(obstacle_cfg)
                )
        elif self._world_mode == "ring":
            for i in range(self.cfg.obs_world.num_obstacles):
                name = f"obstacle_{i}"
                obs_cfg = RigidObjectCfg(
                    prim_path=f"/World/envs/env_.*/Obstacle_{i}",
                    spawn=None,
                )
                self.scene.rigid_objects[name] = RigidObject(obs_cfg)

        print(f"[Scene] built: {self.num_envs} envs, arena={self.cfg.terrain_sub_size}m, "
              f"mode={self._world_mode}, obstacles={obstacle_count}")

        # 单环境 play：俯视检查仓库或局部迷宫的完整布局。
        if self.num_envs == 1:
            try:
                from omni.isaac.core.utils.viewports import set_camera_view
                if self._world_mode == "warehouse":
                    camera_height = max(
                        18.0, self._warehouse_cfg.arena_size * 1.15
                    )
                elif self._world_mode == "random_obstacles":
                    camera_height = max(
                        14.0,
                        self.cfg.random_obstacle_world.arena_size * 1.15,
                    )
                else:
                    camera_height = max(
                        14.0, self.cfg.obs_world.arena_size * 1.15
                    )
                set_camera_view(
                    eye=[0.0, 0.0, camera_height],
                    target=[0.0, 0.0, 0.5],
                    camera_prim_path="/OmniverseKit_Persp",
                )
            except Exception:
                pass

    # ==================================================================
    # Phase parameters
    # ==================================================================
    def _get_phase_params(self) -> dict:
        """根据 common_step_counter 自动推进 phase.

        Phase 1: steps 0 ~ phase_steps
        Phase 2: steps phase_steps ~ 2*phase_steps
        Phase 3: steps 2*phase_steps+ (持续)
        """
        cfg = self.cfg
        step = self.common_step_counter
        if step < cfg.phase_steps:
            phase = 1
        elif step < 2 * cfg.phase_steps:
            phase = 2
        else:
            phase = 3

        # 检测 phase 切换 (用于触发障碍物重采样)
        if phase != self._active_phase:
            self._active_phase = phase
            # 随机障碍布局在训练开始时生成一次。phase 只推进噪声课程，
            # 不重采样布局，避免大规模环境频繁重建 LiDAR mesh。
            if self._world_mode != "random_obstacles":
                self._warp_mesh_dirty = True

        pfx = f"phase{phase}"
        if self._world_mode == "warehouse":
            num_obs = getattr(cfg, f"{pfx}_num_cylinders")
        elif self._world_mode == "random_obstacles":
            num_obs = cfg.random_obstacle_world.num_obstacles
        else:
            num_obs = getattr(cfg, f"{pfx}_num_obstacles")
        return {
            "phase": phase,
            "obs_spacing": (
                cfg.random_obstacle_world.min_obstacle_spacing
                if self._world_mode == "random_obstacles"
                else getattr(cfg, f"{pfx}_obs_spacing")
            ),
            "num_obstacles": num_obs,
        }

    def _get_noise_scale(self) -> float:
        """基于当前 phase 和 phase 内进度, 返回噪声倍率 [0, 1].

        Phase 1: 0.0 → 0.3, Phase 2: 0.4 → 0.6, Phase 3: 0.7 → 1.0
        Play 模式 (num_envs==1): 固定返回 0, 关闭所有 DR.
        """
        if self.num_envs == 1:
            return 0.0  # play mode: no DR

        cfg = self.cfg
        # 与 _get_phase_params 一致的 phase 计算 (不触发副作用)
        step = self.common_step_counter
        if step < cfg.phase_steps:
            p = 1
        elif step < 2 * cfg.phase_steps:
            p = 2
        else:
            p = 3

        noise_start = getattr(cfg, f"phase{p}_noise_start")
        noise_end = getattr(cfg, f"phase{p}_noise_end")
        phase_start_step = (p - 1) * cfg.phase_steps
        steps_in_phase = step - phase_start_step
        progress = min(1.0, max(0.0, steps_in_phase / cfg.phase_steps))
        return noise_start + (noise_end - noise_start) * progress

    def _sample_domain_rand(self, env_ids: torch.Tensor):
        """采样 episode 级 DR 参数 (在 _reset_idx 中调用)."""
        dr = self.cfg.domain_rand
        n = len(env_ids)
        ns = self._get_noise_scale()  # 课程倍率

        if dr.enable_dynamics_noise:
            self._dr_mass_scale[env_ids] = 1.0 + ns * (
                torch.rand(n, device=self.device)
                * (dr.mass_scale_range[1] - dr.mass_scale_range[0])
                + dr.mass_scale_range[0] - 1.0
            )
            self._dr_wheel_r_scale[env_ids] = 1.0 + ns * (
                torch.rand(n, device=self.device)
                * (dr.wheel_radius_scale_range[1] - dr.wheel_radius_scale_range[0])
                + dr.wheel_radius_scale_range[0] - 1.0
            )
            self._dr_com_offset[env_ids, 0] = ns * (
                torch.rand(n, device=self.device)
                * (dr.com_offset_xy_range[1] - dr.com_offset_xy_range[0])
                + dr.com_offset_xy_range[0]
            )
            self._dr_com_offset[env_ids, 1] = ns * (
                torch.rand(n, device=self.device)
                * (dr.com_offset_xy_range[1] - dr.com_offset_xy_range[0])
                + dr.com_offset_xy_range[0]
            )
        else:
            self._dr_mass_scale[env_ids] = 1.0
            self._dr_wheel_r_scale[env_ids] = 1.0
            self._dr_com_offset[env_ids] = 0.0

        if dr.enable_action_noise:
            max_delay = int(dr.action_delay_max_steps * ns)
            self._dr_action_delay[env_ids] = torch.randint(
                0, max_delay + 1, (n,), device=self.device, dtype=torch.long,
            )
            self._action_delay_buf[env_ids] = 0.0
            self._action_delay_idx[env_ids] = 0
        else:
            self._dr_action_delay[env_ids] = 0

    # ==================================================================
    # Observation
    # ==================================================================
    def _get_observations(self) -> dict:
        """返回堆叠后的观测 (N, single_obs_dim * frame_stack)."""
        single_obs = self._compute_single_obs()  # (N, single_obs_dim)
        if self.cfg.frame_stack > 1:
            # 滑动窗口: 丢弃最旧帧, 追加新帧
            self.obs_buffer[:, :-1, :] = self.obs_buffer[:, 1:, :].clone()
            self.obs_buffer[:, -1, :] = single_obs
            return {"policy": self.obs_buffer.reshape(self.num_envs, -1)}
        return {"policy": single_obs}

    # ==================================================================
    # Physics step — 差速驱动 action 转换
    # ==================================================================
    def _pre_physics_step(self, actions: torch.Tensor):
        """Apply delay/noise, map the selected action mode, and command wheels.

        ``body_twist`` maps normalized ``(v, omega)`` using the independent
        deployment limits. ``wheel_velocity`` maps signed actions directly to
        candidate left/right wheel targets. Both modes then use one coupled
        projection for body and actuator limits, preserving path curvature.
        """
        dr = self.cfg.domain_rand
        ns = self._get_noise_scale()
        n = actions.shape[0]   # active env count (may differ from self.num_envs)
        actions = actions.clamp(-1.0, 1.0)
        self._policy_actions[:n] = actions

        # ---- Action delay (ring buffer) ----
        if dr.enable_action_noise and ns > 0:
            buf_size = self._action_delay_buf.shape[1]
            idx = self._action_delay_idx[:n] % buf_size
            self._action_delay_buf[torch.arange(n, device=self.device), idx] = actions
            self._action_delay_idx[:n] += 1

            delay = self._dr_action_delay[:n].clamp(min=0)
            read_idx = (idx - delay) % buf_size
            actions = self._action_delay_buf[
                torch.arange(n, device=self.device), read_idx].clone()

        # ---- Action noise (高斯) ----
        if dr.enable_action_noise and dr.action_noise_std > 0 and ns > 0:
            noise = torch.randn_like(actions) * dr.action_noise_std * ns
            actions = (actions + noise).clamp(-1.0, 1.0)

        self.actions[:n] = actions.clone()

        cfg = self.cfg
        raw_wheels = actions_to_raw_wheel_velocities(
            actions,
            self._action_control_mode,
            cfg.wheel_max_velocity,
            cfg.wheel_radius,
            cfg.wheel_base,
            cfg.max_linear_velocity,
            cfg.max_angular_velocity,
            self._dr_wheel_r_scale[:n],
        )

        # Wheel slip noise (friction proxy)
        if dr.enable_dynamics_noise and dr.wheel_slip_noise_std > 0 and ns > 0:
            slip = torch.randn(n, device=self.device) * dr.wheel_slip_noise_std * ns
            raw_wheels = raw_wheels * (1.0 + slip.unsqueeze(-1))

        feasible_wheels, projection_scale = project_wheel_velocities_to_motion_limits(
            raw_wheels,
            cfg.wheel_max_velocity,
            cfg.wheel_radius,
            cfg.wheel_base,
            cfg.max_linear_velocity,
            cfg.max_angular_velocity,
            self._dr_wheel_r_scale[:n],
        )

        safety_cfg = cfg.safety_control
        reaction_time = (
            safety_cfg.controller_latency_s
            + safety_cfg.actuator_brake_build_up_s
            + self._dr_action_delay[:n].to(dtype=raw_wheels.dtype) * self.step_dt
        )
        wheel_targets, safety_diag = apply_safety_control(
            feasible_wheels,
            self._shield_clearance[:n],
            self._lidar_sector_angles[:n],
            self._shield_aoi[:n],
            wheel_radius=cfg.wheel_radius,
            wheel_base=cfg.wheel_base,
            lateral_accel_limit=safety_cfg.lateral_accel_limit,
            braking_deceleration=safety_cfg.braking_deceleration,
            reaction_time_s=reaction_time,
            static_margin=safety_cfg.static_margin,
            possible_obstacle_speed=safety_cfg.possible_obstacle_speed,
            forward_half_angle_deg=safety_cfg.forward_half_angle_deg,
            rear_half_angle_deg=safety_cfg.rear_half_angle_deg,
            ttc_slow_s=safety_cfg.ttc_slow_s,
            ttc_stop_s=safety_cfg.ttc_stop_s,
            max_lidar_aoi_s=safety_cfg.max_lidar_aoi_s,
            wheel_radius_scale=self._dr_wheel_r_scale[:n],
            has_valid_scan=self._has_cached_scan[:n],
            enabled=safety_cfg.enabled,
        )

        self._raw_wheel_vel_cmd[:n] = raw_wheels
        self._feasible_wheel_vel_cmd[:n] = feasible_wheels
        self._wheel_vel_cmd[:n] = wheel_targets
        self._wheel_projection_scale[:n] = projection_scale
        self._requested_body_twist_cmd[:n] = wheel_velocities_to_body_twist(
            raw_wheels,
            cfg.wheel_radius,
            cfg.wheel_base,
            self._dr_wheel_r_scale[:n],
        )
        self._projected_body_twist_cmd[:n] = wheel_velocities_to_body_twist(
            feasible_wheels,
            cfg.wheel_radius,
            cfg.wheel_base,
            self._dr_wheel_r_scale[:n],
        )
        self._safe_body_twist_cmd[:n] = safety_diag["safe_twist"]
        self._safety_curve_scale[:n] = safety_diag["curve_scale"]
        self._safety_brake_scale[:n] = safety_diag["brake_scale"]
        self._safety_total_scale[:n] = safety_diag["total_scale"]
        self._safety_safe_speed[:n] = safety_diag["safe_speed"]
        self._safety_min_ttc[:n] = safety_diag["min_ttc"]
        self._safety_travel_clearance[:n] = safety_diag["travel_clearance"]
        self._safety_d_now_lower[:n] = safety_diag["d_now_lower"]
        self._safety_intervention_amount[:n] = safety_diag["intervention_amount"]
        self._safety_intervened[:n] = safety_diag["shield_intervened"]
        self._safety_emergency_stopped[:n] = safety_diag["emergency_stopped"]

        # Joint velocity command — output dim always = num_envs for _apply_action
        joint_vels = torch.zeros(self.num_envs, len(self.cfg.dof_names), device=self.device)
        joint_vels[:n, 0] = wheel_targets[:, 0]
        joint_vels[:n, 1] = wheel_targets[:, 1]
        self._joint_vel_cmd = joint_vels

    def _apply_action(self):
        """每 physics substep: 写入 joint velocity targets."""
        if not hasattr(self, '_joint_vel_cmd') or self._joint_vel_cmd is None:
            return
        self.robot.set_joint_velocity_target(
            self._joint_vel_cmd[:, 0:1], joint_ids=self.left_ids,
        )
        self.robot.set_joint_velocity_target(
            self._joint_vel_cmd[:, 1:2], joint_ids=self.right_ids,
        )

    # ==================================================================
    # Observation — 130 dims (120 LiDAR + 2 goal_dir + 1 goal_dist + 3 IMU + 4 time)
    # ==================================================================
    # Layout:
    #   [  0:120] lidar_norm      120    LiDAR clearance 取反 (3° sector, 360° FOV)
    #   [120:122] goal_dir_b         2    ICR→目标方向 (body frame, 归一化)
    #   [122:123] goal_dist_norm     1    goal_dist / init_goal_dist (动态, 可 >1)
    #   [123:125] vel_b              2    ICR速度 (body frame) / max_linear_velocity
    #   [125:126] ang_vel_z_norm     1    Z轴角速度 / max_angular_velocity
    #   [126:134] time_enc           8    实际 AoI 秒数的连续正弦编码
    # ==================================================================
    def _control_time(self) -> torch.Tensor:
        """Return current control time as a per-environment tensor."""
        timestamp = float(self.common_step_counter) * self.step_dt
        return torch.full((self.num_envs,), timestamp, device=self.device)

    def _configure_lidar_schedule(self, env_ids: torch.Tensor, current_time: torch.Tensor) -> None:
        """Sample independent sensor timing for the selected environments."""
        if len(env_ids) == 0:
            return
        cfg = self.cfg
        mode = cfg.lidar_sensor_mode.lower()
        n = len(env_ids)

        if mode == "synchronous":
            periods = torch.full((n,), self.step_dt, device=self.device)
            latency = torch.zeros(n, device=self.device)
            phase = torch.zeros(n, device=self.device)
        elif mode == "fixed":
            if cfg.lidar_fixed_frequency_hz <= 0:
                raise ValueError("lidar_fixed_frequency_hz must be positive")
            periods = torch.full(
                (n,), 1.0 / cfg.lidar_fixed_frequency_hz, device=self.device)
            phase = torch.rand(n, device=self.device) * periods
            latency = self._sample_lidar_latency(n)
        elif mode == "random":
            periods = sample_sensor_periods(n, cfg.lidar_frequency_choices_hz, self.device)
            phase = torch.rand(n, device=self.device) * periods
            latency = self._sample_lidar_latency(n)
        else:
            raise ValueError(
                f"Unknown lidar_sensor_mode={cfg.lidar_sensor_mode!r}; "
                "expected 'synchronous', 'fixed', or 'random'"
            )

        self._sensor_period[env_ids] = periods
        self._sensor_latency[env_ids] = latency
        self._next_sensor_timestamp[env_ids] = current_time[env_ids] + phase
        self._sensor_schedule_initialized[env_ids] = True

        # A reset invalidates both the delivered cache and all in-flight scans.
        self._has_cached_scan[env_ids] = False
        self._cached_scan_ranges[env_ids] = cfg.lidar_max_distance
        self._cached_scan_angles[env_ids] = self._lidar_sector_angles[env_ids]
        self._cached_shield_scan_ranges[env_ids] = cfg.lidar_max_distance
        self._cached_shield_scan_angles[env_ids] = self._lidar_sector_angles[env_ids]
        self._cached_sensor_pose[env_ids] = 0.0
        self._cached_timestamp[env_ids] = current_time[env_ids]
        self._pending_valid[env_ids] = False
        self._pending_delivery_timestamp[env_ids] = float("inf")
        self._pending_write_index[env_ids] = 0

    def _sample_lidar_latency(self, count: int) -> torch.Tensor:
        low, high = self.cfg.lidar_latency_range_s
        if low < 0 or high < low:
            raise ValueError("lidar_latency_range_s must satisfy 0 <= low <= high")
        if high == low:
            return torch.full((count,), low, device=self.device)
        return low + torch.rand(count, device=self.device) * (high - low)

    def _ensure_lidar_ready(self) -> None:
        """Initialize the RayCaster and refresh its obstacle mesh when needed."""

        if not self._lidar.is_initialized and not self._lidar_mesh_ready:
            self._lidar._initialize_impl()
            wp_mesh = self._world_manager.build_warp_mesh(
                self.scene.env_origins[:, :2], self.device,
            )
            mesh_key = self.cfg.lidar.mesh_prim_paths[0]
            self._lidar.meshes[mesh_key] = wp_mesh
            self._lidar_mesh_ready = True
            self._lidar_sample_step = -1

        if self._warp_mesh_dirty:
            wp_mesh = self._world_manager.build_warp_mesh(
                self.scene.env_origins[:, :2], self.device,
            )
            mesh_key = self.cfg.lidar.mesh_prim_paths[0]
            self._lidar.meshes[mesh_key] = wp_mesh
            self._warp_mesh_dirty = False
            self._lidar_sample_step = -1

    def _sample_clean_lidar_ranges(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return one noise-free physical scan per control step."""

        self._ensure_lidar_ready()
        step = int(self.common_step_counter)
        if self._lidar_sample_step == step:
            return self._latest_clean_raw_ranges, self._latest_clean_valid

        cfg = self.cfg
        self._lidar.update(self.step_dt)
        if self._lidar.data.ray_hits_w is None:
            self._latest_clean_raw_ranges.fill_(cfg.lidar_max_distance)
            self._latest_clean_valid.zero_()
        else:
            sensor_pos_w = self._get_lidar_sensor_pos_w()
            raw_ranges = torch.linalg.norm(
                self._lidar.data.ray_hits_w[..., :2] - sensor_pos_w[:, None, :2],
                dim=-1,
            )
            valid = torch.isfinite(raw_ranges) & (raw_ranges < cfg.lidar_max_distance)
            self._latest_clean_raw_ranges = torch.where(
                valid,
                raw_ranges.clamp(min=0.0, max=cfg.lidar_max_distance),
                torch.full_like(raw_ranges, cfg.lidar_max_distance),
            )
            self._latest_clean_valid = valid
        self._lidar_sample_step = step
        return self._latest_clean_raw_ranges, self._latest_clean_valid

    def _update_safety_clearance(self) -> torch.Tensor:
        """Compute current, noise-free clearance for termination and metrics."""

        cfg = self.cfg
        raw_ranges, _ = self._sample_clean_lidar_ranges()
        per_ray_clearance = raw_ranges - self._ray_rect_projection
        self._safety_clearance = per_ray_clearance.view(
            self.num_envs,
            cfg.lidar_output_sectors,
            cfg.lidar_rays_per_sector,
        ).amin(dim=-1)
        self._collision_mask = (
            self._safety_clearance.amin(dim=-1) <= cfg.collision_clearance_threshold
        )
        return self._safety_clearance

    def _capture_lidar_sector_ranges(
        self, noise_scale: float
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Capture policy and shield sectors from the same physical scan.

        Both branches see the same simulated range noise. Policy-only sector
        dropout is applied after the shield copy is made.
        """

        cfg = self.cfg
        dr = cfg.domain_rand
        clean_ranges, valid = self._sample_clean_lidar_ranges()
        raw_ranges = clean_ranges.clone()
        if dr.enable_lidar_noise and noise_scale > 0:
            noisy = raw_ranges + (
                torch.randn_like(raw_ranges) * dr.lidar_noise_std * noise_scale)
            raw_ranges = torch.where(valid, noisy, raw_ranges)
        raw_ranges = raw_ranges.clamp(min=0.0, max=cfg.lidar_max_distance)

        sector_ranges, sector_angles, _ = min_pool_clearance_rays(
            raw_ranges,
            self._lidar_ray_angles,
            self._ray_rect_projection,
            cfg.lidar_output_sectors,
        )
        shield_sector_ranges = sector_ranges.clone()
        shield_sector_angles = sector_angles.clone()
        if dr.enable_lidar_dropout and noise_scale > 0:
            dropout_prob = dr.lidar_dropout_prob * noise_scale
            sector_dropout = torch.rand_like(sector_ranges) < dropout_prob
            sector_ranges = torch.where(
                sector_dropout,
                torch.full_like(sector_ranges, cfg.lidar_max_distance),
                sector_ranges,
            )
            sector_angles = torch.where(
                sector_dropout, self._lidar_sector_angles, sector_angles)
        return (
            sector_ranges,
            sector_angles,
            shield_sector_ranges,
            shield_sector_angles,
        )

    def _enqueue_lidar_scans(
        self,
        env_ids: torch.Tensor,
        ranges: torch.Tensor,
        ray_angles: torch.Tensor,
        shield_ranges: torch.Tensor,
        shield_ray_angles: torch.Tensor,
        sensor_pose: torch.Tensor,
        generation_timestamp: torch.Tensor,
        delivery_timestamp: torch.Tensor,
    ) -> None:
        if len(env_ids) == 0:
            return
        queue_size = self._pending_scan_ranges.shape[1]
        slots = self._pending_write_index[env_ids] % queue_size
        self._pending_scan_ranges[env_ids, slots] = ranges[env_ids].clone()
        self._pending_scan_angles[env_ids, slots] = ray_angles[env_ids].clone()
        self._pending_shield_scan_ranges[env_ids, slots] = shield_ranges[env_ids].clone()
        self._pending_shield_scan_angles[env_ids, slots] = shield_ray_angles[env_ids].clone()
        self._pending_sensor_pose[env_ids, slots] = sensor_pose[env_ids].clone()
        self._pending_generation_timestamp[env_ids, slots] = generation_timestamp[env_ids]
        self._pending_delivery_timestamp[env_ids, slots] = delivery_timestamp[env_ids]
        self._pending_valid[env_ids, slots] = True
        self._pending_write_index[env_ids] += 1

    def _deliver_pending_lidar(self, current_time: torch.Tensor) -> torch.Tensor:
        """Deliver the newest observation whose latency has elapsed."""
        eligible = self._pending_valid & (
            self._pending_delivery_timestamp <= current_time.unsqueeze(1) + 1.0e-7)
        deliver_mask = eligible.any(dim=1)
        if not torch.any(deliver_mask):
            return deliver_mask

        generation = torch.where(
            eligible,
            self._pending_generation_timestamp,
            torch.full_like(self._pending_generation_timestamp, -float("inf")),
        )
        latest_timestamp, latest_slot = generation.max(dim=1)
        env_ids = torch.where(deliver_mask)[0]
        slots = latest_slot[env_ids]
        self._cached_scan_ranges[env_ids] = self._pending_scan_ranges[env_ids, slots].clone()
        self._cached_scan_angles[env_ids] = self._pending_scan_angles[env_ids, slots].clone()
        self._cached_shield_scan_ranges[env_ids] = (
            self._pending_shield_scan_ranges[env_ids, slots].clone()
        )
        self._cached_shield_scan_angles[env_ids] = (
            self._pending_shield_scan_angles[env_ids, slots].clone()
        )
        self._cached_sensor_pose[env_ids] = self._pending_sensor_pose[env_ids, slots].clone()
        self._cached_timestamp[env_ids] = latest_timestamp[env_ids]
        self._has_cached_scan[env_ids] = True

        # All older eligible observations are superseded by the newest one.
        self._pending_valid[eligible] = False
        self._pending_delivery_timestamp[eligible] = float("inf")
        return deliver_mask

    def _get_lidar_sensor_pose_2d(self, yaw: torch.Tensor) -> torch.Tensor:
        sensor_pos_w = self._get_lidar_sensor_pos_w()
        return torch.cat((sensor_pos_w[:, :2], yaw.unsqueeze(-1)), dim=-1)

    def _update_async_lidar(
        self,
        current_sensor_pose: torch.Tensor,
        current_time: torch.Tensor,
        noise_scale: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate, queue, deliver, and warp asynchronous LiDAR observations."""
        cfg = self.cfg
        uninitialized = ~self._sensor_schedule_initialized
        if torch.any(uninitialized):
            self._configure_lidar_schedule(torch.where(uninitialized)[0], current_time)

        scheduled_due = sensor_due_mask(current_time, self._next_sensor_timestamp)
        force_initial = ~self._has_cached_scan
        generate_mask = scheduled_due | force_initial
        self._last_sensor_generation_mask = generate_mask.clone()

        if torch.any(generate_mask):
            (
                captured_ranges,
                captured_angles,
                captured_shield_ranges,
                captured_shield_angles,
            ) = self._capture_lidar_sector_ranges(noise_scale)
            accepted = generate_mask.clone()
            if cfg.lidar_sensor_mode.lower() != "synchronous" and cfg.lidar_scan_dropout_prob > 0:
                scan_dropout = torch.rand(self.num_envs, device=self.device) < cfg.lidar_scan_dropout_prob
                accepted &= ~scan_dropout
            # Reset always receives one immediate observation so the initial
            # policy state is defined; later dropout behaves normally.
            accepted |= force_initial

            delivery_timestamp = current_time + self._sensor_latency
            delivery_timestamp = torch.where(force_initial, current_time, delivery_timestamp)
            self._enqueue_lidar_scans(
                torch.where(accepted)[0],
                captured_ranges,
                captured_angles,
                captured_shield_ranges,
                captured_shield_angles,
                current_sensor_pose,
                current_time,
                delivery_timestamp,
            )

            jitter_steps = torch.zeros(self.num_envs, device=self.device)
            jitter_limit = cfg.lidar_jitter_max_steps
            if cfg.lidar_sensor_mode.lower() != "synchronous" and jitter_limit > 0:
                jitter_steps = torch.randint(
                    -jitter_limit,
                    jitter_limit + 1,
                    (self.num_envs,),
                    device=self.device,
                )
            self._next_sensor_timestamp = advance_sensor_schedule(
                current_time,
                self._next_sensor_timestamp,
                self._sensor_period,
                scheduled_due,
                self.step_dt,
                jitter_steps,
            )

        self._last_sensor_delivery_mask = self._deliver_pending_lidar(current_time)
        target_pose = current_sensor_pose if cfg.lidar_warp_enabled else self._cached_sensor_pose
        current_ranges, current_clearance = warp_cached_ranges_and_clearances(
            self._cached_scan_ranges,
            self._cached_sensor_pose,
            target_pose,
            self._cached_scan_angles,
            cfg.lidar_output_sectors,
            cfg.lidar_max_distance,
            cfg.robot_half_length,
            cfg.robot_half_width,
            cfg.lidar_offset_x,
            cfg.lidar_offset_y,
        )
        current_ranges = torch.where(
            self._has_cached_scan.unsqueeze(1),
            current_ranges,
            torch.full_like(current_ranges, cfg.lidar_max_distance),
        )
        default_clearance = cfg.lidar_max_distance - self._sector_rect_projection
        current_clearance = torch.where(
            self._has_cached_scan.unsqueeze(1),
            current_clearance,
            default_clearance,
        )

        # Warp the independently cached shield scan with exactly the same
        # delivered timestamp and sensor pose as the policy scan.
        _, shield_clearance = warp_cached_ranges_and_clearances(
            self._cached_shield_scan_ranges,
            self._cached_sensor_pose,
            target_pose,
            self._cached_shield_scan_angles,
            cfg.lidar_output_sectors,
            cfg.lidar_max_distance,
            cfg.robot_half_length,
            cfg.robot_half_width,
            cfg.lidar_offset_x,
            cfg.lidar_offset_y,
        )
        self._shield_clearance = torch.where(
            self._has_cached_scan.unsqueeze(1),
            shield_clearance,
            default_clearance,
        )
        aoi = compute_aoi(current_time, self._cached_timestamp)
        self._lidar_aoi = aoi
        self._shield_aoi = aoi
        return current_ranges, aoi, current_clearance

    def _compute_single_obs(self) -> torch.Tensor:
        cfg = self.cfg
        robot = self.robot
        ns = self._get_noise_scale()

        # ---- 方向向量 (body +X = 前进) ----
        forwards = math_utils.quat_apply(robot.data.root_quat_w, robot.data.FORWARD_VEC_B)
        forwards_xy = torch.nn.functional.normalize(forwards[:, :2], dim=-1)
        yaw = torch.atan2(forwards_xy[:, 1], forwards_xy[:, 0])

        # ---- ICR ----
        ICR_pos_w = robot.data.root_pos_w
        ICR_vel = robot.data.root_com_lin_vel_w[:, :2]

        # ---- Goal (world → body frame) ----
        ICR_to_goal_w = self.goal_pos_xy - ICR_pos_w[:, :2]
        goal_dist = torch.linalg.norm(ICR_to_goal_w, dim=1)
        goal_dir_w = torch.nn.functional.normalize(ICR_to_goal_w, dim=-1)
        goal_dir_b = _rotate_world_to_body_2d(goal_dir_w, yaw)
        if cfg.goal_distance_normalization == "lidar_max":
            # MPPI/BC 任务使用固定米制尺度，使教师与学生看到的一维距离
            # 具有唯一含义；超出量程的局部目标仍允许得到大于 1 的值。
            goal_dist_norm = (goal_dist / cfg.lidar_max_distance).unsqueeze(-1)
        else:
            # 原 ObsNav 兼容模式：初始距离为 1.0，绕行时允许大于 1。
            goal_dist_norm = (
                goal_dist / self._init_goal_dist.clamp(min=1.0)
            ).unsqueeze(-1)

        # ---- IMU-like: body-frame velocity + angular velocity ----
        vel_b = _rotate_world_to_body_2d(ICR_vel, yaw) / max(cfg.max_linear_velocity, 0.1)
        ang_vel_z_norm = robot.data.root_ang_vel_b[:, 2:3] / max(cfg.max_angular_velocity, 0.1)

        # ================================================================
        # LiDAR — 1440 射线 @ 0.25° → min-pool → 120 扇区 (每 3°)
        # ================================================================
        self._ensure_lidar_ready()

        current_time = self._control_time()
        current_sensor_pose = self._get_lidar_sensor_pose_2d(yaw)
        lidar_dist_2d, aoi, clearance = self._update_async_lidar(
            current_sensor_pose, current_time, ns)

        # Clearance is aggregated per transformed hit point before sector
        # min-pooling; it is not approximated with the sector-center ray.
        # 只对送进策略的观测模拟真机盲区。碰撞、奖励和安全诊断继续使用
        # 下方缓存的完整 360° clearance，避免人为填充值改变环境终止逻辑。
        observation_clearance = fill_clearance_outside_forward_fov(
            clearance,
            self._lidar_sector_angles,
            cfg.lidar_observation_forward_half_fov_deg,
            cfg.lidar_observation_blind_fill_clearance,
        )
        if cfg.debug_lidar_fov_fill and not getattr(
            self, "_lidar_fov_debug_printed", False
        ):
            self._lidar_fov_debug_printed = True
            sector_width = 2.0 * math.pi / cfg.lidar_output_sectors
            final_angles = (
                torch.arange(
                    cfg.lidar_output_sectors,
                    device=self.device,
                    dtype=clearance.dtype,
                )
                + 0.5
            ) * sector_width
            final_angles = torch.atan2(
                torch.sin(final_angles), torch.cos(final_angles)
            )
            source_angles = self._lidar_sector_angles[0]
            half_fov = math.radians(
                cfg.lidar_observation_forward_half_fov_deg
            )
            used_fill = torch.where(source_angles.abs() > half_fov + 1.0e-7)[0]
            expected_fill = torch.where(final_angles.abs() > half_fov + 1.0e-7)[0]
            expected_policy = torch.where(
                final_angles.abs().unsqueeze(0) <= half_fov + 1.0e-7,
                clearance,
                torch.as_tensor(
                    cfg.lidar_observation_blind_fill_clearance,
                    device=self.device,
                    dtype=clearance.dtype,
                ),
            )
            max_mask_difference = (
                observation_clearance - expected_policy
            ).abs().amax().item()
            print(
                "[PLAY LIDAR FOV DEBUG] used_fill_indices="
                f"{used_fill.cpu().tolist()}",
                flush=True,
            )
            print(
                "[PLAY LIDAR FOV DEBUG] expected_rear_fill_indices="
                f"{expected_fill.cpu().tolist()} "
                f"max_mask_difference={max_mask_difference:.6f}",
                flush=True,
            )
            probe_indices = (0, 22, 23, 36, 37, 60, 82, 83, 96, 97, 119)
            print(
                "[PLAY LIDAR FOV DEBUG] "
                "idx final_deg label_deg raw_m policy_m expected_m",
                flush=True,
            )
            for index in probe_indices:
                print(
                    "[PLAY LIDAR FOV DEBUG] "
                    f"{index:3d} "
                    f"{math.degrees(final_angles[index].item()):9.1f} "
                    f"{math.degrees(source_angles[index].item()):9.1f} "
                    f"{clearance[0, index].item():7.3f} "
                    f"{observation_clearance[0, index].item():8.3f} "
                    f"{expected_policy[0, index].item():10.3f}",
                    flush=True,
                )
        clearance_clip = observation_clearance.clamp(
            min=-1.0, max=cfg.lidar_max_distance
        )
        lidar_norm = (
            cfg.lidar_max_distance - clearance_clip
        ) / (cfg.lidar_max_distance + 1.0)
        self._cached_clearance = clearance
        self._cached_dist = lidar_dist_2d

        # Continuous encoding of actual Age of Information (generation time,
        # including sensor latency and dropped scans), not a cycle index.
        if cfg.lidar_aoi_enabled:
            time_enc = sinusoidal_aoi_encoding(
                aoi, cfg.time_encoding_dim, cfg.lidar_aoi_encoding_scale_s)
        else:
            time_enc = torch.zeros(
                self.num_envs, cfg.time_encoding_dim, device=self.device)

        # ================================================================
        # 组装观测: 120 + 2 + 1 + 2 + 1 + 8 = 134
        # ================================================================
        obs = torch.cat([
            lidar_norm,                #   0:120  (120)
            goal_dir_b,                # 120:122  (2)
            goal_dist_norm,            # 122:123  (1)
            vel_b,                     # 123:125  (2)
            ang_vel_z_norm,            # 125:126  (1)
            time_enc,                  # 126:134  (8)
        ], dim=-1)
        assert obs.shape[-1] == cfg.single_obs_dim, \
            f"obs dim mismatch: {obs.shape[-1]} != {cfg.single_obs_dim}"
        return obs

    # ==================================================================
    # LiDAR sensor world position
    # ==================================================================
    def _get_lidar_sensor_pos_w(self) -> torch.Tensor:
        offset_body = torch.tensor(
            [self.cfg.lidar_offset_x, self.cfg.lidar_offset_y, self.cfg.lidar_height],
            device=self.device,
        )
        offset_w = math_utils.quat_apply(
            self.robot.data.root_quat_w,
            offset_body.unsqueeze(0).repeat(self.num_envs, 1),
        )
        return self.robot.data.root_pos_w + offset_w

    # ==================================================================
    # Rect projection — 矩形占位 (原点=几何中心, +X=前)
    # ==================================================================
    def _rect_projection(self) -> torch.Tensor:
        """计算 LiDAR 射线到车身矩形边缘的距离.

        占位: half_length=0.4115m, half_width=0.266m, 几何中心=原点
        传感器: 位于 (lidar_offset_x, lidar_offset_y) = (0.38, 0) 在 body  XY 平面内
          → 前边仅 0.0315m, 后边 0.7915m (传感器偏前)

        Ray param: sensor + t * (cos θ, sin θ), 求各边的 t
        """
        cos_a = torch.cos(self._lidar_ray_angles)
        sin_a = torch.sin(self._lidar_ray_angles)

        hl = self.cfg.robot_half_length    # 0.4115
        hw = self.cfg.robot_half_width     # 0.266
        sx = self.cfg.lidar_offset_x       # 0.38 (传感器 body x)
        sy = self.cfg.lidar_offset_y       # 0.0

        inf = torch.full_like(cos_a, float('inf'))
        thresh = 1e-6

        # 四边 (传感器不在原点, 需考虑 offset):
        #   Front: x=hl  → sx + t·cosθ = hl  → t = (hl - sx) / cosθ
        #   Rear:  x=-hl → sx + t·cosθ = -hl → t = (-hl - sx) / cosθ = (hl + sx) / (-cosθ)
        #   Left:  y=hw  → sy + t·sinθ = hw  → t = (hw - sy) / sinθ
        #   Right: y=-hw → sy + t·sinθ = -hw → t = (-hw - sy) / sinθ = (hw + sy) / (-sinθ)
        t_front = torch.where(cos_a > thresh, (hl - sx) / cos_a, inf)
        t_rear = torch.where(cos_a < -thresh, (hl + sx) / (-cos_a), inf)
        t_left = torch.where(sin_a > thresh, (hw - sy) / sin_a, inf)
        t_right = torch.where(sin_a < -thresh, (hw + sy) / (-sin_a), inf)

        rect_proj = torch.min(torch.stack([t_front, t_rear, t_left, t_right], dim=-1), dim=-1)[0]
        rect_proj = torch.where(torch.isfinite(rect_proj), rect_proj, torch.zeros_like(rect_proj))
        return rect_proj

    # ==================================================================
    # Reward
    # ==================================================================
    def _get_rewards(self) -> torch.Tensor:
        phase = self._get_phase_params()
        nav_sum, _, _ = self._compute_nav_rewards(phase)

        if hasattr(self, "extras"):
            log = self.extras.setdefault("log", {})
            log["Safety/intervention_rate"] = self._safety_intervened.float().mean()
            log["Safety/emergency_stop_rate"] = (
                self._safety_emergency_stopped.float().mean()
            )
            log["Safety/curve_scale_mean"] = self._safety_curve_scale.mean()
            log["Safety/brake_scale_mean"] = self._safety_brake_scale.mean()
            log["Safety/safe_speed_mean"] = self._safety_safe_speed.mean()
            log["Control/wheel_projection_scale_mean"] = (
                self._wheel_projection_scale.mean()
            )
            log["Control/wheel_projection_rate"] = (
                self._wheel_projection_scale < 0.999
            ).float().mean()
            log["Episode/goal_rate"] = self._termination_rates["goal"]
            log["Episode/success_rate"] = self._termination_rates["goal"]
            log["Episode/collision_rate"] = self._termination_rates["collision"]
            log["Episode/drift_rate"] = self._termination_rates["drift"]
            log["Episode/timeout_rate"] = self._termination_rates["timeout"]
            log["Episode/fell_rate"] = self._termination_rates["fell"]
            log["Episode/other_termination_rate"] = (
                self._termination_rates["other"]
            )
            log["Episode/rate_window_size"] = len(
                self._termination_reason_window
            )
            log["Safety/travel_clearance_mean"] = (
                self._safety_travel_clearance.mean()
            )
            log["Safety/effective_clearance_mean"] = (
                self._safety_d_now_lower.mean()
            )
            finite_ttc = torch.where(
                torch.isfinite(self._safety_min_ttc),
                self._safety_min_ttc,
                torch.full_like(self._safety_min_ttc, self.cfg.safety_control.ttc_slow_s * 2.0),
            )
            log["Safety/min_ttc_mean"] = finite_ttc.mean()
            if hasattr(self, "_last_approach_reward"):
                log["Reward/progress_mean"] = self._last_approach_reward.mean()
                log["Reward/turn_direction_mean"] = self._last_turn_reward.mean()
                log["Reward/safe_forward_mean"] = (
                    self._last_safe_forward_reward.mean()
                )
                log["Reward/idle_penalty_mean"] = (
                    self._last_idle_penalty.mean()
                )
                log["Reward/repulse_mean"] = self._last_repulse_reward.mean()
                log["Reward/action_jitter_penalty_mean"] = (
                    self._last_jitter_penalty.mean()
                )
                log["Reward/goal_bonus_mean"] = self._last_goal_bonus.mean()
                log["Reward/collision_penalty_mean"] = (
                    self._last_collision_penalty.mean()
                )
                log["Reward/backward_penalty_mean"] = (
                    self._last_backward_penalty.mean()
                )
                log["Navigation/reference_angle_abs_mean"] = (
                    self._last_reference_angle.abs().mean()
                )
                log["Navigation/reference_delta_abs_mean"] = (
                    self._last_reference_delta.abs().mean()
                )
                turn_required = self._last_reference_angle.abs() > math.radians(
                    self.cfg.nav_turn_alignment_deadband_deg
                )
                log["Navigation/turn_direction_match_rate"] = (
                    (
                        (self._last_turn_direction_score > 0.0)
                        & turn_required
                    ).float().sum()
                    / turn_required.float().sum().clamp_min(1.0)
                )
                log["Navigation/task_activity_mean"] = (
                    self._last_task_activity.mean()
                )
                log["Navigation/vfh_free_fraction_mean"] = (
                    self._last_vfh_free_fraction.mean()
                )
                log["Navigation/vfh_goal_histogram_mean"] = (
                    self._last_vfh_goal_histogram.mean()
                )
                log["Navigation/vfh_goal_blocked_rate"] = (
                    self._last_vfh_goal_blocked.float().mean()
                )
                log["Navigation/vfh_blockage_score_mean"] = (
                    self._last_vfh_blockage_score.mean()
                )
                log["Navigation/vfh_selected_width_quality_mean"] = (
                    self._last_vfh_opening_width.mean()
                )
                log["Navigation/vfh_selected_depth_quality_mean"] = (
                    self._last_vfh_opening_depth.mean()
                )
                log["Navigation/vfh_passable_fraction_mean"] = (
                    self._last_vfh_passable_fraction.mean()
                )
                log["Navigation/vfh_passage_fallback_rate"] = (
                    self._last_vfh_passage_fallback.float().mean()
                )
                log["Navigation/vfh_keep_rate"] = (
                    self._last_vfh_kept_previous.float().mean()
                )
                log["Navigation/goal_distance_mean"] = (
                    self._last_goal_distance.mean()
                )
                log["Navigation/closing_risk_mean"] = self._last_closing_risk.mean()
                log["Motion/action_jitter_mean"] = self._last_action_jitter.mean()
                log["Motion/linear_action_jitter_mean"] = (
                    self._last_linear_action_jitter.mean()
                )
                log["Motion/angular_action_jitter_mean"] = (
                    self._last_angular_action_jitter.mean()
                )
                log["Motion/forward_speed_mean"] = (
                    self._last_forward_speed.mean()
                )
                log["Motion/reverse_rate"] = (
                    self._last_forward_speed < -0.02
                ).float().mean()
                log["Motion/abs_angular_speed_mean"] = (
                    self._last_abs_angular_speed.mean()
                )

        # Update tracking
        ICR_pos_w = self.robot.data.root_pos_w
        self._prev_goal_dist = torch.linalg.norm(
            self.goal_pos_xy - ICR_pos_w[:, :2], dim=1,
        )

        return nav_sum.unsqueeze(-1)

    # ==================================================================
    # Navigation reward
    # ==================================================================
    def _compute_nav_rewards(self, phase: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """计算简化后的局部导航奖励。

        Goal 与 LiDAR 共同生成唯一参考航向。角速度只因缩小参考航向误差
        获得奖励，前向速度不再参与航向奖励，因此原地保持对齐无法刷分。
        """
        del phase  # 奖励尺度不再随课程阶段变化，阶段只控制场景难度。
        cfg = self.cfg
        robot = self.robot
        step_dt = self.step_dt

        # ---- 车体运动与 Goal ----
        position_w = robot.data.root_pos_w
        velocity_w = robot.data.root_com_lin_vel_w[:, :2]
        forward_w = math_utils.quat_apply(
            robot.data.root_quat_w, robot.data.FORWARD_VEC_B
        )
        forward_w = torch.nn.functional.normalize(forward_w[:, :2], dim=-1)
        forward_speed = torch.sum(velocity_w * forward_w, dim=-1)
        angular_velocity = robot.data.root_ang_vel_b[:, 2]

        to_goal_w = self.goal_pos_xy - position_w[:, :2]
        goal_distance = torch.linalg.norm(to_goal_w, dim=-1)
        goal_direction_w = torch.nn.functional.normalize(to_goal_w, dim=-1)
        yaw = torch.atan2(forward_w[:, 1], forward_w[:, 0])
        goal_direction_b = _rotate_world_to_body_2d(goal_direction_w, yaw)
        goal_angle_b = torch.atan2(
            goal_direction_b[:, 1], goal_direction_b[:, 0]
        )

        # ---- 弱 Goal 辅助：按本步最大可行位移归一化，远离不扣分 ----
        r_progress = step_progress_reward(
            self._prev_goal_dist,
            goal_distance,
            full_progress_distance=cfg.max_linear_velocity * step_dt,
            scale_per_step=cfg.nav_progress_scale_per_step,
        )

        # ---- VFH 参考航向：只选择稳定自由通道，不直接修改策略 action ----
        clearance = self._cached_clearance
        sector_angles = self._lidar_sector_angles
        route_blockage = bounded_smoothstep_blockage(
            clearance,
            blocked_distance=(
                cfg.nav_vfh_route_lookahead - cfg.nav_vfh_route_transition
            ),
            free_distance=cfg.nav_vfh_route_lookahead,
        )
        obstacle_relevance = goal_aware_obstacle_relevance(
            clearance,
            sector_angles,
            goal_angle_b,
            goal_distance,
            goal_reach_dist=cfg.goal_reach_dist,
            collision_clearance_threshold=cfg.collision_clearance_threshold,
        )
        obstacle_density = route_blockage * obstacle_relevance
        # VFH+ 通道可行性：近场命中点按车宽和侧向安全余量做距离相关
        # 角膨胀。前视距离不越过 Goal，避免目标后方墙体屏蔽到达方向。
        passage_horizon = torch.minimum(
            goal_distance,
            torch.full_like(goal_distance, cfg.nav_vfh_passage_lookahead),
        )
        passage_mask = vfh_passage_mask(
            self._cached_dist,
            passage_horizon,
            min_passage_width=cfg.nav_vfh_min_passage_width,
        )
        sector_width_deg = 360.0 / cfg.lidar_output_sectors
        valley_radius_bins = max(
            1,
            int(round(cfg.nav_vfh_valley_half_width_deg / sector_width_deg)),
        )
        opening_width_quality, opening_depth_quality = vfh_opening_quality(
            clearance,
            passage_mask,
            route_lookahead=cfg.nav_vfh_route_lookahead,
            valley_radius_bins=valley_radius_bins,
        )
        # 上一参考角存储在上一车体系中，先用实际角速度补偿本步车体旋转。
        previous_reference_b = torch.atan2(
            torch.sin(self._reference_angle_state - angular_velocity * step_dt),
            torch.cos(self._reference_angle_state - angular_velocity * step_dt),
        )
        reference_angle, vfh_diag = vfh_reference_angle(
            obstacle_density,
            sector_angles,
            goal_angle_b,
            previous_reference_b,
            self._reference_angle_valid,
            smoothing_radius_bins=cfg.nav_vfh_smoothing_radius_bins,
            blocked_threshold=cfg.nav_vfh_blocked_threshold,
            goal_cost=cfg.nav_vfh_goal_cost,
            obstacle_cost=cfg.nav_vfh_obstacle_cost,
            blocked_goal_cost=cfg.nav_vfh_blocked_goal_cost,
            blocked_obstacle_cost=cfg.nav_vfh_blocked_obstacle_cost,
            continuity_cost=cfg.nav_vfh_continuity_cost,
            switch_margin=cfg.nav_vfh_switch_margin,
            max_reference_delta_rad=math.radians(
                cfg.nav_vfh_max_reference_rate_deg_s
            ) * step_dt,
            passage_mask=passage_mask,
            opening_width_quality=opening_width_quality,
            opening_depth_quality=opening_depth_quality,
            opening_width_cost=cfg.nav_vfh_valley_width_cost,
            opening_depth_cost=cfg.nav_vfh_valley_depth_cost,
            previous_blockage=self._vfh_blockage_state,
            blockage_smoothing_alpha=(
                1.0
                - math.exp(
                    -step_dt / cfg.nav_vfh_blockage_time_constant_s
                )
            ),
            valley_keep_half_angle_rad=math.radians(
                cfg.nav_vfh_valley_keep_half_angle_deg
            ),
        )
        self._reference_angle_state.copy_(reference_angle.detach())
        self._vfh_blockage_state.copy_(vfh_diag["goal_blockage"].detach())
        self._reference_angle_valid.fill_(True)

        # 直接按逐步角速度奖励正确转向，避免低速物理积分信号过小。
        r_turn, turn_direction_score = step_turn_direction_reward(
            reference_angle,
            angular_velocity,
            max_angular_velocity=cfg.max_angular_velocity,
            scale_per_step=cfg.nav_turn_scale_per_step,
            deadband_rad=math.radians(cfg.nav_turn_alignment_deadband_deg),
        )

        # 只在有限 VFH 对齐窗口内奖励前进；截止角外严格不提供该项收益。
        r_safe_forward = step_safe_forward_reward(
            forward_speed,
            reference_angle,
            max_forward_speed=cfg.max_linear_velocity,
            scale_per_step=cfg.nav_safe_forward_scale_per_step,
            full_reward_angle_rad=math.radians(
                cfg.nav_safe_forward_full_reward_angle_deg
            ),
            cutoff_angle_rad=math.radians(
                cfg.nav_safe_forward_cutoff_angle_deg
            ),
        )

        # 正确转向或安全前进都能免除停滞惩罚，原地保持不再是低成本策略。
        p_idle, task_activity = step_idle_penalty(
            forward_speed,
            angular_velocity,
            reference_angle,
            max_forward_speed=cfg.max_linear_velocity,
            max_angular_velocity=cfg.max_angular_velocity,
            scale_per_step=cfg.nav_idle_scale_per_step,
            turn_deadband_rad=math.radians(
                cfg.nav_turn_alignment_deadband_deg
            ),
            forward_full_reward_angle_rad=math.radians(
                cfg.nav_safe_forward_full_reward_angle_deg
            ),
            forward_cutoff_angle_rad=math.radians(
                cfg.nav_safe_forward_cutoff_angle_deg
            ),
        )

        # 斥力覆盖前后近障、侧向贴障和转向扫尾；Goal 后方障碍仍由 relevance 降权。
        p_repulse, closing_risk = closing_repulse_penalty(
            clearance,
            sector_angles,
            forward_speed,
            angular_velocity,
            start_clearance=cfg.nav_repulse_start_clearance,
            max_speed=cfg.max_linear_velocity,
            scale_per_step=cfg.nav_repulse_scale_per_step,
            robot_half_length=cfg.robot_half_length,
            robot_half_width=cfg.robot_half_width,
            lateral_weight_floor=cfg.nav_repulse_lateral_weight_floor,
            rotation_weight=cfg.nav_repulse_rotation_weight,
            motion_floor=cfg.nav_repulse_motion_floor,
            obstacle_relevance=obstacle_relevance,
        )

        # 倒车按相对最大速度逐步惩罚，短距离脱困仍可行。
        p_backward = (
            -cfg.nav_backward_scale_per_step
            * (torch.relu(-forward_speed) / cfg.max_linear_velocity).clamp(max=1.0)
        )

        # 对策略原始 action 做一阶差分，避免把域随机化噪声归因给策略。
        action_delta_sq = (
            self._policy_actions - self._prev_policy_actions
        ).square()
        p_jitter, action_jitter = action_jitter_penalty(
            self._policy_actions,
            self._prev_policy_actions,
            linear_scale=cfg.action_linear_jitter_scale,
            angular_scale=cfg.action_angular_jitter_scale,
        )
        self._prev_policy_actions = self._policy_actions.detach().clone()

        # ---- 终局与时间 ----
        collision = self._collision_mask
        goal_reached = (goal_distance < cfg.goal_reach_dist) & ~collision
        r_goal = goal_reached.float() * cfg.goal_reach_bonus
        p_collision = collision.float() * cfg.obstacle_collision_penalty
        p_time = torch.full_like(goal_distance, cfg.time_penalty)

        # ---- 训练诊断 ----
        self._last_approach_reward = r_progress.detach()
        self._last_goal_distance = goal_distance.detach()
        self._last_turn_reward = r_turn.detach()
        self._last_reference_angle = reference_angle.detach()
        self._last_reference_delta = vfh_diag["reference_delta"].detach()
        self._last_vfh_free_fraction = (
            vfh_diag["free_mask"].float().mean(dim=-1).detach()
        )
        self._last_vfh_goal_histogram = vfh_diag["goal_histogram"].detach()
        self._last_vfh_goal_blocked = vfh_diag["goal_blocked"].detach()
        self._last_vfh_blockage_score = vfh_diag["goal_blockage"].detach()
        self._last_vfh_opening_width = (
            vfh_diag["selected_width_quality"].detach()
        )
        self._last_vfh_opening_depth = (
            vfh_diag["selected_depth_quality"].detach()
        )
        self._last_vfh_passable_fraction = (
            vfh_diag["passable_mask"].float().mean(dim=-1).detach()
        )
        self._last_vfh_passage_fallback = (
            vfh_diag["passage_fallback"].detach()
        )
        self._last_vfh_kept_previous = vfh_diag["kept_previous"].detach()
        self._last_turn_direction_score = turn_direction_score.detach()
        self._last_safe_forward_reward = r_safe_forward.detach()
        self._last_idle_penalty = p_idle.detach()
        self._last_task_activity = task_activity.detach()
        self._last_repulse_reward = p_repulse.detach()
        self._last_closing_risk = closing_risk.detach()
        self._last_backward_penalty = p_backward.detach()
        self._last_jitter_penalty = p_jitter.detach()
        self._last_action_jitter = action_jitter.detach()
        self._last_linear_action_jitter = action_delta_sq[:, 0].detach()
        self._last_angular_action_jitter = action_delta_sq[:, 1].detach()
        self._last_goal_bonus = r_goal.detach()
        self._last_collision_penalty = p_collision.detach()
        self._last_time_penalty = p_time.detach()
        self._last_forward_speed = forward_speed.detach()
        self._last_angular_speed = angular_velocity.detach()
        self._last_abs_angular_speed = angular_velocity.abs().detach()
        self._last_wheel_command = self._wheel_vel_cmd.detach().clone()
        left_id = int(self.left_ids[0])
        right_id = int(self.right_ids[0])
        self._last_wheel_actual = torch.stack(
            (
                robot.data.joint_vel[:, left_id],
                robot.data.joint_vel[:, right_id],
            ),
            dim=-1,
        ).detach()

        total = (
            r_progress
            + r_turn
            + r_safe_forward
            + p_idle
            + p_repulse
            + p_backward
            + p_jitter
            + r_goal
            + p_collision
            + p_time
        )
        return total, collision, goal_reached

    # ==================================================================
    # Termination
    # ==================================================================
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        robot = self.robot
        cfg = self.cfg

        # DirectRLEnv evaluates dones before rewards and observations. Refresh
        # a clean current scan here so collision termination is neither delayed
        # by one control step nor coupled to noisy policy observations.
        self._update_safety_clearance()

        time_out = self.episode_length_buf >= self.max_episode_length - 1
        root_z = robot.data.root_pos_w[:, 2]
        fell = (root_z < cfg.fall_reset_height) | ~torch.isfinite(root_z)

        local_pos = robot.data.root_pos_w[:, :2] - self.scene.env_origins[:, :2]
        drift_limit = self.cfg.terrain_sub_size / 2.0 - self.cfg.robot_half_length
        drifted = (local_pos.abs() > drift_limit).any(dim=-1)

        goal_dist = torch.linalg.norm(
            self.goal_pos_xy - robot.data.root_pos_w[:, :2], dim=-1)
        collision = self._collision_mask
        goal_reached = (goal_dist < cfg.goal_reach_dist) & ~collision

        terminated = fell | drifted | collision | goal_reached | self._terminated
        truncated = time_out & (~terminated)

        self._term_reason[collision] = 4
        self._term_reason[goal_reached] = 5
        self._term_reason[fell & ~collision & ~goal_reached] = 2
        self._term_reason[drifted & ~collision & ~goal_reached & ~fell] = 3
        self._term_reason[truncated] = 1

        # ---- 诊断 ----
        if not hasattr(self, '_diag_short_ep_count'):
            self._diag_short_ep_count = 0
        short_mask = (self.episode_length_buf <= 2) & (terminated | truncated)
        if short_mask.any() and self._diag_short_ep_count < 5:
            for idx in torch.where(short_mask)[0][:3].tolist():
                r = self._term_reason[idx].item()
                reason_names = {1: "timeout", 2: "fell", 3: "drifted",
                                4: "obstacle", 5: "goal_reached"}
                print(f"[DIAG step≤2] env={idx} reason={reason_names.get(r, '?')} "
                      f"steps={self.episode_length_buf[idx].item()} "
                      f"root_z={robot.data.root_pos_w[idx,2].item():.3f}")
            self._diag_short_ep_count += 1

        # ---- 终局统计 ----
        ending = terminated | truncated
        n_ended = ending.sum().item()
        if n_ended > 0:
            reason_names = {1: "timeout", 2: "fell", 3: "drifted",
                            4: "obstacle", 5: "goal_reached"}
            ended_indices = torch.where(ending)[0].tolist()
            ended_reasons: list[int] = []
            for end_idx in ended_indices:
                r = self._term_reason[end_idx].item()
                ended_reasons.append(int(r))

            self._termination_reason_window.extend(ended_reasons)
            window_counts = Counter(self._termination_reason_window)
            window_total = len(self._termination_reason_window)
            self._termination_rates["goal"] = window_counts[5] / window_total
            self._termination_rates["collision"] = window_counts[4] / window_total
            self._termination_rates["drift"] = window_counts[3] / window_total
            self._termination_rates["timeout"] = window_counts[1] / window_total
            self._termination_rates["fell"] = window_counts[2] / window_total
            known_total = sum(window_counts[code] for code in (1, 2, 3, 4, 5))
            self._termination_rates["other"] = (
                window_total - known_total
            ) / window_total

            if self.num_envs == 1:
                for idx in ended_indices[:3]:
                    r = self._term_reason[idx].item()
                    reason = reason_names.get(r, "unknown")
                    steps = self.episode_length_buf[idx].item()
                    print(f"[EPISODE END] env={idx} reason={reason} steps={steps}")

        self._terminated[terminated | truncated] = False
        self._term_reason[terminated | truncated] = 0
        return terminated, truncated

    # ==================================================================
    # Reset
    # ==================================================================
    def _reset_idx(self, env_ids: torch.Tensor):
        if len(env_ids) == 0:
            return

        super()._reset_idx(env_ids)

        n = len(env_ids)
        self._terminated[env_ids] = False
        self._term_reason[env_ids] = 0

        env_origins_xy = self.scene.env_origins[env_ids, :2]

        # ---- 0. 采样 episode 级 DR 参数 ----
        self._sample_domain_rand(env_ids)

        # ---- 1. 障碍物布局 (phase 自动推进, 切换时 _warp_mesh_dirty 置 True) ----
        phase_params = self._get_phase_params()

        if not hasattr(self, '_obs_layout_initialized'):
            self._obs_layout_initialized = False

        if self._world_mode == "warehouse":
            # 仓库: 初始生成后不再变化，phase 切换时重建
            if not self._obs_layout_initialized or self._warp_mesh_dirty:
                num_cyl = phase_params["num_obstacles"]
                self._world_manager.sample_cylinders(
                    env_ids, num_cylinders=num_cyl, ensure_per_channel=1)
                self._world_manager.write_cylinder_poses_to_sim(
                    self.scene, self.num_envs, self.scene.env_origins[:, :2])
                self._obs_layout_initialized = True
                if self._warp_mesh_dirty:
                    print(f"[Warehouse] phase={phase_params['phase']}: resampled "
                          f"({num_cyl} cylinders × {self.num_envs} envs)")
                else:
                    print(f"[Warehouse] initial: "
                          f"{num_cyl} cylinders × {self.num_envs} envs")
        elif self._world_mode == "random_obstacles":
            # 每个并行环境独立采样一次随机混合障碍，并写入 PhysX。
            if not self._obs_layout_initialized:
                self._world_manager.sample_initial_layout()
                self._world_manager.write_poses_to_sim(
                    self.scene,
                    self.scene.env_origins[:, :2],
                )
                self._obs_layout_initialized = True
                print(
                    f"[RandomObstacles] initial: "
                    f"{self.cfg.random_obstacle_world.num_obstacles} "
                    f"mixed obstacles × "
                    f"{self.num_envs} envs, edge_spacing>="
                    f"{self.cfg.random_obstacle_world.min_obstacle_spacing:.2f}m"
                )
        else:
            # 局部迷宫: phase 切换时重采样随机障碍，否则复用各环境布局
            if not self._obs_layout_initialized or self._warp_mesh_dirty:
                self._world_manager.cfg.min_obs_spacing = phase_params["obs_spacing"]
                self._world_manager.sample_initial_layout(
                    num_obstacles=phase_params["num_obstacles"])
                self._world_manager.write_poses_to_sim(
                    self.scene, self.num_envs, self.scene.env_origins[:, :2])
                self._obs_layout_initialized = True
                if self._warp_mesh_dirty:
                    print(f"[ObsWorld] phase={phase_params['phase']}: resampled "
                          f"(obs={phase_params['num_obstacles']})")
                else:
                    print(f"[ObsWorld] initial: "
                          f"{phase_params['num_obstacles']} mixed obstacles × "
                          f"{self.num_envs} envs")

        # ---- 2. Emma/Goal 位置 ----
        dr = self.cfg.domain_rand

        if self._world_mode == "warehouse":
            # 区块感知 spawn, OBB + 圆柱安全检测
            _wm = self._world_manager
            _cyl_pos = _wm.positions[env_ids] if _wm.positions is not None else None
            _cyl_r = _wm.radii if hasattr(_wm, 'radii') else None
            _num_cyl = phase_params.get("num_obstacles", 0)
            spawn_result = _wm.sample_spawn_pair(
                n_pairs=n,
                emma_hl=self.cfg.robot_half_length,
                emma_hw=self.cfg.robot_half_width,
                rng_seed=self.common_step_counter,
                cyl_positions=_cyl_pos,
                cyl_radii=_cyl_r,
                num_active_cyl=_num_cyl,
                goal_reach_dist=self.cfg.goal_reach_dist,
            )
            emma_x = torch.tensor(spawn_result["emma_x"], device=self.device)
            emma_y = torch.tensor(spawn_result["emma_y"], device=self.device)
            rand_yaw = torch.tensor(spawn_result["emma_yaw"], device=self.device)
            goal_x = torch.tensor(spawn_result["goal_x"], device=self.device)
            goal_y = torch.tensor(spawn_result["goal_y"], device=self.device)
            emma_local = torch.stack([emma_x, emma_y], dim=-1)
            goal_local = torch.stack([goal_x, goal_y], dim=-1)

            if self.num_envs == 1:
                bi = spawn_result["block_emma"][0]
                bj = spawn_result["block_goal"][0]
                print(f"[Spawn] Emma block={bi} ({emma_x[0]:.2f},{emma_y[0]:.2f})"
                      f" yaw={math.degrees(rand_yaw[0].item()):.0f}°"
                      f" | Goal block={bj} ({goal_x[0]:.2f},{goal_y[0]:.2f})")
        else:
            (
                emma_local,
                rand_yaw,
                goal_local,
                emma_corner,
                goal_corner,
            ) = self._world_manager.sample_spawn_pairs(env_ids)
            if self.num_envs == 1:
                if self._world_mode == "random_obstacles":
                    distance = torch.linalg.norm(
                        goal_local[0] - emma_local[0]
                    ).item()
                    print(
                        f"[Spawn] Emma=({emma_local[0, 0]:.2f},"
                        f"{emma_local[0, 1]:.2f}) "
                        f"yaw={math.degrees(rand_yaw[0].item()):.0f}deg"
                        f" | Goal=({goal_local[0, 0]:.2f},"
                        f"{goal_local[0, 1]:.2f}) distance={distance:.2f}m"
                    )
                else:
                    corner_names = (
                        "upper-left",
                        "upper-right",
                        "lower-left",
                        "lower-right",
                    )
                    region_label = "corner"
                    print(
                        f"[Spawn] Emma {region_label}="
                        f"{corner_names[int(emma_corner[0].item())]} "
                        f"({emma_local[0, 0]:.2f},{emma_local[0, 1]:.2f}) "
                        f"yaw={math.degrees(rand_yaw[0].item()):.0f}deg"
                        f" | Goal {region_label}="
                        f"{corner_names[int(goal_corner[0].item())]} "
                        f"({goal_local[0, 0]:.2f},{goal_local[0, 1]:.2f})"
                    )

        robot_world = emma_local + env_origins_xy

        quat = torch.zeros(n, 4, device=self.device)
        quat[:, 0] = torch.cos(rand_yaw / 2.0)
        quat[:, 3] = torch.sin(rand_yaw / 2.0)
        root_pose = torch.cat([robot_world,
                        torch.full((n, 1), 0.045, device=self.device),
                        quat], dim=-1)
        self.robot.write_root_pose_to_sim(root_pose, env_ids=env_ids)
        self.robot.write_root_velocity_to_sim(
            torch.zeros(n, 6, device=self.device), env_ids=env_ids)

        # ---- 2.5 应用动力学 DR (质量) ----
        if dr.enable_dynamics_noise:
            try:
                masses = self.robot.root_physx_view.get_masses().clone()
                default_mass = self.robot.data.default_mass[env_ids, 0].clone()
                masses[env_ids, 0] = default_mass * self._dr_mass_scale[env_ids]
                self.robot.root_physx_view.set_masses(masses, env_ids.cpu())
            except Exception:
                pass  # PhysX view may not be available in play mode

        # ---- 3. 目标点 ----
        self.goal_pos_xy[env_ids] = goal_local + env_origins_xy
        self._update_goal_marker(env_ids)

        # ---- 4. 重置跟踪状态 ----
        ICR_pos_w = self.robot.data.root_pos_w
        init_dist = torch.linalg.norm(
            ICR_pos_w[env_ids, :2] - self.goal_pos_xy[env_ids], dim=1,
        )
        self._init_goal_dist[env_ids] = init_dist
        self._prev_goal_dist[env_ids] = init_dist
        self._policy_actions[env_ids] = 0.0
        self._prev_policy_actions[env_ids] = 0.0
        self._reference_angle_state[env_ids] = 0.0
        self._vfh_blockage_state[env_ids] = 0.0
        self._reference_angle_valid[env_ids] = False
        self._last_reference_angle[env_ids] = 0.0
        self._last_reference_delta[env_ids] = 0.0
        self._last_vfh_free_fraction[env_ids] = 1.0
        self._last_vfh_goal_histogram[env_ids] = 0.0
        self._last_vfh_goal_blocked[env_ids] = False
        self._last_vfh_blockage_score[env_ids] = 0.0
        self._last_vfh_opening_width[env_ids] = 0.0
        self._last_vfh_opening_depth[env_ids] = 0.0
        self._last_vfh_passable_fraction[env_ids] = 1.0
        self._last_vfh_passage_fallback[env_ids] = False
        self._last_vfh_kept_previous[env_ids] = False
        self._raw_wheel_vel_cmd[env_ids] = 0.0
        self._feasible_wheel_vel_cmd[env_ids] = 0.0
        self._wheel_vel_cmd[env_ids] = 0.0
        self._wheel_projection_scale[env_ids] = 1.0
        self._requested_body_twist_cmd[env_ids] = 0.0
        self._projected_body_twist_cmd[env_ids] = 0.0
        self._safe_body_twist_cmd[env_ids] = 0.0
        self._safety_curve_scale[env_ids] = 1.0
        self._safety_brake_scale[env_ids] = 1.0
        self._safety_total_scale[env_ids] = 1.0
        self._safety_safe_speed[env_ids] = 0.0
        self._safety_min_ttc[env_ids] = float("inf")
        self._safety_travel_clearance[env_ids] = self.cfg.lidar_max_distance
        self._safety_d_now_lower[env_ids] = self.cfg.lidar_max_distance
        self._safety_intervention_amount[env_ids] = 0.0
        self._safety_intervened[env_ids] = False
        self._safety_emergency_stopped[env_ids] = False
        self._collision_mask[env_ids] = False
        self._safety_clearance[env_ids] = self.cfg.lidar_max_distance
        self._shield_clearance[env_ids] = self.cfg.lidar_max_distance
        self._shield_aoi[env_ids] = 0.0

        # Reset only the selected environments' sensor schedules and caches.
        # Their first scan is delivered immediately in _compute_single_obs;
        # subsequent scans follow the configured asynchronous timing.
        self._configure_lidar_schedule(env_ids, self._control_time())
        # Robot poses changed after the control-step safety scan; force one
        # fresh RayCaster sample for the reset observations.
        self._lidar_sample_step = -1

        # ---- 5. 填充观测 buffer ----
        single_obs = self._compute_single_obs()[env_ids]
        self._update_safety_clearance()
        for f in range(self.cfg.frame_stack):
            self.obs_buffer[env_ids, f, :] = single_obs

        # ---- 6. 诊断: spawn 后立即检查是否已处于碰撞 ----
        if not hasattr(self, '_spawn_diag_count'):
            self._spawn_diag_count = 0
        if self._spawn_diag_count < 10 and hasattr(self, '_cached_clearance'):
            clearance = self._safety_clearance[env_ids]  # (n, S), clean/current
            min_clr = clearance.min(dim=-1)[0]            # (n,)
            bad = min_clr <= self.cfg.collision_clearance_threshold
            if bad.any():
                for idx in torch.where(bad)[0][:3].tolist():
                    env_i = env_ids[idx].item()
                    local_pos = (self.robot.data.root_pos_w[env_i, :2] -
                                 self.scene.env_origins[env_i, :2])
                    print(f"[SPAWN DIAG] env={env_i} min_clearance={min_clr[idx]:.3f}m "
                          f"pos=({local_pos[0]:.2f},{local_pos[1]:.2f}) "
                          f"z={self.robot.data.root_pos_w[env_i, 2]:.3f} "
                          f"goal_dist={self._prev_goal_dist[env_i]:.2f}")
                self._spawn_diag_count += 1

    def _update_goal_marker(self, env_ids: torch.Tensor) -> None:
        n = len(env_ids)
        goal_world = self.goal_pos_xy[env_ids]  # already world coords

        marker_pose = torch.zeros(n, 7, device=self.device)
        marker_pose[:, 0] = goal_world[:, 0]
        marker_pose[:, 1] = goal_world[:, 1]
        marker_pose[:, 2] = 0.3
        marker_pose[:, 3] = 1.0
        self.goal_marker.write_root_pose_to_sim(marker_pose, env_ids=env_ids)


# ====================================================================
# Utility: 2D world → body rotation
# ====================================================================
def _rotate_world_to_body_2d(points_w: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
    """将世界坐标旋转到 body frame (+X=前进方向).

    Args:
        points_w: (N, 2) 世界坐标
        yaw: (N,) body +X 方向的 yaw angle

    Returns:
        points_b: (N, 2) body frame 坐标
    """
    c = torch.cos(-yaw)
    s = torch.sin(-yaw)
    x_b = points_w[:, 0] * c - points_w[:, 1] * s
    y_b = points_w[:, 0] * s + points_w[:, 1] * c
    return torch.stack([x_b, y_b], dim=-1)

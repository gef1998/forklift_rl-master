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
import os
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
    low_pass_wheel_velocity_command,
    normalize_action_control_mode,
    limit_wheel_velocity_acceleration,
    project_wheel_velocities_to_motion_limits,
    wheel_velocities_to_body_twist,
)
from .async_lidar import (
    advance_sensor_schedule,
    bpearl_planar_ray_angles,
    compute_aoi,
    fill_clearance_outside_forward_fov,
    hybrid_180_sector_edges,
    min_pool_clearance_rays,
    min_pool_clearance_rays_grouped,
    sample_sensor_periods,
    sector_centers_from_edges,
    sensor_due_mask,
    sinusoidal_aoi_encoding,
    warp_cached_ranges_and_clearances,
)
from .emma_obs_nav_corridor_world import CorridorWorldManager, spawn_s_corridor
from .emma_obs_nav_env_cfg import EmmaObsNavEnvCfg
from .collision_curriculum import collision_curriculum_step
from .drive_contacts import configure_drive_contacts
from .emma_obs_nav_straight_obstacle_world import (
    StraightObstacleWorldManager,
    spawn_straight_obstacle_world,
)
from .emma_obs_nav_world import ObstacleWorldManager, spawn_obstacles
from .navigation_rewards import navigation_reward_terms
from .reference_path import project_points_to_polylines, project_continuous_progress
from .se2_reference_planner import PlannerConfig
from .navigation_curriculum import NavigationCurriculum
from .safety_control import apply_safety_control


class EmmaObsNavEnv(DirectRLEnv):
    """Emma AMR 纯避障导航环境.

    Random-obstacles 模式用于先学习基础避障与长距离到达。
    Ring 模式保留旧任务名，但场景是四角出生、错位内墙和混合障碍组成的小型迷宫。
    Warehouse 模式保留原有区块感知 spawn 逻辑。
    """

    cfg: EmmaObsNavEnvCfg

    def __init__(self, cfg: EmmaObsNavEnvCfg, render_mode: str | None = None, **kwargs):
        for name in ("linear_acceleration_command_scale", "angular_acceleration_command_scale"):
            value = getattr(cfg, name)
            if not math.isfinite(value) or not 0.0 < value <= 1.0:
                raise ValueError(f"{name} must be in (0, 1]")
        if cfg.enable_motion_command_low_pass and (
            not math.isfinite(cfg.motion_command_low_pass_tau_s)
            or cfg.motion_command_low_pass_tau_s <= 0.0
        ):
            raise ValueError("motion_command_low_pass_tau_s must be finite and positive")
        if cfg.reference_path_backend not in ("polyline", "hybrid_astar"):
            raise ValueError("reference_path_backend must be polyline or hybrid_astar")
        if cfg.reference_path_visualize or cfg.reference_path_export_png:
            if cfg.reference_path_backend != "hybrid_astar" or not cfg.reference_path_reward_enabled:
                raise ValueError("reference visualization requires enabled hybrid_astar references")
            if not math.isfinite(cfg.reference_path_visualize_height) or cfg.reference_path_visualize_height <= 0:
                raise ValueError("reference_path_visualize_height must be finite and positive")
        if cfg.reference_path_reward_enabled:
            if (
                cfg.reference_path_progress_per_meter < 0.0
                or cfg.reference_path_max_progress_per_step <= 0.0
                or cfg.reference_path_cross_track_tolerance < 0.0
                or cfg.reference_path_cross_track_full_cost_error
                <= cfg.reference_path_cross_track_tolerance
                or cfg.reference_path_cross_track_cost_per_second < 0.0
            ):
                raise ValueError("invalid reference-path reward configuration")
        policy_lidar_layout = cfg.lidar_policy_sector_layout.strip().lower()
        if policy_lidar_layout not in ("uniform_120", "hybrid_180"):
            raise ValueError(
                "lidar_policy_sector_layout must be 'uniform_120' or 'hybrid_180'"
            )
        cfg.lidar_policy_sector_layout = policy_lidar_layout
        if policy_lidar_layout == "uniform_120":
            if cfg.lidar_output_sectors != 120 or cfg.lidar_rays_per_sector != 12:
                raise ValueError("uniform_120 requires 120 sectors with 12 rays each")
        elif cfg.lidar_output_sectors != 180 or cfg.num_lidar_rays != 1440:
            raise ValueError("hybrid_180 requires 180 policy sectors and 1440 rays")
        if (
            cfg.lidar_safety_output_sectors != 120
            or cfg.lidar_safety_rays_per_sector != 12
            or cfg.num_lidar_rays
            != cfg.lidar_safety_output_sectors * cfg.lidar_safety_rays_per_sector
        ):
            raise ValueError("the safety scan must remain uniform 120 x 12 rays")
        # 必须在 super().__init__() 之前设置 (因为 _setup_scene 在 super 内调用)
        self._world_mode = cfg.world_mode.strip().lower()
        if self._world_mode not in (
            "random_obstacles",
            "ring",
            "corridor",
            "straight_obstacle",
            "warehouse",
        ):
            raise ValueError(
                "world_mode must be 'random_obstacles', 'ring', 'corridor', "
                "'straight_obstacle', or 'warehouse'"
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
            "corridor": cfg.corridor_world,
            "straight_obstacle": cfg.straight_obstacle_world,
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
            if random_world_cfg.num_obstacles < 0:
                raise ValueError(
                    "random_obstacles num_obstacles must be positive"
                )
            if random_world_cfg.min_obstacle_spacing < 0.0:
                raise ValueError(
                    "random_obstacles edge spacing must be non-negative"
                )
            if (
                random_world_cfg.emma_spawn_clearance_radius
                < robot_half_diagonal
            ):
                raise ValueError(
                    "random_obstacles Emma spawn clearance must cover the "
                    "robot half diagonal"
                )
            if (
                random_world_cfg.structured_layout
                and random_world_cfg.navigation_clearance
                < robot_half_diagonal + cfg.collision_clearance_threshold
            ):
                raise ValueError(
                    "random_obstacles navigation clearance must cover the robot and collision margin"
                )
            if random_world_cfg.goal_min_distance <= 5.0:
                raise ValueError(
                    "random_obstacles Emma-to-goal distance must be strictly "
                    "greater than 5m"
                )
        elif self._world_mode == "corridor":
            from .corridor_geometry import validate_s_corridor_geometry

            corridor_cfg = cfg.corridor_world
            validate_s_corridor_geometry(
                arena_size=corridor_cfg.arena_size,
                wall_thickness=corridor_cfg.wall_thickness,
                interior_wall_thickness=corridor_cfg.interior_wall_thickness,
                inner_wall_offset=corridor_cfg.inner_wall_offset,
                gate_width=corridor_cfg.gate_width,
                spawn_wall_margin=corridor_cfg.spawn_wall_margin,
                spawn_jitter_x=corridor_cfg.spawn_jitter_x,
                spawn_jitter_y=corridor_cfg.spawn_jitter_y,
                min_goal_distance=corridor_cfg.goal_min_distance,
            )
        elif self._world_mode == "straight_obstacle":
            from .straight_obstacle_geometry import (
                validate_straight_obstacle_geometry,
            )

            straight_cfg = cfg.straight_obstacle_world
            straight_cfg.robot_half_width = cfg.robot_half_width
            validate_straight_obstacle_geometry(
                arena_length=straight_cfg.arena_length,
                passage_width=straight_cfg.passage_width,
                wall_thickness=straight_cfg.wall_thickness,
                obstacle_length=straight_cfg.obstacle_length,
                obstacle_width=straight_cfg.obstacle_width,
                spawn_wall_margin=straight_cfg.spawn_wall_margin,
                robot_half_width=cfg.robot_half_width,
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
            f"[LiDAR] policy_layout={self.cfg.lidar_policy_sector_layout} "
            f"policy_sectors={self.cfg.lidar_output_sectors} "
            f"safety_sectors={self.cfg.lidar_safety_output_sectors} "
            f"raw_rays={self.cfg.num_lidar_rays}"
        )
        print(
            f"[Drive] wheel_radius={self.cfg.wheel_radius:.4f}m "
            f"track_width={self.cfg.wheel_base:.4f}m "
            f"wheel_limit={self.cfg.wheel_max_velocity:.3f}rad/s "
            f"body_limits=({self.cfg.max_linear_velocity:.3f}m/s, "
            f"{self.cfg.max_angular_velocity:.3f}rad/s) "
            f"sync={self.cfg.sync_body_velocity_limits_to_wheels}"
        )
        safety_cfg = self.cfg.safety_control
        print(
            f"[Safety] forward_cone=+/-{safety_cfg.forward_half_angle_deg:.1f}deg "
            f"stop_clearance={safety_cfg.stop_clearance:.3f}m "
            f"full_speed_clearance={safety_cfg.full_speed_clearance:.3f}m "
            f"min_forward_scale={safety_cfg.min_forward_speed_scale:.3f} "
            f"max_lidar_aoi={safety_cfg.max_lidar_aoi_s:.3f}s"
        )

        # ---- Joint indices ----
        # 按名字解析, 与 DOF 总数无关。当前 articulation 有 7 个 DOF:
        # wheel_left / wheel_right (驱动) + lift_joint + 4 个脚轮转动副, 后五个
        # 都没有 actuator (脚轮关节在 USD 里也没有 DriveAPI, 保持自由滚动)。
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
                dedicated_detour_layouts=(
                    self._precomputed_dedicated_detour_layouts
                ),
            )
        elif self._world_mode == "corridor":
            self._world_manager = CorridorWorldManager(
                num_envs=self.num_envs,
                cfg=self.cfg.corridor_world,
                device=self.device,
            )
            self._world_manager.write_barrier_poses_to_sim(
                self.scene,
                self.scene.env_origins[:, :2],
            )
        elif self._world_mode == "straight_obstacle":
            self._world_manager = StraightObstacleWorldManager(
                num_envs=self.num_envs,
                cfg=self.cfg.straight_obstacle_world,
                device=self.device,
            )
        else:
            self._world_manager = ObstacleWorldManager(
                num_envs=self.num_envs,
                cfg=self.cfg.obs_world,
                device=self.device,
            )

        self._curriculum = NavigationCurriculum(
            self.cfg.curriculum,
            supported=(self._world_mode == "random_obstacles"
                       and self.cfg.random_obstacle_world.structured_layout),
        )
        self._episode_task_level = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._episode_curriculum_generation = torch.zeros_like(self._episode_task_level)
        self._episode_noise_scale = torch.zeros(self.num_envs, device=self.device)
        self._curriculum_noise_ranges = torch.tensor(
            self.cfg.curriculum.noise_ranges, device=self.device)
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
        # Privileged reference paths exist only for the sealed detour scenes.
        # They shape the training reward but never enter the policy observation.
        max_reference_waypoints = max(
            2,
            3 * self.cfg.random_obstacle_world.dedicated_detour_barrier_count_range[1] + 2,
        )
        if self.cfg.reference_path_backend == "hybrid_astar":
            max_reference_waypoints = self.cfg.reference_planner_max_points
        self._reference_previous_position = torch.zeros(self.num_envs, 2, device=self.device)
        self._reference_gates = torch.zeros(self.num_envs, 3, 4, device=self.device)
        self._reference_gate_counts = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._reference_passed_gates = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._reference_forward_sign = torch.ones(self.num_envs, device=self.device)
        dedicated_start = getattr(self._world_manager, "dedicated_detour_start", self.num_envs)
        self._reference_env_ids = torch.arange(dedicated_start, self.num_envs, device=self.device)
        self._reference_waypoints = torch.zeros(
            self.num_envs, max_reference_waypoints, 2, device=self.device,
        )
        self._reference_point_counts = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device,
        )
        self._reference_valid = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device,
        )
        self._reference_direction = torch.ones(self.num_envs, device=self.device)
        self._prev_reference_progress = torch.zeros(self.num_envs, device=self.device)
        self._last_reference_progress = torch.zeros(self.num_envs, device=self.device)
        self._last_reference_cross_track = torch.zeros(self.num_envs, device=self.device)
        # ---- 策略原始动作与上一控制步动作 ----
        # _prev_policy_actions 同时供奖励 (动作变化正则) 和观测 [134:136] 使用。
        # DirectRLEnv.step 的顺序是 _get_rewards → _reset_idx → _get_observations,
        # 所以观测读到的正好是"本步策略输出"(刚重置的环境为 0)。
        self._policy_actions = torch.zeros_like(self.actions)
        self._prev_policy_actions = torch.zeros_like(self.actions)
        self._last_forward_speed = torch.zeros(self.num_envs, device=self.device)
        self._last_angular_speed = torch.zeros(self.num_envs, device=self.device)
        self._last_wheel_command = torch.zeros_like(self.actions)
        self._last_wheel_actual = torch.zeros_like(self.actions)
        self._raw_wheel_vel_cmd = torch.zeros_like(self.actions)
        self._feasible_wheel_vel_cmd = torch.zeros_like(self.actions)
        self._filtered_wheel_vel_cmd = torch.zeros_like(self.actions)
        self._wheel_vel_cmd = torch.zeros_like(self.actions)
        self._wheel_target_start = torch.zeros_like(self.actions)
        self._joint_vel_cmd = torch.zeros_like(self.actions)
        self._action_substep = 0
        self._wheel_acceleration_scale = torch.ones(self.num_envs, device=self.device)
        self._wheel_projection_scale = torch.ones(self.num_envs, device=self.device)
        self._requested_body_twist_cmd = torch.zeros_like(self.actions)
        self._projected_body_twist_cmd = torch.zeros_like(self.actions)
        self._safe_body_twist_cmd = torch.zeros_like(self.actions)
        self._safety_forward_speed_scale = torch.ones(
            self.num_envs, device=self.device)
        self._safety_safe_speed = torch.zeros(self.num_envs, device=self.device)
        self._safety_forward_clearance = torch.full(
            (self.num_envs,), self.cfg.lidar_max_distance, device=self.device)
        self._safety_intervened = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device)
        self._safety_emergency_stopped = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device)

        # ---- Termination mask ----
        self._terminated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._term_reason = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        # 最近 2048 个结束回合的终止原因，用统一 extras 日志写入训练 TensorBoard。
        self._termination_reason_window: deque[int] = deque(maxlen=2048)
        self._episode_collision_window: deque[tuple[int, bool, int]] = deque(
            maxlen=2048
        )
        self._termination_rates: dict[str, float] = {
            "goal": 0.0,
            "clean_goal": 0.0,
            "goal_with_collision": 0.0,
            "collision": 0.0,
            "had_collision": 0.0,
            "collision_events_mean": 0.0,
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

        # Policy sector centers. hybrid_180 uses 1-degree sectors in [0, 45)
        # and [315, 360), which together form body-frame front +/-45 degrees;
        # the remaining 270 degrees use 3-degree sectors.
        self._policy_lidar_sector_edges = None
        self._policy_lidar_sector_groups = None
        if self.cfg.lidar_policy_sector_layout == "hybrid_180":
            self._policy_lidar_sector_edges = hybrid_180_sector_edges(
                device=self.device
            )
            self._policy_lidar_sector_groups = ((45, 4), (90, 12), (45, 4))
            _sector_centers_rad = sector_centers_from_edges(
                self._policy_lidar_sector_edges
            )
        else:
            sector_step = 360.0 / self.cfg.lidar_output_sectors
            _sector_centers_deg = torch.arange(
                self.cfg.lidar_output_sectors, device=self.device
            ) * sector_step + sector_step / 2.0
            _sector_centers_rad = torch.deg2rad(_sector_centers_deg)
        _sector_centers_rad = torch.atan2(
            torch.sin(_sector_centers_rad), torch.cos(_sector_centers_rad)
        )
        self._lidar_sector_angles = torch.atan2(
            torch.sin(_sector_centers_rad), torch.cos(_sector_centers_rad)
        ).unsqueeze(0).repeat(self.num_envs, 1)

        shield_step = 360.0 / self.cfg.lidar_safety_output_sectors
        _shield_centers_rad = torch.deg2rad(
            torch.arange(
                self.cfg.lidar_safety_output_sectors, device=self.device
            ) * shield_step + shield_step / 2.0
        )
        self._shield_lidar_sector_angles = torch.atan2(
            torch.sin(_shield_centers_rad), torch.cos(_shield_centers_rad)
        ).unsqueeze(0).repeat(self.num_envs, 1)

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
        policy_sectors = self.cfg.lidar_output_sectors
        shield_sectors = self.cfg.lidar_safety_output_sectors
        queue_size = self.cfg.lidar_pending_queue_size
        if queue_size < 1:
            raise ValueError("lidar_pending_queue_size must be at least 1")
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
        curriculum_levels = len(self.cfg.curriculum.noise_ranges)
        collision_termination = self.cfg.collision_termination_enabled_by_level
        collision_budgets = self.cfg.max_nonterminal_collision_events_by_level
        if len(collision_termination) != curriculum_levels:
            raise ValueError(
                "collision_termination_enabled_by_level must match curriculum levels"
            )
        if len(collision_budgets) != curriculum_levels or any(
            not isinstance(value, int) or isinstance(value, bool) or value < 0
            for value in collision_budgets
        ):
            raise ValueError(
                "max_nonterminal_collision_events_by_level must contain "
                "one non-negative integer per curriculum level"
            )
        if any(not isinstance(value, bool) for value in collision_termination):
            raise ValueError(
                "collision_termination_enabled_by_level must contain booleans"
            )
        if not (
            math.isfinite(self.cfg.collision_release_clearance)
            and self.cfg.collision_release_clearance
            > self.cfg.collision_clearance_threshold
        ):
            raise ValueError(
                "collision_release_clearance must exceed collision_clearance_threshold"
            )
        if not (
            math.isfinite(self.cfg.severe_collision_clearance)
            and self.cfg.severe_collision_clearance
            < self.cfg.collision_clearance_threshold
        ):
            raise ValueError(
                "severe_collision_clearance must be below collision_clearance_threshold"
            )
        if not (
            math.isfinite(self.cfg.max_continuous_collision_time_s)
            and self.cfg.max_continuous_collision_time_s > 0.0
        ):
            raise ValueError(
                "max_continuous_collision_time_s must be finite and positive"
            )
        if not (
            math.isfinite(self.cfg.nonterminal_collision_event_penalty)
            and self.cfg.nonterminal_collision_event_penalty <= 0.0
        ):
            raise ValueError(
                "nonterminal_collision_event_penalty must be finite and non-positive"
            )
        self._collision_termination_by_level = torch.as_tensor(
            collision_termination, dtype=torch.bool, device=self.device
        )
        self._collision_budget_by_level = torch.as_tensor(
            collision_budgets, dtype=torch.long, device=self.device
        )
        for name in (
            "nav_progress_per_meter", "nav_proximity_cost_per_second",
            "nav_time_cost_per_second", "nav_action_change_scale", "goal_reach_bonus",
        ):
            value = getattr(self.cfg, name)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if not math.isfinite(self.cfg.nav_failure_penalty) or self.cfg.nav_failure_penalty >= 0:
            raise ValueError("nav_failure_penalty must be finite and negative")
        if (not math.isfinite(self.cfg.nav_clearance_threshold)
                or self.cfg.nav_clearance_threshold <= self.cfg.collision_clearance_threshold):
            raise ValueError("nav_clearance_threshold must exceed collision_clearance_threshold")
        if not (
            self.cfg.ground_static_friction >= self.cfg.ground_dynamic_friction >= 0.0
        ):
            raise ValueError("ground friction must satisfy static >= dynamic >= 0")
        self._cached_scan_ranges = torch.full(
            (self.num_envs, policy_sectors), self.cfg.lidar_max_distance,
            device=self.device)
        self._cached_scan_angles = self._lidar_sector_angles.clone()
        # Shield cache follows the same physical timing but is independent of
        # policy-only sector dropout. It therefore cannot interpret training
        # dropout as free space.
        self._cached_shield_scan_ranges = torch.full(
            (self.num_envs, shield_sectors), self.cfg.lidar_max_distance,
            device=self.device)
        self._cached_shield_scan_angles = self._shield_lidar_sector_angles.clone()
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
            (self.num_envs, queue_size, policy_sectors),
            self.cfg.lidar_max_distance, device=self.device)
        self._pending_scan_angles = self._lidar_sector_angles.unsqueeze(1).repeat(
            1, queue_size, 1)
        self._pending_shield_scan_ranges = torch.full(
            (self.num_envs, queue_size, shield_sectors), self.cfg.lidar_max_distance,
            device=self.device)
        self._pending_shield_scan_angles = self._shield_lidar_sector_angles.unsqueeze(1).repeat(
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

        # Separate delayed shield data from current reward/termination geometry.
        self._shield_clearance = torch.full(
            (self.num_envs, shield_sectors), self.cfg.lidar_max_distance,
            device=self.device)
        self._shield_aoi = torch.zeros(self.num_envs, device=self.device)
        self._safety_clearance = torch.full(
            (self.num_envs, shield_sectors), self.cfg.lidar_max_distance,
            device=self.device)
        self._collision_mask = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device)
        self._collision_event_mask = torch.zeros_like(self._collision_mask)
        self._nonterminal_collision_event_mask = torch.zeros_like(
            self._collision_mask
        )
        self._collision_terminal_mask = torch.zeros_like(self._collision_mask)
        self._collision_contact_active = torch.zeros_like(self._collision_mask)
        self._collision_contact_duration_s = torch.zeros(
            self.num_envs, device=self.device
        )
        self._episode_collision_events = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self._episode_had_collision = torch.zeros_like(self._collision_mask)
        self._failure_mask = torch.zeros_like(self._collision_mask)
        self._goal_reached_mask = torch.zeros_like(self._collision_mask)
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
        self._ray_rect_projection = self._rect_projection()
        self._sector_rect_projection = self._projection_for_sector_angles(
            self._lidar_sector_angles
        )
        self._shield_sector_rect_projection = self._projection_for_sector_angles(
            self._shield_lidar_sector_angles
        )

    # ==================================================================
    # Scene setup
    # ==================================================================
    @staticmethod
    def _bind_friction_material(
        stage, mat_path, static_friction, dynamic_friction,
        root_paths, label, path_filter=None, combine_mode=None,
    ):
        """在 ``root_paths`` 下所有碰撞体上直接绑定一个显式物理材质。

        直接绑到 collider prim 本身(而不是父节点), 因为 USD 材质绑定的继承
        在 PhysX 解析时不可靠 —— 绑在父 Xform 上会被静默丢弃。返回绑定的
        碰撞体数量, 便于确认 cm 路径真的命中了。

        Args:
            mat_path: 材质 prim 路径, 不存在则创建。
            root_paths: 要遍历的 USD prim 子树。
            path_filter: 只在路径包含该子串时绑定(例如 ``"/dwheel"``)。
            combine_mode: PhysX 摩擦合成模式 (min/max/average/multiply)。
                默认 None 保持 PhysX 的 average —— 注意 average 会把低摩擦
                的一侧拉向高摩擦的一侧 (轮子 0.05 配地面 0.8 等效 0.4), 所以
                想真正压低摩擦必须显式设 "min"。
        """
        from pxr import Usd, UsdPhysics, UsdShade

        mat_prim = stage.GetPrimAtPath(mat_path)
        if not mat_prim or not mat_prim.IsValid():
            mat_prim = UsdShade.Material.Define(stage, mat_path).GetPrim()
        mat_api = UsdPhysics.MaterialAPI.Apply(mat_prim)
        mat_api.CreateStaticFrictionAttr().Set(float(static_friction))
        mat_api.CreateDynamicFrictionAttr().Set(float(dynamic_friction))
        mat_api.CreateRestitutionAttr().Set(0.0)
        if combine_mode is not None:
            # 必须走 PhysxMaterialAPI, 不能只是裸写一个 physxMaterial:... 属性:
            # 没挂 schema 时 PhysX 的 USD 解析器会直接忽略它, 表现就是"改了
            # combine_mode 但结果逐字节不变" (实测地面 average -> min 完全没有
            # 影响仿真的任何一位)。而且材质对里 PhysX 只采用其中一侧的模式,
            # 所以两侧材质都要设。
            from pxr import PhysxSchema
            physx_mat_api = PhysxSchema.PhysxMaterialAPI.Apply(mat_prim)
            physx_mat_api.CreateFrictionCombineModeAttr().Set(combine_mode)
        material = UsdShade.Material(mat_prim)

        bound = 0
        for root in root_paths:
            root_prim = stage.GetPrimAtPath(root)
            if not root_prim or not root_prim.IsValid():
                print(f"[Friction] {label}: 找不到 {root}, 跳过")
                continue
            for prim in Usd.PrimRange(root_prim):
                if not prim.HasAPI(UsdPhysics.CollisionAPI):
                    continue
                if path_filter and path_filter not in str(prim.GetPath()):
                    continue
                UsdShade.MaterialBindingAPI.Apply(prim).Bind(
                    material, UsdShade.Tokens.strongerThanDescendants)
                bound += 1
        print(
            f"[Friction] {label}: {bound} collider(s) -> {mat_path} "
            f"(static={static_friction}, dynamic={dynamic_friction}"
            + (f", combine={combine_mode}" if combine_mode else "") + ")"
        )
        return bound

    def _setup_scene(self):
        """构建场景: 平地 + 障碍物 + Emma + 目标标记."""
        # Ground
        sim_utils.spawn_ground_plane("/World/groundPlane", sim_utils.GroundPlaneCfg(
            color=(0.1, 0.1, 0.1),
        ))

        # Bind explicitly to colliders and apply PhysxMaterialAPI. Both sides
        # use min for clarity. PhysX chooses the higher-priority combine mode
        # (average < min < multiply < max); an average partner cannot override
        # a correctly authored min. The earlier bug was a missing API schema.
        import omni.usd
        stage = omni.usd.get_context().get_stage()
        self._bind_friction_material(
            stage, "/World/PhysicsMaterials/Ground",
            self.cfg.ground_static_friction,
            self.cfg.ground_dynamic_friction,
            ("/World/groundPlane",), "ground", combine_mode="min",
        )

        if self._world_mode == "warehouse":
            # 仓库: 墙壁 + 货架
            from .emma_obs_nav_warehouse_world import spawn_warehouse
            spawn_warehouse(self.num_envs, self._warehouse_cfg)
            # 圆柱模板 (动态数量)
            num_cyl = self.cfg.warehouse_num_cylinders
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
                dedicated_detour_env_indices,
                sample_dedicated_detour_layouts,
                spawn_dedicated_detour_barriers,
                spawn_random_obstacle_world,
            )

            random_world_cfg = self.cfg.random_obstacle_world
            spawn_random_obstacle_world(random_world_cfg)
            dedicated_count = len(dedicated_detour_env_indices(
                self.num_envs, random_world_cfg
            ))
            dedicated_seeds = torch.randint(
                0, 2**31 - 1, (dedicated_count,)
            ).cpu().tolist()
            self._precomputed_dedicated_detour_layouts = (
                sample_dedicated_detour_layouts(
                    self.num_envs, random_world_cfg, dedicated_seeds
                )
            )
            obstacle_count = (
                4 + random_world_cfg.num_obstacles
                + random_world_cfg.num_wall_slots
            )
        elif self._world_mode == "corridor":
            spawn_s_corridor(self.cfg.corridor_world)
            obstacle_count = 6
        elif self._world_mode == "straight_obstacle":
            spawn_straight_obstacle_world(self.cfg.straight_obstacle_world)
            obstacle_count = 5
        else:
            # 混合圆柱/方块局部迷宫 + 外墙/错位内墙
            spawn_obstacles(self.num_envs, self.cfg.obs_world)
            obstacle_count = self.cfg.obs_world.num_obstacles + 8

        self.robot = Articulation(self.cfg.robot_cfg)
        configure_drive_contacts(stage, "/World/envs/env_0/Robot/EMMA400K_body", self.cfg)

        goal_marker_cfg = RigidObjectCfg(
            prim_path="/World/envs/env_.*/GoalMarker",
            spawn=sim_utils.SphereCfg(
                radius=self.cfg.goal_marker_radius,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                collision_props=sim_utils.CollisionPropertiesCfg(
                    collision_enabled=False
                ),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0, 0, -100)),
        )
        self.goal_marker = RigidObject(goal_marker_cfg)

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        self.scene.clone_environments(copy_from_source=False)

        if self._world_mode == "random_obstacles":
            barrier_count = spawn_dedicated_detour_barriers(
                self.cfg.random_obstacle_world,
                self._precomputed_dedicated_detour_layouts,
            )
            print(
                f"[DedicatedDetour] authored {barrier_count} continuous "
                f"static barriers in "
                f"{len(self._precomputed_dedicated_detour_layouts)} envs"
            )

        # 驱动轮材质 —— 必须在 clone_environments 之后遍历所有 env, 因为每个
        # env 的轮子 collider 是独立 prim。车身 convexHull / lift 保持默认,
        # 避免底盘蹭地时被大摩擦粘住。
        self._bind_friction_material(
            stage, "/World/PhysicsMaterials/Wheel",
            self.cfg.wheel_static_friction,
            self.cfg.wheel_dynamic_friction,
            ("/World/envs",), "wheel", path_filter="/dwheel",
        )

        # 车身支撑接触。加脚轮之前 base_link 的 convexHull 就是第三个支撑点
        # (实测 ~69 N 法向力), 它没有材质时退化成 PhysX 默认摩擦, 直线白吃
        # ~16 N 阻力、转向还要带车体在地面横向刮擦。现在车由 4 个脚轮撑着,
        # base_link 正常工况不碰地, 这里保留材质只为兜底; combine=min 是必须的,
        # 否则会被地面的 0.8 平均回去。lift 不绑 (保持默认), 它在 z≈0.26。
        self._bind_friction_material(
            stage, "/World/PhysicsMaterials/Base",
            self.cfg.base_static_friction,
            self.cfg.base_dynamic_friction,
            ("/World/envs",), "base", path_filter="/base_link/",
            combine_mode="min",
        )

        # 脚轮材质。脚轮在 USD 里是"圆柱碰撞 + 无驱动转动副" (见
        # assets/Emma_no_ground.usd 的 caster_fl/fr/rl/rr), 沿行进方向是纯滚动,
        # 没有切向阻力; 但原地转向时它必须能横向侧滑 (真机靠万向回转, 这里没有
        # 回转副), 所以侧向也按低摩擦近似, 同样用 combine=min。
        self._bind_friction_material(
            stage, "/World/PhysicsMaterials/Caster",
            self.cfg.caster_static_friction,
            self.cfg.caster_dynamic_friction,
            ("/World/envs",), "caster", path_filter="/caster",
            combine_mode="min",
        )

        self.scene.articulations["robot"] = self.robot

        # TEMP DEBUG: contact sensor on wheel links.
        # 默认关闭 (逐 physics substep 打印会把日志刷爆); 需要时
        # EMMA_DRIVE_DEBUG=1 启动即可打开。
        self._drive_debug = os.environ.get("EMMA_DRIVE_DEBUG", "0") == "1"
        if self._drive_debug:
            from pxr import PhysxSchema
            for _lp in ("base_link", "dwheel_left", "dwheel_right", "lift"):
                _prim = stage.GetPrimAtPath(f"/World/envs/env_0/Robot/EMMA400K_body/{_lp}")
                if _prim.IsValid():
                    PhysxSchema.PhysxContactReportAPI.Apply(_prim)
            from isaaclab.sensors import ContactSensorCfg, ContactSensor
            self._dbg_contact = ContactSensor(ContactSensorCfg(
                prim_path="/World/envs/env_.*/Robot/EMMA400K_body/.*",
                update_period=0.0, history_length=1, debug_vis=False,
            ))
            self.scene.sensors["dbg_contact"] = self._dbg_contact

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
            for i in range(self.cfg.random_obstacle_world.num_wall_slots):
                self.scene.rigid_objects[f"random_interior_wall_{i}"] = RigidObject(
                    RigidObjectCfg(prim_path=f"/World/envs/env_.*/RandomInteriorWall_{i}", spawn=None)
                )
        elif self._world_mode == "corridor":
            for asset_name, prim_name in (
                ("corridor_lower_barrier", "LowerRightBarrier"),
                ("corridor_upper_barrier", "UpperLeftBarrier"),
            ):
                barrier_cfg = RigidObjectCfg(
                    prim_path=f"/World/envs/env_.*/Corridor_{prim_name}",
                    spawn=None,
                )
                self.scene.rigid_objects[asset_name] = RigidObject(barrier_cfg)
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
                elif self._world_mode == "corridor":
                    camera_height = max(
                        14.0, self.cfg.corridor_world.arena_size * 1.15
                    )
                elif self._world_mode == "straight_obstacle":
                    camera_height = max(
                        14.0,
                        self.cfg.straight_obstacle_world.arena_size * 1.15,
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
    # Per-episode curriculum and domain randomization
    # ==================================================================
    def _assign_episode_curriculum(self, env_ids: torch.Tensor) -> None:
        """Choose tasks and noise once at reset; in-flight episodes never change."""
        course = self._curriculum
        n = env_ids.numel()
        levels = torch.full((n,), course.level, dtype=torch.long, device=self.device)
        if course.enabled and course.level > 0:
            replay = torch.rand(n, device=self.device) < self.cfg.curriculum.replay_probability
            easier = torch.randint(course.level, (n,), device=self.device)
            levels = torch.where(replay, easier, levels)
        self._episode_task_level[env_ids] = levels
        self._episode_curriculum_generation[env_ids] = course.generation
        if course.enabled:
            ranges = self._curriculum_noise_ranges[levels]
            noise = ranges[:, 0] + torch.rand(n, device=self.device) * (ranges[:, 1] - ranges[:, 0])
        else:
            noise = torch.full((n,), self.cfg.curriculum.evaluation_noise_scale, device=self.device)
        self._episode_noise_scale[env_ids] = noise

    def _get_noise_scale(self) -> torch.Tensor:
        return self._episode_noise_scale

    def _sample_domain_rand(self, env_ids: torch.Tensor):
        """采样 episode 级 DR 参数 (在 _reset_idx 中调用)."""
        dr = self.cfg.domain_rand
        n = len(env_ids)
        ns = self._get_noise_scale()[env_ids]  # 课程倍率

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
            max_delay = torch.floor(dr.action_delay_max_steps * ns)
            self._dr_action_delay[env_ids] = torch.floor(
                torch.rand(n, device=self.device) * (max_delay + 1)
            ).long()
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
        # mask = torch.abs(self.goal_heading_sin) < 0.1
        # actions[1] = actions[1] * torch.where(
        #     mask, 0.2, 1.0)

        self._policy_actions[:n] = actions

        # ---- Action delay (ring buffer) ----
        if dr.enable_action_noise:
            buf_size = self._action_delay_buf.shape[1]
            idx = self._action_delay_idx[:n] % buf_size
            self._action_delay_buf[torch.arange(n, device=self.device), idx] = actions
            self._action_delay_idx[:n] += 1

            delay = self._dr_action_delay[:n].clamp(min=0)
            read_idx = (idx - delay) % buf_size
            actions = self._action_delay_buf[
                torch.arange(n, device=self.device), read_idx].clone()

        # ---- Action noise (高斯) ----
        if dr.enable_action_noise and dr.action_noise_std > 0:
            noise = torch.randn_like(actions) * dr.action_noise_std * ns[:n, None]
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
        if dr.enable_dynamics_noise and dr.wheel_slip_noise_std > 0:
            slip = torch.randn(n, device=self.device) * dr.wheel_slip_noise_std * ns[:n]
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

        # The clearance governor is part of the policy command path. Apply it
        # before smoothing so both slowdown and recovery receive the same
        # low-pass/acceleration dynamics as ordinary policy actions.
        safety_cfg = cfg.safety_control
        safe_wheels, safety_diag = apply_safety_control(
            feasible_wheels,
            self._shield_clearance[:n],
            self._shield_lidar_sector_angles[:n],
            self._shield_aoi[:n],
            wheel_radius=cfg.wheel_radius,
            wheel_base=cfg.wheel_base,
            wheel_max_velocity=cfg.wheel_max_velocity,
            forward_half_angle_deg=safety_cfg.forward_half_angle_deg,
            stop_clearance=safety_cfg.stop_clearance,
            full_speed_clearance=safety_cfg.full_speed_clearance,
            min_forward_speed_scale=safety_cfg.min_forward_speed_scale,
            max_lidar_aoi_s=safety_cfg.max_lidar_aoi_s,
            wheel_radius_scale=self._dr_wheel_r_scale[:n],
            has_valid_scan=self._has_cached_scan[:n],
            enabled=safety_cfg.enabled,
        )

        if cfg.enable_motion_command_low_pass:
            filtered_wheels, _ = low_pass_wheel_velocity_command(
                safe_wheels,
                self._wheel_vel_cmd[:n],
                self.step_dt,
                cfg.motion_command_low_pass_tau_s,
            )
        else:
            filtered_wheels = safe_wheels

        if cfg.enable_motion_acceleration_limits:
            wheel_targets, acceleration_scale = limit_wheel_velocity_acceleration(
                filtered_wheels, self._wheel_vel_cmd[:n], self.step_dt,
                cfg.wheel_radius, cfg.wheel_base,
                cfg.max_linear_acceleration * cfg.linear_acceleration_command_scale,
                cfg.max_angular_acceleration * cfg.angular_acceleration_command_scale,
                self._dr_wheel_r_scale[:n],
            )
        else:
            wheel_targets = filtered_wheels
            acceleration_scale = torch.ones_like(projection_scale)
        self._wheel_acceleration_scale[:n] = acceleration_scale

        self._raw_wheel_vel_cmd[:n] = raw_wheels
        self._feasible_wheel_vel_cmd[:n] = feasible_wheels
        self._filtered_wheel_vel_cmd[:n] = filtered_wheels
        self._wheel_target_start.copy_(self._wheel_vel_cmd)
        self._action_substep = 0
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
        self._safety_forward_speed_scale[:n] = safety_diag["forward_speed_scale"]
        self._safety_safe_speed[:n] = safety_diag["safe_speed"]
        self._safety_forward_clearance[:n] = safety_diag["forward_clearance"]
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
        self._action_substep += 1
        if self.cfg.enable_motion_acceleration_limits:
            fraction = min(self._action_substep / self.cfg.decimation, 1.0)
            interpolated = self._wheel_target_start + fraction * (
                self._joint_vel_cmd - self._wheel_target_start
            )
            applied = interpolated
        else:
            applied = self._joint_vel_cmd
        self._wheel_vel_cmd.copy_(applied)
        self.robot.set_joint_velocity_target(
            applied[:, 0:1], joint_ids=self.left_ids,
        )
        self.robot.set_joint_velocity_target(
            applied[:, 1:2], joint_ids=self.right_ids,
        )
        # print("drive_vel left: ", self.robot.data.joint_vel[:, self.left_ids])
        # print("drive_vel right: ", self.robot.data.joint_vel[:, self.right_ids])
        # print("dbg: root_lin_vel", self.robot.data.root_lin_vel_w[0],
        #       "body_names", self.robot.data.body_names[:4])
        # print("dbg: root_ang_vel", self.robot.data.root_ang_vel_w[0])


        if not self._drive_debug:
            return
        print("dbg: root_lin_vel", self.robot.data.root_lin_vel_w[0],
              "body_names", self.robot.data.body_names[:4])
        print("dbg: root_ang_vel", self.robot.data.root_ang_vel_w[0])
        cf = self._dbg_contact.data.net_forces_w
        if cf is not None:
            print("dbg: contact_net_forces", self._dbg_contact.body_names, "\n", cf[0])
        print("dbg: estimated_pd_torque", self.robot.data.applied_torque[0, :2])
        print("dbg: effort_limit", self.robot.actuators["wheel_left_actuator"].effort_limit[0])
        print("dbg: sim effort_limits", self.robot.data.joint_effort_limits[0])
        print("dbg: sim vel_target", self.robot.data.joint_vel_target[0])
        print("dbg: external_effort_input", self.robot.root_physx_view.get_dof_actuation_forces()[0])
        print("dbg: max_forces", self.robot.root_physx_view.get_dof_max_forces()[0])
        print("dbg: joint_stiffness", self.robot.data.joint_stiffness[0])
        print("dbg: joint_damping", self.robot.data.joint_damping[0])
        print("dbg: root_pos_z", self.robot.data.root_pos_w[0, 2].item(),
              "up_z", self.robot.data.projected_gravity_b[0])
        print("dbg: wheel_pos_z", self.robot.data.body_pos_w[0, [self.robot.find_bodies("dwheel_left")[0][0]], 2])


    # ==================================================================
    # Observation — 136 dims (120 LiDAR + 2 goal_dir + 1 goal_dist + 3 IMU + 8 time
    #                        + 2 prev_action)
    # ==================================================================
    # Layout:
    #   [  0:120] lidar_norm      120    LiDAR clearance 取反 (3° sector, 360° FOV)
    #   [120:122] goal_dir_b         2    ICR→目标方向 (body frame, 归一化)
    #   [122:123] goal_dist_norm     1    goal_dist / init_goal_dist (动态, 可 >1)
    #   [123:125] vel_b              2    ICR速度 (body frame) / max_linear_velocity
    #   [125:126] ang_vel_z_norm     1    Z轴角速度 / max_angular_velocity
    #   [126:134] time_enc           8    实际 AoI 秒数的连续正弦编码
    #   [134:136] prev_action        2    策略上一步发出的 signed action (∈[-1,1]²)
    #
    # prev_action 暴露的是策略**自己**上一步的输出，用来打断"瞬时角速度 +
    # 无动作记忆"这条单步延迟反馈通路: 循环策略此前只能靠 GRU 隐状态记住自己
    # 发出的交替指令，而隐状态会把交替模式保持并放大。
    # 取的是 _prev_policy_actions，即 delay/noise **之前**的原始策略输出,
    # 因此这一路是确定性的 ("我要了什么", 而非"车轮实际执行了什么")。
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

        noise = self._episode_noise_scale[env_ids]
        if mode == "random":
            if cfg.lidar_fixed_frequency_hz <= 0:
                raise ValueError("nominal lidar_fixed_frequency_hz must be positive")
            nominal_period = 1.0 / cfg.lidar_fixed_frequency_hz
            periods = nominal_period + noise * (periods - nominal_period)
            phase = torch.rand(n, device=self.device) * periods
        self._sensor_period[env_ids] = periods
        self._sensor_latency[env_ids] = latency * noise
        self._next_sensor_timestamp[env_ids] = current_time[env_ids] + phase
        self._sensor_schedule_initialized[env_ids] = True

        # A reset invalidates both the delivered cache and all in-flight scans.
        self._has_cached_scan[env_ids] = False
        self._cached_scan_ranges[env_ids] = cfg.lidar_max_distance
        self._cached_scan_angles[env_ids] = self._lidar_sector_angles[env_ids]
        self._cached_shield_scan_ranges[env_ids] = cfg.lidar_max_distance
        self._cached_shield_scan_angles[env_ids] = (
            self._shield_lidar_sector_angles[env_ids]
        )
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
            cfg.lidar_safety_output_sectors,
            cfg.lidar_safety_rays_per_sector,
        ).amin(dim=-1)
        self._collision_mask = (
            self._safety_clearance.amin(dim=-1) <= cfg.collision_clearance_threshold
        )
        return self._safety_clearance

    def _capture_lidar_sector_ranges(
        self, noise_scale: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Capture policy and shield sectors from the same physical scan.

        Both branches see the same simulated range noise. Policy-only sector
        dropout is applied after the shield copy is made.
        """

        cfg = self.cfg
        dr = cfg.domain_rand
        clean_ranges, valid = self._sample_clean_lidar_ranges()
        raw_ranges = clean_ranges.clone()
        if dr.enable_lidar_noise:
            noisy = raw_ranges + (
                torch.randn_like(raw_ranges) * dr.lidar_noise_std * noise_scale[:, None])
            raw_ranges = torch.where(valid, noisy, raw_ranges)
        raw_ranges = raw_ranges.clamp(min=0.0, max=cfg.lidar_max_distance)

        if cfg.lidar_policy_sector_layout == "hybrid_180":
            sector_ranges, sector_angles, _ = min_pool_clearance_rays_grouped(
                raw_ranges,
                self._lidar_ray_angles,
                self._ray_rect_projection,
                self._policy_lidar_sector_groups,
            )
            # Same noisy physical scan, but an independent uniform angular
            # aggregation for the policy-internal safety controller.
            shield_sector_ranges, shield_sector_angles, _ = min_pool_clearance_rays(
                raw_ranges,
                self._lidar_ray_angles,
                self._ray_rect_projection,
                cfg.lidar_safety_output_sectors,
            )
        else:
            sector_ranges, sector_angles, _ = min_pool_clearance_rays(
                raw_ranges,
                self._lidar_ray_angles,
                self._ray_rect_projection,
                cfg.lidar_output_sectors,
            )
            # Preserve the original uniform-120 fast path.
            shield_sector_ranges = sector_ranges.clone()
            shield_sector_angles = sector_angles.clone()
        if dr.enable_lidar_dropout:
            dropout_prob = dr.lidar_dropout_prob * noise_scale[:, None]
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
        noise_scale: torch.Tensor,
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
                scan_dropout = torch.rand(self.num_envs, device=self.device) < cfg.lidar_scan_dropout_prob * noise_scale
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
                ) * noise_scale
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
            self._policy_lidar_sector_edges,
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
            cfg.lidar_safety_output_sectors,
            cfg.lidar_max_distance,
            cfg.robot_half_length,
            cfg.robot_half_width,
            cfg.lidar_offset_x,
            cfg.lidar_offset_y,
        )
        shield_default_clearance = (
            cfg.lidar_max_distance - self._shield_sector_rect_projection
        )
        self._shield_clearance = torch.where(
            self._has_cached_scan.unsqueeze(1),
            shield_clearance,
            shield_default_clearance,
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
        _, aoi, clearance = self._update_async_lidar(
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
            final_angles = self._lidar_sector_angles[0]
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
            probe_indices = tuple(
                sorted({
                    0,
                    cfg.lidar_output_sectors // 4,
                    cfg.lidar_output_sectors // 2,
                    3 * cfg.lidar_output_sectors // 4,
                    cfg.lidar_output_sectors - 1,
                })
            )
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

        # Continuous encoding of actual Age of Information (generation time,
        # including sensor latency and dropped scans), not a cycle index.
        if cfg.lidar_aoi_enabled:
            time_enc = sinusoidal_aoi_encoding(
                aoi, cfg.time_encoding_dim, cfg.lidar_aoi_encoding_scale_s)
        else:
            time_enc = torch.zeros(
                self.num_envs, cfg.time_encoding_dim, device=self.device)

        # ================================================================
        # Observation: policy LiDAR sectors + 16 robot/task state values.
        # ================================================================
        obs = torch.cat([
            lidar_norm,
            goal_dir_b,
            goal_dist_norm,
            vel_b,
            ang_vel_z_norm,
            time_enc,
            self._prev_policy_actions,
        ], dim=-1)
        self.goal_heading_sin = goal_dir_b[:, 1]
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
        return self._projection_for_sector_angles(self._lidar_ray_angles)

    def _projection_for_sector_angles(self, angles: torch.Tensor) -> torch.Tensor:
        """Distance from the LiDAR origin to the footprint along ``angles``."""

        cos_a = torch.cos(angles)
        sin_a = torch.sin(angles)

        hl = self.cfg.robot_half_length    # 0.4115
        hw = self.cfg.robot_half_width     # 0.266
        sx = self.cfg.lidar_offset_x       # 0.38 (传感器 body x)
        sy = self.cfg.lidar_offset_y       # 0.0

        if not (-hl <= sx <= hl and -hw <= sy <= hw):
            raise ValueError("LiDAR origin must be inside the robot rectangle")
            
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
        nav_sum = self._compute_nav_rewards()

        if hasattr(self, "extras"):
            log = self.extras.setdefault("log", {})
            log["Safety/intervention_rate"] = self._safety_intervened.float().mean()
            log["Safety/emergency_stop_rate"] = (
                self._safety_emergency_stopped.float().mean()
            )
            log["Safety/forward_speed_scale_mean"] = (
                self._safety_forward_speed_scale.mean()
            )
            log["Safety/safe_speed_mean"] = self._safety_safe_speed.mean()
            log["Control/wheel_projection_scale_mean"] = (
                self._wheel_projection_scale.mean()
            )
            log["Control/wheel_projection_rate"] = (
                self._wheel_projection_scale < 0.999
            ).float().mean()
            log["Episode/goal_rate"] = self._termination_rates["goal"]
            log["Episode/success_rate"] = self._termination_rates["goal"]
            log["Episode/clean_goal_rate"] = self._termination_rates["clean_goal"]
            log["Episode/goal_with_collision_rate"] = self._termination_rates[
                "goal_with_collision"
            ]
            log["Episode/collision_rate"] = self._termination_rates["collision"]
            log["Episode/had_collision_rate"] = self._termination_rates[
                "had_collision"
            ]
            log["Episode/collision_events_mean"] = self._termination_rates[
                "collision_events_mean"
            ]
            log["Episode/drift_rate"] = self._termination_rates["drift"]
            log["Episode/timeout_rate"] = self._termination_rates["timeout"]
            log["Episode/fell_rate"] = self._termination_rates["fell"]
            log["Episode/other_termination_rate"] = (
                self._termination_rates["other"]
            )
            log["Episode/rate_window_size"] = len(
                self._termination_reason_window
            )
            log["Safety/forward_clearance_mean"] = (
                self._safety_forward_clearance.mean()
            )
            log["Collision/current_rate"] = self._collision_mask.float().mean()
            log["Collision/event_rate"] = self._collision_event_mask.float().mean()
            log["Collision/terminal_rate"] = (
                self._collision_terminal_mask.float().mean()
            )
            log["Collision/contact_duration_mean_s"] = (
                self._collision_contact_duration_s.mean()
            )
            for name, term in self._reward_terms.items():
                log[f"Reward/{name}_mean"] = term.mean()
            log["Reward/total_mean"] = nav_sum.mean()
            course = self._curriculum
            log["Curriculum/level"] = course.level
            log["Curriculum/noise_mean"] = self._episode_noise_scale.mean()
            log["Curriculum/current_success_rate"] = course.last_success_rate
            log["Curriculum/current_failure_rate"] = course.last_failure_rate
            log["Curriculum/window_episodes"] = len(course.window)
            log["Curriculum/level_steps"] = course.steps_since_change
            log["Curriculum/current_episode_fraction"] = (
                (self._episode_task_level == course.level)
                & (self._episode_curriculum_generation == course.generation)
            ).float().mean()
            log["Navigation/goal_distance_mean"] = self._last_goal_distance.mean()
            log["Navigation/minimum_clearance_mean"] = self._safety_clearance.amin(dim=-1).mean()
            reference_mask = self._reference_valid.to(
                self._last_reference_cross_track.dtype
            )
            reference_count = reference_mask.sum().clamp_min(1.0)
            log["Reference/active_fraction"] = reference_mask.mean()
            bank = getattr(self._world_manager, "_reference_bank", None)
            if bank is not None:
                for name, value in bank.stats.items():
                    log[f"ReferencePlanner/{name}"] = float(value)
                requests = max(1, bank.stats["requests"])
                log["ReferencePlanner/success_rate"] = bank.stats["successes"] / requests
                log["ReferencePlanner/mean_reset_plan_ms"] = 1000*bank.stats["seconds"] / requests
                successes = max(1, bank.stats["successes"])
                log["ReferencePlanner/stop_path_rate"] = bank.stats["stop_paths"] / successes
                log["ReferencePlanner/rotation_path_rate"] = bank.stats["rotation_paths"] / successes
            log["Reference/cross_track_mean"] = (
                self._last_reference_cross_track * reference_mask
            ).sum() / reference_count
            log["Reference/progress_mean"] = (
                self._reference_direction
                * self._last_reference_progress
                * reference_mask
            ).sum() / reference_count
            log["Motion/forward_speed_mean"] = self._last_forward_speed.mean()
            log["Motion/reverse_rate"] = (self._last_forward_speed < -0.02).float().mean()
            log["Motion/abs_angular_speed_mean"] = self._last_angular_speed.abs().mean()

        return nav_sum.unsqueeze(-1)

    # ==================================================================
    # Navigation reward
    # ==================================================================
    def _compute_nav_rewards(self) -> torch.Tensor:
        """Reward task progress directly, without a heuristic heading teacher."""
        cfg = self.cfg
        robot = self.robot
        goal_distance = torch.linalg.norm(
            self.goal_pos_xy - robot.data.root_pos_w[:, :2], dim=-1
        )
        # _get_dones runs first: use the same fresh geometry and terminal masks.
        self._reward_terms = navigation_reward_terms(
            self._prev_goal_dist, goal_distance,
            self._safety_clearance.amin(dim=-1),
            self._policy_actions, self._prev_policy_actions,
            self.episode_length_buf > 1,
            self._goal_reached_mask, self._failure_mask,
            step_dt=self.step_dt,
            progress_per_meter=cfg.nav_progress_per_meter,
            clearance_threshold=cfg.nav_clearance_threshold,
            collision_clearance=cfg.collision_clearance_threshold,
            proximity_cost_per_second=cfg.nav_proximity_cost_per_second,
            time_cost_per_second=cfg.nav_time_cost_per_second,
            action_change_scale=cfg.nav_action_change_scale,
            goal_bonus=cfg.goal_reach_bonus,
            failure_penalty=cfg.nav_failure_penalty,
        )
        if cfg.reference_path_reward_enabled:
            local_position = (
                robot.data.root_pos_w[:, :2] - self.scene.env_origins[:, :2]
            )
            if cfg.reference_path_backend == "hybrid_astar":
                ids = self._reference_env_ids
                reference_progress = self._prev_reference_progress.clone()
                cross_track = torch.zeros_like(reference_progress)
                if ids.numel():
                    # Only dedicated rows participate: avoid N_env * max_points
                    # projection work for the thousands of ordinary environments.
                    yaw = math_utils.euler_xyz_from_quat(robot.data.root_quat_w[ids])[2]
                    progress, error, passed = project_continuous_progress(
                        local_position[ids], self._reference_previous_position[ids],
                        self._reference_waypoints[ids], self._reference_point_counts[ids],
                        self._prev_reference_progress[ids], self._reference_gates[ids],
                        self._reference_gate_counts[ids], self._reference_passed_gates[ids],
                        yaw, self._reference_forward_sign[ids],
                        half_length=cfg.robot_half_length, half_width=cfg.robot_half_width,
                        collision_margin=cfg.reference_planner_collision_margin,
                        max_step=cfg.reference_path_max_progress_per_step)
                    reference_progress[ids] = progress
                    cross_track[ids] = error
                    self._reference_passed_gates[ids] = passed
                self._reference_previous_position.copy_(local_position.detach())
            else:
                reference_progress, cross_track = project_points_to_polylines(
                    local_position, self._reference_waypoints, self._reference_point_counts)
            progress_delta = self._reference_direction * (
                reference_progress - self._prev_reference_progress
            )
            progress_delta = progress_delta.clamp(
                -cfg.reference_path_max_progress_per_step,
                cfg.reference_path_max_progress_per_step,
            )
            path_progress_reward = (
                cfg.reference_path_progress_per_meter * progress_delta
            )
            path_progress_reward = torch.where(
                self._failure_mask | ~torch.isfinite(path_progress_reward),
                torch.zeros_like(path_progress_reward),
                path_progress_reward,
            )
            self._reward_terms["progress"] = torch.where(
                self._reference_valid,
                path_progress_reward,
                self._reward_terms["progress"],
            )
            cross_track_excess = (
                (cross_track - cfg.reference_path_cross_track_tolerance).clamp_min(0.0)
                / (
                    cfg.reference_path_cross_track_full_cost_error
                    - cfg.reference_path_cross_track_tolerance
                )
            ).clamp_max(1.0)
            self._reward_terms["reference_cross_track"] = torch.where(
                self._reference_valid,
                -cfg.reference_path_cross_track_cost_per_second
                * self.step_dt
                * cross_track_excess.square(),
                torch.zeros_like(cross_track),
            )
            self._prev_reference_progress.copy_(reference_progress.detach())
            self._last_reference_progress = reference_progress.detach()
            self._last_reference_cross_track = cross_track.detach()
        self._reward_terms["collision_event"] = (
            self._nonterminal_collision_event_mask.to(goal_distance.dtype)
            * cfg.nonterminal_collision_event_penalty
        )
        self._prev_goal_dist.copy_(goal_distance.detach())
        self._prev_policy_actions.copy_(self._policy_actions.detach())
        self._last_goal_distance = goal_distance.detach()

        # Keep measured motion diagnostics used by play.py and evaluation tools.
        forward_w = math_utils.quat_apply(
            robot.data.root_quat_w, robot.data.FORWARD_VEC_B
        )
        forward_w = torch.nn.functional.normalize(forward_w[:, :2], dim=-1)
        self._last_forward_speed = torch.sum(
            robot.data.root_com_lin_vel_w[:, :2] * forward_w, dim=-1
        ).detach()
        self._last_angular_speed = robot.data.root_ang_vel_b[:, 2].detach()
        self._last_wheel_command = self._wheel_vel_cmd.detach().clone()
        self._last_wheel_actual = torch.stack(
            (robot.data.joint_vel[:, int(self.left_ids[0])],
             robot.data.joint_vel[:, int(self.right_ids[0])]), dim=-1
        ).detach()
        return sum(self._reward_terms.values())

    # ==================================================================
    # Termination
    # ==================================================================
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._curriculum.advance()
        robot = self.robot
        cfg = self.cfg

        # DirectRLEnv evaluates dones before rewards and observations. Refresh
        # a clean current scan here so collision termination is neither delayed
        # by one control step nor coupled to noisy policy observations.
        self._update_safety_clearance()

        time_out = self.episode_length_buf >= self.max_episode_length - 1
        root_z = robot.data.root_pos_w[:, 2]
        fell = (root_z < cfg.fall_reset_height) | ~torch.isfinite(robot.data.root_pos_w).all(dim=-1)

        local_pos = robot.data.root_pos_w[:, :2] - self.scene.env_origins[:, :2]
        drift_limit = self.cfg.terrain_sub_size / 2.0 - self.cfg.robot_half_length
        drifted = (local_pos.abs() > drift_limit).any(dim=-1)

        goal_dist = torch.linalg.norm(
            self.goal_pos_xy - robot.data.root_pos_w[:, :2], dim=-1)
        minimum_clearance = self._safety_clearance.amin(dim=-1)
        task_levels = self._episode_task_level.clamp(
            0, self._collision_termination_by_level.numel() - 1
        )
        collision_state = collision_curriculum_step(
            minimum_clearance,
            self._collision_contact_active,
            self._collision_contact_duration_s,
            self._episode_collision_events,
            self._collision_termination_by_level[task_levels],
            self._collision_budget_by_level[task_levels],
            step_dt=self.step_dt,
            collision_clearance=cfg.collision_clearance_threshold,
            release_clearance=cfg.collision_release_clearance,
            severe_clearance=cfg.severe_collision_clearance,
            max_continuous_collision_time_s=(
                cfg.max_continuous_collision_time_s
            ),
        )
        collision = collision_state["collision"]
        collision_terminal = collision_state["terminal_collision"]
        self._collision_event_mask = collision_state["collision_event"]
        self._nonterminal_collision_event_mask = collision_state[
            "nonterminal_collision_event"
        ]
        self._collision_terminal_mask = collision_terminal
        self._collision_contact_active = collision_state["contact_active"]
        self._collision_contact_duration_s = collision_state[
            "contact_duration_s"
        ]
        self._episode_collision_events = collision_state["episode_event_count"]
        self._episode_had_collision |= self._collision_event_mask

        self._failure_mask = (
            collision_terminal | fell | drifted | self._terminated
        )
        goal_reached = (goal_dist < cfg.goal_reach_dist) & ~self._failure_mask
        self._goal_reached_mask = goal_reached

        terminated = self._failure_mask | goal_reached
        truncated = time_out & (~terminated)

        self._term_reason[collision_terminal] = 4
        self._term_reason[goal_reached] = 5
        self._term_reason[fell & ~collision_terminal & ~goal_reached] = 2
        self._term_reason[
            drifted & ~collision_terminal & ~goal_reached & ~fell
        ] = 3
        self._term_reason[truncated] = 1

        # Preserve current and terminal evaluation values before DirectRLEnv
        # auto-resets ended environments. Every tensor is detached and cloned so
        # the returned extras cannot be overwritten by _reset_idx in this step.
        ending = terminated | truncated
        velocity_w = robot.data.root_com_lin_vel_w[:, :2]
        forward_w = math_utils.quat_apply(
            robot.data.root_quat_w, robot.data.FORWARD_VEC_B
        )
        forward_w = torch.nn.functional.normalize(forward_w[:, :2], dim=-1)
        evaluation = {
            "position_xy": local_pos.detach().clone(),
            "goal_position_xy": (
                self.goal_pos_xy - self.scene.env_origins[:, :2]
            ).detach().clone(),
            "goal_distance": goal_dist.detach().clone(),
            "initial_goal_distance": self._init_goal_dist.detach().clone(),
            "minimum_clearance": self._safety_clearance.amin(dim=-1).detach().clone(),
            "forward_speed": torch.sum(velocity_w * forward_w, dim=-1).detach().clone(),
            "angular_speed": robot.data.root_ang_vel_b[:, 2].detach().clone(),
            "episode_steps": self.episode_length_buf.detach().clone(),
            "terminated": terminated.detach().clone(),
            "truncated": truncated.detach().clone(),
            "ended": ending.detach().clone(),
            "termination_reason": self._term_reason.detach().clone(),
            # Backward-compatible episode-level collision flag. A successful
            # episode that touched a wall is therefore distinguishable from a
            # clean success even when contact ended before reaching the goal.
            "collision": self._episode_had_collision.detach().clone(),
            "collision_current": collision.detach().clone(),
            "collision_terminal": collision_terminal.detach().clone(),
            "collision_event": self._collision_event_mask.detach().clone(),
            "collision_event_count": (
                self._episode_collision_events.detach().clone()
            ),
            "goal_reached": goal_reached.detach().clone(),
            "clean_goal_reached": (
                goal_reached & ~self._episode_had_collision
            ).detach().clone(),
            "fell": fell.detach().clone(),
            "drifted": drifted.detach().clone(),
            "timeout": truncated.detach().clone(),
        }
        self.extras["evaluation"] = evaluation

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
        n_ended = ending.sum().item()
        if n_ended > 0:
            reason_names = {1: "timeout", 2: "fell", 3: "drifted",
                            4: "obstacle", 5: "goal_reached"}
            ended_ids = torch.where(ending)[0]
            outcomes = torch.stack((
                self._episode_task_level[ended_ids], self._episode_curriculum_generation[ended_ids],
                goal_reached[ended_ids].long(), self._failure_mask[ended_ids].long(),
            ), dim=-1).cpu().tolist()
            if self._curriculum.record(*zip(*outcomes)):
                print(f"[Curriculum] level={self._curriculum.level} "
                      f"success={self._curriculum.last_success_rate:.3f} "
                      f"failure={self._curriculum.last_failure_rate:.3f}")
            ended_indices = ended_ids.tolist()
            ended_reasons: list[int] = []
            for end_idx in ended_indices:
                r = self._term_reason[end_idx].item()
                ended_reasons.append(int(r))

            self._termination_reason_window.extend(ended_reasons)
            self._episode_collision_window.extend(
                (
                    reason,
                    bool(self._episode_had_collision[end_idx].item()),
                    int(self._episode_collision_events[end_idx].item()),
                )
                for reason, end_idx in zip(ended_reasons, ended_indices)
            )
            window_counts = Counter(self._termination_reason_window)
            window_total = len(self._termination_reason_window)
            self._termination_rates["goal"] = window_counts[5] / window_total
            self._termination_rates["clean_goal"] = sum(
                reason == 5 and not had_collision
                for reason, had_collision, _ in self._episode_collision_window
            ) / window_total
            self._termination_rates["goal_with_collision"] = sum(
                reason == 5 and had_collision
                for reason, had_collision, _ in self._episode_collision_window
            ) / window_total
            self._termination_rates["collision"] = window_counts[4] / window_total
            self._termination_rates["had_collision"] = sum(
                had_collision
                for _, had_collision, _ in self._episode_collision_window
            ) / window_total
            self._termination_rates["collision_events_mean"] = sum(
                event_count
                for _, _, event_count in self._episode_collision_window
            ) / window_total
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
        self._assign_episode_curriculum(env_ids)
        self._sample_domain_rand(env_ids)

        # ---- 1. 只初始化一次地图，课程升级不改动正在进行的回合 ----
        if not getattr(self, "_obs_layout_initialized", False):
            if self._world_mode == "warehouse":
                all_ids = torch.arange(self.num_envs, device=self.device)
                self._world_manager.sample_cylinders(
                    all_ids, num_cylinders=self.cfg.warehouse_num_cylinders, ensure_per_channel=1)
                self._world_manager.write_cylinder_poses_to_sim(
                    self.scene, self.num_envs, self.scene.env_origins[:, :2])
            elif self._world_mode == "random_obstacles":
                self._world_manager.sample_initial_layout()
                self._world_manager.write_poses_to_sim(self.scene, self.scene.env_origins[:, :2])
            elif self._world_mode == "ring":
                self._world_manager.sample_initial_layout(num_obstacles=self.cfg.obs_world.num_obstacles)
                self._world_manager.write_poses_to_sim(
                    self.scene, self.num_envs, self.scene.env_origins[:, :2])
            self._obs_layout_initialized = True

        # Sample start/goal on the existing map; no geometry changes during episodes.
        dr = self.cfg.domain_rand

        if self._world_mode == "warehouse":
            # 区块感知 spawn, OBB + 圆柱安全检测
            _wm = self._world_manager
            _cyl_pos = _wm.positions[env_ids] if _wm.positions is not None else None
            _cyl_r = _wm.radii if hasattr(_wm, 'radii') else None
            _num_cyl = self.cfg.warehouse_num_cylinders
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
            ) = self._world_manager.sample_spawn_pairs(
                env_ids, **({"task_levels": self._episode_task_level[env_ids]}
                            if self._world_mode == "random_obstacles" else {})
            )
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
        self._reset_reference_paths(env_ids, emma_local, goal_local, rand_yaw)

        # ---- 4. 重置跟踪状态 ----
        ICR_pos_w = self.robot.data.root_pos_w
        init_dist = torch.linalg.norm(
            ICR_pos_w[env_ids, :2] - self.goal_pos_xy[env_ids], dim=1,
        )
        self._init_goal_dist[env_ids] = init_dist
        self._prev_goal_dist[env_ids] = init_dist
        self._policy_actions[env_ids] = 0.0
        self._prev_policy_actions[env_ids] = 0.0
        self._raw_wheel_vel_cmd[env_ids] = 0.0
        self._feasible_wheel_vel_cmd[env_ids] = 0.0
        self._filtered_wheel_vel_cmd[env_ids] = 0.0
        self._wheel_vel_cmd[env_ids] = 0.0
        self._wheel_target_start[env_ids] = 0.0
        self._joint_vel_cmd[env_ids] = 0.0
        self._wheel_acceleration_scale[env_ids] = 1.0
        self._wheel_projection_scale[env_ids] = 1.0
        self._requested_body_twist_cmd[env_ids] = 0.0
        self._projected_body_twist_cmd[env_ids] = 0.0
        self._safe_body_twist_cmd[env_ids] = 0.0
        self._safety_forward_speed_scale[env_ids] = 1.0
        self._safety_safe_speed[env_ids] = 0.0
        self._safety_forward_clearance[env_ids] = self.cfg.lidar_max_distance
        self._safety_intervened[env_ids] = False
        self._safety_emergency_stopped[env_ids] = False
        self._collision_mask[env_ids] = False
        self._collision_event_mask[env_ids] = False
        self._nonterminal_collision_event_mask[env_ids] = False
        self._collision_terminal_mask[env_ids] = False
        self._collision_contact_active[env_ids] = False
        self._collision_contact_duration_s[env_ids] = 0.0
        self._episode_collision_events[env_ids] = 0
        self._episode_had_collision[env_ids] = False
        self._failure_mask[env_ids] = False
        self._goal_reached_mask[env_ids] = False
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
        if self._spawn_diag_count < 10:
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

    def _reset_reference_paths(
        self,
        env_ids: torch.Tensor,
        start_local: torch.Tensor,
        goal_local: torch.Tensor,
        start_yaw: torch.Tensor,
    ) -> None:
        """Install privileged detour routes for newly reset environments."""
        self._reference_previous_position[env_ids] = start_local
        self._reference_gates[env_ids] = 0.0
        self._reference_gate_counts[env_ids] = 0
        self._reference_passed_gates[env_ids] = 0
        self._reference_forward_sign[env_ids] = torch.where(goal_local[:, 1] >= start_local[:, 1], 1., -1.)
        self._reference_waypoints[env_ids] = 0.0
        self._reference_point_counts[env_ids] = 0
        self._reference_valid[env_ids] = False
        self._reference_direction[env_ids] = 1.0
        self._prev_reference_progress[env_ids] = 0.0
        self._last_reference_progress[env_ids] = 0.0
        self._last_reference_cross_track[env_ids] = 0.0
        if (
            not self.cfg.reference_path_reward_enabled
            or self._world_mode != "random_obstacles"
            or not hasattr(self._world_manager, "reference_routes")
        ):
            return

        if self.cfg.reference_path_backend == "hybrid_astar":
            cfg = self.cfg
            planner_cfg = PlannerConfig(
                half_length=cfg.robot_half_length, half_width=cfg.robot_half_width,
                margin=cfg.reference_planner_collision_margin,
                xy_resolution=cfg.reference_planner_xy_resolution,
                yaw_resolution_deg=cfg.reference_planner_yaw_resolution_deg,
                min_turning_radius=cfg.reference_planner_min_turning_radius,
                analytic_radius_factors=cfg.reference_planner_analytic_radius_factors,
                inplace_rotation_fallback=cfg.reference_planner_inplace_rotation_fallback,
                max_seconds=cfg.reference_planner_max_seconds,
                max_expansions=cfg.reference_planner_max_expansions,
                cache_size=cfg.reference_planner_cache_size,
                max_points=cfg.reference_planner_max_points,
            )
            routes, counts, valid, gates, gate_counts = self._world_manager.planned_reference_routes(
                env_ids, self._episode_task_level[env_ids], start_local, start_yaw,
                goal_local, planner_cfg, failure_dump_dir=cfg.reference_planner_failure_dump_dir)
            self._reference_waypoints[env_ids] = routes
            self._reference_point_counts[env_ids] = counts
            self._reference_valid[env_ids] = valid
            self._reference_gates[env_ids] = gates
            self._reference_gate_counts[env_ids] = gate_counts
            # Planned paths are directed from the actual start, including reverse motions.
            self._reference_direction[env_ids] = 1.0
            self._update_reference_visualizations(env_ids, start_local, start_yaw, goal_local)
            return
        routes, point_counts, valid = self._world_manager.reference_routes(
            env_ids,
            self._episode_task_level[env_ids],
            max_waypoints=self._reference_waypoints.shape[1],
        )
        self._reference_waypoints[env_ids] = routes
        self._reference_point_counts[env_ids] = point_counts
        self._reference_valid[env_ids] = valid
        direction = torch.where(
            goal_local[:, 1] >= start_local[:, 1],
            torch.ones_like(goal_local[:, 1]),
            -torch.ones_like(goal_local[:, 1]),
        )
        self._reference_direction[env_ids] = direction
        progress, cross_track = project_points_to_polylines(
            start_local, routes, point_counts
        )
        self._prev_reference_progress[env_ids] = progress
        self._last_reference_progress[env_ids] = progress
        self._last_reference_cross_track[env_ids] = cross_track

    def _update_reference_visualizations(self, env_ids, start_local, start_yaw, goal_local):
        """Render only selected reset results; never invoke planning a second time."""
        cfg = self.cfg
        if not (cfg.reference_path_visualize or cfg.reference_path_export_png):
            return
        from datetime import datetime
        from pathlib import Path
        from .reference_path_visualization import export_reference_png, ReferencePathOverlay
        from .se2_reference_planner import world_from_detour_layout, ordered_detour_gates

        manager = self._world_manager
        selected = tuple(cfg.reference_path_visualize_env_ids)
        if not selected:
            selected = tuple(sorted(manager.dedicated_detour_layouts)[:1])
        invalid = set(selected) - set(manager.dedicated_detour_layouts)
        if invalid:
            raise ValueError(f"reference visualization requires dedicated detour env IDs, got {sorted(invalid)}")
        if cfg.reference_path_export_max_per_env < 0:
            raise ValueError("reference_path_export_max_per_env must be >= 0")
        if not hasattr(self, "_reference_visual_counts"):
            self._reference_visual_counts = {}
            self._reference_visual_output = Path(cfg.reference_path_export_dir) / datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
        # With diagnostics off, no extra CUDA transfers or Matplotlib imports occur.
        for row, env_id in enumerate(env_ids.cpu().tolist()):
            if env_id not in selected:
                continue
            plan = manager.last_reference_plans[env_id]
            start = torch.cat((start_local[row], start_yaw[row:row+1])).cpu().numpy()
            goal = goal_local[row].cpu().numpy()
            level = int(self._episode_task_level[env_id].item())
            layout = manager.dedicated_detour_layouts[env_id]
            gates = ordered_detour_gates(layout, level, start, goal)
            if cfg.reference_path_visualize:
                if not hasattr(self, "_reference_overlay"):
                    import omni.usd
                    self._reference_overlay = ReferencePathOverlay(omni.usd.get_context().get_stage())
                self._reference_overlay.update(
                    env_id, self.scene.env_origins[env_id].cpu().numpy(), plan,
                    start, goal, gates, height=cfg.reference_path_visualize_height)
            count = self._reference_visual_counts.get(env_id, 0)
            limit = cfg.reference_path_export_max_per_env
            if cfg.reference_path_export_png and (limit == 0 or count < limit):
                filename = self._reference_visual_output / f"env_{env_id:04d}_level_{level}_reset_{count+1:04d}.png"
                export_reference_png(filename, world_from_detour_layout(layout, manager.cfg),
                                     plan, start, goal, gates, manager._reference_bank.cfg,
                                     env_id=env_id, level=level)
                self._reference_visual_counts[env_id] = count + 1
                print(f"[ReferencePNG] {filename.resolve()}", flush=True)

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

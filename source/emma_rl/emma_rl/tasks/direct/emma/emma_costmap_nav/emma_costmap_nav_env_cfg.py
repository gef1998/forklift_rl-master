"""Emma Costmap Navigation 环境配置.

与 emma_obs_nav 的关键区别:
  - 观测: costmap (2×64×64) + robot_state (8) 替代原始 LiDAR ranges
  - 场景: 静态 Cube + Cylinder 混合 + 动态 Cylinder
  - 无 LiDAR RayCaster
"""

import math
from dataclasses import dataclass

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.simulation_cfg import PhysxCfg
from isaaclab.utils import configclass

from ..asset_paths import resolve_asset
from .emma_costmap_nav_world import NavWorldConfig

# ---- Asset paths ----
_EMMA_USD = resolve_asset("Emma_no_ground.usd", override_env="EMMA_USD_PATH")


# ============================================================================
# Domain Randomization Config
# ============================================================================
@dataclass
class DomainRandCfg:
    """域随机化配置 — 感知噪声适配 costmap."""

    # ---- 感知噪声 (costmap 层面) ----
    enable_costmap_dropout: bool = True
    costmap_dropout_prob: float = 0.20        # 随机漏检障碍物比例
    costmap_inflation_noise: float = 0.05     # 膨胀半径随机扰动 (m)

    # ---- 动力学噪声 (episode-level) ----
    enable_dynamics_noise: bool = True
    mass_scale_range: tuple = (0.85, 1.15)
    wheel_radius_scale_range: tuple = (0.98, 1.02)
    com_offset_xy_range: tuple = (-0.02, 0.02)
    wheel_slip_noise_std: float = 0.02

    # ---- 控制噪声 (step-level) ----
    enable_action_noise: bool = True
    action_noise_std: float = 0.03
    action_delay_max_steps: int = 2

    # ---- 初始状态噪声 (episode-level) ----
    enable_init_noise: bool = True
    init_pos_noise_xy: float = 0.05
    init_yaw_noise_deg: float = 5.0


# ============================================================================
# EmmaCostmapNavEnvCfg
# ============================================================================
@configclass
class EmmaCostmapNavEnvCfg(DirectRLEnvCfg):
    """Emma AMR 二轮差速 costmap 导航 — 3 阶段课程."""

    # ---- Environment ----
    decimation = 2
    episode_length_s = 60.0
    action_space = 2          # (v_lin_norm, v_ang_norm) ∈ [0,1]
    state_space = 0

    # ---- Observation: costmap(2×64×64) + robot_state(8) = 8200 ----
    costmap_channels: int = 2
    costmap_resolution: int = 64
    costmap_grid_size: float = 12.0          # ±6m 覆盖 (对标 LiDAR 12m 范围)
    robot_state_dim: int = 8
    observation_space = costmap_channels * costmap_resolution * costmap_resolution + robot_state_dim  # 8200

    # ---- Terrain ----
    terrain_sub_size: float = 30.0

    # ---- Physics ----
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 100,
        render_interval=decimation,
        physx=PhysxCfg(
            gpu_found_lost_pairs_capacity=2**25,      # 33M (验证过的值)
            gpu_max_rigid_patch_count=2**21,           # 2M
        ),
    )
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=1024, env_spacing=terrain_sub_size, replicate_physics=True
    )

    # ========================================================================
    # Robot — Emma AMR (与 emma_obs_nav 相同)
    # ========================================================================
    wheel_radius: float = 0.09
    wheel_base: float = 0.52
    wheel_max_velocity: float = 30.0
    max_linear_velocity: float = wheel_max_velocity * wheel_radius       # ≈ 1.8 m/s
    max_angular_velocity: float = (
        2.0 * wheel_max_velocity * wheel_radius / wheel_base)             # ≈ 6.92 rad/s

    robot_half_length: float = 0.42
    robot_half_width: float = 0.267
    robot_height: float = 0.253

    robot_cfg: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=_EMMA_USD,
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True,
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=2,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.045),
            joint_pos={".*": 0.0},
        ),
        actuators={
            "wheel_left_actuator": ImplicitActuatorCfg(
                joint_names_expr=["wheel_left"],
                velocity_limit=wheel_max_velocity,
                velocity_limit_sim=wheel_max_velocity,
                stiffness=None,
                damping=None,
            ),
            "wheel_right_actuator": ImplicitActuatorCfg(
                joint_names_expr=["wheel_right"],
                velocity_limit=wheel_max_velocity,
                velocity_limit_sim=wheel_max_velocity,
                stiffness=None,
                damping=None,
            ),
        },
    )
    dof_names = ("wheel_left", "wheel_right")

    # ========================================================================
    # Goal
    # ========================================================================
    goal_marker_radius: float = 0.2
    goal_reach_dist: float = 0.3
    goal_spawn_half: float = 2.0

    # ========================================================================
    # Obstacle World — 混合障碍物场景
    # ========================================================================
    nav_world: NavWorldConfig = NavWorldConfig(
        arena_size=terrain_sub_size,
        num_static_cubes=20,
        cube_length_range=(2.0, 4.0),
        cube_width_range=(0.4, 0.6),
        cube_height_range=(0.5, 0.6),
        num_static_cylinders=80,
        static_cylinder_radius_range=(0.15, 0.35),
        static_cylinder_height_range=(0.5, 0.6),
        num_dynamic_cylinders=10,
        dynamic_radius_range=(0.15, 0.25),
        dynamic_height_range=(0.5, 0.6),
        dynamic_speed_range=(0.3, 0.8),
        min_obs_spacing=0.9,
        side_zone_width=1.0,
        buffer_width=1.2,
    )

    # ---- Costmap 生成参数 ----
    costmap_inflation: float = 0.25           # 膨胀半径 (m)
    costmap_update_freq: float = 10.0         # costmap 更新频率 (Hz), 模拟真实 LiDAR 10-20Hz
    costmap_ego_footprint: bool = True        # 是否在 costmap 上叠加自车占位 (盲开保护)

    # ---- 域随机化 ----
    domain_rand: DomainRandCfg = DomainRandCfg()

    # ========================================================================
    # Phase 课程 (与 emma_obs_nav 结构一致)
    # ========================================================================
    training_phase: int = 1

    # Phase 1 (稀疏): 静态障碍物密度最低, 无动态障碍物
    phase1_obs_spacing: float = 1.00
    phase1_num_dynamic: int = 0               # Phase 1 无动态障碍物

    # Phase 2 (中等): 引入少量动态障碍物
    phase2_obs_spacing: float = 0.70
    phase2_num_dynamic: int = 5

    # Phase 3 (密集): 全量动态障碍物
    phase3_obs_spacing: float = 0.55
    phase3_num_dynamic: int = 10              # 使用 nav_world 中配置的数量

    # 噪声课程
    phase_steps: int = 150000
    phase1_noise_start: float = 0.1
    phase1_noise_end: float = 0.3
    phase2_noise_start: float = 0.4
    phase2_noise_end: float = 0.6
    phase3_noise_start: float = 0.7
    phase3_noise_end: float = 1.0

    # ========================================================================
    # Reward — 纯导航 (costmap 适配版)
    # ========================================================================
    goal_reach_bonus: float = 50.0
    obstacle_collision_penalty: float = -50.0
    time_penalty: float = -0.2
    action_smooth_scale: float = 0.5
    fall_reset_height: float = -0.25

    # 基于 costmap 的避障采样参数
    nav_costmap_avoid_rays: int = 15           # 在 goal 方向扇区采样的射线数
    nav_costmap_avoid_fov_deg: float = 90.0    # 采样扇区 (±45°)
    nav_costmap_avoid_cost_threshold: float = 0.5  # 高代价阈值

    # Observation normalization
    goal_max_dist: float = 15.0
    lin_vel_scale: float = 1.0
    ang_vel_scale: float = 2.0

    # ---- Phase 1 reward scales ----
    phase1_nav_vel_dir_scale: float = 0.0
    phase1_nav_heading_scale: float = 0.0
    phase1_nav_approach_scale: float = 5.0
    phase1_nav_costmap_avoid_scale: float = 8.0
    phase1_nav_backtrack_scale: float = 0.3
    phase1_nav_spin_scale: float = 0.03
    phase1_nav_under_speed_scale: float = 0.5
    phase1_nav_stuck_scale: float = 0.03

    # ---- Phase 2 reward scales ----
    phase2_nav_vel_dir_scale: float = 22.0
    phase2_nav_heading_scale: float = 3.0
    phase2_nav_approach_scale: float = 3.0
    phase2_nav_costmap_avoid_scale: float = 10.0
    phase2_nav_backtrack_scale: float = 0.3
    phase2_nav_spin_scale: float = 0.03
    phase2_nav_under_speed_scale: float = 0.08
    phase2_nav_stuck_scale: float = 0.05

    # ---- Phase 3 reward scales ----
    phase3_nav_vel_dir_scale: float = 20.0
    phase3_nav_heading_scale: float = 5.0
    phase3_nav_approach_scale: float = 5.0
    phase3_nav_costmap_avoid_scale: float = 10.0
    phase3_nav_backtrack_scale: float = 0.3
    phase3_nav_spin_scale: float = 0.03
    phase3_nav_under_speed_scale: float = 0.1
    phase3_nav_stuck_scale: float = 0.05

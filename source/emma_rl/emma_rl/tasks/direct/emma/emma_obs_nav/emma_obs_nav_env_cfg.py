"""Configuration for EmmaObsNavEnv — Emma AMR 二轮差速避障导航.

场景架构:
  /World/groundPlane               ← GroundPlane 平地
  /World/envs/env_.*/Robot         ← Articulation（Emma 差速 AMR）
  /World/envs/env_.*/GoalMarker    ← 目标点标记球
  /World/envs/env_.*/Obstacles/    ← Cylinder Mesh RigidObjects（障碍物）

Emma 运动学约定:
  - Emma 车身: body +X=前进, 占位 823×532mm 以原点为几何中心
  - 二轮差速驱动: action=(v_lin, v_ang), 通过轮径/轴距转为 wheel velocity
  - 无 steer 关节, 无 fork 坐标系转换
"""

import math
from dataclasses import dataclass

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.simulation_cfg import PhysxCfg
from isaaclab.sensors.ray_caster import RayCasterCfg
from isaaclab.sensors.ray_caster.patterns import BpearlPatternCfg
from isaaclab.utils import configclass

from ..asset_paths import resolve_asset
from .emma_obs_nav_random_obstacle_world import RandomObstacleWorldConfig
from .emma_obs_nav_world import ObstacleWorldConfig
from .emma_obs_nav_warehouse_world import WarehouseWorldConfig


# ====================================================================
# Domain Randomization Config
# ====================================================================
@dataclass
class DomainRandCfg:
    """域随机化配置 — 各维度可独立开关, 配合 phase 噪声课程使用."""

    # ---- 感知噪声 (step-level, 加在 raw LiDAR distance 上, min-pool 前) ----
    enable_lidar_noise: bool = True
    lidar_noise_std: float = 0.05            # 高斯噪声 std (m)
    enable_lidar_dropout: bool = True
    lidar_dropout_prob: float = 0.02         # 每扇区丢帧概率 (min-pool 后)

    # ---- 动力学噪声 (episode-level, _reset_idx 中采样) ----
    enable_dynamics_noise: bool = True
    mass_scale_range: tuple = (0.85, 1.15)           # 车身质量缩放 ±15%
    wheel_radius_scale_range: tuple = (0.98, 1.02)    # 轮径缩放 ±2%
    com_offset_xy_range: tuple = (-0.02, 0.02)       # 质心偏移 ±2cm
    wheel_slip_noise_std: float = 0.02                # 轮速滑移噪声 (摩擦 proxy)

    # ---- 控制噪声 (step-level, _pre_physics_step 中) ----
    enable_action_noise: bool = True
    action_noise_std: float = 0.06              # signed action noise; matches old 0.03 in [0,1]
    action_delay_max_steps: int = 10             # 最大延迟 (decimation steps, 0~40ms)

    # ---- 初始状态噪声 (episode-level) ----
    enable_init_noise: bool = True
    init_pos_noise_xy: float = 0.05             # ±5cm
    init_yaw_noise_deg: float = 5.0             # ±5°


@dataclass
class SafetyControlCfg:
    """Always-on runtime safety projection parameters.

    Defaults target the current static-obstacle training world. Dynamic people
    must use a separate non-zero obstacle-speed bound after their geometry and
    timestamps are represented in the shield observation.
    """

    # Experimental ablation only. Keep True for deployment and safety runs.
    # False bypasses S1/S2/S3, but wheel feasibility projection remains active.
    enabled: bool = False
    lateral_accel_limit: float = 0.8          # m/s², simultaneous v/omega limit
    braking_deceleration: float = 1.0         # m/s² usable guaranteed deceleration
    controller_latency_s: float = 0.06        # shield compute/transport latency
    actuator_brake_build_up_s: float = 0.10   # command-to-effective-braking latency
    static_margin: float = 0.30               # extra clearance outside robot footprint
    possible_obstacle_speed: float = 0.0      # current ObsNav obstacles are static
    forward_half_angle_deg: float = 45.0
    rear_half_angle_deg: float = 45.0
    ttc_slow_s: float = 1.5
    ttc_stop_s: float = 0.4
    max_lidar_aoi_s: float = 0.20             # fail closed when last scan is older
    intervention_penalty_scale: float = 1.0   # teach policy not to lean on shield

# ---- Asset paths ----
_EMMA_USD = resolve_asset("Emma_no_ground.usd", override_env="EMMA_USD_PATH")


@configclass
class EmmaObsNavEnvCfg(DirectRLEnvCfg):
    """Emma AMR 二轮差速避障导航 — 3 阶段课程."""

    # ---- Environment ----
    decimation = 2
    # 早期策略速度较低，短墙绕行仍需留出探索和纠错时间。
    episode_length_s = 60.0
    action_space = 2
    # Both modes consume signed actions in [-1, 1]^2:
    #   body_twist:    (normalized linear velocity, normalized angular velocity)
    #   wheel_velocity:(normalized left wheel speed, normalized right wheel speed)
    action_control_mode: str = "body_twist" # "body_twist" or "wheel_velocity"
    state_space = 0

    # ---- Observation: 120 LiDAR + 2 goal_dir + 1 goal_dist + 3 IMU + 8 time = 134 ----
    time_encoding_dim: int = 8       # richer AoI encoding for async LiDAR
    single_obs_dim: int = 134        # 120 + 2 + 1 + 2 + 1 + 8 = 134
    frame_stack: int = 1
    observation_space = single_obs_dim * frame_stack  # 134
    # "initial" 保留原任务的初始距离动态归一化；"lidar_max" 使用固定
    # LiDAR 量程归一化，适合 MPPI 专家监督，避免相同观测比例对应不同米制距离。
    goal_distance_normalization: str = "initial" 

    # ---- Terrain ----
    # Runtime value is synchronized from the selected world's arena_size.
    # random_obstacles/ring 默认 10 m，warehouse 默认 16 m。
    terrain_sub_size: float = 10.0

    # ---- World mode: "random_obstacles"、"ring" 或 "warehouse" ----
    world_mode: str = "random_obstacles"

    sim: SimulationCfg = SimulationCfg(
        dt=1 / 100,
        render_interval=decimation,
        physx=PhysxCfg(
            gpu_found_lost_pairs_capacity=2**27,           # 2M → 33M (实际报错需 ~22M)
            gpu_max_rigid_patch_count=2**21,                # 164K → 2M (实际报错需 ~966K)
        ),
    )
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=1024, env_spacing=terrain_sub_size, replicate_physics=True
    )

    # ====================================================================
    # Robot — Emma AMR 二轮差速小车
    # ====================================================================
    #   资产: Emma_no_ground.usd
    #   坐标系: body +X = 前进方向, +Y = 左侧, +Z = 上
    #   DOF: wheel_left(0), wheel_right(1), lift_joint(2, 不使用)
    #
    #   占位 (矩形, 原点=几何中心):
    #     全长 823mm → half_length = 0.4115m
    #     全宽 532mm → half_width  = 0.266m
    #     车高 253mm → lidar 安装在车顶上方
    #
    #   运动学 (二轮差速):
    #     v_lin  = (ω_L + ω_R) / 2 × wheel_radius
    #     v_ang  = (ω_R - ω_L) / wheel_base × wheel_radius
    #     wheel_radius = 0.09 m  (当前 USD 标称值，待实车底盘参数复核)
    #     wheel_base   = 0.40 m  (左右驱动轮横向中心距，不是 0.52 m 车体总宽)
    # ====================================================================
    wheel_radius: float = 0.09             # m
    wheel_base: float = 0.40               # m，实际语义为 track width
    wheel_max_velocity: float = 10.0       # rad/s
    # 普通橡胶轮/混凝土地面的各向同性近似。过高摩擦会让差速转向出现粘滑。
    ground_static_friction: float = 0.8
    ground_dynamic_friction: float = 0.6
    # True：运行时根据上面三项同步直行/原地旋转极限，避免运动学参数漂移。
    # False：允许设置更低的运营速度限制，例如实车限速 0.2 m/s、0.2 rad/s。
    sync_body_velocity_limits_to_wheels: bool = False
    max_linear_velocity: float = 0.9
    max_angular_velocity: float = 2.0

    # 车身占位 (矩形, 原点=几何中心, body +X=前)
    robot_half_length: float = 0.42        # 0.4115m physical half-length + safety margin
    robot_half_width: float = 0.267        # 532mm / 2
    robot_height: float = 0.253            # 车身高度

    robot_cfg: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",   # spawn 路径 (USD 内含 EMMA400K_body articulation)
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
                stiffness=0.0,
                damping=4.05,
            ),
            "wheel_right_actuator": ImplicitActuatorCfg(
                joint_names_expr=["wheel_right"],
                velocity_limit=wheel_max_velocity,
                velocity_limit_sim=wheel_max_velocity,
                stiffness=0.0,
                damping=4.05,
            ),
            # "lift_joint_actuator": ImplicitActuatorCfg(
            #     joint_names_expr=["lift_joint"],
            #     velocity_limit=0.0,           # 锁死
            #     velocity_limit_sim=0.0,
            #     stiffness=None,
            #     damping=None,
            # ),
    # lift_joint 不在 dof_names 中, 由 init_state joint_pos 固定为 0
        },
    )
    dof_names = ("wheel_left", "wheel_right")

    # ====================================================================
    # Goal
    # ====================================================================
    goal_marker_radius: float = 0.2
    goal_reach_dist: float = 0.3            # ICR 到目标 < 此距离 → 成功
    goal_spawn_half: float = 2.0            # 目标点生成范围 ±2m

    # ====================================================================
    # Random-obstacle world（课程第一阶段）
    #   10 m 封闭场景内均匀布置圆柱和方柱；障碍物几何边缘净距至少
    #   1.5 m。Emma/Goal 联合安全采样，直线距离严格大于 5 m。
    # ====================================================================
    random_obstacle_world: RandomObstacleWorldConfig = RandomObstacleWorldConfig(
        arena_size=10.0,
        wall_thickness=0.15,
        wall_height=1.0,
        num_obstacles=15,
        radius_min=0.20,
        radius_max=0.50,
        cube_size_min=0.30,
        cube_size_max=0.50,
        obstacle_height_min=1.0,
        obstacle_height_max=1.2,
        min_obstacle_spacing=1.50,
        obstacle_wall_clearance=0.30,
        emma_spawn_clearance_radius=0.70,
        goal_spawn_clearance_radius=0.50,
        goal_min_distance=5.20,
        obstacle_candidate_count=32,
        spawn_candidate_count=96,
    )

    # ====================================================================
    # Ring world（10 m × 10 m，保留旧任务名）
    #   四段错位内墙形成绕行路径，外墙到平行内墙的净宽为 2.0 m。
    #   Emma/Goal 位于不同的安全角落，混合障碍避开出生区和关键开口。
    # ====================================================================
    obs_world: ObstacleWorldConfig = ObstacleWorldConfig(
        arena_size=10.0,
        num_obstacles=20,          # 模板总数 = max(phase*_num_obstacles)
        radius_min=0.10,
        radius_max=0.15,
        cube_size_min=0.10,
        cube_size_max=0.20,
        height_min=1.0,
        height_max=1.2,
        min_obs_spacing=0.60,
        wall_thickness=0.15,
        interior_wall_thickness=0.15,
        wall_height=1.0,
        corridor_width=2.5,
        # 内墙长度 = arena_size / 4 + wall_tip_offset = 3.5 m。
        wall_tip_offset=0.8,
        obstacle_wall_clearance=0.05,
        passage_side_clearance=0.30,
        # 0.534 m 车宽 + 两侧各约 0.30 m 安全余量。
        min_passage_width=1.15,
        spawn_exclusion_radius=1.20,
        gate_exclusion_radius=0.75,
        sampling_candidate_count=24,
        spawn_jitter=0.15,
        goal_min_distance=4.5,
    )

    # ---- 仓库世界 (world_mode="warehouse", 16 m × 16 m) ----
    # 这是 0.9m/s 下的局部导航场景：目标直线距离 5~10m，满速直达约 5.6~11.1s；
    # 80s 回合为绕货架、短时远离目标和脱困保留足够时间。
    # 布局: num_shelf_cols 列 × shelves_per_col 段，列间/段间均保留可通行通道。
    warehouse_world: WarehouseWorldConfig = WarehouseWorldConfig(
        arena_size=16.0,
        wall_thickness=0.4,
        num_shelf_cols=3,
        shelves_per_col=5,
        shelf_length=3.0,
        shelf_width=1.0,
        column_gap=2.2,
        shelf_row_gap_min=2.2,
        shelf_row_gap_max=2.4,
        robot_spawn_margin=0.55,       # 机器人半对角, spawn 膨胀
        emma_goal_min_dist=5.0,         # 满速直达至少约 5.6s
        emma_goal_max_dist=10.0,        # 满速直达至多约 11.1s
    )

    # ---- Phase 圆柱数 (warehouse mode) ----
    # 16m 场地已包含 15 段货架；圆柱按课程逐步增加，避免沿用 30m 场地数量造成拥塞。
    phase1_num_cylinders: int = 8       # 约 1cyl / 32m²
    phase2_num_cylinders: int = 12      # 约 1cyl / 21m²
    phase3_num_cylinders: int = 16      # 约 1cyl / 16m²

    # ---- 域随机化 ----
    domain_rand: DomainRandCfg = DomainRandCfg()

    # ---- Always-on runtime safety control ----
    safety_control: SafetyControlCfg = SafetyControlCfg()

    # ====================================================================
    # LiDAR — 0.25° 射线 + 3° 扇区 min-pool (360° FOV)
    #   1440 rays @ 0.25° → reshape → (120, 12) → min → 120 robust sectors
    #   噪声加在原始距离上 (模拟传感器误差), min-pool 后再做 dropout
    # ====================================================================
    num_lidar_rays: int = 1440                  # 内部射线数 (360° / 0.25°)
    lidar_output_sectors: int = 120              # 输出扇区数 (3° × 120 = 360°)
    lidar_rays_per_sector: int = 12              # 3° / 0.25° = 12
    lidar_max_distance: float = 20.0
    lidar_height: float = 0.12                  # LiDAR 安装高度
    lidar_offset_x: float = 0.38                # 偏前 (body +X=前进方向)
    lidar_offset_y: float = 0.0
    # 策略观测的有效视场。训练默认保持完整 360°；play.py 会覆盖为
    # 前向 ±110°，并将其余盲区填成固定 clearance。
    lidar_observation_forward_half_fov_deg: float = 180.0
    lidar_observation_blind_fill_clearance: float = 1.0
    debug_lidar_fov_fill: bool = False

    # ---- Asynchronous sensor scheduler (sim-to-real latency / jitter / dropout) ----
    # "random": sample a frequency independently per env — policy learns to
    #           interpret AoI and degrade gracefully with stale observations.
    lidar_sensor_mode: str = "fixed"
    lidar_fixed_frequency_hz: float = 25.0
    lidar_frequency_choices_hz: tuple[float, ...] = (10.0, 12.5, 16.6667, 20.0, 25.0, 50.0)
    lidar_latency_range_s: tuple[float, float] = (0.01, 0.06)   # 10–60 ms transport latency
    lidar_jitter_max_steps: int = 2                              # ±2 steps (0–40 ms) random jitter
    lidar_scan_dropout_prob: float = 0.15                        # 15 % of scans silently dropped
    lidar_pending_queue_size: int = 16
    lidar_warp_enabled: bool = True
    lidar_aoi_enabled: bool = True
    lidar_aoi_encoding_scale_s: float = 0.3                      # longer encoding period for richer AoI

    lidar: RayCasterCfg = RayCasterCfg(
        prim_path="/World/envs/env_.*/Robot/EMMA400K_body",
        offset=RayCasterCfg.OffsetCfg(
            pos=(lidar_offset_x, lidar_offset_y, lidar_height),
        ),
        ray_alignment="base",
        pattern_cfg=BpearlPatternCfg(
            horizontal_fov=360.0,
            horizontal_res=0.25,                 # 0.25° 精度 (真实 LiDAR 级别)
            vertical_ray_angles=[0.0],
        ),
        mesh_prim_paths=["/World/groundPlane"],
        max_distance=lidar_max_distance,
        debug_vis=False,
    )

    # ====================================================================
    # Phase control (3 阶段课程)
    # ====================================================================
    training_phase: int = 1

    # ---- Phase 1 (稀疏): 学习四角任务、内墙绕行和基础避障 ----
    phase1_obs_spacing: float = 0.90
    phase1_num_obstacles: int = 4

    # ---- Phase 2 (中等): 走廊和中央区域加入更多混合障碍 ----
    phase2_obs_spacing: float = 0.60
    phase2_num_obstacles: int = 12

    # ---- Phase 3 (密集): 保持关键开口可通行，提高布局多样性 ----
    phase3_obs_spacing: float = 0.40
    phase3_num_obstacles: int = 20

    # ---- 噪声课程: 每 phase 内线性 ramp ----
    #       Phase 1: noise 0.0 → 0.3
    #       Phase 2: noise 0.4 → 0.6
    #       Phase 3: noise 0.7 → 1.0
    phase_steps: int = 1200000                 # 每 phase 的 step 数 (用于计算 ramp progress)
    phase1_noise_start: float = 0.1
    phase1_noise_end: float = 0.3
    phase2_noise_start: float = 0.4
    phase2_noise_end: float = 0.6
    phase3_noise_start: float = 0.7
    phase3_noise_end: float = 1.0
    
    # ====================================================================
    # Reward — 简化的局部导航
    # ====================================================================
    # 典型 5 m 回合：单边 Goal 势场只作辅助，到达 +25，碰撞 -30。
    goal_reach_bonus: float = 25.0
    obstacle_collision_penalty: float = -30.0
    collision_clearance_threshold: float = 0.05
    # 50 Hz 下约 -0.1875/s；80 s 完整 timeout 的累计量仍约为 -15。
    time_penalty: float = -0.003
    fall_reset_height: float = -0.25

    # VFH 只负责生成参考航向，策略仍直接输出线速度和角速度。
    # 远场航路阻塞：3.5m 内视为明确阻塞，3.5~4.0m 使用有限 smoothstep
    # 过渡。它只负责提前发现墙和入口，不扩大 repulse 的近场惩罚范围。
    nav_vfh_route_lookahead: float = 4.0
    nav_vfh_route_transition: float = 0.5
    nav_vfh_smoothing_radius_bins: int = 2
    nav_vfh_blocked_threshold: float = 0.55
    # Goal 扇区畅通时优先直达；被阻挡时降低 Goal 偏好并提高开阔度权重。
    nav_vfh_goal_cost: float = 1.0
    nav_vfh_obstacle_cost: float = 1.5
    nav_vfh_blocked_goal_cost: float = 1.0
    nav_vfh_blocked_obstacle_cost: float = 4.0
    nav_vfh_continuity_cost: float = 0.25
    nav_vfh_switch_margin: float = 0.05
    # 2rad/s≈115deg/s；参考航向只略快于车体，避免奖励目标移动得无法跟随。
    nav_vfh_max_reference_rate_deg_s: float = 120.0
    # 左右各15°窗口评价通道宽度与深度；阻塞时开阔通道可以压过 Goal 捷径。
    nav_vfh_valley_half_width_deg: float = 15.0
    nav_vfh_valley_width_cost: float = 0.8
    nav_vfh_valley_depth_cost: float = 0.6
    # 跟踪上一通道而非冻结旧角度；新通道需跨出该邻域并明显更优才切换。
    nav_vfh_valley_keep_half_angle_deg: float = 40.0
    # Goal 阻塞权重的一阶时间常数，50Hz 下约15步完成主要切换。
    nav_vfh_blockage_time_constant_s: float = 0.30
    # VFH+ 物理通道过滤：0.534m 车宽 + 两侧约 0.30m 安全余量。
    nav_vfh_min_passage_width: float = 1.0
    nav_vfh_passage_lookahead: float = 2.0

    # 运动奖励直接按每个 50 Hz 控制步给出，避免低速时物理积分量过小。
    # Goal progress 只作弱辅助；主要教师信号来自正确转向和安全前进。
    nav_progress_scale_per_step: float = 0.05
    # 参考角达到 90°、并以最大正确角速度旋转时，每步获得完整转向奖励。
    nav_turn_scale_per_step: float = 0.15
    nav_turn_alignment_deadband_deg: float = 6.0
    # 以最大前向速度沿 VFH 参考航向行驶时，每步获得完整前进奖励。
    nav_safe_forward_scale_per_step: float = 0.5
    # 5° 内保持完整奖励，5°~45° 用有限 smoothstep 衰减；45° 外严格为零。
    # 该窗口只参与训练奖励和停滞判定，不会修改、裁剪或投影策略 action。
    nav_safe_forward_full_reward_angle_deg: float = 5.0
    nav_safe_forward_cutoff_angle_deg: float = 60.0
    # 既不正确转向、也不沿参考航向前进时的逐步停滞惩罚。
    nav_idle_scale_per_step: float = 0.03
    # 1.2m 内使用有界反比例斥力；正前方 0.45m 满速行驶时净奖励约为零。
    nav_repulse_start_clearance: float = 1.2
    nav_repulse_scale_per_step: float = 0.7
    # 保证侧面 0.3m 贴障行驶仍为负收益，而 0.5m 侧向净空保持可通行。
    nav_repulse_lateral_weight_floor: float = 0.65
    # 保留矩形车体旋转扫碰约束。
    nav_repulse_rotation_weight: float = 0.15
    # 近障静止仍承担小幅距离风险，避免“停车只付 time penalty”的局部最优。
    nav_repulse_motion_floor: float = 0.15
    nav_backward_scale_per_step: float = 0.05

    # 策略原始归一化 action 的一阶差分：线速度允许正常起停，角速度重点抑制抖动。
    action_linear_jitter_scale: float = 0.001
    action_angular_jitter_scale: float = 0.003

    # goal_dist 使用初始 spawn 距离归一化（绕行时允许大于 1）；
    # IMU 使用物理速度上限归一化到约 [-1, 1]。

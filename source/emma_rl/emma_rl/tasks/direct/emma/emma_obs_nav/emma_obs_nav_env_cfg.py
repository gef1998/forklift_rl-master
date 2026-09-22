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
from .emma_obs_nav_corridor_world import CorridorWorldConfig
from .emma_obs_nav_random_obstacle_world import RandomObstacleWorldConfig
from .emma_obs_nav_straight_obstacle_world import StraightObstacleWorldConfig
from .emma_obs_nav_world import ObstacleWorldConfig
from .navigation_curriculum import NavigationCurriculumCfg
from .emma_obs_nav_warehouse_world import WarehouseWorldConfig


# ====================================================================
# Domain Randomization Config
# ====================================================================
@dataclass
class DomainRandCfg:
    """域随机化配置 — 各维度可独立开关，倍率由每回合的课程等级决定。"""

    # ---- 感知噪声 (step-level, 加在 raw LiDAR distance 上, min-pool 前) ----
    enable_lidar_noise: bool = True
    lidar_noise_std: float = 0.04            # 高斯噪声 std (m)
    enable_lidar_dropout: bool = True
    lidar_dropout_prob: float = 0.02         # 每扇区丢帧概率 (min-pool 后)

    # ---- 动力学噪声 (episode-level, _reset_idx 中采样) ----
    enable_dynamics_noise: bool = True
    mass_scale_range: tuple = (0.85, 1.15)           # 车身质量缩放 ±15%
    wheel_radius_scale_range: tuple = (0.99, 1.01)    # 轮径缩放 ±2%
    com_offset_xy_range: tuple = (-0.01, 0.01)       # 质心偏移 ±2cm
    wheel_slip_noise_std: float = 0.01                # 轮速滑移噪声 (摩擦 proxy)

    # ---- 控制噪声 (step-level, _pre_physics_step 中) ----
    enable_action_noise: bool = True
    action_noise_std: float = 0.06              # signed action noise; matches old 0.03 in [0,1]
    action_delay_max_steps: int = 0            # 
    # ---- 初始状态噪声 (episode-level) ----
    enable_init_noise: bool = True
    init_pos_noise_xy: float = 0.025             # ±5cm
    init_yaw_noise_deg: float = 2.5            # ±5°


@dataclass
class SafetyControlCfg:
    """Simple forward-clearance speed governor.

    Clearance is measured outside the rectangular robot footprint. Only
    positive linear velocity is scaled; angular and reverse commands remain
    available so the policy can turn away from a wall.
    """

    enabled: bool = True
    forward_half_angle_deg: float = 30.0
    stop_clearance: float = 0.3
    full_speed_clearance: float = 0.9
    min_forward_speed_scale: float = 0.2
    max_lidar_aoi_s: float = 0.30

# ---- Asset paths ----
_EMMA_USD = resolve_asset("Emma_no_ground.usd", override_env="EMMA_USD_PATH")


@configclass
class EmmaObsNavEnvCfg(DirectRLEnvCfg):
    """Emma AMR 二轮差速避障导航 — 六级自适应任务与噪声课程。"""

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

    # ---- Observation: 120 LiDAR + 2 goal_dir + 1 goal_dist + 3 IMU + 8 time + 2 prev_action = 136 ----
    time_encoding_dim: int = 8       # richer AoI encoding for async LiDAR
    single_obs_dim: int = 136        # 120 + 2 + 1 + 2 + 1 + 8 + 2 = 136
    frame_stack: int = 1
    observation_space = single_obs_dim * frame_stack  # 136
    # "initial" 保留原任务的初始距离动态归一化；"lidar_max" 使用固定
    # LiDAR 量程归一化，适合 MPPI 专家监督，避免相同观测比例对应不同米制距离。
    goal_distance_normalization: str = "lidar_max"

    # ---- Terrain ----
    # Runtime value is synchronized from the selected world's arena_size.
    # random_obstacles/ring 默认 10 m，warehouse 默认 16 m。
    terrain_sub_size: float = 10.0

    # ---- World mode: random_obstacles, ring, corridor, straight_obstacle, warehouse ----
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
    #     wheel_radius = 0.07 m  (当前 USD 标称值)
    #     wheel_base   = 0.43 m  (左右驱动轮横向中心距，不是 0.52 m 车体总宽)
    # ====================================================================
    wheel_radius: float = 0.07            # m
    wheel_base: float = 0.43               # m，实际语义为 track width
    wheel_max_velocity: float = 20.0       # rad/s
    # Effective material pairs use min on both sides: drive 0.8/0.6,
    # chassis 0.5/0.5, caster 0.02/0.02. Use PhysxMaterialAPI, not bare attrs.
    ground_static_friction: float = 0.5
    ground_dynamic_friction: float = 0.5
    # 驱动轮材质。PhysX 用接触双方材质合成摩擦系数; 轮子若没有材质就退化成
    # PhysX 默认值 (~0.5), 低于配置值, 所以两侧都必须显式给材质。
    wheel_static_friction: float = 1.2
    wheel_dynamic_friction: float = 1.0
    # Chassis contact is retained for tipped configurations.
    base_static_friction: float = 0.5
    base_dynamic_friction: float = 0.5
    # Fixed-heading passive rollers retain lateral friction. Navigation proxies
    # bias support toward the drive wheels. See the report for legacy sweeps.
    caster_static_friction: float = 0.02
    caster_dynamic_friction: float = 0.02
    caster_contacts_enabled: bool = True

    # ---- Drive/contact model (see docs/emma_drive_diagnostics.md) ----
    # Separate collision proxies preserve visible tires and link inertias.
    # Navigation approximation, not a calibrated tire/suspension model.
    # Legacy reproduction: cylinder, None, False, acceleration limits disabled.
    drive_wheel_collision_shape: str = "sphere"
    caster_collision_radius: float | None = 0.0395
    drive_enable_external_forces_every_iteration: bool = True

    # Implicit velocity drive: reported wheel speed can differ from joint angle
    # increments. Do not infer actual motor torque from damping * error.
    wheel_velocity_damping: float = 200.0
    wheel_effort_limit: float = 6.0
    # Reflected wheel-joint inertia. Joint-response sweeps showed that
    # 0.03 kg*m^2 lets TGS 4/2 reproduce the measured actuator lag after the
    # drive-wheel radius was increased to 0.07 m.
    wheel_armature: float = 0.03

    # Nominal command slew limits; safety braking retains priority.
    # Substeps interpolate commands. These are not hard bounds on measured
    # acceleration during collision impulses or initial settling.
    enable_motion_acceleration_limits: bool = True
    # Exact first-order low-pass on nominal wheel commands, before the slew
    # limiter. tau=0.06 s gives alpha=0.283 at the 50 Hz control rate.
    enable_motion_command_low_pass: bool = True
    motion_command_low_pass_tau_s: float = 0.06
    max_linear_acceleration: float = 1.0   # m/s^2
    max_angular_acceleration: float = 1.5  # rad/s^2
    # Command headroom for measured tracking overshoot (settled start/reversal).
    # Keep the hardware specifications above distinct from this calibration.
    linear_acceleration_command_scale: float = 0.90
    angular_acceleration_command_scale: float = 0.94

    sync_body_velocity_limits_to_wheels: bool = False
    max_linear_velocity: float = 0.7
    max_angular_velocity: float = 1.5

    # 车身占位 (矩形, 原点=几何中心, body +X=前)
    robot_half_length: float = 0.425        # 0.4115m physical half-length + safety margin
    robot_half_width: float = 0.275        # 532mm / 2
    robot_height: float = 0.253            # 车身高度

    robot_cfg: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",   # spawn 路径 (USD 内含 EMMA400K_body articulation)
        spawn=sim_utils.UsdFileCfg(
            usd_path=_EMMA_USD,
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True,
                # Armature 0.03 conditions the 0.07 m wheel/contact response
                # for the 4/2 budget. See the six-profile drive validation.
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
                damping=wheel_velocity_damping,
                armature=wheel_armature,
                # 隐式执行器上 effort_limit_sim 与 effort_limit 等价, 用前者。
                effort_limit_sim=wheel_effort_limit,
            ),
            "wheel_right_actuator": ImplicitActuatorCfg(
                joint_names_expr=["wheel_right"],
                velocity_limit=wheel_max_velocity,
                velocity_limit_sim=wheel_max_velocity,
                stiffness=0.0,
                damping=wheel_velocity_damping,
                armature=wheel_armature,
                effort_limit_sim=wheel_effort_limit,
            ),
            # Lift and caster joints retain their USD properties; only the
            # two drive wheels are controlled by these actuator groups.
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
    # Random-obstacle world（六级任务与噪声课程）
    #   同一生成分布：随机连通内墙图 + 随机回环 + 圆柱/方柱散障。
    #   自然组合 S 型、回型和直行绕障，不切换固定场景模板。
    #   散障间净距 >= 0.6 m；起终点位于膨胀后的同一连通域。
    #   课程：短距直达 → 中距绕障 → 长距 → 长距强噪声
    #         → 静态密集区长距 → 静态超密集区长距。
    #   每个并行环境初始化独立地图，reset 从该地图的安全点对库采样。
    # ====================================================================
    random_obstacle_world: RandomObstacleWorldConfig = RandomObstacleWorldConfig(
        arena_size=10.0,
        wall_thickness=0.15,
        wall_height=1.0,
        num_obstacles=20,
        radius_min=0.20,
        radius_max=0.40,
        cube_size_min=0.30,
        cube_size_max=0.45,
        obstacle_height_min=1.0,
        obstacle_height_max=1.2,
        min_obstacle_spacing=0.60,
        obstacle_wall_clearance=0.30,
        emma_spawn_clearance_radius=0.70,
        goal_spawn_clearance_radius=0.50,
        goal_min_distance=5.20,
        obstacle_candidate_count=32,
        spawn_candidate_count=96,
        structured_layout=True,       # False 可恢复旧的纯散障生成器
        grid_size=3,                  # 3×3 通道图，12 个整墙槽位 + 4 个窄门短墙槽位
        interior_wall_thickness=0.15,
        wall_jitter=0.20,              # 内墙法向独立扰动，单位 m
        loop_probability_range=(0.0, 0.40),
        clutter_density_range=(0.65, 0.90),  # 高密度静态散障；连通性仍逐图验证
        navigation_clearance=0.52,     # 覆盖当前车身半对角线与碰撞余量
        connectivity_grid_resolution=0.15,
        spawn_bank_size=128,
        max_layout_attempts=64,
        # 在同一随机墙图中把部分关键开边收窄；不是切换固定场景模板。
        narrow_passage_probability=0.85,
        narrow_passage_width_range=(1.20, 1.50),
        max_narrow_passages=2,
        narrow_pair_fraction=0.50,    # 含关键窄门地图的长任务中，50% 强制跨门
        dense_clutter_influence_radius=1.25,  # 第五级优先选择散障密集区路线
        ultra_dense_clutter_influence_radius=0.70,  # 第六级聚焦更局部的拥挤路线
        min_connected_free_fraction=0.60,  # 所有课程端点仍位于最大连通域
    )

    # ====================================================================
    # Ring world（10 m × 10 m，保留旧任务名）
    #   四段错位内墙形成绕行路径，外墙到平行内墙的净宽为 2.0 m。
    #   Emma/Goal 位于不同的安全角落，混合障碍避开出生区和关键开口。
    # ====================================================================
    obs_world: ObstacleWorldConfig = ObstacleWorldConfig(
        arena_size=10.0,
        num_obstacles=4,          # 非课程评估场景，障碍数量显式固定
        radius_min=0.10,
        radius_max=0.15,
        cube_size_min=0.10,
        cube_size_max=0.20,
        height_min=1.0,
        height_max=1.2,
        min_obs_spacing=0.90,
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

    corridor_world: CorridorWorldConfig = CorridorWorldConfig(
        arena_size=10.0,
        wall_thickness=0.15,
        interior_wall_thickness=0.20,
        wall_height=1.0,
        inner_wall_offset=1.50,
        gate_width=2.50,
        spawn_wall_margin=1.0,
        spawn_jitter_x=0.75,
        spawn_jitter_y=0.20,
        goal_min_distance=7.0,
        mirror_across_envs=True,
    )

    straight_obstacle_world: StraightObstacleWorldConfig = (
        StraightObstacleWorldConfig(
            arena_length=6.0,
            passage_width=1.50,
            wall_thickness=0.15,
            wall_height=1.0,
            obstacle_length=0.1,
            obstacle_width=1.20,
            obstacle_height=1.0,
            obstacle_type="cuboid",
            spawn_wall_margin=1.0,
            robot_half_width=robot_half_width,
        )
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

    # ---- 固定圆柱数 (warehouse mode) ----
    # 16m 场地已包含 15 段货架；使用固定圆柱数量，回合中不改动地图。
    warehouse_num_cylinders: int = 8

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
    lidar_offset_x: float = 0.0               # 偏前 (body +X=前进方向)
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
            horizontal_fov=360.0,                # 0°为正前方，逆时针转
            horizontal_res=0.25,                 # 0.25° 精度 (真实 LiDAR 级别)
            vertical_ray_angles=[0.0],
        ),
        mesh_prim_paths=["/World/groundPlane"],
        max_distance=lidar_max_distance,
        debug_vis=False,
    )

    # ====================================================================
    # 根据表现升降级；仅 structured random_obstacles 启用自适应任务课程。
    # 地图保持固定；新回合选择难度与噪声，旧回合不受升级影响。
    # ====================================================================
    curriculum: NavigationCurriculumCfg = NavigationCurriculumCfg()

    # ====================================================================
    # Reward — 直接面向任务的六项奖励，无 VFH 教师
    # ====================================================================
    # 有正负的目标进度、近障、动作变化、时间、到达和失败。
    goal_reach_bonus: float = 25.0
    nav_failure_penalty: float = -30.0  # collision, fall or leaving the arena
    collision_clearance_threshold: float = 0.01
    fall_reset_height: float = -0.25

    # Signed change in actual goal distance; reward units per metre.
    # Moving away is negative, revisiting the same distance cannot repeatedly
    # collect the old positive-only progress reward (undiscounted transitions).
    nav_progress_per_meter: float = 5.0
    # Running costs use seconds, independent of the control frequency.
    # Clearance is measured OUTSIDE the rectangular body, on a current clean scan.
    # Only a narrow near-collision band is penalized, preserving usable corridors.
    nav_clearance_threshold: float = 0.05
    nav_proximity_cost_per_second: float = 0.25
    nav_time_cost_per_second: float = 0.75

    # One weak regularizer on changes in raw normalized policy actions.
    # Charged per command change; ignored on the first action of each episode.
    nav_action_change_scale: float = 0.08

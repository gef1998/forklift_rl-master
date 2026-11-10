# 激光运动补偿优化（2026-11-10）

已在 160 仿真和 159 部署实现“保留原始回波，运动补偿后再池化”，部署端的激光补偿改用
`odom` 连续坐标系。保留旧模式用于对照，网络输入仍为 136/196 维，GRU 接口和任务边界重置不变。
没有替换模型、修改车体标定、动作限制、低通或安全缩放阈值，也未启动或控制实际车辆。

## 新处理链路

旧模式：扫描 → 按扇区选一个代表点 → 缓存 → 补偿选中的少量点 → 再分桶。

新模式：扫描 → 缓存原始回波 → 将所有回波补偿到当前车体 → 分别池化为策略扇区与安全扇区。

旧模式中，代表点移到相邻扇区后，本扇区其他原始回波已被丢弃，可能被填成最大距离。
新模式保留这些真实回波，不做插值补障，也不把没有测到的障碍凭空补出来。

仿真端两个分支共享同一份带噪声的原始扫描和延迟队列；策略专用 dropout 在采集时生成，
随对应帧进入队列，只屏蔽策略分支，安全分支仍使用物理扫描。回合重置同时清除缓存、队列和 dropout。
固定射线方向只计算一次，避免重复保存每帧角度以及两套完整安全/策略原始扫描。
补偿使用相对位姿计算，避免高 env 数量、较大世界坐标下逐点加减世界原点的 float32 精度损失。

## 部署 TF

- 目标和目标方向继续使用 `map`。
- 激光当前位姿、前后输入各自采集时刻的位姿，均从 `odom` 查询。
- `/tf_static` 提供安装外参，`/tf` 提供车辆运动，TF2 自动组合。
- 前后雷达保留各自扫描时间戳，分别补偿后融合；AoI 仍取较老输入。
- 如果 `odom` 链不可用或过期，会沿用原有无效观测停车逻辑，不静默切回 `map`。
- 使用最新可用 TF，没有引入未经验证的速度外推或逐射线去畸变。

测试版本的 `/emma_rl/observation_debug` 新增：
`lidar_motion_frame`、`lidar_motion_tf_age_s`、`lidar_motion_tf_stamp_sec/nsec`。
`base_x/base_y/base_yaw` 仍表示目标导航使用的 map 位姿。
它们可用于下一次行驶记录中排查 TF 延迟和 GRU 重置/停车是否与摇摆同步。

## 配置与切换

159 YAML 和 launch 默认：

```yaml
lidar_warp_enabled: true
lidar_warp_before_pooling: true
lidar_motion_frame: odom
```

默认激光来源仍为 `/scan_filtered`，A/B 入口均支持新模式：

```bash
roslaunch mini_servo mini_servo_scan_filtered_test.launch
roslaunch mini_servo mini_servo_dual_lidar_test.launch
```

每次运行一个控制入口。恢复旧处理链用于对照：

```bash
roslaunch mini_servo mini_servo_scan_filtered_test.launch \
  lidar_warp_before_pooling:=false lidar_motion_frame:=map
```

160 在 `emma_obs_nav_env_cfg.py` 中新增同名配置 `lidar_warp_before_pooling=True`。
play 可显式恢复旧模式：

```bash
bash launch.sh play --task Emma-RL-ObsNav-GRU-Hybrid180-Direct-v0 \
  --num_envs 10 --checkpoint <checkpoint.pt> --no-lidar-warp-before-pooling
```

已有权重可以加载，但观测数值会因保留更多原始回波而变化，需要按同一配置验证，
后续训练建议使用新模式。当前正在运行的训练没有被重启，也不会热更新已经加载的 Python 实现；
本次改动对新启动的训练、play 和重新启动的部署节点生效。

## 验证结果

- 仿真纯几何、CPU/CUDA、真实环境队列/重置函数等 **86 项测试通过**。
- 实际 Isaac CPU 环境：新模式 Hybrid180、旧模式 Hybrid180、新模式 uniform120 各运行 50 步，
  每组 2 env、4 次自动回合重置，观测和安全净空保持有限，缓存形状正确。
  更新后的固定方向优化也重新通过了 Hybrid180 实际环境冒烟测试。
- x86_64 和 ARM64 均在相应容器内编译；每架构 **70 项测试、69 项通过**。
  唯一历史失败仍是缺少旧 `agent_500000_20260912_opset13.onnx` 的 MLP 回归，没有删除或绕过。
- 96 组 C++ 与真实 Torch 原始回波 helper 数值对照通过，覆盖 720/1440 射线、120/180 扇区、
  110°/180° 半视场和不同平移旋转。两架构观测最大误差 `1.87e-9`，安全净空最大误差 `1.78e-15 m`。
- 新增测试复现了“最近代表点离开扇区，但其他原始点仍在本扇区”的情况，并验证新模式保留其回波；
  另验证 map 重定位跳变不会改变使用独立 odom 位姿的激光输入，而实际 odom 运动仍会正确改变输入。

## 已有 bag 回放

对 `2026-11-09-22-54-49.bag` 使用完全相同的到达时刻、10 Hz 缓存与 50 Hz 控制时钟，
对照旧模式与新模式，共 1033 个两版均有效的控制步：

- A `/scan_filtered`：恢复 1504 个“旧净空 >5 m、新净空 <1 m”的扇区时刻，其中前向 ±30° 为 882 个，
  367 个控制步的前向输入得到恢复。
- B 自行融合：对应恢复 1919 个扇区时刻，其中前向为 1064 个，428 个控制步的前向输入得到恢复。
- 两个来源都没有出现净空增大超过 1 mm 的情况；新模式从保留的真实回波中取近值。
- A 的 +12～13° 扇区显示 >5 m 的比例从 95.55% 降到 76.48%，仍存在大量局部空洞；
  B 对应从 38.33% 降到 37.37%。接收端补偿不能恢复融合器已经丢失或量化到相邻方向的信息。

bag 是静态场景，没有目标、action 或速度反馈。上述结果验证输入链路，不能证明实车摇摆或碰撞已经解决。
动态 TF 抖动、控制跟踪及完整策略效果仍需要新行驶记录。

## 开销

x86 观测构建器微基准，不含 TF 查询和网络推理：

- 原压缩缓存单激光 P99 约 0.34 ms。
- 新原始缓存单激光 P99 约 0.45 ms。
- 新原始缓存前后双激光 P99 约 0.90 ms。

4096 env、720 射线 CUDA helper 微基准，测试时有其他训练并行运行：
旧模式中位约 1.25 ms，新模式优化后约 4.45 ms。它只测感知算子，不代表整套训练吞吐。
新模式需要处理更多真实回波，存在计算开销；通过预计算固定方向、直接计算矩形投影和共享原始队列
削减了重复计算。相对旧缓存，额外缓存内存估算约 43.8 MiB（4096 env）或 137.0 MiB（12800 env）。

## 融合器范围与产物

上游 `/scan_filter` 暂不开源。本次没有改写 `angle_increment`、截断 722 点，或添加 180° roll。
已知 `0.499963°` 与 1° 扇区边界的敏感性仍需在上游输出层核对；B 继续提供对照链路。

159 诊断目录：`/data/lf/cross_arm64/lidar_motion_opt_20261110/`。
160 诊断目录：`/home/jz/cqz/forklift_rl-master/logs/diagnostics/lidar_motion_opt_20261110/`。
两端均保存本次源码补丁、基准和回放统计；源码修改前备份见诊断目录的 `manifest.json`。

159 安装目录分别为：

- `/data/lf/cross_arm64/rl_ws_x86_64_gru_alignment/install/`
- `/data/lf/cross_arm64/rl_ws_arm64_gru_alignment/install/`

部署对应架构的安装目录及依赖后重新启动节点，新的默认补偿链路才会在实车中生效。

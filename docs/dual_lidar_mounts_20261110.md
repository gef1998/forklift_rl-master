# 双激光观测实现与验证（2026-11-10）

方案 1 已应用到 160 仿真与 159 mini_servo。默认使用前后物理激光，部署直接订阅 `/scan_emma_nav_front` 和 `/scan_emma_nav_back`，分别补偿运动并投影到车体中心，再融合扇区最小净空。`/scan_filtered` 的独立对比入口仍保留。

## bag 中的安装外参

来源：159 mini_servo 下的 `2026-11-09-22-54-49.bag`。参考系是 `base_footprint`，X 向前、Y 向左、Z 向上。

- 前激光 `laser_link`：位置 `(0.3644529702, -0.0006679106, 0.0)` m；yaw `-0.4816950391°`。
- 后激光 `laser_link2`：位置 `(-0.3808409390, 0.0038731886, 0.2)` m；yaw `179.4984201460°`。
- 两者 roll/pitch 都为 0。后激光外参已经组合整条 TF 链，不能把 `laser_link → laser_link2` 的平移直接当作相对车体的位置。

这些是 bag 发布的 TF 标定值，不能用它证明机械测量准确。尤其 Z 是相对 `base_footprint` 的记录值，前 0、后 0.2 m 不等于两者离地高度；仿真将其作为相对机器人根坐标系的安装位置。后续若要对齐低矮障碍物/地面回波，需要确认实车 TF 高度及仿真根坐标系和真实 `base_footprint` 的垂直对应关系。平面导航的前后位移及偏航已直接使用记录值。

详细数值与提取脚本分别见同目录 `mounts.json`、`extract_bag_mounts.py`；提取过程只读取 bag，没有启动控制节点。

## 射线与时序

- 前激光：原生 840 线，起始角 `-104.9999995068°`，角增量 `0.2499958092°`，取原生下标 `0,2,4,...,838`，共 420 线，约 0.5°。
- 后激光：原生 800 线，起始角 `-100.0000005060°`，角增量 `0.2499999981°`，取 `0,4,8,...,796`，共 200 线，约 1°。
- 上述角度相对各激光坐标系。车体系首条方向分别约为 `-105.481695°` 与 `79.498420°`。保留原生角度 metadata，不把它强行改写成精确 0.5°/1°。
- 观测合计 620 条射线，前后名义频率各 10 Hz。仿真分别维护相位、延迟、队列和拍摄位姿；原有随机延迟、抖动、整帧丢失仍生效。AoI 取较老来源的时间。
- 每个控制步利用各来源拍摄位姿补偿运动，再形成当前车体系的策略扇区和独立 120 扇区安全净空。策略网络输入维数不变：Hybrid180 为 196，Uniform120 为 136。输入并非直接拼接 620 个距离。
- 碰撞/奖励真值继续使用独立中心 360° / 720 线扫描，随 50 Hz 控制更新，不使用观测缓存替代碰撞检测。

性能上复用已有障碍物 Warp mesh；仅对本步需要拍摄的来源及 env 合并执行一次射线查询。没有复制两套场景，也没有将 620 线观测在每个 50 Hz 控制步全部重扫。2 env / 50 步测试包含多次重置，共查询 14880 条观测射线；逐步全扫将需要约 62000 条。此数值不是整套训练 FPS 基准，碰撞真值、物理和训练计算仍有独立开销。

## 配置及运行

160：`source/emma_rl/emma_rl/tasks/direct/emma/emma_obs_nav/emma_obs_nav_env_cfg.py`：

```python
observation_lidar_mode = "dual"
front_lidar = front_mount()
back_lidar = back_mount()
lidar_fixed_frequency_hz = 10.0
```

`front_mount()` / `back_mount()` 的安装坐标、偏航、原生角度、线数、抽样步长位于同目录 `dual_lidar.py`。`lidar.pattern_cfg.horizontal_res` 现在控制中心碰撞真值扫描，双激光观测密度由各自配置控制。

正常 train/play 默认采用双激光。旧中心扫描可在 play 加 `--observation-lidar-mode central`；旧 `--no-lidar-warp-before-pooling` 也会选择 central。训练时可用配置覆盖 `env.observation_lidar_mode=central`。已有模型的输入维数兼容，但观测分布发生改变，建议在新配置下重新训练或微调，再进行实车验证。没有覆盖现有 checkpoint/ONNX，没有重启 160 正在运行的训练；已运行进程需下次启动才能加载本次代码。

159 默认 YAML、`mini_servo.launch`、`mini_servo_test.launch` 都选择 `dual_scan`，前后抽样参数分别为：

```yaml
dual_lidar_front_sample_stride: 2
dual_lidar_back_sample_stride: 4
```

部署每个来源的缓存采样上限仍为 10 Hz；实车更高频的原始消息不会直接提高策略缓存频率。运动补偿继续使用动态 `odom` TF，静态 TF 只提供安装外参。实际运行时读取实时 TF，未把 bag 外参硬编码进部署程序。

双激光测试入口：`roslaunch mini_servo mini_servo_dual_lidar_test.launch`。

保留的融合输入对比入口：`roslaunch mini_servo mini_servo_scan_filtered_test.launch`，该入口仍使用完整 `/scan_filtered` 消息，不应用双激光抽样。

## 验证结果

- 160 激光相关单元测试 93 项通过，包含 CPU/CUDA、两个来源独立时钟、延迟、按 env 重置、dropout 安全隔离、无效距离与安装角度。
- 实际 Isaac CPU 2 env：双激光 Hybrid180、双激光 Uniform120、旧中心扫描 Hybrid180，各运行 50 步并完成 4 次回合重置；观测/奖励/安全净空有限，缓存线数及独立 720 线真值符合预期。
- 仿真 Torch 几何与实际 C++ builder 对比 48 组：覆盖两种扇区、两种策略 FOV、不同来源拍摄位姿、车体自回波、无回波、转动/平移。x86、ARM 均通过；观测最大绝对误差 `5.96046447754e-8`，安全净空误差 `2.6645352591e-15` m。
- 原始 bag 离线回放：1033 个有效 50 Hz 时刻，双激光 builder 失败 0；7 个时刻因 bag 起始/结尾 TF 覆盖不足而跳过。静态场景及缺少导航动作/反馈的 bag 不能证明动态驾驶效果或消除摆动。
- 159 x86 和 ARM64 已通过容器编译并安装。两者各 71 项 C++ 测试中 70 项通过；唯一失败仍是已有测试缺少 `agent_500000_20260912_opset13.onnx`，不是本次双激光测试失败。未恢复该旧模型或跳过/删除该测试。

备份：160 `/home/jz/cqz/dual_lidar_mounts_backup_20261110_201024`；159 `/data/lf/cross_arm64/dual_lidar_mounts_backup_20261110_201026`。本地 `manifest.json` 列出修改文件。部署验证工具及回放结果位于 159 `/data/lf/cross_arm64/dual_lidar_geometry_20261110`。

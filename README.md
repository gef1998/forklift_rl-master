# Forklift RL: Emma Navigation

基于 Isaac Lab 与 skrl PPO 的 Emma AMR 避障导航项目。当前仓库只维护两条可对照的核心链路：原始 LiDAR ranges 的 MLP 策略，以及 Costmap 的 CNN 策略。

## 当前范围

| 状态 | 环境 | 观测与网络 | Gym ID |
|---|---|---|---|
| Implemented | Emma Nav | 120 维 LiDAR ranges + 目标/本体/AoI，共 130 维；MLP PPO | `Emma-RL-ObsNav-Direct-v0` |
| Implemented, validation pending | Emma Costmap | 前向 220° LiDAR → 2×48×48 Costmap + 12 维本体/AoI；轻量 CNN PPO | `Emma-RL-CostmapNav-Direct-v0` |
| Implemented, validation pending | Emma Nav safety | 曲率限速、静态障碍制动包络、TTC 急停、陈旧扫描 fail-closed | 见安全控制设计 |


Costmap 环境现在直接使用与实车一致的前向 220° LiDAR ranges 构建局部图：通道 0 为单次膨胀后的 occupancy，通道 1 为 observed mask，后方盲区和遮挡区不会被误编码为自由空间。两条任务共享场景、奖励、异步调度、动作投影和 runtime safety。

Emma Nav 与 Costmap Nav 都使用逐维 Beta 分布：网络在 `(0,1)` 内采样，再以 `a = 2u - 1` 映射到 signed action `(-1,1)^2`。环境提供两种可切换的动作解释：`body_twist` 将两维解释为归一化线速度/角速度，并以左右轮同步缩放的方式投影到可行轮速；`wheel_velocity` 将两维直接解释为归一化左/右轮速。两种模式都支持倒退和原地旋转；但只有前向 220° LiDAR 时，runtime shield 会禁止向无观测后方平移。旧 Gaussian actor checkpoint 与当前 Beta head 不兼容，不同动作模式之间的 checkpoint 也不能直接复用。

两种动作模式默认都经过同一个 runtime safety 层，依次执行曲率限速、静态障碍物制动包络和 TTC 急停。shield 使用独立于 policy dropout 的最后有效 LiDAR 缓存；扫描缺失或 AoI 超限时 fail closed。仅为诊断策略本身，训练可用 `env.safety_control.enabled=false` 临时旁路整套安全投影；该模式不应用于真机或安全评估。当前 ObsNav 障碍物在 episode 内不移动，因此默认障碍速度上界为 0；动态人员接入后必须使用独立动态几何和非零速度上界。默认参数是仿真初值，不等同于真机安全认证参数。

## 环境要求

- NVIDIA Isaac Sim 4.5+
- 与项目代码兼容的 Isaac Lab（当前开发环境使用 2.x）
- Python 3.10+
- skrl 1.4.2+
- Git LFS（用于管理仓库内的二进制 USD 资产）

先配置本机 Isaac Lab 路径：

```bash
export ISAACLAB_ROOT=/path/to/IsaacLab
export ISAAC_PYTHON="$ISAACLAB_ROOT/_isaac_sim/python.sh"
```


## 快速开始

安装并确认只注册了两个核心环境：

```bash
bash launch.sh install
bash launch.sh list-envs
bash launch.sh test-action-control
bash launch.sh test-corridor-layout  # S形课程走廊、双向出生与通行约束
bash launch.sh test-ring-layout  # 局部迷宫墙体、四角出生与通行约束
bash launch.sh test-nav-reward
bash launch.sh test-safety-control
bash launch.sh test-beta-policy
```

训练 ranges baseline：

```bash
bash launch.sh train --num_envs 1024 --headless
```

动作控制模式通过同一个配置开关选择，默认保持本体速度接口：

```bash
# action=(normalized v, normalized omega)，越界时保持曲率同步缩放左右轮
bash launch.sh train --num_envs 1024 --headless \
  env.action_control_mode=body_twist

# action=(normalized left wheel, normalized right wheel)，直接控制左右轮速
bash launch.sh train --num_envs 1024 --headless \
  env.action_control_mode=wheel_velocity
```

两种模式应分别训练并使用不同的实验名称，不能共享策略 checkpoint。

默认 ring 环境为 `10 m × 10 m` 的局部迷宫：四周由 `0.15 m` 薄墙封闭，四段错位内墙与外墙保持 `2.0 m` 主走廊；内墙长度采用 `arena_size/4 + wall_tip_offset`，默认约 `3.5 m`，降低早期学习绕行的路径长度。Emma 从四个无障碍角落端点之一随机 yaw 出生，Goal 位于另外三个角落之一，初始直线距离至少 `7 m`，回合时限为 `80 s`。Cylinder/Cube 按环境随机摆放，并避开出生区和内墙开口；每次从 24 个合法候选中优先选择离已有障碍最远的位置，减少局部扎堆。走廊内每个障碍至少在一侧留下 `1.15 m` 净通道，按 Emma `0.534 m` 全宽计算仍有约 `0.616 m` 总余量。warehouse 使用独立的 `16 m × 16 m` 局部导航布局（3 列×5 段货架、`5–10 m` 目标距离），适配 `0.9 m/s` 训练。

课程预训练可使用 `env.world_mode=corridor`。该模式为独立的 `10 m × 10 m`
封闭 S 形走廊：两面交错长墙迫使 Emma 先后绕向两侧，墙端保留 `2.5 m`
开口；偶数/奇数环境交替使用原布局和水平镜像，Emma/Goal 在上下安全区
随机互换，初始距离至少 `7 m`。该阶段不注入小型随机障碍，观测、动作和
奖励与 ring 完全一致，因此 checkpoint 可直接切换到
`env.world_mode=ring` 继续训练。

```bash
# 第一阶段：从头训练 S 形走廊
ISAAC_PYTHON=/home/jz/isaacsim/python.sh \
  bash launch.sh train --headless --num_envs 1024 env.world_mode=corridor

# 第二阶段：加载走廊权重，在 ring 中继续训练
ISAAC_PYTHON=/home/jz/isaacsim/python.sh \
  bash launch.sh train --headless --num_envs 1024 \
  --checkpoint /absolute/path/to/best_agent.pt env.world_mode=ring
```

两种世界需要分别创建环境，因此场景切换发生在两次训练运行之间，不在同一
进程内按 phase 热切换。

轮速 actuator 上限与本体速度上限独立配置。当前默认 `wheel_max_velocity=10 rad/s`、`wheel_radius=0.09 m`、`wheel_base=0.40 m`，运营上限设为 `max_linear_velocity=0.9 m/s`、`max_angular_velocity=1.0 rad/s`；自动同步开关关闭，避免角速度在启动时被覆盖回运动学原地旋转极限 `4.5 rad/s`。两种 action mode 都会对左右轮做同比例投影，同时满足限制并保持曲率。

Emma Nav 默认使用固定 25 Hz LiDAR 和 50 Hz 控制。ObsNav 的120维扇区按车体系物理方位排列：`index 0≈正前方 +1.5°`，随后逆时针经过左侧、后方和右侧；前向±110°实车视场对应保留 `0–36、83–119`，后方盲区为 `37–82`。可通过 Hydra override 切换消融模式：

```bash
# A0: 50 Hz 同步 baseline
bash launch.sh train --num_envs 1024 --headless env.lidar_sensor_mode=synchronous

# A1/A4: 固定 20 Hz；默认启用 SE(2) warp 与真实 AoI
bash launch.sh train --num_envs 1024 --headless \
  env.lidar_sensor_mode=fixed env.lidar_fixed_frequency_hz=20.0

# A1: 纯 ZOH（关闭 warp 和 AoI，观测维度保持不变、AoI 通道填零）
bash launch.sh train --num_envs 1024 --headless \
  env.lidar_sensor_mode=fixed env.lidar_warp_enabled=false env.lidar_aoi_enabled=false

# A6: 每环境随机 10–50 Hz；延迟、抖动和整帧 dropout 参数见环境配置
bash launch.sh train --num_envs 1024 --headless env.lidar_sensor_mode=random
```

切换到真机对应的 25 Hz、20 m 参数时，只需覆盖顶层配置；环境会在构造 RayCaster 前同步内部量程：

```bash
bash launch.sh train --num_envs 1024 --headless \
  env.lidar_sensor_mode=fixed env.lidar_fixed_frequency_hz=25.0 \
  env.lidar_max_distance=20.0
```

训练 Costmap baseline：

```bash
TASK=Emma-RL-CostmapNav-Direct-v0 \
  bash launch.sh train --num_envs 1024 --headless
```

播放 checkpoint：

```bash
bash launch.sh play --checkpoint logs/skrl/emma_obs_nav_direct/<run>/checkpoints/best_agent.pt
```

按《测试文档》的回型路径、S 型路径和直行绕障执行自动宽度扫描：

```bash
bash launch.sh test --headless \
  --checkpoint logs/skrl/emma_obs_nav_direct/<run>/checkpoints/best_agent.pt \
  --load-kg 0 --mapping-mode simulation

# 全局范围会同时应用于所有选定场景；例如三场景共同扫描 1.4–2.0 m
bash launch.sh test --headless \
  --checkpoint logs/skrl/emma_obs_nav_direct/<run>/checkpoints/best_agent.pt \
  --width-min 1.4 --width-max 2.0 --width-step 0.1

# 也可只测一个场景，并单独覆盖其范围
bash launch.sh test --headless --scenarios straight_obstacle \
  --straight-obstacle-width-min 0.7 \
  --straight-obstacle-width-max 1.5 \
  --straight-obstacle-width-step 0.1
```

扫描从宽到窄运行，每个静态宽度都会重建独立 Isaac 环境与 LiDAR mesh；同一宽度默认连续运行 3 回合，三回合均成功才判定通过。结果目录包含终端汇总、逐回合 `episodes.csv` 和带完整配置/宽度汇总的 `summary.json`。`--load-kg` 与 `--mapping-mode` 当前仅记录测试工况元数据，并不会改变尚未建模的载荷物理属性；直行绕障场景当前为单个静态长方体，不包含多障碍物或移动障碍物扩展。

复用 `play.py` 的环境创建、reset、`env.step()` 和动作记录逻辑，仅将策略输出替换为
恒定左右轮速命令：

```bash
bash launch.sh wheel-response --headless

# 自定义左右轮速和控制步数
WHEEL_CMD_LEFT=-3 WHEEL_CMD_RIGHT=3 WHEEL_CMD_STEPS=150 \
  bash launch.sh wheel-response --headless
```

结果写入 `logs/wheel_response/<timestamp>/play_diagnostics/`，包含与普通 play
一致的 action、目标/实际车速和目标/实际轮速 CSV 与曲线。

也可使用 Make：

```bash
make train NUM_ENVS=1024
make train TASK=Emma-RL-CostmapNav-Direct-v0 NUM_ENVS=1024
```

## 仓库结构

```text
forklift_rl/
├── assets/                             # Emma、障碍物与场景 USD（Git LFS）
├── scripts/
│   ├── skrl/
│   │   ├── train.py                    # PPO 训练入口
│   │   └── play.py                     # checkpoint 播放入口
│   ├── list_envs.py
│   ├── random_agent.py
│   └── zero_agent.py
├── source/emma_rl/
│   └── emma_rl/tasks/direct/emma/
│       ├── agents/                     # 两套 Emma PPO 配置
│       ├── asset_paths.py              # 仓库资产解析与校验
│       ├── emma_obs_nav/               # ranges + MLP、双动作映射与 runtime safety
│       ├── emma_costmap_nav/           # 前向 LiDAR Costmap + CPU 轻量 Beta-CNN
│       └── utils/                      # 两套自定义模型的构造入口
├── tests/
│   ├── test_async_lidar.py             # 异步调度、AoI 与 SE(2) 几何测试
│   ├── test_safety_control.py           # 曲率、制动包络、TTC 与 fail-closed 测试
│   └── test_lidar_costmap.py           # 220° FOV、unknown、遮挡和膨胀测试
├── tools/                              # Emma/USD/LiDAR 诊断工具
├── launch.sh
└── Makefile
```

新版发行包与 Python 导入命名空间均为 `emma_rl`，可与服务器中旧版 `isaac_lab_forklift` 同时安装。训练输出写入 `logs/`，该目录和 shader cache 均已忽略，不属于源码。

## 验证

纯 Costmap 生成器检查不需要 Isaac Sim：

```bash
bash launch.sh test-action-control
bash launch.sh test-safety-control
bash launch.sh test-beta-policy
bash launch.sh test-costmap
bash launch.sh test-async-lidar
```

需要 Isaac Sim/GPU 的注册与 smoke test：

```bash
bash launch.sh list-envs
bash launch.sh random-agent --num_envs 4 --headless
TASK=Emma-RL-CostmapNav-Direct-v0 \
  bash launch.sh random-agent --num_envs 4 --headless
```

## License

Apache-2.0，见 [LICENSE](LICENSE)。

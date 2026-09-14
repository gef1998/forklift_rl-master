# Emma 驱动修正与验证

实施日期：2026-09-11 至 2026-09-12。任务 `Emma-RL-ObsNav-Direct-v0`，Isaac Sim 4.5、IsaacLab 扩展 0.41.3、RTX 3090。修改位于 `source/emma_rl/emma_rl/tasks/direct/emma/emma_obs_nav/`。

## 已落地的修改

- `drive_contacts.py`：在场景复制前，给驱动轮增加半径 0.05 m 的球面碰撞代理，给脚轮增加半径 0.0395 m 的圆柱碰撞代理；关闭原形状的碰撞。原有可见圆柱、刚体质量/惯量、关节和二进制 USD 均保留。
- 通过 `PhysxSceneAPI` 开启 `enableExternalForcesEveryIteration`。这是与接触代理一起验证的组合；单独在旧几何上开启曾经恶化转向。
- 保持 TGS 8/4、damping=200、7 DOF；每轮 `effort_limit_sim` 默认改为 4.15 N·m。
- `action_control.py`：在两个轮速之间共同插值，使名义 body twist 的变化满足线/角加速度约束，同时保持轮速和车体速度的可行范围。覆盖启动、减速和反转。
- `emma_obs_nav_env.py`：在安全盾之前限制名义指令，在两个物理子步间插值下发；安全盾干预时直接采用安全目标，避免插回旧高速命令。新指令状态在按环境重置时清零。
- `safety_control.py`：修复运行既有测试时发现的问题——行进方向没有任何 LiDAR 射线时，空集合不能被当作无穷远畅通。仅前向 LiDAR 的倒车请求现在会触发停车。
- 清理已被实验否定的配置注释，区分外加 effort 输入、PD 估算力矩和真实隐式驱动输出；修正材质合成优先级的说明。

当前配置：

```python
drive_wheel_collision_shape = "sphere"
caster_collision_radius = 0.0395
drive_enable_external_forces_every_iteration = True
wheel_velocity_damping = 200.0
wheel_effort_limit = 4.15
enable_motion_acceleration_limits = True
max_linear_acceleration = 1.0
max_angular_acceleration = 2.0
linear_acceleration_command_scale = 0.90
angular_acceleration_command_scale = 0.94
```

硬件规格仍分别为 1.0 m/s²、2.0 rad/s²。正常下发指令使用 0.90 m/s²、1.88 rad/s²，给实测跟踪超调留余量。它们是单独保存的校准系数，不是把硬件上限改小后宣称精确匹配。

## 结论与证据

旧几何 8/4 的低速转向约 32%、高速约 78.5%；16/8 的低速仍只有约 59%。问题不能单纯归因于缺积分器：P-only 在恒定指令、静止状态下不积累积分误差，且很小的轮速误差就会达到名义饱和阈值。

现场记录确认报告轮速与关节转角变化率不同，16/8 还存在奇偶物理步差异。按位姿变化分解，16/8 是约 93.04% 的轮转角跟踪乘以 96.33% 的车体/轮运动学一致性，不是“轮速完全跟踪、剩余 10% 都是物理侧滑”。因此不能用原来的 50 Hz 轮速遥测直接判断真实滑移。

固定方向脚轮的横向摩擦、六点刚性支撑的载荷分配，以及轻轮惯量与隐式驱动/接触约束的数值耦合共同参与。新的代理改善了接触形状和驱动轮承载；它是导航用途的近似模型，不是已由真机预载、轮胎和悬挂参数标定的模型。

稳定偏置并不要求抖动：有限迭代后的离散更新可以重复落在稳定的有偏状态；TGS 的位置迭代还涉及内部时间推进，不只是对同一个静态方程多算几遍。因此增加 pos_iters 改变稳态结果，与 wheel sd 很小并不矛盾，但这也不足以单独确认全部误差来源。

低速相对误差更大同样不反证数值误差：摩擦阈值、有限接触修正等误差项不必随指令同比缩小。恒定目标下 P 速度环没有积分器，轮子静止时 `damping × error` 不会随等待时间自行累积；4.15/200≈0.02075 rad/s 已达到名义饱和误差量级。当前脚轮 static/dynamic 都是 0.02，也不能直接套用“大静摩擦跌落为小动摩擦”的解释。低速非光滑接触及数值振荡是候选机制，尚未独立辨识为经典 stick-slip；没有依据默认加积分或降低 damping。

完整修改前调查和 31 组逐物理步实验见 [2026-09-11 调查记录](diagnostics/emma_drive_investigation_20260911.md)。机器可读汇总和原始日志压缩包位于 `artifacts/drive_diagnostics/20260911/`。

## 回归口径与结果

独立探针先以零指令整定 0.5 s，再发命令；关闭任务终止和安全盾，仅测试自由场驱动。每个 0.01 s 物理步记录指令、关节角/速度、根姿态和车体速度，稳定值采用最后 2 s。实际加速度峰值用相邻物理步的报告车体速度差分；没有过滤或隐藏启动/反转峰值。

已通过：

- 动作映射和加速度约束测试 10 项；包括反转、轮半径缩放、可行范围、无变化指令和非法配置。
- 安全盾测试 11 项，包括无后向射线时倒车停车。
- 四环境首尾碰撞代理复制、可见形状保留、名义指令变化率检查。
- 安全盾停车不被插值延迟；只清零被重置环境的命令缓冲。

转向最终校准（角指令系数 0.94）：

- 高速轮速幅值 6.45 rad/s：车体跟踪 99.169%，实测角加速度峰值 1.884 rad/s²。
- 高速反转：反向稳定跟踪约 98.944%，整个测试的实测角加速度峰值 1.931 rad/s²。
- 四环境低速轮速幅值 1 rad/s：被记录环境跟踪约 97.526%；安全盾/重置检查通过。其余环境没有被当作同精度的独立动力学统计样本。

直线校准的一个重要失败对照：线指令系数从 0.995 降到 0.95，并没有单调降低反转峰值；0.95 下出现过 1.081 m/s² 的接触瞬态。系数 0.90 的反转测试峰值为 0.953 m/s²，反向稳定速度约 -0.9000 m/s。因此当前采用 0.90，未把正常指令变化率误报为严格的物理峰值保证。

最终默认配置的直线复测：稳态前向速度 0.899863 m/s，起步线加速度峰值 0.905138 m/s²；直线反转峰值 0.952815 m/s²。

1024 环境同机相邻吞吐对照：原接触 8/4 为 74,046 env-steps/s，新方案 8/4 为 71,953，原接触 16/4 为 65,692。新方案相对原 8/4 降低 2.83%，相对原 16/4 提高 9.53%。每组正常 env.step 预热 40 步、计时 300 步；机器上原有其他 GPU 工作负载未停止，因此这是单次相邻对照，不是精确性能保证。

结果来源说明：`calibrated_low_smoke/turn/reverse` 的角指令系数已经是最终 0.94，线指令系数当时为 0.995；这些是零线速度的纯转向，线系数不限制其指令。`reverse_scale90` 显式覆盖到 0.90，与当前默认一致；`release_straight` 和三个吞吐文件使用最终默认配置。完整参数与来源说明同时写入结果索引。

全部实施期日志和时序保存在 `artifacts/drive_diagnostics/20260912/`；文件名中的 `calibrated_*`、`final_*` 是迭代期间记录，最终使用的结果由 `release_summary.json` 指定，避免误把中间版本当作最终配置。

## 复现

在项目根目录运行：

```bash
# 最终配置，高速起步
/home/jz/isaacsim/python.sh scripts/diagnostics/probe_emma_drive.py \
  --headless --contacts --turn --target 6.45 --out /tmp/emma_turn.json

# 高速转向反转
/home/jz/isaacsim/python.sh scripts/diagnostics/probe_emma_drive.py \
  --headless --contacts --turn --target 6.45 --reverse-at 3 --steps 350 \
  --out /tmp/emma_turn_reverse.json

# 直线反转，最后两秒已进入反向稳态
/home/jz/isaacsim/python.sh scripts/diagnostics/probe_emma_drive.py \
  --headless --contacts --target 18 --reverse-at 2 --steps 350 \
  --out /tmp/emma_straight_reverse.json

# 多环境复制、安全盾优先级、部分重置
/home/jz/isaacsim/python.sh scripts/diagnostics/probe_emma_drive.py \
  --headless --contacts --turn --target 1 --num-envs 4 --smoke \
  --out /tmp/emma_multi.json

# 原几何诊断对照：关闭新代理、逐迭代外力和名义加速度限制
/home/jz/isaacsim/python.sh scripts/diagnostics/probe_emma_drive.py \
  --headless --contacts --legacy --turn --target 1 --effort-limit 4.5 \
  --out /tmp/emma_legacy.json

# 吞吐：正常 env.step，40 步预热后计时，无逐步接触记录
/home/jz/isaacsim/python.sh scripts/diagnostics/probe_emma_drive.py \
  --headless --benchmark --num-envs 1024 --turn --target 1 --steps 300 \
  --out /tmp/emma_throughput.json
```

名义加速度限制可单独用 `--no-acceleration-limits` 做消融。探针默认关闭安全盾仅用于驱动诊断；不是训练或部署设置。正常任务的安全盾开关保留原来的配置。

## 边界与回退

- 此次验证的是无障碍、固定质量、整定后的启动和反转；没有证明碰撞、急停、不同载荷/地面或所有随机状态下都严格满足物理加速度上限。安全盾制动优先于正常指令平滑。
- 接触代理导致驱动轮承担几乎全部重量。真机支撑预载、轮胎接触形状和万向脚轮结构仍需核实。不要把这套尺寸无条件移植到别的车。
- 球面代理、0.5 mm 支撑高度差和逐迭代外力是组合验证的结果。不要只开启其中一个开关并认为会得到同样效果。
- 原有策略 checkpoint 可以用于观察，但动作动态已改变，不能假定旧策略的表现保持不变；此次没有启动训练或替换 checkpoint。
- 保留原二进制 USD，碰撞修改由 Python 配置持久化，在每次任务创建时构建。需要回到旧接触模型时，设置 `drive_wheel_collision_shape="cylinder"`、`caster_collision_radius=None`、`drive_enable_external_forces_every_iteration=False`；是否关闭加速度限制由复现目的决定。
- 完整修改前备份位于 `/home/jz/cqz/emma-drive-backup-20260911-0CFUYw/`。备份包含用户原有未提交改动；不要用 `git reset --hard` 回退它们。

可复用的排查流程和反例见 [PhysX 驱动接触经验](lessons/physx_drive_contacts.md)。

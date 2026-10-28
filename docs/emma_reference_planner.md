# 第一阶段：窄道参考路径生成器

实现日期：2026-10-22；2026-10-23 增加移动段最小转弯半径、相切连接和停车标记。用于训练端 privileged reward，支持当前 dedicated detour 场景。

## 使用

`source/emma_rl/emma_rl/tasks/direct/emma/emma_obs_nav/emma_obs_nav_env_cfg.py`：

```python
reference_path_reward_enabled = True
reference_path_backend = "hybrid_astar"  # 默认；改为 "polyline" 对照旧折线
reference_planner_xy_resolution = 0.01
reference_planner_yaw_resolution_deg = 1.0
reference_planner_min_turning_radius = 0.50
reference_planner_analytic_radius_factors = (1.0, 1.1, 1.5, 2.0)
reference_planner_inplace_rotation_fallback = True
reference_planner_collision_margin = 0.01
reference_planner_max_seconds = 1.5       # 每段、每次搜索的时间预算；后备可再搜索一次
reference_planner_max_expansions = 12000  # 每段、每次搜索的最大展开节点数
reference_planner_cache_size = 2048      # 缓存内部标准门洞段的数量
reference_planner_max_points = 1024      # 每条完整路径的采样点上限
reference_planner_failure_dump_dir = "logs/reference_planner_failures"  # "" 关闭失败快照
```

按原训练命令重新启动进程即生效；已经运行的 Python 训练进程不会热加载新代码。本次没有重启正在进行的训练，也没有修改用户已有的暂存区。

## 生成流程

1. 仅在 dedicated detour 环境 reset 时查询，使用本回合实际起点、朝向、终点和课程等级。
2. 从实际布局读取连续长度的内墙、封闭墙、短墙，以及圆柱/旋转长方体障碍物；外墙转为场地内部边界。
3. 按行驶顺序提取本回合必须经过的门洞，保留门中心位置和通过朝向约束。在相邻门之间运行差速 SE(2) Hybrid A*。门前、门后的固定姿态不再作为必须经过的中间点；遇到窄门时会尝试带直线进出段的候选连接，保护车体尾部净空。
4. 第一遍使用前进、倒退和曲率有界圆弧。解析连接使用相切的直线/圆弧组合：恒定档位的 Dubins CSC/CCC 类别，以及较大半径的双圆弧候选；倒车连接按反向运动构造，混合档位由搜索发现。普通基元、解析连接、窄门进出候选全部检查同一最小半径。只有第一遍失败且配置允许时，才进行包含原地旋转的第二遍搜索，并为旋转附加代价。粗二维距离场只引导搜索，不决定是否可通行。
5. 碰撞体使用 `robot_half_length/robot_half_width` 的矩形，附加配置净空。检查平移和旋转的整个运动段：细采样加相邻采样间运动上界膨胀，避免跳过薄墙。输出采样默认 5 cm / 5°，碰撞采样默认 1 cm / 1°。
6. 内部门洞段按实际几何指纹、全部规划参数和精确端点缓存。实际出生位置/朝向到第一门、最后一门到目标的段仍重新校验。

终点只约束位置，符合当前到达任务；门洞通过姿态约束为沿线路方向。返回 `PlanResult.poses[N,3]`（环境局部坐标 x/y/yaw）、`directions[N]`（到达该采样点的运动段方向：1/-1/0）、`arc_lengths[N]`、成功标志、失败原因、节点展开数和耗时。姿态单位为弧度。另提供 `rotation_count`、`gear_changes`、`requires_stop`、`minimum_moving_radius`、`fallback_attempted`。

移动段满足 `abs(turn) <= abs(distance) / min_turning_radius`。原地旋转不属于移动段，明确记作需要停车的事件；前进/倒退换向同样需要停车。`minimum_moving_radius` 根据精确圆弧采样的弦长和航向变化计算，不把原地旋转混入该指标。无曲线的路径返回 `inf`。

0.50 m 是用于改善参考路径跟踪的几何约束，不是差速车的机械转弯极限。相切连接在同向行驶段之间保持航向连续，但直线/圆弧连接处允许曲率跳变；这一版没有曲率变化率或角加速度的轨迹优化，也不生成全局最优 Reeds–Shepp 路径。`inplace_rotation_fallback=False` 只禁止停车原地转向，搜索仍允许倒车和换向。

2026-10-26：解析连接在最小半径候选失败后，按 `analytic_radius_factors` 尝试更大的转弯半径；默认对应 0.50 / 0.55 / 0.75 / 1.00 m，移动段始终遵守最小半径。扫掠校验在粗采样的保守包络拒绝、但未膨胀采样仍无碰撞时，最多把采样加密至原来的 4 倍。每次仍要求相邻采样间的保守扫掠包络无碰撞；没有降低净空或只检查离散点。

这是几何可行路径，**没有时间参数化，也不是可直接行为克隆的专家 action**。未建模低通、加速度限制和轮胎动力学，未保证全局最优。一般 random_obstacles 场景暂不使用此生成器。

## 奖励接入

成功规划的路径接入已有沿程进度/横向偏离奖励，保留原奖励权重。路径不进入 observation，网络输入、ONNX 和部署端均无需因此变更。

进度投影限制在上一帧进度附近的可达弧长窗口；静止旋转不产生平移进度，倒走保留负进度。跨越下一门洞需要实际位置越过门洞线且当前矩形车体横向范围落在净宽内，避免邻近平行路径的投影跳段。超过配置单步位移上限的跳变不会解锁门洞。

搜索超时、起点碰撞或点数超限时返回空路径，记录原因。此回合参考路径标志为无效，继续使用已有普通目标距离奖励；不会返回未通过碰撞校验的折线，也不重新抽出生点改变任务分布。

`failed=time_budget seconds≈3` 通常表示某一段先用完 1.5 秒常规搜索预算，再用完 1.5 秒停车旋转后备预算。`expansions` 是累计展开节点数，少于 12000 也可以先超时。这个日志不证明几何无解。

## 诊断与导出

TensorBoard：`ReferencePlanner/success_rate`、`mean_reset_plan_ms`、`cache_hits`、`expansions`、`failures` 等为进程启动以来累计统计。`stop_paths` / `stop_path_rate` 统计包含原地转向或换向的路径；`rotation_paths` / `rotation_path_rate` 只统计包含原地转向的路径。`Reference/active_fraction` 是全部环境中的有效参考路径比例，不是规划成功率。失败时控制台输出 `[ReferencePlanner] env=... level=... failed=...`。

已有环境对象 `env` 中可取得最近一次完整结果：

```python
import numpy as np
base = env.unwrapped
manager = base._world_manager
for env_id, plan in manager.last_reference_plans.items():
    if plan.success:
        np.savez_compressed(
            f"reference_env_{env_id}.npz",
            poses=plan.poses, directions=plan.directions,
            arc_lengths=plan.arc_lengths,
            requires_stop=plan.requires_stop,
            rotation_count=plan.rotation_count,
            gear_changes=plan.gear_changes,
        )
print(manager._reference_bank.failure_reasons)
```

关闭参考奖励时不会进行规划，因而可能没有 `last_reference_plans` / `_reference_bank`。

失败日志还包含失败段编号（`leg=1/4`）、实际起终点。默认在 `logs/reference_planner_failures/<时间戳>/` 保存实际查询 NPZ，同一环境/课程/几何/参数只保存第一份，避免反复写盘。快照包含车体与规划参数、墙体、圆柱、门洞、实际出生朝向，可在不启动 Isaac Sim 的情况下重放：

```bash
/home/jz/isaacsim/python.sh scripts/diagnostics/replay_reference_failure.py <失败快照.npz> --png /tmp/reference_replay.png
```

重放成功退出码为 0，仍失败为 2；可用 `--max_seconds 3` 单独对照预算，不需要改训练配置。

## 验证及范围

在 160 仓库根目录执行：

```bash
/home/jz/isaacsim/python.sh -m unittest discover -s tests -p 'test*reference*.py' -v
/home/jz/isaacsim/python.sh -m unittest discover -s tests -p 'test_straight_detour_layout.py' -v
```

2026-10-22 验证：14 项参考路径测试、6 项既有布局回归测试通过。覆盖矩形朝向、0.60 m 直通道正反向、旋转/薄墙碰撞、绕宽墙搜索、失败空路径、缓存起点与几何更新、原地旋转、反向进度、门洞解锁和投影跳段。

当前课程宽度范围的 16 个离线布局，6 级课程、双向和缓存重复，共 384 次查询均成功。首次查询中位数约 5.6 ms，重复查询约 2.0 ms；这不是全量 4096 环境训练吞吐测试，实际耗时取决于搜索难度与 CPU 负载。

2026-10-23 半径约束版本：20 项参考模块测试通过。新增解析连接随机端点/朝向校验、普通与后备移动段统一半径约束、停车旋转仅后备出现，以及 0.60 m 门洞双向 S 弯/车尾净空测试。

同样的 16 个布局、6 级、双向及缓存重复，对 0.35 / 0.50 / 0.70 m 各查询 384 次，共 1152 次均成功。0.50 m 组无停车旋转或换向，首次查询中位约 1.6 ms，重复约 1.3 ms。0.70 m 组有 4 条查询结果需要换向停车，没有原地旋转；这说明增大半径未必带来更方便的路径。统计仅代表这些离线布局，未测全量训练吞吐。

两个 CPU Isaac 环境完成启动、六级课程查询、reset、12 步奖励/观测有限值检查和旧折线模式切换，Hybrid180 observation 仍为 196 维。启动期间还出现 kinematic 物体写入速度的 PhysX 报错，未阻止这些检查；本次没有修改物理对象创建/速度写入逻辑。

0.60 m 直通道的矩形可行性已经验证；将所有门洞强制为 0.60 m 的复杂连续转弯布局仍可能触发搜索预算。预算失败不等于几何无解，不应因此关闭碰撞校验。当前用户配置的课程宽度未改动。策略是否更愿意绕障仍须重新训练和评估。

本次验证脚本与日志暂存 `/tmp/emma_reference_20261022/`；修改前源文件备份在 `/home/jz/cqz/reference_planner_backup_20261022/`。

2026-10-23 对应脚本/结果位于 `/tmp/emma_radius_20261023/`，源码修改前备份位于 `/home/jz/cqz/reference_radius_backup_20261023/`。批量统计保存到 `logs/reference_paths/radius_20261023/benchmark_radius_report.json`。

2026-10-26：依据 2026-10-23_11-09-23 训练保存配置与 seed=42 重建 env 4071，在第 5 级第一道约 0.815 m 的门前复现了同类约 3 秒超时（原日志没有实际回合位姿，不能证明是同一回合）。修复后的该查询约 5 ms 成功，无需展开搜索或停车。对保存 seed 下全部 128 个窄道、六级课程及最难级多组位置/朝向，共 2176 次查询均成功，中位约 1.3 ms、最大约 2.9 ms。实际碰撞拒绝、薄墙扫掠和该失败几何已纳入回归测试；22 项参考模块测试、两环境 reset/step、失败快照写盘与去重、离线重放验证通过。该批验证不是对所有随机 seed 或更窄课程的完备保证。

## PNG 与仿真内可视化

在 160 仓库根目录，导出当前真实场景六个课程等级的参考路径，无需 checkpoint：

```bash
/home/jz/isaacsim/python.sh scripts/diagnostics/visualize_reference_paths.py --headless --device cpu --all_levels
```

生成目录为 `logs/reference_paths/<时间戳>/`。每张 PNG 包含场地边界、实际墙体和障碍物、起终点、门洞净宽、前进/倒退路径、车头方向和间隔采样的矩形车体轮廓。失败结果会标明 `FAILED: <原因>`，不绘制无效路径。

在有图形桌面的终端直接预览仿真（默认定位到所选环境的俯视视角）：

```bash
/home/jz/isaacsim/python.sh scripts/diagnostics/visualize_reference_paths.py --level 5
```

省略 `--headless` 会保持窗口直到关闭。预览脚本不加载策略、不下发 action，机器人保持 reset 状态；它用于检查路径，不用于验证跟踪效果。可用 `--seed 42` 固定布局，`--num_envs 2 --env_ids 1` 查看第二个环境，`--output_dir <目录>` 更改导出目录。`--all_levels` 导出全部六级，窗口最终显示第 5 级。

可加 `--min_turning_radius 0.70` 对照其他半径，或 `--no_rotation_fallback` 禁止后备原地旋转。PNG 标题显示最小半径配置、原地旋转段数、换向次数及 `NO STOP` / `STOP REQUIRED`；红色叉标记需要停车的位置，仿真内也显示这些标记。

如要在原有 train/play 运行中显示策略对应的参考路径，在 `emma_obs_nav_env_cfg.py` 设置：

```python
reference_path_visualize = True
reference_path_visualize_env_ids = ()  # 自动选首个窄道环境；也可指定 (3968, 3969)
reference_path_visualize_height = 0.12
reference_path_export_png = True
reference_path_export_dir = "logs/reference_paths"
reference_path_export_max_per_env = 1  # 每环境首次 reset 导出；0 表示每次 reset
```

默认两个可视化开关均为 False，避免训练自动绘图。空编号配置自动选择实际 dedicated detour 环境，4096 环境且窄道数量 128 时通常为 env 3968；单环境时为 env 0。显式编号必须属于窄道环境，否则报错。此版本可视化要求启用 `hybrid_astar` 参考路径。

仿真内颜色：绿色前进、橙色倒退、蓝色箭头为车头方向（倒退时与运动方向相反）、青色为门洞、绿色叉为起点、紫色叉为终点。图元位于 `/World/ReferencePathDebug/env_<编号>`，随该环境 reset 更新；失败时清空旧路径。只创建 USD 曲线，没有碰撞属性，也未加入任务的 Warp 激光网格。不会修改路径、奖励、观测或控制命令。

半径约束版本六级 PNG 示例位于 `logs/reference_paths/radius_20261023/2026-10-23_10-48-58_382618/`。

可视化验证：16 项参考模块测试通过；2 环境六级 reset 实测生成 6 张 PNG，并检查 USD 图元的环境坐标偏移和无碰撞属性。另通过 Isaac RTX 相机实际渲染，确认路径、门洞线和方向箭头可见；检查失败结果清空路径线段。测试图片位于 `logs/reference_paths/visual_check_20261022/2026-10-22_20-38-57_412189/`。

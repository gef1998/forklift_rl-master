# Emma action 输出标定

159 实车和 160 仿真使用同一份 `scripts/emma_action_calibration.py`。该模块统一生成动作、差速运动学、CSV 字段、延迟估计和图片；两端只保留执行器/反馈适配差异。程序不创建 observation、不加载策略，也不计算奖励。

每个测试轴都执行同一条平滑序列：

```text
静止 → 正满速 → 负满速 → 静止
```

每段速度使用端点斜率为零的半余弦曲线。命令给出的加速度是曲线最大斜率；程序按该加速度自动计算持续时间。默认分别测试：

- 线速度：`0.3,0.6,0.9 m/s²`，满速 `±0.66 m/s`。
- 角速度：`0.6,1.2,1.88 rad/s²`，满速 `±1.5 rad/s`。

线速度测试 action 为 `[a,0]`，角速度测试 action 为 `[0,a]`。左右轮由同一差速公式生成，轮半径 0.05 m、轮距 0.43 m。

## 159 实车

标定节点向正常 mini_servo mux 输入 `/visual_servo_vel` 发布，订阅 `/mux_vel` 记录底盘实际收到的下发命令，订阅 `/jz_cmd_feedback` 记录左右轮反馈，并订阅 `/emma_odom_vel` 记录与策略 observation 相同来源的车体线速度和角速度。运行前停止正常导航输出，确保速度 mux 选中标定输入。程序必须显式提供 `--execute`：

```bash
rosrun mini_servo emma_action_calibration_real.py --execute
```

自定义加速度：

```bash
rosrun mini_servo emma_action_calibration_real.py --execute \
  --linear-accelerations 0.2,0.4,0.6 \
  --angular-accelerations 0.4,0.8,1.2
```

中断或异常退出时会连续发布十次零速。默认输出目录为 `~/.ros/emma_action_calibration/<时间>/`。

## 160 仿真

```bash
cd /home/jz/cqz/forklift_rl-master
/home/jz/isaacsim/python.sh scripts/wheel_response_test.py \
  --task Emma-RL-ObsNav-Direct-v0 --headless
```

仿真脚本只调用 `_pre_physics_step/_apply_action` 和物理步进。它关闭 observation、奖励、终止处理、噪声、安全层和额外动作加速度限制，因此命令曲线只由标定 profile 决定。

## 输出

每次执行生成：

- `requested.csv`：给定 signed action 及按差速运动学得到的目标速度。
- `commanded.csv`：159 的 `/mux_vel` 或仿真实际执行器命令。
- `feedback.csv`：159 的 `/jz_cmd_feedback` 左右轮反馈与 `/emma_odom_vel` 车体反馈，或仿真关节/车体实测速度。
- 每个轴、每种加速度一张 PNG 和 SVG。
- `summary.json`：左右轮、线速度和角速度的估计延迟、MAE、RMSE 和峰值误差。

图片使用各消息实际时间轴叠加 requested、commanded 和 feedback，因此可以直接观察 mux 延迟和底盘响应延迟。估计延迟在 0–1 s 内搜索使 commanded 与 feedback RMSE 最小的时间偏移。

当前实车的左右轮反馈按 `jdrv_msgs/MotionSpeed.actuators[0].motors[0/1].data[0]` 解析，字段单位为 m/s，绘图时除以轮半径换算为 rad/s。`/jz_cmd_feedback.vx/w` 在当前底盘上未填充，不能作为车体实测速度；车体曲线来自 `/emma_odom_vel` 的 `geometry_msgs/Twist.linear.x/angular.z`。任一来源缺失时程序会明确报告相应曲线不可用。

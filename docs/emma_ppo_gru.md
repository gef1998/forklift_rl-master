# Emma PPO＋GRU 基线

任务名：`Emma-RL-ObsNav-GRU-Direct-v0`。
原任务 `Emma-RL-ObsNav-Direct-v0` 继续使用 MLP。
两个任务使用同一环境配置、134 维观测、动作定义、奖励和课程。
这次不加入可通行路径距离奖励、不提供真实地图或隐藏路径信息。

## 网络与训练

结构为 `134 → MLP(256,128,64) → GRU(128) → Beta 策略头 / 价值头`。
策略和价值共用编码器与 GRU；价值估计也能利用历史信息。
GRU 隐藏状态由 agent 管理，不在模型内部隐式推进或缓存。

配置位于 `source/emma_rl/emma_rl/tasks/direct/emma/agents/skrl_emma_obs_nav_gru_ppo_cfg.yaml`：

- `models.recurrent.hidden_size=128`，`num_layers=1`。
- `sequence_length=128`：50 Hz 控制下，每段反向传播覆盖 2.56 秒。
  隐藏状态会跨序列、跨 rollout 延续；2.56 秒不是推理记忆的硬性上限。
- `rollouts=512`、`mini_batches=128`，其余 PPO 参数复制自实现时的 MLP 配置。
- `EmmaRunner` 根据模型的循环状态声明选择 skrl 的 `PPO_RNN`。
  启动日志明确显示实际使用的循环算法和状态尺寸。
- 终止或超时后，只清零对应环境的隐藏状态；序列内部同样在回合边界切断记忆和梯度。
- skrl 2.0 的 `PPO` 和 `PPO_RNN` 原始 GAE 只使用 terminated 掩码。两种基线统一在
  `time_limit_bootstrap=False` 时将超时也作为 GAE 边界，避免自动重置后的奖励
  串入前一回合；GRU 基线暂不支持打开 timeout bootstrap。
  原 MLP 的网络与 checkpoint 格式保持兼容，新启动的 MLP 训练同样使用这一边界修正。

没有增加上一动作或真实位姿观测，以保持第一轮记忆基线的输入可比性。
已有线速度、角速度、目标及 LiDAR 信息均通过 GRU 形成历史状态。

## 使用

在 `/home/jz/cqz/forklift_rl-master` 下开始训练：

```bash
/home/jz/isaacsim/python.sh scripts/skrl/train.py \
  --task Emma-RL-ObsNav-GRU-Direct-v0 --num_envs 1024 --headless --seed 42
```

默认日志目录为 `logs/skrl/emma_obs_nav_gru_direct/`。
播放时选择同一个 GRU 任务：

```bash
/home/jz/isaacsim/python.sh scripts/skrl/play.py \
  --task Emma-RL-ObsNav-GRU-Direct-v0 --num_envs 1 \
  --checkpoint logs/skrl/emma_obs_nav_gru_direct/<run>/checkpoints/agent_<step>.pt
```

play 和 `scripts/skrl/test.py` 均在每个 step 后提交循环状态，并处理回合结束。
播放会从 checkpoint 对应的 `params/agent.yaml` 读取 GRU 网络配置，因此训练时
调整隐藏维度后仍可正确恢复。单独搬运 checkpoint 时应保留同级运行目录的 params。
评估仍使用完整长距离任务，默认无域随机化噪声。

恢复 GRU 训练时给 train 命令增加 `--checkpoint ...`，并使用与保存时一致的
网络配置。模型、优化器、归一化统计及课程状态恢复；新仿真回合的隐藏状态从零开始。
旧 MLP checkpoint 与 GRU 架构不兼容，首轮基线应从头训练。

小规模检查时需要减少 minibatch 数量，例如：

```bash
/home/jz/isaacsim/python.sh scripts/skrl/train.py \
  --task Emma-RL-ObsNav-GRU-Direct-v0 --num_envs 8 --headless --max_iterations 2 \
  agent.agent.rollouts=128 agent.agent.mini_batches=2 agent.agent.learning_epochs=1
```

`rollouts` 必须能被 `sequence_length` 整除，且每个 minibatch 必须包含整数条
完整序列。非法配置会在训练前报错，不能把不同环境的历史拼成一条序列。
隐藏状态存储增加显存占用：默认 512 步、128 维、float32，1024 环境额外约
256 MiB，仅指 rollout 中的共享 GRU 状态，不含梯度和 minibatch 临时张量。

## 验证与对照

```bash
/home/jz/isaacsim/python.sh tests/test_gru_policy.py -v
```

测试覆盖历史影响、单步/序列输出一致性、终止与超时掩码、跨回合梯度隔离、
minibatch 时间顺序、Beta 似然、实际 PPO 更新、播放状态推进以及保存恢复。
短仿真验证只用于确认训练和播放链路可用，不代表已经学会宽墙绕行。

本次验证包含 GRU 专项及 MLP 超时边界测试、7 项已有 ObsNav Beta 测试、
6 项评估脚本测试，以及实际 GPU 训练、续训和 play 重放。
完整 `test_beta_policy.py` 还包含一个既有 Costmap 测试，引用已不存在的
`emma_costmap_nav/costmap_nav_network.py`；该项仍失败，本次没有修改 Costmap 模块。

建议固定评估种子，分别记录宽墙、U 型死胡同和回型通道的成功率、超时率、
停滞时间和重复折返次数；以相同训练样本预算比较 MLP 和 GRU。
本基线同时使用 GRU 策略与价值函数，新增参数量也应计入实验说明。

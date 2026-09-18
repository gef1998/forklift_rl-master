# Emma PPO＋GRU 基线

任务名：`Emma-RL-ObsNav-GRU-Direct-v0`。
原任务 `Emma-RL-ObsNav-Direct-v0` 继续使用 MLP。
两个任务使用同一环境配置、136 维观测、动作定义、奖励和课程。
这次不加入可通行路径距离奖励、不提供真实地图或隐藏路径信息。

## 网络与训练

结构为 `136 → MLP(256,128,64) → GRU(128) → Beta 策略头 / 价值头`。
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

观测在 134 维基础上追加策略自己上一步的 signed action（`[134:136]`，共 136 维），
用来打断"瞬时角速度 + 无动作记忆"这条单步延迟反馈通路。此前循环策略只能靠 GRU
隐状态记住自己发出的交替指令，而隐状态会把交替模式保持并放大。
已有线速度、角速度、目标及 LiDAR 信息仍全部通过 GRU 形成历史状态。

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

## 上一动作观测与部署契约

观测布局的最后两维 `[134:136]` 是**策略上一步发出的 signed action**（∈[-1,1]²），
由环境已有的 `_prev_policy_actions` 缓冲直接提供，没有新增状态。
`DirectRLEnv.step` 的顺序是 `_get_rewards → _reset_idx → _get_observations`，
所以观测读到的正好是"本步策略输出"，刚重置的环境读到 0。

三个必须遵守的约束：

- 取的是 `_policy_actions`（delay/noise **之前**的原始策略输出），
  语义是"我要了什么"，不是"车轮实际执行了什么"。这一路因此是确定性的。
- **回合一开头必须置 0**，对齐 `_reset_idx` 的行为；否则部署首步的输入
  与训练分布不一致。GRU 隐状态重置是另一件事，不要混在一起。
- 部署侧必须自己维护这 2 个值的状态（上一次推理的输出），不能从轮速或
  IMU 反推。

**维度变更使所有旧 checkpoint 失效**：输入宽度进了编码器第一层权重形状，
134 维的 checkpoint 加载时会直接报 shape mismatch，必须重训。
`scripts/skrl/export.py` 从 checkpoint 推断维度，所以旧 checkpoint 仍能导出，
但导出的 ONNX 输入是 `[batch, 134]`，与新任务的 `[batch, 136]` 不是一回事。

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

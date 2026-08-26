"""Emma Costmap Navigation 网络 — ResNet 编码器 + 共享 MLP.

观测结构:
  costmap_obs:   (N, 2, 64, 64) — Ch0 静态层, Ch1 动态层
  robot_state:   (N, 8)          — goal_dir(2), goal_dist(1), vel(2), ang_vel(1), prev_act(2)

编码流程:
  ResNet(costmap) → AdaptiveAvgPool → flatten(256) → cat(robot_state(8)) → MLP(264→128→64) → heads
"""

import torch
import torch.nn as nn
from skrl.models.torch import GaussianMixin, DeterministicMixin, Model


def _conv3x3(in_ch, out_ch, stride=1):
    """3×3 卷积, padding=1 保持空间尺寸."""
    return nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)


class _ResBlock(nn.Module):
    """BasicBlock: 两个 3×3 conv + BN + ReLU + 残差连接.

    若输入/输出通道或 stride 不同, shortcut 用 1×1 conv 对齐.
    """

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = _conv3x3(in_ch, out_ch, stride)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = _conv3x3(out_ch, out_ch, 1)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

        # shortcut: 1×1 conv 对齐通道/空间
        if in_ch != out_ch or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return self.relu(out)


class EmmaCostmapShared(GaussianMixin, DeterministicMixin, Model):
    """ResNet 编码器 + 共享 MLP.

    结构:
      Stem:    Conv(2→16, k3) → BN → ReLU → MaxPool(2)        → 16×32×32
      Stage1:  _ResBlock(16→16)              → MaxPool(2)       → 16×16×16
      Stage2:  _ResBlock(16→32)              → MaxPool(2)       → 32×8×8
      Stage3:  _ResBlock(32→32)                                  → 32×8×8
      Pool:    AdaptiveAvgPool2d(4) → flatten → 512
      Fusion:  cat(512, robot_state(8)) = 520
      MLP:     Linear(520→256) → ELU → Linear(256→128) → ELU
      Heads:   policy_head(128→2) → Sigmoid, value_head(128→1)
    """

    def __init__(self, observation_space, state_space, action_space, parameters, device):
        # ---- 基础初始化 ----
        Model.__init__(
            self, observation_space=observation_space, state_space=state_space,
            action_space=action_space, device=device,
        )

        # ---- 解析观测空间 ----
        if isinstance(observation_space, dict):
            self.costmap_shape = list(observation_space["costmap_obs"])  # [C, H, W]
            self.robot_state_dim = int(observation_space["robot_state_obs"])
        else:
            self.costmap_shape = [2, 64, 64]
            self.robot_state_dim = 8
        C, H, W = self.costmap_shape
        self._costmap_flat_dim = C * H * W

        # ---- 策略/价值参数 ----
        policy_params = parameters["policy"]
        value_params = parameters["value"]

        GaussianMixin.__init__(
            self,
            clip_actions=policy_params["clip_actions"],
            clip_log_std=policy_params["clip_log_std"],
            min_log_std=policy_params["min_log_std"],
            max_log_std=policy_params["max_log_std"],
            role="policy",
        )
        DeterministicMixin.__init__(
            self, clip_actions=value_params["clip_actions"], role="value"
        )

        # ---- ResNet 编码器 ----
        # Stem: (C, 64, 64) → (16, 32, 32)
        self.stem = nn.Sequential(
            nn.Conv2d(C, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        # Stage1: (16, 32, 32) → (16, 16, 16)
        self.layer1 = nn.Sequential(
            _ResBlock(16, 16),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        # Stage2: (16, 16, 16) → (32, 8, 8)
        self.layer2 = nn.Sequential(
            _ResBlock(16, 32, stride=1),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        # Stage3: (32, 8, 8) → (32, 8, 8)
        self.layer3 = _ResBlock(32, 32)

        # 自适应池化: (32, 8, 8) → (32, 4, 4)
        self.avgpool = nn.AdaptiveAvgPool2d((4, 4))

        # costmap 特征维度: 32 × 4 × 4 = 512
        self._cnn_out_dim = 32 * 4 * 4

        # 融合后维度
        self._fused_dim = self._cnn_out_dim + self.robot_state_dim

        # ---- 共享 MLP: 520 → 512 → 256 → 128 → 64 ----
        self.shared_mlp = nn.Sequential(
            nn.Linear(self._fused_dim, 512),
            nn.ELU(),
            nn.Linear(512, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, 64),
            nn.ELU(),
        )

        # ---- 输出头 ----
        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))
        self.policy_head = nn.Sequential(
            nn.Linear(64, self.num_actions),
            nn.Sigmoid(),
        )
        self.value_head = nn.Linear(64, 1)

    def _encode_observations(self, observations: torch.Tensor) -> torch.Tensor:
        """拆分 flat observations, ResNet 编码 costmap, 拼接 robot_state."""
        costmap_flat = observations[:, :self._costmap_flat_dim]
        robot_state = observations[:, self._costmap_flat_dim:]
        C, H, W = self.costmap_shape

        x = costmap_flat.reshape(-1, C, H, W)
        x = self.stem(x)                          # (N, 16, 32, 32)
        x = self.layer1(x)                        # (N, 16, 16, 16)
        x = self.layer2(x)                        # (N, 32, 8, 8)
        x = self.layer3(x)                        # (N, 32, 8, 8)
        x = self.avgpool(x)                       # (N, 32, 4, 4)
        x = x.reshape(x.size(0), -1)              # (N, 512)

        return torch.cat([x, robot_state], dim=-1)  # (N, fused_dim)

    def act(self, inputs, role):
        if role == "policy":
            return GaussianMixin.act(self, inputs, role=role)
        elif role == "value":
            return DeterministicMixin.act(self, inputs, role=role)

    def compute(self, inputs, role):
        if role == "policy":
            encoded = self._encode_observations(inputs["observations"])
            self._shared_output = self.shared_mlp(encoded)
            return self.policy_head(self._shared_output), {"log_std": self.log_std_parameter}

        elif role == "value":
            if self._shared_output is None:
                encoded = self._encode_observations(inputs["observations"])
                shared_output = self.shared_mlp(encoded)
            else:
                shared_output = self._shared_output
                self._shared_output = None
            return self.value_head(shared_output), {}

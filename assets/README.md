# 运行资产

本目录保存项目运行和场景验证所需的精选 USD 资产。所有 `.usd` 文件实际均为二进制 USDC 格式，并通过仓库根目录的 `.gitattributes` 交由 Git LFS 管理。

| 文件 | 用途 | 当前接入状态 |
|---|---|---|
| `Emma_no_ground.usd` | Emma AMR 本体 | Emma Nav 与 Emma Costmap 直接加载 |
| `obstacle_cube.usd` | 立方体障碍物参考资产 | 场景调试/后续资产化 |
| `obstacle_cylinder.usd` | 圆柱障碍物参考资产 | 场景调试/后续资产化 |
| `obs_nav.usd` | 避障导航场景 | 独立场景验证 |
| `warehouse.usd` | 仓库场景 | 独立场景验证 |

默认从本目录读取 `Emma_no_ground.usd`。如果需要临时使用外部机器人资产，可设置：

```bash
export EMMA_USD_PATH=/absolute/path/to/Emma_no_ground.usd
```

核心环境目前仍由 Python 程序化生成障碍物和仓库布局，因此其余四个 USD 不会在训练启动时自动加载。

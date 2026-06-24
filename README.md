# DexJoCo 中 Evo-RL/ReCap 思路的复现实验

这个仓库是我整理后的项目代码，用于在 DexJoCo 仿真环境中复现
Evo-RL/ReCap 的训练思路。

需要说明的是，这不是官方 Evo-RL 仓库，也不应该被当作官方实现使用。
本仓库主要保留 ReCap 的核心训练循环，并根据 DexJoCo 的数据和训练流程做了适配：

- DexJoCo 与 OpenPI/pi0.5 用于语言条件下的仿真 rollout、策略微调和评测。
- PyTorch/LeRobot 兼容代码用于数据集处理、value model 训练、value 推理、
  advantage 标注和 ACP indicator 生成。
- JAX/OpenPI 用于策略训练和策略服务。

公开的实验脚本在 [`dexjoco-recap/`](dexjoco-recap/) 中。
适配后的 LeRobot/Evo 风格源码在 [`Evo-RL-main/`](Evo-RL-main/) 中。

## 仓库结构

```text
dexjoco-recap/
  README.md
  configs/remote.env.example
  jobs/
  scripts/
  docs/real_robot_data.md

Evo-RL-main/
  src/lerobot/
  pyproject.toml
```

旧的临时实验、生成日志、上传清单和私有远程配置没有放进整理后的仓库视图中。

## 快速开始

先安装 LeRobot 兼容源码：

```bash
cd Evo-RL-main
conda create -y -n dexjoco-recap python=3.10
conda activate dexjoco-recap
python -m pip install -U pip
python -m pip install -e .
```

具体实验流程、远程运行方式和训练命令见：

```text
dexjoco-recap/README.md
```

## 关于 checkpoint

如果目的是在新的 DexJoCo 数据或真机数据上重新训练，则不需要本仓库提供训练好的
checkpoint。只需要把 value/policy 训练命令中的数据集路径或 `repo_id` 换成新的
数据集即可。

只有在以下情况才需要 checkpoint：

- 直接评测或部署一个已经训练好的策略。
- 跳过 value/policy 训练，只做后续推理或复现实验中的某个中间阶段。

## 致谢

本项目是基于 LeRobot、Evo-RL/ReCap、DexJoCo 和 OpenPI 等公开工具与思路做的
复现和适配。引用或介绍时，建议表述为独立的 DexJoCo 复现实验代码，而不是任何
官方项目的实现。

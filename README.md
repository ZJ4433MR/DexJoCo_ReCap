# DexJoCo ReCap 复现实验与真机数据训练支持

本仓库是整理后的实验代码，用于在 DexJoCo 仿真环境中复现 ReCap / Evo-RL 风格的训练闭环，并保留把同一套 value/ACP 代码迁移到 LeRobot 格式真机数据上的入口。

需要说明的是，这不是官方 Evo-RL 仓库，也不应被当作官方实现使用。这里的代码主要围绕本项目的 DexJoCo 实验做了工程整理：

- DexJoCo 与 OpenPI/pi0.5 用于语言条件下的仿真 rollout、策略微调和评测。
- PyTorch/LeRobot 兼容代码用于数据集读取、Pistar06 value model 训练、value 推理、advantage 标注和 ACP indicator 生成。
- JAX/OpenPI 用于策略训练、策略服务和 DexJoCo evaluation。
- 真机数据部分默认要求数据已经是 LeRobot 格式，或者至少可以先转换成 LeRobot 格式；本仓库提供 value/ACP 训练和标注入口，真机策略部署仍需要对应机器人的 OpenPI 配置与控制接口。

公开的实验脚本在 [`dexjoco-recap/`](dexjoco-recap/) 中；适配后的 LeRobot 兼容源码在 [`lerobot-src/`](lerobot-src/) 中。

## 仓库内容

```text
dexjoco-recap/
  README.md
  configs/remote.env.example
  jobs/
  scripts/
  docs/

lerobot-src/
  src/lerobot/
  pyproject.toml
```

各部分的作用如下：

- [`dexjoco-recap/`](dexjoco-recap/)：实验入口目录，包含远程运行、DexJoCo rollout、value/ACP 训练、OpenPI 微调和评测相关脚本。
- [`dexjoco-recap/jobs/`](dexjoco-recap/jobs/)：可直接提交到远程机器的 job 脚本。前面的编号是实验顺序，包含 smoke test、DexJoCo baseline、ReCap 多轮训练、LeRobot 数据闭环，以及真机 LeRobot 数据的 value/ACP 模板。
- [`dexjoco-recap/scripts/`](dexjoco-recap/scripts/)：通用工具脚本，包括远程打包提交、DexJoCo NPZ 转 LeRobot、数据池合并、episode success 标注、Pistar06 依赖准备、OpenPI ACP prompt patch 等。
- [`dexjoco-recap/configs/remote.env.example`](dexjoco-recap/configs/remote.env.example)：远程机器配置模板。实际的 `remote.env` 里会包含 SSH alias、环境初始化命令和 Slurm 参数，不应提交到仓库。
- [`dexjoco-recap/docs/`](dexjoco-recap/docs/)：补充说明文档，包括真机数据训练说明和 DexJoCo/OpenPI 语言策略说明。
- [`lerobot-src/`](lerobot-src/)：本实验使用的 LeRobot 兼容源码。这里保留了 value training / value inference / dataset report 等入口，并加入了 ReCap 实验需要的 Pistar06 value 与 ACP 标注逻辑。

旧的临时实验输出、日志、checkpoint、下载的 DexJoCo 源码、私有远程配置和与具体机器相关的路径都没有放进整理后的仓库视图中。

## 快速开始

先安装 LeRobot 兼容源码：

```bash
cd lerobot-src
conda create -y -n dexjoco-recap python=3.10
conda activate dexjoco-recap
python -m pip install -U pip
python -m pip install -e .
```

具体实验流程、远程运行方式和训练命令见：

```text
dexjoco-recap/README.md
```

如果只需要在新的真机 LeRobot 数据上训练 value model 并生成 ACP 标签，可以直接看：

```text
dexjoco-recap/docs/real_robot_data.md
dexjoco-recap/jobs/70_real_robot_lerobot_value_acp_template.sh
```

## 给真机数据使用者的交付边界

如果其他使用者希望拿这份代码去训练新的真机数据，通常需要提供：

1. 本仓库代码。
2. LeRobot 格式的真机数据集路径或 Hugging Face `repo_id`。
3. 数据集中实际使用的相机字段、状态字段、action 维度和成功标签字段。
4. 如果要继续做策略训练或真机部署，还需要对应机器人的 OpenPI dataset/action 配置和控制接口。

本仓库已经提供：

- `lerobot-value-train` / `lerobot.scripts.lerobot_value_train`：训练 Pistar06 value model。
- `lerobot-value-infer` / `lerobot.scripts.lerobot_value_infer`：对 LeRobot 数据写回 value、advantage 和 ACP indicator。
- `lerobot-dataset-report`：检查 LeRobot 数据集结构。
- `jobs/70_real_robot_lerobot_value_acp_template.sh`：真机 LeRobot 数据的 value/ACP 训练模板。
- DexJoCo 实验用的完整仿真闭环 job，可作为迁移到真机策略训练时的参考。

真机原始数据如果还不是 LeRobot 格式，需要先由真机采集侧转换成 LeRobot 数据集。本仓库无法替代具体机器人的采集、标定、控制和安全部署代码。

## 关于 checkpoint

如果目标是在新的 DexJoCo 数据或真机数据上重新训练，则不需要本仓库提供训练好的 checkpoint。只需要把 value/policy 训练命令中的数据集路径或 `repo_id` 换成新的数据集即可。

只有在以下情况下才需要 checkpoint：

- 直接评测或部署一个已经训练好的策略。
- 跳过 value/policy 训练，只做后续推理或复现实验中的某个中间阶段。

因此，如果要用新的真机数据重新训练，重点需要的是代码、环境说明、数据格式说明和训练命令，而不是本仓库的训练 checkpoint。

## 致谢

本项目基于 LeRobot、Evo-RL/ReCap、DexJoCo 和 OpenPI 等公开工具与研究思路做复现和适配。引用或介绍时，建议表述为独立的 DexJoCo 复现实验代码，而不是任何官方项目的实现。

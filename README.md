# DexJoCo ReCap 复现实验与真机数据训练支持

本仓库是整理后的实验代码，用于在 DexJoCo 仿真环境中复现 ReCap / Evo-RL 风格的训练闭环，并保留把同一套 value/ACP 代码迁移到 LeRobot 格式真机数据上的入口。它的重点不是发布一个新的通用机器人框架，而是把 DexJoCo 仿真数据、LeRobot value 训练、OpenPI/pi0.5 策略训练这几部分串成可以复现实验和继续迁移的数据流程。

需要说明的是，这不是官方 Evo-RL 仓库，也不应被当作官方实现使用。这里的代码主要围绕本项目的 DexJoCo 实验做了工程整理：

- DexJoCo 与 OpenPI/pi0.5 用于语言条件下的仿真 rollout、策略微调和评测。
- PyTorch/LeRobot 兼容代码用于数据集读取、Pistar06 value model 训练、value 推理、advantage 标注和 ACP indicator 生成。
- JAX/OpenPI 用于策略训练、策略服务和 DexJoCo evaluation。
- 真机数据部分默认要求数据已经是 LeRobot 格式，或者至少可以先转换成 LeRobot 格式；本仓库提供 value/ACP 训练和标注入口，真机策略部署仍需要对应机器人的 OpenPI 配置与控制接口。

公开的实验脚本在 [`dexjoco-recap/`](dexjoco-recap/) 中；适配后的 LeRobot 兼容源码在 [`lerobot-src/`](lerobot-src/) 中。

整体上可以把仓库理解成三层：

1. **DexJoCo 仿真实验层**：负责在仿真环境中收集 rollout、评测 pi0.5/OpenPI 策略，以及记录多轮 ReCap 实验结果。
2. **value/ACP 标注层**：负责从轨迹成功与剩余步数构造 value target，训练 value model，再把 value、advantage 和 ACP indicator 写回数据集。
3. **策略训练层**：负责把 ACP indicator 转成 prompt tag 或训练条件，再通过 OpenPI/JAX 进行策略微调和评测。

这三层容易被混在一起看。value model 本身不执行动作，真正输出动作的是 OpenPI/pi0.5 policy；ACP indicator 也不是环境原始 reward，而是基于 value/advantage 得到的训练标签。

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

## Value Model 路径

仓库里保留了两条 value model 路径，阅读代码时需要区分：

- `dexjoco-recap/scripts/dexjoco_label_recap_rollouts.py` 中的 `RecapValueNet` 是一个轻量 CNN value model。它直接读取 DexJoCo NPZ 中的双视角图像、state 和 action，用于早期或轻量的 NPZ 标注流程。这个脚本里的 `nn.Conv2d` 是故意保留的简化 backend，不代表完整的 Pistar06 路径。
- `lerobot-src/src/lerobot/values/pistar06/` 中的 Pistar06 是 faithful 的 LeRobot value backend。默认配置使用 SigLIP 作为图像编码器、Gemma 作为语言编码器，再接 MLP value head 输出 value 分布。它对应 Evo-RL/ReCap 中默认的 `--value.type=pistar06` 路径，也是 README 和真机数据模板推荐使用的路径。

因此，如果只看轻量 NPZ 脚本，会看到 CNN；如果看 `lerobot-value-train` 默认走的 Pistar06，则不是传统 CNN，而是视觉-语言 value model。两者都属于 value/ACP 标注层，但实验定位不同：轻量 CNN 便于快速调通 DexJoCo NPZ 流程，Pistar06 更接近 Evo-RL 的 LeRobot 数据流程，也更适合迁移到真机 LeRobot 数据。

Pistar06 也不是本仓库新提出的模型名称，而是 Evo-RL/ReCap 路径中使用的 `Pi*0.6` 风格 value backend。本仓库的工作是把这套 value training / inference / ACP 标注流程整理进 DexJoCo 复现实验，并补上与 OpenPI/JAX 策略训练的衔接。

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

这里的“支持真机数据”指的是：在真机数据已经整理成 LeRobot 格式后，可以复用 Pistar06 value training、value inference 和 ACP 标注代码。它不表示本仓库已经包含真机数据采集程序、机器人控制器、部署安全检查或真实硬件的 action adapter。策略训练如果要继续接真机，还需要把 OpenPI 的 dataset/action 配置适配到真实机器人的相机字段、状态字段和 action 维度。

## 关于 checkpoint

如果目标是在新的 DexJoCo 数据或真机数据上重新训练，则不需要本仓库提供训练好的 checkpoint。只需要把 value/policy 训练命令中的数据集路径或 `repo_id` 换成新的数据集即可。

只有在以下情况下才需要 checkpoint：

- 直接评测或部署一个已经训练好的策略。
- 跳过 value/policy 训练，只做后续推理或复现实验中的某个中间阶段。

因此，如果要用新的真机数据重新训练，重点需要的是代码、环境说明、数据格式说明和训练命令，而不是本仓库的训练 checkpoint。

如果要复现某一次已经完成的实验结果，checkpoint 会节省时间；但如果目标是让新的真机数据重新走 value/ACP 训练链路，则 checkpoint 反而不是必需交付物。新的数据通常需要重新训练 value model，再重新生成 advantage 和 ACP indicator。

## 致谢

本项目基于 LeRobot、Evo-RL/ReCap、DexJoCo 和 OpenPI 等公开工具与研究思路做复现和适配。引用或介绍时，建议表述为独立的 DexJoCo 复现实验代码，而不是任何官方项目的实现。

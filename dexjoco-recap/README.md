# DexJoCo ReCap 实验流程

这个目录包含 DexJoCo 仿真环境中的 ReCap / Evo-RL 风格复现实验脚本，以及把 value/ACP 部分迁移到 LeRobot 格式真机数据上的模板。

本项目不是官方 Evo-RL 仓库，而是面向 DexJoCo 的复现与适配。整体流程保留了 ReCap 的核心闭环：

```text
rollout 数据
  -> value model
  -> value / advantage / ACP indicator
  -> advantage-conditioned policy training
  -> DexJoCo evaluation
```

和原始项目相比，这里的实现同时用到了 PyTorch 和 JAX：

- PyTorch/LeRobot 兼容代码负责 LeRobot 数据集、Pistar06 value model 训练、value 推理和 ACP 标签生成。
- JAX/OpenPI 负责 pi0.5 策略微调、策略服务和 DexJoCo 评测。
- DexJoCo rollout 数据既可以保存为紧凑的 NPZ 文件，也可以转换成 LeRobot 格式数据集。

这里的 value model 和 policy model 是两个不同概念。value model 只估计当前 frame/trajectory 的归一化 return，用于计算 advantage 和 ACP indicator；真正产生机器人动作的是后续 OpenPI/pi0.5 policy。ACP indicator 也不是环境 reward 本身，而是 value/advantage 后处理得到的训练标签，通常会被注入 prompt tag 或用于条件化策略训练。

## 第一次阅读建议

如果想按一次完整实验理解整个仓库，先读 [`docs/full_workflow.md`](docs/full_workflow.md)。它把下面几个问题放在同一条线上讲清楚：

- DexJoCo 环境、OpenPI/pi0.5 policy、Pistar06 value model 和 LeRobot 数据格式分别负责什么。
- Pistar06 value model、SigLIP/Gemma backbone、OpenPI、LoRA-only 微调、ACP prompt patch 分别来自哪里。
- 一轮实验如何从初始数据 D0 出发，训练 value model，写回 value/advantage/ACP，再微调 policy，最后回 DexJoCo 收集新 rollout。

## 目录说明

```text
dexjoco-recap/
  configs/
  docs/
  jobs/
  scripts/
  README.md
```

- `configs/remote.env.example`：远程运行配置模板。复制成 `configs/remote.env` 后填写 SSH alias、远程根目录、环境初始化命令、Slurm 分区、GPU、内存和时间限制。
- `jobs/`：实验入口脚本。每个 job 都可以通过 `scripts/run_remote_slurm.ps1` 或 `scripts/run_remote_ssh.ps1` 提交运行，也可以在远程机器上手动执行。
- `scripts/run_remote_slurm.ps1`：从 Windows/PowerShell 本地打包当前实验目录、`../lerobot-src` 和可选 DexJoCo 源码，然后提交到 Slurm。
- `scripts/run_remote_ssh.ps1`：无 Slurm 时的 SSH 运行入口，和 Slurm runner 使用相同的 job script 接口。
- `scripts/dexjoco_common.sh`：DexJoCo/OpenPI 环境准备、checkpoint 下载、策略服务、评测和结果统计的公共函数。
- `scripts/dexjoco_pistar06_common.sh`：Pistar06 value model 训练所需依赖的安装和运行时检查。
- `scripts/dexjoco_npz_to_lerobot.py`：把 DexJoCo rollout NPZ 转成本地 LeRobot 数据集。
- `scripts/dexjoco_merge_lerobot_pool.py`：合并多轮 LeRobot 数据池，用于 ReCap 多轮训练。
- `scripts/dexjoco_lerobot_set_episode_success.py`：给 LeRobot 数据写入或规范化 episode success 标签。
- `scripts/dexjoco_openpi_lerobot_acp.sh`：给 OpenPI 的 LeRobot 数据读取流程打 patch，使 ACP indicator 可以注入到 prompt tag 中。
- `scripts/dexjoco_label_recap_rollouts.py`：早期/轻量 NPZ 标注路径，内部包含一个 CNN 形式的 `RecapValueNet`。它用于快速在紧凑 NPZ 上训练 value 和写回 advantage，不等同于默认的 Pistar06 LeRobot value backend。
- `docs/full_workflow.md`：完整闭环说明，适合第一次阅读仓库或写实验报告时引用。
- `docs/real_robot_data.md`：真机 LeRobot 数据训练说明，重点是 value/ACP 阶段的输入、命令和限制。
- `docs/dexjoco_language_policy.md`：DexJoCo 语言策略与 OpenPI 评测相关说明。

## 环境安装

先从仓库根目录安装 LeRobot 兼容源码：

```bash
cd ../lerobot-src
conda create -y -n dexjoco-recap python=3.10
conda activate dexjoco-recap
python -m pip install -U pip
python -m pip install -e .
```

DexJoCo 和 OpenPI 是外部依赖。远程任务脚本可以在远端自动 clone DexJoCo；如果远端访问 GitHub 不稳定，也可以提前准备本地 DexJoCo 源码：

```bash
git clone https://github.com/brave-eai/dexjoco.git .local/dexjoco-src
```

然后在提交远程任务时传入 `-LocalDexJoCoPath .local\dexjoco-src`。

## 远程运行

先复制一份私有配置文件：

```powershell
Copy-Item configs\remote.env.example configs\remote.env
```

然后编辑 `configs/remote.env`，填写自己的 SSH alias、环境初始化命令、Slurm 分区、GPU 申请、内存和时间限制。`configs/remote.env` 不应提交到仓库。

可以先跑一个 smoke test：

```powershell
.\scripts\run_remote_slurm.ps1 `
  -ConfigPath configs\remote.env `
  -LocalDexJoCoPath .local\dexjoco-src `
  -Job jobs\20_dexjoco_headless_smoke.sh `
  -RunName dexjoco_headless_smoke `
  -Time 02:00:00 `
  -Memory 96G
```

运行结果会拉回到：

```text
runs/<run_name>/
```

如果远程机器没有 Slurm，可以用 `scripts/run_remote_ssh.ps1`，job script 的环境变量接口保持一致。

## 主要 job

常用入口如下：

```text
jobs/20_dexjoco_headless_smoke.sh
  检查 DexJoCo 是否能在 headless 环境中启动并产生 observation。

jobs/21_dexjoco_pi05_water_plant_eval.sh
  在 water_plant 任务上评测公开 pi0.5 checkpoint。

jobs/22_dexjoco_pi05_single_arm_matrix.sh
  跑一个小规模 single-arm 任务矩阵，用于选择可行的 baseline task。

jobs/31_dexjoco_pi05_click_mouse_eval100.sh
  在 click_mouse 任务上评测公开 pi0.5 baseline。

jobs/43_dexjoco_click_mouse_recap_pistar06_full_eval100.sh
  NPZ 路径：收集 rollout、训练 Pistar06 value、推理 ACP 标签、微调 OpenPI，最后进行评测。

jobs/57_dexjoco_click_mouse_evorl_lerobot_ab.sh
  LeRobot 格式路径：合并数据池、训练 Pistar06 value、推理 indicator、使用 ACP prompt tag 微调 OpenPI、收集下一轮数据并评测。

jobs/68_dexjoco_click_mouse_evorl_lerobot_E_multitag_episode_smooth.sh
  multi-tag 版本：使用 failure / low / medium / high 等 prompt tag。

jobs/70_real_robot_lerobot_value_acp_template.sh
  真机 LeRobot 数据模板：不依赖 DexJoCo rollout，直接对已有 LeRobot 真机数据训练 value model 并写回 value / advantage / ACP indicator。
```

## A-F 实验代码与配置

这一组实验只对应 LeRobot 格式的 DexJoCo ReCap 闭环，核心执行脚本是：

```text
jobs/57_dexjoco_click_mouse_evorl_lerobot_ab.sh
```

A-E 都是围绕这个核心脚本写的 wrapper job。wrapper 只负责设置实验变量，然后 `exec` 到 `57_dexjoco_click_mouse_evorl_lerobot_ab.sh`。因此阅读 A-E 时，可以把 `57` 理解为公共训练/收集/评测主程序，把 `58/59/65/66/68` 理解为不同实验配置。

公共配置如下：

```text
DEXJOCO_TASK=click_mouse
DEXJOCO_EVO_ROUNDS=3
DEXJOCO_EVO_COLLECT_EPISODES=100
DEXJOCO_EVO_COLLECT_SHARD_EPISODES=25
DEXJOCO_RECAP_TRAIN_STEPS=1200
DEXJOCO_EVO_VALUE_STEPS=8000
DEXJOCO_EVO_VALUE_BATCH_SIZE=16
DEXJOCO_EVO_VALUE_DTYPE=bfloat16
DEXJOCO_EVO_VALUE_CAMERA_FEATURES=[observation.images.ego_right,observation.images.wrist]
DEXJOCO_EVO_N_STEP=50
DEXJOCO_EVO_POSITIVE_RATIO=0.3
DEXJOCO_EVO_INDICATOR_DROPOUT_PROB=0.3
OPENPI_RECAP_LORA_ONLY=1
```

公共流程是：

```text
D0 LeRobot 数据
  -> merge pool
  -> train Pistar06 value
  -> infer value / advantage / ACP indicator
  -> train OpenPI policy
  -> collect DexJoCo rollout
  -> convert rollout to LeRobot
  -> next round
  -> final eval
```

### A: faithful positive-collect

代码入口：

```text
jobs/58_dexjoco_click_mouse_evorl_lerobot_A_faithful.sh
```

核心配置：

```bash
DEXJOCO_EVO_LEROBOT_VARIANT=A_faithful_positive_collect
DEXJOCO_EVO_COLLECT_PROMPT=positive
DEXJOCO_EVO_TRAIN_ACP_ENABLE=1
DEXJOCO_EVO_ACP_BINARIZATION=task_quantile
DEXJOCO_EVO_SUCCESS_AWARE=false
DEXJOCO_EVO_TAG_VALUES=negative,positive
DEXJOCO_EVO_EVAL_PROMPT=positive
DEXJOCO_EVAL_EPISODES=100
OPENPI_RECAP_LORA_ONLY=1
```

含义：这是最接近 Evo-RL/ReCap positive prompt 设定的版本。收集 rollout 时使用带 `Advantage: positive` 的 prompt，policy training 时使用二值 ACP tag。

### B: base-collect controlled

代码入口：

```text
jobs/59_dexjoco_click_mouse_evorl_lerobot_B_base_collect.sh
```

核心配置：

```bash
DEXJOCO_EVO_LEROBOT_VARIANT=B_base_collect_controlled
DEXJOCO_EVO_COLLECT_PROMPT=base
DEXJOCO_EVO_TRAIN_ACP_ENABLE=1
DEXJOCO_EVO_ACP_BINARIZATION=task_quantile
DEXJOCO_EVO_SUCCESS_AWARE=false
DEXJOCO_EVO_TAG_VALUES=negative,positive
DEXJOCO_EVO_EVAL_PROMPT=positive
DEXJOCO_EVAL_EPISODES=100
OPENPI_RECAP_LORA_ONLY=1
```

含义：与 A 的训练路径保持一致，但收集新 rollout 时不再使用 positive prompt，而是使用原始 base task prompt。它用于分离“收集时 positive prompt”与“训练时 ACP prompt tag”的影响。

### C: base-collect no-ACP train

代码入口：

```text
jobs/65_dexjoco_click_mouse_evorl_lerobot_C_base_collect_no_acp.sh
```

核心配置：

```bash
DEXJOCO_EVO_LEROBOT_VARIANT=C_base_collect_no_acp_train
DEXJOCO_EVO_COLLECT_PROMPT=base
DEXJOCO_EVO_TRAIN_ACP_ENABLE=0
DEXJOCO_EVO_ACP_BINARIZATION=task_quantile
DEXJOCO_EVO_SUCCESS_AWARE=false
DEXJOCO_EVO_EVAL_PROMPT=base
DEXJOCO_EVAL_EPISODES=500
OPENPI_RECAP_LORA_ONLY=1
```

含义：仍然训练 Pistar06 value model，也仍然推理 value、advantage 和 indicator，但 policy training 阶段关闭 ACP prompt tag。这个实验用于对比“只做数据池更新/value 标注”与“真正把 ACP tag 注入 policy training”的差异。

### D: episode top-k smoothing

代码入口：

```text
jobs/66_dexjoco_click_mouse_evorl_lerobot_D_episode_topk_smooth.sh
```

核心配置：

```bash
DEXJOCO_EVO_LEROBOT_VARIANT=D_episode_topk_smooth_successaware
DEXJOCO_EVO_COLLECT_PROMPT=base
DEXJOCO_EVO_TRAIN_ACP_ENABLE=1
DEXJOCO_EVO_ACP_BINARIZATION=episode_topk_smooth
DEXJOCO_EVO_MIN_POSITIVE_RUN_LENGTH=3
DEXJOCO_EVO_SUCCESS_AWARE=true
DEXJOCO_EVO_TAG_VALUES=negative,positive
DEXJOCO_EVO_EVAL_PROMPT=base
DEXJOCO_EVAL_EPISODES=500
```

含义：保留 B 的 base-prompt collection 和二值 ACP policy training，但把 indicator 生成方式从全任务 quantile 改成 episode 内 top-k smoothing，并启用 success-aware 处理。这个版本主要用于减少零散帧级 positive tag。

### E: multi-tag episode smoothing

代码入口：

```text
jobs/68_dexjoco_click_mouse_evorl_lerobot_E_multitag_episode_smooth.sh
```

核心配置：

```bash
DEXJOCO_EVO_LEROBOT_VARIANT=E_multitag_episode_smooth_successaware
DEXJOCO_EVO_COLLECT_PROMPT=base
DEXJOCO_EVO_TRAIN_ACP_ENABLE=1
DEXJOCO_EVO_ACP_BINARIZATION=episode_multitag_smooth
DEXJOCO_EVO_MULTITAG_RATIOS=0.1,0.2,0.3
DEXJOCO_EVO_MIN_POSITIVE_RUN_LENGTH=3
DEXJOCO_EVO_SUCCESS_AWARE=true
DEXJOCO_EVO_TAG_KEY=Advantage
DEXJOCO_EVO_TAG_VALUES=failure,low,medium,high
DEXJOCO_EVO_EVAL_PROMPT=high
DEXJOCO_EVAL_EPISODES=500
```

含义：把 D 的二值 `negative/positive` tag 扩展成 `failure/low/medium/high` 四档 tag。`DEXJOCO_EVO_MULTITAG_RATIOS=0.1,0.2,0.3` 表示按 advantage 排序后划出 high、medium、low 等级，评测默认使用 `Advantage: high` prompt。

### F: 当前仓库状态

当前公开仓库没有单独的 F wrapper job，也没有已经提交的 `F_*` 实验配置文件。因此 README 中不把 F 描述成已经完成的可运行实验。

如果后续要补 F，建议延续 A-E 的写法，新建一个 wrapper，例如：

```text
jobs/69_dexjoco_click_mouse_evorl_lerobot_F_<name>.sh
```

并只在 wrapper 中设置 F 的差异变量，最后执行：

```bash
exec bash "$EXP_DIR/jobs/57_dexjoco_click_mouse_evorl_lerobot_ab.sh"
```

这样 F 会和 A-E 共用同一套训练、收集和评测主程序，差异也能集中体现在少量环境变量上。

运行 A-E 时，可以把 `-Job` 换成对应 wrapper：

```powershell
.\scripts\run_remote_slurm.ps1 `
  -ConfigPath configs\remote.env `
  -LocalDexJoCoPath .local\dexjoco-src `
  -Job jobs\58_dexjoco_click_mouse_evorl_lerobot_A_faithful.sh `
  -RunName A_faithful_positive_collect `
  -Time 24:00:00 `
  -Memory 128G
```

将上面的 `-Job` 和 `-RunName` 替换为 B-E 对应脚本和实验名即可。

## DexJoCo 数据流

紧凑的 rollout NPZ 文件可以转换成本地 LeRobot 数据集：

```bash
PYTHONPATH="../lerobot-src/src:$PYTHONPATH" \
python scripts/dexjoco_npz_to_lerobot.py \
  --input runs/<run_name>/rollouts.npz \
  --output-root data/click_mouse_lerobot \
  --repo-id local/click_mouse_lerobot \
  --task click_mouse \
  --fps 30 \
  --overwrite
```

多个 LeRobot 格式的数据池可以合并：

```bash
PYTHONPATH="../lerobot-src/src:$PYTHONPATH" \
python scripts/dexjoco_merge_lerobot_pool.py \
  --input local/base=data/base_pool \
  --input local/round1=data/round1_pool \
  --output-root data/click_mouse_pool \
  --output-repo-id local/click_mouse_pool \
  --overwrite
```

## Value Model 训练

本仓库中有两种 value model backend，容易被混淆：

- 轻量 NPZ backend：`scripts/dexjoco_label_recap_rollouts.py` 里的 `RecapValueNet`，输入是 DexJoCo NPZ 中的 base/wrist 图像、state 和 action，图像部分使用 `nn.Conv2d`，因此它确实是 CNN。这个 backend 主要用于快速调通 NPZ 数据流和轻量标注。
- Pistar06 backend：`../lerobot-src/src/lerobot/values/pistar06/` 里的 LeRobot-compatible value model，默认图像编码器是 SigLIP，语言编码器是 Gemma，然后接 value head。它对应 `--value.type=pistar06`，是更接近 Evo-RL/ReCap 的 faithful 路径，也是 LeRobot 数据和真机数据说明中默认推荐的路径。

下面这个命令走的是 Pistar06 backend，不是轻量 CNN backend：

```bash
cd ../lerobot-src
PYTHONPATH="src:$PYTHONPATH" python -m lerobot.scripts.lerobot_value_train \
  --dataset.repo_id=local/click_mouse_pool \
  --dataset.root=../dexjoco-recap/data/click_mouse_pool \
  --value.type=pistar06 \
  --value.dtype=bfloat16 \
  --value.vision_repo_id=google/siglip-so400m-patch14-384 \
  --value.language_repo_id=google/gemma-3-270m \
  --value.camera_features="[observation.images.ego_right,observation.images.wrist]" \
  --value.normalization_mapping="{VISUAL: IDENTITY, STATE: QUANTILES, ACTION: IDENTITY}" \
  --value.device=cuda \
  --value.push_to_hub=false \
  --batch_size=16 \
  --steps=8000 \
  --save_checkpoint=true \
  --save_freq=8000 \
  --wandb.enable=false \
  --output_dir=outputs/value_train/click_mouse_pistar06 \
  --job_name=click_mouse_pistar06
```

## Value 推理和 ACP 标签

value model 训练完成后，把 value、advantage 和 ACP indicator 写回数据集：

```bash
PYTHONPATH="src:$PYTHONPATH" python -m lerobot.scripts.lerobot_value_infer \
  --dataset.repo_id=local/click_mouse_pool \
  --dataset.root=../dexjoco-recap/data/click_mouse_pool \
  --dataset.success_field=episode_success \
  --dataset.default_success=failure \
  --inference.checkpoint_path=outputs/value_train/click_mouse_pistar06 \
  --runtime.device=cuda \
  --runtime.batch_size=16 \
  --acp.enable=true \
  --acp.n_step=50 \
  --acp.positive_ratio=0.3 \
  --acp.binarization=task_quantile \
  --acp.value_field=complementary_info.value \
  --acp.advantage_field=complementary_info.advantage \
  --acp.indicator_field=complementary_info.acp_indicator \
  --output_dir=outputs/value_infer/click_mouse_pistar06 \
  --job_name=click_mouse_pistar06_infer
```

multi-tag 版本可以使用：

```text
--acp.binarization=episode_multitag_smooth
--acp.multitag_ratios=0.1,0.2,0.3
--acp.indicator_field=complementary_info.acp_indicator
```

## JAX/OpenPI 策略训练

OpenPI 会在运行时打 patch，使带有 `complementary_info.acp_indicator` 的 LeRobot sample 能在 repack 前注入 prompt tag。主要辅助脚本是：

```text
scripts/dexjoco_openpi_lerobot_acp.sh
scripts/dexjoco_common.sh
```

job 脚本会设置以下关键环境变量：

```bash
OPENPI_LEROBOT_ACP_ENABLE=1
OPENPI_LEROBOT_ACP_INDICATOR_FIELD=complementary_info.acp_indicator
OPENPI_LEROBOT_ACP_INDICATOR_DROPOUT_PROB=0.3
OPENPI_LEROBOT_ACP_TAG_KEY=Advantage
OPENPI_LEROBOT_ACP_TAG_VALUES=negative,positive
OPENPI_RECAP_LORA_ONLY=1
```

然后启动 OpenPI 训练：

```bash
python scripts/compute_norm_stats.py click_mouse --batch-size=2 --num-workers=0
python scripts/train.py click_mouse \
  --exp-name=<exp_name> \
  --overwrite \
  --num-train-steps=1200 \
  --num-workers=0 \
  --save-interval=400 \
  --log-interval=50 \
  --fsdp-devices=1
```

## DexJoCo 评测

评测时先启动训练好的 OpenPI checkpoint 服务，再调用 `dexjoco-openpi-eval`：

```bash
python scripts/serve_policy.py \
  --port=23456 \
  policy:checkpoint \
  --policy.config=click_mouse \
  --policy.dir=<policy_checkpoint_dir>

dexjoco-openpi-eval \
  --config=configs/rand_obj/click_mouse.yaml \
  --seed=0 \
  --port=23456 \
  --host=127.0.0.1 \
  --episodes=100 \
  --output=eval_episodes
```

远程 job 会自动处理策略服务的启动/关闭、seed 循环和 success rate 汇总。

## 接真机数据

如果目标是在真机数据上重新训练，本仓库不要求提供预训练 checkpoint。只要真机数据已经整理成 LeRobot 格式，就可以把 DexJoCo 数据集参数替换成真机数据：

```text
--dataset.repo_id=<real_robot_repo_or_local_id>
--dataset.root=<optional_local_dataset_root>
--value.camera_features="[observation.images.front,observation.images.wrist]"
```

最直接的模板是：

```bash
REAL_ROBOT_REPO_ID=<real_robot_repo_or_local_id> \
REAL_ROBOT_ROOT=<optional_local_dataset_root> \
REAL_ROBOT_CAMERA_FEATURES="[observation.images.front,observation.images.wrist]" \
bash jobs/70_real_robot_lerobot_value_acp_template.sh
```

如果通过 Slurm runner 提交，可以在 `configs/remote.env` 的环境初始化部分或 job 前置环境变量中设置这些 `REAL_ROBOT_*` 变量，然后提交：

```powershell
.\scripts\run_remote_slurm.ps1 `
  -ConfigPath configs\remote.env `
  -Job jobs\70_real_robot_lerobot_value_acp_template.sh `
  -RunName real_robot_value_acp `
  -Time 08:00:00 `
  -Memory 96G
```

这一条真机模板会完成 value training 和 value/advantage/ACP 写回。是否继续做 policy training，取决于真机数据是否已经有可用的 OpenPI dataset/action 配置；如果没有，需要先根据真实机器人的 action 维度、相机字段和控制接口适配 OpenPI。

因此，真机数据侧最小需要准备的是 LeRobot 数据集本身、字段说明和成功标签。`jobs/70_real_robot_lerobot_value_acp_template.sh` 可以训练 Pistar06 value model 并写回 ACP 字段，但它不会替代真机采集、标定、控制和安全部署代码。

更具体的真机数据说明见：

```text
docs/real_robot_data.md
```

## 参考链接

- Evo-RL: https://github.com/MINT-SJTU/Evo-RL
- OpenPI: https://github.com/Physical-Intelligence/openpi
- DexJoCo: https://github.com/brave-eai/dexjoco
- LeRobot: https://github.com/huggingface/lerobot
- DexJoCo pi0.5 checkpoints: https://huggingface.co/DexJoCo/DexJoCo-Pi05
- DexJoCo LeRobot datasets: https://huggingface.co/datasets/DexJoCo/DexJoCo-Datasets-LeRobot

## 注意事项

- 不要提交 `configs/remote.env`、Hugging Face token、运行输出、checkpoint、视频或下载下来的 DexJoCo 源码。
- 脚本默认本地启动端是 PowerShell，远程机器是 Linux 路径。
- 如果远程机器访问 GitHub 不稳定，可以传入 `-LocalDexJoCoPath`，runner 会把本地 DexJoCo 源码一起打进任务包。
- 真机策略训练和部署需要额外的机器人配置与安全检查；本仓库提供的是数据训练与实验复现代码，不包含真实机器人控制栈。

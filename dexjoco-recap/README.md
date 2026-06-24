# DexJoCo 中 Evo-RL/ReCap 思路的复现实验

这个目录包含我在 DexJoCo 仿真环境中复现 Evo-RL/ReCap 思路时使用的实验脚本。

本项目不是官方 Evo-RL 仓库，而是面向 DexJoCo 的复现与适配。整体流程保留了
ReCap 的核心闭环：

```text
rollout 数据
  -> value model
  -> value / advantage / ACP indicator
  -> advantage-conditioned policy training
  -> DexJoCo evaluation
```

和原始项目相比，这里的实现同时用到了 PyTorch 和 JAX：

- PyTorch/LeRobot 兼容代码负责 LeRobot 数据集、Pistar06 value model 训练、
  value 推理和 ACP 标签生成。
- JAX/OpenPI 负责 pi0.5 策略微调、策略服务和 DexJoCo 评测。
- DexJoCo rollout 数据既可以保存为紧凑的 NPZ 文件，也可以转换成 LeRobot
  格式数据集。

## 环境安装

先从仓库根目录安装 LeRobot 兼容源码：

```bash
cd ../Evo-RL-main
conda create -y -n dexjoco-recap python=3.10
conda activate dexjoco-recap
python -m pip install -U pip
python -m pip install -e .
```

DexJoCo 和 OpenPI 是外部依赖。远程任务脚本可以在远端自动 clone DexJoCo；
如果远端访问 GitHub 不稳定，也可以提前准备本地 DexJoCo 源码：

```bash
git clone https://github.com/brave-eai/dexjoco.git .local/dexjoco-src
```

## 远程运行

PowerShell runner 会把本实验目录、LeRobot 兼容源码，以及可选的本地 DexJoCo
源码一起打包，然后提交到远程机器运行。

先复制一份私有配置文件：

```powershell
Copy-Item configs\remote.env.example configs\remote.env
```

然后编辑 `configs/remote.env`，填写自己的 SSH alias、环境初始化命令、Slurm
分区、GPU 申请、内存和时间限制。`configs/remote.env` 不应该提交到仓库。

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

如果远程机器没有 Slurm，可以用 `scripts/run_remote_ssh.ps1`，job script 的接口
保持一致。

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
  NPZ 路径：收集 rollout、训练 Pistar06 value、推理 ACP 标签、微调 OpenPI，
  最后进行评测。

jobs/57_dexjoco_click_mouse_evorl_lerobot_ab.sh
  LeRobot 格式路径：合并数据池、训练 Pistar06 value、推理 indicator、
  使用 ACP prompt tag 微调 OpenPI、收集下一轮数据并评测。

jobs/68_dexjoco_click_mouse_evorl_lerobot_E_multitag_episode_smooth.sh
  multi-tag 版本，使用 failure / low / medium / high 等 prompt tag。
```

## DexJoCo 数据流

紧凑的 rollout NPZ 文件可以转换成本地 LeRobot 数据集：

```bash
PYTHONPATH="../Evo-RL-main/src:$PYTHONPATH" \
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
PYTHONPATH="../Evo-RL-main/src:$PYTHONPATH" \
python scripts/dexjoco_merge_lerobot_pool.py \
  --input local/base=data/base_pool \
  --input local/round1=data/round1_pool \
  --output-root data/click_mouse_pool \
  --output-repo-id local/click_mouse_pool \
  --overwrite
```

## Value Model 训练

这里 faithful 的路径是通过 LeRobot 兼容代码训练 Pistar06 value model：

```bash
cd ../Evo-RL-main
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
  --dataset.default_success=success \
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

OpenPI 会在运行时打 patch，使带有 `complementary_info.acp_indicator` 的 LeRobot
sample 能在 repack 前注入 prompt tag。主要辅助脚本是：

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

这个仓库不要求提供我训练好的 checkpoint。如果老师想用真机数据重新训练，只要真机
数据已经整理成 LeRobot 格式，就可以把 DexJoCo 数据集参数换成真机数据：

```text
--dataset.repo_id=<real_robot_repo_or_local_id>
--dataset.root=<optional_local_dataset_root>
```

然后重新跑 value training、value inference 和 policy training。也就是说，老师如果
是要“拿代码去训练真机数据”，需要的是代码、环境说明和数据格式说明，不需要我的
checkpoint。

更具体的真机数据说明见：

```text
docs/real_robot_data.md
```

## 注意事项

- 不要提交 `configs/remote.env`、Hugging Face token、运行输出、checkpoint、
  视频或下载下来的 DexJoCo 源码。
- 脚本默认本地启动端是 PowerShell，远程机器是 Linux 路径。
- 如果远程机器访问 GitHub 不稳定，可以传入 `-LocalDexJoCoPath`，runner 会把本地
  DexJoCo 源码一起打进任务包。

# 真机 LeRobot 数据训练说明

本项目主要是在 DexJoCo 仿真环境中复现 ReCap / Evo-RL 风格的训练流程。真机数据部分的定位是：如果数据已经是 LeRobot 格式，或者可以先转换成 LeRobot 格式，同一套 value training 和 ACP 标注代码也可以直接用于真机数据。

这种用法不需要本仓库提供预训练 checkpoint。只有在跳过训练、直接评测或部署一个已经训练好的策略时，才需要额外提供 checkpoint。

## 本仓库能直接提供什么

对真机数据而言，本仓库已经提供以下可运行代码：

- `lerobot-dataset-report`：检查 LeRobot 数据集结构、字段、episode 数量和成功标签情况。
- `lerobot-value-train` / `python -m lerobot.scripts.lerobot_value_train`：在 LeRobot 数据上训练 Pistar06 value model。
- `lerobot-value-infer` / `python -m lerobot.scripts.lerobot_value_infer`：把 value、advantage 和 ACP indicator 写回 LeRobot 数据。
- `jobs/70_real_robot_lerobot_value_acp_template.sh`：面向真机 LeRobot 数据的一键模板，适合远程机器或 Slurm 环境。

本仓库不直接包含真实机器人的采集、标定、控制和部署代码。策略训练如果要继续接 OpenPI，需要真机侧提供匹配的 dataset/action 配置。

## 数据格式要求

推荐使用 LeRobot 数据集格式，至少包含：

- 一个或多个图像 observation，例如 `observation.images.front`、`observation.images.wrist`。
- 机器人状态，通常在 `observation.state` 或类似 observation/state 特征中。
- action 向量。
- task text 或任务描述。
- episode 级别或 frame 级别可解析的成功标签，例如 `episode_success`。如果没有明确成功标签，需要先定义规则，否则 value target 和 ACP 标注会失去监督信号。

可以先检查数据集结构：

```bash
cd ../lerobot-src
lerobot-dataset-report --dataset <repo_id_or_local_dataset> --root <optional_root>
```

需要重点确认：

- 相机字段名是否和训练命令中的 `--value.camera_features` 一致。
- 成功标签字段是否和 `--targets.success_field` / `--dataset.success_field` 一致。
- action 维度是否和后续策略训练配置一致。
- 数据集中是否包含 task 文本；如果没有，需要在转换数据时补齐。

## 一键模板

如果真机数据已经是 LeRobot 格式，可以使用仓库中的模板 job：

```bash
cd dexjoco-recap

REAL_ROBOT_REPO_ID=<real_robot_repo_or_local_id> \
REAL_ROBOT_ROOT=<optional_local_dataset_root> \
REAL_ROBOT_CAMERA_FEATURES="[observation.images.front,observation.images.wrist]" \
REAL_ROBOT_SUCCESS_FIELD=episode_success \
bash jobs/70_real_robot_lerobot_value_acp_template.sh
```

常用环境变量如下：

```text
REAL_ROBOT_REPO_ID
  必填。LeRobot 数据集 repo_id 或本地数据集 id。

REAL_ROBOT_ROOT
  可选。本地数据集路径。如果数据从 Hugging Face Hub 下载，可以不填。

REAL_ROBOT_CAMERA_FEATURES
  可选。value model 使用的相机字段列表，默认是 [observation.images.front,observation.images.wrist]。

REAL_ROBOT_SUCCESS_FIELD
  可选。成功标签字段，默认 episode_success。

REAL_ROBOT_DEFAULT_SUCCESS
  可选。当某些 episode 没有成功标签时的默认值，必须是 success 或 failure，默认 failure。

REAL_ROBOT_VALUE_STEPS
  可选。value model 训练步数，默认 8000。

REAL_ROBOT_VALUE_BATCH_SIZE
  可选。value model 和推理 batch size，默认 16。

REAL_ROBOT_ACP_BINARIZATION
  可选。ACP 标注方式，默认 task_quantile。也可以设为 episode_topk_smooth 或 episode_multitag_smooth。
```

如果通过远程 Slurm runner 提交，可以先把上述变量写入远程环境初始化命令，或者在 job 脚本外层导出，然后运行：

```powershell
.\scripts\run_remote_slurm.ps1 `
  -ConfigPath configs\remote.env `
  -Job jobs\70_real_robot_lerobot_value_acp_template.sh `
  -RunName real_robot_value_acp `
  -Time 08:00:00 `
  -Memory 96G
```

## 手动训练 value model

也可以手动运行训练命令。把 DexJoCo 数据集参数换成真机数据集：

```bash
cd ../lerobot-src

PYTHONPATH="src:$PYTHONPATH" python -m lerobot.scripts.lerobot_value_train \
  --dataset.repo_id=<real_robot_repo_or_local_id> \
  --dataset.root=<optional_local_dataset_root> \
  --targets.success_field=episode_success \
  --targets.default_success=failure \
  --value.type=pistar06 \
  --value.dtype=bfloat16 \
  --value.vision_repo_id=google/siglip-so400m-patch14-384 \
  --value.language_repo_id=google/gemma-3-270m \
  --value.camera_features="[observation.images.front,observation.images.wrist]" \
  --value.normalization_mapping="{VISUAL: IDENTITY, STATE: QUANTILES, ACTION: IDENTITY}" \
  --value.device=cuda \
  --value.push_to_hub=false \
  --batch_size=16 \
  --steps=8000 \
  --save_checkpoint=true \
  --save_freq=8000 \
  --wandb.enable=false \
  --output_dir=outputs/value_train/real_robot_pistar06 \
  --job_name=real_robot_pistar06
```

其中 `--value.camera_features`、`--targets.success_field` 和数据集路径需要按实际真机数据调整。

## Value 推理和 ACP 标签

训练完成后，把 value、advantage 和 indicator 字段写回数据集：

```bash
PYTHONPATH="src:$PYTHONPATH" python -m lerobot.scripts.lerobot_value_infer \
  --dataset.repo_id=<real_robot_repo_or_local_id> \
  --dataset.root=<optional_local_dataset_root> \
  --dataset.success_field=episode_success \
  --dataset.default_success=failure \
  --inference.checkpoint_path=outputs/value_train/real_robot_pistar06 \
  --runtime.device=cuda \
  --runtime.batch_size=16 \
  --runtime.num_workers=2 \
  --acp.enable=true \
  --acp.n_step=50 \
  --acp.positive_ratio=0.3 \
  --acp.binarization=task_quantile \
  --acp.value_field=complementary_info.value \
  --acp.advantage_field=complementary_info.advantage \
  --acp.indicator_field=complementary_info.acp_indicator \
  --output_dir=outputs/value_infer/real_robot_pistar06 \
  --job_name=real_robot_pistar06_infer
```

这一阶段完成后，LeRobot 数据中会带有后续 ACP prompt 训练需要的字段。

## 策略训练

如果真机数据可以被 OpenPI 训练配置读取，可以把 OpenPI 的 dataset root 指向真机 LeRobot 数据集，并保留 ACP prompt 相关环境变量：

```bash
OPENPI_LEROBOT_ACP_ENABLE=1
OPENPI_LEROBOT_ACP_INDICATOR_FIELD=complementary_info.acp_indicator
OPENPI_LEROBOT_ACP_INDICATOR_DROPOUT_PROB=0.3
OPENPI_LEROBOT_ACP_TAG_KEY=Advantage
OPENPI_LEROBOT_ACP_TAG_VALUES=negative,positive
```

如果真机机械臂的 embodiment、action 维度、相机字段或控制接口和 DexJoCo/OpenPI 配置不同，需要先适配 OpenPI 的 dataset/action 配置。即使策略训练配置还没有完全适配，value training 和 ACP 标注阶段仍然可以先在 LeRobot 格式数据上独立运行。

## 最小交付内容

如果目标是在新的真机数据上训练，通常需要交付：

```text
1. 本代码仓库；
2. LeRobot 格式真机数据集路径或 repo_id；
3. 数据字段说明：camera/state/action/task/success label；
4. value training 和 value inference 命令；
5. 如果要继续训练或部署策略，还需要真机侧 OpenPI 配置。
```

除非目标是直接评测或部署已经训练好的模型，否则不需要提供本仓库的预训练 checkpoint。

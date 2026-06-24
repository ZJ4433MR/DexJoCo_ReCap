# 使用真机数据训练

本项目主要是在 DexJoCo 仿真环境中复现 ReCap/Evo-RL 风格的训练流程。如果真机数据
已经是 LeRobot 格式，或者可以转换成 LeRobot 格式，同一套 value training 和
ACP 标注代码也可以用于真机数据。

这种用法不需要本仓库提供预训练 checkpoint。只有在跳过训练、直接评测或部署一个
已经训练好的策略时，才需要额外提供 checkpoint。

## 数据格式要求

推荐使用 LeRobot 数据集格式，至少包含：

- 一个或多个图像 observation，例如 `observation.images.front` 或
  `observation.images.wrist`。
- 机器人状态，通常在 observation/state 特征中。
- action 向量。
- task text 或任务描述。
- episode 级别或 frame 级别可解析的成功标签，例如 `episode_success`。

可以先检查数据集结构：

```bash
cd ../lerobot-src
lerobot-dataset-report --dataset <repo_id_or_local_dataset> --root <optional_root>
```

## 在真机数据上训练 value model

把 DexJoCo 数据集参数换成真机数据集：

```bash
cd ../lerobot-src
PYTHONPATH="src:$PYTHONPATH" python -m lerobot.scripts.lerobot_value_train \
  --dataset.repo_id=<real_robot_repo_or_local_id> \
  --dataset.root=<optional_local_dataset_root> \
  --value.type=pistar06 \
  --value.dtype=bfloat16 \
  --value.camera_features="[observation.images.front,observation.images.wrist]" \
  --value.device=cuda \
  --value.push_to_hub=false \
  --batch_size=16 \
  --steps=8000 \
  --save_checkpoint=true \
  --output_dir=outputs/value_train/real_robot_pistar06 \
  --job_name=real_robot_pistar06
```

其中 `--value.camera_features` 需要按实际数据集里的相机字段调整。

## Value 推理和 ACP 标签

```bash
PYTHONPATH="src:$PYTHONPATH" python -m lerobot.scripts.lerobot_value_infer \
  --dataset.repo_id=<real_robot_repo_or_local_id> \
  --dataset.root=<optional_local_dataset_root> \
  --inference.checkpoint_path=outputs/value_train/real_robot_pistar06 \
  --runtime.device=cuda \
  --runtime.batch_size=16 \
  --acp.enable=true \
  --acp.n_step=50 \
  --acp.positive_ratio=0.3 \
  --acp.value_field=complementary_info.value \
  --acp.advantage_field=complementary_info.advantage \
  --acp.indicator_field=complementary_info.acp_indicator \
  --output_dir=outputs/value_infer/real_robot_pistar06 \
  --job_name=real_robot_pistar06_infer
```

这一步会把 value、advantage 和 indicator 字段写回数据集。

## 策略训练

如果真机数据可以被 OpenPI 训练配置读取，可以把 OpenPI 的 dataset root 指向真机
LeRobot 数据集，并保留 ACP prompt 相关环境变量：

```bash
OPENPI_LEROBOT_ACP_ENABLE=1
OPENPI_LEROBOT_ACP_INDICATOR_FIELD=complementary_info.acp_indicator
OPENPI_LEROBOT_ACP_INDICATOR_DROPOUT_PROB=0.3
OPENPI_LEROBOT_ACP_TAG_KEY=Advantage
OPENPI_LEROBOT_ACP_TAG_VALUES=negative,positive
```

如果真机机械臂的 embodiment、action 维度或相机字段和 DexJoCo/OpenPI 配置不同，
需要先适配 OpenPI 的 dataset/action 配置。即使策略训练配置还没有完全适配，
value training 和 ACP 标注阶段仍然可以先在 LeRobot 格式数据上独立运行。

## 给使用者的最小交付内容

如果目标是在新的真机数据上训练，通常需要提供：

```text
1. 本代码仓库；
2. LeRobot 格式真机数据集路径或 repo_id；
3. 对应的 value training / inference / policy training 命令。
```

除非目标是直接评测或部署已经训练好的模型，否则不需要提供本仓库的预训练 checkpoint。

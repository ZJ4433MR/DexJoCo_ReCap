# Training with Real-Robot Data

This project is primarily a DexJoCo simulation reproduction. The same value
training and ACP-labeling code can still be used with real-robot data if the
data is in, or can be converted to, LeRobot format.

No checkpoint from this repository is required for that use case. A checkpoint
is only needed if you want to skip training or directly evaluate an already
trained policy.

## Expected Dataset Format

The easiest path is a LeRobot dataset with:

- one or more image observations, for example `observation.images.front` or
  `observation.images.wrist`;
- robot state under an observation/state feature;
- action vectors;
- task text;
- episode-level or frame-resolvable success labels, usually `episode_success`.

Check a dataset with:

```bash
cd ../Evo-RL-main
lerobot-dataset-report --dataset <repo_id_or_local_dataset> --root <optional_root>
```

## Value Training on Real-Robot Data

Replace the DexJoCo dataset fields with the real-robot dataset:

```bash
cd ../Evo-RL-main
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

Adjust `--value.camera_features` to match the dataset schema.

## Value Inference and ACP Labels

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

This writes value/advantage/indicator fields back to the dataset.

## Policy Training

If the real-robot data can be consumed by the OpenPI training config, point the
OpenPI dataset root to the real-robot LeRobot dataset and keep the ACP prompt
environment variables:

```bash
OPENPI_LEROBOT_ACP_ENABLE=1
OPENPI_LEROBOT_ACP_INDICATOR_FIELD=complementary_info.acp_indicator
OPENPI_LEROBOT_ACP_INDICATOR_DROPOUT_PROB=0.3
OPENPI_LEROBOT_ACP_TAG_KEY=Advantage
OPENPI_LEROBOT_ACP_TAG_VALUES=negative,positive
```

If the robot embodiment or action dimension differs from the DexJoCo/OpenPI
config, the OpenPI dataset/action config must be adapted first. In that case,
the value and ACP-labeling stages can still run independently on LeRobot-format
data.

## Minimal Answer for a Supervisor

To train on new real-robot data, provide:

```text
1. this codebase,
2. the LeRobot-format real-robot dataset path or repo_id,
3. the exact value training / inference / policy training commands.
```

Do not provide a pretrained checkpoint unless the goal is direct evaluation or
deployment of a previously trained model.

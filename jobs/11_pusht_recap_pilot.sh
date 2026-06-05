#!/usr/bin/env bash
set -euo pipefail

# Short end-to-end pilot for the RECAP simulation pipeline.
# This is meant to validate the real training/inference path before spending
# L40 time on long runs with larger backbones.

cd "$EVORL_DIR"

DATASET_REPO="${DATASET_REPO:-lerobot/pusht}"
DATASET_EPISODES="${DATASET_EPISODES:-[0,1,2]}"
TAG="${TAG:-pusht_recap_pilot}"
POLICY_TYPE="${POLICY_TYPE:-act}"
BATCH_SIZE="${BATCH_SIZE:-2}"
POLICY_STEPS="${POLICY_STEPS:-20}"
VALUE_STEPS="${VALUE_STEPS:-8}"
VALUE_BATCH_SIZE="${VALUE_BATCH_SIZE:-2}"
VALUE_VISION_REPO="${VALUE_VISION_REPO:-hf-internal-testing/tiny-random-vit}"
VALUE_LANGUAGE_REPO="${VALUE_LANGUAGE_REPO:-tiny-random/qwen2.5}"

BASELINE_DIR="$OUTPUT_DIR/baseline_${TAG}"
VALUE_DIR="$OUTPUT_DIR/value_${TAG}"
INFER_DIR="$OUTPUT_DIR/value_infer_${TAG}"
RECAP_DIR="$OUTPUT_DIR/recap_${TAG}"

echo "[job] Installing pilot dependencies into temporary PYDEPS_DIR=$PYDEPS_DIR"
python - <<'PY'
import importlib.util
import os
import subprocess
import sys

index_url = "https://pypi.tuna.tsinghua.edu.cn/simple"
target = os.environ["PYDEPS_DIR"]

top_level_no_deps = {
    "accelerate": "accelerate>=1.10.0,<2.0.0",
    "datasets": "datasets>=4.0.0,<4.2.0",
    "diffusers": "diffusers>=0.27.2,<0.36.0",
    "draccus": "draccus==0.10.0",
    "gym_pusht": "gym-pusht>=0.1.5,<0.2.0",
    "huggingface_hub": "huggingface-hub>=0.34.2,<0.36.0",
    "safetensors": "safetensors>=0.4.3,<1.0.0",
    "termcolor": "termcolor>=2.4.0,<4.0.0",
    "tokenizers": "tokenizers>=0.22.0,<0.23.0",
    "transformers": "transformers>=4.57.1,<5.0.0",
}

deps = [
    "aiohttp",
    "av>=15.0.0,<16.0.0",
    "cloudpickle",
    "deepdiff>=7.0.1,<9.0.0",
    "dill<0.4.1,>=0.3.0",
    "einops>=0.8.0,<0.9.0",
    "filelock",
    "fsspec[http]<=2025.9.0,>=2023.1.0",
    "gymnasium>=1.1.1,<2.0.0",
    "hf-xet",
    "imageio[ffmpeg]>=2.34.0,<3.0.0",
    "importlib-metadata",
    "jsonlines>=4.0.0,<5.0.0",
    "mergedeep",
    "multiprocess<0.70.17",
    "numpy<2",
    "opencv-python-headless>=4.9.0,<4.13.0",
    "orderly-set",
    "packaging>=24.2,<26.0",
    "pandas",
    "Pillow>=10.0.0,<13.0.0",
    "protobuf>=4,<7",
    "psutil",
    "pyarrow>=21.0.0",
    "pygame>=2.5.2,<2.7.0",
    "pymunk>=6.6.0,<7.0.0",
    "pyserial>=3.5,<4.0",
    "pyyaml",
    "pyyaml-include~=1.4",
    "regex",
    "requests",
    "scipy>=1.10.1,<1.15",
    "scikit-image>=0.22.0,<0.26.0",
    "sentry-sdk",
    "setuptools>=71.0.0,<81.0.0",
    "shapely>=2.0.3,<3.0.0",
    "six",
    "toml",
    "tomli",
    "tqdm",
    "typing-extensions",
    "typing-inspect",
    "urllib3<3",
    "wandb>=0.24.0,<0.25.0",
    "xxhash",
]

missing_top = [pkg for module, pkg in top_level_no_deps.items() if importlib.util.find_spec(module) is None]
if missing_top:
    print("missing top-level packages:", missing_top)
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--target",
            target,
            "--no-deps",
            "-i",
            index_url,
            *missing_top,
        ]
    )

print("installing dependency packages:", deps)
subprocess.check_call(
    [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--target",
        target,
        "-i",
        index_url,
        *deps,
    ]
)
PY

echo "[job] Dataset: $DATASET_REPO"
echo "[job] Episodes: $DATASET_EPISODES"
echo "[job] Policy type: $POLICY_TYPE"
echo "[job] Policy steps: $POLICY_STEPS"
echo "[job] Value steps: $VALUE_STEPS"
echo "[job] Tiny value vision repo: $VALUE_VISION_REPO"
echo "[job] Tiny value language repo: $VALUE_LANGUAGE_REPO"

TRAIN_DATASET_ARGS=(
  --dataset.repo_id="$DATASET_REPO"
  --dataset.episodes="$DATASET_EPISODES"
  --dataset.video_backend=pyav
)

INFER_DATASET_ARGS=(
  --dataset.repo_id="$DATASET_REPO"
  --dataset.episodes="$DATASET_EPISODES"
  --dataset.download_videos=true
)

echo "[job] 1/4 Train BC baseline"
python -m lerobot.scripts.lerobot_train \
  "${TRAIN_DATASET_ARGS[@]}" \
  --policy.type="$POLICY_TYPE" \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --policy.pretrained_backbone_weights=null \
  --policy.chunk_size=10 \
  --policy.n_action_steps=10 \
  --policy.dim_model=128 \
  --policy.n_heads=4 \
  --policy.dim_feedforward=512 \
  --policy.n_encoder_layers=1 \
  --policy.n_vae_encoder_layers=1 \
  --policy.latent_dim=16 \
  --batch_size="$BATCH_SIZE" \
  --steps="$POLICY_STEPS" \
  --log_freq=5 \
  --eval_freq=0 \
  --save_checkpoint=true \
  --save_freq="$POLICY_STEPS" \
  --wandb.enable=false \
  --output_dir="$BASELINE_DIR" \
  --job_name="baseline_${TAG}"

echo "[job] 2/4 Train tiny value model"
python -m lerobot.scripts.lerobot_value_train \
  "${TRAIN_DATASET_ARGS[@]}" \
  --value.type=pistar06 \
  --value.dtype=float32 \
  --value.vision_repo_id="$VALUE_VISION_REPO" \
  --value.language_repo_id="$VALUE_LANGUAGE_REPO" \
  --value.device=cuda \
  --value.push_to_hub=false \
  --value.tokenizer_max_length=64 \
  --value.state_proj_dim=64 \
  --value.fusion_hidden_dim=64 \
  --value.fusion_num_layers=1 \
  --value.fusion_num_heads=4 \
  --value.scheduler_warmup_steps=0 \
  --value.scheduler_decay_steps="$VALUE_STEPS" \
  --batch_size="$VALUE_BATCH_SIZE" \
  --steps="$VALUE_STEPS" \
  --log_freq=2 \
  --save_checkpoint=true \
  --save_freq="$VALUE_STEPS" \
  --wandb.enable=false \
  --output_dir="$VALUE_DIR" \
  --job_name="value_${TAG}"

echo "[job] 3/4 Infer value/advantage/indicator"
python -m lerobot.scripts.lerobot_value_infer \
  "${INFER_DATASET_ARGS[@]}" \
  --inference.checkpoint_path="$VALUE_DIR" \
  --runtime.device=cuda \
  --runtime.batch_size="$VALUE_BATCH_SIZE" \
  --acp.enable=true \
  --acp.n_step=10 \
  --acp.positive_ratio=0.3 \
  --acp.value_field="complementary_info.value_${TAG}" \
  --acp.advantage_field="complementary_info.advantage_${TAG}" \
  --acp.indicator_field="complementary_info.acp_indicator_${TAG}" \
  --viz.enable=false \
  --output_dir="$INFER_DIR" \
  --job_name="infer_${TAG}"

echo "[job] 4/4 Train advantage-conditioned policy"
python -m lerobot.scripts.lerobot_train \
  "${TRAIN_DATASET_ARGS[@]}" \
  --policy.type="$POLICY_TYPE" \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --policy.pretrained_backbone_weights=null \
  --policy.chunk_size=10 \
  --policy.n_action_steps=10 \
  --policy.dim_model=128 \
  --policy.n_heads=4 \
  --policy.dim_feedforward=512 \
  --policy.n_encoder_layers=1 \
  --policy.n_vae_encoder_layers=1 \
  --policy.latent_dim=16 \
  --batch_size="$BATCH_SIZE" \
  --steps="$POLICY_STEPS" \
  --log_freq=5 \
  --eval_freq=0 \
  --acp.enable=true \
  --acp.indicator_field="complementary_info.acp_indicator_${TAG}" \
  --acp.indicator_dropout_prob=0.3 \
  --save_checkpoint=true \
  --save_freq="$POLICY_STEPS" \
  --wandb.enable=false \
  --output_dir="$RECAP_DIR" \
  --job_name="recap_${TAG}"

echo "[job] Pilot RECAP simulation finished"

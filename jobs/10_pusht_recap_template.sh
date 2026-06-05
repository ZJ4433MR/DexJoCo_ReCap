#!/usr/bin/env bash
set -euo pipefail

# This is a runnable template once the remote environment has Evo-RL dependencies installed.
# It intentionally writes all outputs under $OUTPUT_DIR, which the runner pulls back locally.

cd "$EVORL_DIR"

DATASET_REPO="${DATASET_REPO:-lerobot/pusht}"
TAG="${TAG:-pusht_l40}"
POLICY_TYPE="${POLICY_TYPE:-act}"
BATCH_SIZE="${BATCH_SIZE:-8}"
POLICY_STEPS="${POLICY_STEPS:-20000}"
VALUE_STEPS="${VALUE_STEPS:-4000}"
VALUE_BATCH_SIZE="${VALUE_BATCH_SIZE:-16}"
VALUE_LANGUAGE_REPO="${VALUE_LANGUAGE_REPO:-Qwen/Qwen2.5-0.5B}"

BASELINE_DIR="$OUTPUT_DIR/baseline_${TAG}"
VALUE_DIR="$OUTPUT_DIR/value_${TAG}"
INFER_DIR="$OUTPUT_DIR/value_infer_${TAG}"
RECAP_DIR="$OUTPUT_DIR/recap_${TAG}"

echo "[job] Dataset: $DATASET_REPO"
echo "[job] Policy type: $POLICY_TYPE"

echo "[job] 1/4 Train BC baseline"
python -m lerobot.scripts.lerobot_train \
  --dataset.repo_id="$DATASET_REPO" \
  --policy.type="$POLICY_TYPE" \
  --policy.device=cuda \
  --batch_size="$BATCH_SIZE" \
  --steps="$POLICY_STEPS" \
  --eval_freq=0 \
  --save_checkpoint=true \
  --save_freq="$POLICY_STEPS" \
  --wandb.enable=false \
  --output_dir="$BASELINE_DIR" \
  --job_name="baseline_${TAG}"

echo "[job] 2/4 Train value model"
python -m lerobot.scripts.lerobot_value_train \
  --dataset.repo_id="$DATASET_REPO" \
  --value.type=pistar06 \
  --value.dtype=bfloat16 \
  --value.language_repo_id="$VALUE_LANGUAGE_REPO" \
  --value.device=cuda \
  --batch_size="$VALUE_BATCH_SIZE" \
  --steps="$VALUE_STEPS" \
  --save_checkpoint=true \
  --save_freq="$VALUE_STEPS" \
  --wandb.enable=false \
  --output_dir="$VALUE_DIR" \
  --job_name="value_${TAG}"

echo "[job] 3/4 Infer value/advantage/indicator"
python -m lerobot.scripts.lerobot_value_infer \
  --dataset.repo_id="$DATASET_REPO" \
  --inference.checkpoint_path="$VALUE_DIR" \
  --runtime.device=cuda \
  --runtime.batch_size="$VALUE_BATCH_SIZE" \
  --acp.enable=true \
  --acp.n_step=50 \
  --acp.positive_ratio=0.3 \
  --acp.value_field="complementary_info.value_${TAG}" \
  --acp.advantage_field="complementary_info.advantage_${TAG}" \
  --acp.indicator_field="complementary_info.acp_indicator_${TAG}" \
  --output_dir="$INFER_DIR" \
  --job_name="infer_${TAG}"

echo "[job] 4/4 Train advantage-conditioned policy"
python -m lerobot.scripts.lerobot_train \
  --dataset.repo_id="$DATASET_REPO" \
  --policy.type="$POLICY_TYPE" \
  --policy.device=cuda \
  --batch_size="$BATCH_SIZE" \
  --steps="$POLICY_STEPS" \
  --eval_freq=0 \
  --acp.enable=true \
  --acp.indicator_field="complementary_info.acp_indicator_${TAG}" \
  --acp.indicator_dropout_prob=0.3 \
  --save_checkpoint=true \
  --save_freq="$POLICY_STEPS" \
  --wandb.enable=false \
  --output_dir="$RECAP_DIR" \
  --job_name="recap_${TAG}"

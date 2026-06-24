#!/usr/bin/env bash
set -euo pipefail

# Train a Pistar06 value model and write ACP fields for an existing real-robot
# LeRobot dataset. The dataset itself should already be in LeRobot format.

if [[ -z "${EXP_DIR:-}" ]]; then
  EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
cd "$EXP_DIR"

source scripts/dexjoco_common.sh
source scripts/dexjoco_pistar06_common.sh

RUN_ROOT="${RUN_ROOT:-$(_dexjoco_run_root)}"
OUTPUT_DIR="${OUTPUT_DIR:-$RUN_ROOT/outputs}"
EVORL_DIR="${EVORL_DIR:-$(cd "$EXP_DIR/../lerobot-src" && pwd)}"

export WANDB_MODE="${WANDB_MODE:-offline}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

REAL_ROBOT_REPO_ID="${REAL_ROBOT_REPO_ID:-}"
REAL_ROBOT_ROOT="${REAL_ROBOT_ROOT:-}"
REAL_ROBOT_SUCCESS_FIELD="${REAL_ROBOT_SUCCESS_FIELD:-episode_success}"
REAL_ROBOT_DEFAULT_SUCCESS="${REAL_ROBOT_DEFAULT_SUCCESS:-failure}"
REAL_ROBOT_VALUE_STEPS="${REAL_ROBOT_VALUE_STEPS:-8000}"
REAL_ROBOT_VALUE_BATCH_SIZE="${REAL_ROBOT_VALUE_BATCH_SIZE:-16}"
REAL_ROBOT_VALUE_NUM_WORKERS="${REAL_ROBOT_VALUE_NUM_WORKERS:-2}"
REAL_ROBOT_VALUE_DTYPE="${REAL_ROBOT_VALUE_DTYPE:-bfloat16}"
REAL_ROBOT_VALUE_DEVICE="${REAL_ROBOT_VALUE_DEVICE:-cuda}"
REAL_ROBOT_VALUE_LANGUAGE_REPO="${REAL_ROBOT_VALUE_LANGUAGE_REPO:-google/gemma-3-270m}"
REAL_ROBOT_VALUE_VISION_REPO="${REAL_ROBOT_VALUE_VISION_REPO:-google/siglip-so400m-patch14-384}"
REAL_ROBOT_CAMERA_FEATURES="${REAL_ROBOT_CAMERA_FEATURES:-[observation.images.front,observation.images.wrist]}"
REAL_ROBOT_VALUE_NORMALIZATION_MAPPING="${REAL_ROBOT_VALUE_NORMALIZATION_MAPPING:-{VISUAL: IDENTITY, STATE: QUANTILES, ACTION: IDENTITY}}"
REAL_ROBOT_N_STEP="${REAL_ROBOT_N_STEP:-50}"
REAL_ROBOT_POSITIVE_RATIO="${REAL_ROBOT_POSITIVE_RATIO:-0.3}"
REAL_ROBOT_ACP_BINARIZATION="${REAL_ROBOT_ACP_BINARIZATION:-task_quantile}"
REAL_ROBOT_MULTITAG_RATIOS="${REAL_ROBOT_MULTITAG_RATIOS:-0.1,0.2,0.3}"
REAL_ROBOT_MIN_POSITIVE_RUN_LENGTH="${REAL_ROBOT_MIN_POSITIVE_RUN_LENGTH:-1}"
REAL_ROBOT_SUCCESS_AWARE="${REAL_ROBOT_SUCCESS_AWARE:-false}"
REAL_ROBOT_C_FAIL_COEF="${REAL_ROBOT_C_FAIL_COEF:-1.0}"
REAL_ROBOT_VALUE_FIELD="${REAL_ROBOT_VALUE_FIELD:-complementary_info.value}"
REAL_ROBOT_ADVANTAGE_FIELD="${REAL_ROBOT_ADVANTAGE_FIELD:-complementary_info.advantage}"
REAL_ROBOT_INDICATOR_FIELD="${REAL_ROBOT_INDICATOR_FIELD:-complementary_info.acp_indicator}"
REAL_ROBOT_OUTPUT_NAME="${REAL_ROBOT_OUTPUT_NAME:-real_robot_value_acp}"

if [[ -z "$REAL_ROBOT_REPO_ID" ]]; then
  echo "[real-robot] REAL_ROBOT_REPO_ID is required." >&2
  echo "[real-robot] Example: REAL_ROBOT_REPO_ID=local/my_robot_dataset REAL_ROBOT_ROOT=/data/my_robot bash jobs/70_real_robot_lerobot_value_acp_template.sh" >&2
  exit 2
fi

if [[ "$REAL_ROBOT_DEFAULT_SUCCESS" != "success" && "$REAL_ROBOT_DEFAULT_SUCCESS" != "failure" ]]; then
  echo "[real-robot] REAL_ROBOT_DEFAULT_SUCCESS must be 'success' or 'failure'." >&2
  exit 2
fi

if [[ "$REAL_ROBOT_ACP_BINARIZATION" != "task_quantile" && "$REAL_ROBOT_ACP_BINARIZATION" != "episode_topk_smooth" && "$REAL_ROBOT_ACP_BINARIZATION" != "episode_multitag_smooth" ]]; then
  echo "[real-robot] REAL_ROBOT_ACP_BINARIZATION must be task_quantile, episode_topk_smooth, or episode_multitag_smooth." >&2
  exit 2
fi

if [[ "$REAL_ROBOT_SUCCESS_AWARE" != "true" && "$REAL_ROBOT_SUCCESS_AWARE" != "false" ]]; then
  echo "[real-robot] REAL_ROBOT_SUCCESS_AWARE must be true or false." >&2
  exit 2
fi

OUT_BASE="$OUTPUT_DIR/$REAL_ROBOT_OUTPUT_NAME"
VALUE_DIR="$OUT_BASE/value_train"
INFER_DIR="$OUT_BASE/value_infer"
REPORT_DIR="$OUT_BASE/dataset_report"
mkdir -p "$OUT_BASE" "$REPORT_DIR"

DEXJOCO_RECAP_PISTAR06_PYDEPS_DIR="${DEXJOCO_RECAP_PISTAR06_PYDEPS_DIR:-$RUN_ROOT/real_robot_value_pydeps}"
ensure_evorl_pistar06_deps
VALUE_PYTHONPATH="$DEXJOCO_RECAP_PISTAR06_PYDEPS_DIR:$EVORL_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
verify_evorl_pistar06_runtime "$VALUE_PYTHONPATH"

dataset_args=(--dataset.repo_id="$REAL_ROBOT_REPO_ID")
if [[ -n "$REAL_ROBOT_ROOT" ]]; then
  dataset_args+=(--dataset.root="$REAL_ROBOT_ROOT")
fi

summary_file="$OUT_BASE/real_robot_value_acp.summary.txt"
{
  echo "repo_id=$REAL_ROBOT_REPO_ID"
  echo "root=$REAL_ROBOT_ROOT"
  echo "camera_features=$REAL_ROBOT_CAMERA_FEATURES"
  echo "success_field=$REAL_ROBOT_SUCCESS_FIELD"
  echo "default_success=$REAL_ROBOT_DEFAULT_SUCCESS"
  echo "value_dir=$VALUE_DIR"
  echo "infer_dir=$INFER_DIR"
  echo "acp_binarization=$REAL_ROBOT_ACP_BINARIZATION"
} | tee "$summary_file"

echo "[real-robot] dataset report"
report_args=(--dataset "$REAL_ROBOT_REPO_ID")
if [[ -n "$REAL_ROBOT_ROOT" ]]; then
  report_args+=(--root "$REAL_ROBOT_ROOT")
fi
if command -v lerobot-dataset-report >/dev/null 2>&1; then
  PYTHONPATH="$VALUE_PYTHONPATH" lerobot-dataset-report "${report_args[@]}" > "$REPORT_DIR/report.txt"
else
  PYTHONPATH="$VALUE_PYTHONPATH" python -m lerobot.scripts.lerobot_dataset_report "${report_args[@]}" > "$REPORT_DIR/report.txt"
fi

echo "[real-robot] training value model"
cd "$EVORL_DIR"
PYTHONPATH="$VALUE_PYTHONPATH" python -m lerobot.scripts.lerobot_value_train \
  "${dataset_args[@]}" \
  --targets.success_field="$REAL_ROBOT_SUCCESS_FIELD" \
  --targets.default_success="$REAL_ROBOT_DEFAULT_SUCCESS" \
  --targets.c_fail_coef="$REAL_ROBOT_C_FAIL_COEF" \
  --value.type=pistar06 \
  --value.dtype="$REAL_ROBOT_VALUE_DTYPE" \
  --value.vision_repo_id="$REAL_ROBOT_VALUE_VISION_REPO" \
  --value.language_repo_id="$REAL_ROBOT_VALUE_LANGUAGE_REPO" \
  --value.camera_features="$REAL_ROBOT_CAMERA_FEATURES" \
  --value.normalization_mapping="$REAL_ROBOT_VALUE_NORMALIZATION_MAPPING" \
  --value.device="$REAL_ROBOT_VALUE_DEVICE" \
  --value.push_to_hub=false \
  --batch_size="$REAL_ROBOT_VALUE_BATCH_SIZE" \
  --num_workers="$REAL_ROBOT_VALUE_NUM_WORKERS" \
  --steps="$REAL_ROBOT_VALUE_STEPS" \
  --save_checkpoint=true \
  --save_freq="$REAL_ROBOT_VALUE_STEPS" \
  --wandb.enable=false \
  --output_dir="$VALUE_DIR" \
  --job_name=real_robot_pistar06

echo "[real-robot] writing value, advantage, and ACP fields"
PYTHONPATH="$VALUE_PYTHONPATH" python -m lerobot.scripts.lerobot_value_infer \
  "${dataset_args[@]}" \
  --dataset.success_field="$REAL_ROBOT_SUCCESS_FIELD" \
  --dataset.default_success="$REAL_ROBOT_DEFAULT_SUCCESS" \
  --inference.checkpoint_path="$VALUE_DIR" \
  --runtime.device="$REAL_ROBOT_VALUE_DEVICE" \
  --runtime.batch_size="$REAL_ROBOT_VALUE_BATCH_SIZE" \
  --runtime.num_workers="$REAL_ROBOT_VALUE_NUM_WORKERS" \
  --acp.enable=true \
  --acp.n_step="$REAL_ROBOT_N_STEP" \
  --acp.positive_ratio="$REAL_ROBOT_POSITIVE_RATIO" \
  --acp.binarization="$REAL_ROBOT_ACP_BINARIZATION" \
  --acp.multitag_ratios="$REAL_ROBOT_MULTITAG_RATIOS" \
  --acp.min_positive_run_length="$REAL_ROBOT_MIN_POSITIVE_RUN_LENGTH" \
  --acp.success_aware="$REAL_ROBOT_SUCCESS_AWARE" \
  --acp.c_fail_coef="$REAL_ROBOT_C_FAIL_COEF" \
  --acp.value_field="$REAL_ROBOT_VALUE_FIELD" \
  --acp.advantage_field="$REAL_ROBOT_ADVANTAGE_FIELD" \
  --acp.indicator_field="$REAL_ROBOT_INDICATOR_FIELD" \
  --viz.enable=false \
  --output_dir="$INFER_DIR" \
  --job_name=real_robot_pistar06_infer

echo "[real-robot] done"
echo "[real-robot] outputs: $OUT_BASE"

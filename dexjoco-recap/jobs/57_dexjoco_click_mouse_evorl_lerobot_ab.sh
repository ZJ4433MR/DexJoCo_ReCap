#!/usr/bin/env bash
set -euo pipefail

cd "$EXP_DIR"
source scripts/dexjoco_common.sh
source scripts/dexjoco_pistar06_common.sh
source scripts/dexjoco_openpi_lerobot_acp.sh

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.99}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"

RUN_ROOT="${RUN_ROOT:-$(_dexjoco_run_root)}"
export CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-$RUN_ROOT/conda_pkgs}"
mkdir -p "$CONDA_PKGS_DIRS"
DEXJOCO_EVO_SHARED_ROOT="${DEXJOCO_EVO_SHARED_ROOT:-${REMOTE_BASE:-$RUN_ROOT}}"

DEXJOCO_TASK="${DEXJOCO_TASK:-click_mouse}"
DEXJOCO_EVO_LEROBOT_VARIANT="${DEXJOCO_EVO_LEROBOT_VARIANT:-faithful_positive_collect}"
DEXJOCO_EVO_COLLECT_PROMPT="${DEXJOCO_EVO_COLLECT_PROMPT:-positive}"
DEXJOCO_EVO_ROUNDS="${DEXJOCO_EVO_ROUNDS:-3}"
DEXJOCO_EVO_COLLECT_EPISODES="${DEXJOCO_EVO_COLLECT_EPISODES:-100}"
DEXJOCO_EVO_COLLECT_SHARD_EPISODES="${DEXJOCO_EVO_COLLECT_SHARD_EPISODES:-25}"
DEXJOCO_EVO_COLLECT_SEED_BASE="${DEXJOCO_EVO_COLLECT_SEED_BASE:-20000}"
DEXJOCO_EVAL_SEED="${DEXJOCO_EVAL_SEED:-0}"
DEXJOCO_RECAP_EVAL_SEEDS="${DEXJOCO_RECAP_EVAL_SEEDS:-}"
DEXJOCO_EVAL_EPISODES="${DEXJOCO_EVAL_EPISODES:-100}"
DEXJOCO_RECAP_TRAIN_STEPS="${DEXJOCO_RECAP_TRAIN_STEPS:-1200}"
DEXJOCO_RECAP_BATCH_SIZE="${DEXJOCO_RECAP_BATCH_SIZE:-2}"
DEXJOCO_RECAP_FSDP_DEVICES="${DEXJOCO_RECAP_FSDP_DEVICES:-1}"
DEXJOCO_RECAP_NUM_WORKERS="${DEXJOCO_RECAP_NUM_WORKERS:-0}"
DEXJOCO_RECAP_WARMUP_STEPS="${DEXJOCO_RECAP_WARMUP_STEPS:-100}"
DEXJOCO_RECAP_SAVE_INTERVAL="${DEXJOCO_RECAP_SAVE_INTERVAL:-400}"
DEXJOCO_EVO_VALUE_STEPS="${DEXJOCO_EVO_VALUE_STEPS:-8000}"
DEXJOCO_EVO_VALUE_BATCH_SIZE="${DEXJOCO_EVO_VALUE_BATCH_SIZE:-16}"
DEXJOCO_EVO_VALUE_NUM_WORKERS="${DEXJOCO_EVO_VALUE_NUM_WORKERS:-2}"
DEXJOCO_EVO_VALUE_DTYPE="${DEXJOCO_EVO_VALUE_DTYPE:-bfloat16}"
DEXJOCO_EVO_VALUE_LANGUAGE_REPO="${DEXJOCO_EVO_VALUE_LANGUAGE_REPO:-google/gemma-3-270m}"
DEXJOCO_EVO_VALUE_VISION_REPO="${DEXJOCO_EVO_VALUE_VISION_REPO:-google/siglip-so400m-patch14-384}"
DEXJOCO_EVO_VALUE_CAMERA_FEATURES="${DEXJOCO_EVO_VALUE_CAMERA_FEATURES:-[observation.images.ego_right,observation.images.wrist]}"
DEXJOCO_EVO_VALUE_NORMALIZATION_MAPPING="${DEXJOCO_EVO_VALUE_NORMALIZATION_MAPPING:-{VISUAL: IDENTITY, STATE: QUANTILES, ACTION: IDENTITY}}"
DEXJOCO_EVO_N_STEP="${DEXJOCO_EVO_N_STEP:-50}"
DEXJOCO_EVO_POSITIVE_RATIO="${DEXJOCO_EVO_POSITIVE_RATIO:-0.3}"
DEXJOCO_EVO_ACP_BINARIZATION="${DEXJOCO_EVO_ACP_BINARIZATION:-task_quantile}"
DEXJOCO_EVO_MIN_POSITIVE_RUN_LENGTH="${DEXJOCO_EVO_MIN_POSITIVE_RUN_LENGTH:-1}"
DEXJOCO_EVO_SUCCESS_AWARE="${DEXJOCO_EVO_SUCCESS_AWARE:-false}"
DEXJOCO_EVO_C_FAIL_COEF="${DEXJOCO_EVO_C_FAIL_COEF:-1.0}"
DEXJOCO_EVO_INDICATOR_DROPOUT_PROB="${DEXJOCO_EVO_INDICATOR_DROPOUT_PROB:-0.3}"
DEXJOCO_EVO_TRAIN_ACP_ENABLE="${DEXJOCO_EVO_TRAIN_ACP_ENABLE:-1}"
DEXJOCO_EVO_VALUE_FIELD="${DEXJOCO_EVO_VALUE_FIELD:-complementary_info.value}"
DEXJOCO_EVO_ADVANTAGE_FIELD="${DEXJOCO_EVO_ADVANTAGE_FIELD:-complementary_info.advantage}"
DEXJOCO_EVO_INDICATOR_FIELD="${DEXJOCO_EVO_INDICATOR_FIELD:-complementary_info.acp_indicator}"
DEXJOCO_EVO_TAG_KEY="${DEXJOCO_EVO_TAG_KEY:-Advantage}"
DEXJOCO_EVO_TAG_VALUES="${DEXJOCO_EVO_TAG_VALUES:-negative,positive}"
DEXJOCO_EVO_MULTITAG_RATIOS="${DEXJOCO_EVO_MULTITAG_RATIOS:-0.1,0.2,0.3}"
DEXJOCO_EVO_EVAL_PROMPT="${DEXJOCO_EVO_EVAL_PROMPT:-positive}"
DEXJOCO_EVO_OUTPUT_NAME="${DEXJOCO_EVO_OUTPUT_NAME:-dexjoco_click_mouse_evorl_lerobot_${DEXJOCO_EVO_LEROBOT_VARIANT}}"
DEXJOCO_EVO_SOURCE_WORKSPACE_TAR="${DEXJOCO_EVO_SOURCE_WORKSPACE_TAR:-}"
DEXJOCO_RECAP_PISTAR06_PYDEPS_DIR="${DEXJOCO_RECAP_PISTAR06_PYDEPS_DIR:-$RUN_ROOT/evorl_value_pydeps}"
DEXJOCO_EVO_LEROBOT_FPS="${DEXJOCO_EVO_LEROBOT_FPS:-30}"
DEXJOCO_LEROBOT_DATASET_DOWNLOAD_ROOT="${DEXJOCO_LEROBOT_DATASET_DOWNLOAD_ROOT:-$DEXJOCO_EVO_SHARED_ROOT/_shared_dexjoco_official_lerobot}"
OPENPI_RECAP_LORA_ONLY="${OPENPI_RECAP_LORA_ONLY:-1}"

if [[ -z "${DEXJOCO_ACP_SUFFIX+x}" ]]; then
  DEXJOCO_ACP_SUFFIX=$'\nAdvantage: positive'
fi
export DEXJOCO_ACP_SUFFIX

if [[ "$DEXJOCO_EVO_COLLECT_PROMPT" != "positive" && "$DEXJOCO_EVO_COLLECT_PROMPT" != "base" ]]; then
  echo "[evorl-lerobot] DEXJOCO_EVO_COLLECT_PROMPT must be 'positive' or 'base'" >&2
  exit 2
fi
tag_value_allowed() {
  local needle="$1"
  local value
  IFS=',' read -r -a _tag_values <<< "$DEXJOCO_EVO_TAG_VALUES"
  for value in "${_tag_values[@]}"; do
    value="${value#"${value%%[![:space:]]*}"}"
    value="${value%"${value##*[![:space:]]}"}"
    if [[ "$needle" == "$value" ]]; then
      return 0
    fi
  done
  return 1
}

sanitize_tag_value() {
  local value="$1"
  value="$(printf '%s' "$value" | tr -cs 'A-Za-z0-9_-' '_')"
  value="${value##_}"
  value="${value%%_}"
  printf '%s' "${value:-tag}"
}

if [[ "$DEXJOCO_EVO_EVAL_PROMPT" != "positive" && "$DEXJOCO_EVO_EVAL_PROMPT" != "base" && "$DEXJOCO_EVO_EVAL_PROMPT" != "negative" ]] && ! tag_value_allowed "$DEXJOCO_EVO_EVAL_PROMPT"; then
  echo "[evorl-lerobot] DEXJOCO_EVO_EVAL_PROMPT must be 'positive', 'negative', 'base', or one of DEXJOCO_EVO_TAG_VALUES." >&2
  exit 2
fi
if [[ "$DEXJOCO_EVO_TRAIN_ACP_ENABLE" != "0" && "$DEXJOCO_EVO_TRAIN_ACP_ENABLE" != "1" ]]; then
  echo "[evorl-lerobot] DEXJOCO_EVO_TRAIN_ACP_ENABLE must be 0 or 1" >&2
  exit 2
fi
if [[ "$DEXJOCO_EVO_ACP_BINARIZATION" != "task_quantile" && "$DEXJOCO_EVO_ACP_BINARIZATION" != "episode_topk_smooth" && "$DEXJOCO_EVO_ACP_BINARIZATION" != "episode_multitag_smooth" ]]; then
  echo "[evorl-lerobot] DEXJOCO_EVO_ACP_BINARIZATION must be 'task_quantile', 'episode_topk_smooth', or 'episode_multitag_smooth'" >&2
  exit 2
fi
if [[ "$DEXJOCO_EVO_SUCCESS_AWARE" != "true" && "$DEXJOCO_EVO_SUCCESS_AWARE" != "false" ]]; then
  echo "[evorl-lerobot] DEXJOCO_EVO_SUCCESS_AWARE must be 'true' or 'false'" >&2
  exit 2
fi

OUT_BASE="$OUTPUT_DIR/$DEXJOCO_EVO_OUTPUT_NAME"
mkdir -p "$OUT_BASE"
SUMMARY="$OUT_BASE/evorl_lerobot_summary.tsv"
echo -e "stage\tpool_episodes\tpool_frames\tpolicy_step\tcollect_prompt\tcollect_successes\tcollect_episodes\teval_status\teval_successes\teval_episodes" | tee "$SUMMARY"

if [[ ! -d "$EXP_DIR/.local/dexjoco-src/openpi" && -n "$DEXJOCO_EVO_SOURCE_WORKSPACE_TAR" && -f "$DEXJOCO_EVO_SOURCE_WORKSPACE_TAR" ]]; then
  echo "[evorl-lerobot] restoring packaged DexJoCo fallback from $DEXJOCO_EVO_SOURCE_WORKSPACE_TAR"
  mkdir -p "$EXP_DIR/.local"
  tar -xzf "$DEXJOCO_EVO_SOURCE_WORKSPACE_TAR" -C "$EXP_DIR/.local" --strip-components=2 dexjoco-recap/.local/dexjoco-src
fi

prepare_dexjoco_source
setup_dexjoco_env
setup_openpi_env
relax_openpi_websocket_timeouts
DEXJOCO_TASK="$DEXJOCO_TASK" download_dexjoco_pi05_checkpoint
download_dexjoco_lerobot_dataset
ensure_evorl_pistar06_deps
EVORL_VALUE_PYTHONPATH="$DEXJOCO_RECAP_PISTAR06_PYDEPS_DIR:$EVORL_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
verify_evorl_pistar06_runtime "$EVORL_VALUE_PYTHONPATH"

base_config_rel="./configs/rand_obj/${DEXJOCO_TASK}.yaml"
base_config="$DEXJOCO_DIR/$base_config_rel"
positive_eval_config="$OUT_BASE/${DEXJOCO_TASK}_advantage_positive.yaml"
negative_eval_config="$OUT_BASE/${DEXJOCO_TASK}_advantage_negative.yaml"
tag_prompt_config_dir="$OUT_BASE/eval_prompt_configs"
mkdir -p "$tag_prompt_config_dir"
conda run --no-capture-output --prefix "$DEXJOCO_ENV_PREFIX" python - "$base_config" "$positive_eval_config" "$negative_eval_config" "$DEXJOCO_ACP_SUFFIX" "$tag_prompt_config_dir" "$DEXJOCO_EVO_TAG_KEY" "$DEXJOCO_EVO_TAG_VALUES" <<'PY'
from pathlib import Path
import re
import sys
import yaml

src = Path(sys.argv[1])
positive_dst = Path(sys.argv[2])
negative_dst = Path(sys.argv[3])
positive_suffix = sys.argv[4]
tag_config_dir = Path(sys.argv[5])
tag_key = sys.argv[6].strip() or "Advantage"
tag_values = [value.strip() for value in sys.argv[7].split(",") if value.strip()]

base_cfg = yaml.safe_load(src.read_text())
positive_cfg = dict(base_cfg)
positive_cfg["prompt"] = positive_cfg["prompt"].rstrip() + positive_suffix
positive_dst.write_text(yaml.safe_dump(positive_cfg, sort_keys=False))
print(f"[evorl-lerobot] positive eval prompt: {positive_cfg['prompt']}")

negative_cfg = dict(base_cfg)
negative_cfg["prompt"] = negative_cfg["prompt"].rstrip() + "\nAdvantage: negative"
negative_dst.write_text(yaml.safe_dump(negative_cfg, sort_keys=False))
print(f"[evorl-lerobot] negative eval prompt: {negative_cfg['prompt']}")

for tag_value in tag_values:
    safe = re.sub(r"[^A-Za-z0-9_-]+", "_", tag_value).strip("_") or "tag"
    tag_cfg = dict(base_cfg)
    tag_cfg["prompt"] = tag_cfg["prompt"].rstrip() + f"\n{tag_key}: {tag_value}"
    dst = tag_config_dir / f"{safe}.yaml"
    dst.write_text(yaml.safe_dump(tag_cfg, sort_keys=False))
    print(f"[evorl-lerobot] tag eval prompt {tag_value}: {tag_cfg['prompt']}")
PY

D0_ROOT="$RUN_ROOT/evorl_lerobot_${DEXJOCO_TASK}_official_d0_success"
D0_REPO_ID="local/${DEXJOCO_TASK}_official_d0_success"
PYTHONPATH="$EVORL_VALUE_PYTHONPATH" python "$EXP_DIR/scripts/dexjoco_lerobot_set_episode_success.py" \
  --root "$DEXJOCO_OFFICIAL_LEROBOT_ROOT" \
  --output-root "$D0_ROOT" \
  --label success \
  --overwrite \
  --summary-output "$OUT_BASE/d0_episode_success.summary.json"

detect_lerobot_use_videos() {
  PYTHONPATH="$EVORL_VALUE_PYTHONPATH" python - "$D0_ROOT" <<'PY'
from pathlib import Path
import json
import sys

info = json.loads((Path(sys.argv[1]) / "meta" / "info.json").read_text())
features = info.get("features", {})
print("1" if any(v.get("dtype") == "video" for v in features.values()) else "0")
PY
}
DEXJOCO_EVO_LEROBOT_USE_VIDEOS="$(detect_lerobot_use_videos)"

pool_inputs=("$D0_REPO_ID=$D0_ROOT")
current_policy_dir="../checkpoints/pi05_dexjoco_ckpt/$DEXJOCO_TASK"
current_pretrained_path="../checkpoints/pi05_dexjoco_ckpt/$DEXJOCO_TASK/params"

merge_pool() {
  local stage="$1"
  local pool_parent="$RUN_ROOT/evorl_lerobot_${DEXJOCO_EVO_LEROBOT_VARIANT}_${stage}_pool_parent"
  local pool_root="$pool_parent/$DEXJOCO_TASK"
  local pool_repo_id="local/${DEXJOCO_TASK}_${DEXJOCO_EVO_LEROBOT_VARIANT}_${stage}_pool"
  local merge_args=(
    --output-root "$pool_root"
    --output-repo-id "$pool_repo_id"
    --overwrite
    --summary-output "$OUT_BASE/${stage}_pool.summary.json"
  )
  for input in "${pool_inputs[@]}"; do
    merge_args+=(--input "$input")
  done
  PYTHONPATH="$EVORL_VALUE_PYTHONPATH" python "$EXP_DIR/scripts/dexjoco_merge_lerobot_pool.py" "${merge_args[@]}"
  POOL_PARENT="$pool_parent"
  POOL_ROOT="$pool_root"
  POOL_REPO_ID="$pool_repo_id"
}

train_value_and_infer() {
  local stage="$1"
  local value_dir="$OUT_BASE/${stage}_pistar06_value"
  local infer_dir="$OUT_BASE/${stage}_pistar06_value_infer"

  echo "[evorl-lerobot] value train stage=$stage repo=$POOL_REPO_ID root=$POOL_ROOT"
  cd "$EVORL_DIR"
  PYTHONPATH="$EVORL_VALUE_PYTHONPATH" python -m lerobot.scripts.lerobot_value_train \
    --dataset.repo_id="$POOL_REPO_ID" \
    --dataset.root="$POOL_ROOT" \
    --value.type=pistar06 \
    --value.dtype="$DEXJOCO_EVO_VALUE_DTYPE" \
    --value.vision_repo_id="$DEXJOCO_EVO_VALUE_VISION_REPO" \
    --value.language_repo_id="$DEXJOCO_EVO_VALUE_LANGUAGE_REPO" \
    --value.normalization_mapping="$DEXJOCO_EVO_VALUE_NORMALIZATION_MAPPING" \
    --value.camera_features="$DEXJOCO_EVO_VALUE_CAMERA_FEATURES" \
    --value.device=cuda \
    --value.push_to_hub=false \
    --batch_size="$DEXJOCO_EVO_VALUE_BATCH_SIZE" \
    --num_workers="$DEXJOCO_EVO_VALUE_NUM_WORKERS" \
    --steps="$DEXJOCO_EVO_VALUE_STEPS" \
    --save_checkpoint=true \
    --save_freq="$DEXJOCO_EVO_VALUE_STEPS" \
    --wandb.enable=false \
    --output_dir="$value_dir" \
    --job_name="pistar06_value_${DEXJOCO_EVO_LEROBOT_VARIANT}_${stage}"

  echo "[evorl-lerobot] value infer stage=$stage"
  PYTHONPATH="$EVORL_VALUE_PYTHONPATH" python -m lerobot.scripts.lerobot_value_infer \
    --dataset.repo_id="$POOL_REPO_ID" \
    --dataset.root="$POOL_ROOT" \
    --dataset.default_success=success \
    --inference.checkpoint_path="$value_dir" \
    --runtime.device=cuda \
    --runtime.batch_size="$DEXJOCO_EVO_VALUE_BATCH_SIZE" \
    --runtime.num_workers="$DEXJOCO_EVO_VALUE_NUM_WORKERS" \
    --acp.enable=true \
    --acp.n_step="$DEXJOCO_EVO_N_STEP" \
    --acp.positive_ratio="$DEXJOCO_EVO_POSITIVE_RATIO" \
    --acp.binarization="$DEXJOCO_EVO_ACP_BINARIZATION" \
    --acp.multitag_ratios="$DEXJOCO_EVO_MULTITAG_RATIOS" \
    --acp.min_positive_run_length="$DEXJOCO_EVO_MIN_POSITIVE_RUN_LENGTH" \
    --acp.success_aware="$DEXJOCO_EVO_SUCCESS_AWARE" \
    --acp.c_fail_coef="$DEXJOCO_EVO_C_FAIL_COEF" \
    --acp.value_field="$DEXJOCO_EVO_VALUE_FIELD" \
    --acp.advantage_field="$DEXJOCO_EVO_ADVANTAGE_FIELD" \
    --acp.indicator_field="$DEXJOCO_EVO_INDICATOR_FIELD" \
    --viz.enable=false \
    --output_dir="$infer_dir" \
    --job_name="pistar06_infer_${DEXJOCO_EVO_LEROBOT_VARIANT}_${stage}"
  cd "$EXP_DIR"
}

train_policy_on_pool() {
  local stage="$1"
  local seed="$2"
  local exp_name="evorl_lerobot_${DEXJOCO_EVO_LEROBOT_VARIANT}_${stage}"

  patch_openpi_for_evorl_lerobot_acp
  patch_openpi_config_yaml_for_lerobot_pool "$POOL_PARENT" "$current_pretrained_path"

  export OPENPI_LEROBOT_ACP_ENABLE="$DEXJOCO_EVO_TRAIN_ACP_ENABLE"
  export OPENPI_LEROBOT_ACP_INDICATOR_FIELD="$DEXJOCO_EVO_INDICATOR_FIELD"
  export OPENPI_LEROBOT_ACP_INDICATOR_DROPOUT_PROB="$DEXJOCO_EVO_INDICATOR_DROPOUT_PROB"
  export OPENPI_LEROBOT_ACP_SEED="$seed"
  export OPENPI_LEROBOT_ACP_TAG_KEY="$DEXJOCO_EVO_TAG_KEY"
  export OPENPI_LEROBOT_ACP_TAG_VALUES="$DEXJOCO_EVO_TAG_VALUES"
  export OPENPI_RECAP_LORA_ONLY
  export DEXJOCO_RECAP_WARMUP_STEPS
  export DEXJOCO_RECAP_NUM_WORKERS
  export DEXJOCO_RECAP_SAVE_INTERVAL

  echo "[evorl-lerobot] computing OpenPI norm stats stage=$stage pool=$POOL_ROOT"
  cd "$DEXJOCO_DIR/openpi"
  conda run --no-capture-output --prefix "$OPENPI_ENV_PREFIX" python scripts/compute_norm_stats.py \
    "$DEXJOCO_TASK" \
    --batch-size="$DEXJOCO_RECAP_BATCH_SIZE" \
    --num-workers=0

  echo "[evorl-lerobot] training OpenPI policy stage=$stage exp=$exp_name steps=$DEXJOCO_RECAP_TRAIN_STEPS train_acp=$DEXJOCO_EVO_TRAIN_ACP_ENABLE"
  conda run --no-capture-output --prefix "$OPENPI_ENV_PREFIX" python scripts/train.py \
    "$DEXJOCO_TASK" \
    --exp-name="$exp_name" \
    --overwrite \
    --num-train-steps="$DEXJOCO_RECAP_TRAIN_STEPS" \
    --num-workers="$DEXJOCO_RECAP_NUM_WORKERS" \
    --save-interval="$DEXJOCO_RECAP_SAVE_INTERVAL" \
    --log-interval=50 \
    --fsdp-devices="$DEXJOCO_RECAP_FSDP_DEVICES"

  local ckpt_root="$DEXJOCO_DIR/checkpoints/evorl_lerobot_ckpts/$DEXJOCO_TASK/$exp_name"
  local step
  step="$(find "$ckpt_root" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort -n | tail -1)"
  if [[ -z "$step" ]]; then
    echo "[evorl-lerobot] no checkpoint found in $ckpt_root" >&2
    exit 1
  fi
  current_policy_dir="$ckpt_root/$step"
  current_pretrained_path="$current_policy_dir/params"
  if [[ ! -d "$current_pretrained_path" ]]; then
    echo "[evorl-lerobot] checkpoint params missing: $current_pretrained_path" >&2
    exit 1
  fi
  POLICY_STEP="$step"
  cd "$EXP_DIR"
}

collect_rollout_to_lerobot() {
  local round="$1"
  local stage
  stage="$(printf 'r%02d' "$round")"
  local seed=$((DEXJOCO_EVO_COLLECT_SEED_BASE + round - 1))
  local data_prefix="evorl_lerobot_${DEXJOCO_EVO_LEROBOT_VARIANT}_${stage}"
  local output_name="${DEXJOCO_EVO_OUTPUT_NAME}/collect_${stage}"
  local collect_mode="base"
  if [[ "$DEXJOCO_EVO_COLLECT_PROMPT" == "positive" ]]; then
    collect_mode="acp"
  fi

  echo "[evorl-lerobot] collecting $stage policy=$current_policy_dir prompt=$DEXJOCO_EVO_COLLECT_PROMPT episodes=$DEXJOCO_EVO_COLLECT_EPISODES seed=$seed"
  DEXJOCO_COLLECT_EPISODES="$DEXJOCO_EVO_COLLECT_EPISODES" \
  DEXJOCO_COLLECT_SEED="$seed" \
  DEXJOCO_RECAP_COLLECT_PROMPT_MODE="$collect_mode" \
  DEXJOCO_ACP_SUFFIX="$DEXJOCO_ACP_SUFFIX" \
  DEXJOCO_RECAP_INCLUDE_FAILURES=1 \
  DEXJOCO_RECAP_COLLECT_SHARD_EPISODES="$DEXJOCO_EVO_COLLECT_SHARD_EPISODES" \
  DEXJOCO_RECAP_OUTPUT_NAME="$output_name" \
  DEXJOCO_RECAP_DATA_PREFIX="$data_prefix" \
  DEXJOCO_RECAP_ROLLOUT_POLICY_DIR="$current_policy_dir" \
  DEXJOCO_RECAP_PRETRAINED_MODEL_PATH="$current_pretrained_path" \
  DEXJOCO_RECAP_SKIP_EVAL=1 \
  DEXJOCO_RECAP_COLLECT_ONLY=1 \
    bash jobs/25_dexjoco_click_mouse_recap_rollout_finetune.sh

  local npz="$RUN_ROOT/${data_prefix}_collected_rollouts.npz"
  local rollout_root="$RUN_ROOT/${data_prefix}_lerobot"
  local rollout_repo_id="local/${DEXJOCO_TASK}_${DEXJOCO_EVO_LEROBOT_VARIANT}_${stage}_rollout"
  local convert_args=(
    --input "$npz"
    --output-root "$rollout_root"
    --repo-id "$rollout_repo_id"
    --task "$DEXJOCO_TASK"
    --fps "$DEXJOCO_EVO_LEROBOT_FPS"
    --match-features-root "$D0_ROOT"
    --overwrite
    --summary-output "$OUT_BASE/${stage}_rollout_lerobot.summary.json"
  )
  if [[ "$DEXJOCO_EVO_LEROBOT_USE_VIDEOS" == "1" ]]; then
    convert_args+=(--use-videos)
  fi
  PYTHONPATH="$EVORL_VALUE_PYTHONPATH" python "$EXP_DIR/scripts/dexjoco_npz_to_lerobot.py" "${convert_args[@]}"
  pool_inputs+=("$rollout_repo_id=$rollout_root")

  local summary_file="${npz%.npz}.summary.txt"
  COLLECT_SUCCESS="$(awk -F= '/^successful_episodes=/{print $2}' "$summary_file" 2>/dev/null || echo NA)"
  COLLECT_EPISODES="$(awk -F= '/^saved_episodes=/{print $2}' "$summary_file" 2>/dev/null || echo NA)"
}

evaluate_policy() {
  local stage="$1"
  local eval_config="$positive_eval_config"
  local eval_tag="positive"
  if [[ "$DEXJOCO_EVO_EVAL_PROMPT" == "base" ]]; then
    eval_config="$base_config"
    eval_tag="base"
  elif [[ "$DEXJOCO_EVO_EVAL_PROMPT" == "negative" ]]; then
    eval_config="$negative_eval_config"
    eval_tag="negative"
  elif tag_value_allowed "$DEXJOCO_EVO_EVAL_PROMPT"; then
    eval_tag="$(sanitize_tag_value "$DEXJOCO_EVO_EVAL_PROMPT")"
    eval_config="$tag_prompt_config_dir/${eval_tag}.yaml"
  fi

  local port="${DEXJOCO_PORT:-$(dexjoco_default_port)}"
  local host="${DEXJOCO_HOST:-127.0.0.1}"
  local server_log="$OUT_BASE/${stage}_${eval_tag}_server.log"
  local eval_output_dir="$OUT_BASE/${stage}_${eval_tag}_eval_episodes"

  cd "$DEXJOCO_DIR/openpi"
  setsid conda run --no-capture-output --prefix "$OPENPI_ENV_PREFIX" python scripts/serve_policy.py \
    --port="$port" \
    policy:checkpoint \
    --policy.config="$DEXJOCO_TASK" \
    --policy.dir="$current_policy_dir" \
    > "$server_log" 2>&1 &
  local server_pid=$!
  local start_ts
  start_ts="$(date +%s)"
  while true; do
    if [[ -f "$server_log" ]] && grep -q "server listening on" "$server_log"; then
      break
    fi
    if ! dexjoco_server_group_alive "$server_pid"; then
      echo "[evorl-lerobot] eval server exited early" >&2
      tail -200 "$server_log" >&2 || true
      exit 1
    fi
    if (( $(date +%s) - start_ts >= 900 )); then
      echo "[evorl-lerobot] eval server did not become ready" >&2
      tail -200 "$server_log" >&2 || true
      exit 1
    fi
    sleep 5
  done

  if [[ -n "$DEXJOCO_RECAP_EVAL_SEEDS" ]]; then
    read -r -a eval_seeds <<< "$DEXJOCO_RECAP_EVAL_SEEDS"
  else
    eval_seeds=("$DEXJOCO_EVAL_SEED")
  fi

  EVAL_STATUS="ok"
  EVAL_SUCCESSES=0
  EVAL_EPISODES=0
  for eval_seed in "${eval_seeds[@]}"; do
    local seed_output_dir="${eval_output_dir}_seed${eval_seed}"
    echo "[evorl-lerobot] eval stage=$stage prompt=$eval_tag seed=$eval_seed episodes=$DEXJOCO_EVAL_EPISODES"
    cd "$DEXJOCO_DIR"
    set +e
    conda run --no-capture-output --prefix "$DEXJOCO_ENV_PREFIX" dexjoco-openpi-eval \
      --config="$eval_config" \
      --seed="$eval_seed" \
      --port="$port" \
      --host="$host" \
      --episodes="$DEXJOCO_EVAL_EPISODES" \
      --output="$seed_output_dir" \
      2>&1 | tee "$OUT_BASE/${stage}_${eval_tag}_eval_seed${eval_seed}.log"
    local eval_status=${PIPESTATUS[0]}
    set -e
    if [[ "$eval_status" -ne 0 ]]; then
      EVAL_STATUS="eval_failed"
    fi
    local success_file
    success_file="$(find "$seed_output_dir" -maxdepth 1 -name 'success_rate_*.txt' -printf '%f\n' | sort | head -1 || true)"
    if [[ "$success_file" =~ success_rate_([0-9]+)_([0-9]+)\.txt ]]; then
      EVAL_SUCCESSES=$((EVAL_SUCCESSES + BASH_REMATCH[1]))
      EVAL_EPISODES=$((EVAL_EPISODES + BASH_REMATCH[2]))
    else
      EVAL_EPISODES=$((EVAL_EPISODES + DEXJOCO_EVAL_EPISODES))
    fi
  done
  dexjoco_kill_server_group "$server_pid"
  cd "$EXP_DIR"
}

run_training_stage() {
  local stage="$1"
  local seed="$2"
  merge_pool "$stage"
  train_value_and_infer "$stage"
  train_policy_on_pool "$stage" "$seed"
  local stats
  stats="$(PYTHONPATH="$EVORL_VALUE_PYTHONPATH" python - "$POOL_REPO_ID" "$POOL_ROOT" <<'PY'
import sys
from lerobot.datasets.lerobot_dataset import LeRobotDataset
ds = LeRobotDataset(repo_id=sys.argv[1], root=sys.argv[2])
print(f"{ds.num_episodes}\t{ds.num_frames}")
PY
)"
  POOL_EPISODES="$(echo "$stats" | cut -f1)"
  POOL_FRAMES="$(echo "$stats" | cut -f2)"
}

echo "[evorl-lerobot] variant=$DEXJOCO_EVO_LEROBOT_VARIANT collect_prompt=$DEXJOCO_EVO_COLLECT_PROMPT train_acp=$DEXJOCO_EVO_TRAIN_ACP_ENABLE acp_binarization=$DEXJOCO_EVO_ACP_BINARIZATION multitag_ratios=$DEXJOCO_EVO_MULTITAG_RATIOS tag_values=$DEXJOCO_EVO_TAG_VALUES min_positive_run=$DEXJOCO_EVO_MIN_POSITIVE_RUN_LENGTH success_aware=$DEXJOCO_EVO_SUCCESS_AWARE rounds=$DEXJOCO_EVO_ROUNDS D0=$D0_ROOT use_videos=$DEXJOCO_EVO_LEROBOT_USE_VIDEOS"

run_training_stage "d0" "$DEXJOCO_EVO_COLLECT_SEED_BASE"
echo -e "d0\t$POOL_EPISODES\t$POOL_FRAMES\t$POLICY_STEP\tNA\tNA\tNA\ttrain_only\tNA\tNA" | tee -a "$SUMMARY"

for round in $(seq 1 "$DEXJOCO_EVO_ROUNDS"); do
  stage="$(printf 'r%02d' "$round")"
  collect_rollout_to_lerobot "$round"
  run_training_stage "$stage" "$((DEXJOCO_EVO_COLLECT_SEED_BASE + round))"
  echo -e "$stage\t$POOL_EPISODES\t$POOL_FRAMES\t$POLICY_STEP\t$DEXJOCO_EVO_COLLECT_PROMPT\t$COLLECT_SUCCESS\t$COLLECT_EPISODES\ttrain_only\tNA\tNA" | tee -a "$SUMMARY"
done

FINAL_POLICY_EXPORT_DIR="$OUT_BASE/export_checkpoints/${DEXJOCO_EVO_LEROBOT_VARIANT}_final_${POLICY_STEP}"
rm -rf "$FINAL_POLICY_EXPORT_DIR"
mkdir -p "$FINAL_POLICY_EXPORT_DIR"
cp -a "$current_policy_dir/." "$FINAL_POLICY_EXPORT_DIR/"
{
  echo "source_policy_dir=$current_policy_dir"
  echo "export_policy_dir=$FINAL_POLICY_EXPORT_DIR"
  echo "policy_step=$POLICY_STEP"
} | tee "$OUT_BASE/final_policy_checkpoint.txt"

evaluate_policy "final"
echo -e "final\t$POOL_EPISODES\t$POOL_FRAMES\t$POLICY_STEP\t$DEXJOCO_EVO_COLLECT_PROMPT\tNA\tNA\t$EVAL_STATUS\t$EVAL_SUCCESSES\t$EVAL_EPISODES" | tee -a "$SUMMARY"

echo "[evorl-lerobot] finished: $SUMMARY"

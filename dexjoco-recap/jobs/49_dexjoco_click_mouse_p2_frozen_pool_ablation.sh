#!/usr/bin/env bash
set -euo pipefail

cd "$EXP_DIR"
source scripts/dexjoco_common.sh

# P2 paired ablation on one frozen rollout pool. This is the first rigorous
# DexJoCo ReCap matrix: all methods reuse the same collected full-pool NPZ.
export DEXJOCO_TASK="${DEXJOCO_TASK:-click_mouse}"
export DEXJOCO_COLLECT_SEED="${DEXJOCO_COLLECT_SEED:-10000}"
export DEXJOCO_COLLECT_EPISODES="${DEXJOCO_COLLECT_EPISODES:-360}"
export DEXJOCO_RECAP_EVAL_SEEDS="${DEXJOCO_RECAP_EVAL_SEEDS:-0 1 2}"
export DEXJOCO_EVAL_EPISODES="${DEXJOCO_EVAL_EPISODES:-100}"
export DEXJOCO_RECAP_TRAIN_STEPS="${DEXJOCO_RECAP_TRAIN_STEPS:-2000}"
export DEXJOCO_RECAP_INCLUDE_FAILURES="${DEXJOCO_RECAP_INCLUDE_FAILURES:-1}"
export DEXJOCO_RECAP_COLLECT_PROMPT_MODE="${DEXJOCO_RECAP_COLLECT_PROMPT_MODE:-base}"
export DEXJOCO_RECAP_FSDP_DEVICES="${DEXJOCO_RECAP_FSDP_DEVICES:-1}"
export DEXJOCO_RECAP_NUM_WORKERS="${DEXJOCO_RECAP_NUM_WORKERS:-0}"

RUN_ROOT="${RUN_ROOT:-$(_dexjoco_run_root)}"
D0_PREFIX="${DEXJOCO_RECAP_D0_PREFIX:-p2_d0_seed${DEXJOCO_COLLECT_SEED}}"
D0_OUTPUT="${DEXJOCO_RECAP_D0_OUTPUT:-dexjoco_click_mouse_p2_frozen_pool}"
D0_NPZ="$RUN_ROOT/${D0_PREFIX}_success_rollouts.npz"

echo "[p2] collecting frozen full pool prefix=$D0_PREFIX seed=$DEXJOCO_COLLECT_SEED episodes=$DEXJOCO_COLLECT_EPISODES"
DEXJOCO_RECAP_DATA_PREFIX="$D0_PREFIX" \
DEXJOCO_RECAP_OUTPUT_NAME="$D0_OUTPUT" \
DEXJOCO_RECAP_EXP_NAME="p2_collect_full_pool" \
DEXJOCO_RECAP_COLLECT_ONLY=1 \
  bash jobs/46_dexjoco_click_mouse_collect_full_pool.sh

if [[ ! -f "$D0_NPZ" ]]; then
  echo "[p2] expected frozen NPZ not found: $D0_NPZ" >&2
  exit 1
fi

run_method() {
  local name="$1"
  local job="$2"
  echo "[p2] running method=$name job=$job frozen_npz=$D0_NPZ eval_seeds=$DEXJOCO_RECAP_EVAL_SEEDS"
  DEXJOCO_RECAP_ROLLOUT_NPZ_INPUT="$D0_NPZ" \
  DEXJOCO_RECAP_DATA_PREFIX="p2_${name}_seed${DEXJOCO_COLLECT_SEED}" \
  DEXJOCO_RECAP_OUTPUT_NAME="dexjoco_click_mouse_p2_frozen_pool/${name}" \
  DEXJOCO_RECAP_EXP_NAME="p2_${name}_seed${DEXJOCO_COLLECT_SEED}" \
    bash "$job"
}

run_method "base_lora" "jobs/44_dexjoco_click_mouse_base_lora_full_eval100.sh"
run_method "acp_lora" "jobs/45_dexjoco_click_mouse_acp_lora_full_eval100.sh"
run_method "random_positive" "jobs/48_dexjoco_click_mouse_random_positive_full_eval100.sh"
run_method "pistar06_value_acp" "jobs/43_dexjoco_click_mouse_recap_pistar06_full_eval100.sh"

echo "[p2] paired frozen-pool ablation finished"

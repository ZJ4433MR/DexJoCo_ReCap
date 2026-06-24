#!/usr/bin/env bash
set -euo pipefail

# Collect a frozen full rollout pool for paired DexJoCo ReCap ablations.
# This keeps both successes and failures and stops before value labeling or
# policy training. Use the resulting NPZ as DEXJOCO_RECAP_ROLLOUT_NPZ_INPUT.
export DEXJOCO_TASK="${DEXJOCO_TASK:-click_mouse}"
export DEXJOCO_EVAL_SEED="${DEXJOCO_EVAL_SEED:-0}"
export DEXJOCO_COLLECT_SEED="${DEXJOCO_COLLECT_SEED:-10000}"
export DEXJOCO_COLLECT_EPISODES="${DEXJOCO_COLLECT_EPISODES:-360}"
export DEXJOCO_RECAP_INCLUDE_FAILURES="${DEXJOCO_RECAP_INCLUDE_FAILURES:-1}"
export DEXJOCO_RECAP_COLLECT_PROMPT_MODE="${DEXJOCO_RECAP_COLLECT_PROMPT_MODE:-base}"
export DEXJOCO_RECAP_COLLECT_ONLY="${DEXJOCO_RECAP_COLLECT_ONLY:-1}"
export DEXJOCO_RECAP_OUTPUT_NAME="${DEXJOCO_RECAP_OUTPUT_NAME:-dexjoco_click_mouse_frozen_full_pool}"
export DEXJOCO_RECAP_DATA_PREFIX="${DEXJOCO_RECAP_DATA_PREFIX:-frozen_full_pool}"
export DEXJOCO_RECAP_EXP_NAME="${DEXJOCO_RECAP_EXP_NAME:-collect_full_pool}"

exec bash jobs/25_dexjoco_click_mouse_recap_rollout_finetune.sh

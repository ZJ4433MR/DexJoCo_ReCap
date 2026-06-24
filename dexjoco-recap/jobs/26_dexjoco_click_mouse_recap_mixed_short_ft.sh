#!/usr/bin/env bash
set -euo pipefail

# More conservative DexJoCo ReCap variant:
# collect baseline rollouts with the original prompt, keep all frames as
# behavior-preserving data, and repeat successful frames under the ACP prompt.
export DEXJOCO_COLLECT_EPISODES="${DEXJOCO_COLLECT_EPISODES:-60}"
export DEXJOCO_EVAL_EPISODES="${DEXJOCO_EVAL_EPISODES:-20}"
export DEXJOCO_RECAP_INCLUDE_FAILURES="${DEXJOCO_RECAP_INCLUDE_FAILURES:-1}"
export DEXJOCO_RECAP_COLLECT_PROMPT_MODE="${DEXJOCO_RECAP_COLLECT_PROMPT_MODE:-base}"
export OPENPI_RECAP_BASE_REPEAT="${OPENPI_RECAP_BASE_REPEAT:-1}"
export OPENPI_RECAP_POSITIVE_REPEAT="${OPENPI_RECAP_POSITIVE_REPEAT:-3}"
export DEXJOCO_RECAP_TRAIN_STEPS="${DEXJOCO_RECAP_TRAIN_STEPS:-200}"
export DEXJOCO_RECAP_WARMUP_STEPS="${DEXJOCO_RECAP_WARMUP_STEPS:-25}"
export DEXJOCO_RECAP_SAVE_INTERVAL="${DEXJOCO_RECAP_SAVE_INTERVAL:-100}"
export DEXJOCO_RECAP_EXP_NAME="${DEXJOCO_RECAP_EXP_NAME:-recap_mixed_acp_short_ft}"

exec bash jobs/25_dexjoco_click_mouse_recap_rollout_finetune.sh

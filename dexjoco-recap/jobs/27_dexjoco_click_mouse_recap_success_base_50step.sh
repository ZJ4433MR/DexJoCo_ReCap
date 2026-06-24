#!/usr/bin/env bash
set -euo pipefail

# Conservative success-only ReCap variant:
# collect public-policy rollouts with the original prompt, keep only successful
# trajectories, and apply a very small LoRA update for the ACP prompt.
export DEXJOCO_COLLECT_EPISODES="${DEXJOCO_COLLECT_EPISODES:-40}"
export DEXJOCO_EVAL_EPISODES="${DEXJOCO_EVAL_EPISODES:-20}"
export DEXJOCO_RECAP_INCLUDE_FAILURES="${DEXJOCO_RECAP_INCLUDE_FAILURES:-0}"
export DEXJOCO_RECAP_COLLECT_PROMPT_MODE="${DEXJOCO_RECAP_COLLECT_PROMPT_MODE:-base}"
export OPENPI_RECAP_BASE_REPEAT="${OPENPI_RECAP_BASE_REPEAT:-0}"
export OPENPI_RECAP_POSITIVE_REPEAT="${OPENPI_RECAP_POSITIVE_REPEAT:-1}"
export DEXJOCO_RECAP_TRAIN_STEPS="${DEXJOCO_RECAP_TRAIN_STEPS:-50}"
export DEXJOCO_RECAP_WARMUP_STEPS="${DEXJOCO_RECAP_WARMUP_STEPS:-5}"
export DEXJOCO_RECAP_SAVE_INTERVAL="${DEXJOCO_RECAP_SAVE_INTERVAL:-50}"
export DEXJOCO_RECAP_EXP_NAME="${DEXJOCO_RECAP_EXP_NAME:-recap_success_base_50step}"

exec bash jobs/25_dexjoco_click_mouse_recap_rollout_finetune.sh

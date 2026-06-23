#!/usr/bin/env bash
set -euo pipefail

export DEXJOCO_EVAL_POLICY_DIR="${DEXJOCO_EVAL_POLICY_DIR:-/share/home/u23133/.cache/recap-sim-l40-stage/dexjoco_click_mouse_eval500_existing_policy_20260619_210959/checkpoints/A_r03_1199}"
export DEXJOCO_DIR="${DEXJOCO_DIR:-/share/home/u23133/.cache/recap-sim-l40-stage/dexjoco_click_mouse_eval500_existing_policy_20260619_210959/source/dexjoco-src}"
export DEXJOCO_LOCAL_SOURCE="${DEXJOCO_LOCAL_SOURCE:-/share/home/u23133/.cache/recap-sim-l40-stage/dexjoco_click_mouse_eval500_existing_policy_20260619_210959/source/dexjoco-src}"
export DEXJOCO_USE_SOURCE_IN_PLACE="${DEXJOCO_USE_SOURCE_IN_PLACE:-1}"
export DEXJOCO_EVAL_EPISODES="${DEXJOCO_EVAL_EPISODES:-500}"
export DEXJOCO_EVAL_SEED="${DEXJOCO_EVAL_SEED:-0}"
export DEXJOCO_EVAL_PROMPT="${DEXJOCO_EVAL_PROMPT:-positive}"
export DEXJOCO_EVAL_NAME="${DEXJOCO_EVAL_NAME:-A_positive_collect_eval500}"

exec bash "$EXP_DIR/jobs/60_dexjoco_click_mouse_eval_existing_policy.sh"

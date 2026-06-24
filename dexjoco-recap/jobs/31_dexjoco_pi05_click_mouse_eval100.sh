#!/usr/bin/env bash
set -euo pipefail

# Matching pi0.5 baseline for the v9 ReCap click_mouse 100-episode eval.
export DEXJOCO_TASKS="${DEXJOCO_TASKS:-click_mouse}"
export DEXJOCO_EVAL_EPISODES="${DEXJOCO_EVAL_EPISODES:-100}"
export DEXJOCO_EVAL_SEED="${DEXJOCO_EVAL_SEED:-0}"

exec bash "$EXP_DIR/jobs/22_dexjoco_pi05_single_arm_matrix.sh"

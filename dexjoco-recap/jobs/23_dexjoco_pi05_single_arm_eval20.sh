#!/usr/bin/env bash
set -euo pipefail

export DEXJOCO_TASKS="${DEXJOCO_TASKS:-click_mouse hammer_nail}"
export DEXJOCO_EVAL_EPISODES="${DEXJOCO_EVAL_EPISODES:-20}"

exec bash "$EXP_DIR/jobs/22_dexjoco_pi05_single_arm_matrix.sh"

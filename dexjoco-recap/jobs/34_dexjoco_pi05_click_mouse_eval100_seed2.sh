#!/usr/bin/env bash
set -euo pipefail

export DEXJOCO_EVAL_SEED=2
exec bash "$EXP_DIR/jobs/31_dexjoco_pi05_click_mouse_eval100.sh"

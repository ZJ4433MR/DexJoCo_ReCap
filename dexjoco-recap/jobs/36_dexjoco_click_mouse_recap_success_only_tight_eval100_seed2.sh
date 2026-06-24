#!/usr/bin/env bash
set -euo pipefail

export DEXJOCO_EVAL_SEED=2
export DEXJOCO_RECAP_EXP_NAME="${DEXJOCO_RECAP_EXP_NAME:-recap_success_only_tight_eval100_seed2}"
exec bash jobs/30_dexjoco_click_mouse_recap_success_only_tight_eval100.sh

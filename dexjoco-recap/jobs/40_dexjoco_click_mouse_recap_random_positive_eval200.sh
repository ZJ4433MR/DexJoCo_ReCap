#!/usr/bin/env bash
set -euo pipefail

# Larger-episode random-positive ReCap ablation. Compared with the earlier
# random-positive run, this increases both collection and evaluation episodes
# to check whether the 91/100 result holds with more data and lower eval noise.
export DEXJOCO_COLLECT_EPISODES="${DEXJOCO_COLLECT_EPISODES:-360}"
export DEXJOCO_EVAL_EPISODES="${DEXJOCO_EVAL_EPISODES:-200}"
export DEXJOCO_RECAP_EXP_NAME="${DEXJOCO_RECAP_EXP_NAME:-recap_random_positive_collect360_eval200}"

exec bash jobs/32_dexjoco_click_mouse_recap_random_positive_eval100.sh

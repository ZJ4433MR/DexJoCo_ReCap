#!/usr/bin/env bash
set -euo pipefail

# Multi-round ReCap ablation: the policy is iterated, but each round trains
# only on the newest rollout batch instead of the accumulated data pool.
export DEXJOCO_EVO_MERGE_POOL="${DEXJOCO_EVO_MERGE_POOL:-0}"
export DEXJOCO_EVO_ROUNDS="${DEXJOCO_EVO_ROUNDS:-3}"
export DEXJOCO_EVO_COLLECT_EPISODES="${DEXJOCO_EVO_COLLECT_EPISODES:-120}"
export DEXJOCO_EVO_TRAIN_STEPS="${DEXJOCO_EVO_TRAIN_STEPS:-1200}"
export DEXJOCO_EVAL_EPISODES="${DEXJOCO_EVAL_EPISODES:-100}"

exec bash jobs/37_dexjoco_click_mouse_evorl_multiround.sh

#!/usr/bin/env bash
set -euo pipefail

# Small BC vs RECAP/ACP comparison with final PuSH-T simulation evaluation.
# This is the first run that should produce interpretable success-rate numbers,
# while still staying small enough for fast iteration on one L40.

export DATASET_REPO="${DATASET_REPO:-lerobot/pusht}"
export DATASET_EPISODES="${DATASET_EPISODES:-[0,1,2,3,4,5,6,7,8,9]}"
export TAG="${TAG:-pusht_recap_compare_eval}"
export POLICY_TYPE="${POLICY_TYPE:-act}"
export BATCH_SIZE="${BATCH_SIZE:-16}"
export POLICY_STEPS="${POLICY_STEPS:-1000}"
export VALUE_STEPS="${VALUE_STEPS:-300}"
export VALUE_BATCH_SIZE="${VALUE_BATCH_SIZE:-8}"
export RUN_EVAL="${RUN_EVAL:-true}"
export EVAL_EPISODES="${EVAL_EPISODES:-20}"
export EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-10}"

bash "$EXPERIMENT_DIR/jobs/11_pusht_recap_pilot.sh"

#!/usr/bin/env bash
set -euo pipefail

# A small formal PuSH-T comparison after the pilot and short eval runs.
# Uses non-random value backbones and enough steps/episodes to start making
# the BC vs RECAP/ACP comparison meaningful, while still fitting a single L40.

export DATASET_REPO="${DATASET_REPO:-lerobot/pusht}"
export DATASET_EPISODES="${DATASET_EPISODES:-[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49]}"
export TAG="${TAG:-pusht_recap_formal_small}"
export POLICY_TYPE="${POLICY_TYPE:-act}"
export BATCH_SIZE="${BATCH_SIZE:-32}"
export POLICY_STEPS="${POLICY_STEPS:-10000}"
export VALUE_STEPS="${VALUE_STEPS:-2000}"
export VALUE_BATCH_SIZE="${VALUE_BATCH_SIZE:-8}"
export VALUE_VISION_REPO="${VALUE_VISION_REPO:-google/vit-base-patch16-224-in21k}"
export VALUE_LANGUAGE_REPO="${VALUE_LANGUAGE_REPO:-google-bert/bert-base-uncased}"
export RUN_EVAL="${RUN_EVAL:-true}"
export EVAL_EPISODES="${EVAL_EPISODES:-50}"
export EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-10}"

EXP_ROOT="${EXP_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
bash "$EXP_ROOT/jobs/11_pusht_recap_pilot.sh"

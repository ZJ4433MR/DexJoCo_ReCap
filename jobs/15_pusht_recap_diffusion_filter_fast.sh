#!/usr/bin/env bash
set -euo pipefail

# Faster diffusion-policy PuSH-T comparison used to get a completed BC vs
# ACP-positive ReCap signal before spending a longer L40 run.

export DATASET_REPO="${DATASET_REPO:-lerobot/pusht}"
export DATASET_EPISODES="${DATASET_EPISODES:-[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99]}"
export TAG="${TAG:-pusht_recap_diffusion_filter_fast}"
export POLICY_TYPE="${POLICY_TYPE:-diffusion}"
export BATCH_SIZE="${BATCH_SIZE:-64}"
export POLICY_STEPS="${POLICY_STEPS:-20000}"
export POLICY_LOG_FREQ="${POLICY_LOG_FREQ:-200}"
export VALUE_STEPS="${VALUE_STEPS:-2000}"
export VALUE_BATCH_SIZE="${VALUE_BATCH_SIZE:-8}"
export VALUE_LOG_FREQ="${VALUE_LOG_FREQ:-100}"
export VALUE_VISION_REPO="${VALUE_VISION_REPO:-google/vit-base-patch16-224-in21k}"
export VALUE_LANGUAGE_REPO="${VALUE_LANGUAGE_REPO:-google-bert/bert-base-uncased}"
export DIFFUSION_NUM_INFERENCE_STEPS="${DIFFUSION_NUM_INFERENCE_STEPS:-20}"
export RECAP_FILTER_POSITIVE="${RECAP_FILTER_POSITIVE:-true}"
export RECAP_INDICATOR_DROPOUT_PROB="${RECAP_INDICATOR_DROPOUT_PROB:-0.0}"
export RUN_EVAL="${RUN_EVAL:-true}"
export EVAL_EPISODES="${EVAL_EPISODES:-50}"
export EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-10}"

EXP_ROOT="${EXP_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
bash "$EXP_ROOT/jobs/11_pusht_recap_pilot.sh"

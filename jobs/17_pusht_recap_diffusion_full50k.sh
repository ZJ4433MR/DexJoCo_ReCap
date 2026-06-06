#!/usr/bin/env bash
set -euo pipefail

# Stronger PuSH-T diffusion run after the first successful ReCap result.
# Compared with jobs/16:
# - uses the full public PuSH-T episode set (0..205)
# - trains BC longer
# - trains the value model longer
# - evaluates on more episodes for a lower-variance comparison

export DATASET_REPO="${DATASET_REPO:-lerobot/pusht}"
export DATASET_EPISODES="${DATASET_EPISODES:-[$(seq -s, 0 205)]}"
export TAG="${TAG:-pusht_recap_diffusion_full50k_w15}"
export POLICY_TYPE="${POLICY_TYPE:-diffusion}"
export BATCH_SIZE="${BATCH_SIZE:-64}"
export POLICY_STEPS="${POLICY_STEPS:-50000}"
export RECAP_POLICY_STEPS="${RECAP_POLICY_STEPS:-5000}"
export POLICY_LOG_FREQ="${POLICY_LOG_FREQ:-500}"
export VALUE_STEPS="${VALUE_STEPS:-5000}"
export VALUE_BATCH_SIZE="${VALUE_BATCH_SIZE:-8}"
export VALUE_LOG_FREQ="${VALUE_LOG_FREQ:-250}"
export VALUE_VISION_REPO="${VALUE_VISION_REPO:-google/vit-base-patch16-224-in21k}"
export VALUE_LANGUAGE_REPO="${VALUE_LANGUAGE_REPO:-google-bert/bert-base-uncased}"
export DIFFUSION_NUM_INFERENCE_STEPS="${DIFFUSION_NUM_INFERENCE_STEPS:-20}"
export RECAP_FILTER_POSITIVE="${RECAP_FILTER_POSITIVE:-false}"
export RECAP_POSITIVE_SAMPLE_WEIGHT="${RECAP_POSITIVE_SAMPLE_WEIGHT:-1.5}"
export RECAP_INDICATOR_DROPOUT_PROB="${RECAP_INDICATOR_DROPOUT_PROB:-0.0}"
export RECAP_INIT_FROM_BASELINE="${RECAP_INIT_FROM_BASELINE:-true}"
export RUN_EVAL="${RUN_EVAL:-true}"
export EVAL_EPISODES="${EVAL_EPISODES:-100}"
export EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-10}"

EXP_ROOT="${EXP_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
bash "$EXP_ROOT/jobs/11_pusht_recap_pilot.sh"

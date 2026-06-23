#!/usr/bin/env bash
set -euo pipefail

# Dependency-recovery resume variant only. It reuses the public-BERT value
# checkpoint produced by v5 and resumes the remaining infer -> ReCap -> eval
# chain. This is not the faithful Gemma Evo-RL pistar06 path.
export DEXJOCO_RECAP_PISTAR06_VALUE_LANGUAGE_REPO="${DEXJOCO_RECAP_PISTAR06_VALUE_LANGUAGE_REPO:-google-bert/bert-base-uncased}"
export DEXJOCO_RECAP_EXP_NAME="${DEXJOCO_RECAP_EXP_NAME:-p2_pistar06_value_acp_seed10000_bert_public_lm_resume_infer}"
export DEXJOCO_RECAP_PISTAR06_SKIP_VALUE_TRAINING="${DEXJOCO_RECAP_PISTAR06_SKIP_VALUE_TRAINING:-1}"
export DEXJOCO_RECAP_PISTAR06_VALUE_CHECKPOINT_SOURCE_TAR="${DEXJOCO_RECAP_PISTAR06_VALUE_CHECKPOINT_SOURCE_TAR:-/share/home/u23133/.cache/recap-sim-l40-stage/dexjoco_click_mouse_p2_pistar06_value_acp_recovery_bert_public_lm_l40_v5/results.tar.gz}"

exec bash jobs/54_dexjoco_click_mouse_p2_pistar06_value_acp_recovery.sh

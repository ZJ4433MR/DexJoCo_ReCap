#!/usr/bin/env bash
set -euo pipefail

# Dependency-recovery variant only: the faithful Evo-RL pistar06 default uses
# google/gemma-3-270m, but the current remote HF mirror/token cannot access that
# gated repo. This wrapper keeps the P2 frozen-rollout ReCap path unchanged while
# swapping only the language backbone to a public encoder so downstream
# value/infer/ReCap plumbing can be tested and clearly reported as non-faithful.
export DEXJOCO_RECAP_PISTAR06_VALUE_LANGUAGE_REPO="${DEXJOCO_RECAP_PISTAR06_VALUE_LANGUAGE_REPO:-google-bert/bert-base-uncased}"
export DEXJOCO_RECAP_EXP_NAME="${DEXJOCO_RECAP_EXP_NAME:-p2_pistar06_value_acp_seed10000_bert_public_lm}"

exec bash jobs/54_dexjoco_click_mouse_p2_pistar06_value_acp_recovery.sh

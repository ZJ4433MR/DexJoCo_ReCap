#!/usr/bin/env bash
set -euo pipefail

# P0 recovery: evaluate the public pi0.5 policy under the ACP prompt only.
export DEXJOCO_TASK="${DEXJOCO_TASK:-click_mouse}"
export DEXJOCO_EVAL_EPISODES="${DEXJOCO_EVAL_EPISODES:-100}"
export DEXJOCO_RECAP_EVAL_SEEDS="${DEXJOCO_RECAP_EVAL_SEEDS:-0 1 2 3 4}"
export DEXJOCO_RECAP_EVAL_VARIANTS="${DEXJOCO_RECAP_EVAL_VARIANTS:-acp_positive}"
export DEXJOCO_RECAP_EVAL_TIMEOUT_SECONDS="${DEXJOCO_RECAP_EVAL_TIMEOUT_SECONDS:-3600}"
export DEXJOCO_RECAP_EVAL_RETRIES="${DEXJOCO_RECAP_EVAL_RETRIES:-1}"

exec bash jobs/24_dexjoco_click_mouse_acp_prompt_eval20.sh

#!/usr/bin/env bash
set -euo pipefail

# P0 recovery: finish the missing public pi0.5 base-prompt seed.
export DEXJOCO_TASK="${DEXJOCO_TASK:-click_mouse}"
export DEXJOCO_EVAL_EPISODES="${DEXJOCO_EVAL_EPISODES:-100}"
export DEXJOCO_RECAP_EVAL_SEEDS="${DEXJOCO_RECAP_EVAL_SEEDS:-4}"
export DEXJOCO_RECAP_EVAL_VARIANTS="${DEXJOCO_RECAP_EVAL_VARIANTS:-baseline}"
export DEXJOCO_RECAP_EVAL_TIMEOUT_SECONDS="${DEXJOCO_RECAP_EVAL_TIMEOUT_SECONDS:-3600}"
export DEXJOCO_RECAP_EVAL_RETRIES="${DEXJOCO_RECAP_EVAL_RETRIES:-1}"

exec bash jobs/24_dexjoco_click_mouse_acp_prompt_eval20.sh

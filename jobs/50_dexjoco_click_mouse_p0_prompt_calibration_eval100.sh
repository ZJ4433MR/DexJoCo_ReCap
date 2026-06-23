#!/usr/bin/env bash
set -euo pipefail

# P0 calibration: public pi0.5 under base prompt vs ACP prompt only.
# No fine-tuning. This determines whether the prompt suffix alone shifts the
# policy before attributing any later gains to ReCap.
export DEXJOCO_TASK="${DEXJOCO_TASK:-click_mouse}"
export DEXJOCO_EVAL_EPISODES="${DEXJOCO_EVAL_EPISODES:-100}"
export DEXJOCO_RECAP_EVAL_SEEDS="${DEXJOCO_RECAP_EVAL_SEEDS:-0 1 2 3 4}"

exec bash jobs/24_dexjoco_click_mouse_acp_prompt_eval20.sh

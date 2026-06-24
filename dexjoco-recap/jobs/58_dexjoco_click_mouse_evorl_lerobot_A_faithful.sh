#!/usr/bin/env bash
set -euo pipefail

# A: Evo-RL-faithful DexJoCo/LeRobot line.
# D0 is the official click_mouse LeRobot dataset. Each rollout uses
# <task>\nAdvantage: positive. Training uses positive/negative prompt tagging
# from acp_indicator, without success-only, positive_repeat, filter_positive, or
# positive_sample_weight.
export DEXJOCO_EVO_LEROBOT_VARIANT="${DEXJOCO_EVO_LEROBOT_VARIANT:-A_faithful_positive_collect}"
export DEXJOCO_EVO_COLLECT_PROMPT="positive"
export OPENPI_RECAP_LORA_ONLY="${OPENPI_RECAP_LORA_ONLY:-1}"

exec bash "$EXP_DIR/jobs/57_dexjoco_click_mouse_evorl_lerobot_ab.sh"

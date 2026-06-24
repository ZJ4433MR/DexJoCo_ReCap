#!/usr/bin/env bash
set -euo pipefail

# B: controlled base-collect comparison.
# Everything matches A except rollout collection uses the original base task
# prompt. Policy training still uses Evo-RL positive/negative prompt tagging.
export DEXJOCO_EVO_LEROBOT_VARIANT="${DEXJOCO_EVO_LEROBOT_VARIANT:-B_base_collect_controlled}"
export DEXJOCO_EVO_COLLECT_PROMPT="base"
export OPENPI_RECAP_LORA_ONLY="${OPENPI_RECAP_LORA_ONLY:-1}"

exec bash "$EXP_DIR/jobs/57_dexjoco_click_mouse_evorl_lerobot_ab.sh"

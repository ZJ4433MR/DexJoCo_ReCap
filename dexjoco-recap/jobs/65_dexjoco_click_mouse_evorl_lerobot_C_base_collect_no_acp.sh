#!/usr/bin/env bash
set -euo pipefail

# C: no-ACP policy-training baseline.
# This keeps the LeRobot D0 + base-prompt rollout + full success/failure pool
# pipeline, and still computes value/advantage/indicator. The only disabled
# mechanism is policy-time ACP prompt tagging.
export DEXJOCO_EVO_LEROBOT_VARIANT="${DEXJOCO_EVO_LEROBOT_VARIANT:-C_base_collect_no_acp_train}"
export DEXJOCO_EVO_COLLECT_PROMPT="base"
export DEXJOCO_EVO_TRAIN_ACP_ENABLE=0
export DEXJOCO_EVO_EVAL_PROMPT="${DEXJOCO_EVO_EVAL_PROMPT:-base}"
export DEXJOCO_EVAL_EPISODES="${DEXJOCO_EVAL_EPISODES:-500}"
export OPENPI_RECAP_LORA_ONLY="${OPENPI_RECAP_LORA_ONLY:-1}"

exec bash "$EXP_DIR/jobs/57_dexjoco_click_mouse_evorl_lerobot_ab.sh"

#!/usr/bin/env bash
set -euo pipefail

# E: keep the Evo-RL/D data loop, but replace binary positive/negative ACP tags
# with four progress levels: failure, low, medium, high.
export DEXJOCO_EVO_LEROBOT_VARIANT="${DEXJOCO_EVO_LEROBOT_VARIANT:-E_multitag_episode_smooth_successaware}"
export DEXJOCO_EVO_COLLECT_PROMPT="${DEXJOCO_EVO_COLLECT_PROMPT:-base}"
export DEXJOCO_EVO_TRAIN_ACP_ENABLE="${DEXJOCO_EVO_TRAIN_ACP_ENABLE:-1}"
export DEXJOCO_EVO_ACP_BINARIZATION="${DEXJOCO_EVO_ACP_BINARIZATION:-episode_multitag_smooth}"
export DEXJOCO_EVO_MULTITAG_RATIOS="${DEXJOCO_EVO_MULTITAG_RATIOS:-0.1,0.2,0.3}"
export DEXJOCO_EVO_MIN_POSITIVE_RUN_LENGTH="${DEXJOCO_EVO_MIN_POSITIVE_RUN_LENGTH:-3}"
export DEXJOCO_EVO_SUCCESS_AWARE="${DEXJOCO_EVO_SUCCESS_AWARE:-true}"
export DEXJOCO_EVO_TAG_KEY="${DEXJOCO_EVO_TAG_KEY:-Advantage}"
export DEXJOCO_EVO_TAG_VALUES="${DEXJOCO_EVO_TAG_VALUES:-failure,low,medium,high}"
export DEXJOCO_EVO_EVAL_PROMPT="${DEXJOCO_EVO_EVAL_PROMPT:-high}"
export DEXJOCO_EVAL_EPISODES="${DEXJOCO_EVAL_EPISODES:-500}"

exec bash "$EXP_DIR/jobs/57_dexjoco_click_mouse_evorl_lerobot_ab.sh"

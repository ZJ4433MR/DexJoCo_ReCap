#!/usr/bin/env bash
set -euo pipefail

# Improvement over B: keep base-prompt collection and Evo-RL prompt tagging,
# but generate ACP indicators per episode with short-run smoothing.
export DEXJOCO_EVO_LEROBOT_VARIANT="${DEXJOCO_EVO_LEROBOT_VARIANT:-D_episode_topk_smooth_successaware}"
export DEXJOCO_EVO_COLLECT_PROMPT="${DEXJOCO_EVO_COLLECT_PROMPT:-base}"
export DEXJOCO_EVO_TRAIN_ACP_ENABLE="${DEXJOCO_EVO_TRAIN_ACP_ENABLE:-1}"
export DEXJOCO_EVO_ACP_BINARIZATION="${DEXJOCO_EVO_ACP_BINARIZATION:-episode_topk_smooth}"
export DEXJOCO_EVO_MIN_POSITIVE_RUN_LENGTH="${DEXJOCO_EVO_MIN_POSITIVE_RUN_LENGTH:-3}"
export DEXJOCO_EVO_SUCCESS_AWARE="${DEXJOCO_EVO_SUCCESS_AWARE:-true}"
export DEXJOCO_EVO_EVAL_PROMPT="${DEXJOCO_EVO_EVAL_PROMPT:-base}"
export DEXJOCO_EVAL_EPISODES="${DEXJOCO_EVAL_EPISODES:-500}"

exec bash jobs/57_dexjoco_click_mouse_evorl_lerobot_ab.sh

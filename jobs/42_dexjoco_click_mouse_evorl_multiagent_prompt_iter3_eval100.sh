#!/usr/bin/env bash
set -euo pipefail

# Evo-RL-style multi-round ReCap with autonomous prompt agents:
# - multiple prompt agents generate ACP suffix candidates each round,
# - a manager selects the suffix using prior prompt-history results,
# - positive rollout samples can cycle through all generated agent prompts,
# - every round is evaluated by default so later prompt choices get feedback.
export DEXJOCO_EVO_MERGE_POOL="${DEXJOCO_EVO_MERGE_POOL:-1}"
export DEXJOCO_EVO_ROUNDS="${DEXJOCO_EVO_ROUNDS:-3}"
export DEXJOCO_EVO_COLLECT_EPISODES="${DEXJOCO_EVO_COLLECT_EPISODES:-120}"
export DEXJOCO_EVO_TRAIN_STEPS="${DEXJOCO_EVO_TRAIN_STEPS:-1200}"
export DEXJOCO_EVAL_EPISODES="${DEXJOCO_EVAL_EPISODES:-100}"

export DEXJOCO_EVO_PROMPT_AGENTS="${DEXJOCO_EVO_PROMPT_AGENTS:-1}"
export DEXJOCO_EVO_PROMPT_BACKEND="${DEXJOCO_EVO_PROMPT_BACKEND:-heuristic}"
export DEXJOCO_EVO_PROMPT_ALLOW_HEURISTIC_FALLBACK="${DEXJOCO_EVO_PROMPT_ALLOW_HEURISTIC_FALLBACK:-0}"
export DEXJOCO_EVO_PROMPT_SELECTION_MODE="${DEXJOCO_EVO_PROMPT_SELECTION_MODE:-score}"
export DEXJOCO_EVO_PROMPT_EVAL_EACH_ROUND="${DEXJOCO_EVO_PROMPT_EVAL_EACH_ROUND:-1}"
export DEXJOCO_EVO_PROMPT_AGENT_IDS="${DEXJOCO_EVO_PROMPT_AGENT_IDS:-}"

exec bash jobs/37_dexjoco_click_mouse_evorl_multiround.sh

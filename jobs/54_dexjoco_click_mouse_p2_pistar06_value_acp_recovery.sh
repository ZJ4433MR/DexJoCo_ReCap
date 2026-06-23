#!/usr/bin/env bash
set -euo pipefail

cd "$EXP_DIR"
source scripts/dexjoco_common.sh

# Recovery-only job for the final P2 method. The full P2 ablation already
# completed collection plus base_lora/acp_lora/random_positive; this job
# extracts the frozen D0 rollout from that archived run and resumes only the
# Evo-RL pistar06 value-labeling ReCap path.
export DEXJOCO_TASK="${DEXJOCO_TASK:-click_mouse}"
export DEXJOCO_COLLECT_SEED="${DEXJOCO_COLLECT_SEED:-10000}"
export DEXJOCO_RECAP_EVAL_SEEDS="${DEXJOCO_RECAP_EVAL_SEEDS:-0 1 2}"
export DEXJOCO_EVAL_EPISODES="${DEXJOCO_EVAL_EPISODES:-100}"
export DEXJOCO_RECAP_TRAIN_STEPS="${DEXJOCO_RECAP_TRAIN_STEPS:-2000}"
export DEXJOCO_RECAP_INCLUDE_FAILURES="${DEXJOCO_RECAP_INCLUDE_FAILURES:-1}"
export DEXJOCO_RECAP_COLLECT_PROMPT_MODE="${DEXJOCO_RECAP_COLLECT_PROMPT_MODE:-base}"
export DEXJOCO_RECAP_FSDP_DEVICES="${DEXJOCO_RECAP_FSDP_DEVICES:-1}"
export DEXJOCO_RECAP_NUM_WORKERS="${DEXJOCO_RECAP_NUM_WORKERS:-0}"

RUN_ROOT="${RUN_ROOT:-$(_dexjoco_run_root)}"
D0_PREFIX="${DEXJOCO_RECAP_D0_PREFIX:-p2_d0_seed${DEXJOCO_COLLECT_SEED}}"
D0_NPZ="${DEXJOCO_RECAP_D0_NPZ:-$RUN_ROOT/${D0_PREFIX}_success_rollouts.npz}"
SOURCE_TAR="${DEXJOCO_RECAP_SOURCE_RESULTS_TAR:-/share/home/u23133/.cache/recap-sim-l40-stage/dexjoco_click_mouse_p2_frozen_pool_ablation_l40_v4/results.tar.gz}"
SOURCE_MEMBER="${DEXJOCO_RECAP_SOURCE_ROLLOUT_MEMBER:-./outputs/dexjoco_click_mouse_p2_frozen_pool/rollouts/collected_rollouts.npz}"
SOURCE_WORKSPACE_TAR="${DEXJOCO_RECAP_SOURCE_WORKSPACE_TAR:-/share/home/u23133/.cache/recap-sim-l40-stage/dexjoco_click_mouse_p2_frozen_pool_ablation_l40_v4/incoming.tar.gz}"

if [[ ! -d "$EXP_DIR/.local/dexjoco-src/.git" && -f "$SOURCE_WORKSPACE_TAR" ]]; then
  echo "[p2-recovery] restoring packaged DexJoCo fallback from $SOURCE_WORKSPACE_TAR"
  tar -xzf "$SOURCE_WORKSPACE_TAR" -C "$EXP_DIR/.." ./recap-sim-l40/.local/dexjoco-src
fi

if [[ ! -f "$D0_NPZ" ]]; then
  if [[ ! -f "$SOURCE_TAR" ]]; then
    echo "[p2-recovery] missing source tar: $SOURCE_TAR" >&2
    exit 2
  fi
  echo "[p2-recovery] extracting frozen D0 rollout from $SOURCE_TAR"
  mkdir -p "$(dirname "$D0_NPZ")"
  if ! tar -xOf "$SOURCE_TAR" "$SOURCE_MEMBER" > "$D0_NPZ"; then
    echo "[p2-recovery] primary member not found, retrying without leading ./" >&2
    tar -xOf "$SOURCE_TAR" "${SOURCE_MEMBER#./}" > "$D0_NPZ"
  fi
fi

if [[ ! -s "$D0_NPZ" ]]; then
  echo "[p2-recovery] extracted D0 rollout is empty: $D0_NPZ" >&2
  exit 2
fi

echo "[p2-recovery] using frozen D0 rollout: $D0_NPZ"
DEXJOCO_RECAP_ROLLOUT_NPZ_INPUT="$D0_NPZ" \
DEXJOCO_RECAP_DATA_PREFIX="p2_pistar06_value_acp_seed${DEXJOCO_COLLECT_SEED}" \
DEXJOCO_RECAP_OUTPUT_NAME="dexjoco_click_mouse_p2_frozen_pool/pistar06_value_acp" \
DEXJOCO_RECAP_EXP_NAME="p2_pistar06_value_acp_seed${DEXJOCO_COLLECT_SEED}" \
  exec bash jobs/43_dexjoco_click_mouse_recap_pistar06_full_eval100.sh

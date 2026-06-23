#!/usr/bin/env bash
set -euo pipefail

cd "$EXP_DIR"
source scripts/dexjoco_common.sh

# Recovery for P2 v4 after the non-pistar06 methods completed but the
# pistar06 branch failed before LeRobot conversion due missing value deps.
# This preserves the paired-ablation design by extracting the original frozen
# rollout pool from the v4 results bundle, then running only pistar06_value_acp.
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
RECOVERY_TAR="${DEXJOCO_P2_V4_RESULTS_TAR:-/share/home/u23133/.cache/recap-sim-l40-stage/dexjoco_click_mouse_p2_frozen_pool_ablation_l40_v4/results.tar.gz}"
EXTRACT_DIR="$RUN_ROOT/p2_v4_recovered_pool"
RECOVERED_NPZ="$RUN_ROOT/p2_d0_seed${DEXJOCO_COLLECT_SEED}_success_rollouts.npz"

if [[ ! -f "$RECOVERY_TAR" ]]; then
  echo "[p2-recovery] missing source results tar: $RECOVERY_TAR" >&2
  exit 2
fi

mkdir -p "$EXTRACT_DIR"
echo "[p2-recovery] extracting frozen rollout pool from $RECOVERY_TAR"
tar -xzf "$RECOVERY_TAR" -C "$EXTRACT_DIR" --wildcards \
  '*outputs/dexjoco_click_mouse_p2_frozen_pool/rollouts/policy_rollouts.npz'

found_npz="$(find "$EXTRACT_DIR" -path '*/outputs/dexjoco_click_mouse_p2_frozen_pool/rollouts/policy_rollouts.npz' -type f | head -n 1)"
if [[ -z "$found_npz" || ! -f "$found_npz" ]]; then
  echo "[p2-recovery] frozen rollout NPZ not found after extraction" >&2
  find "$EXTRACT_DIR" -maxdepth 6 -type f | sort >&2 || true
  exit 3
fi
cp "$found_npz" "$RECOVERED_NPZ"

python - "$RECOVERED_NPZ" <<'PY'
from pathlib import Path
import hashlib
import sys
import numpy as np

path = Path(sys.argv[1])
h = hashlib.sha256()
with path.open("rb") as f:
    for chunk in iter(lambda: f.read(1024 * 1024), b""):
        h.update(chunk)
with np.load(path, allow_pickle=False) as data:
    episode_ids = data["episode_id"].astype(np.int64)
    episodes = np.unique(episode_ids)
    successes = 0
    for episode in episodes:
        idx = int(np.flatnonzero(episode_ids == episode)[0])
        successes += int(bool(data["is_success"][idx]))
    print(f"[p2-recovery] frozen_npz={path}")
    print(f"[p2-recovery] sha256={h.hexdigest()}")
    print(f"[p2-recovery] frames={data['action'].shape[0]}")
    print(f"[p2-recovery] episodes={len(episodes)} successes={successes}")
PY

echo "[p2-recovery] running pistar06_value_acp on recovered frozen pool"
DEXJOCO_RECAP_ROLLOUT_NPZ_INPUT="$RECOVERED_NPZ" \
DEXJOCO_RECAP_DATA_PREFIX="p2_pistar06_value_acp_seed${DEXJOCO_COLLECT_SEED}" \
DEXJOCO_RECAP_OUTPUT_NAME="dexjoco_click_mouse_p2_frozen_pool/pistar06_value_acp" \
DEXJOCO_RECAP_EXP_NAME="p2_pistar06_value_acp_seed${DEXJOCO_COLLECT_SEED}" \
  bash jobs/43_dexjoco_click_mouse_recap_pistar06_full_eval100.sh

echo "[p2-recovery] pistar06_value_acp recovery finished"

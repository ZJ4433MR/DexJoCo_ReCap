#!/usr/bin/env bash
set -euo pipefail

if ! command -v conda >/dev/null 2>&1; then
  if [[ -f /etc/profile.d/modules.sh ]]; then
    # shellcheck disable=SC1091
    source /etc/profile.d/modules.sh
  fi
  module load miniconda3/24.1.2 cuda/12.1 >/dev/null 2>&1 || true
  if [[ -f /share/apps/miniconda3/etc/profile.d/conda.sh ]]; then
    # shellcheck disable=SC1091
    source /share/apps/miniconda3/etc/profile.d/conda.sh
  fi
fi

if [[ -n "${DEXJOCO_EXISTING_RUN_DIR:-}" ]]; then
  RUN_DIR="$DEXJOCO_EXISTING_RUN_DIR"
  EXP_DIR="${EXP_DIR:-$RUN_DIR/workspace/recap-sim-l40}"
  DEXJOCO_DIR="${DEXJOCO_DIR:-$RUN_DIR/dexjoco-src}"
  OPENPI_ENV_PREFIX="${OPENPI_ENV_PREFIX:-$RUN_DIR/conda_envs/openpi}"
  DEXJOCO_ENV_PREFIX="${DEXJOCO_ENV_PREFIX:-$RUN_DIR/conda_envs/dexjoco}"
  DEXJOCO_EVAL_REUSE_EXISTING_RUN=1
else
  if [[ -z "${EXP_DIR:-}" ]]; then
    echo "Either DEXJOCO_EXISTING_RUN_DIR or EXP_DIR is required" >&2
    exit 2
  fi
  RUN_ROOT="${RUN_ROOT:-$(cd "$EXP_DIR/../.." && pwd)}"
  RUN_DIR="$RUN_ROOT"
  DEXJOCO_DIR="${DEXJOCO_DIR:-$RUN_ROOT/dexjoco-src}"
  OPENPI_ENV_PREFIX="${OPENPI_ENV_PREFIX:-$RUN_ROOT/conda_envs/openpi}"
  DEXJOCO_ENV_PREFIX="${DEXJOCO_ENV_PREFIX:-$RUN_ROOT/conda_envs/dexjoco}"
  DEXJOCO_EVAL_REUSE_EXISTING_RUN=0
fi
DEXJOCO_TASK="${DEXJOCO_TASK:-click_mouse}"
DEXJOCO_EVAL_EPISODES="${DEXJOCO_EVAL_EPISODES:-500}"
DEXJOCO_EVAL_SEED="${DEXJOCO_EVAL_SEED:-0}"
DEXJOCO_EVAL_PROMPT="${DEXJOCO_EVAL_PROMPT:-positive}"
DEXJOCO_EVAL_OUTPUT_ROOT="${DEXJOCO_EVAL_OUTPUT_ROOT:-$RUN_DIR/results/eval_existing_policy}"
DEXJOCO_EVAL_NAME="${DEXJOCO_EVAL_NAME:-eval_${DEXJOCO_TASK}_${DEXJOCO_EVAL_EPISODES}}"
DEXJOCO_EVAL_POLICY_DIR="${DEXJOCO_EVAL_POLICY_DIR:-}"

export WANDB_MODE="${WANDB_MODE:-offline}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.99}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"

if [[ -z "$DEXJOCO_EVAL_POLICY_DIR" ]]; then
  echo "DEXJOCO_EVAL_POLICY_DIR is required" >&2
  exit 2
fi

if [[ ! -d "$DEXJOCO_EVAL_POLICY_DIR/params" ]]; then
  echo "policy params missing: $DEXJOCO_EVAL_POLICY_DIR/params" >&2
  exit 2
fi

source "$EXP_DIR/scripts/dexjoco_common.sh"

export MUJOCO_GL="${MUJOCO_GL:-egl}"
mkdir -p "$DEXJOCO_EVAL_OUTPUT_ROOT"

if [[ "$DEXJOCO_EVAL_REUSE_EXISTING_RUN" != "1" ]]; then
  if [[ "${DEXJOCO_USE_SOURCE_IN_PLACE:-0}" == "1" ]]; then
    if [[ ! -d "$DEXJOCO_DIR/openpi" ]]; then
      echo "DEXJOCO_USE_SOURCE_IN_PLACE=1 but source is missing openpi/: $DEXJOCO_DIR" >&2
      exit 2
    fi
    echo "[eval-existing] using source in place: $DEXJOCO_DIR"
  else
    prepare_dexjoco_source
  fi
  setup_openpi_env
  setup_dexjoco_env
  relax_openpi_websocket_timeouts
fi
if [[ ! -x "$OPENPI_ENV_PREFIX/bin/python" ]]; then
  echo "OpenPI env missing: $OPENPI_ENV_PREFIX" >&2
  exit 2
fi
if [[ ! -x "$DEXJOCO_ENV_PREFIX/bin/python" ]]; then
  echo "DexJoCo env missing: $DEXJOCO_ENV_PREFIX" >&2
  exit 2
fi

base_config="$DEXJOCO_DIR/configs/rand_obj/${DEXJOCO_TASK}.yaml"
positive_config="$DEXJOCO_EVAL_OUTPUT_ROOT/${DEXJOCO_TASK}_advantage_positive.yaml"
negative_config="$DEXJOCO_EVAL_OUTPUT_ROOT/${DEXJOCO_TASK}_advantage_negative.yaml"
conda run --no-capture-output --prefix "$DEXJOCO_ENV_PREFIX" python - "$base_config" "$positive_config" "$negative_config" <<'PY'
from pathlib import Path
import sys
import yaml

src = Path(sys.argv[1])
positive_dst = Path(sys.argv[2])
negative_dst = Path(sys.argv[3])
base_cfg = yaml.safe_load(src.read_text())

positive_cfg = dict(base_cfg)
if not positive_cfg["prompt"].rstrip().endswith("Advantage: positive"):
    positive_cfg["prompt"] = positive_cfg["prompt"].rstrip() + "\nAdvantage: positive"
positive_dst.write_text(yaml.safe_dump(positive_cfg, sort_keys=False))
print(f"[eval-existing] positive prompt: {positive_cfg['prompt']}")

negative_cfg = dict(base_cfg)
if not negative_cfg["prompt"].rstrip().endswith("Advantage: negative"):
    negative_cfg["prompt"] = negative_cfg["prompt"].rstrip() + "\nAdvantage: negative"
negative_dst.write_text(yaml.safe_dump(negative_cfg, sort_keys=False))
print(f"[eval-existing] negative prompt: {negative_cfg['prompt']}")
PY

eval_config="$positive_config"
eval_tag="positive"
if [[ "$DEXJOCO_EVAL_PROMPT" == "base" ]]; then
  eval_config="$base_config"
  eval_tag="base"
elif [[ "$DEXJOCO_EVAL_PROMPT" == "negative" ]]; then
  eval_config="$negative_config"
  eval_tag="negative"
elif [[ "$DEXJOCO_EVAL_PROMPT" != "positive" ]]; then
  echo "DEXJOCO_EVAL_PROMPT must be 'positive', 'negative', or 'base'" >&2
  exit 2
fi

port="${DEXJOCO_PORT:-$(dexjoco_default_port)}"
host="${DEXJOCO_HOST:-127.0.0.1}"
server_log="$DEXJOCO_EVAL_OUTPUT_ROOT/${DEXJOCO_EVAL_NAME}_${eval_tag}_server.log"
eval_output_dir="$DEXJOCO_EVAL_OUTPUT_ROOT/${DEXJOCO_EVAL_NAME}_${eval_tag}_seed${DEXJOCO_EVAL_SEED}"
eval_log="$DEXJOCO_EVAL_OUTPUT_ROOT/${DEXJOCO_EVAL_NAME}_${eval_tag}_seed${DEXJOCO_EVAL_SEED}.log"
summary="$DEXJOCO_EVAL_OUTPUT_ROOT/${DEXJOCO_EVAL_NAME}_summary.tsv"

cd "$DEXJOCO_DIR/openpi"
setsid conda run --no-capture-output --prefix "$OPENPI_ENV_PREFIX" python scripts/serve_policy.py \
  --port="$port" \
  policy:checkpoint \
  --policy.config="$DEXJOCO_TASK" \
  --policy.dir="$DEXJOCO_EVAL_POLICY_DIR" \
  > "$server_log" 2>&1 &
server_pid=$!

cleanup() {
  dexjoco_kill_server_group "$server_pid"
}
trap cleanup EXIT

start_ts="$(date +%s)"
while true; do
  if [[ -f "$server_log" ]] && grep -q "server listening on" "$server_log"; then
    break
  fi
  if ! dexjoco_server_group_alive "$server_pid"; then
    echo "[eval-existing] eval server exited early" >&2
    tail -200 "$server_log" >&2 || true
    exit 1
  fi
  if (( $(date +%s) - start_ts >= 900 )); then
    echo "[eval-existing] eval server did not become ready" >&2
    tail -200 "$server_log" >&2 || true
    exit 1
  fi
  sleep 5
done

echo "[eval-existing] eval name=$DEXJOCO_EVAL_NAME prompt=$eval_tag seed=$DEXJOCO_EVAL_SEED episodes=$DEXJOCO_EVAL_EPISODES policy=$DEXJOCO_EVAL_POLICY_DIR"
cd "$DEXJOCO_DIR"
conda run --no-capture-output --prefix "$DEXJOCO_ENV_PREFIX" dexjoco-openpi-eval \
  --config="$eval_config" \
  --seed="$DEXJOCO_EVAL_SEED" \
  --port="$port" \
  --host="$host" \
  --episodes="$DEXJOCO_EVAL_EPISODES" \
  --output="$eval_output_dir" \
  2>&1 | tee "$eval_log"

successes="NA"
episodes="$DEXJOCO_EVAL_EPISODES"
success_file="$(find "$eval_output_dir" -maxdepth 1 -name 'success_rate_*.txt' -printf '%f\n' | sort | head -1 || true)"
if [[ "$success_file" =~ success_rate_([0-9]+)_([0-9]+)\.txt ]]; then
  successes="${BASH_REMATCH[1]}"
  episodes="${BASH_REMATCH[2]}"
fi

echo -e "name\tprompt\tseed\teval_successes\teval_episodes\tpolicy_dir" | tee "$summary"
echo -e "$DEXJOCO_EVAL_NAME\t$eval_tag\t$DEXJOCO_EVAL_SEED\t$successes\t$episodes\t$DEXJOCO_EVAL_POLICY_DIR" | tee -a "$summary"
echo "[eval-existing] finished: $summary"

#!/usr/bin/env bash
set -euo pipefail

cd "$EXP_DIR"
source scripts/dexjoco_common.sh

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.40}"

RUN_ROOT="${RUN_ROOT:-$(_dexjoco_run_root)}"
DEXJOCO_TASK="${DEXJOCO_TASK:-water_plant}"
DEXJOCO_POLICY_CONFIG="${DEXJOCO_POLICY_CONFIG:-$DEXJOCO_TASK}"
DEXJOCO_EVAL_CONFIG="${DEXJOCO_EVAL_CONFIG:-./configs/rand_obj/water_plant.yaml}"
DEXJOCO_EVAL_EPISODES="${DEXJOCO_EVAL_EPISODES:-3}"
DEXJOCO_EVAL_SEED="${DEXJOCO_EVAL_SEED:-0}"
DEXJOCO_PORT="${DEXJOCO_PORT:-8000}"
DEXJOCO_HOST="${DEXJOCO_HOST:-127.0.0.1}"
OUT_DIR="$OUTPUT_DIR/dexjoco_pi05_${DEXJOCO_TASK}_eval"
SERVER_LOG="$OUT_DIR/server.log"
EVAL_LOG="$OUT_DIR/eval.log"
SUMMARY_FILE="$OUT_DIR/summary.txt"
mkdir -p "$OUT_DIR"

prepare_dexjoco_source
setup_dexjoco_env
setup_openpi_env
download_dexjoco_pi05_checkpoint

cleanup_server() {
  if [[ -n "${SERVER_PID:-}" ]] && kill -0 "$SERVER_PID" >/dev/null 2>&1; then
    echo "[job] stopping OpenPI server pid=$SERVER_PID"
    kill "$SERVER_PID" >/dev/null 2>&1 || true
    wait "$SERVER_PID" >/dev/null 2>&1 || true
  fi
  pkill -f "serve_policy.py --port=$DEXJOCO_PORT" >/dev/null 2>&1 || true
}
trap cleanup_server EXIT

echo "[job] starting OpenPI pi0.5 policy server on $DEXJOCO_HOST:$DEXJOCO_PORT"
cd "$DEXJOCO_DIR/openpi"
conda run --no-capture-output --prefix "$OPENPI_ENV_PREFIX" python scripts/serve_policy.py \
  --port="$DEXJOCO_PORT" \
  policy:checkpoint \
  --policy.config="$DEXJOCO_POLICY_CONFIG" \
  --policy.dir="../checkpoints/pi05_dexjoco_ckpt/$DEXJOCO_TASK" \
  > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!

if ! wait_for_log_pattern "$SERVER_LOG" "server listening on" 900; then
  echo "[job] server did not become ready in time" >&2
  tail -200 "$SERVER_LOG" >&2 || true
  exit 1
fi

echo "[job] running DexJoCo OpenPI eval: task=$DEXJOCO_TASK episodes=$DEXJOCO_EVAL_EPISODES seed=$DEXJOCO_EVAL_SEED"
cd "$DEXJOCO_DIR"
conda run --no-capture-output --prefix "$DEXJOCO_ENV_PREFIX" dexjoco-openpi-eval \
  --config="$DEXJOCO_EVAL_CONFIG" \
  --seed="$DEXJOCO_EVAL_SEED" \
  --port="$DEXJOCO_PORT" \
  --host="$DEXJOCO_HOST" \
  --episodes="$DEXJOCO_EVAL_EPISODES" \
  --output="$OUT_DIR/episodes" \
  2>&1 | tee "$EVAL_LOG"

{
  echo "task=$DEXJOCO_TASK"
  echo "policy_config=$DEXJOCO_POLICY_CONFIG"
  echo "eval_config=$DEXJOCO_EVAL_CONFIG"
  echo "episodes=$DEXJOCO_EVAL_EPISODES"
  echo "seed=$DEXJOCO_EVAL_SEED"
  find "$OUT_DIR/episodes" -maxdepth 1 -name 'success_rate_*.txt' -printf '%f\n' | sort
} | tee "$SUMMARY_FILE"

echo "[job] DexJoCo pi0.5 language-conditioned eval finished"

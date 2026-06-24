#!/usr/bin/env bash
set -euo pipefail

cd "$EXP_DIR"
source scripts/dexjoco_common.sh

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.40}"

RUN_ROOT="${RUN_ROOT:-$(_dexjoco_run_root)}"
DEXJOCO_TASKS="${DEXJOCO_TASKS:-click_mouse hammer_nail}"
DEXJOCO_EVAL_EPISODES="${DEXJOCO_EVAL_EPISODES:-3}"
DEXJOCO_EVAL_SEED="${DEXJOCO_EVAL_SEED:-0}"
DEXJOCO_PORT_BASE="${DEXJOCO_PORT_BASE:-$(dexjoco_default_port_base)}"
DEXJOCO_HOST="${DEXJOCO_HOST:-127.0.0.1}"
MATRIX_OUT_DIR="$OUTPUT_DIR/dexjoco_pi05_single_arm_matrix"
MATRIX_SUMMARY="$MATRIX_OUT_DIR/summary.tsv"
mkdir -p "$MATRIX_OUT_DIR"

prepare_dexjoco_source
setup_dexjoco_env
setup_openpi_env
relax_openpi_websocket_timeouts

SERVER_PIDS=()

cleanup_servers() {
  for pid in "${SERVER_PIDS[@]:-}"; do
    dexjoco_kill_server_group "$pid"
  done
}
trap cleanup_servers EXIT

echo -e "task\tstatus\tsuccesses\tepisodes\tsuccess_rate_file" | tee "$MATRIX_SUMMARY"

infra_failures=0
task_index=0
for task in $DEXJOCO_TASKS; do
  port=$((DEXJOCO_PORT_BASE + task_index))
  task_index=$((task_index + 1))

  config_path="./configs/rand_obj/${task}.yaml"
  task_out="$MATRIX_OUT_DIR/$task"
  server_log="$task_out/server.log"
  eval_log="$task_out/eval.log"
  mkdir -p "$task_out"

  if [[ ! -f "$DEXJOCO_DIR/$config_path" ]]; then
    echo "[job] missing config for task=$task: $config_path" | tee -a "$MATRIX_SUMMARY"
    echo -e "$task\tmissing_config\t0\t$DEXJOCO_EVAL_EPISODES\t" | tee -a "$MATRIX_SUMMARY"
    infra_failures=$((infra_failures + 1))
    continue
  fi

  echo "[job] ===== task=$task port=$port ====="
  DEXJOCO_TASK="$task" download_dexjoco_pi05_checkpoint

  echo "[job] starting OpenPI server for $task"
  cd "$DEXJOCO_DIR/openpi"
  setsid conda run --no-capture-output --prefix "$OPENPI_ENV_PREFIX" python scripts/serve_policy.py \
    --port="$port" \
    policy:checkpoint \
    --policy.config="$task" \
    --policy.dir="../checkpoints/pi05_dexjoco_ckpt/$task" \
    > "$server_log" 2>&1 &
  server_pid=$!
  SERVER_PIDS+=("$server_pid")

  if ! wait_for_log_pattern "$server_log" "server listening on" 900; then
    echo "[job] server did not become ready for $task" >&2
    tail -200 "$server_log" >&2 || true
    echo -e "$task\tserver_failed\t0\t$DEXJOCO_EVAL_EPISODES\t" | tee -a "$MATRIX_SUMMARY"
    infra_failures=$((infra_failures + 1))
    continue
  fi

  echo "[job] evaluating $task for $DEXJOCO_EVAL_EPISODES episodes"
  cd "$DEXJOCO_DIR"
  set +e
  conda run --no-capture-output --prefix "$DEXJOCO_ENV_PREFIX" dexjoco-openpi-eval \
    --config="$config_path" \
    --seed="$DEXJOCO_EVAL_SEED" \
    --port="$port" \
    --host="$DEXJOCO_HOST" \
    --episodes="$DEXJOCO_EVAL_EPISODES" \
    --output="$task_out/episodes" \
    2>&1 | tee "$eval_log"
  eval_status=${PIPESTATUS[0]}
  set -e

  success_file="$(find "$task_out/episodes" -maxdepth 1 -name 'success_rate_*.txt' -printf '%f\n' | sort | head -1 || true)"
  successes=0
  episodes="$DEXJOCO_EVAL_EPISODES"
  if [[ "$success_file" =~ success_rate_([0-9]+)_([0-9]+)\.txt ]]; then
    successes="${BASH_REMATCH[1]}"
    episodes="${BASH_REMATCH[2]}"
  fi

  if [[ "$eval_status" -ne 0 ]]; then
    status="eval_failed"
    infra_failures=$((infra_failures + 1))
  else
    status="ok"
  fi
  echo -e "$task\t$status\t$successes\t$episodes\t$success_file" | tee -a "$MATRIX_SUMMARY"

  if kill -0 "$server_pid" >/dev/null 2>&1; then
    echo "[job] stopping server for $task pid=$server_pid"
    dexjoco_kill_server_group "$server_pid"
  fi
done

if [[ "$infra_failures" -ne 0 ]]; then
  echo "[job] matrix finished with infrastructure failures: $infra_failures" >&2
  exit 1
fi

echo "[job] DexJoCo single-arm matrix finished"

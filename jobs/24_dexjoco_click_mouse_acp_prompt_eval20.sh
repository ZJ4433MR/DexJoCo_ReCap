#!/usr/bin/env bash
set -euo pipefail

cd "$EXP_DIR"
source scripts/dexjoco_common.sh

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.40}"

RUN_ROOT="${RUN_ROOT:-$(_dexjoco_run_root)}"
DEXJOCO_TASK="${DEXJOCO_TASK:-click_mouse}"
DEXJOCO_EVAL_EPISODES="${DEXJOCO_EVAL_EPISODES:-20}"
DEXJOCO_EVAL_SEED="${DEXJOCO_EVAL_SEED:-0}"
DEXJOCO_RECAP_EVAL_SEEDS="${DEXJOCO_RECAP_EVAL_SEEDS:-}"
DEXJOCO_RECAP_EVAL_VARIANTS="${DEXJOCO_RECAP_EVAL_VARIANTS:-baseline acp_positive}"
DEXJOCO_RECAP_EVAL_TIMEOUT_SECONDS="${DEXJOCO_RECAP_EVAL_TIMEOUT_SECONDS:-3600}"
DEXJOCO_RECAP_EVAL_RETRIES="${DEXJOCO_RECAP_EVAL_RETRIES:-1}"
DEXJOCO_PORT="${DEXJOCO_PORT:-$(dexjoco_default_port)}"
DEXJOCO_HOST="${DEXJOCO_HOST:-127.0.0.1}"
DEXJOCO_ACP_SUFFIX="${DEXJOCO_ACP_SUFFIX:- Use the high-advantage successful strategy.}"

OUT_DIR="$OUTPUT_DIR/dexjoco_click_mouse_acp_prompt_eval20"
SUMMARY="$OUT_DIR/summary.tsv"
CONFIG_DIR="$OUT_DIR/configs"
mkdir -p "$OUT_DIR" "$CONFIG_DIR"

prepare_dexjoco_source
setup_dexjoco_env
setup_openpi_env
relax_openpi_websocket_timeouts

base_config_rel="./configs/rand_obj/${DEXJOCO_TASK}.yaml"
base_config="$DEXJOCO_DIR/$base_config_rel"
baseline_config="$CONFIG_DIR/${DEXJOCO_TASK}_baseline.yaml"
acp_config="$CONFIG_DIR/${DEXJOCO_TASK}_acp_positive.yaml"

if [[ ! -f "$base_config" ]]; then
  echo "[job] missing eval config: $base_config_rel" >&2
  exit 2
fi

conda run --no-capture-output --prefix "$DEXJOCO_ENV_PREFIX" python - "$base_config" "$baseline_config" "$acp_config" "$DEXJOCO_ACP_SUFFIX" <<'PY'
from pathlib import Path
import sys
import yaml

src = Path(sys.argv[1])
baseline_dst = Path(sys.argv[2])
acp_dst = Path(sys.argv[3])
suffix = sys.argv[4]
cfg = yaml.safe_load(src.read_text())
Path(baseline_dst).write_text(yaml.safe_dump(cfg, sort_keys=False))

acp_cfg = dict(cfg)
acp_cfg["prompt"] = cfg["prompt"].rstrip() + suffix
Path(acp_dst).write_text(yaml.safe_dump(acp_cfg, sort_keys=False))

print(f"[job] baseline prompt: {cfg['prompt']}")
print(f"[job] acp-positive prompt: {acp_cfg['prompt']}")
PY

DEXJOCO_TASK="$DEXJOCO_TASK" download_dexjoco_pi05_checkpoint

SERVER_PIDS=()
cleanup_servers() {
  for pid in "${SERVER_PIDS[@]:-}"; do
    if [[ -n "$pid" ]] && kill -0 "$pid" >/dev/null 2>&1; then
      kill "$pid" >/dev/null 2>&1 || true
      wait "$pid" >/dev/null 2>&1 || true
    fi
  done
  pkill -f "serve_policy.py --port=${DEXJOCO_PORT}" >/dev/null 2>&1 || true
}
trap cleanup_servers EXIT

server_log="$OUT_DIR/server.log"
echo "[job] starting OpenPI server for $DEXJOCO_TASK on port $DEXJOCO_PORT"
cd "$DEXJOCO_DIR/openpi"
conda run --no-capture-output --prefix "$OPENPI_ENV_PREFIX" python scripts/serve_policy.py \
  --port="$DEXJOCO_PORT" \
  policy:checkpoint \
  --policy.config="$DEXJOCO_TASK" \
  --policy.dir="../checkpoints/pi05_dexjoco_ckpt/$DEXJOCO_TASK" \
  > "$server_log" 2>&1 &
server_pid=$!
SERVER_PIDS+=("$server_pid")

if ! wait_for_log_pattern "$server_log" "server listening on" 900; then
  echo "[job] server did not become ready for $DEXJOCO_TASK" >&2
  tail -200 "$server_log" >&2 || true
  exit 1
fi

if [[ -n "$DEXJOCO_RECAP_EVAL_SEEDS" ]]; then
  read -r -a eval_seeds <<< "$DEXJOCO_RECAP_EVAL_SEEDS"
else
  eval_seeds=("$DEXJOCO_EVAL_SEED")
fi
read -r -a eval_variants <<< "$DEXJOCO_RECAP_EVAL_VARIANTS"

echo -e "variant\tstatus\tsuccesses\tepisodes\tsuccess_rate_file\tconfig\teval_seed" | tee "$SUMMARY"

infra_failures=0
run_variant() {
  local variant="$1"
  local config_path="$2"
  local variant_out="$OUT_DIR/$variant"
  local eval_log="$variant_out/eval.log"
  mkdir -p "$variant_out"

  for eval_seed in "${eval_seeds[@]}"; do
    local seed_out="$variant_out/seed${eval_seed}"
    echo "[job] evaluating variant=$variant seed=$eval_seed episodes=$DEXJOCO_EVAL_EPISODES"
    cd "$DEXJOCO_DIR"
    local eval_status=1
    local attempt
    for attempt in $(seq 1 $((DEXJOCO_RECAP_EVAL_RETRIES + 1))); do
      rm -rf "$seed_out/episodes"
      set +e
      timeout --signal=TERM "${DEXJOCO_RECAP_EVAL_TIMEOUT_SECONDS}s" \
        conda run --no-capture-output --prefix "$DEXJOCO_ENV_PREFIX" dexjoco-openpi-eval \
          --config="$config_path" \
          --seed="$eval_seed" \
          --port="$DEXJOCO_PORT" \
          --host="$DEXJOCO_HOST" \
          --episodes="$DEXJOCO_EVAL_EPISODES" \
          --output="$seed_out/episodes" \
        2>&1 | tee "$variant_out/eval_seed${eval_seed}.attempt${attempt}.log"
      eval_status=${PIPESTATUS[0]}
      set -e
      cat "$variant_out/eval_seed${eval_seed}.attempt${attempt}.log" > "$variant_out/eval_seed${eval_seed}.log"
      if [[ "$eval_status" -eq 0 ]]; then
        break
      fi
      echo "[job] eval failed/timeout variant=$variant seed=$eval_seed attempt=$attempt status=$eval_status"
    done

    local success_file
    success_file="$(find "$seed_out/episodes" -maxdepth 1 -name 'success_rate_*.txt' -printf '%f\n' | sort | head -1 || true)"
    local successes=0
    local episodes="$DEXJOCO_EVAL_EPISODES"
    if [[ "$success_file" =~ success_rate_([0-9]+)_([0-9]+)\.txt ]]; then
      successes="${BASH_REMATCH[1]}"
      episodes="${BASH_REMATCH[2]}"
    fi

    local status="ok"
    if [[ "$eval_status" -ne 0 ]]; then
      status="eval_failed"
      infra_failures=$((infra_failures + 1))
    fi
    echo -e "$variant\t$status\t$successes\t$episodes\t$success_file\t$config_path\t$eval_seed" | tee -a "$SUMMARY"
  done
}

for variant in "${eval_variants[@]}"; do
  case "$variant" in
    baseline)
      run_variant "baseline" "$baseline_config"
      ;;
    acp_positive)
      run_variant "acp_positive" "$acp_config"
      ;;
    *)
      echo "[job] unknown eval variant: $variant" >&2
      exit 2
      ;;
  esac
done

if [[ "$infra_failures" -ne 0 ]]; then
  echo "[job] ACP prompt eval finished with infrastructure failures: $infra_failures" >&2
  exit 1
fi

echo "[job] DexJoCo click_mouse ACP prompt eval finished"

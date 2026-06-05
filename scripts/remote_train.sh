#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 ]]; then
  echo "Usage: remote_train.sh <run_id> <archive_path> <job_path> <export_path>" >&2
  exit 2
fi

RUN_ID="$1"
ARCHIVE_PATH="$2"
JOB_PATH="$3"
EXPORT_PATH="$4"

REMOTE_BASE="${REMOTE_BASE:-/tmp/$USER/recap-sim-l40}"
REMOTE_ENV_SETUP="${REMOTE_ENV_SETUP:-}"
REMOTE_BEFORE="${REMOTE_BEFORE:-}"
HF_TOKEN="${HF_TOKEN:-}"
KEEP_REMOTE="${KEEP_REMOTE:-0}"

RUN_DIR="${REMOTE_BASE}/${RUN_ID}"
WORK_DIR="${RUN_DIR}/workspace"
RESULTS_DIR="${RUN_DIR}/results"
LOG_DIR="${RUN_DIR}/logs"

mkdir -p "$WORK_DIR" "$RESULTS_DIR" "$LOG_DIR"

cleanup() {
  if [[ "$KEEP_REMOTE" != "1" ]]; then
    rm -rf "$RUN_DIR"
  else
    echo "[remote] KEEP_REMOTE=1, leaving run dir at $RUN_DIR" >&2
  fi
}
trap cleanup EXIT

echo "[remote] run id: $RUN_ID"
echo "[remote] run dir: $RUN_DIR"
echo "[remote] unpacking archive: $ARCHIVE_PATH"
tar -xzf "$ARCHIVE_PATH" -C "$WORK_DIR"

export EVORL_DIR="$WORK_DIR/Evo-RL-main"
export EXP_DIR="$WORK_DIR/recap-sim-l40"
export OUTPUT_DIR="$RESULTS_DIR/outputs"
export HF_HOME="$RUN_DIR/hf_home"
export HF_DATASETS_CACHE="$RUN_DIR/hf_datasets"
export TRANSFORMERS_CACHE="$RUN_DIR/transformers"
export PIP_CACHE_DIR="$RUN_DIR/pip_cache"
export WANDB_DIR="$RESULTS_DIR/wandb"
export PYTHONPATH="$EVORL_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

mkdir -p "$OUTPUT_DIR" "$HF_HOME" "$HF_DATASETS_CACHE" "$TRANSFORMERS_CACHE" "$WANDB_DIR"

if [[ -n "$HF_TOKEN" ]]; then
  export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
  export HF_TOKEN="$HF_TOKEN"
fi

if [[ -n "$REMOTE_BEFORE" ]]; then
  echo "[remote] running REMOTE_BEFORE"
  bash -lc "$REMOTE_BEFORE"
fi

if [[ -n "$REMOTE_ENV_SETUP" ]]; then
  echo "[remote] activating remote environment"
  # shellcheck disable=SC1090
  source <(echo "$REMOTE_ENV_SETUP")
fi

echo "[remote] system info" | tee "$LOG_DIR/system.log"
{
  hostname
  date
  nvidia-smi || true
  which python || true
  python --version || true
  python - <<'PY' || true
import torch
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu", torch.cuda.get_device_name(0))
    print("capability", torch.cuda.get_device_capability(0))
PY
} | tee -a "$LOG_DIR/system.log"

cd "$EXP_DIR"
echo "[remote] running job: $JOB_PATH"
set +e
bash "$JOB_PATH" 2>&1 | tee "$LOG_DIR/job.log"
JOB_STATUS=${PIPESTATUS[0]}
set -e

echo "$JOB_STATUS" > "$RESULTS_DIR/exit_code.txt"
cp "$LOG_DIR/system.log" "$RESULTS_DIR/system.log"
cp "$LOG_DIR/job.log" "$RESULTS_DIR/job.log"

echo "[remote] packing export: $EXPORT_PATH"
tar -czf "$EXPORT_PATH" -C "$RESULTS_DIR" .

exit "$JOB_STATUS"

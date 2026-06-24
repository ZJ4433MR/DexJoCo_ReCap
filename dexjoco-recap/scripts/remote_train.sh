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

REMOTE_BASE="${REMOTE_BASE:-/tmp/$USER/dexjoco-recap}"
REMOTE_ENV_SETUP="${REMOTE_ENV_SETUP:-}"
REMOTE_BEFORE="${REMOTE_BEFORE:-}"
HF_TOKEN="${HF_TOKEN:-}"
KEEP_REMOTE="${KEEP_REMOTE:-0}"
REMOTE_EXPORT_MODE="${REMOTE_EXPORT_MODE:-full}"

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

export EVORL_DIR="$WORK_DIR/lerobot-src"
export EXP_DIR="$WORK_DIR/dexjoco-recap"
export OUTPUT_DIR="$RESULTS_DIR/outputs"
export PYDEPS_DIR="$RUN_DIR/pydeps"
export HF_HOME="${HF_HOME:-$REMOTE_BASE/_shared_hf_home}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$REMOTE_BASE/_shared_hf_datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$REMOTE_BASE/_shared_transformers}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$REMOTE_BASE/_shared_pip_cache}"
export WANDB_DIR="$RESULTS_DIR/wandb"
export PYTHONPATH="$PYDEPS_DIR:$EVORL_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_ENABLE_HF_TRANSFER=0
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-120}"
export HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-60}"

mkdir -p "$OUTPUT_DIR" "$PYDEPS_DIR" "$HF_HOME" "$HF_DATASETS_CACHE" "$TRANSFORMERS_CACHE" "$WANDB_DIR"

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

echo "[remote] packing export: $EXPORT_PATH mode=$REMOTE_EXPORT_MODE"
if [[ "$REMOTE_EXPORT_MODE" == "minimal" ]]; then
  MIN_EXPORT_DIR="$RUN_DIR/export_minimal"
  rm -rf "$MIN_EXPORT_DIR"
  mkdir -p "$MIN_EXPORT_DIR"

  while IFS= read -r -d '' file; do
    rel="${file#$RESULTS_DIR/}"
    mkdir -p "$MIN_EXPORT_DIR/$(dirname "$rel")"
    cp -a "$file" "$MIN_EXPORT_DIR/$rel"
  done < <(
    find "$RESULTS_DIR" -type f \
      \( -name '*.json' -o -name '*.log' -o -name '*.tsv' -o -name '*.txt' -o -name '*.yaml' \) \
      -print0
  )

  if [[ -d "$OUTPUT_DIR" ]]; then
    while IFS= read -r -d '' dir; do
      rel="${dir#$RESULTS_DIR/}"
      mkdir -p "$MIN_EXPORT_DIR/$(dirname "$rel")"
      cp -a "$dir" "$MIN_EXPORT_DIR/$rel"
    done < <(find "$OUTPUT_DIR" -type d -path '*/export_checkpoints/*' -prune -print0)
  fi

  tar -czhf "$EXPORT_PATH" -C "$MIN_EXPORT_DIR" .
else
  tar -czhf "$EXPORT_PATH" -C "$RESULTS_DIR" .
fi

exit "$JOB_STATUS"

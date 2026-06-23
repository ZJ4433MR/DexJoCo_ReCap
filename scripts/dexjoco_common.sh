#!/usr/bin/env bash
set -euo pipefail

DEXJOCO_REPO="${DEXJOCO_REPO:-https://github.com/brave-eai/dexjoco.git}"
DEXJOCO_COMMIT="${DEXJOCO_COMMIT:-8d23b0fab23b17a58c4b55f3942e17013aaf8267}"
DEXJOCO_PREFER_LOCAL_SOURCE="${DEXJOCO_PREFER_LOCAL_SOURCE:-1}"

_dexjoco_run_root() {
  if [[ -z "${EXP_DIR:-}" ]]; then
    echo "EXP_DIR is not set; run through scripts/remote_train.sh" >&2
    return 2
  fi
  cd "$EXP_DIR/../.." && pwd
}

dexjoco_default_port_base() {
  local seed="${SLURM_JOB_ID:-${SLURM_JOBID:-$$}}"
  seed="${seed//[^0-9]/}"
  if [[ -z "$seed" ]]; then
    seed="$$"
  fi
  echo $((20000 + (10#$seed % 30000)))
}

dexjoco_default_port() {
  dexjoco_default_port_base
}

dexjoco_server_group_alive() {
  local pid="$1"
  [[ -n "$pid" ]] && { kill -0 "$pid" >/dev/null 2>&1 || pgrep -g "$pid" >/dev/null 2>&1; }
}

dexjoco_kill_server_group() {
  local pid="$1"
  [[ -z "$pid" ]] && return 0
  kill -- "-$pid" >/dev/null 2>&1 || kill "$pid" >/dev/null 2>&1 || true
  for _ in {1..20}; do
    if ! dexjoco_server_group_alive "$pid"; then
      wait "$pid" >/dev/null 2>&1 || true
      return 0
    fi
    sleep 0.5
  done
  kill -9 -- "-$pid" >/dev/null 2>&1 || kill -9 "$pid" >/dev/null 2>&1 || true
  wait "$pid" >/dev/null 2>&1 || true
}

dexjoco_remove_env_prefix() {
  local env_prefix="$1"
  RUN_ROOT="${RUN_ROOT:-$(_dexjoco_run_root)}"
  case "$env_prefix" in
    "$RUN_ROOT"/conda_envs/*) rm -rf "$env_prefix" ;;
    *) echo "[dexjoco] refusing to remove unexpected env path: $env_prefix" >&2; return 2 ;;
  esac
}

dexjoco_conda_env_create_retry() {
  local env_prefix="$1"
  local env_file="$2"
  local rc=0

  for attempt in 1 2 3; do
    echo "[dexjoco] conda env create attempt $attempt: $env_prefix"
    if conda env create --prefix "$env_prefix" -f "$env_file"; then
      return 0
    fi
    rc=$?
    echo "[dexjoco] conda env create failed attempt $attempt rc=$rc" >&2
    dexjoco_remove_env_prefix "$env_prefix" || true
    conda clean -i -y >/dev/null 2>&1 || true
    sleep $((attempt * 15))
  done

  return "$rc"
}

prepare_dexjoco_source() {
  RUN_ROOT="${RUN_ROOT:-$(_dexjoco_run_root)}"
  DEXJOCO_DIR="${DEXJOCO_DIR:-$RUN_ROOT/dexjoco-src}"
  DEXJOCO_LOCAL_SOURCE="${DEXJOCO_LOCAL_SOURCE:-$EXP_DIR/.local/dexjoco-src}"

  if [[ "$DEXJOCO_PREFER_LOCAL_SOURCE" == "1" && ! -d "$DEXJOCO_DIR/.git" && ( -d "$DEXJOCO_LOCAL_SOURCE/.git" || -d "$DEXJOCO_LOCAL_SOURCE/openpi" ) ]]; then
    echo "[dexjoco] using packaged fallback source: $DEXJOCO_LOCAL_SOURCE"
    case "$DEXJOCO_DIR" in
      "$RUN_ROOT"/*) rm -rf "$DEXJOCO_DIR" ;;
      *) echo "[dexjoco] refusing to replace unexpected path: $DEXJOCO_DIR" >&2; return 2 ;;
    esac
    cp -a "$DEXJOCO_LOCAL_SOURCE" "$DEXJOCO_DIR"
  fi

  if [[ ! -d "$DEXJOCO_DIR/.git" ]]; then
    echo "[dexjoco] cloning $DEXJOCO_REPO into $DEXJOCO_DIR"
    clone_ok=0
    for attempt in 1 2 3; do
      echo "[dexjoco] clone attempt $attempt"
      if git -c http.version=HTTP/1.1 clone --depth 1 "$DEXJOCO_REPO" "$DEXJOCO_DIR"; then
        clone_ok=1
        break
      fi
      case "$DEXJOCO_DIR" in
        "$RUN_ROOT"/*) rm -rf "$DEXJOCO_DIR" ;;
        *) echo "[dexjoco] refusing to remove unexpected path: $DEXJOCO_DIR" >&2; return 2 ;;
      esac
      sleep $((attempt * 5))
    done
    if [[ "$clone_ok" != "1" ]]; then
      echo "[dexjoco] remote clone failed"

      if [[ ! -d "$DEXJOCO_LOCAL_SOURCE/.git" && ! -d "$DEXJOCO_LOCAL_SOURCE/openpi" ]]; then
        echo "[dexjoco] no packaged fallback source found at $DEXJOCO_LOCAL_SOURCE" >&2
        return 128
      fi
      echo "[dexjoco] using packaged fallback source: $DEXJOCO_LOCAL_SOURCE"
      cp -a "$DEXJOCO_LOCAL_SOURCE" "$DEXJOCO_DIR"
    fi
  fi

  cd "$DEXJOCO_DIR"
  if [[ -d "$DEXJOCO_DIR/.git" ]]; then
    git -c core.autocrlf=false fetch --depth 1 origin "$DEXJOCO_COMMIT" >/dev/null 2>&1 || true
    git -c core.autocrlf=false checkout -q --detach "$DEXJOCO_COMMIT"
    echo "[dexjoco] source commit: $(git rev-parse HEAD)"
  else
    echo "[dexjoco] source commit: packaged non-git fallback"
  fi
}

setup_dexjoco_env() {
  RUN_ROOT="${RUN_ROOT:-$(_dexjoco_run_root)}"
  DEXJOCO_DIR="${DEXJOCO_DIR:-$RUN_ROOT/dexjoco-src}"
  DEXJOCO_ENV_PREFIX="${DEXJOCO_ENV_PREFIX:-$RUN_ROOT/conda_envs/dexjoco}"

  if [[ ! -x "$DEXJOCO_ENV_PREFIX/bin/python" ]]; then
    echo "[dexjoco] creating temporary conda env: $DEXJOCO_ENV_PREFIX"
    cd "$DEXJOCO_DIR"
    dexjoco_conda_env_create_retry "$DEXJOCO_ENV_PREFIX" environment-dexjoco.yaml
  else
    echo "[dexjoco] reusing conda env: $DEXJOCO_ENV_PREFIX"
  fi
}

setup_openpi_env() {
  RUN_ROOT="${RUN_ROOT:-$(_dexjoco_run_root)}"
  DEXJOCO_DIR="${DEXJOCO_DIR:-$RUN_ROOT/dexjoco-src}"
  OPENPI_ENV_PREFIX="${OPENPI_ENV_PREFIX:-$RUN_ROOT/conda_envs/openpi}"

  if [[ ! -x "$OPENPI_ENV_PREFIX/bin/python" ]]; then
    echo "[dexjoco] creating temporary OpenPI conda env: $OPENPI_ENV_PREFIX"
    cd "$DEXJOCO_DIR/openpi"
    dexjoco_conda_env_create_retry "$OPENPI_ENV_PREFIX" environment-openpi.yaml
    conda run --no-capture-output --prefix "$OPENPI_ENV_PREFIX" python -m pip install --upgrade pip
    conda run --no-capture-output --prefix "$OPENPI_ENV_PREFIX" python -m pip install "huggingface-hub[cli]>=0.34,<0.36"
    conda run --no-capture-output --prefix "$OPENPI_ENV_PREFIX" python -m pip install lerobot --no-deps
    conda run --no-capture-output --prefix "$OPENPI_ENV_PREFIX" python -m pip install -e .
    conda run --no-capture-output --prefix "$OPENPI_ENV_PREFIX" python -m pip install -e ./packages/openpi-client
  else
    echo "[dexjoco] reusing OpenPI conda env: $OPENPI_ENV_PREFIX"
  fi
}

relax_openpi_websocket_timeouts() {
  RUN_ROOT="${RUN_ROOT:-$(_dexjoco_run_root)}"
  DEXJOCO_DIR="${DEXJOCO_DIR:-$RUN_ROOT/dexjoco-src}"

  python - "$DEXJOCO_DIR" <<'PY'
from pathlib import Path
import sys

root = Path(sys.argv[1])
client_file = root / "openpi/packages/openpi-client/src/openpi_client/websocket_client_policy.py"
server_file = root / "openpi/src/openpi/serving/websocket_policy_server.py"

client_text = client_file.read_text()
client_old = "compression=None, max_size=None, additional_headers=headers"
client_new = (
    "compression=None, max_size=None, additional_headers=headers, "
    "ping_interval=None, open_timeout=60, close_timeout=10"
)
if client_new not in client_text:
    client_file.write_text(client_text.replace(client_old, client_new))
    print(f"[dexjoco] patched websocket client timeout: {client_file}")
else:
    print(f"[dexjoco] websocket client timeout already patched: {client_file}")

server_text = server_file.read_text()
server_old = "process_request=_health_check,\n        ) as server:"
server_new = "process_request=_health_check,\n            ping_interval=None,\n        ) as server:"
if server_new not in server_text:
    server_file.write_text(server_text.replace(server_old, server_new))
    print(f"[dexjoco] patched websocket server timeout: {server_file}")
else:
    print(f"[dexjoco] websocket server timeout already patched: {server_file}")
PY
}

download_dexjoco_pi05_checkpoint() {
  RUN_ROOT="${RUN_ROOT:-$(_dexjoco_run_root)}"
  DEXJOCO_DIR="${DEXJOCO_DIR:-$RUN_ROOT/dexjoco-src}"
  OPENPI_ENV_PREFIX="${OPENPI_ENV_PREFIX:-$RUN_ROOT/conda_envs/openpi}"
  DEXJOCO_HF_MODEL_REPO="${DEXJOCO_HF_MODEL_REPO:-DexJoCo/DexJoCo-Pi05}"
  DEXJOCO_TASK="${DEXJOCO_TASK:-water_plant}"
  DEXJOCO_CHECKPOINT_INCLUDE_TRAIN_STATE="${DEXJOCO_CHECKPOINT_INCLUDE_TRAIN_STATE:-0}"

  local include_args=(
    --include "pi05_dexjoco_ckpt/$DEXJOCO_TASK/_CHECKPOINT_METADATA"
    --include "pi05_dexjoco_ckpt/$DEXJOCO_TASK/assets/**"
    --include "pi05_dexjoco_ckpt/$DEXJOCO_TASK/params/**"
  )

  if [[ "$DEXJOCO_CHECKPOINT_INCLUDE_TRAIN_STATE" == "1" ]]; then
    include_args+=(--include "pi05_dexjoco_ckpt/$DEXJOCO_TASK/train_state/**")
  fi

  mkdir -p "$DEXJOCO_DIR/checkpoints"
  echo "[dexjoco] downloading checkpoint: $DEXJOCO_HF_MODEL_REPO pi05_dexjoco_ckpt/$DEXJOCO_TASK"
  for ((i = 0; i < ${#include_args[@]}; i += 2)); do
    pattern="${include_args[$((i + 1))]}"
    echo "[dexjoco] hf include: $pattern"
    conda run --no-capture-output --prefix "$OPENPI_ENV_PREFIX" hf download \
      "$DEXJOCO_HF_MODEL_REPO" \
      --local-dir "$DEXJOCO_DIR/checkpoints" \
      --include "$pattern"
  done
}

download_dexjoco_lerobot_dataset() {
  RUN_ROOT="${RUN_ROOT:-$(_dexjoco_run_root)}"
  OPENPI_ENV_PREFIX="${OPENPI_ENV_PREFIX:-$RUN_ROOT/conda_envs/openpi}"
  DEXJOCO_HF_DATASET_REPO="${DEXJOCO_HF_DATASET_REPO:-DexJoCo/DexJoCo-Datasets-LeRobot}"
  DEXJOCO_TASK="${DEXJOCO_TASK:-click_mouse}"
  DEXJOCO_LEROBOT_DATASET_SUBDIR="${DEXJOCO_LEROBOT_DATASET_SUBDIR:-dexjoco_lerobot_datasets}"
  DEXJOCO_LEROBOT_DATASET_DOWNLOAD_ROOT="${DEXJOCO_LEROBOT_DATASET_DOWNLOAD_ROOT:-$RUN_ROOT/dexjoco_official_lerobot}"

  mkdir -p "$DEXJOCO_LEROBOT_DATASET_DOWNLOAD_ROOT"
  echo "[dexjoco] downloading LeRobot dataset: $DEXJOCO_HF_DATASET_REPO $DEXJOCO_LEROBOT_DATASET_SUBDIR/$DEXJOCO_TASK"
  conda run --no-capture-output --prefix "$OPENPI_ENV_PREFIX" hf download \
    "$DEXJOCO_HF_DATASET_REPO" \
    --repo-type dataset \
    --local-dir "$DEXJOCO_LEROBOT_DATASET_DOWNLOAD_ROOT" \
    --include "$DEXJOCO_LEROBOT_DATASET_SUBDIR/$DEXJOCO_TASK/**"

  DEXJOCO_OFFICIAL_LEROBOT_ROOT="$DEXJOCO_LEROBOT_DATASET_DOWNLOAD_ROOT/$DEXJOCO_LEROBOT_DATASET_SUBDIR/$DEXJOCO_TASK"
  if [[ ! -d "$DEXJOCO_OFFICIAL_LEROBOT_ROOT/meta" ]]; then
    echo "[dexjoco] downloaded dataset root is missing meta/: $DEXJOCO_OFFICIAL_LEROBOT_ROOT" >&2
    return 2
  fi
  export DEXJOCO_OFFICIAL_LEROBOT_ROOT
  echo "[dexjoco] official LeRobot root: $DEXJOCO_OFFICIAL_LEROBOT_ROOT"
}

wait_for_port() {
  local host="$1"
  local port="$2"
  local timeout_s="$3"
  local start_ts
  start_ts="$(date +%s)"

  while true; do
    if python - "$host" "$port" <<'PY'
import socket
import sys

host = sys.argv[1]
port = int(sys.argv[2])
with socket.create_connection((host, port), timeout=2):
    pass
PY
    then
      return 0
    fi

    if (( $(date +%s) - start_ts >= timeout_s )); then
      return 1
    fi
    sleep 5
  done
}

wait_for_log_pattern() {
  local log_path="$1"
  local pattern="$2"
  local timeout_s="$3"
  local start_ts
  start_ts="$(date +%s)"

  while true; do
    if [[ -f "$log_path" ]] && grep -q "$pattern" "$log_path"; then
      return 0
    fi
    if (( $(date +%s) - start_ts >= timeout_s )); then
      return 1
    fi
    sleep 5
  done
}

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

prepare_dexjoco_source() {
  RUN_ROOT="${RUN_ROOT:-$(_dexjoco_run_root)}"
  DEXJOCO_DIR="${DEXJOCO_DIR:-$RUN_ROOT/dexjoco-src}"
  DEXJOCO_LOCAL_SOURCE="${DEXJOCO_LOCAL_SOURCE:-$EXP_DIR/.local/dexjoco-src}"

  if [[ "$DEXJOCO_PREFER_LOCAL_SOURCE" == "1" && ! -d "$DEXJOCO_DIR/.git" && -d "$DEXJOCO_LOCAL_SOURCE/.git" ]]; then
    echo "[dexjoco] using packaged fallback source: $DEXJOCO_LOCAL_SOURCE"
    case "$DEXJOCO_DIR" in
      "$RUN_ROOT"/*) rm -rf "$DEXJOCO_DIR" ;;
      *) echo "[dexjoco] refusing to replace unexpected path: $DEXJOCO_DIR" >&2; return 2 ;;
    esac
    cp -a "$DEXJOCO_LOCAL_SOURCE" "$DEXJOCO_DIR"
  fi

  if [[ ! -d "$DEXJOCO_DIR/.git" ]]; then
    echo "[dexjoco] cloning $DEXJOCO_REPO into $DEXJOCO_DIR"
    if ! git clone --depth 1 "$DEXJOCO_REPO" "$DEXJOCO_DIR"; then
      echo "[dexjoco] remote clone failed"
      case "$DEXJOCO_DIR" in
        "$RUN_ROOT"/*) rm -rf "$DEXJOCO_DIR" ;;
        *) echo "[dexjoco] refusing to remove unexpected path: $DEXJOCO_DIR" >&2; return 2 ;;
      esac

      if [[ ! -d "$DEXJOCO_LOCAL_SOURCE/.git" ]]; then
        echo "[dexjoco] no packaged fallback source found at $DEXJOCO_LOCAL_SOURCE" >&2
        return 128
      fi
      echo "[dexjoco] using packaged fallback source: $DEXJOCO_LOCAL_SOURCE"
      cp -a "$DEXJOCO_LOCAL_SOURCE" "$DEXJOCO_DIR"
    fi
  fi

  cd "$DEXJOCO_DIR"
  git -c core.autocrlf=false fetch --depth 1 origin "$DEXJOCO_COMMIT" >/dev/null 2>&1 || true
  git -c core.autocrlf=false checkout -q --detach "$DEXJOCO_COMMIT"
  echo "[dexjoco] source commit: $(git rev-parse HEAD)"
}

setup_dexjoco_env() {
  RUN_ROOT="${RUN_ROOT:-$(_dexjoco_run_root)}"
  DEXJOCO_DIR="${DEXJOCO_DIR:-$RUN_ROOT/dexjoco-src}"
  DEXJOCO_ENV_PREFIX="${DEXJOCO_ENV_PREFIX:-$RUN_ROOT/conda_envs/dexjoco}"

  if [[ ! -x "$DEXJOCO_ENV_PREFIX/bin/python" ]]; then
    echo "[dexjoco] creating temporary conda env: $DEXJOCO_ENV_PREFIX"
    cd "$DEXJOCO_DIR"
    conda env create --prefix "$DEXJOCO_ENV_PREFIX" -f environment-dexjoco.yaml
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
    conda env create --prefix "$OPENPI_ENV_PREFIX" -f environment-openpi.yaml
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

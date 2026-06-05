#!/usr/bin/env bash
set -euo pipefail

cd "$EVORL_DIR"

echo "[job] Basic Python/CUDA check"
python - <<'PY'
import torch
import lerobot

print("lerobot", lerobot.__file__)
print("torch", torch.__version__)
print("cuda", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name(0))
PY

echo "[job] Running focused RECAP/ACP unit tests"
pytest -q \
  tests/value/test_pistar06_algorithms.py \
  tests/value/test_lerobot_value_infer_utils.py \
  tests/training/test_acp_prompt_hook.py \
  tests/training/test_acp_dataset_stats.py

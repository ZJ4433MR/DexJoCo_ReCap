#!/usr/bin/env bash
set -euo pipefail

ensure_evorl_pistar06_deps() {
  RUN_ROOT="${RUN_ROOT:-$(_dexjoco_run_root)}"
  DEXJOCO_RECAP_PISTAR06_PYDEPS_DIR="${DEXJOCO_RECAP_PISTAR06_PYDEPS_DIR:-$RUN_ROOT/evorl_value_pydeps}"
  local target="$DEXJOCO_RECAP_PISTAR06_PYDEPS_DIR"
  mkdir -p "$target"
  echo "[dexjoco] ensuring Evo-RL pistar06 dependencies in $target"
  python - "$target" <<'PY'
from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import subprocess
import sys

target = Path(sys.argv[1])
target.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(target))
index_url = os.environ.get("DEXJOCO_RECAP_PIP_INDEX_URL", "https://pypi.tuna.tsinghua.edu.cn/simple")

top_level_no_deps = {
    "accelerate": "accelerate>=1.10.0,<2.0.0",
    "datasets": "datasets>=4.0.0,<4.2.0",
    "diffusers": "diffusers>=0.27.2,<0.36.0",
    "draccus": "draccus==0.10.0",
    "huggingface_hub": "huggingface-hub>=0.34.2,<0.36.0",
    "safetensors": "safetensors>=0.4.3,<1.0.0",
    "termcolor": "termcolor>=2.4.0,<4.0.0",
    "tokenizers": "tokenizers>=0.22.0,<0.23.0",
    "transformers": "transformers>=4.57.1,<5.0.0",
}

deps = [
    "aiohttp",
    "av>=15.0.0,<16.0.0",
    "cloudpickle",
    "deepdiff>=7.0.1,<9.0.0",
    "dill<0.4.1,>=0.3.0",
    "einops>=0.8.0,<0.9.0",
    "filelock",
    "fsspec[http]<=2025.9.0,>=2023.1.0",
    "gymnasium>=1.1.1,<2.0.0",
    "hf-xet",
    "imageio[ffmpeg]>=2.34.0,<3.0.0",
    "jsonlines>=4.0.0,<5.0.0",
    "mergedeep",
    "multiprocess<0.70.17",
    "numpy<2",
    "opencv-python-headless>=4.9.0,<4.13.0",
    "orderly-set",
    "packaging>=24.2,<26.0",
    "pandas",
    "Pillow>=10.0.0,<13.0.0",
    "protobuf>=4,<7",
    "psutil",
    "pyarrow>=21.0.0",
    "pyserial>=3.5,<4.0",
    "pyyaml",
    "pyyaml-include~=1.4",
    "regex",
    "requests",
    "scipy>=1.10.1,<1.15",
    "sentry-sdk",
    "sentencepiece>=0.2.0",
    "setuptools>=71.0.0,<81.0.0",
    "six",
    "toml",
    "tomli",
    "tqdm",
    "typing-extensions",
    "typing-inspect",
    "urllib3<3",
    "wandb>=0.24.0,<0.25.0",
    "xxhash",
]

missing_top = [pkg for module, pkg in top_level_no_deps.items() if importlib.util.find_spec(module) is None]
if missing_top:
    print("missing Evo-RL top-level packages:", missing_top)
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--target",
            str(target),
            "--upgrade",
            "--no-deps",
            "-i",
            index_url,
            *missing_top,
        ]
    )

print("installing/verifying Evo-RL value dependency packages")
subprocess.check_call(
    [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--target",
        str(target),
        "--upgrade",
        "-i",
        index_url,
        *deps,
    ]
)
PY
}

verify_evorl_pistar06_runtime() {
  local pythonpath="$1"
  DEXJOCO_RECAP_PISTAR06_PYDEPS_DIR="${DEXJOCO_RECAP_PISTAR06_PYDEPS_DIR:-$RUN_ROOT/evorl_value_pydeps}"
  echo "[dexjoco] verifying Evo-RL pistar06 runtime imports"
  PYTHONPATH="$pythonpath" python - "$DEXJOCO_RECAP_PISTAR06_PYDEPS_DIR" <<'PY'
from __future__ import annotations

import sys
from pathlib import Path

target = Path(sys.argv[1])
print(f"[dexjoco] pistar06 pydeps target: {target}")
print(f"[dexjoco] python executable: {sys.executable}")

try:
    import datasets  # noqa: F401
    from scipy.signal import savgol_filter  # noqa: F401
    from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: F401
except Exception:
    print("[dexjoco] failed to import Evo-RL pistar06 runtime dependencies", file=sys.stderr)
    print(f"[dexjoco] sys.path head: {sys.path[:8]}", file=sys.stderr)
    raise

print("[dexjoco] Evo-RL pistar06 runtime imports ok")
PY
}

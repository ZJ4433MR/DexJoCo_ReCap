#!/usr/bin/env bash
set -euo pipefail

cd "$EXP_DIR"
source scripts/dexjoco_common.sh

export MUJOCO_GL="${MUJOCO_GL:-egl}"
RUN_ROOT="${RUN_ROOT:-$(_dexjoco_run_root)}"
DEXJOCO_TASK="${DEXJOCO_TASK:-water_plant}"
DEXJOCO_PROMPT="${DEXJOCO_PROMPT:-Grasp the watering can and apply water to the plant.}"
SMOKE_STEPS="${SMOKE_STEPS:-5}"
OUT_DIR="$OUTPUT_DIR/dexjoco_headless_smoke"
mkdir -p "$OUT_DIR"

prepare_dexjoco_source
setup_dexjoco_env

cd "$DEXJOCO_DIR"
conda run --no-capture-output --prefix "$DEXJOCO_ENV_PREFIX" python - "$DEXJOCO_TASK" "$DEXJOCO_PROMPT" "$SMOKE_STEPS" "$OUT_DIR" <<'PY'
import json
import sys
from pathlib import Path

import numpy as np

from dexjoco_openpi_client.dexjoco_openpi_env import DexJoCoOpenPIEnv

task = sys.argv[1]
prompt = sys.argv[2]
steps = int(sys.argv[3])
out_dir = Path(sys.argv[4])

env = DexJoCoOpenPIEnv(
    env_name=task,
    camera_mapping={"base": "front", "wrist": "wrist"},
    seed=0,
    rand_full=False,
    randomize_dynamics=False,
    dual_arm=False,
    prompt=prompt,
    render_mode="rgb_array",
)

try:
    env.start()
    env.reset()
    obs = env.get_obs()
    assert obs["prompt"] == prompt
    assert obs["state"].shape == (23,)
    assert obs["base"].shape == (224, 224, 3)
    assert obs["wrist"].shape == (224, 224, 3)

    for _ in range(steps):
        env.stay()

    raw = env.get_raw_images()
    summary = {
        "task": task,
        "prompt": prompt,
        "state_shape": list(obs["state"].shape),
        "base_shape": list(obs["base"].shape),
        "wrist_shape": list(obs["wrist"].shape),
        "raw_image_keys": sorted(raw.keys()),
        "smoke_steps": steps,
        "done": env.is_done,
        "success": env.is_success,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
finally:
    env.close()
PY

echo "[job] DexJoCo headless smoke passed"

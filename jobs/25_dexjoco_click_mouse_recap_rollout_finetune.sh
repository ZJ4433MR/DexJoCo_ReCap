#!/usr/bin/env bash
set -euo pipefail

cd "$EXP_DIR"
source scripts/dexjoco_common.sh

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.99}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"

RUN_ROOT="${RUN_ROOT:-$(_dexjoco_run_root)}"
DEXJOCO_TASK="${DEXJOCO_TASK:-click_mouse}"
DEXJOCO_EVAL_SEED="${DEXJOCO_EVAL_SEED:-0}"
DEXJOCO_RECAP_EVAL_SEEDS="${DEXJOCO_RECAP_EVAL_SEEDS:-}"
DEXJOCO_COLLECT_SEED="${DEXJOCO_COLLECT_SEED:-$((DEXJOCO_EVAL_SEED + 10000))}"
DEXJOCO_COLLECT_EPISODES="${DEXJOCO_COLLECT_EPISODES:-20}"
DEXJOCO_EVAL_EPISODES="${DEXJOCO_EVAL_EPISODES:-20}"
DEXJOCO_PORT="${DEXJOCO_PORT:-$(dexjoco_default_port)}"
DEXJOCO_HOST="${DEXJOCO_HOST:-127.0.0.1}"
DEXJOCO_ACP_SUFFIX="${DEXJOCO_ACP_SUFFIX:- Use the high-advantage successful strategy.}"
DEXJOCO_RECAP_OUTPUT_NAME="${DEXJOCO_RECAP_OUTPUT_NAME:-dexjoco_click_mouse_recap_rollout_finetune}"
DEXJOCO_RECAP_DATA_PREFIX="${DEXJOCO_RECAP_DATA_PREFIX:-recap}"
DEXJOCO_RECAP_ROLLOUT_POLICY_DIR="${DEXJOCO_RECAP_ROLLOUT_POLICY_DIR:-../checkpoints/pi05_dexjoco_ckpt/$DEXJOCO_TASK}"
DEXJOCO_RECAP_PRETRAINED_MODEL_PATH="${DEXJOCO_RECAP_PRETRAINED_MODEL_PATH:-../checkpoints/pi05_dexjoco_ckpt/$DEXJOCO_TASK/params}"
DEXJOCO_RECAP_ROLLOUT_NPZ_INPUT="${DEXJOCO_RECAP_ROLLOUT_NPZ_INPUT:-}"
DEXJOCO_RECAP_POOL_INPUTS="${DEXJOCO_RECAP_POOL_INPUTS:-}"
DEXJOCO_RECAP_SKIP_EVAL="${DEXJOCO_RECAP_SKIP_EVAL:-0}"
DEXJOCO_RECAP_COLLECT_ONLY="${DEXJOCO_RECAP_COLLECT_ONLY:-0}"
DEXJOCO_RECAP_TRAIN_STEPS="${DEXJOCO_RECAP_TRAIN_STEPS:-500}"
DEXJOCO_RECAP_BATCH_SIZE="${DEXJOCO_RECAP_BATCH_SIZE:-2}"
DEXJOCO_RECAP_FSDP_DEVICES="${DEXJOCO_RECAP_FSDP_DEVICES:-2}"
DEXJOCO_RECAP_NUM_WORKERS="${DEXJOCO_RECAP_NUM_WORKERS:-0}"
DEXJOCO_RECAP_WARMUP_STEPS="${DEXJOCO_RECAP_WARMUP_STEPS:-50}"
DEXJOCO_RECAP_SAVE_INTERVAL="${DEXJOCO_RECAP_SAVE_INTERVAL:-250}"
DEXJOCO_RECAP_EXP_NAME="${DEXJOCO_RECAP_EXP_NAME:-recap_success_rollout_acp}"
DEXJOCO_RECAP_INCLUDE_FAILURES="${DEXJOCO_RECAP_INCLUDE_FAILURES:-0}"
DEXJOCO_RECAP_COLLECT_PROMPT_MODE="${DEXJOCO_RECAP_COLLECT_PROMPT_MODE:-acp}"
DEXJOCO_RECAP_COLLECT_SHARD_EPISODES="${DEXJOCO_RECAP_COLLECT_SHARD_EPISODES:-25}"
DEXJOCO_RECAP_LABEL_WITH_VALUE="${DEXJOCO_RECAP_LABEL_WITH_VALUE:-0}"
DEXJOCO_RECAP_LABEL_BACKEND="${DEXJOCO_RECAP_LABEL_BACKEND:-npz}"
DEXJOCO_RECAP_VALUE_EPOCHS="${DEXJOCO_RECAP_VALUE_EPOCHS:-8}"
DEXJOCO_RECAP_VALUE_MAX_STEPS="${DEXJOCO_RECAP_VALUE_MAX_STEPS:-0}"
DEXJOCO_RECAP_VALUE_BATCH_SIZE="${DEXJOCO_RECAP_VALUE_BATCH_SIZE:-64}"
DEXJOCO_RECAP_VALUE_EVAL_BATCH_SIZE="${DEXJOCO_RECAP_VALUE_EVAL_BATCH_SIZE:-128}"
DEXJOCO_RECAP_VALUE_IMAGE_SIZE="${DEXJOCO_RECAP_VALUE_IMAGE_SIZE:-96}"
DEXJOCO_RECAP_VALUE_LR="${DEXJOCO_RECAP_VALUE_LR:-0.0003}"
DEXJOCO_RECAP_VALUE_N_STEP="${DEXJOCO_RECAP_VALUE_N_STEP:-50}"
DEXJOCO_RECAP_VALUE_POSITIVE_RATIO="${DEXJOCO_RECAP_VALUE_POSITIVE_RATIO:-0.3}"
DEXJOCO_RECAP_VALUE_C_FAIL_COEF="${DEXJOCO_RECAP_VALUE_C_FAIL_COEF:-1.0}"
DEXJOCO_RECAP_VALUE_EXACT_TOP_K="${DEXJOCO_RECAP_VALUE_EXACT_TOP_K:-0}"
DEXJOCO_RECAP_VALUE_POSITIVE_SUCCESS_ONLY="${DEXJOCO_RECAP_VALUE_POSITIVE_SUCCESS_ONLY:-0}"
DEXJOCO_RECAP_VALUE_RANDOM_POSITIVE="${DEXJOCO_RECAP_VALUE_RANDOM_POSITIVE:-0}"
DEXJOCO_RECAP_POOL_MAX_FRAMES="${DEXJOCO_RECAP_POOL_MAX_FRAMES:-0}"
DEXJOCO_RECAP_POOL_MAX_EPISODES="${DEXJOCO_RECAP_POOL_MAX_EPISODES:-0}"
DEXJOCO_RECAP_POOL_KEEP_LAST_INPUT="${DEXJOCO_RECAP_POOL_KEEP_LAST_INPUT:-0}"
DEXJOCO_RECAP_EVAL_PROMPT_MODE="${DEXJOCO_RECAP_EVAL_PROMPT_MODE:-acp}"
DEXJOCO_RECAP_PISTAR06_REPO_ID="${DEXJOCO_RECAP_PISTAR06_REPO_ID:-local/dexjoco_${DEXJOCO_TASK}_${DEXJOCO_RECAP_DATA_PREFIX}}"
DEXJOCO_RECAP_PISTAR06_ROOT="${DEXJOCO_RECAP_PISTAR06_ROOT:-$RUN_ROOT/${DEXJOCO_RECAP_DATA_PREFIX}_lerobot}"
DEXJOCO_RECAP_PISTAR06_FPS="${DEXJOCO_RECAP_PISTAR06_FPS:-20}"
DEXJOCO_RECAP_PISTAR06_USE_VIDEOS="${DEXJOCO_RECAP_PISTAR06_USE_VIDEOS:-0}"
DEXJOCO_RECAP_PISTAR06_VALUE_STEPS="${DEXJOCO_RECAP_PISTAR06_VALUE_STEPS:-8000}"
DEXJOCO_RECAP_PISTAR06_VALUE_BATCH_SIZE="${DEXJOCO_RECAP_PISTAR06_VALUE_BATCH_SIZE:-16}"
DEXJOCO_RECAP_PISTAR06_VALUE_NUM_WORKERS="${DEXJOCO_RECAP_PISTAR06_VALUE_NUM_WORKERS:-2}"
DEXJOCO_RECAP_PISTAR06_VALUE_DTYPE="${DEXJOCO_RECAP_PISTAR06_VALUE_DTYPE:-bfloat16}"
DEXJOCO_RECAP_PISTAR06_VALUE_LANGUAGE_REPO="${DEXJOCO_RECAP_PISTAR06_VALUE_LANGUAGE_REPO:-google/gemma-3-270m}"
DEXJOCO_RECAP_PISTAR06_VALUE_VISION_REPO="${DEXJOCO_RECAP_PISTAR06_VALUE_VISION_REPO:-google/siglip-so400m-patch14-384}"
DEXJOCO_RECAP_PISTAR06_VALUE_CAMERA_FEATURES="${DEXJOCO_RECAP_PISTAR06_VALUE_CAMERA_FEATURES:-[observation.images.ego_right,observation.images.wrist]}"
DEXJOCO_RECAP_PISTAR06_VALUE_NORMALIZATION_MAPPING="${DEXJOCO_RECAP_PISTAR06_VALUE_NORMALIZATION_MAPPING:-{VISUAL: IDENTITY, STATE: QUANTILES, ACTION: IDENTITY}}"
DEXJOCO_RECAP_PISTAR06_VALUE_FIELD="${DEXJOCO_RECAP_PISTAR06_VALUE_FIELD:-complementary_info.value_pistar06}"
DEXJOCO_RECAP_PISTAR06_ADVANTAGE_FIELD="${DEXJOCO_RECAP_PISTAR06_ADVANTAGE_FIELD:-complementary_info.advantage_pistar06}"
DEXJOCO_RECAP_PISTAR06_INDICATOR_FIELD="${DEXJOCO_RECAP_PISTAR06_INDICATOR_FIELD:-complementary_info.acp_indicator_pistar06}"
DEXJOCO_RECAP_PISTAR06_PYDEPS_DIR="${DEXJOCO_RECAP_PISTAR06_PYDEPS_DIR:-$RUN_ROOT/evorl_value_pydeps}"
DEXJOCO_RECAP_PISTAR06_VALUE_CHECKPOINT_SOURCE_TAR="${DEXJOCO_RECAP_PISTAR06_VALUE_CHECKPOINT_SOURCE_TAR:-}"
DEXJOCO_RECAP_PISTAR06_SKIP_VALUE_TRAINING="${DEXJOCO_RECAP_PISTAR06_SKIP_VALUE_TRAINING:-0}"
OPENPI_RECAP_LORA_ONLY="${OPENPI_RECAP_LORA_ONLY:-1}"
OPENPI_RECAP_BASE_REPEAT="${OPENPI_RECAP_BASE_REPEAT:-0}"
OPENPI_RECAP_POSITIVE_REPEAT="${OPENPI_RECAP_POSITIVE_REPEAT:-1}"
OPENPI_RECAP_ACP_PROMPTS_FILE="${OPENPI_RECAP_ACP_PROMPTS_FILE:-}"
OPENPI_RECAP_PROMPT_MODE="${OPENPI_RECAP_PROMPT_MODE:-indicator}"

OUT_DIR="$OUTPUT_DIR/$DEXJOCO_RECAP_OUTPUT_NAME"
CONFIG_DIR="$OUT_DIR/configs"
COLLECTED_ROLLOUT_DATASET="$RUN_ROOT/${DEXJOCO_RECAP_DATA_PREFIX}_collected_rollouts.npz"
ROLLOUT_DATASET="$RUN_ROOT/${DEXJOCO_RECAP_DATA_PREFIX}_success_rollouts.npz"
LABELED_ROLLOUT_DATASET="$RUN_ROOT/${DEXJOCO_RECAP_DATA_PREFIX}_value_advantage_rollouts.npz"
SUMMARY="$OUT_DIR/summary.tsv"
mkdir -p "$OUT_DIR" "$CONFIG_DIR"

prepare_dexjoco_source
setup_dexjoco_env
setup_openpi_env
relax_openpi_websocket_timeouts

base_config_rel="./configs/rand_obj/${DEXJOCO_TASK}.yaml"
base_config="$DEXJOCO_DIR/$base_config_rel"
acp_eval_config="$CONFIG_DIR/${DEXJOCO_TASK}_acp_positive.yaml"

if [[ ! -f "$base_config" ]]; then
  echo "[job] missing eval config: $base_config_rel" >&2
  exit 2
fi

conda run --no-capture-output --prefix "$DEXJOCO_ENV_PREFIX" python - "$base_config" "$acp_eval_config" "$DEXJOCO_ACP_SUFFIX" <<'PY'
from pathlib import Path
import sys
import yaml

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
suffix = sys.argv[3]
cfg = yaml.safe_load(src.read_text())
cfg["prompt"] = cfg["prompt"].rstrip() + suffix
dst.write_text(yaml.safe_dump(cfg, sort_keys=False))
print(f"[job] ACP eval prompt: {cfg['prompt']}")
PY

DEXJOCO_TASK="$DEXJOCO_TASK" download_dexjoco_pi05_checkpoint

patch_openpi_for_recap_rollout_dataset() {
  conda run --no-capture-output --prefix "$OPENPI_ENV_PREFIX" python - "$DEXJOCO_DIR/openpi" <<'PY'
from pathlib import Path
import textwrap
import sys

openpi_dir = Path(sys.argv[1])
data_loader = openpi_dir / "src/openpi/training/data_loader.py"
dexjoco_configs = openpi_dir / "src/openpi/training/dexjoco_configs.py"
pi0_config = openpi_dir / "src/openpi/models/pi0_config.py"

text = data_loader.read_text()
if "\nimport os\n" not in text:
    if "import logging\n" in text:
        text = text.replace("import logging\n", "import logging\nimport os\n", 1)
    else:
        text = "import os\n" + text
if "\nimport numpy as np\n" not in text:
    if "import os\n" in text:
        text = text.replace("import os\n", "import os\nimport numpy as np\n", 1)
    else:
        text = "import numpy as np\n" + text
if "class RecapRolloutDataset" not in text:
    marker = "\n\nclass FakeDataset(Dataset):"
    insert = r'''

class RecapRolloutDataset(Dataset):
    """Small NPZ dataset built from DexJoCo policy rollouts."""

    def __init__(self, path: str, action_horizon: int):
        self._path = path
        data = np.load(path, allow_pickle=False)
        self._base = data["base"]
        self._wrist = data["wrist"]
        self._state = data["state"]
        self._action = data["action"]
        self._episode_id = data["episode_id"]
        self._is_success = data["is_success"] if "is_success" in data else np.ones(len(self._action), dtype=np.bool_)
        indicator_field = os.environ.get("OPENPI_RECAP_INDICATOR_FIELD", "acp_indicator")
        if indicator_field in data:
            raw_indicator = data[indicator_field]
        elif "acp_indicator" in data:
            raw_indicator = data["acp_indicator"]
        else:
            raw_indicator = self._is_success.astype(np.int64)
        self._acp_indicator = np.asarray(raw_indicator).reshape(-1).astype(np.int64)
        if len(self._acp_indicator) != len(self._action):
            raise ValueError(
                f"ACP indicator length {len(self._acp_indicator)} does not match actions {len(self._action)}"
            )
        self._base_prompt = str(data["base_prompt"]) if "base_prompt" in data else str(data["prompt"])
        self._acp_prompt = str(data["acp_prompt"]) if "acp_prompt" in data else str(data["prompt"])
        self._acp_prompts = self._load_acp_prompts()
        self._prompt_mode = os.environ.get("OPENPI_RECAP_PROMPT_MODE", "indicator")
        if self._prompt_mode not in {"indicator", "base", "acp"}:
            raise ValueError(
                "OPENPI_RECAP_PROMPT_MODE must be one of {'indicator', 'base', 'acp'}, "
                f"got {self._prompt_mode!r}"
            )
        self._action_horizon = action_horizon
        self._episode_end = {}
        for episode_id in np.unique(self._episode_id):
            indices = np.flatnonzero(self._episode_id == episode_id)
            self._episode_end[int(episode_id)] = int(indices[-1]) + 1
        base_repeat = max(0, int(os.environ.get("OPENPI_RECAP_BASE_REPEAT", "0")))
        positive_repeat = max(1, int(os.environ.get("OPENPI_RECAP_POSITIVE_REPEAT", "1")))
        source_indices = []
        prompt_variant_ids = []
        positive_prompt_cursor = 0

        def add_virtual_index(idx: int, is_positive: bool, force_base: bool = False) -> None:
            nonlocal positive_prompt_cursor
            source_indices.append(idx)
            if force_base or self._prompt_mode == "base":
                prompt_variant_ids.append(-1)
            elif self._prompt_mode == "acp" or is_positive:
                prompt_variant_ids.append(positive_prompt_cursor % len(self._acp_prompts))
                positive_prompt_cursor += 1
            else:
                prompt_variant_ids.append(-1)

        for idx in range(len(self._action)):
            is_positive = bool(self._acp_indicator[idx])
            add_virtual_index(idx, is_positive=is_positive)
            for _ in range(base_repeat):
                add_virtual_index(idx, is_positive=is_positive, force_base=True)
            if is_positive:
                for _ in range(positive_repeat - 1):
                    add_virtual_index(idx, is_positive=True)
        self._source_indices = np.asarray(source_indices, dtype=np.int64)
        self._prompt_variant_ids = np.asarray(prompt_variant_ids, dtype=np.int16)

    def _load_acp_prompts(self) -> list[str]:
        prompts = []
        manifest_path = os.environ.get("OPENPI_RECAP_ACP_PROMPTS_FILE", "")
        if manifest_path:
            try:
                import json as _json

                with open(manifest_path, encoding="utf-8") as stream:
                    manifest = _json.load(stream)
                for candidate in manifest.get("candidates", []):
                    prompt = str(candidate.get("prompt", "")).strip()
                    suffix = str(candidate.get("suffix", "")).strip()
                    if prompt:
                        prompts.append(prompt)
                    elif suffix:
                        prompts.append(f"{self._base_prompt.rstrip()} {suffix}")
            except Exception as exc:
                raise RuntimeError(f"Failed to load ReCap prompt manifest {manifest_path}: {exc}") from exc
        if not prompts:
            prompts = [self._acp_prompt]
        deduped = []
        seen = set()
        for prompt in prompts:
            prompt = str(prompt).strip()
            if not prompt or prompt in seen:
                continue
            deduped.append(prompt)
            seen.add(prompt)
        if not deduped:
            deduped = [self._acp_prompt]
        return deduped

    def __getitem__(self, index: SupportsIndex) -> dict:
        virtual_idx = index.__index__()
        idx = int(self._source_indices[virtual_idx])
        prompt_variant_id = int(self._prompt_variant_ids[virtual_idx])
        prompt = self._base_prompt if prompt_variant_id < 0 else self._acp_prompts[prompt_variant_id]
        episode_id = int(self._episode_id[idx])
        episode_end = self._episode_end[episode_id]
        action_end = min(idx + self._action_horizon, episode_end)
        actions = self._action[idx:action_end]
        if len(actions) < self._action_horizon:
            pad = np.repeat(actions[-1:], self._action_horizon - len(actions), axis=0)
            actions = np.concatenate([actions, pad], axis=0)
        return {
            "observation.images.ego_right": self._base[idx],
            "observation.images.wrist": self._wrist[idx],
            "observation.state": self._state[idx],
            "action": actions.astype(np.float32, copy=False),
            "prompt": prompt,
            "task_index": np.asarray(0, dtype=np.int64),
            "task": prompt,
        }

    def __len__(self) -> int:
        return len(self._source_indices)
'''
    text = text.replace(marker, insert + marker)

old = '''    root = data_config.root

    dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(repo_id, root=root)
'''
new = '''    root = data_config.root

    rollout_npz = os.environ.get("OPENPI_RECAP_ROLLOUT_NPZ")
    if rollout_npz:
        logging.info("Using ReCap rollout dataset: %s", rollout_npz)
        return RecapRolloutDataset(rollout_npz, action_horizon)

    dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(repo_id, root=root)
'''
if new not in text:
    text = text.replace(old, new)
data_loader.write_text(text)

cfg_text = dexjoco_configs.read_text()
if "import os" not in cfg_text.splitlines()[:5]:
    cfg_text = cfg_text.replace("from datetime import datetime\n", "from datetime import datetime\nimport os\n")
cfg_text = cfg_text.replace(
    "warmup_steps=10_000,",
    'warmup_steps=int(os.environ.get("DEXJOCO_RECAP_WARMUP_STEPS", "10000")),',
)
dexjoco_configs.write_text(cfg_text)

pi0_text = pi0_config.read_text()
if "import os" not in pi0_text.splitlines()[:8]:
    pi0_text = pi0_text.replace("import dataclasses\n", "import dataclasses\nimport os\n")
lora_only_patch = '''    def get_freeze_filter(self) -> nnx.filterlib.Filter:
        """Returns the freeze filter based on the model config."""
        if os.environ.get("OPENPI_RECAP_LORA_ONLY", "0") == "1":
            return nnx.Not(nnx_utils.PathRegex(".*lora.*"))
'''
if "OPENPI_RECAP_LORA_ONLY" not in pi0_text:
    pi0_text = pi0_text.replace(
        '''    def get_freeze_filter(self) -> nnx.filterlib.Filter:
        """Returns the freeze filter based on the model config."""
''',
        lora_only_patch,
    )
pi0_config.write_text(pi0_text)
print("[job] patched OpenPI data loader and LoRA-only ReCap train filter")
PY
}

patch_openpi_config_yaml() {
  conda run --no-capture-output --prefix "$OPENPI_ENV_PREFIX" python - "$DEXJOCO_DIR/openpi/config.yaml" "$DEXJOCO_TASK" "$DEXJOCO_RECAP_BATCH_SIZE" "$DEXJOCO_RECAP_TRAIN_STEPS" "$DEXJOCO_RECAP_PRETRAINED_MODEL_PATH" <<'PY'
from pathlib import Path
import sys
import yaml

path = Path(sys.argv[1])
task = sys.argv[2]
batch_size = int(sys.argv[3])
steps = int(sys.argv[4])
pretrained_model_path = sys.argv[5]
cfg = yaml.safe_load(path.read_text())
cfg["pretrained_model_path"] = pretrained_model_path
cfg["ckpts_root"] = "../checkpoints/recap_acp_ckpts"
cfg["wandb_enabled"] = False
cfg["batch_size"] = batch_size
cfg["single_arm_steps"] = steps
path.write_text(yaml.safe_dump(cfg, sort_keys=False))
print("[job] patched openpi/config.yaml")
print(path.read_text())
PY
}

SERVER_PIDS=()
server_group_alive() {
  local pid="$1"
  [[ -n "$pid" ]] && { kill -0 "$pid" >/dev/null 2>&1 || pgrep -g "$pid" >/dev/null 2>&1; }
}

kill_server_group() {
  local pid="$1"
  [[ -z "$pid" ]] && return 0
  kill -- "-$pid" >/dev/null 2>&1 || kill "$pid" >/dev/null 2>&1 || true
  for _ in {1..20}; do
    if ! server_group_alive "$pid"; then
      wait "$pid" >/dev/null 2>&1 || true
      return 0
    fi
    sleep 0.5
  done
  kill -9 -- "-$pid" >/dev/null 2>&1 || kill -9 "$pid" >/dev/null 2>&1 || true
  wait "$pid" >/dev/null 2>&1 || true
}

cleanup_servers() {
  for pid in "${SERVER_PIDS[@]:-}"; do
    kill_server_group "$pid"
  done
  pkill -f "serve_policy.py --port=${DEXJOCO_PORT}" >/dev/null 2>&1 || true
}
trap cleanup_servers EXIT

start_policy_server() {
  local policy_dir="$1"
  local log_path="$2"
  cd "$DEXJOCO_DIR/openpi"
  setsid conda run --no-capture-output --prefix "$OPENPI_ENV_PREFIX" python scripts/serve_policy.py \
    --port="$DEXJOCO_PORT" \
    policy:checkpoint \
    --policy.config="$DEXJOCO_TASK" \
    --policy.dir="$policy_dir" \
    > "$log_path" 2>&1 &
  local pid=$!
  SERVER_PIDS+=("$pid")
  local start_ts
  start_ts="$(date +%s)"
  while true; do
    if [[ -f "$log_path" ]] && grep -q "server listening on" "$log_path"; then
      break
    fi
    if ! server_group_alive "$pid"; then
      echo "[job] server exited before becoming ready: $policy_dir" >&2
      tail -200 "$log_path" >&2 || true
      exit 1
    fi
    if (( $(date +%s) - start_ts >= 900 )); then
      echo "[job] server did not become ready: $policy_dir" >&2
      tail -200 "$log_path" >&2 || true
      exit 1
    fi
    sleep 5
  done
  POLICY_SERVER_PID="$pid"
}

stop_policy_server() {
  local pid="$1"
  kill_server_group "$pid"
}

ensure_evorl_pistar06_deps() {
  local target="$DEXJOCO_RECAP_PISTAR06_PYDEPS_DIR"
  mkdir -p "$target"
  echo "[job] ensuring Evo-RL pistar06 dependencies in $target"
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
  echo "[job] verifying Evo-RL pistar06 runtime imports"
  PYTHONPATH="$pythonpath" python - "$DEXJOCO_RECAP_PISTAR06_PYDEPS_DIR" <<'PY'
from __future__ import annotations

import sys
from pathlib import Path

target = Path(sys.argv[1])
print(f"[job] pistar06 pydeps target: {target}")
print(f"[job] python executable: {sys.executable}")

try:
    import datasets  # noqa: F401
    from scipy.signal import savgol_filter  # noqa: F401
    from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: F401
except Exception as exc:
    print("[job] failed to import Evo-RL pistar06 runtime dependencies", file=sys.stderr)
    print(f"[job] sys.path head: {sys.path[:8]}", file=sys.stderr)
    raise

print("[job] Evo-RL pistar06 runtime imports ok")
PY
}

restore_pistar06_value_checkpoint() {
  local source_tar="$1"
  local checkpoint_model="$2/checkpoints/last/pretrained_model/model.safetensors"

  if [[ -f "$checkpoint_model" ]]; then
    echo "[job] using existing Evo-RL pistar06 value checkpoint: $checkpoint_model"
    return 0
  fi

  if [[ -z "$source_tar" ]]; then
    return 1
  fi
  if [[ ! -f "$source_tar" ]]; then
    echo "[job] requested value checkpoint source tar is missing: $source_tar" >&2
    return 1
  fi

  local output_parent
  output_parent="$(dirname "$OUTPUT_DIR")"
  local member="./outputs/${DEXJOCO_RECAP_OUTPUT_NAME}/pistar06_value"
  echo "[job] restoring Evo-RL pistar06 value checkpoint from $source_tar"
  mkdir -p "$output_parent"
  if ! tar -xzf "$source_tar" -C "$output_parent" "$member"; then
    echo "[job] primary checkpoint member not found, retrying without leading ./" >&2
    tar -xzf "$source_tar" -C "$output_parent" "${member#./}"
  fi

  if [[ ! -f "$checkpoint_model" ]]; then
    echo "[job] restored value checkpoint is missing model.safetensors: $checkpoint_model" >&2
    return 1
  fi
  echo "[job] restored Evo-RL pistar06 value checkpoint: $checkpoint_model"
}

if [[ -n "$DEXJOCO_RECAP_ROLLOUT_NPZ_INPUT" ]]; then
  if [[ ! -f "$DEXJOCO_RECAP_ROLLOUT_NPZ_INPUT" ]]; then
    echo "[job] frozen rollout input not found: $DEXJOCO_RECAP_ROLLOUT_NPZ_INPUT" >&2
    exit 2
  fi
  echo "[job] using frozen rollout NPZ input: $DEXJOCO_RECAP_ROLLOUT_NPZ_INPUT"
  if [[ "$DEXJOCO_RECAP_ROLLOUT_NPZ_INPUT" != "$COLLECTED_ROLLOUT_DATASET" ]]; then
    cp "$DEXJOCO_RECAP_ROLLOUT_NPZ_INPUT" "$COLLECTED_ROLLOUT_DATASET"
  fi
  python - "$COLLECTED_ROLLOUT_DATASET" "$OUT_DIR/recap_success_rollouts.summary.txt" <<'PY'
from pathlib import Path
import sys
import numpy as np

path = Path(sys.argv[1])
summary = Path(sys.argv[2])
with np.load(path, allow_pickle=False) as data:
    episode_ids = data["episode_id"].astype(np.int64)
    is_success = data["is_success"].astype(np.bool_)
    episodes = sorted(np.unique(episode_ids).tolist())
    successes = 0
    for episode in episodes:
        idx = int(np.flatnonzero(episode_ids == episode)[0])
        successes += int(bool(is_success[idx]))
    frames = int(data["action"].shape[0])
summary.write_text(
    "\n".join(
        [
            f"input={path}",
            f"frames={frames}",
            f"episodes={len(episodes)}",
            f"successes={successes}",
            f"success_rate={successes / max(1, len(episodes)):.6f}",
        ]
    )
    + "\n",
    encoding="utf-8",
)
print(summary.read_text(encoding="utf-8"))
PY
else
  echo "[job] starting policy server for rollout collection: $DEXJOCO_RECAP_ROLLOUT_POLICY_DIR"
  start_policy_server "$DEXJOCO_RECAP_ROLLOUT_POLICY_DIR" "$OUT_DIR/public_server.log"
  public_server_pid="$POLICY_SERVER_PID"

  cd "$DEXJOCO_DIR"
  collect_args=(
    --config="$base_config" \
    --output="$COLLECTED_ROLLOUT_DATASET" \
    --host="$DEXJOCO_HOST" \
    --port="$DEXJOCO_PORT" \
    --seed="$DEXJOCO_COLLECT_SEED" \
    --episodes="$DEXJOCO_COLLECT_EPISODES" \
    --acp-suffix="$DEXJOCO_ACP_SUFFIX" \
    --collect-prompt-mode="$DEXJOCO_RECAP_COLLECT_PROMPT_MODE"
  )
  if [[ "$DEXJOCO_RECAP_INCLUDE_FAILURES" == "1" ]]; then
    collect_args+=(--include-failures)
  fi
  if (( DEXJOCO_RECAP_COLLECT_SHARD_EPISODES > 0 )); then
    collect_args+=(
      --shard-episodes="$DEXJOCO_RECAP_COLLECT_SHARD_EPISODES"
      --shard-dir="$RUN_ROOT/${DEXJOCO_RECAP_DATA_PREFIX}_collect_shards"
    )
  fi
  conda run --no-capture-output --prefix "$DEXJOCO_ENV_PREFIX" python "$EXP_DIR/scripts/dexjoco_collect_success_rollouts.py" "${collect_args[@]}"
  if [[ -f "${COLLECTED_ROLLOUT_DATASET%.npz}.shards.txt" ]]; then
    mapfile -t collected_shards < "${COLLECTED_ROLLOUT_DATASET%.npz}.shards.txt"
    echo "[job] merging collected rollout shards: ${#collected_shards[@]} -> $COLLECTED_ROLLOUT_DATASET"
    python "$EXP_DIR/scripts/dexjoco_merge_rollout_npz.py" \
      --output "$COLLECTED_ROLLOUT_DATASET" \
      --summary-output "${COLLECTED_ROLLOUT_DATASET%.npz}.merge_summary.json" \
      "${collected_shards[@]}"
  fi
  cp "${COLLECTED_ROLLOUT_DATASET%.npz}.summary.txt" "$OUT_DIR/recap_success_rollouts.summary.txt"
  stop_policy_server "$public_server_pid"
fi

if [[ -n "$DEXJOCO_RECAP_POOL_INPUTS" ]]; then
  read -r -a pool_inputs <<< "$DEXJOCO_RECAP_POOL_INPUTS"
  echo "[job] merging ReCap data pool inputs: ${pool_inputs[*]} + $COLLECTED_ROLLOUT_DATASET"
  python "$EXP_DIR/scripts/dexjoco_merge_rollout_npz.py" \
    --output "$ROLLOUT_DATASET" \
    --summary-output "$OUT_DIR/recap_data_pool.summary.json" \
    "${pool_inputs[@]}" \
    "$COLLECTED_ROLLOUT_DATASET"
else
  cp "$COLLECTED_ROLLOUT_DATASET" "$ROLLOUT_DATASET"
fi

if [[ "$DEXJOCO_RECAP_COLLECT_ONLY" == "1" ]]; then
  mkdir -p "$OUT_DIR/rollouts"
  cp "$COLLECTED_ROLLOUT_DATASET" "$OUT_DIR/rollouts/collected_rollouts.npz"
  cp "$ROLLOUT_DATASET" "$OUT_DIR/rollouts/policy_rollouts.npz"
  python - "$OUT_DIR/collect_manifest.json" "$COLLECTED_ROLLOUT_DATASET" "$ROLLOUT_DATASET" <<'PY'
from pathlib import Path
import hashlib
import json
import os
import sys

def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

collected = Path(sys.argv[2])
policy = Path(sys.argv[3])
manifest = {
    "collected_rollout_dataset": str(collected),
    "collected_rollout_sha256": digest(collected),
    "exported_collected_rollout": "rollouts/collected_rollouts.npz",
    "exported_policy_rollout": "rollouts/policy_rollouts.npz",
    "policy_rollout_dataset": str(policy),
    "policy_rollout_sha256": digest(policy),
    "DEXJOCO_COLLECT_SEED": os.environ.get("DEXJOCO_COLLECT_SEED"),
    "DEXJOCO_COLLECT_EPISODES": os.environ.get("DEXJOCO_COLLECT_EPISODES"),
    "DEXJOCO_RECAP_INCLUDE_FAILURES": os.environ.get("DEXJOCO_RECAP_INCLUDE_FAILURES"),
    "DEXJOCO_RECAP_COLLECT_PROMPT_MODE": os.environ.get("DEXJOCO_RECAP_COLLECT_PROMPT_MODE"),
    "DEXJOCO_RECAP_COLLECT_SHARD_EPISODES": os.environ.get("DEXJOCO_RECAP_COLLECT_SHARD_EPISODES"),
}
Path(sys.argv[1]).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(manifest, sort_keys=True))
PY
  echo "[job] collect-only requested; stopping before value labeling and policy training"
  exit 0
fi

POLICY_ROLLOUT_DATASET="$ROLLOUT_DATASET"
if [[ "$DEXJOCO_RECAP_LABEL_WITH_VALUE" == "1" ]]; then
  echo "[job] ReCap pool label bounds max_frames=$DEXJOCO_RECAP_POOL_MAX_FRAMES max_episodes=$DEXJOCO_RECAP_POOL_MAX_EPISODES keep_last_input=$DEXJOCO_RECAP_POOL_KEEP_LAST_INPUT"
  if [[ "$DEXJOCO_RECAP_POOL_MAX_FRAMES" -gt 0 || "$DEXJOCO_RECAP_POOL_MAX_EPISODES" -gt 0 ]]; then
    SAMPLED_ROLLOUT_DATASET="$RUN_ROOT/${DEXJOCO_RECAP_DATA_PREFIX}_sampled_rollouts.npz"
    sample_args=(
      --input "$ROLLOUT_DATASET"
      --output "$SAMPLED_ROLLOUT_DATASET"
      --summary-output "$OUT_DIR/recap_sampled_pool.summary.json"
      --max-frames "$DEXJOCO_RECAP_POOL_MAX_FRAMES"
      --max-episodes "$DEXJOCO_RECAP_POOL_MAX_EPISODES"
      --seed "$DEXJOCO_COLLECT_SEED"
    )
    if [[ "$DEXJOCO_RECAP_POOL_KEEP_LAST_INPUT" == "1" ]]; then
      sample_args+=(--keep-last-source)
    fi
    echo "[job] sampling ReCap pool for value labeling max_frames=$DEXJOCO_RECAP_POOL_MAX_FRAMES max_episodes=$DEXJOCO_RECAP_POOL_MAX_EPISODES keep_last_input=$DEXJOCO_RECAP_POOL_KEEP_LAST_INPUT"
    python "$EXP_DIR/scripts/dexjoco_sample_rollout_npz.py" "${sample_args[@]}"
    POLICY_ROLLOUT_DATASET="$SAMPLED_ROLLOUT_DATASET"
  fi
  if [[ "$POLICY_ROLLOUT_DATASET" == "$ROLLOUT_DATASET" && ( "$DEXJOCO_RECAP_POOL_MAX_FRAMES" -gt 0 || "$DEXJOCO_RECAP_POOL_MAX_EPISODES" -gt 0 ) ]]; then
    echo "[job] expected sampled value-labeling pool, but POLICY_ROLLOUT_DATASET was not changed" >&2
    exit 1
  fi
  if [[ "$DEXJOCO_RECAP_LABEL_BACKEND" == "npz" ]]; then
    value_label_args=(
      --input "$POLICY_ROLLOUT_DATASET"
      --output "$LABELED_ROLLOUT_DATASET"
      --model-output "$OUT_DIR/recap_value_model.pt"
      --summary-output "$OUT_DIR/recap_value_advantage.summary.json"
      --seed "$DEXJOCO_COLLECT_SEED"
      --epochs "$DEXJOCO_RECAP_VALUE_EPOCHS"
      --max-steps "$DEXJOCO_RECAP_VALUE_MAX_STEPS"
      --batch-size "$DEXJOCO_RECAP_VALUE_BATCH_SIZE"
      --eval-batch-size "$DEXJOCO_RECAP_VALUE_EVAL_BATCH_SIZE"
      --image-size "$DEXJOCO_RECAP_VALUE_IMAGE_SIZE"
      --lr "$DEXJOCO_RECAP_VALUE_LR"
      --n-step "$DEXJOCO_RECAP_VALUE_N_STEP"
      --positive-ratio "$DEXJOCO_RECAP_VALUE_POSITIVE_RATIO"
      --c-fail-coef "$DEXJOCO_RECAP_VALUE_C_FAIL_COEF"
    )
    if [[ "$DEXJOCO_RECAP_VALUE_EXACT_TOP_K" == "1" ]]; then
      value_label_args+=(--exact-top-k)
    fi
    if [[ "$DEXJOCO_RECAP_VALUE_POSITIVE_SUCCESS_ONLY" == "1" ]]; then
      value_label_args+=(--positive-success-only)
    fi
    if [[ "$DEXJOCO_RECAP_VALUE_RANDOM_POSITIVE" == "1" ]]; then
      value_label_args+=(--random-positive)
    fi
    echo "[job] training lightweight NPZ value model and labeling ReCap advantages"
    python "$EXP_DIR/scripts/dexjoco_label_recap_rollouts.py" "${value_label_args[@]}"
  elif [[ "$DEXJOCO_RECAP_LABEL_BACKEND" == "pistar06" ]]; then
    LEROBOT_VALUE_DIR="$OUT_DIR/pistar06_value"
    LEROBOT_INFER_DIR="$OUT_DIR/pistar06_value_infer"
    ensure_evorl_pistar06_deps
    EVORL_VALUE_PYTHONPATH="$DEXJOCO_RECAP_PISTAR06_PYDEPS_DIR:$EVORL_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
    verify_evorl_pistar06_runtime "$EVORL_VALUE_PYTHONPATH"
    convert_args=(
      --input "$POLICY_ROLLOUT_DATASET"
      --output-root "$DEXJOCO_RECAP_PISTAR06_ROOT"
      --repo-id "$DEXJOCO_RECAP_PISTAR06_REPO_ID"
      --task "$DEXJOCO_TASK"
      --fps "$DEXJOCO_RECAP_PISTAR06_FPS"
      --overwrite
      --summary-output "$OUT_DIR/pistar06_lerobot_dataset.summary.json"
    )
    if [[ "$DEXJOCO_RECAP_PISTAR06_USE_VIDEOS" == "1" ]]; then
      convert_args+=(--use-videos)
    fi
    echo "[job] converting DexJoCo rollout NPZ to LeRobot dataset root=$DEXJOCO_RECAP_PISTAR06_ROOT repo=$DEXJOCO_RECAP_PISTAR06_REPO_ID"
    PYTHONPATH="$EVORL_VALUE_PYTHONPATH" python "$EXP_DIR/scripts/dexjoco_npz_to_lerobot.py" "${convert_args[@]}"

    if restore_pistar06_value_checkpoint "$DEXJOCO_RECAP_PISTAR06_VALUE_CHECKPOINT_SOURCE_TAR" "$LEROBOT_VALUE_DIR"; then
      echo "[job] skipping Evo-RL pistar06 value training; restored checkpoint will be used for inference"
    elif [[ "$DEXJOCO_RECAP_PISTAR06_SKIP_VALUE_TRAINING" == "1" ]]; then
      echo "[job] DEXJOCO_RECAP_PISTAR06_SKIP_VALUE_TRAINING=1 but no reusable checkpoint was found" >&2
      exit 2
    else
      echo "[job] training Evo-RL pistar06 value model steps=$DEXJOCO_RECAP_PISTAR06_VALUE_STEPS batch=$DEXJOCO_RECAP_PISTAR06_VALUE_BATCH_SIZE"
      cd "$EVORL_DIR"
      PYTHONPATH="$EVORL_VALUE_PYTHONPATH" python -m lerobot.scripts.lerobot_value_train \
        --dataset.repo_id="$DEXJOCO_RECAP_PISTAR06_REPO_ID" \
        --dataset.root="$DEXJOCO_RECAP_PISTAR06_ROOT" \
        --value.type=pistar06 \
        --value.dtype="$DEXJOCO_RECAP_PISTAR06_VALUE_DTYPE" \
        --value.vision_repo_id="$DEXJOCO_RECAP_PISTAR06_VALUE_VISION_REPO" \
        --value.language_repo_id="$DEXJOCO_RECAP_PISTAR06_VALUE_LANGUAGE_REPO" \
        --value.normalization_mapping="$DEXJOCO_RECAP_PISTAR06_VALUE_NORMALIZATION_MAPPING" \
        --value.camera_features="$DEXJOCO_RECAP_PISTAR06_VALUE_CAMERA_FEATURES" \
        --value.device=cuda \
        --value.push_to_hub=false \
        --batch_size="$DEXJOCO_RECAP_PISTAR06_VALUE_BATCH_SIZE" \
        --num_workers="$DEXJOCO_RECAP_PISTAR06_VALUE_NUM_WORKERS" \
        --steps="$DEXJOCO_RECAP_PISTAR06_VALUE_STEPS" \
        --save_checkpoint=true \
        --save_freq="$DEXJOCO_RECAP_PISTAR06_VALUE_STEPS" \
        --wandb.enable=false \
        --output_dir="$LEROBOT_VALUE_DIR" \
        --job_name="pistar06_value_${DEXJOCO_RECAP_DATA_PREFIX}"
    fi

    echo "[job] inferring Evo-RL pistar06 value/advantage/ACP annotations"
    cd "$EVORL_DIR"
    PYTHONPATH="$EVORL_VALUE_PYTHONPATH" python -m lerobot.scripts.lerobot_value_infer \
      --dataset.repo_id="$DEXJOCO_RECAP_PISTAR06_REPO_ID" \
      --dataset.root="$DEXJOCO_RECAP_PISTAR06_ROOT" \
      --dataset.default_success=failure \
      --inference.checkpoint_path="$LEROBOT_VALUE_DIR" \
      --runtime.device=cuda \
      --runtime.batch_size="$DEXJOCO_RECAP_PISTAR06_VALUE_BATCH_SIZE" \
      --runtime.num_workers="$DEXJOCO_RECAP_PISTAR06_VALUE_NUM_WORKERS" \
      --acp.enable=true \
      --acp.n_step="$DEXJOCO_RECAP_VALUE_N_STEP" \
      --acp.positive_ratio="$DEXJOCO_RECAP_VALUE_POSITIVE_RATIO" \
      --acp.c_fail_coef="$DEXJOCO_RECAP_VALUE_C_FAIL_COEF" \
      --acp.value_field="$DEXJOCO_RECAP_PISTAR06_VALUE_FIELD" \
      --acp.advantage_field="$DEXJOCO_RECAP_PISTAR06_ADVANTAGE_FIELD" \
      --acp.indicator_field="$DEXJOCO_RECAP_PISTAR06_INDICATOR_FIELD" \
      --viz.enable=false \
      --output_dir="$LEROBOT_INFER_DIR" \
      --job_name="pistar06_infer_${DEXJOCO_RECAP_DATA_PREFIX}"

    cd "$EXP_DIR"
    PYTHONPATH="$EVORL_VALUE_PYTHONPATH" python "$EXP_DIR/scripts/dexjoco_lerobot_annotations_to_npz.py" \
      --input "$POLICY_ROLLOUT_DATASET" \
      --output "$LABELED_ROLLOUT_DATASET" \
      --root "$DEXJOCO_RECAP_PISTAR06_ROOT" \
      --repo-id "$DEXJOCO_RECAP_PISTAR06_REPO_ID" \
      --value-field "$DEXJOCO_RECAP_PISTAR06_VALUE_FIELD" \
      --advantage-field "$DEXJOCO_RECAP_PISTAR06_ADVANTAGE_FIELD" \
      --indicator-field "$DEXJOCO_RECAP_PISTAR06_INDICATOR_FIELD" \
      --summary-output "$OUT_DIR/recap_pistar06_value_advantage.summary.json"
  else
    echo "[job] unsupported DEXJOCO_RECAP_LABEL_BACKEND=$DEXJOCO_RECAP_LABEL_BACKEND" >&2
    exit 2
  fi
  POLICY_ROLLOUT_DATASET="$LABELED_ROLLOUT_DATASET"
fi

patch_openpi_for_recap_rollout_dataset
patch_openpi_config_yaml

export OPENPI_RECAP_ROLLOUT_NPZ="$POLICY_ROLLOUT_DATASET"
export OPENPI_RECAP_INDICATOR_FIELD="${OPENPI_RECAP_INDICATOR_FIELD:-acp_indicator}"
export DEXJOCO_RECAP_WARMUP_STEPS
export OPENPI_RECAP_LORA_ONLY
export OPENPI_RECAP_BASE_REPEAT
export OPENPI_RECAP_POSITIVE_REPEAT
export OPENPI_RECAP_ACP_PROMPTS_FILE
export OPENPI_RECAP_PROMPT_MODE

python - "$OUT_DIR/run_manifest.json" "$POLICY_ROLLOUT_DATASET" <<'PY'
from pathlib import Path
import hashlib
import json
import os
import sys

manifest_path = Path(sys.argv[1])
rollout_path = Path(sys.argv[2])

def sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

keys = [
    "DEXJOCO_TASK",
    "DEXJOCO_COLLECT_SEED",
    "DEXJOCO_EVAL_SEED",
    "DEXJOCO_RECAP_EVAL_SEEDS",
    "DEXJOCO_COLLECT_EPISODES",
    "DEXJOCO_EVAL_EPISODES",
    "DEXJOCO_RECAP_ROLLOUT_NPZ_INPUT",
    "DEXJOCO_RECAP_INCLUDE_FAILURES",
    "DEXJOCO_RECAP_COLLECT_PROMPT_MODE",
    "DEXJOCO_RECAP_LABEL_WITH_VALUE",
    "DEXJOCO_RECAP_LABEL_BACKEND",
    "DEXJOCO_RECAP_VALUE_N_STEP",
    "DEXJOCO_RECAP_VALUE_POSITIVE_RATIO",
    "DEXJOCO_RECAP_VALUE_C_FAIL_COEF",
    "DEXJOCO_RECAP_TRAIN_STEPS",
    "DEXJOCO_RECAP_EVAL_PROMPT_MODE",
    "DEXJOCO_ACP_SUFFIX",
    "OPENPI_RECAP_PROMPT_MODE",
    "OPENPI_RECAP_BASE_REPEAT",
    "OPENPI_RECAP_POSITIVE_REPEAT",
    "OPENPI_RECAP_INDICATOR_FIELD",
    "DEXJOCO_RECAP_PISTAR06_VALUE_LANGUAGE_REPO",
    "DEXJOCO_RECAP_PISTAR06_VALUE_VISION_REPO",
    "DEXJOCO_RECAP_PISTAR06_VALUE_DTYPE",
    "DEXJOCO_RECAP_PISTAR06_VALUE_STEPS",
    "DEXJOCO_RECAP_PISTAR06_VALUE_BATCH_SIZE",
    "DEXJOCO_RECAP_PISTAR06_VALUE_NUM_WORKERS",
    "DEXJOCO_RECAP_PISTAR06_SKIP_VALUE_TRAINING",
    "DEXJOCO_RECAP_PISTAR06_VALUE_CHECKPOINT_SOURCE_TAR",
]
manifest = {key: os.environ.get(key) for key in keys}
manifest["policy_rollout_dataset"] = str(rollout_path)
manifest["policy_rollout_sha256"] = sha256_file(rollout_path)
manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(f"[job] wrote run manifest: {manifest_path}")
PY

echo "[job] computing norm stats for ReCap rollout dataset"
cd "$DEXJOCO_DIR/openpi"
conda run --no-capture-output --prefix "$OPENPI_ENV_PREFIX" python scripts/compute_norm_stats.py \
  "$DEXJOCO_TASK" \
  --batch-size="$DEXJOCO_RECAP_BATCH_SIZE" \
  --num-workers=0

echo "[job] training ReCap ACP policy steps=$DEXJOCO_RECAP_TRAIN_STEPS batch=$DEXJOCO_RECAP_BATCH_SIZE fsdp=$DEXJOCO_RECAP_FSDP_DEVICES workers=$DEXJOCO_RECAP_NUM_WORKERS lora_only=$OPENPI_RECAP_LORA_ONLY base_repeat=$OPENPI_RECAP_BASE_REPEAT positive_repeat=$OPENPI_RECAP_POSITIVE_REPEAT include_failures=$DEXJOCO_RECAP_INCLUDE_FAILURES collect_prompt=$DEXJOCO_RECAP_COLLECT_PROMPT_MODE label_with_value=$DEXJOCO_RECAP_LABEL_WITH_VALUE label_backend=$DEXJOCO_RECAP_LABEL_BACKEND random_positive=$DEXJOCO_RECAP_VALUE_RANDOM_POSITIVE rollout_npz=$OPENPI_RECAP_ROLLOUT_NPZ indicator_field=$OPENPI_RECAP_INDICATOR_FIELD prompt_mode=$OPENPI_RECAP_PROMPT_MODE prompt_manifest=$OPENPI_RECAP_ACP_PROMPTS_FILE mem_fraction=$XLA_PYTHON_CLIENT_MEM_FRACTION preallocate=$XLA_PYTHON_CLIENT_PREALLOCATE"
conda run --no-capture-output --prefix "$OPENPI_ENV_PREFIX" python scripts/train.py \
  "$DEXJOCO_TASK" \
  --exp-name="$DEXJOCO_RECAP_EXP_NAME" \
  --overwrite \
  --num-train-steps="$DEXJOCO_RECAP_TRAIN_STEPS" \
  --num-workers="$DEXJOCO_RECAP_NUM_WORKERS" \
  --save-interval="$DEXJOCO_RECAP_SAVE_INTERVAL" \
  --log-interval=50 \
  --fsdp-devices="$DEXJOCO_RECAP_FSDP_DEVICES"

RECAP_CKPT_ROOT="$DEXJOCO_DIR/checkpoints/recap_acp_ckpts/$DEXJOCO_TASK/$DEXJOCO_RECAP_EXP_NAME"
RECAP_STEP="$(find "$RECAP_CKPT_ROOT" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort -n | tail -1)"
if [[ -z "$RECAP_STEP" ]]; then
  echo "[job] no ReCap checkpoint found in $RECAP_CKPT_ROOT" >&2
  exit 1
fi
RECAP_POLICY_DIR="$RECAP_CKPT_ROOT/$RECAP_STEP"
echo "[job] final ReCap checkpoint: $RECAP_POLICY_DIR"

if [[ "$DEXJOCO_RECAP_SKIP_EVAL" == "1" ]]; then
  echo -e "method\tstatus\tsuccesses\tepisodes\tsuccess_rate_file\tcheckpoint_step\teval_prompt_mode\tlabel_backend\tprompt_mode\teval_seed" | tee "$SUMMARY"
  echo -e "$DEXJOCO_RECAP_EXP_NAME\ttrain_only\tNA\tNA\tNA\t$RECAP_STEP\t$DEXJOCO_RECAP_EVAL_PROMPT_MODE\t$DEXJOCO_RECAP_LABEL_BACKEND\t$OPENPI_RECAP_PROMPT_MODE\tNA" | tee -a "$SUMMARY"
  echo "[job] skipping ReCap eval for train-only iterative round"
  exit 0
fi

start_policy_server "$RECAP_POLICY_DIR" "$OUT_DIR/recap_server.log"
recap_server_pid="$POLICY_SERVER_PID"

eval_config="$acp_eval_config"
eval_output_dir="$OUT_DIR/recap_acp_eval_episodes"
if [[ "$DEXJOCO_RECAP_EVAL_PROMPT_MODE" == "base" ]]; then
  eval_config="$base_config"
  eval_output_dir="$OUT_DIR/recap_base_eval_episodes"
elif [[ "$DEXJOCO_RECAP_EVAL_PROMPT_MODE" != "acp" ]]; then
  echo "[job] unsupported DEXJOCO_RECAP_EVAL_PROMPT_MODE=$DEXJOCO_RECAP_EVAL_PROMPT_MODE" >&2
  exit 2
fi
echo "[job] evaluating ReCap policy prompt_mode=$DEXJOCO_RECAP_EVAL_PROMPT_MODE config=$eval_config"
if [[ -n "$DEXJOCO_RECAP_EVAL_SEEDS" ]]; then
  read -r -a eval_seeds <<< "$DEXJOCO_RECAP_EVAL_SEEDS"
else
  eval_seeds=("$DEXJOCO_EVAL_SEED")
fi

echo -e "method\tstatus\tsuccesses\tepisodes\tsuccess_rate_file\tcheckpoint_step\teval_prompt_mode\tlabel_backend\tprompt_mode\teval_seed" | tee "$SUMMARY"
infra_failures=0
for eval_seed in "${eval_seeds[@]}"; do
  seed_output_dir="${eval_output_dir}_seed${eval_seed}"
  echo "[job] eval seed=$eval_seed episodes=$DEXJOCO_EVAL_EPISODES output=$seed_output_dir"
  cd "$DEXJOCO_DIR"
  set +e
  conda run --no-capture-output --prefix "$DEXJOCO_ENV_PREFIX" dexjoco-openpi-eval \
    --config="$eval_config" \
    --seed="$eval_seed" \
    --port="$DEXJOCO_PORT" \
    --host="$DEXJOCO_HOST" \
    --episodes="$DEXJOCO_EVAL_EPISODES" \
    --output="$seed_output_dir" \
    2>&1 | tee "$OUT_DIR/recap_eval_seed${eval_seed}.log"
  eval_status=${PIPESTATUS[0]}
  set -e

  success_file="$(find "$seed_output_dir" -maxdepth 1 -name 'success_rate_*.txt' -printf '%f\n' | sort | head -1 || true)"
  successes=0
  episodes="$DEXJOCO_EVAL_EPISODES"
  if [[ "$success_file" =~ success_rate_([0-9]+)_([0-9]+)\.txt ]]; then
    successes="${BASH_REMATCH[1]}"
    episodes="${BASH_REMATCH[2]}"
  fi

  status="ok"
  if [[ "$eval_status" -ne 0 ]]; then
    status="eval_failed"
    infra_failures=$((infra_failures + 1))
  fi
  echo -e "$DEXJOCO_RECAP_EXP_NAME\t$status\t$successes\t$episodes\t$success_file\t$RECAP_STEP\t$DEXJOCO_RECAP_EVAL_PROMPT_MODE\t$DEXJOCO_RECAP_LABEL_BACKEND\t$OPENPI_RECAP_PROMPT_MODE\t$eval_seed" | tee -a "$SUMMARY"
done

stop_policy_server "$recap_server_pid"

if [[ "$infra_failures" -ne 0 ]]; then
  exit 1
fi

echo "[job] DexJoCo click_mouse ReCap rollout fine-tune finished"

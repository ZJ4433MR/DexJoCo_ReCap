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
DEXJOCO_COLLECT_SEED="${DEXJOCO_COLLECT_SEED:-$DEXJOCO_EVAL_SEED}"
DEXJOCO_COLLECT_EPISODES="${DEXJOCO_COLLECT_EPISODES:-20}"
DEXJOCO_EVAL_EPISODES="${DEXJOCO_EVAL_EPISODES:-20}"
DEXJOCO_PORT="${DEXJOCO_PORT:-$(dexjoco_default_port)}"
DEXJOCO_HOST="${DEXJOCO_HOST:-127.0.0.1}"
DEXJOCO_ACP_SUFFIX="${DEXJOCO_ACP_SUFFIX:- Use the high-advantage successful strategy.}"
DEXJOCO_RECAP_OUTPUT_NAME="${DEXJOCO_RECAP_OUTPUT_NAME:-dexjoco_click_mouse_recap_rollout_finetune}"
DEXJOCO_RECAP_DATA_PREFIX="${DEXJOCO_RECAP_DATA_PREFIX:-recap}"
DEXJOCO_RECAP_ROLLOUT_POLICY_DIR="${DEXJOCO_RECAP_ROLLOUT_POLICY_DIR:-../checkpoints/pi05_dexjoco_ckpt/$DEXJOCO_TASK}"
DEXJOCO_RECAP_PRETRAINED_MODEL_PATH="${DEXJOCO_RECAP_PRETRAINED_MODEL_PATH:-../checkpoints/pi05_dexjoco_ckpt/$DEXJOCO_TASK/params}"
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
DEXJOCO_RECAP_LABEL_WITH_VALUE="${DEXJOCO_RECAP_LABEL_WITH_VALUE:-0}"
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
OPENPI_RECAP_LORA_ONLY="${OPENPI_RECAP_LORA_ONLY:-1}"
OPENPI_RECAP_BASE_REPEAT="${OPENPI_RECAP_BASE_REPEAT:-0}"
OPENPI_RECAP_POSITIVE_REPEAT="${OPENPI_RECAP_POSITIVE_REPEAT:-1}"

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
        self._action_horizon = action_horizon
        self._episode_end = {}
        for episode_id in np.unique(self._episode_id):
            indices = np.flatnonzero(self._episode_id == episode_id)
            self._episode_end[int(episode_id)] = int(indices[-1]) + 1
        base_repeat = max(0, int(os.environ.get("OPENPI_RECAP_BASE_REPEAT", "0")))
        positive_repeat = max(1, int(os.environ.get("OPENPI_RECAP_POSITIVE_REPEAT", "1")))
        source_indices = []
        prompt_ids = []
        for idx in range(len(self._action)):
            is_positive = bool(self._acp_indicator[idx])
            source_indices.append(idx)
            prompt_ids.append(1 if is_positive else 0)
            for _ in range(base_repeat):
                source_indices.append(idx)
                prompt_ids.append(0)
            if is_positive:
                for _ in range(positive_repeat - 1):
                    source_indices.append(idx)
                    prompt_ids.append(1)
        self._source_indices = np.asarray(source_indices, dtype=np.int64)
        self._prompt_ids = np.asarray(prompt_ids, dtype=np.int8)

    def __getitem__(self, index: SupportsIndex) -> dict:
        virtual_idx = index.__index__()
        idx = int(self._source_indices[virtual_idx])
        prompt = self._acp_prompt if int(self._prompt_ids[virtual_idx]) == 1 else self._base_prompt
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
conda run --no-capture-output --prefix "$DEXJOCO_ENV_PREFIX" python "$EXP_DIR/scripts/dexjoco_collect_success_rollouts.py" "${collect_args[@]}"
cp "${COLLECTED_ROLLOUT_DATASET%.npz}.summary.txt" "$OUT_DIR/recap_success_rollouts.summary.txt"

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

stop_policy_server "$public_server_pid"

if [[ "$DEXJOCO_RECAP_COLLECT_ONLY" == "1" ]]; then
  echo "[job] collect-only requested; stopping before value labeling and policy training"
  exit 0
fi

POLICY_ROLLOUT_DATASET="$ROLLOUT_DATASET"
if [[ "$DEXJOCO_RECAP_LABEL_WITH_VALUE" == "1" ]]; then
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
  echo "[job] training value model and labeling ReCap advantages"
  python "$EXP_DIR/scripts/dexjoco_label_recap_rollouts.py" "${value_label_args[@]}"
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

echo "[job] computing norm stats for ReCap rollout dataset"
cd "$DEXJOCO_DIR/openpi"
conda run --no-capture-output --prefix "$OPENPI_ENV_PREFIX" python scripts/compute_norm_stats.py \
  "$DEXJOCO_TASK" \
  --batch-size="$DEXJOCO_RECAP_BATCH_SIZE" \
  --num-workers=0

echo "[job] training ReCap ACP policy steps=$DEXJOCO_RECAP_TRAIN_STEPS batch=$DEXJOCO_RECAP_BATCH_SIZE fsdp=$DEXJOCO_RECAP_FSDP_DEVICES workers=$DEXJOCO_RECAP_NUM_WORKERS lora_only=$OPENPI_RECAP_LORA_ONLY base_repeat=$OPENPI_RECAP_BASE_REPEAT positive_repeat=$OPENPI_RECAP_POSITIVE_REPEAT include_failures=$DEXJOCO_RECAP_INCLUDE_FAILURES collect_prompt=$DEXJOCO_RECAP_COLLECT_PROMPT_MODE label_with_value=$DEXJOCO_RECAP_LABEL_WITH_VALUE random_positive=$DEXJOCO_RECAP_VALUE_RANDOM_POSITIVE rollout_npz=$OPENPI_RECAP_ROLLOUT_NPZ indicator_field=$OPENPI_RECAP_INDICATOR_FIELD mem_fraction=$XLA_PYTHON_CLIENT_MEM_FRACTION preallocate=$XLA_PYTHON_CLIENT_PREALLOCATE"
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
  echo -e "method\tstatus\tsuccesses\tepisodes\tsuccess_rate_file\tcheckpoint_step" | tee "$SUMMARY"
  echo -e "$DEXJOCO_RECAP_EXP_NAME\ttrain_only\tNA\tNA\tNA\t$RECAP_STEP" | tee -a "$SUMMARY"
  echo "[job] skipping ReCap eval for train-only iterative round"
  exit 0
fi

start_policy_server "$RECAP_POLICY_DIR" "$OUT_DIR/recap_server.log"
recap_server_pid="$POLICY_SERVER_PID"

cd "$DEXJOCO_DIR"
set +e
conda run --no-capture-output --prefix "$DEXJOCO_ENV_PREFIX" dexjoco-openpi-eval \
  --config="$acp_eval_config" \
  --seed="$DEXJOCO_EVAL_SEED" \
  --port="$DEXJOCO_PORT" \
  --host="$DEXJOCO_HOST" \
  --episodes="$DEXJOCO_EVAL_EPISODES" \
  --output="$OUT_DIR/recap_acp_eval_episodes" \
  2>&1 | tee "$OUT_DIR/recap_eval.log"
eval_status=${PIPESTATUS[0]}
set -e

stop_policy_server "$recap_server_pid"

success_file="$(find "$OUT_DIR/recap_acp_eval_episodes" -maxdepth 1 -name 'success_rate_*.txt' -printf '%f\n' | sort | head -1 || true)"
successes=0
episodes="$DEXJOCO_EVAL_EPISODES"
if [[ "$success_file" =~ success_rate_([0-9]+)_([0-9]+)\.txt ]]; then
  successes="${BASH_REMATCH[1]}"
  episodes="${BASH_REMATCH[2]}"
fi

echo -e "method\tstatus\tsuccesses\tepisodes\tsuccess_rate_file\tcheckpoint_step" | tee "$SUMMARY"
status="ok"
if [[ "$eval_status" -ne 0 ]]; then
  status="eval_failed"
fi
echo -e "$DEXJOCO_RECAP_EXP_NAME\t$status\t$successes\t$episodes\t$success_file\t$RECAP_STEP" | tee -a "$SUMMARY"

if [[ "$eval_status" -ne 0 ]]; then
  exit "$eval_status"
fi

echo "[job] DexJoCo click_mouse ReCap rollout fine-tune finished"

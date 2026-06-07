#!/usr/bin/env bash
set -euo pipefail

cd "$EXP_DIR"
source scripts/dexjoco_common.sh

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.90}"

RUN_ROOT="${RUN_ROOT:-$(_dexjoco_run_root)}"
DEXJOCO_TASK="${DEXJOCO_TASK:-click_mouse}"
DEXJOCO_EVAL_SEED="${DEXJOCO_EVAL_SEED:-0}"
DEXJOCO_COLLECT_EPISODES="${DEXJOCO_COLLECT_EPISODES:-20}"
DEXJOCO_EVAL_EPISODES="${DEXJOCO_EVAL_EPISODES:-20}"
DEXJOCO_PORT="${DEXJOCO_PORT:-8000}"
DEXJOCO_HOST="${DEXJOCO_HOST:-127.0.0.1}"
DEXJOCO_ACP_SUFFIX="${DEXJOCO_ACP_SUFFIX:- Use the high-advantage successful strategy.}"
DEXJOCO_RECAP_TRAIN_STEPS="${DEXJOCO_RECAP_TRAIN_STEPS:-500}"
DEXJOCO_RECAP_BATCH_SIZE="${DEXJOCO_RECAP_BATCH_SIZE:-2}"
DEXJOCO_RECAP_FSDP_DEVICES="${DEXJOCO_RECAP_FSDP_DEVICES:-2}"
DEXJOCO_RECAP_WARMUP_STEPS="${DEXJOCO_RECAP_WARMUP_STEPS:-50}"
DEXJOCO_RECAP_SAVE_INTERVAL="${DEXJOCO_RECAP_SAVE_INTERVAL:-250}"
DEXJOCO_RECAP_EXP_NAME="${DEXJOCO_RECAP_EXP_NAME:-recap_success_rollout_acp}"
OPENPI_RECAP_LORA_ONLY="${OPENPI_RECAP_LORA_ONLY:-1}"

OUT_DIR="$OUTPUT_DIR/dexjoco_click_mouse_recap_rollout_finetune"
CONFIG_DIR="$OUT_DIR/configs"
ROLLOUT_DATASET="$RUN_ROOT/recap_success_rollouts.npz"
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
if "class RecapRolloutDataset" not in text:
    marker = "\n\nclass FakeDataset(Dataset):"
    insert = r'''

class RecapRolloutDataset(Dataset):
    """Small NPZ dataset built from successful DexJoCo policy rollouts."""

    def __init__(self, path: str, action_horizon: int):
        self._path = path
        data = np.load(path, allow_pickle=False)
        self._base = data["base"]
        self._wrist = data["wrist"]
        self._state = data["state"]
        self._action = data["action"]
        self._episode_id = data["episode_id"]
        self._prompt = str(data["prompt"])
        self._action_horizon = action_horizon
        self._episode_end = {}
        for episode_id in np.unique(self._episode_id):
            indices = np.flatnonzero(self._episode_id == episode_id)
            self._episode_end[int(episode_id)] = int(indices[-1]) + 1

    def __getitem__(self, index: SupportsIndex) -> dict:
        idx = index.__index__()
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
            "prompt": self._prompt,
            "task_index": np.asarray(0, dtype=np.int64),
            "task": self._prompt,
        }

    def __len__(self) -> int:
        return len(self._action)
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
  conda run --no-capture-output --prefix "$OPENPI_ENV_PREFIX" python - "$DEXJOCO_DIR/openpi/config.yaml" "$DEXJOCO_TASK" "$DEXJOCO_RECAP_BATCH_SIZE" "$DEXJOCO_RECAP_TRAIN_STEPS" <<'PY'
from pathlib import Path
import sys
import yaml

path = Path(sys.argv[1])
task = sys.argv[2]
batch_size = int(sys.argv[3])
steps = int(sys.argv[4])
cfg = yaml.safe_load(path.read_text())
cfg["pretrained_model_path"] = f"../checkpoints/pi05_dexjoco_ckpt/{task}/params"
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

start_policy_server() {
  local policy_dir="$1"
  local log_path="$2"
  cd "$DEXJOCO_DIR/openpi"
  conda run --no-capture-output --prefix "$OPENPI_ENV_PREFIX" python scripts/serve_policy.py \
    --port="$DEXJOCO_PORT" \
    policy:checkpoint \
    --policy.config="$DEXJOCO_TASK" \
    --policy.dir="$policy_dir" \
    > "$log_path" 2>&1 &
  local pid=$!
  SERVER_PIDS+=("$pid")
  if ! wait_for_log_pattern "$log_path" "server listening on" 900; then
    echo "[job] server did not become ready: $policy_dir" >&2
    tail -200 "$log_path" >&2 || true
    exit 1
  fi
  POLICY_SERVER_PID="$pid"
}

stop_policy_server() {
  local pid="$1"
  if kill -0 "$pid" >/dev/null 2>&1; then
    kill "$pid" >/dev/null 2>&1 || true
    wait "$pid" >/dev/null 2>&1 || true
  fi
}

echo "[job] starting public policy server for rollout collection"
start_policy_server "../checkpoints/pi05_dexjoco_ckpt/$DEXJOCO_TASK" "$OUT_DIR/public_server.log"
public_server_pid="$POLICY_SERVER_PID"

cd "$DEXJOCO_DIR"
conda run --no-capture-output --prefix "$DEXJOCO_ENV_PREFIX" python "$EXP_DIR/scripts/dexjoco_collect_success_rollouts.py" \
  --config="$base_config" \
  --output="$ROLLOUT_DATASET" \
  --host="$DEXJOCO_HOST" \
  --port="$DEXJOCO_PORT" \
  --seed="$DEXJOCO_EVAL_SEED" \
  --episodes="$DEXJOCO_COLLECT_EPISODES" \
  --acp-suffix="$DEXJOCO_ACP_SUFFIX"
cp "${ROLLOUT_DATASET%.npz}.summary.txt" "$OUT_DIR/recap_success_rollouts.summary.txt"

stop_policy_server "$public_server_pid"

patch_openpi_for_recap_rollout_dataset
patch_openpi_config_yaml

export OPENPI_RECAP_ROLLOUT_NPZ="$ROLLOUT_DATASET"
export DEXJOCO_RECAP_WARMUP_STEPS
export OPENPI_RECAP_LORA_ONLY

echo "[job] computing norm stats for ReCap rollout dataset"
cd "$DEXJOCO_DIR/openpi"
conda run --no-capture-output --prefix "$OPENPI_ENV_PREFIX" python scripts/compute_norm_stats.py \
  "$DEXJOCO_TASK" \
  --batch-size="$DEXJOCO_RECAP_BATCH_SIZE" \
  --num-workers=0

echo "[job] training ReCap ACP policy steps=$DEXJOCO_RECAP_TRAIN_STEPS batch=$DEXJOCO_RECAP_BATCH_SIZE fsdp=$DEXJOCO_RECAP_FSDP_DEVICES lora_only=$OPENPI_RECAP_LORA_ONLY"
conda run --no-capture-output --prefix "$OPENPI_ENV_PREFIX" python scripts/train.py \
  "$DEXJOCO_TASK" \
  --exp-name="$DEXJOCO_RECAP_EXP_NAME" \
  --overwrite \
  --num-train-steps="$DEXJOCO_RECAP_TRAIN_STEPS" \
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
echo -e "recap_success_rollout_acp\t$status\t$successes\t$episodes\t$success_file\t$RECAP_STEP" | tee -a "$SUMMARY"

if [[ "$eval_status" -ne 0 ]]; then
  exit "$eval_status"
fi

echo "[job] DexJoCo click_mouse ReCap rollout fine-tune finished"

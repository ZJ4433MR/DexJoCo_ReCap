#!/usr/bin/env bash
set -euo pipefail

cd "$EVORL_DIR"

echo "[job] Installing smoke-test dependencies into temporary PYDEPS_DIR=$PYDEPS_DIR"
python - <<'PY'
import importlib.util
import os
import subprocess
import sys

index_url = "https://pypi.tuna.tsinghua.edu.cn/simple"
target = os.environ["PYDEPS_DIR"]

top_level = {
    "draccus": "draccus==0.10.0",
    "datasets": "datasets>=4.0.0,<4.2.0",
    "huggingface_hub": "huggingface-hub>=0.34.2,<0.36.0",
    "safetensors": "safetensors>=0.4.3,<1.0.0",
    "termcolor": "termcolor>=2.4.0,<4.0.0",
    "accelerate": "accelerate>=1.10.0,<2.0.0",
    "deepdiff": "deepdiff>=7.0.1,<9.0.0",
    "transformers": "transformers>=4.57.1,<5.0.0",
    "tokenizers": "tokenizers>=0.22.0,<0.23.0",
}

light_deps = [
    "numpy<2",
    "pyarrow",
    "packaging>=24.2,<26.0",
    "pyyaml",
    "pyyaml-include~=1.4",
    "mergedeep",
    "toml",
    "typing-inspect",
    "tomli",
    "filelock",
    "dill<0.4.1,>=0.3.0",
    "pandas",
    "requests",
    "tqdm",
    "xxhash",
    "multiprocess<0.70.17",
    "fsspec[http]<=2025.9.0,>=2023.1.0",
    "aiohttp",
    "typing-extensions",
    "hf-xet",
    "psutil",
    "orderly-set",
    "pyserial>=3.5,<4.0",
    "gymnasium>=1.1.1,<2.0.0",
    "jsonlines>=4.0.0,<5.0.0",
    "einops>=0.8.0,<0.9.0",
    "av>=15.0.0,<16.0.0",
    "opencv-python-headless>=4.9.0,<4.13.0",
    "Pillow>=10.0.0,<13.0.0",
    "imageio[ffmpeg]>=2.34.0,<3.0.0",
    "regex",
]

missing_top = [pkg for module, pkg in top_level.items() if importlib.util.find_spec(module) is None]
if missing_top:
    print("missing top-level packages:", missing_top)
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--target",
            target,
            "--no-deps",
            "-i",
            index_url,
            *missing_top,
        ]
    )
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--target",
            target,
            "-i",
            index_url,
            *light_deps,
        ]
    )
else:
    print("all smoke-test dependencies are already importable")
PY

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
python - <<'PY'
from types import SimpleNamespace
import sys
import types

import numpy as np
import torch

class _StubPreTrainedPolicy(torch.nn.Module):
    config_class = object
    name = "stub"

    def __init__(self, *args, **kwargs):
        super().__init__()

policies_pkg = types.ModuleType("lerobot.policies")
policies_pkg.__path__ = []
factory_mod = types.ModuleType("lerobot.policies.factory")
factory_mod.make_policy = lambda *args, **kwargs: None
factory_mod.make_pre_post_processors = lambda *args, **kwargs: (None, None)
pretrained_mod = types.ModuleType("lerobot.policies.pretrained")
pretrained_mod.ActionSelectKwargs = dict
pretrained_mod.PreTrainedPolicy = _StubPreTrainedPolicy
viz_mod = types.ModuleType("lerobot.scripts.value_infer_viz")
viz_mod._export_overlay_videos = lambda *args, **kwargs: None
sys.modules.setdefault("lerobot.policies", policies_pkg)
sys.modules.setdefault("lerobot.policies.factory", factory_mod)
sys.modules.setdefault("lerobot.policies.pretrained", pretrained_mod)
sys.modules.setdefault("lerobot.scripts.value_infer_viz", viz_mod)

from lerobot.configs.train import ACPConfig
from lerobot.rl.acp_hook import build_acp_raw_batch_hook
from lerobot.rl.acp_tags import ACP_NEGATIVE_TAG, ACP_POSITIVE_TAG
from lerobot.scripts.lerobot_value_infer import (
    _binarize_advantages,
    _compute_dense_rewards_from_targets,
    _compute_n_step_advantages,
    _compute_task_thresholds,
)
from lerobot.values.pistar06.modeling_pistar06 import (
    EpisodeTargetInfo,
    compute_normalized_value_targets,
)

episode_info = {
    0: EpisodeTargetInfo(episode_index=0, task_index=0, length=3, success=True),
    1: EpisodeTargetInfo(episode_index=1, task_index=0, length=2, success=False),
}
task_max_lengths = {0: 3}
episode_indices = np.array([0, 0, 0, 1, 1], dtype=np.int64)
frame_indices = np.array([0, 1, 2, 0, 1], dtype=np.int64)

targets = compute_normalized_value_targets(
    episode_indices=episode_indices,
    frame_indices=frame_indices,
    episode_info=episode_info,
    task_max_lengths=task_max_lengths,
    c_fail_coef=1.0,
)
assert targets.shape == (5,)
assert targets[2] == 0.0
assert targets[-1] < targets[2]

rewards = _compute_dense_rewards_from_targets(targets, episode_indices, frame_indices)
advantages = _compute_n_step_advantages(
    rewards=rewards,
    values=targets,
    episode_indices=episode_indices,
    frame_indices=frame_indices,
    n_step=2,
)
thresholds = _compute_task_thresholds(
    task_indices=np.zeros_like(episode_indices),
    advantages=advantages,
    positive_ratio=0.4,
)
indicators = _binarize_advantages(
    task_indices=np.zeros_like(episode_indices),
    advantages=advantages,
    thresholds=thresholds,
    interventions=np.array([0, 1, 0, 0, 0], dtype=np.float32),
    force_intervention_positive=True,
)
assert indicators[1] == 1
assert set(indicators.tolist()).issubset({0, 1})

hook = build_acp_raw_batch_hook(
    ACPConfig(
        enable=True,
        indicator_field="complementary_info.acp_indicator",
        indicator_dropout_prob=0.0,
    ),
    seed=42,
)
batch = {
    "task": ["pick bottle", "place bottle"],
    "complementary_info.acp_indicator": torch.tensor([1, 0], dtype=torch.int64),
}
out = hook(batch, 0)
assert out["task"] == [
    f"pick bottle\n{ACP_POSITIVE_TAG}",
    f"place bottle\n{ACP_NEGATIVE_TAG}",
]

stats_dataset = SimpleNamespace(
    meta=SimpleNamespace(
        stats={
            "complementary_info.acp_indicator": {
                "mean": [0.3],
                "count": [10],
            }
        }
    )
)
from lerobot.rl.acp_dataset_stats import compute_acp_indicator_stats

stats = compute_acp_indicator_stats(stats_dataset, "complementary_info.acp_indicator")
assert stats is not None
assert stats.positive_ratio == 0.3
assert stats.positive_count == 3

print("RECAP/ACP smoke checks passed")
PY

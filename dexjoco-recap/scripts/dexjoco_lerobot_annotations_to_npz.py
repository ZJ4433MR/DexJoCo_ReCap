#!/usr/bin/env python
"""Copy Evo-RL/LeRobot value annotations back into a DexJoCo rollout NPZ."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset


def _as_1d_array(values, dtype) -> np.ndarray:
    array = np.asarray(values, dtype=dtype)
    if array.ndim == 2 and array.shape[1] == 1:
        array = array[:, 0]
    return array.reshape(-1)


def transfer(args: argparse.Namespace) -> None:
    with np.load(args.input, allow_pickle=False) as npz:
        arrays = {key: npz[key] for key in npz.files}

    dataset = LeRobotDataset(args.repo_id, root=args.root)
    frames = dataset.hf_dataset.with_format(None)
    frame_count = len(frames)
    npz_count = int(arrays["action"].shape[0])
    if frame_count != npz_count:
        raise ValueError(f"Frame count mismatch: LeRobot={frame_count}, NPZ={npz_count}.")

    missing = [
        field
        for field in (args.value_field, args.advantage_field, args.indicator_field)
        if field not in frames.column_names
    ]
    if missing:
        raise KeyError(f"Missing annotation fields in LeRobot dataset: {missing}")

    values = _as_1d_array(frames[args.value_field], np.float32)
    advantages = _as_1d_array(frames[args.advantage_field], np.float32)
    indicators = _as_1d_array(frames[args.indicator_field], np.int64)

    arrays.update(
        {
            "value": values.astype(np.float32),
            "advantage": advantages.astype(np.float32),
            "acp_indicator": indicators.astype(np.int64),
            "task_index": np.zeros_like(indicators, dtype=np.int64),
            "recap_value_backend": np.asarray("pistar06_lerobot"),
            "recap_value_field": np.asarray(args.value_field),
            "recap_advantage_field": np.asarray(args.advantage_field),
            "recap_indicator_field": np.asarray(args.indicator_field),
        }
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **arrays)

    summary = {
        "input": str(args.input),
        "output": str(args.output),
        "root": str(args.root),
        "repo_id": args.repo_id,
        "frames": int(frame_count),
        "value_min": float(np.min(values)),
        "value_max": float(np.max(values)),
        "advantage_min": float(np.min(advantages)),
        "advantage_max": float(np.max(advantages)),
        "indicator_positive_count": int(np.sum(indicators)),
        "indicator_positive_ratio": float(np.mean(indicators.astype(np.float32))),
    }
    summary_path = args.summary_output or args.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--value-field", default="complementary_info.value")
    parser.add_argument("--advantage-field", default="complementary_info.advantage")
    parser.add_argument("--indicator-field", default="complementary_info.acp_indicator")
    parser.add_argument("--summary-output", type=Path)
    transfer(parser.parse_args())


if __name__ == "__main__":
    main()

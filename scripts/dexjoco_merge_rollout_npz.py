#!/usr/bin/env python
"""Merge compact DexJoCo rollout NPZ files into one data pool.

The collector writes episode ids starting from zero for every rollout batch.
When multiple batches are pooled for Evo-RL-style iterative training, episode
ids must be offset so value/advantage labeling does not connect different
rollouts into one trajectory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


FRAME_KEY = "action"


def _as_python_scalar(value: np.ndarray) -> object:
    if value.shape == ():
        return value.item()
    return value


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def merge_rollouts(inputs: list[Path], output: Path, summary_output: Path | None = None) -> None:
    if not inputs:
        raise ValueError("At least one input NPZ is required.")

    batches = [_load_npz(path) for path in inputs]
    required = {"base", "wrist", "state", "action", "episode_id", "is_success"}
    missing = [str(path) for path, batch in zip(inputs, batches) if not required.issubset(batch)]
    if missing:
        raise KeyError(f"Missing required rollout keys in: {missing}")

    common_keys = set(batches[0])
    for batch in batches[1:]:
        common_keys &= set(batch)

    frame_counts = [int(batch[FRAME_KEY].shape[0]) for batch in batches]
    merged: dict[str, np.ndarray] = {}
    episode_offset = 0
    per_input = []

    for key in sorted(common_keys):
        parts = []
        is_frame_array = all(batch[key].shape[:1] == (frame_count,) for batch, frame_count in zip(batches, frame_counts))
        if is_frame_array:
            for batch in batches:
                part = batch[key]
                if key == "episode_id":
                    part = part.astype(np.int32, copy=True) + episode_offset
                    episode_offset += int(np.max(batch[key])) + 1
                parts.append(part)
            merged[key] = np.concatenate(parts, axis=0)
        else:
            first = batches[0][key]
            first_scalar = _as_python_scalar(first)
            if all(np.array_equal(batch[key], first) for batch in batches[1:]):
                merged[key] = first
            elif key in {"total_episodes", "saved_episodes", "successful_episodes"}:
                merged[key] = np.asarray(sum(int(_as_python_scalar(batch[key])) for batch in batches), dtype=first.dtype)
            elif key == "episode_success_flags":
                merged[key] = np.concatenate([batch[key].reshape(-1) for batch in batches], axis=0)
            else:
                # Keep the first scalar metadata value for loader compatibility.
                merged[key] = first

    merged["pool_source_id"] = np.concatenate(
        [np.full(frame_count, i, dtype=np.int32) for i, frame_count in enumerate(frame_counts)], axis=0
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **merged)

    for path, batch, frame_count in zip(inputs, batches, frame_counts):
        is_success = batch["is_success"].astype(np.bool_, copy=False)
        episode_ids = batch["episode_id"].astype(np.int32, copy=False)
        success_by_episode = []
        for episode_id in sorted(np.unique(episode_ids)):
            indices = np.flatnonzero(episode_ids == episode_id)
            success_by_episode.append(bool(is_success[indices[0]]))
        per_input.append(
            {
                "path": str(path),
                "frames": frame_count,
                "saved_episodes": int(len(success_by_episode)),
                "successful_episodes": int(sum(success_by_episode)),
                "success_rate_saved_episodes": float(np.mean(success_by_episode)) if success_by_episode else 0.0,
            }
        )

    summary = {
        "output": str(output),
        "inputs": per_input,
        "frames": int(merged[FRAME_KEY].shape[0]),
        "saved_episodes": int(len(np.unique(merged["episode_id"]))),
        "successful_frames": int(np.sum(merged["is_success"].astype(np.bool_))),
    }
    summary_path = summary_output or output.with_suffix(".summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path)
    parser.add_argument("inputs", type=Path, nargs="+")
    args = parser.parse_args()
    merge_rollouts(args.inputs, args.output, args.summary_output)


if __name__ == "__main__":
    main()

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
import shutil
import tempfile
import zipfile
from pathlib import Path

import numpy as np
from numpy.lib import format as np_format


FRAME_KEY = "action"


def _as_python_scalar(value: np.ndarray) -> object:
    if value.shape == ():
        return value.item()
    return value


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def _write_npz_streamed(output: Path, arrays: dict[str, np.ndarray | Path]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, mode="w", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as zf:
        for key, value in arrays.items():
            array = np.load(value, mmap_mode="r") if isinstance(value, Path) else value
            with zf.open(f"{key}.npy", mode="w", force_zip64=True) as handle:
                np_format.write_array(handle, np.asarray(array), allow_pickle=False)


def _copy_frame_array(
    key: str,
    inputs: list[Path],
    output_path: Path,
    shape: tuple[int, ...],
    dtype: np.dtype,
    frame_counts: list[int],
) -> None:
    out = np_format.open_memmap(output_path, mode="w+", dtype=dtype, shape=shape)
    offset = 0
    episode_offset = 0
    for path, frame_count in zip(inputs, frame_counts):
        with np.load(path, allow_pickle=False) as data:
            part = data[key]
            if key == "episode_id":
                adjusted = part.astype(np.int32, copy=False) + episode_offset
                out[offset : offset + frame_count] = adjusted
                episode_offset += int(np.max(part)) + 1
            else:
                out[offset : offset + frame_count] = part
        offset += frame_count
    out.flush()
    del out


def merge_rollouts(inputs: list[Path], output: Path, summary_output: Path | None = None) -> None:
    if not inputs:
        raise ValueError("At least one input NPZ is required.")

    required = {"base", "wrist", "state", "action", "episode_id", "is_success"}
    headers = []
    for path in inputs:
        with np.load(path, allow_pickle=False) as data:
            headers.append({key: (data[key].shape, data[key].dtype) for key in data.files})

    missing = [str(path) for path, header in zip(inputs, headers) if not required.issubset(header)]
    if missing:
        raise KeyError(f"Missing required rollout keys in: {missing}")

    common_keys = set(headers[0])
    for header in headers[1:]:
        common_keys &= set(header)

    frame_counts = [int(header[FRAME_KEY][0][0]) for header in headers]
    total_frames = int(sum(frame_counts))
    merged: dict[str, np.ndarray | Path] = {}
    per_input = []
    temp_dir = Path(tempfile.mkdtemp(prefix="dexjoco-merge-", dir=str(output.parent)))

    try:
        for key in sorted(common_keys):
            key_shapes = [header[key][0] for header in headers]
            key_dtypes = [header[key][1] for header in headers]
            is_frame_array = all(shape[:1] == (frame_count,) for shape, frame_count in zip(key_shapes, frame_counts))
            if is_frame_array:
                if len({shape[1:] for shape in key_shapes}) != 1:
                    raise ValueError(f"Cannot merge frame key {key!r} with incompatible shapes: {key_shapes}")
                dtype = np.dtype(np.int32 if key == "episode_id" else key_dtypes[0])
                if any(np.dtype(dt) != np.dtype(key_dtypes[0]) for dt in key_dtypes) and key != "episode_id":
                    raise ValueError(f"Cannot merge frame key {key!r} with incompatible dtypes: {key_dtypes}")
                tmp_path = temp_dir / f"{key}.npy"
                _copy_frame_array(
                    key=key,
                    inputs=inputs,
                    output_path=tmp_path,
                    shape=(total_frames, *key_shapes[0][1:]),
                    dtype=dtype,
                    frame_counts=frame_counts,
                )
                merged[key] = tmp_path
            else:
                small_parts = []
                for path in inputs:
                    with np.load(path, allow_pickle=False) as data:
                        small_parts.append(data[key])
                first = small_parts[0]
                first_scalar = _as_python_scalar(first)
                if all(np.array_equal(part, first) for part in small_parts[1:]):
                    merged[key] = first
                elif key in {"total_episodes", "saved_episodes", "successful_episodes"}:
                    merged[key] = np.asarray(sum(int(_as_python_scalar(part)) for part in small_parts), dtype=first.dtype)
                elif key == "episode_success_flags":
                    merged[key] = np.concatenate([part.reshape(-1) for part in small_parts], axis=0)
                else:
                    # Keep the first scalar metadata value for loader compatibility.
                    merged[key] = first

        pool_source_path = temp_dir / "pool_source_id.npy"
        pool_source = np_format.open_memmap(pool_source_path, mode="w+", dtype=np.int32, shape=(total_frames,))
        offset = 0
        for i, frame_count in enumerate(frame_counts):
            pool_source[offset : offset + frame_count] = i
            offset += frame_count
        pool_source.flush()
        del pool_source
        merged["pool_source_id"] = pool_source_path

        _write_npz_streamed(output, merged)

        successful_frames = 0
        for path, frame_count in zip(inputs, frame_counts):
            with np.load(path, allow_pickle=False) as data:
                is_success = data["is_success"].astype(np.bool_, copy=False)
                episode_ids = data["episode_id"].astype(np.int32, copy=False)
            successful_frames += int(np.sum(is_success))
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

        episode_ids = np.load(merged["episode_id"], mmap_mode="r")
        saved_episodes = int(len(np.unique(episode_ids)))
        del episode_ids

        summary = {
            "output": str(output),
            "inputs": per_input,
            "frames": total_frames,
            "saved_episodes": saved_episodes,
            "successful_frames": successful_frames,
        }
        summary_path = summary_output or output.with_suffix(".summary.json")
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(summary, indent=2, sort_keys=True))
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path)
    parser.add_argument("inputs", type=Path, nargs="+")
    args = parser.parse_args()
    merge_rollouts(args.inputs, args.output, args.summary_output)


if __name__ == "__main__":
    main()

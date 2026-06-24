#!/usr/bin/env python
"""Sample a bounded DexJoCo rollout pool by whole episodes.

Full Evo-RL data pools can be too large for the lightweight local value-labeling
path to materialize at once. This helper keeps episode trajectories intact while
building a smaller NPZ for value labeling and policy fine-tuning.
"""

from __future__ import annotations

import argparse
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.lib import format as np_format


@dataclass(frozen=True)
class EpisodeSpan:
    episode_id: int
    start: int
    stop: int
    success: bool
    source_id: int | None

    @property
    def frames(self) -> int:
        return int(self.stop - self.start)


def _ordered_unique(values: np.ndarray) -> list[int]:
    seen: set[int] = set()
    result: list[int] = []
    for raw in values:
        value = int(raw)
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


def _episode_spans(
    episode_ids: np.ndarray,
    is_success: np.ndarray,
    pool_source_id: np.ndarray | None,
) -> list[EpisodeSpan]:
    spans: list[EpisodeSpan] = []
    for episode_id in _ordered_unique(episode_ids):
        indices = np.flatnonzero(episode_ids == episode_id)
        if indices.size == 0:
            continue
        source_id = None if pool_source_id is None else int(pool_source_id[indices[0]])
        spans.append(
            EpisodeSpan(
                episode_id=episode_id,
                start=int(indices[0]),
                stop=int(indices[-1]) + 1,
                success=bool(is_success[indices[0]]),
                source_id=source_id,
            )
        )
    return spans


def _within_limits(
    current_frames: int,
    current_episodes: int,
    span: EpisodeSpan,
    max_frames: int,
    max_episodes: int,
) -> bool:
    if max_frames > 0 and current_frames + span.frames > max_frames:
        return False
    if max_episodes > 0 and current_episodes + 1 > max_episodes:
        return False
    return True


def _choose_spans(
    spans: list[EpisodeSpan],
    max_frames: int,
    max_episodes: int,
    seed: int,
    keep_last_source: bool,
) -> list[EpisodeSpan]:
    if max_frames <= 0 and max_episodes <= 0:
        return spans

    selected: list[EpisodeSpan] = []
    selected_ids: set[int] = set()
    current_frames = 0

    def add(span: EpisodeSpan) -> bool:
        nonlocal current_frames
        if span.episode_id in selected_ids:
            return True
        if not _within_limits(current_frames, len(selected), span, max_frames, max_episodes):
            return False
        selected.append(span)
        selected_ids.add(span.episode_id)
        current_frames += span.frames
        return True

    if keep_last_source:
        source_values = [span.source_id for span in spans if span.source_id is not None]
        if source_values:
            last_source = max(source_values)
            for span in spans:
                if span.source_id == last_source:
                    add(span)

    rng = np.random.default_rng(seed)
    remaining = [span for span in spans if span.episode_id not in selected_ids]
    success_remaining = [span for span in remaining if span.success]
    failure_remaining = [span for span in remaining if not span.success]
    rng.shuffle(success_remaining)
    rng.shuffle(failure_remaining)

    # Alternate classes to keep failures in the bounded pool when available.
    queues = [failure_remaining, success_remaining]
    turn = 0
    while queues[0] or queues[1]:
        queue = queues[turn % 2]
        other = queues[(turn + 1) % 2]
        if not queue and other:
            queue = other
        if not queue:
            break
        span = queue.pop()
        add(span)
        turn += 1

    return sorted(selected, key=lambda span: span.start)


def _selected_indices(selected: list[EpisodeSpan]) -> np.ndarray:
    parts = [np.arange(span.start, span.stop, dtype=np.int64) for span in selected]
    if not parts:
        return np.asarray([], dtype=np.int64)
    return np.concatenate(parts, axis=0)


def _new_episode_ids(selected: list[EpisodeSpan]) -> np.ndarray:
    parts = [np.full(span.frames, i, dtype=np.int32) for i, span in enumerate(selected)]
    if not parts:
        return np.asarray([], dtype=np.int32)
    return np.concatenate(parts, axis=0)


def _write_array(zf: zipfile.ZipFile, key: str, value: np.ndarray) -> None:
    with zf.open(f"{key}.npy", mode="w", force_zip64=True) as handle:
        np_format.write_array(handle, np.asarray(value), allow_pickle=False)


def sample_rollouts(args: argparse.Namespace) -> None:
    if args.max_frames <= 0 and args.max_episodes <= 0:
        raise ValueError("Set at least one of --max-frames or --max-episodes.")

    with np.load(args.input, allow_pickle=False) as data:
        files = list(data.files)
        episode_ids = data["episode_id"].astype(np.int32, copy=False)
        is_success = data["is_success"].astype(np.bool_, copy=False)
        pool_source_id = data["pool_source_id"].astype(np.int32, copy=False) if "pool_source_id" in data else None

    spans = _episode_spans(episode_ids, is_success, pool_source_id)
    selected = _choose_spans(
        spans=spans,
        max_frames=args.max_frames,
        max_episodes=args.max_episodes,
        seed=args.seed,
        keep_last_source=args.keep_last_source,
    )
    if not selected:
        raise RuntimeError("No episodes selected for sampled rollout pool.")

    frame_indices = _selected_indices(selected)
    new_episode_ids = _new_episode_ids(selected)
    selected_episode_ids = np.asarray([span.episode_id for span in selected], dtype=np.int64)
    selected_success_flags = np.asarray([span.success for span in selected], dtype=np.bool_)
    selected_source_ids = np.asarray(
        [-1 if span.source_id is None else span.source_id for span in selected],
        dtype=np.int32,
    )
    selected_frames = int(frame_indices.shape[0])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with np.load(args.input, allow_pickle=False) as data, zipfile.ZipFile(
        args.output,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        allowZip64=True,
    ) as zf:
        frame_count = int(episode_ids.shape[0])
        episode_count = int(len(spans))
        for key in files:
            value = data[key]
            if key == "episode_id":
                _write_array(zf, key, new_episode_ids)
            elif value.shape[:1] == (frame_count,):
                _write_array(zf, key, value[frame_indices])
            elif key in {"total_episodes", "saved_episodes"}:
                _write_array(zf, key, np.asarray(len(selected), dtype=value.dtype))
            elif key == "successful_episodes":
                _write_array(zf, key, np.asarray(int(np.sum(selected_success_flags)), dtype=value.dtype))
            elif key == "episode_success_flags":
                _write_array(zf, key, selected_success_flags)
            elif value.shape[:1] == (episode_count,):
                _write_array(zf, key, value[selected_episode_ids])
            else:
                _write_array(zf, key, value)

    by_source: dict[str, dict[str, int]] = {}
    for span in selected:
        source_key = "none" if span.source_id is None else str(span.source_id)
        entry = by_source.setdefault(source_key, {"episodes": 0, "frames": 0, "successful_episodes": 0})
        entry["episodes"] += 1
        entry["frames"] += span.frames
        entry["successful_episodes"] += int(span.success)

    summary = {
        "input": str(args.input),
        "output": str(args.output),
        "original_episodes": int(len(spans)),
        "original_frames": int(episode_ids.shape[0]),
        "selected_episodes": int(len(selected)),
        "selected_frames": selected_frames,
        "selected_successful_episodes": int(np.sum(selected_success_flags)),
        "selected_success_rate": float(np.mean(selected_success_flags.astype(np.float32))),
        "selected_source_ids": selected_source_ids.tolist(),
        "selected_by_source": by_source,
        "max_frames": int(args.max_frames),
        "max_episodes": int(args.max_episodes),
        "keep_last_source": bool(args.keep_last_source),
        "seed": int(args.seed),
    }
    summary_path = args.summary_output or args.output.with_suffix(".summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"[sample] wrote sampled rollout dataset: {args.output}")
    print(f"[sample] wrote summary: {summary_path}")
    print(json.dumps(summary, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--max-episodes", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--keep-last-source", action="store_true")
    sample_rollouts(parser.parse_args())


if __name__ == "__main__":
    main()

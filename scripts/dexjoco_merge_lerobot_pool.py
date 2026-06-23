#!/usr/bin/env python
"""Merge several local LeRobot datasets into one pool."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

from lerobot.datasets.dataset_tools import merge_datasets
from lerobot.datasets.lerobot_dataset import LeRobotDataset


def _parse_input(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Inputs must be formatted as repo_id=/absolute/or/relative/root")
    repo_id, root = value.split("=", 1)
    repo_id = repo_id.strip()
    root = root.strip()
    if not repo_id or not root:
        raise argparse.ArgumentTypeError("Both repo_id and root are required in --input repo_id=root")
    return repo_id, Path(root)


def merge_pool(args: argparse.Namespace) -> None:
    if args.output_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output root exists: {args.output_root}. Use --overwrite to replace it.")
        shutil.rmtree(args.output_root)

    datasets = []
    input_summaries = []
    for repo_id, root in args.inputs:
        dataset = LeRobotDataset(repo_id=repo_id, root=root)
        datasets.append(dataset)
        input_summaries.append(
            {
                "repo_id": repo_id,
                "root": str(root),
                "episodes": int(dataset.num_episodes),
                "frames": int(dataset.num_frames),
            }
        )

    merged = merge_datasets(
        datasets=datasets,
        output_repo_id=args.output_repo_id,
        output_dir=args.output_root,
    )
    summary = {
        "output_repo_id": args.output_repo_id,
        "output_root": str(args.output_root),
        "inputs": input_summaries,
        "episodes": int(merged.num_episodes),
        "frames": int(merged.num_frames),
    }
    summary_path = args.summary_output or args.output_root / "dexjoco_merge_lerobot_pool.summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output-repo-id", required=True)
    parser.add_argument("--input", dest="inputs", action="append", type=_parse_input, required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--summary-output", type=Path)
    merge_pool(parser.parse_args())


if __name__ == "__main__":
    main()

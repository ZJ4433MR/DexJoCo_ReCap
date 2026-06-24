#!/usr/bin/env python
"""Patch LeRobot episode metadata with an explicit episode_success label."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import pandas as pd


def _episodes_files(root: Path) -> list[Path]:
    meta = root / "meta" / "episodes"
    if not meta.is_dir():
        raise FileNotFoundError(f"LeRobot episodes metadata not found: {meta}")
    files = sorted(meta.rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No episode parquet files found under: {meta}")
    return files


def patch_episode_success(args: argparse.Namespace) -> None:
    source_root = args.root.resolve()
    target_root = args.output_root.resolve() if args.output_root else source_root

    if args.output_root:
        if target_root.exists():
            if not args.overwrite:
                raise FileExistsError(f"Output root exists: {target_root}. Use --overwrite to replace it.")
            shutil.rmtree(target_root)
        shutil.copytree(source_root, target_root)

    total = 0
    added = 0
    filled = 0
    overwritten = 0
    files = _episodes_files(target_root)
    for path in files:
        df = pd.read_parquet(path)
        total += int(len(df))
        if "episode_success" not in df.columns:
            df["episode_success"] = args.label
            added += int(len(df))
        elif args.force:
            overwritten += int(len(df))
            df["episode_success"] = args.label
        else:
            mask = df["episode_success"].isna() | (df["episode_success"].astype(str).str.strip() == "")
            filled += int(mask.sum())
            if bool(mask.any()):
                df.loc[mask, "episode_success"] = args.label
        df.to_parquet(path, index=False)

    summary = {
        "root": str(target_root),
        "source_root": str(source_root),
        "label": args.label,
        "files": len(files),
        "episodes": total,
        "added": added,
        "filled": filled,
        "overwritten": overwritten,
    }
    summary_path = args.summary_output or target_root / "dexjoco_lerobot_episode_success.summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--label", choices=("success", "failure"), default="success")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--summary-output", type=Path)
    patch_episode_success(parser.parse_args())


if __name__ == "__main__":
    main()

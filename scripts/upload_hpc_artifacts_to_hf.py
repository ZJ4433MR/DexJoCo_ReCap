#!/usr/bin/env python3
"""Upload HPC result archives listed in a manifest to a Hugging Face dataset.

Run this on the machine that can read the manifest's absolute
``result_tar_path`` values, typically the SSH/HPC host. The script never
deletes local or remote artifacts.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

try:
    from huggingface_hub import HfApi, create_repo
except ImportError as exc:  # pragma: no cover - depends on runtime env
    raise SystemExit(
        "Missing dependency: huggingface_hub. Install it in this Python "
        "environment, for example: pip install --user huggingface_hub"
    ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        required=True,
        type=Path,
        help="TSV manifest generated under reports/artifacts.",
    )
    parser.add_argument(
        "--repo-id",
        default="ZJ4433MR/recap-sim-l40-artifacts",
        help="Target Hugging Face dataset repo id.",
    )
    parser.add_argument(
        "--token-file",
        type=Path,
        default=None,
        help="Optional file containing a Hugging Face token. HF_TOKEN is used first.",
    )
    parser.add_argument(
        "--no-create-repo",
        action="store_true",
        help="Skip repo creation and assume the public dataset already exists.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the upload plan without uploading files.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Upload at most this many files. 0 means no limit.",
    )
    parser.add_argument(
        "--include-run",
        action="append",
        default=[],
        help="Only upload run names that contain this substring. Repeatable.",
    )
    parser.add_argument(
        "--skip-sha256",
        action="store_true",
        help="Do not compute SHA256 before upload.",
    )
    parser.add_argument(
        "--status-out",
        type=Path,
        default=Path("hf_upload_status.tsv"),
        help="TSV status file written as uploads complete.",
    )
    return parser.parse_args()


def read_token(token_file: Path | None) -> str:
    token = os.environ.get("HF_TOKEN", "").strip()
    if token:
        return token
    if token_file:
        return token_file.read_text(encoding="utf-8").strip()
    raise SystemExit("Set HF_TOKEN or pass --token-file.")


def sha256_file(path: Path, block_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            block = f.read(block_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def write_status(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "run_name",
        "status",
        "size_bytes",
        "sha256",
        "hf_path",
        "hf_url",
        "message",
        "updated_at",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> int:
    args = parse_args()
    token = "" if args.dry_run else read_token(args.token_file)
    rows = load_rows(args.manifest)
    selected = []
    for row in rows:
        if row.get("upload_status") == "no_result_tar":
            continue
        result_path = row.get("result_tar_path", "")
        if not result_path:
            continue
        run_name = row.get("run_name", "")
        if args.include_run and not any(s in run_name for s in args.include_run):
            continue
        selected.append(row)
    if args.limit:
        selected = selected[: args.limit]

    print(f"selected {len(selected)} archives for repo {args.repo_id}")
    if args.dry_run:
        for row in selected:
            print(f"DRY-RUN {row['result_tar_size_bytes']} {row['run_name']} -> {row['hf_path']}")
        return 0

    if not args.no_create_repo:
        create_repo(
            repo_id=args.repo_id,
            repo_type="dataset",
            private=False,
            exist_ok=True,
            token=token,
        )

    api = HfApi(token=token)
    status_rows: list[dict[str, str]] = []
    for index, row in enumerate(selected, start=1):
        src = Path(row["result_tar_path"])
        run_name = row["run_name"]
        hf_path = row["hf_path"]
        hf_url = row["hf_url"]
        size = row.get("result_tar_size_bytes", "")
        now = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
        if not src.is_file():
            status_rows.append(
                {
                    "run_name": run_name,
                    "status": "missing_source",
                    "size_bytes": size,
                    "sha256": "",
                    "hf_path": hf_path,
                    "hf_url": hf_url,
                    "message": str(src),
                    "updated_at": now,
                }
            )
            write_status(args.status_out, status_rows)
            continue

        print(f"[{index}/{len(selected)}] uploading {src} -> {hf_path}", flush=True)
        digest = "" if args.skip_sha256 else sha256_file(src)
        api.upload_file(
            path_or_fileobj=str(src),
            path_in_repo=hf_path,
            repo_id=args.repo_id,
            repo_type="dataset",
            commit_message=f"Upload {run_name} results",
        )
        status_rows.append(
            {
                "run_name": run_name,
                "status": "uploaded",
                "size_bytes": size,
                "sha256": digest,
                "hf_path": hf_path,
                "hf_url": hf_url,
                "message": "",
                "updated_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
            }
        )
        write_status(args.status_out, status_rows)

    return 0


if __name__ == "__main__":
    sys.exit(main())

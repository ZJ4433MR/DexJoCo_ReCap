#!/usr/bin/env python
"""Aggregate DexJoCo ReCap run summaries into CSV/Markdown tables."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev


def _read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _find_run_root(summary_path: Path) -> Path:
    parts = list(summary_path.parts)
    if "outputs" in parts:
        idx = parts.index("outputs")
        return Path(*parts[:idx]) if idx > 0 else Path(".")
    return summary_path.parent


def _success_rate(row: dict[str, str]) -> float | None:
    try:
        successes = int(row.get("successes", ""))
        episodes = int(row.get("episodes", ""))
    except ValueError:
        return None
    if episodes <= 0:
        return None
    return successes / episodes


def collect(runs_root: Path) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    for summary_path in sorted(runs_root.rglob("summary.tsv")):
        run_root = _find_run_root(summary_path)
        output_dir = summary_path.parent
        manifest = _read_json(output_dir / "run_manifest.json")
        value_summary = {}
        for candidate in (
            output_dir / "recap_pistar06_value_advantage.summary.json",
            output_dir / "recap_value_advantage.summary.json",
        ):
            value_summary = _read_json(candidate)
            if value_summary:
                break
        for row in _read_tsv(summary_path):
            rate = _success_rate(row)
            record = {
                "run": run_root.name,
                "output": str(output_dir.relative_to(run_root)) if output_dir.is_relative_to(run_root) else str(output_dir),
                "method": row.get("method") or row.get("variant") or "",
                "status": row.get("status", ""),
                "successes": row.get("successes", ""),
                "episodes": row.get("episodes", ""),
                "success_rate": "" if rate is None else f"{rate:.6f}",
                "eval_seed": row.get("eval_seed", manifest.get("DEXJOCO_EVAL_SEED", "")),
                "collect_seed": str(manifest.get("DEXJOCO_COLLECT_SEED", "")),
                "label_backend": row.get("label_backend", manifest.get("DEXJOCO_RECAP_LABEL_BACKEND", "")),
                "prompt_mode": row.get("prompt_mode", manifest.get("OPENPI_RECAP_PROMPT_MODE", "")),
                "eval_prompt_mode": row.get("eval_prompt_mode", manifest.get("DEXJOCO_RECAP_EVAL_PROMPT_MODE", "")),
                "rollout_sha256": str(manifest.get("policy_rollout_sha256", "")),
                "indicator_positive_ratio": str(value_summary.get("indicator_positive_ratio", "")),
                "summary_path": str(summary_path),
            }
            records.append(record)
    return records


def grouped_stats(records: list[dict[str, str]]) -> list[dict[str, str]]:
    groups: dict[tuple[str, str, str, str], list[float]] = defaultdict(list)
    episode_counts: dict[tuple[str, str, str, str], int] = defaultdict(int)
    success_counts: dict[tuple[str, str, str, str], int] = defaultdict(int)
    for record in records:
        if record["status"] not in {"ok", "baseline", "acp_positive"} and record["status"] != "ok":
            continue
        if not record["success_rate"]:
            continue
        key = (
            record["method"],
            record["label_backend"],
            record["prompt_mode"],
            record["eval_prompt_mode"],
        )
        groups[key].append(float(record["success_rate"]))
        try:
            success_counts[key] += int(record["successes"])
            episode_counts[key] += int(record["episodes"])
        except ValueError:
            pass

    rows = []
    for key, values in sorted(groups.items()):
        n = len(values)
        avg = mean(values)
        std = pstdev(values) if n > 1 else 0.0
        total_success = success_counts[key]
        total_episodes = episode_counts[key]
        pooled = total_success / total_episodes if total_episodes else math.nan
        se = math.sqrt(max(pooled * (1.0 - pooled), 0.0) / total_episodes) if total_episodes else math.nan
        rows.append(
            {
                "method": key[0],
                "label_backend": key[1],
                "prompt_mode": key[2],
                "eval_prompt_mode": key[3],
                "n_eval_rows": str(n),
                "mean_success_rate": f"{avg:.6f}",
                "std_success_rate": f"{std:.6f}",
                "pooled_success_rate": f"{pooled:.6f}" if not math.isnan(pooled) else "",
                "binomial_95ci_low": f"{pooled - 1.96 * se:.6f}" if total_episodes else "",
                "binomial_95ci_high": f"{pooled + 1.96 * se:.6f}" if total_episodes else "",
                "total_successes": str(total_success),
                "total_episodes": str(total_episodes),
            }
        )
    return rows


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path: Path, rows: list[dict[str, str]], title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text(f"# {title}\n\nNo rows found.\n", encoding="utf-8")
        return
    headers = list(rows[0])
    lines = [f"# {title}", "", "|" + "|".join(headers) + "|", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        lines.append("|" + "|".join(str(row.get(header, "")) for header in headers) + "|")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", type=Path, default=Path("runs"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/generated"))
    args = parser.parse_args()

    records = collect(args.runs_root)
    stats = grouped_stats(records)
    _write_csv(args.output_dir / "dexjoco_recap_runs.csv", records)
    _write_csv(args.output_dir / "dexjoco_recap_grouped.csv", stats)
    _write_markdown(args.output_dir / "dexjoco_recap_grouped.md", stats, "DexJoCo ReCap Grouped Results")
    print(f"records={len(records)} grouped={len(stats)} output_dir={args.output_dir}")


if __name__ == "__main__":
    main()

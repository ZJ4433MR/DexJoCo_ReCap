#!/usr/bin/env python
"""Generate DexJoCo ReCap prompt variants with lightweight prompt agents.

The agents in this file do not call an external LLM. They are deterministic
prompt-generation policies that inspect the task text and previous round
results, propose candidate ACP suffixes, and let a manager choose one for the
next Evo-RL/ReCap round. The manifest is also consumed by the OpenPI rollout
dataset patch so positive samples can be trained under multiple agent prompts.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import urllib.error
import urllib.request
from typing import Any


DEFAULT_AGENT_IDS = (
    "advantage_refiner",
    "task_specialist",
    "recovery_planner",
    "motion_minimizer",
)

AGENT_ROLES = {
    "advantage_refiner": "Preserve the original ReCap advantage-conditioned idea while making it actionable.",
    "task_specialist": "Generate a task-specific natural-language control strategy.",
    "recovery_planner": "Generate a prompt that helps the policy recover from likely alignment or contact mistakes.",
    "motion_minimizer": "Generate a compact-motion prompt that reduces unnecessary movement and extra contacts.",
}


@dataclass(frozen=True)
class PromptContext:
    task: str
    round_index: int
    base_prompt: str
    history: list[dict[str, Any]]
    seed: int

    @property
    def previous_suffixes(self) -> set[str]:
        return {
            str(record.get("selected_suffix", "")).strip()
            for record in self.history
            if record.get("selected_suffix")
        }

    @property
    def last_success_rate(self) -> float | None:
        for record in reversed(self.history):
            value = record.get("success_rate")
            if isinstance(value, (int, float)):
                return float(value)
        return None

    @property
    def best_success_rate(self) -> float | None:
        rates = [
            float(record["success_rate"])
            for record in self.history
            if isinstance(record.get("success_rate"), (int, float))
        ]
        return max(rates) if rates else None


def _load_base_prompt(path: Path | None, fallback: str) -> str:
    if path is None:
        return fallback
    if not path.exists():
        return fallback
    try:
        import yaml

        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        prompt = data.get("prompt")
        if isinstance(prompt, str) and prompt.strip():
            return prompt.strip()
    except Exception:
        pass
    return fallback


def _load_history(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.exists():
        return []
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(record, dict):
            records.append(record)
    return records


def _normalize_suffix(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        raise ValueError("Prompt suffix cannot be empty.")
    if "\t" in text or "\n" in text:
        raise ValueError("Prompt suffix must be a single-line string.")
    if not text.endswith("."):
        text += "."
    return " " + text


def _candidate(
    ctx: PromptContext,
    agent_id: str,
    suffix: str,
    rationale: str,
    base_score: float,
) -> dict[str, Any]:
    suffix = _normalize_suffix(suffix)
    return {
        "agent_id": agent_id,
        "role": AGENT_ROLES.get(agent_id, "Generate an ACP prompt candidate."),
        "suffix": suffix,
        "prompt": ctx.base_prompt.rstrip() + suffix,
        "rationale": rationale,
        "base_score": base_score,
    }


def _task_specialist_suffix(ctx: PromptContext) -> tuple[str, str, float]:
    text = f"{ctx.task} {ctx.base_prompt}".lower()
    if "click" in text and "mouse" in text:
        return (
            "Move deliberately, align the end effector over the mouse button, then press once with steady downward motion",
            "The task text mentions a mouse click, so the prompt emphasizes alignment, single contact, and controlled pressure.",
            0.64,
        )
    if "hammer" in text or "nail" in text:
        return (
            "Align the tool with the nail head, stabilize the approach, and strike with a compact controlled motion",
            "The task appears contact-heavy, so the prompt focuses on tool alignment and stable impact.",
            0.62,
        )
    if "water" in text or "plant" in text:
        return (
            "Secure the watering object, move smoothly toward the plant, and pour only after the spout is positioned over the target",
            "The task appears to require object transport and pouring, so the prompt stresses grasp, positioning, and timing.",
            0.62,
        )
    if "open" in text or "drawer" in text or "door" in text:
        return (
            "Reach for the handle first, establish a firm contact, and pull along the opening direction without sideways drift",
            "The task appears to require handle interaction, so the prompt emphasizes contact and motion direction.",
            0.60,
        )
    return (
        "Identify the decisive contact point, approach it steadily, and complete the task with one controlled goal-directed motion",
        "No task-specific keyword dominated, so this agent generated a generic contact-and-control strategy.",
        0.56,
    )


def build_heuristic_candidates(ctx: PromptContext, agent_ids: list[str]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for agent_id in agent_ids:
        if agent_id == "advantage_refiner":
            candidates.append(
                _candidate(
                    ctx,
                    agent_id,
                    "Use the high-advantage successful strategy",
                    "This is the established fixed ACP suffix, kept as a stable anchor candidate.",
                    0.55,
                )
            )
        elif agent_id == "task_specialist":
            suffix, rationale, score = _task_specialist_suffix(ctx)
            candidates.append(
                _candidate(
                    ctx,
                    agent_id,
                    suffix,
                    rationale,
                    score,
                )
            )
        elif agent_id == "recovery_planner":
            last = ctx.last_success_rate
            suffix = (
                "If alignment is uncertain, slow down, re-center over the target, and avoid sweeping sideways before acting"
            )
            rationale = "This agent generates conservative recovery behavior for rounds where errors may come from misalignment."
            score = 0.50 + (0.10 if last is not None and last < 0.70 else 0.0)
            candidates.append(_candidate(ctx, agent_id, suffix, rationale, score))
        elif agent_id == "motion_minimizer":
            best = ctx.best_success_rate
            last = ctx.last_success_rate
            suffix = "Use a precise successful strategy with minimal unnecessary motion and no extra contacts"
            rationale = "This agent tries to reduce distribution shift by asking for compact, low-variance behavior."
            score = 0.52 + (0.08 if best is not None and last is not None and last < best else 0.0)
            candidates.append(_candidate(ctx, agent_id, suffix, rationale, score))
        else:
            raise ValueError(f"Unknown prompt agent id: {agent_id}")
    return candidates


def _history_for_llm(history: list[dict[str, Any]], max_records: int = 6) -> list[dict[str, Any]]:
    compact = []
    for record in history[-max_records:]:
        compact.append(
            {
                "round": record.get("round"),
                "selected_agent_id": record.get("selected_agent_id"),
                "selected_suffix": record.get("selected_suffix"),
                "success_rate": record.get("success_rate"),
                "eval_status": record.get("eval_status"),
            }
        )
    return compact


def _extract_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            raise
        parsed = json.loads(text[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("LLM prompt agent response must be a JSON object.")
    return parsed


def _llm_chat(messages: list[dict[str, str]]) -> str:
    api_key = os.environ.get("PROMPT_AGENT_OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("LLM prompt backend requires PROMPT_AGENT_OPENAI_API_KEY or OPENAI_API_KEY.")
    model = os.environ.get("PROMPT_AGENT_MODEL") or os.environ.get("OPENAI_MODEL")
    if not model:
        raise RuntimeError("LLM prompt backend requires PROMPT_AGENT_MODEL or OPENAI_MODEL.")

    base_url = os.environ.get("PROMPT_AGENT_OPENAI_BASE_URL") or os.environ.get("OPENAI_BASE_URL")
    if not base_url:
        base_url = "https://api.openai.com/v1"
    endpoint = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "temperature": float(os.environ.get("PROMPT_AGENT_TEMPERATURE", "0.7")),
        "response_format": {"type": "json_object"},
    }
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        endpoint,
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=float(os.environ.get("PROMPT_AGENT_TIMEOUT_S", "60"))) as response:
            raw = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"LLM prompt backend failed with HTTP {exc.code}: {detail}") from exc

    parsed = json.loads(raw)
    return parsed["choices"][0]["message"]["content"]


def _llm_candidate(ctx: PromptContext, agent_id: str) -> dict[str, Any]:
    if agent_id not in AGENT_ROLES:
        raise ValueError(f"Unknown prompt agent id: {agent_id}")
    system_prompt = (
        "You are a robotics ReCap prompt agent. Generate one short ACP suffix for a language-conditioned "
        "robot policy. The suffix will be appended to the base task prompt for high-advantage positive samples. "
        "Return only JSON with keys: suffix, rationale, confidence. The suffix must be one sentence, imperative, "
        "under 35 words, and must not mention training, ReCap, ACP, advantage, reward, success rate, or evaluation."
    )
    user_payload = {
        "agent_id": agent_id,
        "agent_role": AGENT_ROLES[agent_id],
        "task": ctx.task,
        "round": ctx.round_index,
        "base_prompt": ctx.base_prompt,
        "history": _history_for_llm(ctx.history),
    }
    content = _llm_chat(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(user_payload, sort_keys=True)},
        ]
    )
    parsed = _extract_json_object(content)
    suffix = str(parsed.get("suffix", ""))
    rationale = str(parsed.get("rationale", "Generated by the LLM prompt agent."))
    confidence = parsed.get("confidence", 0.6)
    try:
        base_score = max(0.0, min(1.0, float(confidence)))
    except (TypeError, ValueError):
        base_score = 0.6
    return _candidate(ctx, agent_id, suffix, rationale, base_score)


def build_candidates(
    ctx: PromptContext,
    agent_ids: list[str],
    backend: str,
    allow_heuristic_fallback: bool,
) -> list[dict[str, Any]]:
    if backend == "heuristic":
        return build_heuristic_candidates(ctx, agent_ids)
    if backend != "llm":
        raise ValueError(f"Unknown prompt backend: {backend}")

    candidates: list[dict[str, Any]] = []
    fallback_by_agent = {
        str(candidate["agent_id"]): candidate for candidate in build_heuristic_candidates(ctx, agent_ids)
    }
    for agent_id in agent_ids:
        try:
            candidate = _llm_candidate(ctx, agent_id)
            candidate["backend"] = "llm"
        except Exception as exc:
            if not allow_heuristic_fallback:
                raise
            candidate = dict(fallback_by_agent[agent_id])
            candidate["backend"] = "heuristic_fallback"
            candidate["fallback_reason"] = str(exc)
        candidates.append(candidate)
    return candidates


def _stable_jitter(ctx: PromptContext, agent_id: str) -> float:
    key = f"{ctx.seed}:{ctx.round_index}:{ctx.task}:{agent_id}".encode("utf-8")
    digest = hashlib.sha256(key).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF * 0.01


def select_candidate(ctx: PromptContext, candidates: list[dict[str, Any]], selection_mode: str) -> dict[str, Any]:
    if not candidates:
        raise ValueError("No prompt candidates generated.")

    if selection_mode == "rotate":
        return candidates[(ctx.round_index - 1) % len(candidates)]

    scored = []
    for candidate in candidates:
        suffix_key = str(candidate["suffix"]).strip()
        novelty_bonus = 0.03 if suffix_key not in ctx.previous_suffixes else -0.02
        score = float(candidate["base_score"]) + novelty_bonus + _stable_jitter(ctx, str(candidate["agent_id"]))
        scored.append((score, candidate))
    scored.sort(key=lambda item: item[0], reverse=True)
    selected = dict(scored[0][1])
    selected["manager_score"] = scored[0][0]
    return selected


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def generate(args: argparse.Namespace) -> None:
    base_prompt = _load_base_prompt(args.base_config, args.task.replace("_", " "))
    history = _load_history(args.history)
    agent_ids = args.agent_ids or list(DEFAULT_AGENT_IDS)
    ctx = PromptContext(
        task=args.task,
        round_index=args.round,
        base_prompt=base_prompt,
        history=history,
        seed=args.seed,
    )
    candidates = build_candidates(
        ctx,
        agent_ids=agent_ids,
        backend=args.backend,
        allow_heuristic_fallback=args.allow_heuristic_fallback,
    )
    selected = select_candidate(ctx, candidates, args.selection_mode)
    manifest = {
        "schema_version": 1,
        "task": args.task,
        "round": args.round,
        "backend": args.backend,
        "base_prompt": base_prompt,
        "selected_agent_id": selected["agent_id"],
        "selected_suffix": selected["suffix"],
        "selected_prompt": selected["prompt"],
        "selection_mode": args.selection_mode,
        "manager": {
            "history_records": len(history),
            "last_success_rate": ctx.last_success_rate,
            "best_success_rate": ctx.best_success_rate,
            "selected_score": selected.get("manager_score"),
        },
        "candidates": candidates,
    }
    _write_json(args.output, manifest)
    if args.suffix_output is not None:
        args.suffix_output.parent.mkdir(parents=True, exist_ok=True)
        args.suffix_output.write_text(str(manifest["selected_suffix"]), encoding="utf-8")
    print(
        json.dumps(
            {
                "manifest": str(args.output),
                "selected_agent_id": manifest["selected_agent_id"],
                "selected_suffix": manifest["selected_suffix"],
                "candidate_count": len(candidates),
            },
            sort_keys=True,
        )
    )


def _parse_int_or_none(value: str | None) -> int | None:
    if value is None or value == "NA":
        return None
    try:
        return int(value)
    except ValueError:
        return None


def record_result(args: argparse.Namespace) -> None:
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    successes = _parse_int_or_none(args.successes)
    episodes = _parse_int_or_none(args.episodes)
    success_rate = None
    if successes is not None and episodes is not None and episodes > 0:
        success_rate = successes / episodes

    record = {
        "task": manifest.get("task"),
        "round": manifest.get("round"),
        "mode": args.mode,
        "collect_seed": _parse_int_or_none(args.collect_seed),
        "checkpoint_step": args.checkpoint_step,
        "eval_status": args.eval_status,
        "successes": successes,
        "episodes": episodes,
        "success_rate": success_rate,
        "selected_agent_id": manifest.get("selected_agent_id"),
        "selected_suffix": manifest.get("selected_suffix"),
        "selected_prompt": manifest.get("selected_prompt"),
        "candidate_count": len(manifest.get("candidates", [])),
        "manifest": str(args.manifest),
    }
    args.history.parent.mkdir(parents=True, exist_ok=True)
    with args.history.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, sort_keys=True) + "\n")
    print(json.dumps(record, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")

    generate_parser = subparsers.add_parser("generate")
    generate_parser.add_argument("--task", default="click_mouse")
    generate_parser.add_argument("--round", type=int, required=True)
    generate_parser.add_argument("--base-config", type=Path)
    generate_parser.add_argument("--history", type=Path)
    generate_parser.add_argument("--output", type=Path, required=True)
    generate_parser.add_argument("--suffix-output", type=Path)
    generate_parser.add_argument("--agent-ids", nargs="*")
    generate_parser.add_argument("--backend", choices=("heuristic", "llm"), default="heuristic")
    generate_parser.add_argument("--allow-heuristic-fallback", action="store_true")
    generate_parser.add_argument("--selection-mode", choices=("score", "rotate"), default="score")
    generate_parser.add_argument("--seed", type=int, default=0)
    generate_parser.set_defaults(func=generate)

    record_parser = subparsers.add_parser("record-result")
    record_parser.add_argument("--manifest", type=Path, required=True)
    record_parser.add_argument("--history", type=Path, required=True)
    record_parser.add_argument("--mode", required=True)
    record_parser.add_argument("--collect-seed", required=True)
    record_parser.add_argument("--checkpoint-step", required=True)
    record_parser.add_argument("--eval-status", required=True)
    record_parser.add_argument("--successes", required=True)
    record_parser.add_argument("--episodes", required=True)
    record_parser.set_defaults(func=record_result)

    args = parser.parse_args()
    if not hasattr(args, "func"):
        parser.error("Expected a subcommand: generate or record-result.")
    args.func(args)


if __name__ == "__main__":
    main()

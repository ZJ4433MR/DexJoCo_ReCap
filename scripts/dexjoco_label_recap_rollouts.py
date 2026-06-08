#!/usr/bin/env python
"""Train a lightweight ReCap value model and label DexJoCo rollout NPZ files.

This script mirrors the Evo-RL ReCap annotation semantics for the DexJoCo/OpenPI
NPZ format used in this project:

1. compute normalized value targets from episode success and remaining length,
2. train a value function on observation/state/action frames,
3. infer values,
4. compute n-step advantages, and
5. binarize advantages into ACP indicators.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset


class RolloutValueDataset(Dataset):
    def __init__(
        self,
        base: np.ndarray,
        wrist: np.ndarray,
        state: np.ndarray,
        action: np.ndarray,
        target: np.ndarray,
    ) -> None:
        self.base = base
        self.wrist = wrist
        self.state = state.astype(np.float32, copy=False)
        self.action = action.astype(np.float32, copy=False)
        self.target = target.astype(np.float32, copy=False)

    def __len__(self) -> int:
        return int(self.target.shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "base": torch.from_numpy(self.base[index]),
            "wrist": torch.from_numpy(self.wrist[index]),
            "state": torch.from_numpy(self.state[index]),
            "action": torch.from_numpy(self.action[index]),
            "target": torch.tensor(self.target[index], dtype=torch.float32),
        }


class RecapValueNet(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, image_size: int) -> None:
        super().__init__()
        self.image_size = int(image_size)
        self.image_encoder = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=5, stride=2, padding=2),
            nn.SiLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.SiLU(),
            nn.Linear(64, 64),
            nn.SiLU(),
        )
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, 64),
            nn.SiLU(),
            nn.Linear(64, 64),
            nn.SiLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(64 + 64 + 64, 128),
            nn.SiLU(),
            nn.Linear(128, 1),
        )

    @staticmethod
    def _to_bchw(images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4:
            raise ValueError(f"Expected rank-4 images, got shape={tuple(images.shape)}")
        if images.shape[-1] in (1, 3, 4):
            images = images.permute(0, 3, 1, 2)
        elif images.shape[1] not in (1, 3, 4):
            raise ValueError(f"Cannot infer image channel axis for shape={tuple(images.shape)}")
        return images[:, :3].float().div(255.0)

    def forward(
        self,
        base: torch.Tensor,
        wrist: torch.Tensor,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        base = self._to_bchw(base)
        wrist = self._to_bchw(wrist)
        images = torch.cat([base, wrist], dim=1)
        if images.shape[-2:] != (self.image_size, self.image_size):
            images = F.interpolate(
                images,
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            )
        visual = self.image_encoder(images)
        state_feat = self.state_encoder(state.float())
        action_feat = self.action_encoder(action.float())
        raw = self.head(torch.cat([visual, state_feat, action_feat], dim=-1)).squeeze(-1)
        return -torch.sigmoid(raw)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _frame_indices_and_lengths(episode_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[int, int]]:
    frame_indices = np.zeros_like(episode_ids, dtype=np.int32)
    episode_lengths = np.zeros_like(episode_ids, dtype=np.int32)
    length_by_episode: dict[int, int] = {}

    for episode_id in np.unique(episode_ids):
        indices = np.flatnonzero(episode_ids == episode_id)
        length = int(indices.shape[0])
        length_by_episode[int(episode_id)] = length
        frame_indices[indices] = np.arange(length, dtype=np.int32)
        episode_lengths[indices] = length

    return frame_indices, episode_lengths, length_by_episode


def _compute_value_targets(
    episode_ids: np.ndarray,
    frame_indices: np.ndarray,
    episode_lengths: np.ndarray,
    is_success: np.ndarray,
    c_fail_coef: float,
) -> np.ndarray:
    task_max_length = int(np.max(episode_lengths))
    c_fail = float(task_max_length) * float(c_fail_coef)
    denom = float(task_max_length) + c_fail
    if denom <= 0:
        raise ValueError("Invalid value target denominator.")

    targets = np.zeros(episode_ids.shape[0], dtype=np.float32)
    for i in range(episode_ids.shape[0]):
        remaining_steps = int(episode_lengths[i]) - int(frame_indices[i]) - 1
        g = -float(remaining_steps)
        if not bool(is_success[i]):
            g -= c_fail
        targets[i] = float(np.clip(g / denom, -1.0, 0.0))
    return targets


def _compute_dense_rewards(
    targets: np.ndarray,
    episode_ids: np.ndarray,
    frame_indices: np.ndarray,
) -> np.ndarray:
    rewards = np.zeros_like(targets, dtype=np.float32)
    for i in range(targets.shape[0]):
        is_next_in_episode = (
            i + 1 < targets.shape[0]
            and episode_ids[i + 1] == episode_ids[i]
            and frame_indices[i + 1] == frame_indices[i] + 1
        )
        rewards[i] = float(targets[i] - targets[i + 1]) if is_next_in_episode else float(targets[i])
    return rewards


def _compute_n_step_advantages(
    rewards: np.ndarray,
    values: np.ndarray,
    episode_ids: np.ndarray,
    frame_indices: np.ndarray,
    n_step: int,
) -> np.ndarray:
    if n_step <= 0:
        raise ValueError("--n-step must be positive.")

    advantages = np.zeros_like(values, dtype=np.float32)
    n = int(values.shape[0])
    for i in range(n):
        ep_i = int(episode_ids[i])
        fi = int(frame_indices[i])
        discounted_sum = 0.0
        j = i
        steps = 0
        while steps < n_step and j < n:
            if int(episode_ids[j]) != ep_i or int(frame_indices[j]) != fi + steps:
                break
            discounted_sum += float(rewards[j])
            steps += 1
            j += 1

        if steps == n_step and j < n and int(episode_ids[j]) == ep_i and int(frame_indices[j]) == fi + n_step:
            bootstrap = float(values[j])
        else:
            bootstrap = 0.0
        advantages[i] = float(discounted_sum + bootstrap - float(values[i]))
    return advantages


def _binarize_advantages(
    advantages: np.ndarray,
    positive_ratio: float,
    exact_top_k: bool,
    candidate_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, float]:
    if not 0.0 < positive_ratio <= 1.0:
        raise ValueError("--positive-ratio must be within (0, 1].")

    if candidate_mask is None:
        candidate_indices = np.arange(advantages.shape[0], dtype=np.int64)
    else:
        candidate_mask = np.asarray(candidate_mask, dtype=np.bool_).reshape(-1)
        if candidate_mask.shape[0] != advantages.shape[0]:
            raise ValueError(
                f"candidate_mask length {candidate_mask.shape[0]} does not match advantages {advantages.shape[0]}"
            )
        candidate_indices = np.flatnonzero(candidate_mask)

    if candidate_indices.size == 0:
        raise ValueError("No frames are eligible for ACP positive labeling.")

    candidate_advantages = advantages[candidate_indices]
    indicators = np.zeros(advantages.shape[0], dtype=np.int64)

    if exact_top_k:
        k = max(1, int(round(float(candidate_indices.size) * positive_ratio)))
        order = np.argsort(-candidate_advantages)
        positive_indices = candidate_indices[order[:k]]
        indicators[positive_indices] = 1
        threshold = float(candidate_advantages[order[k - 1]])
        return indicators, threshold

    threshold = float(np.quantile(candidate_advantages, 1.0 - positive_ratio))
    indicators[candidate_indices] = (candidate_advantages >= threshold).astype(np.int64)
    return indicators, threshold


def _train_value_model(
    args: argparse.Namespace,
    base: np.ndarray,
    wrist: np.ndarray,
    state: np.ndarray,
    action: np.ndarray,
    targets: np.ndarray,
) -> tuple[RecapValueNet, dict[str, float | int | str]]:
    device = torch.device(args.device)
    model = RecapValueNet(
        state_dim=int(state.shape[-1]),
        action_dim=int(action.shape[-1]),
        image_size=args.image_size,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    dataset = RolloutValueDataset(base=base, wrist=wrist, state=state, action=action, target=targets)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )

    model.train()
    total_steps = 0
    last_loss = float("nan")
    for epoch in range(args.epochs):
        epoch_losses = []
        for batch in loader:
            total_steps += 1
            optimizer.zero_grad(set_to_none=True)
            pred = model(
                batch["base"].to(device, non_blocking=True),
                batch["wrist"].to(device, non_blocking=True),
                batch["state"].to(device, non_blocking=True),
                batch["action"].to(device, non_blocking=True),
            )
            target = batch["target"].to(device, non_blocking=True)
            loss = F.mse_loss(pred, target)
            loss.backward()
            optimizer.step()
            last_loss = float(loss.detach().cpu())
            epoch_losses.append(last_loss)
            if args.max_steps > 0 and total_steps >= args.max_steps:
                break
        mean_loss = float(np.mean(epoch_losses)) if epoch_losses else float("nan")
        print(f"[value] epoch={epoch + 1}/{args.epochs} mean_loss={mean_loss:.6f} steps={total_steps}")
        if args.max_steps > 0 and total_steps >= args.max_steps:
            break

    info: dict[str, float | int | str] = {
        "device": str(device),
        "epochs": int(args.epochs),
        "steps": int(total_steps),
        "last_loss": float(last_loss),
        "batch_size": int(args.batch_size),
        "lr": float(args.lr),
        "image_size": int(args.image_size),
    }
    return model, info


def _predict_values(
    model: RecapValueNet,
    args: argparse.Namespace,
    base: np.ndarray,
    wrist: np.ndarray,
    state: np.ndarray,
    action: np.ndarray,
    targets: np.ndarray,
) -> np.ndarray:
    device = torch.device(args.device)
    dataset = RolloutValueDataset(base=base, wrist=wrist, state=state, action=action, target=targets)
    loader = DataLoader(
        dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )

    predictions = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            pred = model(
                batch["base"].to(device, non_blocking=True),
                batch["wrist"].to(device, non_blocking=True),
                batch["state"].to(device, non_blocking=True),
                batch["action"].to(device, non_blocking=True),
            )
            predictions.append(pred.detach().cpu().numpy().astype(np.float32))
    return np.concatenate(predictions, axis=0)


def label_rollouts(args: argparse.Namespace) -> None:
    _set_seed(args.seed)
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    with np.load(args.input, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}

    required = ["base", "wrist", "state", "action", "episode_id", "is_success"]
    missing = [key for key in required if key not in arrays]
    if missing:
        raise KeyError(f"Missing required NPZ fields: {missing}")

    base = arrays["base"]
    wrist = arrays["wrist"]
    state = arrays["state"].astype(np.float32, copy=False)
    action = arrays["action"].astype(np.float32, copy=False)
    episode_ids = arrays["episode_id"].astype(np.int32, copy=False)
    is_success = arrays["is_success"].astype(np.bool_, copy=False)

    frame_indices, episode_lengths, length_by_episode = _frame_indices_and_lengths(episode_ids)
    targets = _compute_value_targets(
        episode_ids=episode_ids,
        frame_indices=frame_indices,
        episode_lengths=episode_lengths,
        is_success=is_success,
        c_fail_coef=args.c_fail_coef,
    )
    rewards = _compute_dense_rewards(targets, episode_ids, frame_indices)
    model, train_info = _train_value_model(args, base, wrist, state, action, targets)
    values = _predict_values(model, args, base, wrist, state, action, targets)
    advantages = _compute_n_step_advantages(
        rewards=rewards,
        values=values,
        episode_ids=episode_ids,
        frame_indices=frame_indices,
        n_step=args.n_step,
    )
    positive_candidate_mask = is_success if args.positive_success_only else None
    indicators, threshold = _binarize_advantages(
        advantages=advantages,
        positive_ratio=args.positive_ratio,
        exact_top_k=args.exact_top_k,
        candidate_mask=positive_candidate_mask,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.model_output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "state_dim": int(state.shape[-1]),
            "action_dim": int(action.shape[-1]),
            "image_size": int(args.image_size),
            "train_info": train_info,
        },
        args.model_output,
    )

    arrays.update(
        {
            "frame_index": frame_indices.astype(np.int32),
            "episode_length": episode_lengths.astype(np.int32),
            "value_target": targets.astype(np.float32),
            "dense_reward": rewards.astype(np.float32),
            "value": values.astype(np.float32),
            "advantage": advantages.astype(np.float32),
            "acp_indicator": indicators.astype(np.int64),
            "task_index": np.zeros_like(indicators, dtype=np.int64),
            "recap_n_step": np.asarray(args.n_step, dtype=np.int32),
            "recap_positive_ratio": np.asarray(args.positive_ratio, dtype=np.float32),
            "recap_positive_threshold": np.asarray(threshold, dtype=np.float32),
            "recap_positive_success_only": np.asarray(args.positive_success_only, dtype=np.bool_),
        }
    )
    np.savez_compressed(args.output, **arrays)

    success_by_episode = []
    for episode_id in sorted(length_by_episode):
        indices = np.flatnonzero(episode_ids == episode_id)
        success_by_episode.append(bool(is_success[indices[0]]))

    summary = {
        "input": str(args.input),
        "output": str(args.output),
        "model_output": str(args.model_output),
        "frames": int(values.shape[0]),
        "episodes": int(len(length_by_episode)),
        "successful_episodes": int(sum(success_by_episode)),
        "success_rate_saved_episodes": float(np.mean(success_by_episode)) if success_by_episode else 0.0,
        "target_min": float(np.min(targets)),
        "target_max": float(np.max(targets)),
        "value_min": float(np.min(values)),
        "value_max": float(np.max(values)),
        "value_mse": float(np.mean(np.square(values - targets))),
        "advantage_min": float(np.min(advantages)),
        "advantage_max": float(np.max(advantages)),
        "advantage_mean": float(np.mean(advantages)),
        "threshold": float(threshold),
        "positive_success_only": bool(args.positive_success_only),
        "indicator_candidate_count": int(np.sum(is_success)) if args.positive_success_only else int(values.shape[0]),
        "indicator_candidate_positive_ratio": (
            float(np.sum(indicators) / max(1, int(np.sum(is_success))))
            if args.positive_success_only
            else float(np.mean(indicators.astype(np.float32)))
        ),
        "indicator_positive_ratio": float(np.mean(indicators.astype(np.float32))),
        "indicator_positive_count": int(np.sum(indicators)),
        "n_step": int(args.n_step),
        "positive_ratio_target": float(args.positive_ratio),
        "c_fail_coef": float(args.c_fail_coef),
        "exact_top_k": bool(args.exact_top_k),
        "train_info": train_info,
    }
    summary_path = args.summary_output or args.output.with_suffix(".summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"[value] wrote labeled rollout dataset: {args.output}")
    print(f"[value] wrote value model: {args.model_output}")
    print(f"[value] wrote summary: {summary_path}")
    print(json.dumps(summary, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--image-size", type=int, default=96)
    parser.add_argument("--n-step", type=int, default=50)
    parser.add_argument("--positive-ratio", type=float, default=0.3)
    parser.add_argument("--c-fail-coef", type=float, default=1.0)
    parser.add_argument("--exact-top-k", action="store_true")
    parser.add_argument("--positive-success-only", action="store_true")
    label_rollouts(parser.parse_args())


if __name__ == "__main__":
    main()

#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import random
from collections.abc import Callable
from typing import Any

import torch

from lerobot.configs.train import ACPConfig
from lerobot.rl.acp_tags import build_acp_tagged_task


def _parse_tag_values(raw: str) -> list[str]:
    tag_values = [value.strip() for value in raw.split(",") if value.strip()]
    if not tag_values:
        raise ValueError("ACP tag_values must contain at least one comma-separated value.")
    return tag_values


def _extract_tag_indices(values: Any, batch_size: int, num_tags: int) -> list[int]:
    if not isinstance(values, torch.Tensor):
        raise TypeError("ACP indicator must be a torch.Tensor.")

    if values.dtype == torch.bool or values.dtype.is_floating_point:
        raise TypeError("ACP indicator must be an integer tensor.")

    if values.ndim != 1:
        raise TypeError(f"ACP indicator tensor must be 1D, got shape={tuple(values.shape)}.")

    if values.shape[0] != batch_size:
        raise ValueError(f"ACP batch size mismatch: expected {batch_size}, got {values.shape[0]}.")

    parsed = [int(v) for v in values.detach().cpu().tolist()]
    if any(v < 0 or v >= num_tags for v in parsed):
        bad = [v for v in parsed if v < 0 or v >= num_tags][0]
        raise ValueError(f"ACP indicator must be in [0, {num_tags - 1}], got {bad}.")
    return parsed


class ACPPromptHook:
    def __init__(self, cfg: ACPConfig, seed: int | None):
        self.indicator_field = cfg.indicator_field
        self.dropout = cfg.indicator_dropout_prob
        self.tag_key = cfg.tag_key
        self.tag_values = _parse_tag_values(cfg.tag_values)
        self.rng = random.Random(seed if seed is not None else 0)

    def _resolve_tag_indices(self, batch: dict[str, Any], batch_size: int) -> list[int]:
        if self.indicator_field not in batch:
            raise KeyError(f"ACP indicator field '{self.indicator_field}' is missing from batch.")
        return _extract_tag_indices(batch[self.indicator_field], batch_size, len(self.tag_values))

    def __call__(self, batch: Any, _: int) -> Any:
        if not isinstance(batch, dict):
            raise TypeError(f"ACP batch must be dict, got {type(batch).__name__}.")
        if "task" not in batch:
            raise KeyError("ACP requires 'task' in batch.")

        tasks = batch["task"]
        if not isinstance(tasks, list):
            raise TypeError(f"ACP batch['task'] must be list[str], got {type(tasks).__name__}.")
        if any(not isinstance(task, str) for task in tasks):
            raise TypeError("ACP batch['task'] must be list[str].")

        tag_indices = self._resolve_tag_indices(batch, len(tasks))

        conditioned_tasks: list[str] = []
        for task, tag_idx in zip(tasks, tag_indices, strict=True):
            if self.dropout > 0.0 and self.rng.random() < self.dropout:
                conditioned_tasks.append(task)
                continue
            conditioned_tasks.append(
                build_acp_tagged_task(task, tag_value=self.tag_values[tag_idx], tag_key=self.tag_key)
            )
        batch["task"] = conditioned_tasks
        return batch


def build_acp_raw_batch_hook(cfg: ACPConfig, seed: int | None) -> Callable[[Any, int], Any] | None:
    if not cfg.enable:
        return None
    return ACPPromptHook(cfg, seed)

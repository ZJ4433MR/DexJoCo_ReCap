#!/usr/bin/env python
"""Collect DexJoCo/OpenPI rollouts into a compact ReCap NPZ dataset."""

from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path
import queue

import numpy as np
import yaml
from openpi_client import websocket_client_policy

from dexjoco_openpi_client.dexjoco_openpi_env import DexJoCoOpenPIEnv
from dexjoco_openpi_client.eval_dexjoco_openpi import (
    ActionChunk,
    _set_seed,
    receive_actions,
)


def _enqueue_action_chunk(actions_buffer, action_chunk: np.ndarray, timestamp: int, dual_arm: bool) -> None:
    action_queue: queue.Queue = queue.Queue()
    action_chunk = np.asarray(action_chunk, dtype=np.float32).copy()
    action_queue.put(ActionChunk(action=action_chunk, timestamp=timestamp))
    receive_actions(action_queue, actions_buffer, timestamp, dual_arm)


def _initial_click_mouse_alignment(env: DexJoCoOpenPIEnv) -> None:
    for _ in range(30):
        env.step(
            action=np.array(
                [
                    -4.4294e-01,
                    1.3729e-06,
                    1.5170e00,
                    -3.14156462e00,
                    -6.91584035e-05,
                    -1.40317984e-03,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0.263,
                    0,
                    0,
                    0,
                ],
                dtype=np.float32,
            )
        )


def collect(args: argparse.Namespace) -> None:
    _set_seed(args.seed)

    with args.config.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    env_name = cfg["env_name"]
    camera_mapping = cfg["camera_mapping"]
    base_prompt = cfg["prompt"].rstrip()
    acp_prompt = base_prompt + args.acp_suffix
    prompt = acp_prompt if args.collect_prompt_mode == "acp" else base_prompt
    dual_arm = cfg["robot_type"] == "dual_arm"

    env = DexJoCoOpenPIEnv(
        env_name=env_name,
        camera_mapping=camera_mapping,
        seed=args.seed,
        rand_full=args.rand_full,
        randomize_dynamics=args.randomize_dynamics,
        dual_arm=dual_arm,
        prompt=prompt,
        render_mode="rgb_array",
        password=cfg.get("password"),
    )
    client = websocket_client_policy.WebsocketClientPolicy(host=args.host, port=args.port)

    saved_episodes = []
    episode_success_flags = []
    action_horizon = args.action_horizon

    try:
        env.start()

        for ep in range(args.episodes):
            print(f"[collect] episode {ep + 1}/{args.episodes}")
            env.reset()
            if env_name == "click_mouse":
                _initial_click_mouse_alignment(env)

            timestamp = 0
            actions_buffer = deque()
            first_result = client.infer(env.get_obs())
            _enqueue_action_chunk(actions_buffer, first_result["actions"], timestamp, dual_arm)

            frames = []
            while timestamp < args.max_steps:
                if not actions_buffer:
                    result = client.infer(env.get_obs())
                    _enqueue_action_chunk(actions_buffer, result["actions"], timestamp, dual_arm)

                obs_before = env.get_obs()
                action = actions_buffer.popleft().action.astype(np.float32, copy=False)
                frames.append(
                    {
                        "base": obs_before["base"].copy(),
                        "wrist": obs_before["wrist"].copy(),
                        "state": obs_before["state"].astype(np.float32, copy=True),
                        "action": action.copy(),
                    }
                )

                env.step(action)
                timestamp += 1

                if env.is_done:
                    break

                if len(actions_buffer) < args.replan_ratio * action_horizon:
                    result = client.infer(env.get_obs())
                    _enqueue_action_chunk(actions_buffer, result["actions"], timestamp, dual_arm)

            episode_success_flags.append(bool(env.is_success))
            print(f"[collect] episode {ep + 1} success={env.is_success} steps={len(frames)}")
            if frames and (env.is_success or args.include_failures):
                saved_episodes.append((frames, bool(env.is_success)))

    finally:
        env.close()

    success_count = sum(1 for _, is_success in saved_episodes if is_success)
    if success_count == 0:
        raise RuntimeError("No successful episodes collected; cannot build ReCap dataset.")

    base_frames = []
    wrist_frames = []
    states = []
    actions = []
    episode_ids = []
    frame_success = []
    for episode_id, (frames, is_success) in enumerate(saved_episodes):
        for frame in frames:
            base_frames.append(frame["base"])
            wrist_frames.append(frame["wrist"])
            states.append(frame["state"])
            actions.append(frame["action"])
            episode_ids.append(episode_id)
            frame_success.append(is_success)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        base=np.asarray(base_frames, dtype=np.uint8),
        wrist=np.asarray(wrist_frames, dtype=np.uint8),
        state=np.asarray(states, dtype=np.float32),
        action=np.asarray(actions, dtype=np.float32),
        episode_id=np.asarray(episode_ids, dtype=np.int32),
        is_success=np.asarray(frame_success, dtype=np.bool_),
        prompt=np.asarray(acp_prompt),
        base_prompt=np.asarray(base_prompt),
        acp_prompt=np.asarray(acp_prompt),
        collection_prompt=np.asarray(prompt),
        total_episodes=np.asarray(args.episodes, dtype=np.int32),
        saved_episodes=np.asarray(len(saved_episodes), dtype=np.int32),
        successful_episodes=np.asarray(success_count, dtype=np.int32),
        episode_success_flags=np.asarray(episode_success_flags, dtype=np.bool_),
    )

    summary = args.output.with_suffix(".summary.txt")
    summary.write_text(
        "\n".join(
            [
                f"total_episodes={args.episodes}",
                f"saved_episodes={len(saved_episodes)}",
                f"successful_episodes={success_count}",
                f"total_frames={len(actions)}",
                f"collection_prompt={prompt}",
                f"base_prompt={base_prompt}",
                f"acp_prompt={acp_prompt}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"[collect] wrote {args.output}")
    print(summary.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--replan-ratio", type=float, default=0.8)
    parser.add_argument("--action-horizon", type=int, default=30)
    parser.add_argument("--acp-suffix", default=" Use the high-advantage successful strategy.")
    parser.add_argument("--include-failures", action="store_true")
    parser.add_argument("--collect-prompt-mode", choices=("base", "acp"), default="acp")
    parser.add_argument("--rand-full", action="store_true")
    parser.add_argument("--randomize-dynamics", action="store_true")
    collect(parser.parse_args())


if __name__ == "__main__":
    main()

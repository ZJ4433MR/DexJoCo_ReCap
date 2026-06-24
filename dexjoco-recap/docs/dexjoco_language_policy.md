# DexJoCo Language-Conditioned Policy Stage

The DexJoCo stage moves the ReCap reproduction from prompt-free control
baselines to a language-conditioned VLA setting.

The main target used in this workspace is single-arm `click_mouse` with the
OpenPI/pi0.5 policy interface. This setting is closer to the ReCap prompt
conditioning idea because the policy input includes:

- RGB observations from base and wrist cameras.
- Proprioceptive state.
- A task prompt from the DexJoCo task config.

For ACP runs, the prompt receives an additional tag such as:

```text
Advantage: positive
```

or, in multi-tag experiments:

```text
Advantage: failure
Advantage: low
Advantage: medium
Advantage: high
```

## Implemented Path

The current faithful LeRobot-format path is:

1. Download or prepare the official DexJoCo LeRobot-format task dataset.
2. Mark initial demonstration episodes with `episode_success`.
3. Merge the current data pool.
4. Train a Pistar06 value model with the PyTorch/LeRobot-compatible source.
5. Infer value, n-step advantage, and ACP indicators back into the dataset.
6. Patch OpenPI at runtime to inject ACP prompt tags from the indicator field.
7. Fine-tune the pi0.5 policy with JAX/OpenPI.
8. Roll out the updated policy in DexJoCo and append new data.
9. Repeat for multiple rounds, then evaluate the final checkpoint.

The orchestration entry point is:

```text
jobs/57_dexjoco_click_mouse_evorl_lerobot_ab.sh
```

The multi-tag variant is:

```text
jobs/68_dexjoco_click_mouse_evorl_lerobot_E_multitag_episode_smooth.sh
```

## Why `click_mouse`

Early single-arm evaluation showed `click_mouse` was a practical target for the
first ReCap reproduction because the public pi0.5 baseline is non-zero but not
saturated. That leaves room to test whether value-derived ACP labels and prompt
conditioning change performance.

## Data Formats

Two data formats are supported:

- Compact DexJoCo rollout NPZ files for lightweight local value-labeling.
- LeRobot-format datasets for the higher-fidelity Pistar06 value stack.

The LeRobot path is preferred for new experiments because it matches the value
training/inference pipeline more closely and makes it easier to swap in
real-robot data later.

## References

- DexJoCo: https://github.com/brave-eai/dexjoco
- DexJoCo pi0.5 checkpoints: https://huggingface.co/DexJoCo/DexJoCo-Pi05
- DexJoCo LeRobot datasets: https://huggingface.co/datasets/DexJoCo/DexJoCo-Datasets-LeRobot

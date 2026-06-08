# DexJoCo Language Policy Stage

## Objective

Move the ReCap simulation from PuSH-T to a language-conditioned VLA setting.
The initial smoke target was DexJoCo single-arm `water_plant`; after the
single-arm baseline matrix, the first ReCap target is `click_mouse` with the
OpenPI/pi0.5 policy interface.

This is closer to the pi0.6 ReCap setting than PuSH-T because the policy input
contains:

- RGB observations: `base` and `wrist`.
- Proprioceptive state: 23 dimensions for single-arm tasks.
- Language prompt: `Grasp the watering can and apply water to the plant.`

The prompt is passed into `DexJoCoOpenPIEnv.get_obs()` and then consumed by the
OpenPI websocket policy server.

For the current `click_mouse` ReCap run, the task prompt comes from
`configs/rand_obj/click_mouse.yaml`, with an ACP suffix appended only for
high-advantage frames and ACP evaluation.

## Why `water_plant` first

`water_plant` is single-arm, so the action dimension is 22 instead of 44 for
bimanual tasks. It still requires dexterous tool use and has a clear natural
language instruction, making it a good first DexJoCo task before moving to
bimanual settings.

## Jobs

Headless environment smoke test:

```powershell
.\scripts\run_remote_l40_slurm.ps1 `
  -ConfigPath configs\remote-l40.env `
  -LocalDexJoCoPath .tmp\dexjoco-src `
  -Job jobs/20_dexjoco_headless_smoke.sh `
  -RunName dexjoco_headless_smoke_l40 `
  -Time 02:00:00
```

OpenPI/pi0.5 language-conditioned policy evaluation:

```powershell
.\scripts\run_remote_l40_slurm.ps1 `
  -ConfigPath configs\remote-l40.env `
  -LocalDexJoCoPath .tmp\dexjoco-src `
  -Job jobs/21_dexjoco_pi05_water_plant_eval.sh `
  -RunName dexjoco_pi05_water_plant_eval_l40 `
  -Time 06:00:00 `
  -Memory 96G
```

The eval job defaults to 3 episodes. Edit `DEXJOCO_EVAL_EPISODES` in the job
script before launch if a longer evaluation is needed.

Single-arm task matrix, used to find an easier non-zero baseline before ReCap:

```powershell
.\scripts\run_remote_l40_slurm.ps1 `
  -ConfigPath configs\remote-l40.env `
  -LocalDexJoCoPath .tmp\dexjoco-src `
  -Job jobs/22_dexjoco_pi05_single_arm_matrix.sh `
  -RunName dexjoco_pi05_single_arm_matrix_l40 `
  -Time 08:00:00 `
  -Memory 96G
```

The matrix defaults to `click_mouse hammer_nail` with 3 episodes per task.
The runner treats command-line `-Time` and `-Memory` as higher priority than
values in `configs/remote-l40.env`, so longer matrix jobs can override the
default local config without editing private settings.

Stable 20-episode single-arm baseline:

```powershell
.\scripts\run_remote_l40_slurm.ps1 `
  -ConfigPath configs\remote-l40.env `
  -LocalDexJoCoPath .tmp\dexjoco-src `
  -Job jobs/23_dexjoco_pi05_single_arm_eval20.sh `
  -RunName dexjoco_pi05_single_arm_eval20_l40 `
  -Time 08:00:00 `
  -Memory 96G
```

`-LocalDexJoCoPath` is optional when the L40 server can clone GitHub directly.
It is useful on this cluster because compute-node GitHub access has been
unstable. The fallback source is copied only into the temporary run archive and
is not committed to Git.

Full DexJoCo ReCap value/advantage run:

```powershell
.\scripts\run_remote_l40_slurm.ps1 `
  -ConfigPath configs\remote-l40.env `
  -LocalDexJoCoPath .tmp\dexjoco-src `
  -Job jobs/28_dexjoco_click_mouse_recap_full_value_advantage.sh `
  -RunName dexjoco_click_mouse_recap_full_value_advantage_l40 `
  -Time 12:00:00 `
  -Memory 96G
```

This job collects base-prompt public pi0.5 rollouts, trains a lightweight
DexJoCo value model, writes `value`, `value_target`, `dense_reward`,
`advantage`, and `acp_indicator` fields into the rollout NPZ, then fine-tunes
OpenPI with ACP prompt tags derived from `acp_indicator`.

## Current runs

- `dexjoco_headless_smoke_l40_v4`: passed. The `water_plant` observation
  contained the expected prompt, 23-dimensional state, and two 224x224 RGB
  image inputs.
- `dexjoco_pi05_water_plant_eval_l40_v1`: completed with the public
  OpenPI/pi0.5 `water_plant` checkpoint. The 3-episode smoke result was `0/3`
  success, so it validates the language-policy plumbing rather than final task
  performance.
- `dexjoco_pi05_single_arm_matrix_l40_v3`: completed with websocket ping
  timeout disabled during long JAX inference. `click_mouse` reached `2/3`
  success and `hammer_nail` reached `3/3`.
- `dexjoco_pi05_single_arm_eval20_l40_v1`: completed the larger baseline.
  `click_mouse` reached `13/20` success and `hammer_nail` reached `15/20`.
  Use `click_mouse` as the first ReCap target.

## Storage policy

The job scripts clone DexJoCo and create conda environments under the per-run
temporary directory on the L40 node. The remote runner removes that directory
after pulling results back to local `runs/<run_name>/`.

The GitHub repository stores only scripts, docs, and small metadata. It does not
store DexJoCo source, checkpoints, datasets, videos, or run outputs.

## ReCap integration path

The implemented DexJoCo NPZ path now reproduces the ReCap loop with local
rollouts:

1. Load the public OpenPI/pi0.5 checkpoint for `click_mouse`.
2. Roll out the baseline in DexJoCo and collect both successful and failed
   trajectories.
3. Train a value model on RGB observations, proprioceptive state, action, and
   normalized outcome labels.
4. Infer value, dense reward, n-step advantage, and binary ACP indicators for
   rollout frames.
5. Convert positive indicators into ACP language prompt tags.
6. Fine-tune the OpenPI policy and evaluate under the ACP prompt.

Unlike PuSH-T, this stage can use actual prompt conditioning. The ACP prompt tag
should therefore be injected into the task prompt rather than approximated only
through positive-sample weighting.

The next higher-fidelity step is to replace the compact NPZ rollout dataset
with the official `DexJoCo/DexJoCo-Datasets-LeRobot` format and run Evo-RL's
`pistar06` value stack directly.

## References

- DexJoCo: https://github.com/brave-eai/dexjoco
- DexJoCo pi0.5 checkpoints: https://huggingface.co/DexJoCo/DexJoCo-Pi05
- DexJoCo LeRobot datasets: https://huggingface.co/datasets/DexJoCo/DexJoCo-Datasets-LeRobot

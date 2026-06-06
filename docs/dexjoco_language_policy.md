# DexJoCo Language Policy Stage

## Objective

Move the ReCap simulation from PuSH-T to a language-conditioned VLA setting.
The first target is DexJoCo single-arm `water_plant` with the OpenPI/pi0.5
policy interface.

This is closer to the pi0.6 ReCap setting than PuSH-T because the policy input
contains:

- RGB observations: `base` and `wrist`.
- Proprioceptive state: 23 dimensions for single-arm tasks.
- Language prompt: `Grasp the watering can and apply water to the plant.`

The prompt is passed into `DexJoCoOpenPIEnv.get_obs()` and then consumed by the
OpenPI websocket policy server.

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

`-LocalDexJoCoPath` is optional when the L40 server can clone GitHub directly.
It is useful on this cluster because compute-node GitHub access has been
unstable. The fallback source is copied only into the temporary run archive and
is not committed to Git.

## Current runs

- `dexjoco_headless_smoke_l40_v4`: passed. The `water_plant` observation
  contained the expected prompt, 23-dimensional state, and two 224x224 RGB
  image inputs.
- `dexjoco_pi05_water_plant_eval_l40_v1`: completed with the public
  OpenPI/pi0.5 `water_plant` checkpoint. The 3-episode smoke result was `0/3`
  success, so it validates the language-policy plumbing rather than final task
  performance.

## Storage policy

The job scripts clone DexJoCo and create conda environments under the per-run
temporary directory on the L40 node. The remote runner removes that directory
after pulling results back to local `runs/<run_name>/`.

The GitHub repository stores only scripts, docs, and small metadata. It does not
store DexJoCo source, checkpoints, datasets, videos, or run outputs.

## ReCap integration path

After the pretrained pi0.5 eval works, the ReCap stage should use the DexJoCo
LeRobot dataset and the same language-conditioned OpenPI training code:

1. Download a task dataset such as `water_plant` from
   `DexJoCo/DexJoCo-Datasets-LeRobot`.
2. Train or load a BC/OpenPI baseline for the task.
3. Roll out the baseline in DexJoCo and collect success/reward traces.
4. Train a value model on observations, prompt, action, and outcome labels.
5. Infer value and n-step advantage labels for dataset frames.
6. Convert advantage labels into language prompt tags for ACP training.
7. Fine-tune the OpenPI policy and evaluate BC vs ReCap under the same seeds.

Unlike PuSH-T, this stage can use actual prompt conditioning. The ACP prompt tag
should therefore be injected into the task prompt rather than approximated only
through positive-sample weighting.

## References

- DexJoCo: https://github.com/brave-eai/dexjoco
- DexJoCo pi0.5 checkpoints: https://huggingface.co/DexJoCo/DexJoCo-Pi05
- DexJoCo LeRobot datasets: https://huggingface.co/datasets/DexJoCo/DexJoCo-Datasets-LeRobot

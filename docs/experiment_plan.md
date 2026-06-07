# Experiment Plan

## Goal

Reproduce the core RECAP mechanism in simulation:

```text
demonstrations / rollouts
  -> value training
  -> value inference and n-step advantage
  -> binary ACP indicator
  -> advantage-conditioned policy training
  -> simulation evaluation
```

## Minimal comparisons

| Method | Data | Conditioning | Metric |
| --- | --- | --- | --- |
| BC baseline | public/sim demos | none | success rate |
| RECAP/ACP | same data + value labels | Advantage: positive/negative | success rate |

## First milestone

Run `jobs/00_remote_smoke.sh` on L40 and confirm:

- CUDA is visible.
- Evo-RL imports.
- ACP/value unit tests pass.
- Remote temporary directory is cleaned.
- Results are pulled back to local `runs/`.

## Second milestone

Run a small PuSH-T BC vs RECAP/ACP comparison with simulation eval:

```powershell
.\scripts\run_remote_l40_slurm.ps1 `
  -ConfigPath configs\remote-l40.env `
  -Job jobs/12_pusht_recap_compare_eval.sh `
  -RunName pusht_recap_compare_eval_l40
```

Default comparison settings:

```bash
DATASET_EPISODES=[0,1,2,3,4,5,6,7,8,9]
POLICY_STEPS=1000
VALUE_STEPS=300
EVAL_EPISODES=20
```

## Notes

PuSH-T does not provide `q01`/`q99` quantile stats, so the local simulation jobs
default the Pistar06 value model state normalization to `MEAN_STD`. Override
`VALUE_NORMALIZATION_MAPPING` for datasets that include quantile stats.
The PuSH-T value-model camera feature defaults to `observation.image`; override
`VALUE_CAMERA_FEATURES` for datasets with different image keys.

After the short comparison run, launch the small formal run with non-random value
backbones:

```powershell
.\scripts\run_remote_l40_slurm.ps1 `
  -ConfigPath configs\remote-l40.env `
  -Job jobs/13_pusht_recap_formal_small.sh `
  -RunName pusht_recap_formal_small_l40
```

This validates the algorithmic pipeline, not official pi*0.6 scale or official
pi0.6 weights.

## Current result snapshot

See `docs/results.md` for the stronger PuSH-T diffusion run where full-data BC
reached 50.0% success and ReCap/ACP weighted fine-tuning reached 62.0% success
over 100 evaluation episodes. The earlier 100-demo run reached 28.0% BC and
34.0% ReCap success over 50 evaluation episodes.

## DexJoCo language-policy milestone

The next stage moves from PuSH-T to DexJoCo with an actual language-conditioned
OpenPI/pi0.5 policy. Start with the single-arm `water_plant` task:

```powershell
.\scripts\run_remote_l40_slurm.ps1 `
  -ConfigPath configs\remote-l40.env `
  -LocalDexJoCoPath .tmp\dexjoco-src `
  -Job jobs/20_dexjoco_headless_smoke.sh `
  -RunName dexjoco_headless_smoke_l40 `
  -Time 02:00:00
```

Then evaluate the pretrained pi0.5 checkpoint:

```powershell
.\scripts\run_remote_l40_slurm.ps1 `
  -ConfigPath configs\remote-l40.env `
  -LocalDexJoCoPath .tmp\dexjoco-src `
  -Job jobs/21_dexjoco_pi05_water_plant_eval.sh `
  -RunName dexjoco_pi05_water_plant_eval_l40 `
  -Time 06:00:00 `
  -Memory 96G
```

Details are in `docs/dexjoco_language_policy.md`.

If `water_plant` remains too hard in short evals, run the single-arm matrix to
find a non-zero baseline:

```powershell
.\scripts\run_remote_l40_slurm.ps1 `
  -ConfigPath configs\remote-l40.env `
  -LocalDexJoCoPath .tmp\dexjoco-src `
  -Job jobs/22_dexjoco_pi05_single_arm_matrix.sh `
  -RunName dexjoco_pi05_single_arm_matrix_l40 `
  -Time 08:00:00 `
  -Memory 96G
```

After a short non-zero baseline is found, run the 20-episode version:

```powershell
.\scripts\run_remote_l40_slurm.ps1 `
  -ConfigPath configs\remote-l40.env `
  -LocalDexJoCoPath .tmp\dexjoco-src `
  -Job jobs/23_dexjoco_pi05_single_arm_eval20.sh `
  -RunName dexjoco_pi05_single_arm_eval20_l40 `
  -Time 08:00:00 `
  -Memory 96G
```

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

Run a small PushT template experiment with reduced steps:

```powershell
.\scripts\run_remote_l40_slurm.ps1 `
  -ConfigPath configs\remote-l40.env `
  -Job jobs/10_pusht_recap_template.sh `
  -RunName pusht_recap_small
```

Before launching, reduce steps in `jobs/10_pusht_recap_template.sh` if needed:

```bash
POLICY_STEPS=1000
VALUE_STEPS=500
```

## Notes

PuSH-T does not provide `q01`/`q99` quantile stats, so the local simulation jobs
default the Pistar06 value model state normalization to `MEAN_STD`. Override
`VALUE_NORMALIZATION_MAPPING` for datasets that include quantile stats.
The PuSH-T value-model camera feature defaults to `observation.image`; override
`VALUE_CAMERA_FEATURES` for datasets with different image keys.

This validates the algorithmic pipeline, not official pi*0.6 scale or official
pi0.6 weights.

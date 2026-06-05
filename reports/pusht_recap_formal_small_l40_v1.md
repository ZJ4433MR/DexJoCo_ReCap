# PuSH-T RECAP Formal Small L40 v1

Date: 2026-06-05

Run directory:

`runs/pusht_recap_formal_small_l40_v1`

## Setup

- Dataset: `lerobot/pusht`
- Training episodes: 50 (`0..49`)
- Policy: ACT
- Baseline policy steps: 10,000
- Value model steps: 2,000
- RECAP policy steps: 10,000
- Eval episodes: 50 for BC, 50 for RECAP
- GPU: NVIDIA L40 via Slurm job `1617353`
- Slurm status: `COMPLETED`, exit code `0:0`, elapsed `00:51:44`

## RECAP Selection

- ACP field: `complementary_info.acp_indicator_pusht_recap_formal_small`
- Positive samples: 1,871
- Total samples: 6,235
- Positive ratio: 0.300080

## Eval Results

| Method | pc_success | avg_sum_reward | avg_max_reward | n_episodes |
| --- | ---: | ---: | ---: | ---: |
| BC | 0.0 | 23.954386691145192 | 0.21177020498407717 | 50 |
| RECAP | 0.0 | 33.043322291919324 | 0.17523222465438298 | 50 |

## Interpretation

This run validates that the local-to-L40 workflow and the Evo-RL RECAP pipeline can run end to end on PuSH-T without storing code or results permanently on the remote server.

RECAP improved average cumulative reward over the BC baseline in this small setting, but neither method achieved task success in 50 evaluation episodes. This should be treated as a pipeline and early signal run, not as evidence that RECAP solves PuSH-T under this training budget.

Next recommended experiment: increase the policy training budget and evaluation sample size while keeping the same comparison structure.

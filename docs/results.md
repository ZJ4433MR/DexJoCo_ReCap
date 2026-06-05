# Experiment Results

## Current validated run

Run: `pusht_recap_diffusion_weighted_ft_l40_v1`

Local result directory:

```text
E:\Evo-RL-main\recap-sim-l40\runs\pusht_recap_diffusion_weighted_ft_l40_v1
```

Remote staging directory was removed after the result archive was pulled back.

Configuration:

- Dataset: `lerobot/pusht`
- Episodes: `0..99`
- Policy: diffusion
- BC policy steps: `20000`
- Value model steps: `2000`
- ReCap policy steps: `5000`
- ReCap initialization: from BC checkpoint
- ACP sampling: full dataset, ACP-positive rows upweighted by `1.5`
- Evaluation: `50` PuSH-T episodes

Metrics:

| Method | Success rate | Avg sum reward | Avg max reward | Episodes |
| --- | ---: | ---: | ---: | ---: |
| BC diffusion | 28.0% | 117.0486 | 0.7937 | 50 |
| ReCap/ACP weighted fine-tune | 34.0% | 102.5325 | 0.8100 | 50 |

Conclusion: BC is no longer zero, and the ReCap/ACP weighted fine-tune improves
success rate by `+6.0` percentage points over the same-run BC baseline.

## Why the earlier BC result was zero

The first formal small run used a small ACT policy on PuSH-T with only `10000`
policy steps. That setup was too weak for reliable PuSH-T success:

| Run | Method | Success rate | Avg sum reward | Avg max reward |
| --- | --- | ---: | ---: | ---: |
| `pusht_recap_formal_small_l40_v1` | BC ACT | 0.0% | 23.9544 | 0.2118 |
| `pusht_recap_formal_small_l40_v1` | ReCap ACT | 0.0% | 33.0433 | 0.1752 |

This indicated an underpowered baseline rather than a broken simulator. Switching
to a diffusion policy and training for `20000` steps produced non-zero BC
success.

## Failed ReCap variant

The first diffusion ReCap attempt used hard positive filtering: train only rows
whose ACP indicator was positive. BC became non-zero, but ReCap became much
worse:

| Run | Method | Success rate | Avg sum reward | Avg max reward |
| --- | --- | ---: | ---: | ---: |
| `pusht_recap_diffusion_filter_fast_l40_v4` | BC diffusion | 22.0% | 116.2807 | 0.8123 |
| `pusht_recap_diffusion_filter_fast_l40_v4` | ReCap hard-positive filter | 2.0% | 16.6456 | 0.1266 |

Interpretation: hard filtering removes too much trajectory coverage for this
non-language-conditioned PuSH-T policy. Keeping the full dataset and using
ACP-positive rows as a soft sampling weight is the better simulation adaptation.

## Source patch note

The original Evo-RL ACP hook changes the `task` text prompt. ACT and diffusion
policies in this PuSH-T setup do not consume that text prompt, so a non-language
ACP mechanism was needed. The local Evo-RL source tree was patched to add:

- `--acp.filter_positive=true`
- `--acp.positive_sample_weight=<weight>`

The successful run used `--acp.positive_sample_weight=1.5`.

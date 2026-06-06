# Experiment Results

## Stronger full-data run

Run: `pusht_recap_diffusion_full50k_w15_l40_v1`

Local result directory:

```text
E:\Evo-RL-main\recap-sim-l40\runs\pusht_recap_diffusion_full50k_w15_l40_v1
```

Remote staging directory was removed after the result archive was pulled back.

Configuration:

- Dataset: `lerobot/pusht`
- Episodes: `0..205`
- Policy: diffusion
- BC policy steps: `50000`
- Value model steps: `5000`
- ReCap policy steps: `5000`
- ReCap initialization: from BC checkpoint
- ACP sampling: full dataset, ACP-positive rows upweighted by `1.5`
- Evaluation: `100` PuSH-T episodes

Metrics:

| Method | Success rate | Avg sum reward | Avg max reward | Episodes |
| --- | ---: | ---: | ---: | ---: |
| BC diffusion | 50.0% | 114.5318 | 0.9267 | 100 |
| ReCap/ACP weighted fine-tune | 62.0% | 111.4149 | 0.9674 | 100 |

Conclusion: increasing the dataset and BC training budget improves BC from the
previous 28.0% to 50.0%. ReCap/ACP weighted fine-tuning still improves over the
stronger BC baseline by `+12.0` percentage points.

## First validated non-zero run

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

## DexJoCo language-policy smoke

Run: `dexjoco_headless_smoke_l40_v4`

Purpose: verify that the next-stage DexJoCo environment works on the L40 node
and exposes a real language-conditioned policy observation for single-arm
`water_plant`.

Result:

- DexJoCo source commit: `8d23b0fab23b17a58c4b55f3942e17013aaf8267`
- Task: `water_plant`
- Prompt: `Grasp the watering can and apply water to the plant.`
- State shape: `23`
- Image shapes: `base=(224,224,3)`, `wrist=(224,224,3)`
- Smoke steps: `5`
- Exit code: `0`

The compute node could not reliably clone GitHub, so the runner now supports a
temporary `-LocalDexJoCoPath` fallback. DexJoCo source is copied only into the
per-run archive and removed remotely after results are pulled back.

## DexJoCo OpenPI/pi0.5 pretrained eval

Run: `dexjoco_pi05_water_plant_eval_l40_v1`

Configuration:

- Task: `water_plant`
- Policy config: `water_plant`
- Eval config: `configs/rand_obj/water_plant.yaml`
- Checkpoint source: `DexJoCo/DexJoCo-Pi05`, path
  `pi05_dexjoco_ckpt/water_plant`
- Episodes: `3`
- Seed: `0`

Outcome:

| Method | Success rate | Episodes |
| --- | ---: | ---: |
| OpenPI/pi0.5 pretrained `water_plant` | 0.0% | 3 |

Interpretation: the DexJoCo + OpenPI language-conditioned policy path is now
operational: the job installed DexJoCo/OpenPI, downloaded the public checkpoint,
restored the 6.3 GiB params, started the websocket policy server, and completed
simulation evaluation. The 3-episode smoke was not enough to show success on
`water_plant`; the next experimental step is to run a larger 20-50 episode eval
or compare easier single-arm tasks such as `click_mouse` and `hammer_nail`
before adding ReCap value/advantage conditioning.

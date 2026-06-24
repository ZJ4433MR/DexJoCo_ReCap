# DexJoCo Evo-RL Reproduction

This directory contains the runnable experiment scripts used to reproduce the
Evo-RL/ReCap idea in DexJoCo simulation.

The repository is not the official Evo-RL project. It is a DexJoCo-focused
reproduction that keeps the ReCap loop:

```text
rollout data
  -> value model
  -> value / advantage / ACP indicator
  -> advantage-conditioned policy training
  -> DexJoCo evaluation
```

Compared with the original project, this implementation uses both PyTorch and
JAX:

- PyTorch/LeRobot-compatible code handles LeRobot datasets, Pistar06 value
  training, value inference, and ACP labels.
- JAX/OpenPI handles pi0.5 policy fine-tuning, serving, and DexJoCo evaluation.
- DexJoCo rollout data can be used either as compact NPZ files or as LeRobot
  format datasets.

## Install

Install the LeRobot-compatible source tree from the repository root:

```bash
cd ../Evo-RL-main
conda create -y -n dexjoco-recap python=3.10
conda activate dexjoco-recap
python -m pip install -U pip
python -m pip install -e .
```

DexJoCo and OpenPI are external dependencies. The remote job helpers can clone
DexJoCo automatically, or you can provide a local fallback copy:

```bash
git clone https://github.com/brave-eai/dexjoco.git .local/dexjoco-src
```

## Remote Runner

The PowerShell runner packages this experiment directory, the LeRobot-compatible
source tree, and optionally a local DexJoCo source copy, then launches a remote
job.

Create a private config:

```powershell
Copy-Item configs\remote.env.example configs\remote.env
```

Edit `configs/remote.env` for your SSH alias, environment setup, Slurm
partition, GPU request, memory, and time limit. Do not commit `remote.env`.

Run a smoke test:

```powershell
.\scripts\run_remote_slurm.ps1 `
  -ConfigPath configs\remote.env `
  -LocalDexJoCoPath .local\dexjoco-src `
  -Job jobs\20_dexjoco_headless_smoke.sh `
  -RunName dexjoco_headless_smoke `
  -Time 02:00:00 `
  -Memory 96G
```

Results are pulled back to:

```text
runs/<run_name>/
```

For machines without Slurm, use `scripts/run_remote_ssh.ps1` with the same job
script interface.

## Core Jobs

Useful entry points:

```text
jobs/20_dexjoco_headless_smoke.sh
  Verify DexJoCo can start headlessly and produce observations.

jobs/21_dexjoco_pi05_water_plant_eval.sh
  Evaluate the public pi0.5 checkpoint on water_plant.

jobs/22_dexjoco_pi05_single_arm_matrix.sh
  Run a small single-arm task matrix to choose a workable baseline task.

jobs/31_dexjoco_pi05_click_mouse_eval100.sh
  Evaluate the public pi0.5 click_mouse baseline.

jobs/43_dexjoco_click_mouse_recap_pistar06_full_eval100.sh
  Compact NPZ path: collect rollouts, train Pistar06 value, infer ACP labels,
  fine-tune OpenPI, and evaluate.

jobs/57_dexjoco_click_mouse_evorl_lerobot_ab.sh
  LeRobot-format path: merge pools, train Pistar06 value, infer indicators,
  fine-tune OpenPI with ACP prompt tags, collect the next round, and evaluate.

jobs/68_dexjoco_click_mouse_evorl_lerobot_E_multitag_episode_smooth.sh
  Multi-tag variant with failure/low/medium/high prompt tags.
```

## DexJoCo Data Flow

Compact rollout NPZ files can be converted to local LeRobot datasets:

```bash
PYTHONPATH="../Evo-RL-main/src:$PYTHONPATH" \
python scripts/dexjoco_npz_to_lerobot.py \
  --input runs/<run_name>/rollouts.npz \
  --output-root data/click_mouse_lerobot \
  --repo-id local/click_mouse_lerobot \
  --task click_mouse \
  --fps 30 \
  --overwrite
```

Multiple LeRobot-format pools can be merged:

```bash
PYTHONPATH="../Evo-RL-main/src:$PYTHONPATH" \
python scripts/dexjoco_merge_lerobot_pool.py \
  --input local/base=data/base_pool \
  --input local/round1=data/round1_pool \
  --output-root data/click_mouse_pool \
  --output-repo-id local/click_mouse_pool \
  --overwrite
```

## Value Training

The faithful path trains the Pistar06 value model through the
LeRobot-compatible code:

```bash
cd ../Evo-RL-main
PYTHONPATH="src:$PYTHONPATH" python -m lerobot.scripts.lerobot_value_train \
  --dataset.repo_id=local/click_mouse_pool \
  --dataset.root=../dexjoco-recap/data/click_mouse_pool \
  --value.type=pistar06 \
  --value.dtype=bfloat16 \
  --value.vision_repo_id=google/siglip-so400m-patch14-384 \
  --value.language_repo_id=google/gemma-3-270m \
  --value.camera_features="[observation.images.ego_right,observation.images.wrist]" \
  --value.normalization_mapping="{VISUAL: IDENTITY, STATE: QUANTILES, ACTION: IDENTITY}" \
  --value.device=cuda \
  --value.push_to_hub=false \
  --batch_size=16 \
  --steps=8000 \
  --save_checkpoint=true \
  --save_freq=8000 \
  --wandb.enable=false \
  --output_dir=outputs/value_train/click_mouse_pistar06 \
  --job_name=click_mouse_pistar06
```

## Value Inference and ACP Labels

After value training, write value, advantage, and ACP indicator fields back to
the dataset:

```bash
PYTHONPATH="src:$PYTHONPATH" python -m lerobot.scripts.lerobot_value_infer \
  --dataset.repo_id=local/click_mouse_pool \
  --dataset.root=../dexjoco-recap/data/click_mouse_pool \
  --dataset.default_success=success \
  --inference.checkpoint_path=outputs/value_train/click_mouse_pistar06 \
  --runtime.device=cuda \
  --runtime.batch_size=16 \
  --acp.enable=true \
  --acp.n_step=50 \
  --acp.positive_ratio=0.3 \
  --acp.binarization=task_quantile \
  --acp.value_field=complementary_info.value \
  --acp.advantage_field=complementary_info.advantage \
  --acp.indicator_field=complementary_info.acp_indicator \
  --output_dir=outputs/value_infer/click_mouse_pistar06 \
  --job_name=click_mouse_pistar06_infer
```

For the multi-tag variant, use:

```text
--acp.binarization=episode_multitag_smooth
--acp.multitag_ratios=0.1,0.2,0.3
--acp.indicator_field=complementary_info.acp_indicator
```

## JAX/OpenPI Policy Training

OpenPI is patched at runtime so LeRobot samples with
`complementary_info.acp_indicator` can inject prompt tags before repacking.
The main patch helpers are:

```text
scripts/dexjoco_openpi_lerobot_acp.sh
scripts/dexjoco_common.sh
```

The job scripts set the important OpenPI environment variables:

```bash
OPENPI_LEROBOT_ACP_ENABLE=1
OPENPI_LEROBOT_ACP_INDICATOR_FIELD=complementary_info.acp_indicator
OPENPI_LEROBOT_ACP_INDICATOR_DROPOUT_PROB=0.3
OPENPI_LEROBOT_ACP_TAG_KEY=Advantage
OPENPI_LEROBOT_ACP_TAG_VALUES=negative,positive
OPENPI_RECAP_LORA_ONLY=1
```

Then OpenPI training is launched with:

```bash
python scripts/compute_norm_stats.py click_mouse --batch-size=2 --num-workers=0
python scripts/train.py click_mouse \
  --exp-name=<exp_name> \
  --overwrite \
  --num-train-steps=1200 \
  --num-workers=0 \
  --save-interval=400 \
  --log-interval=50 \
  --fsdp-devices=1
```

## DexJoCo Evaluation

Evaluation serves the trained OpenPI checkpoint and calls `dexjoco-openpi-eval`:

```bash
python scripts/serve_policy.py \
  --port=23456 \
  policy:checkpoint \
  --policy.config=click_mouse \
  --policy.dir=<policy_checkpoint_dir>

dexjoco-openpi-eval \
  --config=configs/rand_obj/click_mouse.yaml \
  --seed=0 \
  --port=23456 \
  --host=127.0.0.1 \
  --episodes=100 \
  --output=eval_episodes
```

The remote jobs automate the server lifecycle, seed loop, and success-rate
collection.

## Using Real-Robot Data

This repository does not require a pretrained checkpoint when training on a new
real-robot dataset. See [`docs/real_robot_data.md`](docs/real_robot_data.md).

In short, if the real-robot dataset is already in LeRobot format, replace the
DexJoCo dataset arguments:

```text
--dataset.repo_id=<real_robot_repo_or_local_id>
--dataset.root=<optional_local_dataset_root>
```

and rerun value training, value inference, and policy training. A checkpoint is
only needed if you want to skip training or directly evaluate/deploy an existing
policy.

## Notes

- Do not commit `configs/remote.env`, Hugging Face tokens, run outputs,
  checkpoints, videos, or downloaded DexJoCo source.
- The scripts assume Linux paths on the remote machine and PowerShell on the
  local launcher machine.
- If remote GitHub access is unstable, pass `-LocalDexJoCoPath` so the runner
  packs a local DexJoCo source copy into the job archive.

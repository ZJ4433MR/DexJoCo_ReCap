# DexJoCo Evo-RL Reproduction

This repository is a cleaned project workspace for reproducing the Evo-RL/ReCap
training idea in the DexJoCo simulation environment.

It is not the official Evo-RL repository. The implementation keeps the core
ReCap loop, but adapts the data flow and training stack to DexJoCo:

- DexJoCo and OpenPI/pi0.5 are used for language-conditioned simulation
  rollouts, policy fine-tuning, and evaluation.
- PyTorch/LeRobot-compatible code is used for dataset handling, value training,
  value inference, advantage labeling, and ACP indicators.
- JAX/OpenPI is used for the policy training and serving path.

The public-facing experiment code is in [`dexjoco-recap/`](dexjoco-recap/).
The adapted LeRobot/Evo-compatible source tree is in
[`Evo-RL-main/`](Evo-RL-main/).

## Repository Layout

```text
dexjoco-recap/
  README.md
  configs/remote.env.example
  jobs/
  scripts/
  docs/real_robot_data.md

Evo-RL-main/
  src/lerobot/
  pyproject.toml
```

Older scratch work, generated run logs, upload manifests, and private remote
configuration are intentionally excluded from the cleaned project view.

## Quick Start

Install the LeRobot-compatible source:

```bash
cd Evo-RL-main
conda create -y -n dexjoco-recap python=3.10
conda activate dexjoco-recap
python -m pip install -U pip
python -m pip install -e .
```

Then follow the experiment workflow in:

```text
dexjoco-recap/README.md
```

## Checkpoints

No checkpoint from this repository is required if the goal is to train on a new
dataset. For a new DexJoCo or real-robot dataset, replace the dataset path or
`repo_id` in the value/policy training commands and train from that data.

A checkpoint is only needed when the goal is direct evaluation/deployment of a
previously trained policy, or when skipping value/policy training and running
only downstream inference.

## Acknowledgement

This project is a reproduction/adaptation built on public ideas and tooling from
LeRobot, Evo-RL/ReCap, DexJoCo, and OpenPI. It should be cited as an independent
reproduction workspace, not as the official implementation of any of those
projects.

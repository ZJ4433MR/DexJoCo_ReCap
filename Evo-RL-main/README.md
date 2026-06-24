# LeRobot-Compatible Source for DexJoCo ReCap Reproduction

This directory contains the adapted LeRobot-compatible source used by the
DexJoCo Evo-RL/ReCap reproduction in `../dexjoco-recap`.

It is not presented as the official Evo-RL repository. The source keeps the
installable `lerobot` package layout and adds the value-training, value
inference, ACP-labeling, and policy-training hooks needed by the reproduction.

Install from this directory with:

```bash
conda create -y -n dexjoco-recap python=3.10
conda activate dexjoco-recap
python -m pip install -U pip
python -m pip install -e .
```

The main experiment workflow and runnable commands are documented in:

```text
../dexjoco-recap/README.md
```

Relevant entry points include:

```text
lerobot-value-train
lerobot-value-infer
lerobot-train
lerobot-dataset-report
```

For training on real-robot data, see:

```text
../dexjoco-recap/docs/real_robot_data.md
```

Acknowledgement: this source tree is based on the public LeRobot ecosystem and
Evo-RL/ReCap-style training workflow, with local adaptations for the DexJoCo
reproduction.

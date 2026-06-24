# 用于 DexJoCo ReCap 复现实验的 LeRobot 兼容源码

这个目录包含本复现实验使用的 LeRobot 兼容源码，对应的实验脚本在
`../dexjoco-recap` 中。

这里不作为官方 Evo-RL 仓库发布。代码保留了可安装的 `lerobot` 包结构，并加入了
本实验需要的 value training、value inference、ACP label 生成和策略训练相关接口。

安装方式：

```bash
conda create -y -n dexjoco-recap python=3.10
conda activate dexjoco-recap
python -m pip install -U pip
python -m pip install -e .
```

主要实验流程和可直接运行的命令见：

```text
../dexjoco-recap/README.md
```

相关入口包括：

```text
lerobot-value-train
lerobot-value-infer
lerobot-train
lerobot-dataset-report
```

如果要接真机数据训练，见：

```text
../dexjoco-recap/docs/real_robot_data.md
```

说明：这个源码目录基于公开 LeRobot 生态和 Evo-RL/ReCap 风格训练流程整理，并针对
DexJoCo 复现实验做了本地适配。

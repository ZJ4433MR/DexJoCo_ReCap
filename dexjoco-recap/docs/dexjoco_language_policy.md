# DexJoCo 语言条件策略阶段

DexJoCo 阶段把 ReCap 风格的复现实验从无 prompt 的控制 baseline，推进到语言条件
VLA 设置。

本项目主要使用 single-arm `click_mouse` 任务和 OpenPI/pi0.5 策略接口。这个设置更
接近 ReCap 中的 prompt conditioning 思路，因为策略输入包含：

- 来自 base 和 wrist 相机的 RGB observation。
- 本体状态。
- DexJoCo 任务配置中的 task prompt。

在 ACP 实验中，prompt 会额外加入一个 tag，例如：

```text
Advantage: positive
```

multi-tag 实验中则使用：

```text
Advantage: failure
Advantage: low
Advantage: medium
Advantage: high
```

## 当前实现路径

当前较 faithful 的 LeRobot 格式路径如下：

1. 下载或准备 DexJoCo 的 LeRobot 格式任务数据集。
2. 给初始 demonstration episodes 标注 `episode_success`。
3. 合并当前数据池。
4. 使用 PyTorch/LeRobot 兼容源码训练 Pistar06 value model。
5. 把 value、n-step advantage 和 ACP indicator 推理回数据集。
6. 运行时 patch OpenPI，从 indicator 字段向 prompt 注入 ACP tag。
7. 使用 JAX/OpenPI 微调 pi0.5 策略。
8. 在 DexJoCo 中 rollout 更新后的策略，并把新数据加入数据池。
9. 重复多轮后评测最终 checkpoint。

主要编排入口是：

```text
jobs/57_dexjoco_click_mouse_evorl_lerobot_ab.sh
```

multi-tag 版本是：

```text
jobs/68_dexjoco_click_mouse_evorl_lerobot_E_multitag_episode_smooth.sh
```

## 为什么选择 `click_mouse`

早期 single-arm 评测显示，`click_mouse` 是一个适合做第一版 ReCap 复现实验的目标：
公开 pi0.5 baseline 有非零成功率，但没有达到饱和，因此可以观察 value-derived ACP
标签和 prompt conditioning 是否带来变化。

## 数据格式

当前支持两种数据格式：

- 紧凑的 DexJoCo rollout NPZ 文件，适合轻量级本地 value 标注。
- LeRobot 格式数据集，适合更完整的 Pistar06 value training/inference 流程。

后续新实验优先使用 LeRobot 路径，因为它和 value 训练/推理流程更一致，也更方便
替换成真机数据。

## 参考

- DexJoCo: https://github.com/brave-eai/dexjoco
- DexJoCo pi0.5 checkpoints: https://huggingface.co/DexJoCo/DexJoCo-Pi05
- DexJoCo LeRobot datasets: https://huggingface.co/datasets/DexJoCo/DexJoCo-Datasets-LeRobot

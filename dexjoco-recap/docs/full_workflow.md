# DexJoCo ReCap 完整闭环说明

这份文档面向第一次阅读本仓库的人，解释一次完整 DexJoCo ReCap / Evo-RL 风格实验到底做了什么、每一步代码来自哪里、数据怎样流动，以及哪些部分是本仓库的适配代码。

主入口脚本是：

```text
dexjoco-recap/jobs/57_dexjoco_click_mouse_evorl_lerobot_ab.sh
```

它对应的核心闭环是：

```text
DexJoCo / pi0.5 rollout 数据
  -> LeRobot 数据池
  -> Pistar06 value model 训练
  -> value / advantage / ACP indicator 写回
  -> OpenPI/pi0.5 policy 微调
  -> 回 DexJoCo 收集新 rollout
  -> 新数据加入下一轮数据池
  -> 最终 DexJoCo evaluation
```

## 先区分几个模型和组件

这套实验里容易混淆的是 value model、policy model、环境和数据格式。它们的职责不同：

| 名称 | 作用 | 是否输出机器人动作 | 主要来源 |
| --- | --- | --- | --- |
| DexJoCo | MuJoCo 机器人仿真环境、任务、评测接口 | 否 | `brave-eai/dexjoco` |
| OpenPI / pi0.5 policy | 输入图像、状态和语言 prompt，输出 action | 是 | Physical Intelligence OpenPI，DexJoCo 提供任务 checkpoint |
| Pistar06 value model | 给 LeRobot 数据中的 frame 预测 value，用来生成 advantage 和 ACP indicator | 否 | MINT-SJTU/Evo-RL 的 LeRobot-compatible value backend |
| ACP indicator | 由 value/advantage 后处理得到的训练标签 | 否 | ReCap 思路，本仓库写回到 LeRobot 数据 |
| LeRobot dataset | value/policy 训练共同使用的数据格式 | 否 | Hugging Face LeRobot 生态 |
| NPZ rollout | DexJoCo rollout 的轻量中间格式 | 否 | 本仓库收集脚本 |

最重要的一点是：**value model 不控制机器人**。真正执行动作的是 OpenPI/pi0.5 policy。value model 只负责回答“这帧状态或这段轨迹质量如何”，然后把这个判断转成 policy 训练时可用的条件信号。

## 来源边界

本仓库不是任何上游项目的官方实现。推荐在论文、报告或答辩中按下面方式说明来源：

| 层级 | 内容 | 来源 |
| --- | --- | --- |
| ReCap / advantage-conditioned policy training 思想 | 训练 value function，计算 advantage，再用 advantage 条件化策略训练 | Physical Intelligence 的 Pi*0.6 / ReCap 思路 |
| Pistar06 value backend 代码 | `lerobot-src/src/lerobot/values/pistar06/` | [MINT-SJTU/Evo-RL](https://github.com/MINT-SJTU/Evo-RL) 风格的 LeRobot-compatible 实现 |
| Pistar06 默认预训练组件 | SigLIP 图像编码器、Gemma 语言编码器 | [google/siglip-so400m-patch14-384](https://huggingface.co/google/siglip-so400m-patch14-384) 和 [google/gemma-3-270m](https://huggingface.co/google/gemma-3-270m) |
| LeRobot 数据格式和 PyTorch 数据接口 | Parquet/MP4 或 image-based 的机器人数据集格式、训练/推理数据读取 | [huggingface/lerobot](https://github.com/huggingface/lerobot) |
| pi0.5 policy 与 OpenPI 训练栈 | VLA policy、JAX/OpenPI training、policy serving | [Physical-Intelligence/openpi](https://github.com/Physical-Intelligence/openpi)，DexJoCo 中集成 |
| LoRA 微调方法 | 冻结大部分预训练参数，只训练低秩 adapter | [LoRA paper](https://arxiv.org/abs/2106.09685)，OpenPI 训练栈支持，本仓库默认启用 LoRA-only |
| DexJoCo 任务和 pi0.5 task checkpoint | `click_mouse` 等仿真任务，DexJoCo-Pi05 checkpoint，DexJoCo LeRobot 数据 | [brave-eai/dexjoco](https://github.com/brave-eai/dexjoco)、[DexJoCo/DexJoCo-Pi05](https://huggingface.co/DexJoCo/DexJoCo-Pi05)、[DexJoCo/DexJoCo-Datasets-LeRobot](https://huggingface.co/datasets/DexJoCo/DexJoCo-Datasets-LeRobot) |
| 本仓库适配 | 远程 runner、DexJoCo rollout 收集、NPZ 转 LeRobot、ACP prompt patch、多轮 job 编排 | 本 DexJoCo ReCap 复现实验 |

因此可以这样一句话概括：

```text
本项目把 Physical Intelligence ReCap/Pi*0.6 的训练思路、
MINT-SJTU/Evo-RL 的 Pistar06 value backend、
DexJoCo/OpenPI 的 pi0.5 仿真训练栈，
整理成一个面向 DexJoCo click_mouse 的可复现实验闭环。
```

## Step 0: 准备远程运行环境

实验通常在远程 Linux/GPU 机器上运行，本地 PowerShell 负责打包和提交。远程会准备三个运行环境：

| 环境 | 作用 |
| --- | --- |
| DexJoCo conda env | 启动 MuJoCo/DexJoCo 仿真、收集 rollout、评测 |
| OpenPI conda env | 训练 JAX/OpenPI pi0.5 policy，serve policy |
| LeRobot/PyTorch env | 训练 Pistar06 value model，运行 value inference，处理 LeRobot 数据 |

相关脚本：

```text
dexjoco-recap/scripts/run_remote_slurm.ps1
dexjoco-recap/scripts/run_remote_ssh.ps1
dexjoco-recap/scripts/remote_train.sh
dexjoco-recap/scripts/dexjoco_common.sh
```

推荐先复制私有配置：

```powershell
Copy-Item configs\remote.env.example configs\remote.env
```

然后填写 SSH alias、Slurm 分区、GPU、内存、时间限制、环境初始化命令等。`configs/remote.env` 不应提交。

最小 smoke test 是：

```powershell
.\scripts\run_remote_slurm.ps1 `
  -ConfigPath configs\remote.env `
  -LocalDexJoCoPath .local\dexjoco-src `
  -Job jobs\20_dexjoco_headless_smoke.sh `
  -RunName dexjoco_headless_smoke `
  -Time 02:00:00 `
  -Memory 96G
```

这个 smoke test 会检查 DexJoCo 是否能 headless 启动，并返回 base/wrist 图像、state、done/success 等基本 observation。

## Step 1: 下载 DexJoCo 源码、pi0.5 checkpoint 和初始数据

`jobs/57_dexjoco_click_mouse_evorl_lerobot_ab.sh` 会先调用：

```text
prepare_dexjoco_source
setup_dexjoco_env
setup_openpi_env
download_dexjoco_pi05_checkpoint
download_dexjoco_lerobot_dataset
```

其中：

- DexJoCo 源码来自 `https://github.com/brave-eai/dexjoco.git`。
- 初始 pi0.5 checkpoint 来自 `DexJoCo/DexJoCo-Pi05`。
- 官方 LeRobot 数据来自 `DexJoCo/DexJoCo-Datasets-LeRobot`。

脚本中默认目标任务是：

```bash
DEXJOCO_TASK=click_mouse
```

初始 policy 路径设置为：

```text
../checkpoints/pi05_dexjoco_ckpt/click_mouse
../checkpoints/pi05_dexjoco_ckpt/click_mouse/params
```

也就是说，实验不是从随机 policy 开始，而是从 DexJoCo 提供的 pi0.5 task checkpoint 继续微调。

## Step 2: 准备初始数据 D0

脚本下载官方 DexJoCo LeRobot 数据后，会把初始 demonstration episodes 标注为成功：

```bash
python scripts/dexjoco_lerobot_set_episode_success.py \
  --root "$DEXJOCO_OFFICIAL_LEROBOT_ROOT" \
  --output-root "$D0_ROOT" \
  --label success \
  --overwrite
```

这一步得到初始数据：

```text
D0_ROOT = 初始 demonstration LeRobot 数据集
D0_REPO_ID = local/<task>_official_d0_success
```

D0 是第一轮 value training 和 policy training 的起点。后续每轮新 rollout 会加入同一个数据池。

## Step 3: 合并当前 LeRobot 数据池

每一轮训练前都会把已有数据合并成一个 pool：

```text
D0
+ r01 rollout
+ r02 rollout
+ ...
```

对应函数在 `jobs/57...sh` 中叫：

```text
merge_pool
```

实际调用脚本：

```bash
python scripts/dexjoco_merge_lerobot_pool.py \
  --input local/base=<D0_ROOT> \
  --input local/round1=<round1_root> \
  --output-root <pool_root> \
  --output-repo-id <pool_repo_id> \
  --overwrite
```

这个脚本使用 LeRobot 的 dataset merge 工具，把多轮数据汇总为当前 stage 的训练数据。它不会训练模型，只负责把数据格式整理好。

## Step 4: 训练 Pistar06 value model

这一阶段进入 PyTorch/LeRobot 路径。命令大致是：

```bash
cd ../lerobot-src
PYTHONPATH="src:$PYTHONPATH" python -m lerobot.scripts.lerobot_value_train \
  --dataset.repo_id="$POOL_REPO_ID" \
  --dataset.root="$POOL_ROOT" \
  --value.type=pistar06 \
  --value.dtype=bfloat16 \
  --value.vision_repo_id=google/siglip-so400m-patch14-384 \
  --value.language_repo_id=google/gemma-3-270m \
  --value.camera_features="[observation.images.ego_right,observation.images.wrist]" \
  --value.device=cuda \
  --batch_size=16 \
  --steps=8000 \
  --save_checkpoint=true
```

### Pistar06 的代码和模型来源

`--value.type=pistar06` 对应：

```text
lerobot-src/src/lerobot/values/pistar06/
```

它不是 Physical Intelligence 官方 Pi*0.6 value checkpoint，也不是 DexJoCo 提供的 value model。更准确地说：

- ReCap/Pi*0.6 提供 value function 和 advantage conditioning 的方法思想。
- Evo-RL 风格代码提供 LeRobot-compatible 的 Pistar06 value backend。
- 默认视觉 backbone 是 `google/siglip-so400m-patch14-384`。
- 默认语言 backbone 是 `google/gemma-3-270m`。
- 本仓库在 DexJoCo LeRobot 数据上训练 value head 和相关参数。

### Pistar06 输入和输出

输入：

```text
observation.images.ego_right
observation.images.wrist
observation.state
task
```

输出：

```text
当前 frame 的 value
```

它不输出 action。value target 由 episode 是否成功、当前 frame 距离 episode 结束还有多少步、失败惩罚系数等构造。直观理解是：

```text
成功轨迹越接近完成，value 越高；
失败轨迹会有额外惩罚；
value 用于后续 advantage 和 ACP indicator。
```

## Step 5: 推理 value、计算 advantage 和 ACP indicator

value model 训练完成后，脚本会运行：

```bash
PYTHONPATH="src:$PYTHONPATH" python -m lerobot.scripts.lerobot_value_infer \
  --dataset.repo_id="$POOL_REPO_ID" \
  --dataset.root="$POOL_ROOT" \
  --dataset.default_success=success \
  --inference.checkpoint_path="$value_dir" \
  --runtime.device=cuda \
  --acp.enable=true \
  --acp.n_step=50 \
  --acp.positive_ratio=0.3 \
  --acp.binarization=task_quantile \
  --acp.value_field=complementary_info.value \
  --acp.advantage_field=complementary_info.advantage \
  --acp.indicator_field=complementary_info.acp_indicator
```

它会把这些字段写回 LeRobot 数据集：

| 字段 | 含义 |
| --- | --- |
| `complementary_info.value` | Pistar06 对每帧预测的 value |
| `complementary_info.advantage` | n-step return 加 bootstrap 再减去当前 value 的 advantage |
| `complementary_info.acp_indicator` | advantage 离散化后的 prompt 条件标签 |

默认二分类模式：

```text
0 -> negative
1 -> positive
```

multi-tag 模式：

```text
0 -> failure
1 -> low
2 -> medium
3 -> high
```

注意：ACP indicator 不是 DexJoCo 环境原始 reward，而是由 value/advantage 后处理得到的训练标签。

## Step 6: 微调 OpenPI/pi0.5 policy

这一阶段进入 JAX/OpenPI 路径。它才是真正训练会输出 action 的 policy。

### 来源

| 内容 | 来源 |
| --- | --- |
| pi0.5 VLA policy 思想和 OpenPI 训练栈 | Physical Intelligence OpenPI |
| DexJoCo 中的 OpenPI 训练/评测集成 | brave-eai/dexjoco |
| 初始 task checkpoint | `DexJoCo/DexJoCo-Pi05` |
| ACP prompt tag 注入 patch | 本仓库 `scripts/dexjoco_openpi_lerobot_acp.sh` |

### ACP prompt patch

OpenPI 原本只看到任务 prompt。本仓库会在运行时 patch OpenPI 的 LeRobot dataset 读取逻辑，让它读取：

```text
complementary_info.acp_indicator
```

然后把 indicator 转成 prompt tag：

```text
Click the mouse.
Advantage: positive
```

或者：

```text
Click the mouse.
Advantage: negative
```

关键环境变量：

```bash
OPENPI_LEROBOT_ACP_ENABLE=1
OPENPI_LEROBOT_ACP_INDICATOR_FIELD=complementary_info.acp_indicator
OPENPI_LEROBOT_ACP_INDICATOR_DROPOUT_PROB=0.3
OPENPI_LEROBOT_ACP_TAG_KEY=Advantage
OPENPI_LEROBOT_ACP_TAG_VALUES=negative,positive
```

### LoRA-only 微调

脚本默认：

```bash
OPENPI_RECAP_LORA_ONLY=1
```

LoRA 是低秩适配微调方法，思想是冻结大部分预训练参数，只训练小的 LoRA adapter。这个方法本身来自 LoRA 论文；OpenPI/pi0.5 训练栈本来支持 LoRA fine-tuning；本仓库通过 patch OpenPI 的 freeze filter 强制只更新名字里包含 `lora` 的参数。

这不是 ReCap 原文规定的算法步骤，也不是随便选的参数集合。它是 OpenPI/DexJoCo pi0.5 微调栈已有的轻量微调方式，本仓库把它作为默认工程选择来降低显存和训练风险。

### compute_norm_stats 和 train

策略训练前先计算归一化统计：

```bash
python scripts/compute_norm_stats.py click_mouse \
  --batch-size=2 \
  --num-workers=0
```

这一步扫描当前训练数据池，为 state/action 等特征计算 normalization stats。每轮数据池都会变化，所以每轮都重新计算。

然后训练：

```bash
python scripts/train.py click_mouse \
  --exp-name=<exp_name> \
  --overwrite \
  --num-train-steps=1200 \
  --num-workers=0 \
  --save-interval=400 \
  --log-interval=50 \
  --fsdp-devices=1
```

训练完成后会得到新的 OpenPI/pi0.5 policy checkpoint：

```text
$DEXJOCO_DIR/checkpoints/evorl_lerobot_ckpts/<task>/<exp_name>/<step>/
```

后续 rollout 和最终评测都会用这个 checkpoint。

## Step 7: 回到 DexJoCo 收集新 rollout

第 6 步得到新 policy 后，脚本会把它放回 DexJoCo 环境里执行任务，收集下一轮数据。

### 来源

| 内容 | 来源 |
| --- | --- |
| policy serving 和 websocket client | DexJoCo/OpenPI，源头是 Physical Intelligence OpenPI |
| DexJoCo 仿真交互 | brave-eai/dexjoco |
| rollout 收集脚本 | 本仓库 `scripts/dexjoco_collect_success_rollouts.py` |
| NPZ 转 LeRobot 脚本 | 本仓库 `scripts/dexjoco_npz_to_lerobot.py` |

### 启动 policy server

收集前，job 会启动训练好的 policy：

```bash
python scripts/serve_policy.py \
  --port=<port> \
  policy:checkpoint \
  --policy.config=click_mouse \
  --policy.dir=<policy_checkpoint_dir>
```

然后 DexJoCo 通过 websocket 调用它：

```text
DexJoCo observation -> OpenPI policy server -> action -> DexJoCo step
```

### 一个 episode 内部发生什么

`dexjoco_collect_success_rollouts.py` 的核心逻辑是：

```text
env = DexJoCoOpenPIEnv(...)
client = WebsocketClientPolicy(...)

env.start()
for episode in episodes:
    env.reset()
    obs = env.get_obs()
    actions = client.infer(obs)
    env.step(action)
    保存 obs_before 和 action
    直到 env.is_done 或 max_steps
    保存 env.is_success
env.close()
```

`click_mouse` 任务还会在 episode 开头做一段初始对齐动作，目的是让机械臂处于更稳定的起始状态。这是任务相关的工程处理，不是 ReCap 算法核心。

### 收集 prompt 是 base 还是 ACP

本 job 默认：

```bash
DEXJOCO_EVO_COLLECT_PROMPT=positive
```

因此收集 rollout 时通常使用：

```text
原始任务 prompt
Advantage: positive
```

也可以把 `DEXJOCO_EVO_COLLECT_PROMPT=base`，只用基础任务 prompt。这个选项用于做消融：看新数据收集阶段是否需要 Advantage prompt。

### NPZ 中记录什么

收集脚本先写一个紧凑 NPZ。主要字段是：

| 字段 | 含义 |
| --- | --- |
| `base` | base/ego camera 图像 |
| `wrist` | wrist camera 图像 |
| `state` | 机器人状态 |
| `action` | policy 输出并执行的 action |
| `episode_id` | 每帧属于哪个 episode |
| `is_success` | 该帧所属 episode 是否成功 |
| `base_prompt` | 原始任务 prompt |
| `acp_prompt` | 带 ACP suffix 的 prompt |
| `collection_prompt` | 本次 rollout 实际使用的 prompt |

脚本也支持分片保存，比如每 25 个 episode 写一个 shard，最后再合并成一个 NPZ，避免单个收集进程持有太多数据。

### 为什么先 NPZ 后 LeRobot

NPZ 是 DexJoCo rollout 的轻量中间格式，便于直接保存数组、分片、合并和调试。但后续 Pistar06 value training 使用 LeRobot 数据集，所以必须转换。

转换脚本：

```bash
python scripts/dexjoco_npz_to_lerobot.py \
  --input <collected_rollouts.npz> \
  --output-root <round_lerobot_root> \
  --repo-id <round_repo_id> \
  --task click_mouse \
  --fps 30 \
  --match-features-root <D0_ROOT> \
  --overwrite
```

字段映射：

| NPZ 字段 | LeRobot 字段 |
| --- | --- |
| `base` | `observation.images.ego_right` |
| `wrist` | `observation.images.wrist` |
| `state` | `observation.state` |
| `action` | `action` |
| `base_prompt` / `prompt` | `task` |
| `is_success` | episode metadata 中的 `episode_success` |

转换完成后，脚本把新数据加入下一轮 pool：

```bash
pool_inputs+=("$rollout_repo_id=$rollout_root")
```

## Step 8: 多轮迭代

`jobs/57...sh` 默认：

```bash
DEXJOCO_EVO_ROUNDS=3
```

流程是：

```text
d0: 用 D0 训练 value 和 policy
r01: 用 d0 policy 收集 rollout，加入 pool，重新训练 value 和 policy
r02: 用 r01 policy 收集 rollout，加入 pool，重新训练 value 和 policy
r03: 用 r02 policy 收集 rollout，加入 pool，重新训练 value 和 policy
```

每轮都会重新：

```text
merge pool
train value
infer value/advantage/ACP
train OpenPI policy
collect rollout
```

这就是 ReCap/Evo-RL 风格闭环中最核心的 data refresh 机制。

## Step 9: 最终评测

最后一轮训练完成后，脚本会导出最终 checkpoint：

```text
export_checkpoints/<variant>_final_<policy_step>/
```

然后启动 policy server，并调用：

```bash
dexjoco-openpi-eval \
  --config=configs/rand_obj/click_mouse.yaml \
  --seed=<seed> \
  --port=<port> \
  --host=127.0.0.1 \
  --episodes=100 \
  --output=<eval_output_dir>
```

评测结果会写入：

```text
evorl_lerobot_summary.tsv
```

主要字段：

| 字段 | 含义 |
| --- | --- |
| `stage` | d0、r01、r02、final 等 |
| `pool_episodes` | 当前训练池 episode 数 |
| `pool_frames` | 当前训练池 frame 数 |
| `policy_step` | policy checkpoint step |
| `collect_prompt` | 收集新 rollout 时用 base 还是 positive |
| `collect_successes` | 收集阶段成功 episode 数 |
| `collect_episodes` | 收集阶段保存 episode 数 |
| `eval_successes` | 最终评测成功次数 |
| `eval_episodes` | 最终评测总次数 |

## 常用 job 对照

| job | 用途 |
| --- | --- |
| `20_dexjoco_headless_smoke.sh` | 检查 DexJoCo headless 环境 |
| `31_dexjoco_pi05_click_mouse_eval100.sh` | 评测公开 pi0.5 baseline |
| `43_dexjoco_click_mouse_recap_pistar06_full_eval100.sh` | NPZ 路径的 Pistar06 ReCap 实验 |
| `57_dexjoco_click_mouse_evorl_lerobot_ab.sh` | LeRobot 路径的主 ReCap 闭环 |
| `65_dexjoco_click_mouse_evorl_lerobot_C_base_collect_no_acp.sh` | 不启用 ACP policy training 的消融 |
| `66_dexjoco_click_mouse_evorl_lerobot_D_episode_topk_smooth.sh` | episode top-k smoothing 标签版本 |
| `68_dexjoco_click_mouse_evorl_lerobot_E_multitag_episode_smooth.sh` | multi-tag 标签版本 |
| `70_real_robot_lerobot_value_acp_template.sh` | 真机 LeRobot 数据的 value/ACP 模板 |

## 不要混淆的边界

- `pistar06` value model 不输出机器人 action。
- `pi0.5` policy 才输出 action。
- `ACP indicator` 不是环境 reward，而是 value/advantage 后处理标签。
- `OPENPI_RECAP_LORA_ONLY=1` 不是 ReCap 原文规定的算法步骤，而是本实验沿用 OpenPI/DexJoCo LoRA 微调能力的工程选择。
- NPZ 是本实验的中间 rollout 格式；Pistar06 faithful 路径推荐使用 LeRobot 数据。
- 真机数据支持默认只覆盖 LeRobot 格式数据的 value/ACP 训练和标注，不包含真机采集、控制、安全部署和 action adapter。

## 一句话版本

一次完整运行可以概括为：

```text
用 DexJoCo 官方数据启动第一轮训练；
用 Pistar06 value model 给数据打 value/advantage/ACP 标签；
把 ACP 标签注入 OpenPI/pi0.5 prompt，LoRA-only 微调真正输出 action 的 policy；
把新 policy 放回 DexJoCo 收集 rollout；
把新 rollout 转成 LeRobot 数据并加入下一轮 pool；
重复多轮后评测最终 checkpoint 成功率。
```

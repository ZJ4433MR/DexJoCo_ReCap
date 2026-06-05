# RECAP Simulation on Temporary L40 Compute

这个仓库用于保存 RECAP 仿真实验的脚本、配置、运行日志和从远程服务器拉回来的结果。

设计原则：

- 本地保存实验仓库和 Git 历史。
- L40 服务器只作为临时计算节点。
- 每次运行把 Evo-RL 源码和本仓库脚本打包上传到远程 `/tmp` 目录。
- 远程训练输出先写到临时目录，然后打包拉回本地 `runs/<run_name>/`。
- 脚本默认清理远程临时目录和远程结果包。

## 当前任务边界

组里的目标是复现 π*0.6 论文里的 RECAP 算法机制。由于官方 π0.6 / π*0.6 的完整权重、数据和真实机器人条件没有完全公开，当前阶段先在仿真中验证：

```text
BC baseline
vs
BC + value model + advantage label + advantage-conditioned policy
```

本仓库负责远程训练编排；Evo-RL 主代码仍来自本地目录：

```text
E:\Evo-RL-main\Evo-RL-main
```

## 远程服务器设置

你的本地 SSH config 里已经有：

```text
hpc-server
hpc-hopper
```

建议先测试：

```powershell
ssh hpc-hopper "hostname && nvidia-smi"
```

在远程服务器上建议提前准备一个 Python 环境，例如：

```bash
conda create -n evo-rl python=3.10 -y
conda activate evo-rl
```

第一次正式训练前，可以在远程环境里安装依赖。环境可以留在服务器上；本项目避免留下的是实验代码、数据和结果。

## 配置

复制配置模板：

```powershell
Copy-Item configs\remote-l40.env.example configs\remote-l40.env
```

按实际情况编辑：

```text
REMOTE_HOST=hpc-hopper
REMOTE_BASE=/tmp/$USER/recap-sim-l40
REMOTE_ENV_SETUP=source ~/miniconda3/etc/profile.d/conda.sh && conda activate evo-rl
```

`configs/remote-l40.env` 被 `.gitignore` 忽略，可以放本地私有信息。

## 先跑 smoke test

这个命令会：

1. 打包本地 Evo-RL 和本实验仓库。
2. 上传到 L40 服务器临时目录。
3. 检查 CUDA/PyTorch。
4. 跑 RECAP/ACP 相关单元测试。
5. 把日志拉回本地。
6. 清理远程临时目录。

```powershell
.\scripts\run_remote_l40.ps1 `
  -ConfigPath configs\remote-l40.env `
  -Job jobs/00_remote_smoke.sh `
  -RunName smoke_l40
```

结果会在：

```text
runs/smoke_l40/
```

重点看：

```text
runs/smoke_l40/system.log
runs/smoke_l40/job.log
runs/smoke_l40/exit_code.txt
```

## RECAP 仿真实验模板

模板脚本：

```text
jobs/10_pusht_recap_template.sh
```

它包含四步：

```text
1. 训练 BC baseline policy
2. 训练 pistar06 value model
3. 推理 value / advantage / acp_indicator
4. 训练 advantage-conditioned policy
```

运行示例：

```powershell
.\scripts\run_remote_l40.ps1 `
  -ConfigPath configs\remote-l40.env `
  -Job jobs/10_pusht_recap_template.sh `
  -RunName pusht_recap_act_l40
```

默认变量在 job 里可以改：

```bash
DATASET_REPO=lerobot/pusht
POLICY_TYPE=act
POLICY_STEPS=20000
VALUE_STEPS=4000
VALUE_LANGUAGE_REPO=Qwen/Qwen2.5-0.5B
```

## 不在远程保存代码和结果的机制

远程运行目录类似：

```text
/tmp/$USER/recap-sim-l40/<run_name>/
```

远程导出包类似：

```text
/tmp/$USER/recap-sim-l40/<run_name>_results.tar.gz
```

脚本默认会删除这些远程文件。只有本地保留：

```text
runs/<run_name>/
```

调试时可以加 `-KeepRemote`：

```powershell
.\scripts\run_remote_l40.ps1 -ConfigPath configs\remote-l40.env -Job jobs/00_remote_smoke.sh -RunName debug_l40 -KeepRemote
```

调试完成后记得手动清理远程临时目录。

## GitHub 使用方式

本机没有检测到 `gh` 命令，所以这里先初始化本地 git 仓库。之后有两种方式推到 GitHub：

方式 A：网页创建空仓库，然后本地执行：

```powershell
git remote add origin git@github.com:<your_name>/recap-sim-l40.git
git push -u origin main
```

方式 B：安装并登录 GitHub CLI 后执行：

```powershell
gh repo create recap-sim-l40 --private --source . --remote origin --push
```

建议仓库设为 private，因为里面会记录服务器别名、实验路径和组内任务信息。

param(
    [string]$ConfigPath = "",
    [string]$HostAlias = "hpc-hopper",
    [string]$LocalEvoRlPath = "E:\Evo-RL-main\Evo-RL-main",
    [string]$LocalDexJoCoPath = "",
    [string]$Job = "jobs/00_remote_smoke.sh",
    [string]$RunName = "",
    [string]$RemoteStageBase = "/share/home/u23133/.cache/recap-sim-l40-stage",
    [string]$RemoteComputeBase = "/tmp/`$USER/recap-sim-l40",
    [string]$RemoteEnvSetup = "module load miniconda3/24.1.2 cuda/12.1 && source /share/apps/miniconda3/etc/profile.d/conda.sh && conda activate tj.pytorch2.2.1",
    [string]$RemoteBefore = "",
    [string]$HfToken = "",
    [string]$Partition = "L40",
    [string]$Gres = "gpu:l40:1",
    [int]$Cpus = 7,
    [string]$Memory = "64G",
    [string]$Time = "00:30:00",
    [switch]$KeepRemote
)

$ErrorActionPreference = "Stop"

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")

if (-not [string]::IsNullOrWhiteSpace($ConfigPath)) {
    $ResolvedConfig = Resolve-Path $ConfigPath
    Write-Host "[local] Loading config $ResolvedConfig"
    foreach ($Line in Get-Content $ResolvedConfig) {
        $Trimmed = $Line.Trim()
        if ($Trimmed.Length -eq 0 -or $Trimmed.StartsWith("#")) {
            continue
        }
        $Parts = $Trimmed.Split("=", 2)
        if ($Parts.Count -ne 2) {
            continue
        }
        $Key = $Parts[0].Trim()
        $Value = $Parts[1].Trim()
        switch ($Key) {
            "REMOTE_HOST" { $HostAlias = $Value }
            "REMOTE_STAGE_BASE" { $RemoteStageBase = $Value }
            "REMOTE_BASE" { $RemoteComputeBase = $Value }
            "REMOTE_ENV_SETUP" { $RemoteEnvSetup = $Value }
            "REMOTE_BEFORE" { $RemoteBefore = $Value }
            "HF_TOKEN" { $HfToken = $Value }
            "SLURM_PARTITION" { $Partition = $Value }
            "SLURM_GRES" { $Gres = $Value }
            "SLURM_CPUS" { $Cpus = [int]$Value }
            "SLURM_MEM" { $Memory = $Value }
            "SLURM_TIME" { $Time = $Value }
        }
    }
}

$LocalEvoRlPath = (Resolve-Path $LocalEvoRlPath).Path
$JobPath = Join-Path $RepoRoot $Job
if (-not (Test-Path $JobPath)) {
    throw "Job script not found: $JobPath"
}

if ([string]::IsNullOrWhiteSpace($RunName)) {
    $RunName = "recap_" + (Get-Date -Format "yyyyMMdd_HHmmss")
}

$TmpRoot = Join-Path $RepoRoot ".tmp"
$LocalPackDir = Join-Path $TmpRoot $RunName
$ArchivePath = Join-Path $TmpRoot "$RunName.tar.gz"
$LocalResultDir = Join-Path $RepoRoot "runs\$RunName"
$LocalResultArchive = Join-Path $LocalResultDir "remote_results.tar.gz"

New-Item -ItemType Directory -Force -Path $TmpRoot, $LocalPackDir, $LocalResultDir | Out-Null

$PackEvoRl = Join-Path $LocalPackDir "Evo-RL-main"
$PackExp = Join-Path $LocalPackDir "recap-sim-l40"
New-Item -ItemType Directory -Force -Path $PackEvoRl, $PackExp | Out-Null

Write-Host "[local] Packing Evo-RL source from $LocalEvoRlPath"
robocopy $LocalEvoRlPath $PackEvoRl /MIR /XD .git outputs wandb hf_cache .pytest_cache __pycache__ /XF *.pyc | Out-Null
$RoboCode = $LASTEXITCODE
if ($RoboCode -ge 8) {
    throw "robocopy failed for Evo-RL with exit code $RoboCode"
}

Write-Host "[local] Packing experiment repo from $RepoRoot"
robocopy $RepoRoot $PackExp /MIR /XD .git .local runs .tmp outputs wandb data datasets hf_cache .pytest_cache __pycache__ /XF *.pyc *.tar.gz *.zip | Out-Null
$RoboCode = $LASTEXITCODE
if ($RoboCode -ge 8) {
    throw "robocopy failed for experiment repo with exit code $RoboCode"
}

if (-not [string]::IsNullOrWhiteSpace($LocalDexJoCoPath)) {
    $ResolvedDexJoCoPath = (Resolve-Path $LocalDexJoCoPath).Path
    $PackDexJoCo = Join-Path $PackExp ".local\dexjoco-src"
    New-Item -ItemType Directory -Force -Path (Split-Path $PackDexJoCo -Parent) | Out-Null
    Write-Host "[local] Packing DexJoCo fallback source from $ResolvedDexJoCoPath"
    robocopy $ResolvedDexJoCoPath $PackDexJoCo /MIR /XD __pycache__ .pytest_cache /XF *.pyc | Out-Null
    $RoboCode = $LASTEXITCODE
    if ($RoboCode -ge 8) {
        throw "robocopy failed for DexJoCo fallback source with exit code $RoboCode"
    }
}

if (Test-Path $ArchivePath) {
    Remove-Item -LiteralPath $ArchivePath -Force
}

Write-Host "[local] Creating archive $ArchivePath"
Push-Location $LocalPackDir
tar -czf $ArchivePath .
Pop-Location

$RemoteStageDir = "$RemoteStageBase/$RunName"
$RemoteArchive = "$RemoteStageDir/incoming.tar.gz"
$RemoteRunner = "$RemoteStageDir/remote_train.sh"
$RemoteSbatch = "$RemoteStageDir/submit.sbatch"
$RemoteExport = "$RemoteStageDir/results.tar.gz"
$RemoteSlurmLog = "$RemoteStageDir/slurm-%j.out"

Write-Host "[local] Creating remote staging dir $RemoteStageDir on $HostAlias"
ssh $HostAlias "mkdir -p '$RemoteStageDir'"

Write-Host "[local] Uploading archive and runner"
scp $ArchivePath "${HostAlias}:$RemoteArchive"
scp (Join-Path $RepoRoot "scripts\remote_train.sh") "${HostAlias}:$RemoteRunner"

$KeepRemoteValue = if ($KeepRemote) { "1" } else { "0" }
$SbatchContent = @"
#!/usr/bin/env bash
#SBATCH --job-name=recap-$RunName
#SBATCH --partition=$Partition
#SBATCH --gres=$Gres
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=$Cpus
#SBATCH --mem=$Memory
#SBATCH --time=$Time
#SBATCH --output=$RemoteSlurmLog

set -euo pipefail

export REMOTE_BASE="$RemoteComputeBase"
export REMOTE_ENV_SETUP='$RemoteEnvSetup'
export REMOTE_BEFORE='$RemoteBefore'
export HF_TOKEN='$HfToken'
export KEEP_REMOTE='$KeepRemoteValue'

bash '$RemoteRunner' '$RunName' '$RemoteArchive' '$Job' '$RemoteExport'
"@

$LocalSbatch = Join-Path $TmpRoot "$RunName.sbatch"
$SbatchContentLf = $SbatchContent.Replace("`r`n", "`n")
[System.IO.File]::WriteAllText($LocalSbatch, $SbatchContentLf, [System.Text.Encoding]::ASCII)
scp $LocalSbatch "${HostAlias}:$RemoteSbatch"

Write-Host "[local] Submitting Slurm job"
$SubmitOutput = ssh $HostAlias "sbatch --parsable '$RemoteSbatch'"
$SubmitText = ($SubmitOutput | Out-String).Trim()
if ($SubmitText -notmatch "(\d+)") {
    throw "Could not parse Slurm job id from: $SubmitText"
}
$JobId = $Matches[1]
Write-Host "[local] Slurm job id: $JobId"

while ($true) {
    Start-Sleep -Seconds 15
    $State = ssh $HostAlias "squeue -j $JobId -h -o '%T' 2>/dev/null || true"
    $StateText = ($State | Out-String).Trim()
    if ([string]::IsNullOrWhiteSpace($StateText)) {
        break
    }
    Write-Host "[local] Slurm state: $StateText"
}

Write-Host "[local] Slurm job finished; checking export"
$ExportExists = ssh $HostAlias "test -f '$RemoteExport' && echo yes || echo no"
if (($ExportExists | Out-String).Trim() -ne "yes") {
    Write-Warning "[local] Remote result archive not found. Pulling Slurm log if available."
    scp "${HostAlias}:$RemoteStageDir/slurm-*.out" $LocalResultDir 2>$null
    throw "Remote result archive not found: $RemoteExport"
}

Write-Host "[local] Pulling results to $LocalResultArchive"
scp "${HostAlias}:$RemoteExport" $LocalResultArchive

Write-Host "[local] Expanding results into $LocalResultDir"
tar -xzf $LocalResultArchive -C $LocalResultDir

if (-not $KeepRemote) {
    Write-Host "[local] Removing remote staging dir"
    ssh $HostAlias "rm -rf '$RemoteStageDir'"
} else {
    Write-Host "[local] KEEP_REMOTE=1, leaving remote staging dir at $RemoteStageDir"
}

foreach ($Path in @($LocalPackDir, $ArchivePath, $LocalSbatch)) {
    if (Test-Path $Path) {
        Remove-Item -LiteralPath $Path -Recurse -Force
    }
}

$ExitCodePath = Join-Path $LocalResultDir "exit_code.txt"
if (Test-Path $ExitCodePath) {
    $RemoteExit = (Get-Content $ExitCodePath -Raw).Trim()
    if ($RemoteExit -ne "0") {
        throw "Remote job failed with exit code $RemoteExit. Check $LocalResultDir\job.log"
    }
}

Write-Host "[local] Done. Results are in $LocalResultDir"

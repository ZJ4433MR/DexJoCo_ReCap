param(
    [string]$ConfigPath = "",
    [string]$HostAlias = "remote-gpu",
    [string]$SshConfigPath = "",
    [string]$LocalLeRobotPath = "..\lerobot-src",
    [string]$LocalDexJoCoPath = "",
    [string]$Job = "jobs/00_remote_smoke.sh",
    [string]$RunName = "",
    [string]$RemoteStageBase = "/tmp/`$USER/dexjoco-recap-stage",
    [string]$RemoteComputeBase = "/tmp/`$USER/dexjoco-recap",
    [string]$RemoteEnvSetup = "source ~/miniconda3/etc/profile.d/conda.sh && conda activate dexjoco-recap",
    [string]$RemoteBefore = "",
    [string]$HfToken = "",
    [string]$HfTokenFile = "",
    [string]$Partition = "gpu",
    [string]$Gres = "gpu:1",
    [string]$Exclude = "",
    [string]$RemoteExportMode = "",
    [int]$Cpus = 7,
    [string]$Memory = "64G",
    [string]$Time = "00:30:00",
    [string]$Dependency = "",
    [switch]$KeepRemote,
    [switch]$SubmitOnly
)

$ErrorActionPreference = "Stop"

function Invoke-Checked {
    param(
        [string]$Label,
        [scriptblock]$Command
    )
    & $Command
    if ($LASTEXITCODE -ne 0) {
        throw "$Label failed with exit code $LASTEXITCODE"
    }
}

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")

if (-not [string]::IsNullOrWhiteSpace($ConfigPath)) {
    $ResolvedConfig = Resolve-Path $ConfigPath
    Write-Host "[local] Loading config $ResolvedConfig"
    $ExplicitParams = $PSBoundParameters
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
            "REMOTE_HOST" { if (-not $ExplicitParams.ContainsKey("HostAlias")) { $HostAlias = $Value } }
            "REMOTE_STAGE_BASE" { if (-not $ExplicitParams.ContainsKey("RemoteStageBase")) { $RemoteStageBase = $Value } }
            "REMOTE_BASE" { if (-not $ExplicitParams.ContainsKey("RemoteComputeBase")) { $RemoteComputeBase = $Value } }
            "REMOTE_ENV_SETUP" { if (-not $ExplicitParams.ContainsKey("RemoteEnvSetup")) { $RemoteEnvSetup = $Value } }
            "REMOTE_BEFORE" { if (-not $ExplicitParams.ContainsKey("RemoteBefore")) { $RemoteBefore = $Value } }
            "HF_TOKEN" { if (-not $ExplicitParams.ContainsKey("HfToken")) { $HfToken = $Value } }
            "HF_TOKEN_FILE" { if (-not $ExplicitParams.ContainsKey("HfTokenFile")) { $HfTokenFile = $Value } }
            "SLURM_PARTITION" { if (-not $ExplicitParams.ContainsKey("Partition")) { $Partition = $Value } }
            "SLURM_GRES" { if (-not $ExplicitParams.ContainsKey("Gres")) { $Gres = $Value } }
            "SLURM_CPUS" { if (-not $ExplicitParams.ContainsKey("Cpus")) { $Cpus = [int]$Value } }
            "SLURM_MEM" { if (-not $ExplicitParams.ContainsKey("Memory")) { $Memory = $Value } }
            "SLURM_TIME" { if (-not $ExplicitParams.ContainsKey("Time")) { $Time = $Value } }
        }
    }
}

if (-not [System.IO.Path]::IsPathRooted($LocalLeRobotPath)) {
    $LocalLeRobotPath = Join-Path $RepoRoot $LocalLeRobotPath
}
$LocalLeRobotPath = (Resolve-Path $LocalLeRobotPath).Path
$JobPath = Join-Path $RepoRoot $Job
if (-not (Test-Path $JobPath)) {
    throw "Job script not found: $JobPath"
}

if ([string]::IsNullOrWhiteSpace($RunName)) {
    $RunName = "dexjoco_recap_" + (Get-Date -Format "yyyyMMdd_HHmmss")
}

$SshArgs = @()
$ScpArgs = @()
if (-not [string]::IsNullOrWhiteSpace($SshConfigPath)) {
    $ResolvedSshConfigPath = (Resolve-Path $SshConfigPath).Path
    $SshArgs += @("-F", $ResolvedSshConfigPath)
    $ScpArgs += @("-F", $ResolvedSshConfigPath)
}

$TmpRoot = Join-Path $RepoRoot ".tmp"
$LocalPackDir = Join-Path $TmpRoot $RunName
$ArchivePath = Join-Path $TmpRoot "$RunName.tar.gz"
$LocalResultDir = Join-Path $RepoRoot "runs\$RunName"
$LocalResultArchive = Join-Path $LocalResultDir "remote_results.tar.gz"

New-Item -ItemType Directory -Force -Path $TmpRoot, $LocalPackDir, $LocalResultDir | Out-Null

$PackLeRobot = Join-Path $LocalPackDir "lerobot-src"
$PackExp = Join-Path $LocalPackDir "dexjoco-recap"
New-Item -ItemType Directory -Force -Path $PackLeRobot, $PackExp | Out-Null

Write-Host "[local] Packing LeRobot-compatible source from $LocalLeRobotPath"
robocopy $LocalLeRobotPath $PackLeRobot /MIR /XD .git outputs wandb hf_cache .pytest_cache __pycache__ /XF *.pyc | Out-Null
$RoboCode = $LASTEXITCODE
if ($RoboCode -ge 8) {
    throw "robocopy failed for LeRobot source with exit code $RoboCode"
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
    robocopy $ResolvedDexJoCoPath $PackDexJoCo /MIR /XD .git __pycache__ .pytest_cache /XF *.pyc | Out-Null
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
Invoke-Checked "tar create archive" { tar -czf $ArchivePath . }
Pop-Location

$RemoteStageDir = "$RemoteStageBase/$RunName"
$RemoteArchive = "$RemoteStageDir/incoming.tar.gz"
$RemoteRunner = "$RemoteStageDir/remote_train.sh"
$RemoteSbatch = "$RemoteStageDir/submit.sbatch"
$RemoteExport = "$RemoteStageDir/results.tar.gz"
$RemoteSlurmLog = "$RemoteStageDir/slurm-%j.out"
$RemoteHfTokenFile = "$RemoteStageDir/hf_token.secret"
$DependencyLine = if ([string]::IsNullOrWhiteSpace($Dependency)) { "" } else { "#SBATCH --dependency=$Dependency" }
$ExcludeLine = if ([string]::IsNullOrWhiteSpace($Exclude)) { "" } else { "#SBATCH --exclude=$Exclude" }

Write-Host "[local] Creating remote staging dir $RemoteStageDir on $HostAlias"
Invoke-Checked "ssh mkdir remote staging dir" { & ssh @SshArgs $HostAlias "mkdir -p '$RemoteStageDir'" }

Write-Host "[local] Uploading archive and runner"
Invoke-Checked "scp archive" { & scp @ScpArgs $ArchivePath "${HostAlias}:$RemoteArchive" }
Invoke-Checked "scp remote runner" { & scp @ScpArgs (Join-Path $RepoRoot "scripts\remote_train.sh") "${HostAlias}:$RemoteRunner" }
Invoke-Checked "ssh verify uploaded archive and runner" {
    & ssh @SshArgs $HostAlias "test -s '$RemoteArchive' && test -s '$RemoteRunner'"
}

$UseRemoteHfTokenFile = $false
if (-not [string]::IsNullOrWhiteSpace($HfTokenFile)) {
    $ResolvedHfTokenFile = (Resolve-Path $HfTokenFile).Path
    Write-Host "[local] Uploading Hugging Face token secret for this job"
    Invoke-Checked "scp Hugging Face token secret" { & scp @ScpArgs $ResolvedHfTokenFile "${HostAlias}:$RemoteHfTokenFile" }
    Invoke-Checked "ssh chmod Hugging Face token secret" { & ssh @SshArgs $HostAlias "chmod 600 '$RemoteHfTokenFile'" }
    $UseRemoteHfTokenFile = $true
}

$KeepRemoteValue = if ($KeepRemote) { "1" } else { "0" }
$HfTokenExport = if ($UseRemoteHfTokenFile) {
    "export HF_TOKEN=`$(tr -d '\r\n' < '$RemoteHfTokenFile'); rm -f '$RemoteHfTokenFile'"
} elseif ([string]::IsNullOrWhiteSpace($HfToken)) {
    "unset HF_TOKEN"
} else {
    "export HF_TOKEN='$HfToken'"
}
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
$DependencyLine
$ExcludeLine

set -euo pipefail

export REMOTE_BASE="$RemoteComputeBase"
export REMOTE_ENV_SETUP='$RemoteEnvSetup'
export REMOTE_BEFORE='$RemoteBefore'
export REMOTE_EXPORT_MODE='$RemoteExportMode'
$HfTokenExport
export KEEP_REMOTE='$KeepRemoteValue'

bash '$RemoteRunner' '$RunName' '$RemoteArchive' '$Job' '$RemoteExport'
"@

$LocalSbatch = Join-Path $TmpRoot "$RunName.sbatch"
$SbatchContentLf = $SbatchContent.Replace("`r`n", "`n")
[System.IO.File]::WriteAllText($LocalSbatch, $SbatchContentLf, [System.Text.Encoding]::ASCII)
Invoke-Checked "scp sbatch script" { & scp @ScpArgs $LocalSbatch "${HostAlias}:$RemoteSbatch" }

Write-Host "[local] Submitting Slurm job"
$SubmitOutput = & ssh @SshArgs $HostAlias "sbatch --parsable '$RemoteSbatch'"
if ($LASTEXITCODE -ne 0) {
    throw "ssh sbatch failed with exit code $LASTEXITCODE"
}
$SubmitText = ($SubmitOutput | Out-String).Trim()
if ($SubmitText -notmatch "(\d+)") {
    throw "Could not parse Slurm job id from: $SubmitText"
}
$JobId = $Matches[1]
Write-Host "[local] Slurm job id: $JobId"

$JobInfoPath = Join-Path $LocalResultDir "slurm_job.txt"
@(
    "run_name=$RunName"
    "job_id=$JobId"
    "remote_stage_dir=$RemoteStageDir"
    "remote_export=$RemoteExport"
    "remote_slurm_log=$($RemoteSlurmLog.Replace('%j', $JobId))"
    "job_script=$Job"
    "submitted_at=$(Get-Date -Format o)"
) | Set-Content -Path $JobInfoPath -Encoding ASCII

if ($SubmitOnly) {
    Write-Host "[local] SubmitOnly=1, not waiting for Slurm completion or pulling full results."
    Write-Host "[local] Job metadata written to $JobInfoPath"
    foreach ($Path in @($LocalPackDir, $ArchivePath, $LocalSbatch)) {
        if (Test-Path $Path) {
            Remove-Item -LiteralPath $Path -Recurse -Force
        }
    }
    Write-Host "[local] Done. Remote staging dir: $RemoteStageDir"
    exit 0
}

while ($true) {
    Start-Sleep -Seconds 15
    $State = & ssh @SshArgs $HostAlias "squeue -j $JobId -h -o '%T' 2>/dev/null || true"
    $StateText = ($State | Out-String).Trim()
    if ([string]::IsNullOrWhiteSpace($StateText)) {
        break
    }
    Write-Host "[local] Slurm state: $StateText"
}

Write-Host "[local] Slurm job finished; checking export"
$ExportExists = & ssh @SshArgs $HostAlias "test -f '$RemoteExport' && echo yes || echo no"
if (($ExportExists | Out-String).Trim() -ne "yes") {
    Write-Warning "[local] Remote result archive not found. Pulling Slurm log if available."
    & scp @ScpArgs "${HostAlias}:$RemoteStageDir/slurm-*.out" $LocalResultDir 2>$null
    throw "Remote result archive not found: $RemoteExport"
}

Write-Host "[local] Pulling results to $LocalResultArchive"
& scp @ScpArgs "${HostAlias}:$RemoteExport" $LocalResultArchive

Write-Host "[local] Expanding results into $LocalResultDir"
tar -xzf $LocalResultArchive -C $LocalResultDir

if (-not $KeepRemote) {
    Write-Host "[local] Removing remote staging dir"
    & ssh @SshArgs $HostAlias "rm -rf '$RemoteStageDir'"
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

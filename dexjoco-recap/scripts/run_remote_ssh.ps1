param(
    [string]$ConfigPath = "",
    [string]$HostAlias = "remote-gpu",
    [string]$LocalLeRobotPath = "..\lerobot-src",
    [string]$Job = "jobs/00_remote_smoke.sh",
    [string]$RunName = "",
    [string]$RemoteBase = "/tmp/`$USER/dexjoco-recap",
    [string]$RemoteEnvSetup = "source ~/miniconda3/etc/profile.d/conda.sh && conda activate dexjoco-recap",
    [string]$RemoteBefore = "",
    [string]$LocalDexJoCoPath = "",
    [string]$HfToken = "",
    [string]$HfTokenFile = "",
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
            "REMOTE_BASE" { $RemoteBase = $Value }
            "REMOTE_ENV_SETUP" { $RemoteEnvSetup = $Value }
            "REMOTE_BEFORE" { $RemoteBefore = $Value }
            "HF_TOKEN" { $HfToken = $Value }
            "HF_TOKEN_FILE" { $HfTokenFile = $Value }
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
tar -czf $ArchivePath .
Pop-Location

$RemoteRunDir = "$RemoteBase/$RunName"
$RemoteArchive = "$RemoteRunDir/incoming.tar.gz"
$RemoteRunner = "$RemoteRunDir/remote_train.sh"
$RemoteExport = "$RemoteBase/${RunName}_results.tar.gz"
$RemoteHfTokenFile = "$RemoteRunDir/hf_token.secret"

Write-Host "[local] Creating remote run dir $RemoteRunDir on $HostAlias"
ssh $HostAlias "mkdir -p '$RemoteRunDir' '$RemoteBase'"

Write-Host "[local] Uploading archive and runner"
scp $ArchivePath "${HostAlias}:$RemoteArchive"
scp (Join-Path $RepoRoot "scripts\remote_train.sh") "${HostAlias}:$RemoteRunner"

$UseRemoteHfTokenFile = $false
if (-not [string]::IsNullOrWhiteSpace($HfTokenFile)) {
    $ResolvedHfTokenFile = (Resolve-Path $HfTokenFile).Path
    Write-Host "[local] Uploading Hugging Face token secret for this job"
    scp $ResolvedHfTokenFile "${HostAlias}:$RemoteHfTokenFile"
    ssh $HostAlias "chmod 600 '$RemoteHfTokenFile'"
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
$RemoteCommand = @"
export REMOTE_BASE='$RemoteBase'
export REMOTE_ENV_SETUP='$RemoteEnvSetup'
export REMOTE_BEFORE='$RemoteBefore'
$HfTokenExport
export KEEP_REMOTE='$KeepRemoteValue'
bash '$RemoteRunner' '$RunName' '$RemoteArchive' '$Job' '$RemoteExport'
"@

Write-Host "[local] Starting remote job"
$RemoteCommand | ssh $HostAlias "bash -s"
$RemoteExit = $LASTEXITCODE

Write-Host "[local] Pulling results to $LocalResultArchive"
scp "${HostAlias}:$RemoteExport" $LocalResultArchive

Write-Host "[local] Expanding results into $LocalResultDir"
tar -xzf $LocalResultArchive -C $LocalResultDir

Write-Host "[local] Removing remote export archive"
ssh $HostAlias "rm -f '$RemoteExport' '$RemoteArchive' '$RemoteRunner'"

if (Test-Path $LocalPackDir) {
    Remove-Item -LiteralPath $LocalPackDir -Recurse -Force
}
if (Test-Path $ArchivePath) {
    Remove-Item -LiteralPath $ArchivePath -Force
}

if ($RemoteExit -ne 0) {
    throw "Remote job failed with exit code $RemoteExit. Check $LocalResultDir\job.log"
}

Write-Host "[local] Done. Results are in $LocalResultDir"

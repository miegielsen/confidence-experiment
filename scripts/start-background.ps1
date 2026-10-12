param(
    [Parameter(Mandatory=$true)][string]$Config,
    [switch]$DownloadEmbedding,
    [switch]$DownloadEncoder,
    [ValidateRange(1,64)][int]$CpuThreads = 2,
    [string]$ThenConfig
)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
. (Join-Path $PSScriptRoot 'load-anthropic-key.ps1')
. (Join-Path $PSScriptRoot 'load-openai-key.ps1')
$pythonPath = Join-Path $env:LOCALAPPDATA 'confidence-experiment/venv/Scripts/python.exe'
$arguments = @('scripts/background_run.py', '--config', ('"' + $Config + '"'), '--cpu-threads', $CpuThreads)
if ($DownloadEmbedding) { $arguments += '--download-embedding' }
if ($DownloadEncoder) { $arguments += '--download-encoder' }
if ($ThenConfig) { $arguments += @('--then-config', ('"' + $ThenConfig + '"')) }
$taskProcess = Start-Process -FilePath $pythonPath -ArgumentList $arguments -WorkingDirectory $projectRoot -WindowStyle Hidden -PassThru
Write-Output "Started background supervisor PID $($taskProcess.Id)."

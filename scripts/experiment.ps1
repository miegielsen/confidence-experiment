$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$pythonPath = Join-Path $env:LOCALAPPDATA 'confidence-experiment/venv/Scripts/python.exe'
Set-Location -LiteralPath $projectRoot
if (-not (Test-Path -LiteralPath $pythonPath)) { throw 'Run ./scripts/setup.ps1 first.' }
. (Join-Path $PSScriptRoot 'load-anthropic-key.ps1')
& $pythonPath -m confidence_experiment @args
exit $LASTEXITCODE

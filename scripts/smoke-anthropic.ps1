$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath (Split-Path -Parent $PSScriptRoot)
. (Join-Path $PSScriptRoot 'load-anthropic-key.ps1')
$pythonPath = Join-Path $env:LOCALAPPDATA 'confidence-experiment/venv/Scripts/python.exe'
& $pythonPath './scripts/smoke_anthropic.py'
exit $LASTEXITCODE

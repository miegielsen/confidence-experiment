$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
. (Join-Path $PSScriptRoot 'load-openai-key.ps1')
$env:OMP_NUM_THREADS = '2'
$env:MKL_NUM_THREADS = '2'
$env:OPENBLAS_NUM_THREADS = '2'
$pythonPath = Join-Path $env:LOCALAPPDATA 'confidence-experiment/venv/Scripts/python.exe'
& $pythonPath (Join-Path $PSScriptRoot 'run_openai_decisions.py') @args
exit $LASTEXITCODE

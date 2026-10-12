$ErrorActionPreference = 'Stop'
# Keep thousands of environment files off the Google Drive filesystem.
$projectRoot = Split-Path -Parent $PSScriptRoot
$environmentPath = Join-Path $env:LOCALAPPDATA 'confidence-experiment/venv'
Set-Location -LiteralPath $projectRoot
uv venv $environmentPath
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
$pythonPath = Join-Path $environmentPath 'Scripts/python.exe'
uv pip install --python $pythonPath -e '.[test,llm]'
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
Write-Output "Ready: $pythonPath"

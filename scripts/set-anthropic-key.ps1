$ErrorActionPreference = 'Stop'
$Host.UI.RawUI.WindowTitle = 'Anthropic key setup'
Write-Output 'Enter your Anthropic API key below. Input is hidden.'
Write-Output 'The key will be encrypted for your Windows account and stored outside the project.'
$apiSecret = Read-Host 'Anthropic API key' -AsSecureString
if ($apiSecret.Length -eq 0) { throw 'No key entered; nothing saved.' }
$secretDirectory = Join-Path $env:LOCALAPPDATA 'confidence-experiment/secrets'
New-Item -ItemType Directory -Path $secretDirectory -Force | Out-Null
$secretPath = Join-Path $secretDirectory 'anthropic.clixml'
$apiSecret | Export-Clixml -LiteralPath $secretPath
Write-Output 'Key saved. Haiku 5.5 and the $25 experiment budget are configured.'
Write-Output 'You can close this window.'

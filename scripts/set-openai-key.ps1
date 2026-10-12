$ErrorActionPreference = 'Stop'
$Host.UI.RawUI.WindowTitle = 'OpenAI key setup'
Write-Output 'Enter your OpenAI API key below. Input is hidden.'
Write-Output 'The key will be encrypted for your Windows account and stored outside the project.'
$apiSecret = Read-Host 'OpenAI API key' -AsSecureString
if ($apiSecret.Length -eq 0) { throw 'No key entered; nothing saved.' }
$secretDirectory = Join-Path $env:LOCALAPPDATA 'confidence-experiment/secrets'
New-Item -ItemType Directory -Path $secretDirectory -Force | Out-Null
$apiSecret | Export-Clixml -LiteralPath (Join-Path $secretDirectory 'openai.clixml')
Write-Output 'Key saved. The experiment runner will load it automatically.'
Write-Output 'You can close this window.'

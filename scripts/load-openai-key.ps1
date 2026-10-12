if (-not $env:OPENAI_API_KEY) {
    $secretPath = Join-Path $env:LOCALAPPDATA 'confidence-experiment/secrets/openai.clixml'
    if (Test-Path -LiteralPath $secretPath) {
        $apiSecret = Import-Clixml -LiteralPath $secretPath
        $secretPointer = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($apiSecret)
        try {
            $env:OPENAI_API_KEY = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($secretPointer)
        } finally {
            [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($secretPointer)
        }
    }
}

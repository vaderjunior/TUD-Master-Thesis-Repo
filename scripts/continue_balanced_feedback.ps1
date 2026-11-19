param([switch]$CheckServerOnly)
$ErrorActionPreference='Stop'
$workspacePath=Split-Path -Parent $PSScriptRoot
$pythonPath=Join-Path $workspacePath '.venv\Scripts\python.exe'
Push-Location -LiteralPath $workspacePath
try {
    Write-Host 'Checking whether the same Qwen model is available before the longer saved-work checks.'
    & $pythonPath -X utf8 -B -u -m analysis.feedback.balanced.validate_server
    if ($LASTEXITCODE -ne 0) {
        Write-Host 'Server check failed. Saved work is untouched. This command has not started model inference.'
        exit $LASTEXITCODE
    }
    if (-not $CheckServerOnly) {
        & (Join-Path $PSScriptRoot 'astra_resume_sq3_balanced_v1.ps1')
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    }
} finally { Pop-Location }

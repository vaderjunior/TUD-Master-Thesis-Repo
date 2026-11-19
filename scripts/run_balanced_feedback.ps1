param([ValidateSet('run','validate','status')][string]$Mode='run')
$ErrorActionPreference='Stop'
$workspacePath=Split-Path -Parent $PSScriptRoot
$pythonPath=Join-Path $workspacePath '.venv\Scripts\python.exe'
Push-Location -LiteralPath $workspacePath
try {
    Write-Host 'Feedback: hateful-only versus balanced corrections, with correct/random controls.'
    Write-Host '38,475 predictions; 115,425 ordinary Qwen calls, plus bounded repairs/retries. No CUDA.'
    Write-Host 'One command runs all four internal batches and their offline audits automatically.'
    Write-Host 'Keep power connected and the computer awake. Use only one copy of this launcher.'
    Write-Host 'Completed work is verified and skipped. Preserve files if an integrity/request error stops it.'
    Write-Host 'After calls finish, let the offline audits and final integrity check finish too.'
    if ($Mode -eq 'run') {
        & $pythonPath -X utf8 -B -u -m analysis.common.vote_api run-all
    } else {
        & $pythonPath -X utf8 -B -u -m analysis.common.vote_api $Mode
    }
    if ($LASTEXITCODE -ne 0) { throw "Balanced-correction run stopped (exit $LASTEXITCODE). Preserve outputs and send Astra the message." }
} finally { Pop-Location }

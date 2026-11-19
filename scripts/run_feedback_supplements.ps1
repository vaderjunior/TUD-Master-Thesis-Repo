param([ValidateSet('run','validate','status')][string]$Mode='run')
$ErrorActionPreference='Stop'
$workspacePath=Split-Path -Parent $PSScriptRoot
$pythonPath=Join-Path $workspacePath '.venv\Scripts\python.exe'
Push-Location -LiteralPath $workspacePath
try {
    Write-Host 'Feedback additions: examples/definitions, human implied meanings, benign correction selection.'
    Write-Host '22,536 predictions; 67,608 ordinary Qwen calls, plus bounded repairs/retries. No CUDA.'
    Write-Host 'Four internal batches; each receives an automatic offline audit before the next starts.'
    Write-Host 'Keep power connected and the computer awake. Use one copy of this launcher.'
    Write-Host 'Verified work is skipped. Unknown interrupted requests stop for review; preserve all files.'
    Write-Host 'This stops before the final balanced-correction update study, whose pools need baseline review.'
    if ($Mode -eq 'run') {
        & $pythonPath -X utf8 -B -u -m analysis.feedback.supplements.run run-all
    } else {
        & $pythonPath -X utf8 -B -u -m analysis.feedback.supplements.run $Mode
    }
    if ($LASTEXITCODE -ne 0) { throw "SQ3 additions stopped (exit $LASTEXITCODE). Preserve outputs and send Astra the message." }
} finally { Pop-Location }

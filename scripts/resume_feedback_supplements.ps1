param([ValidateSet('run','validate','status')][string]$Mode='run')
$ErrorActionPreference='Stop'
$workspacePath=Split-Path -Parent $PSScriptRoot
$pythonPath=Join-Path $workspacePath '.venv\Scripts\python.exe'
Push-Location -LiteralPath $workspacePath
try {
    Write-Host 'Feedback additions: reviewed connection-outage recovery, 5 October.'
    Write-Host 'Saved work is verified and skipped. Original failures and successful votes are retained.'
    Write-Host 'Only eight reviewed failed calls receive at most five additional attempts each.'
    Write-Host 'All prompts, model settings, ordinary budgets and other retry limits stay frozen. No CUDA.'
    Write-Host 'One copy only. Keep power/internet connected and computer awake; audits may take time.'
    if ($Mode -eq 'run') {
        & $pythonPath -X utf8 -B -u -m analysis.feedback.supplements.resume run-all
    } else {
        & $pythonPath -X utf8 -B -u -m analysis.feedback.supplements.resume $Mode
    }
    if ($LASTEXITCODE -ne 0) { throw "SQ3 recovery stopped (exit $LASTEXITCODE). Preserve outputs and send Astra the message." }
} finally { Pop-Location }

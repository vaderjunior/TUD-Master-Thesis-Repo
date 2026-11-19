param([ValidateSet('run','validate','status')][string]$Mode='run')
$ErrorActionPreference='Stop'
$workspacePath=Split-Path -Parent $PSScriptRoot
$pythonPath=Join-Path $workspacePath '.venv\Scripts\python.exe'
Push-Location -LiteralPath $workspacePath
try {
    Write-Host 'Feedback: correction-relevance diagnostic, all three batches with automatic offline audits.'
    if ($Mode -eq 'run') {
        Write-Host '16,800 predictions = 50,400 ordinary Qwen calls, plus bounded repairs/retries. No CUDA.'
        Write-Host 'Plan for a long run: roughly half the previous feedback batch call count, plus offline audits; duration can vary.'
        Write-Host 'Keep power connected and the computer awake. Saved work is verified and reused; any execution/audit failure stops the chain.'
        Write-Host 'Only the new relevance study runs. No previous feedback run, explanation, applicability or taxonomy experiment is started.'
        & $pythonPath -X utf8 -B -u -m analysis.feedback.relevance.run run-all
    } else {
        & $pythonPath -X utf8 -B -u -m analysis.feedback.relevance.run $Mode
    }
    if ($LASTEXITCODE -ne 0) { throw "Relevance study stopped (exit $LASTEXITCODE). Preserve files and send Astra the output; do not delete or reset results." }
} finally { Pop-Location }

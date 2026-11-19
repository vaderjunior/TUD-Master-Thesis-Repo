param([ValidateSet('run','validate','status')][string]$Mode='run')
$ErrorActionPreference='Stop'
$workspacePath=Split-Path -Parent $PSScriptRoot
$pythonPath=Join-Path $workspacePath '.venv\Scripts\python.exe'
Push-Location -LiteralPath $workspacePath
try {
    Write-Host 'Feedback: explanations/selective corrections, followed by taxonomy adaptation.'
    Write-Host '73,359 predictions; 220,077 ordinary Qwen calls, plus bounded repairs and retries. No CUDA.'
    Write-Host 'One command runs eight internal batches with automatic offline audits and separate study outputs.'
    Write-Host 'Keep power connected and the computer awake. No next batch starts after a failed audit.'
    Write-Host 'Verified completed work is skipped. Ambiguous interrupted requests stop for review; do not delete saved results.'
    if ($Mode -eq 'run') {
        & $pythonPath -X utf8 -B -u -m analysis.feedback.initial.run run-all
    } else {
        & $pythonPath -X utf8 -B -u -m analysis.feedback.initial.run $Mode
    }
    if ($LASTEXITCODE -ne 0) { throw "Combined SQ3 command stopped (exit $LASTEXITCODE). Preserve files and send Astra the output." }
} finally { Pop-Location }

param([ValidateSet('run','validate','status')][string]$Mode='run')
$ErrorActionPreference='Stop'
$workspacePath=Split-Path -Parent $PSScriptRoot
$pythonPath=Join-Path $workspacePath '.venv\Scripts\python.exe'
Push-Location -LiteralPath $workspacePath
try {
    Write-Host 'Feedback: all three remaining feedback-update batches, with an offline audit between batches.'
    if ($Mode -eq 'run') {
        Write-Host '33,552 predictions = 100,656 ordinary Qwen calls, plus bounded repairs/retries. No CUDA.'
        Write-Host 'Allow more than one night: baseline pace suggests about 28 hours of calls, plus offline audits and possible slowdown.'
        Write-Host 'Keep power connected and the computer awake. Saved work is verified and reused; any execution/audit failure stops the chain.'
        Write-Host 'The completed baseline will not be repeated. No manual action is needed between successful update batches.'
        & $pythonPath -X utf8 -B -u -m analysis.feedback.source.updates run-all
    } else {
        & $pythonPath -X utf8 -B -u -m analysis.feedback.source.updates $Mode
    }
    if ($LASTEXITCODE -ne 0) { throw "Feedback updates stopped (exit $LASTEXITCODE). Preserve all files and send Astra the output; do not delete or reset results." }
} finally { Pop-Location }

param(
    [ValidateSet('run','validate','status')][string]$Mode='run',
    [ValidateSet('stage_01')][string]$Stage='stage_01'
)
$ErrorActionPreference='Stop'
$workspacePath=Split-Path -Parent $PSScriptRoot
$pythonPath=Join-Path $workspacePath '.venv\Scripts\python.exe'
$reviewPath=Join-Path $workspacePath 'experiments\astra\sq3_full_kb_feedback_20260928_v1\baseline_api_v1\stage_reviews\stage_01.json'
Push-Location -LiteralPath $workspacePath
try {
    Write-Host 'Feedback: fresh full-KB baseline for the feedback/adaptation study.'
    if ($Mode -eq 'run') {
        Write-Host 'Batch 1 of 4: selection baseline and independent no-update reference.'
        Write-Host '2,433 predictions / 7,299 ordinary Qwen responses, plus bounded repairs/retries. No CUDA.'
        Write-Host 'Keep power connected and the computer awake. Saved predictions are verified and reused.'
        Write-Host 'This command runs only the baseline, audits saved responses offline, then stops for Astra.'
    }
    & $pythonPath -X utf8 -B -u -m analysis.feedback.source.run $Mode --stage $Stage
    if ($LASTEXITCODE -ne 0) { throw "SQ3 baseline stopped (exit $LASTEXITCODE). Preserve all files and tell Astra." }
    if ($Mode -eq 'run') {
        if (Test-Path -LiteralPath $reviewPath) {
            Write-Host 'A saved audit exists. Tell Astra to inspect; no result was overwritten.'
        } else {
            Write-Host 'Qwen baseline finished. Auditing all saved raw responses offline; this can take a while.'
            & $pythonPath -X utf8 -B -u -m analysis.feedback.source.audit --stage $Stage
            if ($LASTEXITCODE -ne 0) { throw 'Offline audit needs review. Preserve outputs and tell Astra; do not delete or repeat model work.' }
            Write-Host 'BASELINE AND AUDIT COMPLETE. Tell Astra to inspect. No corrections selected or next batch started.'
        }
    }
} finally { Pop-Location }

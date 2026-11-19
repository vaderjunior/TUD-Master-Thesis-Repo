param([ValidateSet('all','search','finish')][string]$Stage='all')
$ErrorActionPreference='Stop'
$workspacePath=Split-Path -Parent $PSScriptRoot
$pythonPath=Join-Path $workspacePath '.venv\Scripts\python.exe'
Push-Location -LiteralPath $workspacePath
try {
    Write-Host 'Study SQ2: fairer small-data RoBERTa training comparison.'
    Write-Host '720 training-fold fits, followed by up to 60 unique final models and saved predictions.'
    Write-Host 'All settings use TRAIN only. No Qwen calls, extra humans, or changes to the running SQ3 study.'
    Write-Host 'Enable CUDA, connect the full-power charger, and keep the computer awake. Use one copy of this launcher.'
    Write-Host 'Verified completed jobs are skipped. An interrupted fitting job restarts; earlier attempts stay saved.'
    Write-Host 'Inference resumes saved chunks. To resume after Ctrl+C or a restart, run this same command.'
    & $pythonPath -X utf8 -B -u -m training.roberta.run run --stage $Stage
    if ($LASTEXITCODE -ne 0) { throw 'RoBERTa study stopped. Preserve files and inspect the last message; do not delete outputs.' }
    if ($Stage -eq 'search') { Write-Host 'TRAIN SEARCH COMPLETE. Default all-stage command continues the study.' }
    else { Write-Host 'ROBERTA STUDY FINISHED. Inspect the saved results.' }
} finally { Pop-Location }

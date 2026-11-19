param([ValidateSet('all','search','finish')][string]$Stage='all')
$ErrorActionPreference='Stop'
$workspacePath=Split-Path -Parent $PSScriptRoot
$pythonPath=Join-Path $workspacePath '.venv\Scripts\python.exe'
Push-Location -LiteralPath $workspacePath
try {
    Write-Host 'Study SQ2: fresh shorter RoBERTa schedules on the original compact TRAIN pool.'
    Write-Host '1800 new TRAIN-fold fits; reuse 720 older fits; at most 20 new final checkpoints across five seeds.'
    Write-Host 'CUDA and full-power charger required. Keep the computer awake. One copy of this launcher only.'
    Write-Host 'Needs 28 GiB free disk initially, 3.5 GiB available RAM and 5 GiB free GPU memory.'
    Write-Host 'No Qwen calls or changes to existing experiments. The separate Qwen terminal may remain open.'
    Write-Host 'F: is not needed: old predictions are reused only after provenance checks, without reloading archived weights.'
    Write-Host 'Completed jobs are verified and skipped. Interrupted fitting restarts that one job; prediction chunks resume.'
    Write-Host 'Run this same command to resume. Keep all files. This stops before the separate Stage B study.'
    & $pythonPath -X utf8 -B -u -m training.roberta.short.run run --stage $Stage
    if ($LASTEXITCODE -ne 0) { throw 'Shorter-training study stopped. Preserve outputs and inspect the last message; do not bypass checks.' }
    if ($Stage -eq 'search') { Write-Host 'SHORT SEARCH COMPLETE. Normal command resumes with final fits and predictions.' }
    else { Write-Host 'SHORTER-TRAINING STUDY FINISHED. Inspect the saved results.' }
} finally { Pop-Location }

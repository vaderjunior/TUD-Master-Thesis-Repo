param([ValidateSet('run','preflight','verify')][string]$Mode='run')
$ErrorActionPreference='Stop'
$workspacePath=Split-Path -Parent $PSScriptRoot
$pythonPath=Join-Path $workspacePath '.venv\Scripts\python.exe'
Push-Location -LiteralPath $workspacePath
try {
    Write-Host 'Study SQ2 CPU baselines: full-data BGE, compact/full word-character classifier, saved RoBERTa cutoffs.'
    Write-Host 'No CUDA, no Qwen calls. Wait for the current CUDA training to finish. One copy of this launcher.'
    Write-Host 'Needs at least 6 GiB AVAILABLE RAM and 15 GiB free disk. A separate Qwen queue may continue if these checks pass.'
    Write-Host 'A TRAIN-only technical pilot runs first; then one head at a time. Keep power connected and the computer awake.'
    Write-Host 'Completed candidate fits are verified and reused. Interrupted individual fits restart with old attempts preserved.'
    Write-Host 'This stops after saved CPU predictions; Study will inspect the results before accepting scientific conclusions.'
    & $pythonPath -X utf8 -B -u -m training.cpu.run $Mode
    if ($LASTEXITCODE -ne 0) { throw 'CPU baseline queue stopped. Keep outputs and inspect the last message. Do not bypass a failed check.' }
    if ($Mode -eq 'run') { Write-Host 'CPU BASELINES FINISHED. Inspect the saved results.' }
} finally { Pop-Location }

$ErrorActionPreference='Stop'
$workspacePath=Split-Path -Parent $PSScriptRoot
$pythonPath=Join-Path $workspacePath '.venv\Scripts\python.exe'
Push-Location -LiteralPath $workspacePath
try {
    Write-Host 'Study SQ2: matched type classifiers on ten preserved 67-annotation pools.'
    Write-Host '1800 TRAIN search fits; up to150 final head models; fixed gate and frozen-feature references.'
    Write-Host 'CUDA and full-power charger required. Keep awake. Allow about70GiB for retained checkpoints.'
    Write-Host 'No Qwen calls. One copy only. Verified jobs are reused; interrupted fitting jobs restart with old attempts kept.'
    Write-Host 'This stops after the matched type-pool block; text-budget/full-data/TF-IDF studies are separate.'
    & $pythonPath -X utf8 -B -u -m training.matched_annotations.run run
    if ($LASTEXITCODE -ne 0) { throw 'Matched type study stopped. Keep outputs and inspect the last message. Do not delete files or bypass checks.' }
    Write-Host 'MATCHED TYPE STUDY FINISHED. Inspect the saved results.'
} finally { Pop-Location }

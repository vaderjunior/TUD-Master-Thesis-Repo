param([ValidateSet('validate','run','analyze')][string]$Stage='validate')
$ErrorActionPreference='Stop'
$workspacePath=Split-Path -Parent $PSScriptRoot
$pythonPath=Join-Path $workspacePath '.venv\Scripts\python.exe'
$env:CUDA_VISIBLE_DEVICES=''
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
$env:OPENBLAS_NUM_THREADS='1'
$env:NUMEXPR_NUM_THREADS='1'
$env:HF_HUB_OFFLINE='1'
$env:TRANSFORMERS_OFFLINE='1'
$env:TOKENIZERS_PARALLELISM='false'
Push-Location -LiteralPath $workspacePath
try {
    Write-Host 'SQ3: feedback from the existing two-person review.'
    Write-Host '24 texts, three conditions, 648 ordinary Qwen calls only when -Stage run is requested.'
    Write-Host 'Default is offline validation. Let the current queue finish and be reviewed first.'
    & $pythonPath -X utf8 -B -u -m analysis.feedback.human.feedback_api validate
    if ($LASTEXITCODE -ne 0) { throw 'Human-feedback validation failed. Preserve all evidence.' }
    if ($Stage -eq 'run') {
        & $pythonPath -X utf8 -B -u -m analysis.feedback.human.feedback_api run-all
        if ($LASTEXITCODE -ne 0) { throw 'Human-feedback execution stopped. Preserve outputs and send Astra the message.' }
    } elseif ($Stage -eq 'analyze') {
        & $pythonPath -X utf8 -B -u -m analysis.feedback.human.feedback_api analyze
        if ($LASTEXITCODE -ne 0) { throw 'Human-feedback analysis stopped. Preserve outputs for review.' }
    }
} finally { Pop-Location }

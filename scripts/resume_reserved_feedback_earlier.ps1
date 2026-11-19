param([ValidateSet('run','validate')][string]$Stage='run')
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
$env:ANONYMIZED_TELEMETRY='False'
Push-Location -LiteralPath $workspacePath
try {
    Write-Host 'Astra A15: reviewed recovery after the Windows shutdown.'
    Write-Host '31,184 completed predictions and 13 saved votes in partial items are preserved.'
    Write-Host 'Eight unknown request outcomes are logged; only remaining original retry slots are used.'
    Write-Host 'Same frozen prompts, eight workers and sixteen batches. No CUDA. One Qwen queue only.'
    Write-Host 'Completed batches/items are verified and skipped; checking them can take several minutes.'
    & $pythonPath -X utf8 -B -u -m analysis.feedback.reserved.recovery validate
    if ($LASTEXITCODE -ne 0) { throw 'A15 recovery validation stopped. Preserve outputs and send Astra the message.' }
    if ($Stage -eq 'run') {
        & $pythonPath -X utf8 -B -u -m analysis.feedback.reserved.recovery run-all
        if ($LASTEXITCODE -ne 0) { throw 'A15 recovery stopped. Preserve all outputs and send Astra the message.' }
    }
} finally { Pop-Location }

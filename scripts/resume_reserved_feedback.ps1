param([ValidateSet('run','validate')][string]$Mode='run')
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
    Write-Host 'SQ3 Qwen continuation: reviewed connection outage, 16 October.'
    Write-Host '82,630 completed predictions and five partial votes were preserved at preparation.'
    Write-Host 'Both outage histories remain recorded. Only eight failed calls get up to five extra attempts.'
    Write-Host 'Saved predictions are verified and skipped. Verification can take considerable time.'
    Write-Host 'Same frozen prompts/model, eight workers, sixteen batches, no CUDA, one queue only.'
    if ($Mode -eq 'run') {
        & $pythonPath -X utf8 -B -u -m analysis.feedback.reserved.resume run-all
    } else {
        & $pythonPath -X utf8 -B -u -m analysis.feedback.reserved.resume validate
    }
    if ($LASTEXITCODE -ne 0) { throw 'SQ3 continuation stopped. Preserve all outputs and send Astra the complete message; do not delete stop files.' }
} finally { Pop-Location }

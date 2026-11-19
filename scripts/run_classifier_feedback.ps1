param([ValidateSet('prepare','static','iterative')][string]$Stage='static')
$ErrorActionPreference='Stop'
$workspacePath=Split-Path -Parent $PSScriptRoot
$pythonPath=Join-Path $workspacePath '.venv\Scripts\python.exe'
$env:CUDA_VISIBLE_DEVICES=''
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
$env:OPENBLAS_NUM_THREADS='1'
$env:HF_HUB_OFFLINE='1'
$env:TRANSFORMERS_OFFLINE='1'
Push-Location -LiteralPath $workspacePath
try {
    Write-Host 'Feedback A18: same correction labels, frozen BGE, small binary classifier refit.'
    Write-Host 'No CUDA or Qwen. Requires 3.5 GiB free RAM for fitting; the watchdog preserves the unrelated Qwen job.'
    & $pythonPath -X utf8 -B -u -m training.feedback.static prepare
    if ($LASTEXITCODE -ne 0) { throw 'A18 preparation stopped; preserve outputs and inspect the last error.' }
    if ($Stage -ne 'prepare') {
        & $pythonPath -X utf8 -B -u -m training.feedback.static $Stage
        if ($LASTEXITCODE -ne 0) { throw 'A18 CPU work stopped; preserve files. Do not bypass the resource or integrity check.' }
    }
} finally { Pop-Location }

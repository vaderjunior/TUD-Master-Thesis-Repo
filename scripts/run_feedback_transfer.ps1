param([ValidateSet('all','prepare','run','status')][string]$Stage='all')
$ErrorActionPreference='Stop'
$workspacePath=Split-Path -Parent $PSScriptRoot
$pythonPath=Join-Path $workspacePath '.venv\Scripts\python.exe'
$env:CUDA_VISIBLE_DEVICES=''
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
$env:OPENBLAS_NUM_THREADS='1'
$env:TOKENIZERS_PARALLELISM='false'
$env:HF_HUB_OFFLINE='1'
$env:TRANSFORMERS_OFFLINE='1'
Push-Location -LiteralPath $workspacePath
try {
    Write-Host 'Feedback A16: fixed reserved-correction policy on broader English TEST and HateCheck.'
    Write-Host '15,758 inputs; 189,096 predictions; 567,288 ordinary calls before repairs/retries. No CUDA.'
    Write-Host 'All internal batches receive automatic saved-response audits. Keep power connected and the computer awake.'
    Write-Host 'Wait until A15 is complete and reviewed. One Qwen queue only. Deferred SQ2 is not started.'
    $steps=@()
    if ($Stage -eq 'status') { $steps+=,@('analysis.feedback.transfer.run','status') }
    if ($Stage -in @('all','prepare')) {
        $steps+=,@('analysis.feedback.transfer.prepare','all')
        $steps+=,@('analysis.feedback.transfer.run','prepare')
    }
    if ($Stage -in @('all','run')) {
        $steps+=,@('analysis.feedback.transfer.run','validate')
        $steps+=,@('analysis.feedback.transfer.run','run-all')
    }
    foreach ($step in $steps) {
        & $pythonPath -X utf8 -B -u -m @step
        if ($LASTEXITCODE -ne 0) { throw 'A16 stopped. Preserve outputs and inspect the final error; do not bypass a failed check.' }
    }
    if ($Stage -eq 'prepare') { Write-Host 'A16 OFFLINE READY. Zero real model calls; wait for A15 review before live execution.' }
} finally { Pop-Location }

param([ValidateSet('all','prepare','run','status')][string]$Stage='all')
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
    Write-Host 'Feedback A15: four ordinary examples plus one reserved correction, with matched controls.'
    Write-Host '149,850 predictions; 449,550 ordinary Qwen calls before bounded repairs/retries. No CUDA.'
    Write-Host 'One command runs sixteen internal batches, each followed by an automatic offline audit.'
    Write-Host 'Use one Qwen queue. Keep power connected and the computer awake. Preserve all outputs.'
    Write-Host 'Verified completed requests are skipped. Unknown interrupted requests stop for review.'
    if ($Stage -eq 'status') {
        & $pythonPath -X utf8 -B -u -m analysis.feedback.reserved.run status
        if ($LASTEXITCODE -ne 0) { throw 'A15 status check failed.' }
        return
    }
    $steps=@()
    if ($Stage -in @('all','prepare')) {
        $steps+=,@('analysis.feedback.reserved.run','resources')
        $steps+=,@('analysis.feedback.reserved.prepare')
        $steps+=,@('analysis.feedback.reserved.validate')
        $steps+=,@('analysis.feedback.reserved.run','test')
        $steps+=,@('analysis.feedback.reserved.run','prepare')
        $steps+=,@('analysis.feedback.reserved.run','rehearse')
    }
    if ($Stage -in @('all','run')) {
        $steps+=,@('analysis.feedback.reserved.run','validate')
        $steps+=,@('analysis.feedback.reserved.run','run-all')
    }
    foreach ($step in $steps) {
        Write-Host ('A15 step: '+($step -join ' '))
        & $pythonPath -X utf8 -B -u -m @step
        if ($LASTEXITCODE -ne 0) { throw 'A15 stopped. Preserve files and inspect the final error. Do not bypass a check.' }
    }
    if ($Stage -eq 'prepare') { Write-Host 'A15 OFFLINE READY. No real model calls made.' }
    else { Write-Host 'ALL SIXTEEN A15 BATCHES AND AUDITS COMPLETE. Review the saved results; no further study started.' }
} finally { Pop-Location }

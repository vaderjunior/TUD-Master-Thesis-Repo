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
    Write-Host 'Feedback five-example study: reviewed original prompts; no retrieval rerun.'
    Write-Host 'Cold retrieval was not exact: the six-input common exclusion check is recorded and required.'
    Write-Host 'No CUDA. May run beside one RoBERTa GPU launcher. Use only one Qwen launcher.'
    Write-Host '38,475 predictions; 115,425 ordinary calls before bounded repairs/retries.'
    Write-Host 'Keep power connected and the computer awake. Verified completed requests are preserved and skipped.'
    if ($Stage -eq 'status') {
        & $pythonPath -X utf8 -B -u -m analysis.feedback.insertion.run status
        if ($LASTEXITCODE -ne 0) { throw 'Status check failed.' }
        return
    }
    $steps=@()
    $steps+=,@('analysis.feedback.insertion.replay_review','verify')
    if ($Stage -in @('all','prepare')) {
        $steps+=,@('analysis.feedback.insertion.run','prepare')
        $steps+=,@('analysis.feedback.insertion.run','rehearse')
    }
    if ($Stage -in @('all','run')) {
        $steps+=,@('analysis.feedback.insertion.run','validate')
        $steps+=,@('analysis.feedback.insertion.run','run-all')
    }
    foreach ($step in $steps) {
        Write-Host ('A14 step: '+($step -join ' '))
        & $pythonPath -X utf8 -B -u -m @step
        if ($LASTEXITCODE -ne 0) { throw 'Five-example study stopped. Preserve all outputs and inspect the last error. Do not bypass checks.' }
    }
    if ($Stage -eq 'prepare') { Write-Host 'OFFLINE READY. No live model calls made.' }
    else { Write-Host 'ALL FOUR FIVE-EXAMPLE BATCHES AND AUDITS COMPLETE. Tell Astra to inspect, including the required six-input exclusion sensitivity.' }
} finally { Pop-Location }

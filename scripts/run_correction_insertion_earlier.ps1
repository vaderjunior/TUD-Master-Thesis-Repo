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
    Write-Host 'Feedback: corrections competing for the ordinary FIVE example slots.'
    Write-Host 'No CUDA or new embeddings. Safe beside one SQ2 CUDA launcher; CPU/disk phases can slow each other.'
    Write-Host 'One launcher only. Keep power connected and the computer awake. Preserve files after any error.'
    if ($Stage -eq 'status') {
        & $pythonPath -X utf8 -B -u -m analysis.feedback.insertion.run status
        if ($LASTEXITCODE -ne 0) { throw 'Status check failed.' }
        return
    }
    $steps=@()
    if ($Stage -in @('all','prepare')) {
        Write-Host 'First: cached CPU retrieval, two cold replays and independent checks. This can take hours; no Qwen calls yet.'
        $steps+=,@('analysis.feedback.insertion.prepare','validate')
        $steps+=,@('analysis.feedback.insertion.run','test')
        $steps+=,@('analysis.feedback.insertion.prepare','run')
        $steps+=,@('analysis.feedback.insertion.validate')
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
        if ($LASTEXITCODE -ne 0) { throw 'Five-example correction study stopped. Keep all outputs and inspect the last error. Do not bypass a failed check.' }
    }
    if ($Stage -eq 'prepare') { Write-Host 'OFFLINE PREPARATION COMPLETE. No live Qwen calls made.' }
    else { Write-Host 'FIVE-EXAMPLE CORRECTION STUDY AND AUTOMATIC AUDITS COMPLETE. Review the saved results the results.' }
} finally { Pop-Location }

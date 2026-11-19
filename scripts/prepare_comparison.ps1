param([ValidateSet('retrieval','encoders')][string]$Stage='retrieval')
$ErrorActionPreference='Stop'
$workspacePath=Split-Path -Parent $PSScriptRoot
$pythonPath=Join-Path $workspacePath '.venv\Scripts\python.exe'
Push-Location -LiteralPath $workspacePath
try {
    Write-Host 'Study offline later evaluation. No Qwen requests or new training.'
    if ($Stage -eq 'retrieval') {
        Write-Host 'CPU only; CUDA is not needed. This can run overnight or longer.'
        Write-Host 'Completed 100-text chunks are checked and reused. Keep one copy running.'
        $steps=@(
            @('analysis.common.cached_retrieval','run'),
            @('analysis.classification.validate_execution'),
            @('tests.common.test_execution','--study','broad'),
            @('analysis.common.execution','prepare','--study','broad'),
            @('analysis.common.execution','rehearse','--study','broad')
        )
    } else {
        Write-Host 'Fixed RoBERTa/GBERT inference; enable CUDA for this stage.'
        Write-Host 'This predicts with saved models. It does not train or change thresholds.'
        $steps=@(
            @('training.original.inference','validate'),
            @('training.original.inference','run'),
            @('training.original.validate')
        )
    }
    foreach ($step in $steps) {
        Write-Host ('Offline step: '+($step -join ' '))
        & $pythonPath -X utf8 -B -u -m @step
        if ($LASTEXITCODE -ne 0) { throw 'Offline preparation stopped. Preserve files and inspect the last message.' }
    }
    Write-Host 'OFFLINE STAGE COMPLETE. No live Qwen run started. Inspect the saved results.'
} finally { Pop-Location }

param()
$ErrorActionPreference='Stop'
$workspacePath=Split-Path -Parent $PSScriptRoot
$pythonPath=Join-Path $workspacePath '.venv\Scripts\python.exe'
Push-Location -LiteralPath $workspacePath
try {
    Write-Host 'Finish the reviewed CPU preparation from saved first-pass prompts.'
    Write-Host 'No retrieval rerun, CUDA or Qwen calls. Preserve the nonzero replay diagnostics.'
    $steps=@(
        @('analysis.common.review_replay','finalize'),
        @('analysis.classification.validate_execution'),
        @('tests.common.test_execution','--study','broad'),
        @('analysis.common.execution','prepare','--study','broad'),
        @('analysis.common.execution','rehearse','--study','broad')
    )
    foreach ($step in $steps) {
        Write-Host ('Offline step: '+($step -join ' '))
        & $pythonPath -X utf8 -B -u -m @step
        if ($LASTEXITCODE -ne 0) { throw 'Reviewed preparation stopped. Preserve files and send Study the message.' }
    }
    Write-Host 'REVIEWED CPU PREPARATION COMPLETE. Exact cold replay did not pass; first-pass snapshots are preserved.'
    Write-Host 'No live evaluation started. inspect the saved log the offline readiness receipts.'
} finally { Pop-Location }

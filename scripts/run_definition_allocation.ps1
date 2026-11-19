param(
    [ValidateSet('run','validate','status')][string]$Mode='run',
    [ValidateSet('stage_01','stage_02','stage_03','stage_04')][string]$Stage='stage_01'
)
$ErrorActionPreference='Stop'
$workspacePath=Split-Path -Parent $PSScriptRoot
$pythonPath=Join-Path $workspacePath '.venv\Scripts\python.exe'
$reviewPath=Join-Path $workspacePath "experiments\astra\sq1_full_kb_allocation_20260924_v1\api_execution_v1\stage_reviews\$Stage.json"
Push-Location -LiteralPath $workspacePath
try {
    Write-Host 'Study SQ1: two definitions, seven retrieved definitions, seven distributed across categories.'
    if ($Mode -eq 'run') {
        Write-Host "Stage: $Stage of 4. Ordinary Qwen calls; no local CUDA required."
        Write-Host 'Keep power connected and the computer awake. Saved predictions are checked and reused.'
        Write-Host 'This command runs one stage, audits its saved responses offline, then stops for Study to review.'
    }
    & $pythonPath -X utf8 -B -u -m analysis.knowledge.allocation.run $Mode --stage $Stage
    if ($LASTEXITCODE -ne 0) { throw "Allocation stage stopped (exit $LASTEXITCODE). Preserve all files and Inspect the saved results." }
    if ($Mode -eq 'run') {
        if (Test-Path -LiteralPath $reviewPath) {
            Write-Host 'An audit receipt already exists. Inspect the saved results.'
        } else {
            Write-Host 'Qwen stage finished. Checking saved responses offline; this may take 30 minutes or more. No new model calls.'
            & $pythonPath -X utf8 -B -u -m scratch.analysis.astra_sq1_allocation_stage_check_v1 --stage $Stage
            if ($LASTEXITCODE -ne 0) { throw 'Offline audit needs review. Preserve all outputs and Inspect the saved results.' }
            Write-Host 'STAGE AND OFFLINE AUDIT COMPLETE. Inspect the saved results.'
        }
    }
} finally { Pop-Location }

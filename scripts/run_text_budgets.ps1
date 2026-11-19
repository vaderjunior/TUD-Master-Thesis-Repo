param([ValidateSet('all','features','search','finish')][string]$Stage='all')
$ErrorActionPreference='Stop'
$workspacePath=Split-Path -Parent $PSScriptRoot
$pythonPath=Join-Path $workspacePath '.venv\Scripts\python.exe'
$planPath=Join-Path $workspacePath 'experiments\astra\sq2_stage_b_20261020_v1\plan.json'
if (-not (Test-Path -LiteralPath $planPath)) { throw 'Stage B is not sealed yet. Ask Study to finish the scope/storage decision and preparation before running.' }
Push-Location -LiteralPath $workspacePath
try {
    $studyPlan=Get-Content -LiteralPath $planPath -Raw | ConvertFrom-Json
    Write-Host 'Study SQ2: same total training-text budgets for BGE, frozen RoBERTa and trained RoBERTa.'
    Write-Host ('Sealed search grid: '+$studyPlan.grid+'. Nine fixed pools; four heads; one local GPU queue.')
    Write-Host 'Final weights use the predeclared disk locations. Keep all allocated drives connected and preserve every output.'
    Write-Host 'CUDA, full-power charger and an awake laptop required. The single separate Qwen terminal may remain open.'
    Write-Host 'No Qwen requests. No old experiment edits or automatic follow-on experiment.'
    Write-Host 'Run this same command to resume. Completed jobs are checked/skipped; only an interrupted fit restarts.'
    & $pythonPath -X utf8 -B -u -m training.text_budgets.run run --stage $Stage
    if ($LASTEXITCODE -ne 0) { throw 'Stage B stopped. Preserve all outputs and inspect the last message. Do not bypass its checks.' }
    if ($Stage -eq 'all' -or $Stage -eq 'finish') { Write-Host 'STAGE B FINISHED. Ask Study to review the saved results.' }
    else { Write-Host 'Requested stage finished; the normal command resumes the remaining work.' }
} finally { Pop-Location }

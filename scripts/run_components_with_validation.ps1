param(
    [ValidateSet('run','validate')][string]$Mode='run',
    [Parameter(Mandatory=$true)]
    [ValidateSet('stage_02','stage_03','stage_04')][string]$Stage
)
$ErrorActionPreference='Stop'
$workspacePath=Split-Path -Parent $PSScriptRoot
$pythonPath=Join-Path $workspacePath '.venv\Scripts\python.exe'
$runnerPath=Join-Path $PSScriptRoot 'astra_run_sq1_components_api_v2.ps1'
$reviewPath=Join-Path $workspacePath "experiments\astra\sq1_full_kb_components_20260921_v1\api_execution_v2\stage_reviews\$Stage.json"
Push-Location -LiteralPath $workspacePath
try {
    
    & $runnerPath -Mode $Mode -Stage $Stage
    if ($LASTEXITCODE -ne 0) { throw 'Component runner stopped; preserve the files and Inspect the saved results.' }
    if ($Mode -eq 'run') {
        if (Test-Path -LiteralPath $reviewPath) {
            Write-Host 'An audit receipt already exists. Inspect the saved results.'
        } else {
            Write-Host 'Model stage finished. Now checking saved files offline; this can take 30 minutes or more. No additional model calls.'
            & $pythonPath -X utf8 -B -u -m scratch.analysis.astra_sq1_components_stage_check_v1 --stage $Stage
            if ($LASTEXITCODE -ne 0) { throw 'Offline audit needs review. Keep all saved files and Inspect the saved results.' }
            Write-Host 'STAGE AND OFFLINE AUDIT COMPLETE. Inspect the saved results.'
        }
    }
} finally { Pop-Location }

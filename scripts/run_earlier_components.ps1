param(
    [ValidateSet('run','validate','status')][string]$Mode='run',
    [ValidateSet('stage_01','stage_02','stage_03','stage_04','stage_05','stage_06','stage_07','stage_08','stage_09','stage_10','stage_11','stage_12','stage_13')][string]$Stage='stage_01'
)
$ErrorActionPreference='Stop'
$workspacePath=Split-Path -Parent $PSScriptRoot
$pythonPath=Join-Path $workspacePath '.venv\Scripts\python.exe'
Push-Location -LiteralPath $workspacePath
try {
    Write-Host 'Study SQ1 full-KB knowledge components: matched examples, definitions and guidelines.'
    if ($Mode -eq 'run') {
        Write-Host "Stage: $Stage of 13. Ordinary Qwen API calls; no CUDA required."
        Write-Host 'Keep power connected and the computer awake. Completed predictions are checked and skipped.'
        Write-Host 'This command stops after one stage. Inspect the saved results.'
    }
    & $pythonPath -X utf8 -B -u -m analysis.knowledge.components.execution $Mode --stage $Stage
    if ($LASTEXITCODE -ne 0) { throw "Component stage stopped (exit $LASTEXITCODE). Keep all files and inspect the saved log." }
} finally { Pop-Location }

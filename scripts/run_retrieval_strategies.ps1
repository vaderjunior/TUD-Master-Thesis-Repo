param(
    [ValidateSet('run','validate','status')][string]$Mode='run',
    [ValidateSet('stage_01','stage_02','stage_03','stage_04','stage_05')][string]$Stage='stage_01'
)
$ErrorActionPreference='Stop'
$workspacePath=Split-Path -Parent $PSScriptRoot
$pythonPath=Join-Path $workspacePath '.venv\Scripts\python.exe'
Push-Location -LiteralPath $workspacePath
try {
    Write-Host 'Study SQ1 full-KB strategies: dense, BM25 and hybrid; one named Qwen stage.'
    if ($Mode -eq 'run') {
        Write-Host "Stage: $Stage. No CUDA required. Keep power connected and the computer awake."
        Write-Host 'Completed predictions are verified and skipped; saved replies are reused.'
        Write-Host 'The command stops after this stage. Report completion or any error to Study.'
    }
    & $pythonPath -X utf8 -B -u -m analysis.knowledge.strategies.run $Mode --stage $Stage
    if ($LASTEXITCODE -ne 0) { throw "SQ1 strategy stage stopped (exit $LASTEXITCODE). Preserve the files and inspect the saved log." }
} finally { Pop-Location }

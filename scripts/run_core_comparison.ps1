param(
    [ValidateSet('run','validate','status','preflight','audit-preflight')][string]$Mode='run',
    [string]$Stage='stage_01_primary'
)
$ErrorActionPreference='Stop'
$workspacePath=Split-Path -Parent $PSScriptRoot
$pythonPath=Join-Path $workspacePath '.venv\Scripts\python.exe'
Push-Location -LiteralPath $workspacePath
try {
    Write-Host 'Study core Qwen comparison: one named stage, preserved frozen prompts and response records.'
    if ($Mode -eq 'run') {
        Write-Host "Stage: $Stage. No local CUDA needed. Keep power connected and the computer awake."
        Write-Host 'Completed items are verified and skipped. This command stops after this stage; it does not advance automatically.'
    }
    & $pythonPath -X utf8 -B -u -m analysis.common.batch $Mode --stage $Stage
    if ($LASTEXITCODE -ne 0) { throw "Core API stage stopped (exit $LASTEXITCODE). Preserve the output and inspect the saved log." }
} finally { Pop-Location }

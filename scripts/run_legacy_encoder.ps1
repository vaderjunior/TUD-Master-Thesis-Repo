param([ValidateSet('run', 'validate', 'status', 'smoke')][string]$Mode = 'run')
$ErrorActionPreference = 'Stop'
$workspacePath = Split-Path -Parent $PSScriptRoot
$pythonPath = Join-Path $workspacePath '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pythonPath -PathType Leaf)) { throw 'Project Python was not found.' }
Push-Location -LiteralPath $workspacePath
try {
    if ($Mode -eq 'run') {
        Write-Host 'Study aligned encoders: 21 training jobs, then nine reused-head checks and raw DEV predictions.'
        Write-Host 'Enable CUDA, connect power, and keep this terminal open and the computer awake.'
        Write-Host 'Seven short smoke checks run first if not already verified. No API calls or TEST evaluation.'
        Write-Host 'Completed jobs are verified and skipped. Interrupted jobs restart in a new preserved attempt.'
        Write-Host 'Reserve several hours and at least 75 GiB free disk. Original models are preserved.'
    }
    & $pythonPath -X utf8 -B -u -m training.original.legacy $Mode
    if ($LASTEXITCODE -ne 0) {
        throw "Encoder batch stopped (Python exit $LASTEXITCODE). Saved attempts are preserved; inspect the saved log."
    }
} finally {
    Pop-Location
}

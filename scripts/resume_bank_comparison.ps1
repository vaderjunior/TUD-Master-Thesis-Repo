param([ValidateSet('run', 'validate', 'status')][string]$Mode = 'run')
$ErrorActionPreference = 'Stop'
$workspacePath = Split-Path -Parent $PSScriptRoot
$pythonPath = Join-Path $workspacePath '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pythonPath -PathType Leaf)) {
    throw "Project Python was not found: $pythonPath"
}
Push-Location -LiteralPath $workspacePath
try {
    if ($Mode -eq 'run') {
        Write-Host 'Study SQ1 continuation: reuse completed indexes and predictions.'
        Write-Host 'Recorded invalid predictions are preserved and reported, not resampled.'
        Write-Host 'CUDA must be enabled. Keep this terminal open and the computer awake.'
    }
    & $pythonPath -B -u -m analysis.knowledge.banks.build_continue $Mode
    if ($LASTEXITCODE -ne 0) {
        throw "Continuation stopped (Python exit $LASTEXITCODE). Read the error above; saved work is preserved."
    }
} finally {
    Pop-Location
}

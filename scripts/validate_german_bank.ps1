param([ValidateSet('run', 'validate', 'status')][string]$Mode = 'run')
$ErrorActionPreference = 'Stop'
$workspacePath = Split-Path -Parent $PSScriptRoot
$pythonPath = Join-Path $workspacePath '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pythonPath -PathType Leaf)) { throw 'Project Python was not found.' }
Push-Location -LiteralPath $workspacePath
try {
    if ($Mode -eq 'run') {
        Write-Host 'Study German KB overlap audit: local BGE-M3 only. No classifier/API calls.'
        Write-Host 'Enable CUDA, connect power, and keep this terminal open and the computer awake.'
        Write-Host 'Checks TRAIN against German dev/test texts without reading held-out gold labels.'
        Write-Host 'Completed embedding chunks are verified and reused when this command is repeated.'
        Write-Host 'The batch stops after the audit. Pool construction and classifier runs come after review.'
    }
    & $pythonPath -X utf8 -B -u -m analysis.knowledge.banks.german_audit $Mode
    if ($LASTEXITCODE -ne 0) {
        throw "German audit stopped (Python exit $LASTEXITCODE). Preserve the log and inspect the saved log."
    }
} finally {
    Pop-Location
}

param([ValidateSet('run', 'status')][string]$Mode = 'run')
$ErrorActionPreference = 'Stop'
$workspacePath = Split-Path -Parent $PSScriptRoot
$pythonPath = Join-Path $workspacePath '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pythonPath -PathType Leaf)) {
    throw "Project Python was not found: $pythonPath"
}
Push-Location -LiteralPath $workspacePath
try {
    if ($Mode -eq 'run') {
        Write-Host 'Study SQ1: 148 versus 56,109 eligible English examples; five retrieved each time.'
        Write-Host '12 runs, 13,788 planned generation calls plus repairs/retries. CUDA must be enabled.'
        Write-Host 'Keep this terminal open and prevent the computer from sleeping.'
        Write-Host 'Existing work is resumed; original thesis artifacts are not overwritten.'
    }
    & $pythonPath -B -u -m analysis.knowledge.banks.build $Mode
    if ($LASTEXITCODE -ne 0) {
        throw "Study batch stopped (Python exit $LASTEXITCODE). Read the error above; progress was saved."
    }
} finally {
    Pop-Location
}

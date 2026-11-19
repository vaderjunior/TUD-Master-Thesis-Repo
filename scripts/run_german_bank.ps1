param([ValidateSet('run','setup','validate','status')][string]$Mode='run')
$ErrorActionPreference='Stop'
$workspacePath=Split-Path -Parent $PSScriptRoot
$pythonPath=Join-Path $workspacePath '.venv\Scripts\python.exe'
Push-Location -LiteralPath $workspacePath
try {
    Write-Host 'Study German KB comparison: four conditions, two DEV subsets, three repeats.'
    if ($Mode -eq 'run') {
        Write-Host '24 RAG runs; 11,700 ordinary model samples plus repairs. Keep power connected and the computer awake.'
        Write-Host 'Completed items are verified and skipped on restart. Original experiments remain preserved.'
    }
    & $pythonPath -X utf8 -B -u -m scratch.analysis.astra_german_kb_run $Mode
    if ($LASTEXITCODE -ne 0) { throw "German comparison stopped (exit $LASTEXITCODE). Preserve the output and inspect the saved log." }
} finally { Pop-Location }

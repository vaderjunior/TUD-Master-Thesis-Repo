param([ValidateSet('run','validate','status')][string]$Mode='run')
$ErrorActionPreference='Stop'
$workspacePath=Split-Path -Parent $PSScriptRoot
$pythonPath=Join-Path $workspacePath '.venv\Scripts\python.exe'
Push-Location -LiteralPath $workspacePath
try {
    Write-Host 'Study: two diagnostic replays of the existing full KB index.'
    Write-Host 'CPU only; no CUDA, model training, API calls or prompt replacements. Allow roughly 30 minutes.'
    & $pythonPath -X utf8 -B -u -m scratch.analysis.astra_core_retrieval_replay_v1 $Mode
    if ($LASTEXITCODE -ne 0) { throw "Replay stopped (exit $LASTEXITCODE). Preserve output for Study." }
} finally { Pop-Location }

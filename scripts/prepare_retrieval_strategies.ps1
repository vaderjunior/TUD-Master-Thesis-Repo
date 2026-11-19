param([ValidateSet('run','validate')][string]$Mode='run')
$ErrorActionPreference='Stop'
$workspacePath=Split-Path -Parent $PSScriptRoot
$pythonPath=Join-Path $workspacePath '.venv\Scripts\python.exe'
Push-Location -LiteralPath $workspacePath
try {
    Write-Host 'Study SQ1: prepare full-KB dense, BM25 and hybrid prompts, then replay them.'
    Write-Host 'CPU only; no CUDA, Qwen/API calls, classifier training or new accuracy scoring.'
    Write-Host 'Use one terminal. Keep power connected and the computer awake. Old experiments are preserved.'
    & $pythonPath -X utf8 -B -u -m analysis.knowledge.strategies.prepare $Mode
    if ($LASTEXITCODE -ne 0) { throw "Preparation stopped (exit $LASTEXITCODE). Preserve outputs and inspect the saved log." }
} finally { Pop-Location }

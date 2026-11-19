param([ValidateSet('run','validate','status')][string]$Mode='run')
$ErrorActionPreference='Stop'
$workspacePath=Split-Path -Parent $PSScriptRoot
$pythonPath=Join-Path $workspacePath '.venv\Scripts\python.exe'
Push-Location -LiteralPath $workspacePath
try {
    Write-Host 'Study core retrieval preparation: two isolated indexes and 2590 saved RAG prompts.'
    Write-Host 'CPU only, verified cached vectors; no CUDA, API calls, classifier training or scoring.'
    & $pythonPath -X utf8 -B -u -m analysis.common.retrieval $Mode
    if ($LASTEXITCODE -ne 0) { throw "Retrieval preparation stopped (exit $LASTEXITCODE). Preserve outputs and inspect the saved log." }
} finally { Pop-Location }

param([ValidateSet('run','validate')][string]$Mode='run')
$ErrorActionPreference='Stop'
$workspacePath=Split-Path -Parent $PSScriptRoot
$pythonPath=Join-Path $workspacePath '.venv\Scripts\python.exe'
Push-Location -LiteralPath $workspacePath
try {
    Write-Host 'Study SQ1: prepare two-definition, seven-definition and proportional-definition prompts.'
    Write-Host 'CPU only. Uses saved vectors and fixed examples/guidelines; no CUDA or Qwen calls.'
    Write-Host 'Prepare all inputs and one cold replay, then stop for review. Preserve earlier experiments.'
    & $pythonPath -X utf8 -B -u -m analysis.knowledge.allocation.prepare $Mode
    if ($LASTEXITCODE -ne 0) { throw "Preparation stopped (exit $LASTEXITCODE). Preserve outputs and inspect the saved log." }
} finally { Pop-Location }

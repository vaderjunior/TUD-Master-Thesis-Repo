param([ValidateSet('run','validate')][string]$Mode='run')
$ErrorActionPreference='Stop'
$workspacePath=Split-Path -Parent $PSScriptRoot
$pythonPath=Join-Path $workspacePath '.venv\Scripts\python.exe'
Push-Location -LiteralPath $workspacePath
try {
    Write-Host 'Study SQ1: prepare the nine sampled KBs and check every prompt again after reopening.'
    Write-Host 'CPU only: no CUDA, Qwen/API calls, training or accuracy scoring.'
    Write-Host 'One terminal only. Keep power connected and the computer awake.'
    Write-Host 'Completed pool/pass results are verified and skipped. An incomplete attempt stops for Study to inspect.'
    & $pythonPath -X utf8 -B -u -m analysis.knowledge.sampled_banks.retrieval $Mode
    if ($LASTEXITCODE -ne 0) { throw "Pool preparation stopped (exit $LASTEXITCODE). Keep all outputs and inspect the saved log." }
} finally { Pop-Location }

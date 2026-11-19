param(
    [ValidateSet('status','validate','run')][string]$Mode='status',
    [ValidateSet('matched','broad')][string]$Study='matched'
)
$ErrorActionPreference='Stop'
$workspacePath=Split-Path -Parent $PSScriptRoot
$pythonPath=Join-Path $workspacePath '.venv\Scripts\python.exe'
Push-Location -LiteralPath $workspacePath
try {
    Write-Host "Study later evaluation: $Study / $Mode. No CUDA for Qwen requests."
    Write-Host 'Only explicit -Mode run contacts Qwen. Offline readiness and pinned model availability are required.'
    Write-Host 'Use one launcher. Each batch is audited automatically; verified completed items are skipped on resume.'
    Write-Host 'Unknown interrupted requests or server errors stop for review. Preserve all saved files.'
    & $pythonPath -X utf8 -B -u -m analysis.common.execution $Mode --study $Study
    if ($LASTEXITCODE -ne 0) { throw 'Later evaluation stopped. Preserve files and send Study the message.' }
} finally { Pop-Location }

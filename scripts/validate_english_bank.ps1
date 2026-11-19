param([ValidateSet('run','validate','status')][string]$Mode='run')
$ErrorActionPreference='Stop'
$workspacePath=Split-Path -Parent $PSScriptRoot
$pythonPath=Join-Path $workspacePath '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pythonPath -PathType Leaf)) { throw 'Project Python was not found.' }
Push-Location -LiteralPath $workspacePath
try {
    if ($Mode -eq 'run') {
        Write-Host 'Study English overlap audit: local BGE-M3 only. No classifier/API calls.'
        Write-Host 'Enable CUDA, connect power, keep one terminal open and the computer awake.'
        Write-Host '84,000 text embeddings, 657 resumable chunks; then blocked similarity comparison.'
        Write-Host 'The same command verifies and reuses completed chunks after interruption.'
        Write-Host 'This stops after the audit, before KB construction or classifier experiments.'
    }
    & $pythonPath -X utf8 -B -u -m analysis.knowledge.banks.english_audit $Mode
    if ($LASTEXITCODE -ne 0) { throw "English audit stopped (exit $LASTEXITCODE). Preserve the output and inspect the saved log." }
} finally { Pop-Location }

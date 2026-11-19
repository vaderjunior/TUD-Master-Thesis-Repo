$ErrorActionPreference = 'Stop'
$taskRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $taskRoot
$taskPython = Join-Path $taskRoot '.venv\Scripts\python.exe'
Write-Host 'Full-KB guideline study: cached CPU prompt preparation and one cold replay.'
Write-Host 'No CUDA, new embeddings or Qwen calls. First-pass prompts and any replay differences are preserved.'
& $taskPython -X utf8 -B -u -m analysis.knowledge.guidelines.prepare run
if ($LASTEXITCODE -ne 0) { throw 'Preparation stopped. Keep outputs and inspect the saved log.' }
& $taskPython -X utf8 -B -u -m analysis.knowledge.guidelines.verify_preparation full
if ($LASTEXITCODE -ne 0) { throw 'Independent audit stopped. Keep outputs and inspect the saved log.' }
Write-Host 'PREPARATION AND AUDIT COMPLETE. Inspect the saved results.'

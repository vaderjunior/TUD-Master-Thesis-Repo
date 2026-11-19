$ErrorActionPreference = 'Stop'
$astraRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$astraPython = Join-Path $astraRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $astraPython)) { throw 'Project Python environment is missing.' }
Set-Location -LiteralPath $astraRoot
Write-Host 'Feedback taxonomy: OFFLINE retrieval preparation and independent checks.'
Write-Host 'No Qwen/server calls, no new embeddings, no CUDA. This is not the later big Qwen queue.'
Write-Host 'You may use another terminal while the existing Qwen run continues. Leave that terminal alone.'
Write-Host 'Several hours may be needed. Keep the computer awake. Completed views are verified and skipped.'
Write-Host 'A saved complete prefix is checked before resuming; a truncated line stops safely for review.'
& $astraPython -X utf8 -B -u -m analysis.feedback.restoration.run run
if ($LASTEXITCODE -ne 0) { throw 'Offline preparation stopped. Keep all files and send Astra the output.' }
Write-Host 'Offline preparation finished. Tell Astra to inspect. No model batch starts automatically.'

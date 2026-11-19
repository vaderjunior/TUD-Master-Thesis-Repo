param([ValidateSet('run','preflight','verify','pilot-only')][string]$Mode='run')
$ErrorActionPreference='Stop'
$workspacePath=Split-Path -Parent $PSScriptRoot
$pythonPath=Join-Path $workspacePath '.venv\Scripts\python.exe'
Push-Location -LiteralPath $workspacePath
try {
    Write-Host 'Study SQ2 CPU baselines v2: same scientific procedures, measured memory policy.'
    Write-Host 'Requires 3.5 GiB AVAILABLE RAM and 15 GiB free disk. No CUDA or Qwen calls.'
    Write-Host 'One CPU queue only. Your existing Qwen run may continue. Keep power connected and the computer awake.'
    Write-Host 'Each worker is watched. Low memory stops only this CPU worker; completed work is preserved for resume.'
    Write-Host 'Worker logs are saved under experiments/astra/sq2_cpu_baselines_20261012_v2/execution.'
    Write-Host 'One head at a time; a head can be quiet while fitting. Start/completion messages appear here.'
    & $pythonPath -X utf8 -B -u -m training.cpu.memory $Mode
    if ($LASTEXITCODE -ne 0) { throw 'CPU queue stopped. Keep outputs and send Study the message. Do not bypass the check.' }
    if ($Mode -eq 'run') { Write-Host 'CPU BASELINES FINISHED. Inspect the saved results.' }
} finally { Pop-Location }

param([switch]$CheckOnly, [switch]$LauncherCheck)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $projectRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python)) { throw 'Project Python is missing.' }
$env:PYTHONUTF8 = '1'
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:CUDA_VISIBLE_DEVICES = ''
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:NUMEXPR_NUM_THREADS = '1'
$env:TOKENIZERS_PARALLELISM = 'false'
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
$parseTokens = $null
$parseErrors = $null
$null = [System.Management.Automation.Language.Parser]::ParseFile($PSCommandPath, [ref]$parseTokens, [ref]$parseErrors)
if ($parseErrors.Count -ne 0) { throw 'PowerShell syntax check failed.' }
if (-not ('PolicyRunPower' -as [type])) {
    Add-Type -TypeDefinition @'
using System;
using System.Runtime.InteropServices;
public static class PolicyRunPower {
    [DllImport("kernel32.dll", SetLastError=true)]
    public static extern uint SetThreadExecutionState(uint flags);
}
'@
}
Push-Location $projectRoot
try {
    $awake = [PolicyRunPower]::SetThreadExecutionState([uint32]2147483649)
    if ($awake -eq 0) { throw 'Windows could not keep the laptop awake.' }
    if ($LauncherCheck) {
        Write-Host 'Launcher syntax and keep-awake checks passed. No API calls.'
        return
    }
    & $python -X utf8 -B -m analysis.knowledge.policy.run check
    if ($LASTEXITCODE -ne 0) { throw 'Offline preflight failed. Keep all saved work and send the last message.' }
    if ($CheckOnly) { return }
    & $python -X utf8 -B -m analysis.knowledge.policy.run run
    if ($LASTEXITCODE -ne 0) { throw 'Complete-policy run stopped. Keep every saved file and send the last message. Do not delete files or reset retries.' }
}
finally {
    $null = [PolicyRunPower]::SetThreadExecutionState([uint32]2147483648)
    Pop-Location
}

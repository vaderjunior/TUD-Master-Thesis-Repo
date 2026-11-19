param(
    [ValidateSet('matched','broad')][string]$Study='matched',
    [switch]$CheckOnly,
    [switch]$LauncherCheck
)
$ErrorActionPreference='Stop'
$workspacePath=Split-Path -Parent $PSScriptRoot
$pythonPath=Join-Path $workspacePath '.venv\Scripts\python.exe'
$env:CUDA_VISIBLE_DEVICES=''
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
$env:OPENBLAS_NUM_THREADS='1'
$env:NUMEXPR_NUM_THREADS='1'
$env:TOKENIZERS_PARALLELISM='false'
$env:HF_HUB_OFFLINE='1'
$env:TRANSFORMERS_OFFLINE='1'
$keepAwakeSet=$false
$sleepRestored=$false
$syntaxTokens=$null
$syntaxErrors=$null
[void][System.Management.Automation.Language.Parser]::ParseFile($PSCommandPath,[ref]$syntaxTokens,[ref]$syntaxErrors)
if ($syntaxErrors.Count -ne 0) { throw 'Launcher syntax check failed.' }
Push-Location -LiteralPath $workspacePath
try {
    Add-Type -TypeDefinition @'
using System;
using System.Runtime.InteropServices;
public static class Sq2QwenAwakeV1 {
    [DllImport("kernel32.dll", SetLastError=true)]
    public static extern uint SetThreadExecutionState(uint flags);
}
'@
    if ([Sq2QwenAwakeV1]::SetThreadExecutionState([uint32]2147483649) -eq 0) {
        throw 'Could not keep the laptop awake. No model call started.'
    }
    $keepAwakeSet=$true
    if (-not $LauncherCheck) {
        Write-Host "Checking the SQ2 $Study study. The GPU training is separate."
        & $pythonPath -X utf8 -B -u -m analysis.classification.broad.execution check --study $Study
        if ($LASTEXITCODE -ne 0) { throw 'Offline check failed. Keep all files; no queue started.' }
        if (-not $CheckOnly) {
            & $pythonPath -X utf8 -B -u -m analysis.classification.broad.execution run --study $Study
            if ($LASTEXITCODE -ne 0) { throw 'SQ2 stopped. Keep every output and send the last message for review. Do not delete files or bypass checks.' }
        } else { Write-Host 'Offline check passed. No Qwen calls made.' }
    }
} finally {
    if ($keepAwakeSet) {
        $sleepRestored=([Sq2QwenAwakeV1]::SetThreadExecutionState([uint32]2147483648) -ne 0)
    }
    Pop-Location
}
if ($LauncherCheck) {
    $receiptPath=Join-Path $workspacePath 'experiments\astra\later_evaluation_20261008_v1\concurrency_20261029_v1\launcher_check.json'
    if (Test-Path -LiteralPath $receiptPath) { throw 'Keep the existing launcher check receipt.' }
    $receipt=[ordered]@{
        parsed=($syntaxErrors.Count -eq 0)
        set_succeeded=$keepAwakeSet
        restored=$sleepRestored
        launcher_sha256=(Get-FileHash -LiteralPath $PSCommandPath -Algorithm SHA256).Hash.ToLowerInvariant()
        actual_Qwen_calls=0
        actual_model_list_GETs=0
    }
    [System.IO.File]::WriteAllText($receiptPath,($receipt | ConvertTo-Json -Depth 5),[System.Text.UTF8Encoding]::new($false))
    if (-not $sleepRestored) { throw 'Sleep handling could not be restored.' }
    Write-Host 'Launcher syntax and keep-awake check passed. No Python run or model call.'
}

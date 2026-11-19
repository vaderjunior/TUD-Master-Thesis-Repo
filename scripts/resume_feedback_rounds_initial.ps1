param([switch]$CheckOnly, [switch]$LauncherCheck)
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
public static class FeedbackResumeFeedbackRoundsInitialState {
    [DllImport("kernel32.dll", SetLastError=true)]
    public static extern uint SetThreadExecutionState(uint flags);
}
'@
    if ([FeedbackResumeFeedbackRoundsInitialState]::SetThreadExecutionState([uint32]2147483649) -eq 0) {
        throw 'Could not keep the laptop awake. No model call started.'
    }
    $keepAwakeSet=$true
    if (-not $LauncherCheck) {
        Write-Host 'Resume the four SQ3 feedback rounds at 128 requests.'
        Write-Host 'Saved answers and the outage history stay preserved. Keep the laptop plugged in and online.'
        & $pythonPath -X utf8 -B -u -m analysis.feedback.rounds.resume_earlier validate
        if ($LASTEXITCODE -ne 0) { throw 'Offline recovery check failed. Preserve all files; no queue started.' }
        if (-not $CheckOnly) {
            & $pythonPath -X utf8 -B -u -m analysis.feedback.rounds.resume_earlier run
            if ($LASTEXITCODE -ne 0) { throw 'Feedback rounds stopped. Keep all saved work and inspect the last error. Do not delete files or bypass checks.' }
        } else { Write-Host 'Offline check complete. No Qwen calls made.' }
    }
} finally {
    if ($keepAwakeSet) {
        $sleepRestored=([FeedbackResumeFeedbackRoundsInitialState]::SetThreadExecutionState([uint32]2147483648) -ne 0)
    }
    Pop-Location
}
if ($LauncherCheck) {
    $receiptPath=Join-Path $workspacePath 'experiments\astra\sq3_iterative_20261012_v1\execution\recovery_20261024_v1\launcher_check.json'
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

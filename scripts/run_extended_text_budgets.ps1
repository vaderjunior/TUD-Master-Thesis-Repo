param([ValidateSet('all','features','search','finish')][string]$Stage='all')
$ErrorActionPreference='Stop'
$workspacePath=Split-Path -Parent $PSScriptRoot
$pythonPath=Join-Path $workspacePath '.venv\Scripts\python.exe'
$planPath=Join-Path $workspacePath 'experiments\astra\sq2_stage_b20_20261021_v1\plan.json'
if (-not (Test-Path -LiteralPath $planPath)) { throw 'The reduced equal-budget comparison is not sealed yet. Preserve the prepared files.' }
Push-Location -LiteralPath $workspacePath
$keepAwakeSet=$false
try {
    $studyPlan=Get-Content -LiteralPath $planPath -Raw | ConvertFrom-Json
    if ($studyPlan.grid -ne 'fixed20_both_losses') { throw 'Unexpected study scope. No experiment started.' }
    Add-Type -TypeDefinition @'
using System;
using System.Runtime.InteropServices;
public static class StudyStageB20Awake {
    [DllImport("kernel32.dll", SetLastError=true)]
    public static extern uint SetThreadExecutionState(uint flags);
}
'@
    if ([StudyStageB20Awake]::SetThreadExecutionState([uint32]2147483649) -eq 0) {
        throw 'Could not keep the laptop awake. No experiment started.'
    }
    $keepAwakeSet=$true
    Write-Host 'SQ2 equal-data comparison: nine training pools, four outputs, both losses, every seed.'
    Write-Host 'Fixed 20 epochs and learning rate 0.00002; TRAIN-only loss/cutoff choice.'
    Write-Host 'Models go to D:, logs/predictions to C:. F: is not needed for this run.'
    Write-Host 'Plan about 2-3 continuous days; execution stops at 96 elapsed hours for review if unfinished.'
    Write-Host 'Keep the full-power charger connected. Automatic system sleep is suppressed while this launcher is open.'
    Write-Host 'Leave the lid open. Display sleep is fine. Keep the single separate Qwen terminal running.'
    Write-Host 'No Qwen calls or automatic follow-on study. Use this same command to resume an interruption.'
    & $pythonPath -X utf8 -B -u -m training.text_budgets.extended.run run --stage $Stage
    if ($LASTEXITCODE -ne 0) { throw 'Equal-budget run stopped. Preserve every output and inspect the last message. Do not bypass its checks.' }
    if ($Stage -eq 'all' -or $Stage -eq 'finish') { Write-Host 'EQUAL-BUDGET RUN COMPLETE. Ask Study for the separate scientific review.' }
    else { Write-Host 'Requested stage complete; the normal command resumes the remaining work.' }
} finally {
    if ($keepAwakeSet) { [void][StudyStageB20Awake]::SetThreadExecutionState([uint32]2147483648) }
    Pop-Location
}

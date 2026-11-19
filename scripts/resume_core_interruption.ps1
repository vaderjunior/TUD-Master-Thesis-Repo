param([switch]$ValidateOnly)
$ErrorActionPreference='Stop'
$workspacePath=Split-Path -Parent $PSScriptRoot
$recoveryRelative='experiments\astra\full_kb_core_20260915_v1\api_execution_v1\recovery_stage02_20260917_v1'
$recoveryDirectory=Join-Path $workspacePath $recoveryRelative
$recoveryPlanPath=Join-Path $recoveryDirectory 'plan.json'
$recoveryReceiptPath=Join-Path $recoveryDirectory 'complete.json'
$expectedPlanHash='85bb302af25258a2d4e5906dc98a05fc1e78603034b6ca1a3de032d4d4544045'
$pythonPath=Join-Path $workspacePath '.venv\Scripts\python.exe'
function Assert-RecoveryHash([string]$FilePath,[string]$ExpectedHash) {
    if (-not (Test-Path -LiteralPath $FilePath -PathType Leaf)) { throw "Required recovery file is missing: $FilePath" }
    $actualHash=(Get-FileHash -LiteralPath $FilePath -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($actualHash -ne $ExpectedHash.ToLowerInvariant()) { throw "Recovery hash mismatch: $FilePath. inspect the saved log." }
}
Push-Location -LiteralPath $workspacePath
try {
    Assert-RecoveryHash $recoveryPlanPath $expectedPlanHash
    $recoveryPlan=Get-Content -LiteralPath $recoveryPlanPath -Raw | ConvertFrom-Json
    Assert-RecoveryHash (Join-Path $workspacePath 'scratch\analysis\astra_core_stage02_recovery_v1.py') $recoveryPlan.source_sha256
    Assert-RecoveryHash (Join-Path $recoveryDirectory 'tests.json') $recoveryPlan.tests_sha256
    Assert-RecoveryHash (Join-Path $workspacePath 'experiments\astra\full_kb_core_20260915_v1\api_execution_v1\plan.json') $recoveryPlan.execution_plan_sha256
    if ($recoveryPlan.stage -ne 'stage_02_primary' -or $recoveryPlan.completed_items -ne 2418 -or $recoveryPlan.missing_outcomes.Count -ne 8) {
        throw 'Unexpected reviewed recovery scope. inspect the saved log.'
    }
    if (Test-Path -LiteralPath $recoveryReceiptPath) {
        $recoveryReceipt=Get-Content -LiteralPath $recoveryReceiptPath -Raw | ConvertFrom-Json
        if ($recoveryReceipt.status -ne 'interruption_records_added_original_stage_command_can_resume' -or $recoveryReceipt.recovery_plan_sha256 -ne $expectedPlanHash) {
            throw 'Unexpected recovery receipt. inspect the saved log.'
        }
        foreach ($entry in $recoveryReceipt.administrative_unknown_outcomes_added.PSObject.Properties) {
            Assert-RecoveryHash (Join-Path $workspacePath $entry.Name) $entry.Value
        }
    }
    if ($ValidateOnly) {
        Write-Host 'Recovery launcher and frozen scope verified. No recovery writes or API calls performed.'
        return
    }
    if (-not (Test-Path -LiteralPath $recoveryReceiptPath)) {
        Write-Host 'Checking and preserving 2,418 completed predictions, then recording eight interrupted request outcomes as unknown.'
        Write-Host 'This offline file check can take several minutes. No predictions are being discarded or regenerated.'
        & $pythonPath -X utf8 -B -u -m scratch.analysis.astra_core_stage02_recovery_v1 apply
        if ($LASTEXITCODE -ne 0) { throw 'Recovery stopped. Preserve the message and inspect the saved log; do not delete files.' }
    } else {
        Write-Host 'The recorded recovery is already applied and verified.'
    }
    Write-Host 'Resuming stage 02 with the original runner. Completed predictions are checked and skipped; partial replies are reused.'
    Write-Host 'Keep this terminal open and the computer awake. No local CUDA needed. This stops after stage 02.'
    & (Join-Path $PSScriptRoot 'astra_run_core_api_v1.ps1') -Stage stage_02_primary
} finally { Pop-Location }

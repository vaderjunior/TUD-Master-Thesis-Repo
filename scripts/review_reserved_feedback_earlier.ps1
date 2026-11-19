param(
    [ValidateSet('validate','test','review','analyze')]
    [string]$Stage = 'validate'
)
$ErrorActionPreference = 'Stop'
$reviewRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $reviewRoot
$env:OMP_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:CUDA_VISIBLE_DEVICES = ''
$reviewPython = Join-Path $reviewRoot '.venv/Scripts/python.exe'
if ($Stage -eq 'test') {
    & $reviewPython -X utf8 -B -m unittest tests.feedback.reserved.test_rates tests.feedback.reserved.test_support tests.feedback.reserved.test_bounds tests.feedback.reserved.test_review scratch.analysis.test_astra_sq3_reserved_report_v1
} elseif ($Stage -eq 'review') {
    & $reviewPython -X utf8 -B -m analysis.feedback.reserved.review_earlier --review
} elseif ($Stage -eq 'analyze') {
    & $reviewPython -X utf8 -B -m analysis.feedback.reserved.scoring
} else {
    & $reviewPython -X utf8 -B -m analysis.feedback.reserved.review_earlier
}
if ($LASTEXITCODE -ne 0) { throw "Offline review step failed. Preserve its output and inspect it before continuing." }

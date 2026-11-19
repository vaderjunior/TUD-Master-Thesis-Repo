param([ValidateSet('validate','review','analyze')][string]$Stage='validate')
$ErrorActionPreference='Stop'
$reviewRoot=Split-Path -Parent $PSScriptRoot
$reviewPython=Join-Path $reviewRoot '.venv\Scripts\python.exe'
$env:CUDA_VISIBLE_DEVICES=''
$env:OMP_NUM_THREADS='1'
$env:OPENBLAS_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
Push-Location -LiteralPath $reviewRoot
try {
    if ($Stage -eq 'review') {
        & $reviewPython -X utf8 -B -u -m analysis.feedback.reserved.review --review
    } elseif ($Stage -eq 'analyze') {
        & $reviewPython -X utf8 -B -u -m analysis.feedback.reserved.scoring
    } else {
        & $reviewPython -X utf8 -B -u -m analysis.feedback.reserved.review
    }
    if ($LASTEXITCODE -ne 0) { throw 'Offline review stopped. Preserve output and inspect before continuing.' }
} finally { Pop-Location }

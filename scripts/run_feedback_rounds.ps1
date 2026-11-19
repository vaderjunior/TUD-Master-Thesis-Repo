param([ValidateSet('all','prepare','run')][string]$Stage='all')
$ErrorActionPreference='Stop'
$workspacePath=Split-Path -Parent $PSScriptRoot
$pythonPath=Join-Path $workspacePath '.venv\Scripts\python.exe'
$env:CUDA_VISIBLE_DEVICES=''
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
$env:OPENBLAS_NUM_THREADS='1'
$env:TOKENIZERS_PARALLELISM='false'
$env:HF_HUB_OFFLINE='1'
$env:TRANSFORMERS_OFFLINE='1'
Push-Location -LiteralPath $workspacePath
try {
    Write-Host 'Feedback A17: four successive simulated correction rounds, three fixed source streams.'
    Write-Host '259,224 predictions; 777,672 ordinary calls before repairs/retries. No CUDA.'
    Write-Host 'Wait for A15 and A16 reviews. Use one Qwen queue. Every phase receives automatic audits.'
    $steps=@()
    if ($Stage -in @('all','prepare')) { $steps+=,@('analysis.feedback.rounds.run','initial') }
    if ($Stage -in @('all','run')) { $steps+=,@('analysis.feedback.rounds.run','validate'); $steps+=,@('analysis.feedback.rounds.run','run-all') }
    foreach ($step in $steps) {
        & $pythonPath -X utf8 -B -u -m @step
        if ($LASTEXITCODE -ne 0) { throw 'A17 stopped. Preserve every file and inspect the last error. Do not bypass a failed check.' }
    }
    if ($Stage -eq 'prepare') { Write-Host 'A17 initial phase prepared offline. Later pools will be derived only from audited source responses.' }
} finally { Pop-Location }

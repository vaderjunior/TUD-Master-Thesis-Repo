param([ValidateSet('all','encoders','representations')][string]$Stage='all')
$ErrorActionPreference='Stop'
$workspacePath=Split-Path -Parent $PSScriptRoot
$pythonPath=Join-Path $workspacePath '.venv\Scripts\python.exe'
Push-Location -LiteralPath $workspacePath
try {
    Write-Host 'Study SQ2 extension: retained encoder predictions plus frozen RoBERTa representation controls.'
    Write-Host 'Enable CUDA, connect the full-power charger, and keep the computer awake. No Qwen calls.'
    Write-Host 'This does not train a neural network. Small linear classifiers are fitted on CPU afterward.'
    Write-Host 'Use one copy of this launcher. Verified completed inference is reused; preserve all files after an interruption.'
    $steps=@()
    if ($Stage -in @('all','encoders')) {
        $steps+=,@('training.original.inference_entry','validate')
        $steps+=,@('training.original.inference_entry','preflight')
        $steps+=,@('training.original.inference_entry','run')
        $steps+=,@('training.original.validate')
        $steps+=,@('training.frozen_features.compare')
    }
    if ($Stage -in @('all','representations')) {
        $steps+=,@('training.frozen_features.roberta','run')
        $steps+=,@('training.frozen_features.roberta','fit-score')
    }
    foreach ($step in $steps) {
        Write-Host ('SQ2 step: '+($step -join ' '))
        & $pythonPath -X utf8 -B -u -m @step
        if ($LASTEXITCODE -ne 0) { throw 'SQ2 GPU extension stopped. Preserve all outputs and inspect the last message.' }
    }
    Write-Host 'SQ2 GPU EXTENSION COMPLETE. Inspect the saved results.'
} finally { Pop-Location }

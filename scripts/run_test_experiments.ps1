param([int]$Repeats = 5)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path $PSScriptRoot -Parent
$pythonExe = Join-Path $projectRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath (Join-Path $projectRoot 'configs\python_optimized.json'))) { throw 'Run src\tune_parameters.py on validation first.' }
foreach ($profile in @('baseline','optimized')) {
    foreach ($script in @('otsu.py','watershed.py','stardist_predict.py')) {
        & $pythonExe (Join-Path $projectRoot "src\$script") --split test --profile $profile
        if ($LASTEXITCODE -ne 0) { throw "Prediction failed: $script / $profile" }
    }
    foreach ($method in @('otsu','watershed','stardist_python')) {
        & $pythonExe (Join-Path $projectRoot 'src\evaluate.py') --split test --method $method --profile $profile --figures demo
        if ($LASTEXITCODE -ne 0) { throw "Evaluation failed: $method / $profile" }
    }
    & $pythonExe (Join-Path $projectRoot 'src\compare_methods.py') --split test --profile $profile --figures demo
    if ($LASTEXITCODE -ne 0) { throw "Comparison failed: $profile" }
}
& $pythonExe (Join-Path $projectRoot 'src\benchmark.py') --split test --profile optimized --repeats $Repeats
if ($LASTEXITCODE -ne 0) { throw 'Controlled benchmark failed.' }
& $pythonExe (Join-Path $projectRoot 'src\summarize_test.py')
if ($LASTEXITCODE -ne 0) { throw 'Final summary failed.' }
Write-Output 'DONE. See docs\TEST_RESULTS.md.'

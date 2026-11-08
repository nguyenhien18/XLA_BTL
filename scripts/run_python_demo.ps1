param([switch]$Overwrite, [string]$PythonExe)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path $PSScriptRoot -Parent
$pythonExe = if ($PythonExe) { $PythonExe } else { Join-Path $projectRoot '.venv\Scripts\python.exe' }
if (-not (Test-Path -LiteralPath $pythonExe)) { throw 'Missing project .venv. See docs/PYTHON_START_HERE.md.' }
foreach ($script in @('otsu.py','watershed.py','stardist_predict.py')) {
    $scriptArgs = @((Join-Path $projectRoot "src\$script"), '--split', 'demo')
    if ($Overwrite) { $scriptArgs += '--overwrite' }
    & $pythonExe @scriptArgs
    if ($LASTEXITCODE -ne 0) { throw "Prediction failed: $script" }
}
$evaluationMethods = @('otsu','watershed','stardist_python')
$demoNames = Get-Content -LiteralPath (Join-Path $projectRoot 'data\splits\demo.txt') | Where-Object { $_.Trim() }
$missingFiji = @($demoNames | Where-Object { -not (Test-Path -LiteralPath (Join-Path $projectRoot "results\stardist_fiji\$_.tif")) })
if ($missingFiji.Count -eq 0) {
    $evaluationMethods += 'stardist_fiji'
} else {
    Write-Output 'Fiji output is not complete. Evaluating the three Python methods; run Fiji batch before adding its comparison.'
}
foreach ($method in $evaluationMethods) {
    & $pythonExe (Join-Path $projectRoot 'src\evaluate.py') --method $method --split demo
    if ($LASTEXITCODE -ne 0) { throw "Evaluation failed: $method" }
}
& $pythonExe (Join-Path $projectRoot 'src\compare_methods.py') --split demo
if ($LASTEXITCODE -ne 0) { throw 'Method comparison failed.' }
Write-Output 'DONE. Open docs\methods_demo_comparison.md and results\figures\methods_demo\overview.png.'

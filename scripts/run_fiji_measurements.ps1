param([string]$FijiRoot = 'D:\Downloads\fiji-stable-win64-jdk (1)\Fiji.app')
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path $PSScriptRoot -Parent
$javaExe = Join-Path $FijiRoot 'java\win64\zulu8.86.0.25-ca-fx-jdk8.0.452-win_x64\bin\java.exe'
$ijJar = Join-Path $FijiRoot 'jars\ij-1.54p.jar'
$groovyJar = Join-Path $FijiRoot 'jars\groovy-4.0.23.jar'
foreach ($taskPath in @($javaExe,$ijJar,$groovyJar)) {
    if (-not (Test-Path -LiteralPath $taskPath)) { throw "Missing runtime: $taskPath" }
}
& $javaExe '-Djava.awt.headless=true' '-cp' "$ijJar;$groovyJar" 'groovy.ui.GroovyMain' (Join-Path $projectRoot 'fiji\macros\evaluate_demo.groovy')
if ($LASTEXITCODE -ne 0) { throw 'Fiji measurement failed.' }

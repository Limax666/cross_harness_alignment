[CmdletBinding()]
param(
    [string]$AgentDojoRef = '089ed468cf3ed0322acc66b0211f26d9d90dbf60',
    [string]$PythonVersion = '3.11'
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$Root = (Resolve-Path (Join-Path $PSScriptRoot '..\..\..')).Path
$Exp = Join-Path $Root 'experiments\cross_harness_sft'
$Venv = Join-Path $Exp '.venv'
$Python = Join-Path $Venv 'Scripts\python.exe'
$Vendor = Join-Path $Exp 'vendor'
$AgentDojo = Join-Path $Vendor 'agentdojo'

& py "-$PythonVersion" --version
if ($LASTEXITCODE -ne 0) {
    throw "Python $PythonVersion is required. Install 64-bit Python $PythonVersion, then rerun this script."
}

if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    & py "-$PythonVersion" -m venv $Venv
    if ($LASTEXITCODE -ne 0) { throw 'Failed to create the project virtual environment.' }
}

& $Python -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw 'Failed to upgrade pip.' }
& $Python -m pip install -e $Exp
if ($LASTEXITCODE -ne 0) { throw 'Failed to install cross-harness-sft.' }

New-Item -ItemType Directory -Force -Path $Vendor | Out-Null
if (-not (Test-Path -LiteralPath (Join-Path $AgentDojo '.git'))) {
    & git clone https://github.com/ethz-spylab/agentdojo.git $AgentDojo
    if ($LASTEXITCODE -ne 0) { throw 'Failed to clone AgentDojo.' }
}
& git -C $AgentDojo fetch --tags origin
if ($LASTEXITCODE -ne 0) { throw 'Failed to fetch AgentDojo refs.' }
& git -C $AgentDojo switch --detach $AgentDojoRef
if ($LASTEXITCODE -ne 0) { throw "Failed to checkout AgentDojo ref $AgentDojoRef." }
& $Python -m pip install -e $AgentDojo
if ($LASTEXITCODE -ne 0) { throw 'Failed to install AgentDojo.' }

Write-Host "Local collection runtime is ready: $Python"
Write-Host 'Next run: .\experiments\cross_harness_sft\scripts\run_codex_native_agentdojo_full.ps1'

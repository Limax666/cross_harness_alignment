<# Run HarnessAudit-Bench with native Codex + GPT-5.6 Sol and export full trajectories. #>
[CmdletBinding()]
param(
    [string]$Task = 'multi_agent\tasks\daily_life\wellness\dl-t4.yaml',
    [int]$AgentTimeout = 900,
    [switch]$WithJudge
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$env:PYTHONUTF8 = '1'
if (Test-Path variable:PSNativeCommandUseErrorActionPreference) {
    $PSNativeCommandUseErrorActionPreference = $false
}

$Root = (Resolve-Path (Join-Path $PSScriptRoot '..\..\..')).Path
$HarnessAudit = Join-Path $Root 'experiments\cross_harness_sft\vendor\harnessaudit'
$Python = Join-Path $HarnessAudit '.venv\Scripts\python.exe'
$TaskPath = if ([IO.Path]::IsPathRooted($Task)) { $Task } else { Join-Path $HarnessAudit $Task }
$CodexNpm = Join-Path $env:APPDATA 'npm\node_modules\@openai\codex\node_modules\@openai\codex-win32-x64\vendor\x86_64-pc-windows-msvc\bin\codex.exe'
$CodexVsCode = Join-Path $env:USERPROFILE '.vscode\extensions\openai.chatgpt-26.901.22334-win32-x64\bin\windows-x86_64\codex.exe'

if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) { throw "Missing HarnessAudit venv: $Python" }
if (-not (Test-Path -LiteralPath $TaskPath -PathType Leaf)) { throw "Missing task: $TaskPath" }
if (Test-Path -LiteralPath $CodexNpm -PathType Leaf) {
    $CodexBinary = $CodexNpm
} elseif (Test-Path -LiteralPath $CodexVsCode -PathType Leaf) {
    $CodexBinary = $CodexVsCode
} else {
    throw 'Native codex.exe not found.'
}

$ArtifactRoot = Join-Path $HarnessAudit 'collection\codex\gpt-5.6-sol'
$TraceDir = Join-Path $ArtifactRoot 'benchmark_traces'
$ResultDir = Join-Path $ArtifactRoot 'benchmark_results'
$TrajectoryDir = Join-Path $ArtifactRoot 'trajectories'
$IsolationDir = Join-Path $ArtifactRoot 'isolation'
$McpLogDir = Join-Path $ArtifactRoot 'mcp_logs'
@($TraceDir, $ResultDir, $TrajectoryDir, $IsolationDir, $McpLogDir) | ForEach-Object {
    New-Item -ItemType Directory -Force -Path $_ | Out-Null
}

$env:CODEX_AUTH_MODE = 'cli_login'
$env:HARNESSAUDIT_CODEX_BINARY = $CodexBinary
$env:HARNESSAUDIT_CODEX_TRAJECTORY_DIR = $TrajectoryDir
$env:HARNESSAUDIT_TASK_ID = [IO.Path]::GetFileNameWithoutExtension($TaskPath)
$env:MASP_CODEX_ISOLATION_ROOT = $IsolationDir
$env:MASP_MCP_LOG_DIR = $McpLogDir
$env:MASP_KEEP_CODEX_ISOLATION = '0'

$SavedErrorActionPreference = $ErrorActionPreference
$ErrorActionPreference = 'Continue'
& $CodexBinary login status 2>$null
$LoginExitCode = $LASTEXITCODE
$ErrorActionPreference = $SavedErrorActionPreference
if ($LoginExitCode -ne 0) { throw 'Codex CLI login is unavailable.' }

$Arguments = @(
    '-m', 'multi_agent', 'run', $TaskPath,
    '--framework', 'clawteam', '--harness', 'codex', '--model', 'gpt-5.6-sol',
    '--trace-dir', $TraceDir, '--output', $ResultDir,
    '--agent-timeout', $AgentTimeout
)
if (-not $WithJudge) { $Arguments += '--skip-judge' }

Push-Location $HarnessAudit
try {
    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) { throw "HarnessAudit exited with code $LASTEXITCODE" }
} finally {
    Pop-Location
}

Write-Host "Complete trajectories: $TrajectoryDir"
Write-Host "Official HarnessAudit traces: $TraceDir"
Write-Host "HarnessAudit reports: $ResultDir"

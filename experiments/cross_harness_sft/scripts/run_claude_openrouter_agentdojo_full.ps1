[CmdletBinding()]
param(
    [string]$Config,
    [string]$EnvFile,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$CollectArgs
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
[Console]::InputEncoding = [Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = [Text.UTF8Encoding]::new($false)
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'

$Root = (Resolve-Path (Join-Path $PSScriptRoot '..\..\..')).Path
$Exp = Join-Path $Root 'experiments\cross_harness_sft'
if (-not $Config) {
    $Config = Join-Path $Exp 'configs\claude_agentdojo_openrouter_gpt56_sol_full_windows.yaml'
}
if (-not $EnvFile) { $EnvFile = Join-Path $Exp '.env' }
$Config = (Resolve-Path $Config).Path
$Python = Join-Path $Exp '.venv\Scripts\python.exe'
$WorkerConfig = Join-Path $Exp 'configs\agentdojo_worker.yaml'
$WorkerDir = Join-Path $Exp 'runs\workers'
$WorkerOut = Join-Path $WorkerDir 'agentdojo-claude-openrouter.windows.stdout.log'
$WorkerErr = Join-Path $WorkerDir 'agentdojo-claude-openrouter.windows.stderr.log'
$WorkerPid = Join-Path $WorkerDir 'agentdojo-claude-openrouter.windows.pid'
$HealthUrl = 'http://127.0.0.1:8211/health'
$CompatHealthUrl = 'http://127.0.0.1:8311/health'
$CompatOut = Join-Path $WorkerDir 'openrouter-compat.windows.stdout.log'
$CompatErr = Join-Path $WorkerDir 'openrouter-compat.windows.stderr.log'

if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "Missing local environment: $Python`nRun scripts\setup_local_windows.ps1 first."
}

# Load only OPENROUTER_API_KEY from .env; never execute the file as PowerShell.
if (-not $env:OPENROUTER_API_KEY -and (Test-Path -LiteralPath $EnvFile -PathType Leaf)) {
    foreach ($line in Get-Content -LiteralPath $EnvFile) {
        if ($line -match '^\s*(?:export\s+)?OPENROUTER_API_KEY\s*=\s*(.*)\s*$') {
            $value = $Matches[1].Trim()
            if (($value.StartsWith('"') -and $value.EndsWith('"')) -or
                ($value.StartsWith("'") -and $value.EndsWith("'"))) {
                $value = $value.Substring(1, $value.Length - 2)
            }
            $env:OPENROUTER_API_KEY = $value
            break
        }
    }
}
if (-not $env:OPENROUTER_API_KEY) {
    throw "OPENROUTER_API_KEY is missing. Set it in the current process or add it to $EnvFile"
}

$ClaudeCommand = Get-Command claude -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1
if (-not $ClaudeCommand) { throw 'Native Claude Code executable was not found on PATH.' }
$env:CLAUDE_BINARY = $ClaudeCommand.Source

# Claude Code/OpenRouter traffic uses the external proxy, while benchmark MCP traffic must stay local.
$LocalNoProxy = '127.0.0.1,localhost'
$ExistingNoProxy = [Environment]::GetEnvironmentVariable('NO_PROXY')
if ($ExistingNoProxy) { $LocalNoProxy = "$LocalNoProxy,$ExistingNoProxy" }
$env:NO_PROXY = $LocalNoProxy
$env:no_proxy = $LocalNoProxy

New-Item -ItemType Directory -Force -Path $WorkerDir | Out-Null
$StartedWorker = $false
$Worker = $null
$StartedCompat = $false
$Compat = $null

function Test-AgentDojoHealth {
    try {
        $response = Invoke-RestMethod -Uri $HealthUrl -TimeoutSec 3
        return $null -ne $response
    } catch { return $false }
}

try {
    try { $null = Invoke-RestMethod -Uri $CompatHealthUrl -TimeoutSec 2 } catch {
        $CompatScript = Join-Path $PSScriptRoot 'openrouter_anthropic_compat_proxy.py'
        $Compat = Start-Process -FilePath $Python -ArgumentList @(
            ('"{0}"' -f $CompatScript), '--port', '8311'
        ) -WorkingDirectory $Root -WindowStyle Hidden -PassThru `
          -RedirectStandardOutput $CompatOut -RedirectStandardError $CompatErr
        $StartedCompat = $true
        for ($attempt = 1; $attempt -le 30; $attempt++) {
            if ($Compat.HasExited) { throw "OpenRouter compatibility proxy exited. See $CompatErr" }
            try { $null = Invoke-RestMethod -Uri $CompatHealthUrl -TimeoutSec 2; break } catch { Start-Sleep -Milliseconds 500 }
        }
    }
    if (-not (Test-AgentDojoHealth)) {
        $workerArguments = @(
            '-m', 'cross_harness_sft.benchmark_server',
            '--driver', 'cross_harness_sft.backends.agentdojo:create_driver',
            '--config', ('"{0}"' -f $WorkerConfig), '--port', '8211'
        )
        $Worker = Start-Process -FilePath $Python -ArgumentList $workerArguments `
            -WorkingDirectory $Root -WindowStyle Hidden -PassThru `
            -RedirectStandardOutput $WorkerOut -RedirectStandardError $WorkerErr
        Set-Content -LiteralPath $WorkerPid -Value $Worker.Id -Encoding ascii
        $StartedWorker = $true
        $Ready = $false
        for ($attempt = 1; $attempt -le 60; $attempt++) {
            if ($Worker.HasExited) {
                throw "AgentDojo worker exited with code $($Worker.ExitCode). See $WorkerOut and $WorkerErr"
            }
            if (Test-AgentDojoHealth) { $Ready = $true; break }
            Start-Sleep -Seconds 1
            $Worker.Refresh()
        }
        if (-not $Ready) {
            throw "AgentDojo worker did not become healthy within 60 seconds. See $WorkerOut and $WorkerErr"
        }
    }

    Invoke-RestMethod -Uri $HealthUrl -TimeoutSec 5 | ConvertTo-Json -Depth 8
    Push-Location $Root
    try {
        & $Python -m cross_harness_sft.preflight --config $Config --mode collect
        if ($LASTEXITCODE -ne 0) { throw 'Collection preflight failed.' }
        & $Python -m cross_harness_sft.collect_native --config $Config @CollectArgs
        if ($LASTEXITCODE -ne 0) { throw "Collection failed with exit code $LASTEXITCODE." }
    } finally { Pop-Location }
} finally {
    if ($StartedCompat -and $null -ne $Compat -and -not $Compat.HasExited) {
        Stop-Process -Id $Compat.Id -ErrorAction SilentlyContinue
        $Compat.WaitForExit(5000) | Out-Null
    }
    if ($StartedWorker -and $null -ne $Worker -and -not $Worker.HasExited) {
        Stop-Process -Id $Worker.Id -ErrorAction SilentlyContinue
        $Worker.WaitForExit(5000) | Out-Null
    }
    if ($StartedWorker -and (Test-Path -LiteralPath $WorkerPid)) {
        Remove-Item -LiteralPath $WorkerPid -Force
    }
}

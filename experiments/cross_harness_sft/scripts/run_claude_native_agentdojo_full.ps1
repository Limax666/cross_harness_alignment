[CmdletBinding()]
param(
    [string]$Config,
    [string]$EnvFile,
    # 'auto' keeps the ambient route (environment + Windows registry); 'direct' forbids any
    # proxy; any other value is used as an explicit proxy url, e.g. http://127.0.0.1:7897.
    [string]$Egress = 'auto',
    [int]$ForwarderPort = 8311,
    [int]$WorkerPort = 8211,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$CollectArgs
)

# This launcher mirrors scripts\run_codex_native_agentdojo_full.ps1 one-to-one. The two
# extra stages (forwarder + route gate) exist only because `openai/gpt-5.6-sol` is not a
# native Claude Code model: Codex authenticates to its own ChatGPT account with auth.json
# and needs neither an LLM gateway nor a pinned egress route.
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
[Console]::OutputEncoding = [Text.UTF8Encoding]::new($false)
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'

$Root = (Resolve-Path (Join-Path $PSScriptRoot '..\..\..')).Path
$Exp = Join-Path $Root 'experiments\cross_harness_sft'
if (-not $Config) {
    $Config = Join-Path $Exp 'configs\claude_agentdojo_openrouter_gpt56_sol_full_windows.yaml'
}
$Config = (Resolve-Path $Config).Path
if (-not $EnvFile) { $EnvFile = Join-Path $Exp '.env' }
$Python = Join-Path $Exp '.venv\Scripts\python.exe'
$Forwarder = Join-Path $PSScriptRoot 'openrouter_anthropic_compat_proxy.py'
$WorkerConfig = Join-Path $Exp 'configs\agentdojo_worker.yaml'
$LogDir = Join-Path $Exp 'runs\workers'
$ForwarderOut = Join-Path $LogDir "anthropic-forward.$ForwarderPort.stdout.log"
$ForwarderErr = Join-Path $LogDir "anthropic-forward.$ForwarderPort.stderr.log"
$ForwarderErrors = Join-Path $LogDir "anthropic-forward.$ForwarderPort.upstream-errors.jsonl"
$WorkerOut = Join-Path $LogDir "agentdojo-claude.$WorkerPort.stdout.log"
$WorkerErr = Join-Path $LogDir "agentdojo-claude.$WorkerPort.stderr.log"
$WorkerPid = Join-Path $LogDir "agentdojo-claude.$WorkerPort.pid"
$WorkerHealth = "http://127.0.0.1:$WorkerPort/health"
$ForwarderHealth = "http://127.0.0.1:$ForwarderPort/health"

if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "Missing local environment: $Python`nRun scripts\setup_local_windows.ps1 first."
}
if (-not (Get-Command claude -CommandType Application -ErrorAction SilentlyContinue)) {
    throw 'Native Claude Code executable was not found on PATH.'
}
$env:CLAUDE_BINARY = (Get-Command claude -CommandType Application | Select-Object -First 1).Source

# Benchmark MCP traffic is loopback; the teacher route is decided by -Egress below.
$LocalNoProxy = '127.0.0.1,localhost'
$ExistingNoProxy = [Environment]::GetEnvironmentVariable('NO_PROXY')
if ($ExistingNoProxy) { $LocalNoProxy = "$LocalNoProxy,$ExistingNoProxy" }
$env:NO_PROXY = $LocalNoProxy
$env:no_proxy = $LocalNoProxy

function Test-Health($Url, $Seconds = 3) {
    try { return $null -ne (Invoke-RestMethod -Uri $Url -TimeoutSec $Seconds) } catch { return $false }
}

function Get-Health($Url, $Seconds = 3) {
    try { return Invoke-RestMethod -Uri $Url -TimeoutSec $Seconds } catch { return $null }
}

function Get-RouteProbe($Url, $Seconds = 15) {
    $value = Get-Health $Url $Seconds
    if ($null -eq $value) { return $null }
    $property = $value.PSObject.Properties['route_probe']
    if ($null -eq $property) { return $null }
    return $property.Value
}

# The upstream key is loaded only long enough to hand it to the forwarder process. After
# that it is removed from this process, so neither the collector nor Claude Code inherits it.
function Get-OpenRouterKey {
    if ($env:OPENROUTER_API_KEY) { return $env:OPENROUTER_API_KEY.Trim() }
    if (-not (Test-Path -LiteralPath $EnvFile -PathType Leaf)) { return $null }
    foreach ($line in Get-Content -LiteralPath $EnvFile) {
        if ($line -match '^\s*(?:export\s+)?OPENROUTER_API_KEY\s*=\s*(.*?)\s*$') {
            $value = $Matches[1].Trim()
            if (($value.StartsWith('"') -and $value.EndsWith('"')) -or
                ($value.StartsWith("'") -and $value.EndsWith("'"))) {
                $value = $value.Substring(1, $value.Length - 2)
            }
            return $value
        }
    }
    return $null
}

# Guard: `-Config x --limit-per-cell 1` would otherwise bind --limit-per-cell to $EnvFile.
if ($EnvFile -like '-*') {
    $CollectArgs = @($EnvFile) + @($CollectArgs)
    $EnvFile = ''
}
if (-not $EnvFile) { $EnvFile = Join-Path $Exp '.env' }

# The model slug the route probe must check, straight from the experiment config.
$TeacherModel = (& $Python -c "import sys,yaml; c=yaml.safe_load(open(sys.argv[1],encoding='utf-8')); print(c['harnesses'][0]['model'])" $Config).Trim()
if (-not $TeacherModel) { throw "Could not read harnesses[0].model from $Config" }
$env:TEACHER_ROUTE_PROBE_MODEL = $TeacherModel

# urllib resolves proxies as `getproxies_environment() or getproxies_registry()`, so the
# NO_PROXY line above hides the Windows system proxy: "auto" would silently mean direct
# egress, which OpenRouter region-blocks for this model. Resolve the system route explicitly.
if ($Egress -eq 'auto') {
    $registry = $null
    try { $registry = (& $Python -c "import json,urllib.request as u; print(json.dumps(u.getproxies_registry()))").Trim() | ConvertFrom-Json } catch { $registry = $null }
    if ($registry -and $registry.https) {
        Write-Host "[egress] auto -> $($registry.https) (Windows system proxy)"
        $Egress = [string]$registry.https
    } else {
        Write-Host '[egress] auto -> direct (no Windows system proxy configured)'
        $Egress = 'direct'
    }
}

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$StartedForwarder = $false
$StartedWorker = $false
$ForwarderProcess = $null
$Worker = $null

try {
    # ---- 1. LLM transport gate (the Claude analogue of `codex login status`) ----
    $probe = Get-RouteProbe $ForwarderHealth 15
    if ($null -eq $probe) {
        if (Test-Health $ForwarderHealth 2) {
            throw "Port $ForwarderPort answers but is not our forwarder (no route_probe field). " +
                  'Stop that process or choose another -ForwarderPort.'
        }
        $Key = Get-OpenRouterKey
        if (-not $Key) { throw "OPENROUTER_API_KEY is missing (process environment or $EnvFile)." }
        $LocalToken = [guid]::NewGuid().ToString('N')
        $env:OPENROUTER_API_KEY = $Key
        $forwarderArgs = ('"{0}" --port {1} --egress "{2}" --local-token {3} --error-log "{4}"' -f
            $Forwarder, $ForwarderPort, $Egress, $LocalToken, $ForwarderErrors)
        $ForwarderProcess = Start-Process -FilePath $Python -ArgumentList $forwarderArgs `
            -WorkingDirectory $Exp -WindowStyle Hidden -PassThru `
            -RedirectStandardOutput $ForwarderOut -RedirectStandardError $ForwarderErr
        $StartedForwarder = $true
        for ($attempt = 1; $attempt -le 40; $attempt++) {
            if ($ForwarderProcess.HasExited) {
                throw "Forwarder exited before it became healthy (code $($ForwarderProcess.ExitCode)). See $ForwarderErr"
            }
            $probe = Get-RouteProbe $ForwarderHealth 20
            if ($null -ne $probe) { break }
            Start-Sleep -Milliseconds 500
        }
        if ($null -eq $probe) {
            throw "Forwarder never reported a route probe. See $ForwarderOut and $ForwarderErr"
        }
        $env:FACADE_LOCAL_TOKEN = $LocalToken
    } elseif (-not $env:FACADE_LOCAL_TOKEN) {
        # Reusing a forwarder this script did not start: its token is not guessable.
        throw "Reusing the forwarder already on port $ForwarderPort requires its token in " +
              '$env:FACADE_LOCAL_TOKEN. Stop that process to let this script start its own.'
    }

    Write-Host ("[route] egress={0} status={1} ok={2}" -f $probe.egress, $probe.status, $probe.ok)
    if (-not $probe.ok) {
        throw ("Teacher model {0} is not reachable through egress {1} (HTTP {2}): {3}`n" -f
            $TeacherModel, $probe.egress, $probe.status, $probe.detail) +
            "Fix the egress route (or pass -Egress http://127.0.0.1:<port>) before collecting."
    }
    Remove-Item Env:OPENROUTER_API_KEY -ErrorAction SilentlyContinue   # never past this point

    # ---- 2. official AgentDojo worker ----
    if (-not (Test-Health $WorkerHealth)) {
        $workerArgs = @('-m', 'cross_harness_sft.benchmark_server',
                        '--driver', 'cross_harness_sft.backends.agentdojo:create_driver',
                        '--config', ('"{0}"' -f $WorkerConfig), '--port', "$WorkerPort")
        $Worker = Start-Process -FilePath $Python -ArgumentList $workerArgs `
            -WorkingDirectory $Root -WindowStyle Hidden -PassThru `
            -RedirectStandardOutput $WorkerOut -RedirectStandardError $WorkerErr
        Set-Content -LiteralPath $WorkerPid -Value $Worker.Id -Encoding ascii
        $StartedWorker = $true
        $Ready = $false
        for ($attempt = 1; $attempt -le 60; $attempt++) {
            if ($Worker.HasExited) {
                throw "AgentDojo worker exited with code $($Worker.ExitCode). See $WorkerOut and $WorkerErr"
            }
            if (Test-Health $WorkerHealth) { $Ready = $true; break }
            Start-Sleep -Seconds 1
            $Worker.Refresh()
        }
        if (-not $Ready) { throw "AgentDojo worker did not become healthy within 60 seconds. See $WorkerOut" }
    }
    Invoke-RestMethod -Uri $WorkerHealth -TimeoutSec 5 | ConvertTo-Json -Depth 8

    # ---- 3. fail-closed preflight, then collection ----
    Push-Location $Root
    try {
        & $Python -m cross_harness_sft.preflight --config $Config --mode collect
        if ($LASTEXITCODE -ne 0) { throw 'Collection preflight failed.' }
        & $Python -m cross_harness_sft.collect_native --config $Config @CollectArgs
        if ($LASTEXITCODE -ne 0) { throw "Collection failed with exit code $LASTEXITCODE." }
    } finally { Pop-Location }
} finally {
    if ($StartedForwarder -and $null -ne $ForwarderProcess -and -not $ForwarderProcess.HasExited) {
        Stop-Process -Id $ForwarderProcess.Id -ErrorAction SilentlyContinue
        $ForwarderProcess.WaitForExit(5000) | Out-Null
    }
    if ($StartedWorker -and $null -ne $Worker -and -not $Worker.HasExited) {
        Stop-Process -Id $Worker.Id -ErrorAction SilentlyContinue
        $Worker.WaitForExit(5000) | Out-Null
    }
    if ($StartedWorker -and (Test-Path -LiteralPath $WorkerPid)) {
        Remove-Item -LiteralPath $WorkerPid -Force
    }
}

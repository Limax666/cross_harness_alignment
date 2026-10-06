[CmdletBinding()]
param(
    [string]$Config,
    [string]$CodexAuthPath,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$CollectArgs
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$Root = (Resolve-Path (Join-Path $PSScriptRoot '..\..\..')).Path
$Exp = Join-Path $Root 'experiments\cross_harness_sft'
if (-not $Config) {
    $Config = Join-Path $Exp 'configs\codex_agentdojo_native_full_windows.yaml'
}
$Config = (Resolve-Path $Config).Path
$Python = Join-Path $Exp '.venv\Scripts\python.exe'
$WorkerConfig = Join-Path $Exp 'configs\agentdojo_worker.yaml'
$WorkerDir = Join-Path $Exp 'runs\workers'
$WorkerOut = Join-Path $WorkerDir 'agentdojo.windows.stdout.log'
$WorkerErr = Join-Path $WorkerDir 'agentdojo.windows.stderr.log'
$WorkerPid = Join-Path $WorkerDir 'agentdojo.windows.pid'
$HealthUrl = 'http://127.0.0.1:8111/health'

# Keep benchmark traffic local even when the workstation uses an HTTP(S) proxy.
$LocalNoProxy = '127.0.0.1,localhost'
$ExistingNoProxy = [Environment]::GetEnvironmentVariable('NO_PROXY')
if ($ExistingNoProxy) { $LocalNoProxy = "$LocalNoProxy,$ExistingNoProxy" }
$env:NO_PROXY = $LocalNoProxy
$env:no_proxy = $LocalNoProxy

if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "Missing local environment: $Python`nRun scripts\setup_local_windows.ps1 first."
}
if (-not (Get-Command codex -ErrorAction SilentlyContinue)) {
    throw 'Codex CLI is not available on PATH.'
}
$NpmCodexExe = Join-Path $env:APPDATA 'npm\node_modules\@openai\codex\node_modules\@openai\codex-win32-x64\vendor\x86_64-pc-windows-msvc\bin\codex.exe'
$VsCodeCodexExe = Join-Path $env:USERPROFILE '.vscode\extensions\openai.chatgpt-26.901.22334-win32-x64\bin\windows-x86_64\codex.exe'
if (Test-Path -LiteralPath $NpmCodexExe -PathType Leaf) {
    $env:CODEX_BINARY = $NpmCodexExe
} elseif (Test-Path -LiteralPath $VsCodeCodexExe -PathType Leaf) {
    $env:CODEX_BINARY = $VsCodeCodexExe
} else {
    throw 'Native codex.exe was not found. The codex.cmd wrapper cannot safely carry multi-line benchmark prompts.'
}
if (-not $CodexAuthPath) {
    $CodexAuthPath = Join-Path $env:USERPROFILE '.codex\auth.json'
}
$CodexAuthPath = [IO.Path]::GetFullPath($CodexAuthPath)
if (-not (Test-Path -LiteralPath $CodexAuthPath -PathType Leaf)) {
    throw "Codex authentication was not found at $CodexAuthPath. Run: codex login"
}
$env:CODEX_AUTH_PATH = $CodexAuthPath

& codex login status
if ($LASTEXITCODE -ne 0) { throw 'codex login status failed.' }

New-Item -ItemType Directory -Force -Path $WorkerDir | Out-Null
$StartedWorker = $false
$Worker = $null

function Test-AgentDojoHealth {
    try {
        $response = Invoke-RestMethod -Uri $HealthUrl -TimeoutSec 3
        return $null -ne $response
    } catch {
        return $false
    }
}

try {
    if (-not (Test-AgentDojoHealth)) {
        $workerArguments = @(
            '-m', 'cross_harness_sft.benchmark_server',
            '--driver', 'cross_harness_sft.backends.agentdojo:create_driver',
            '--config', ('"{0}"' -f $WorkerConfig),
            '--port', '8111'
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
            if (Test-AgentDojoHealth) {
                $Ready = $true
                break
            }
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

        # Collection only. Verification and rejection filtering remain separate stages.
        & $Python -m cross_harness_sft.collect_native --config $Config @CollectArgs
        if ($LASTEXITCODE -ne 0) { throw "Collection failed with exit code $LASTEXITCODE." }
    } finally {
        Pop-Location
    }
} finally {
    if ($StartedWorker -and $null -ne $Worker -and -not $Worker.HasExited) {
        Stop-Process -Id $Worker.Id -ErrorAction SilentlyContinue
        $Worker.WaitForExit(5000) | Out-Null
    }
    if ($StartedWorker -and (Test-Path -LiteralPath $WorkerPid)) {
        Remove-Item -LiteralPath $WorkerPid -Force
    }
}

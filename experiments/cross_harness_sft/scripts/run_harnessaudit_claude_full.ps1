<#
.SYNOPSIS
    Batch-collect Claude Code + GPT-5.6-sol trajectories on HarnessAudit-bench.

.DESCRIPTION
    Runs every task YAML under vendor/harnessaudit/multi_agent/tasks/ through
    the ClawTeam adapter with --harness claude, forwarding LLM traffic through
    the local Anthropic-compatible forwarder to OpenRouter's openai/gpt-5.6-sol.

    Prerequisites:
      - Forwarder running on $ForwarderPort (default 8312)
      - OPENROUTER_API_KEY in experiments/cross_harness_sft/.env
      - vendor/harnessaudit/.venv with HarnessAudit + clawteam installed

.PARAMETER ForwarderPort
    Port of the local Anthropic forwarder. Default: 8312.

.PARAMETER AgentTimeout
    Max seconds per task. Default: 300.

.PARAMETER TaskGlob
    Glob pattern under tasks/ to select tasks. Default: **/*.yaml (all).

.PARAMETER Resume
    Skip tasks whose report JSON already exists in the output dir.

.PARAMETER DryRun
    Print the task list without executing.

.EXAMPLE
    .\run_harnessaudit_claude_full.ps1
    .\run_harnessaudit_claude_full.ps1 -TaskGlob 'finance/**/*.yaml' -Resume
    .\run_harnessaudit_claude_full.ps1 -DryRun
#>

param(
    [int]$ForwarderPort = 8312,
    [int]$AgentTimeout = 300,
    [string]$TaskGlob = '**/*.yaml',
    [switch]$Resume,
    [switch]$DryRun
)

$ErrorActionPreference = 'Stop'
$env:PYTHONUTF8 = '1'

# ── Paths ──────────────────────────────────────────────────────────────
# Script lives in experiments/cross_harness_sft/scripts/, repo root is 3 levels up
$RepoRoot   = (Resolve-Path (Join-Path $PSScriptRoot '..\..\..')).Path
$VenvPython = Join-Path $RepoRoot 'experiments\cross_harness_sft\vendor\harnessaudit\.venv\Scripts\python.exe'
$TasksRoot  = Join-Path $RepoRoot 'experiments\cross_harness_sft\vendor\harnessaudit\multi_agent\tasks'
$TraceDir   = Join-Path $RepoRoot 'experiments\cross_harness_sft\vendor\harnessaudit\multi_agent\traces'
$OutputDir  = Join-Path $RepoRoot 'experiments\cross_harness_sft\vendor\harnessaudit\results'
$EnvFile    = Join-Path $RepoRoot 'experiments\cross_harness_sft\.env'
$IsolationRoot = 'C:\tmp\ha'
$McpLogDir     = 'C:\tmp\ha\mcp_logs'

# ── Load OPENROUTER_API_KEY from .env (forwarder needs it) ────────────
if (-not (Test-Path $EnvFile)) { throw ".env not found at $EnvFile" }
foreach ($line in Get-Content $EnvFile) {
    if ($line -match '^([^#][^=]+)=(.*)$') {
        [System.Environment]::SetEnvironmentVariable($matches[1].Trim(), $matches[2].Trim(), 'Process')
    }
}

# ── Environment for Claude Code ───────────────────────────────────────
$env:ANTHROPIC_API_KEY       = 'harnessaudit-local-token'
$env:ANTHROPIC_BASE_URL      = "http://127.0.0.1:$ForwarderPort"
$env:CLAUDE_CODE_AUTH_MODE   = 'api_key'
$env:MASP_CLAUDE_ISOLATION_ROOT = $IsolationRoot
$env:MASP_MCP_LOG_DIR        = $McpLogDir
$env:MASP_KEEP_CLAUDE_ISOLATION = '1'   # keep isolation dirs for raw JSONL extraction
Remove-Item Env:OPENROUTER_API_KEY -ErrorAction SilentlyContinue

# Output dir for raw Claude Code session JSONLs
$RawJsonlDir = Join-Path $RepoRoot 'experiments\cross_harness_sft\vendor\harnessaudit\raw_sessions\claude_gpt56sol'
New-Item -ItemType Directory -Path $RawJsonlDir -Force | Out-Null

# ── Verify forwarder ──────────────────────────────────────────────────
try {
    $health = Invoke-RestMethod -Uri "http://127.0.0.1:$ForwarderPort/health" -TimeoutSec 5
    if (-not $health.route_probe.ok) {
        throw "Forwarder route_probe is not OK"
    }
    Write-Host "[OK] Forwarder healthy on port $ForwarderPort" -ForegroundColor Green
} catch {
    # PowerShell may route localhost through WinINET proxy; try Python fallback
    $checkResult = & $VenvPython -c @"
import urllib.request, json
handler = urllib.request.ProxyHandler({})
opener = urllib.request.build_opener(handler)
try:
    r = opener.open('http://127.0.0.1:$ForwarderPort/health', timeout=5)
    h = json.loads(r.read().decode())
    print('ok' if h.get('route_probe',{}).get('ok') else 'fail')
except Exception as e:
    print(f'error:{e}')
"@ 2>&1
    if ($checkResult -eq 'ok') {
        Write-Host "[OK] Forwarder healthy on port $ForwarderPort (verified via Python)" -ForegroundColor Green
    } else {
        throw "Forwarder not reachable on port ${ForwarderPort}: $checkResult"
    }
}

# ── Discover tasks ────────────────────────────────────────────────────
$taskFiles = Get-ChildItem -Path $TasksRoot -Filter '*.yaml' -Recurse |
    Where-Object { $_.FullName -like (Join-Path $TasksRoot $TaskGlob) } |
    Sort-Object FullName

Write-Host "`nFound $($taskFiles.Count) tasks matching '$TaskGlob'" -ForegroundColor Cyan

if ($DryRun) {
    $taskFiles | ForEach-Object { Write-Host "  $($_.FullName.Replace($TasksRoot, ''))" }
    exit 0
}

# ── Ensure output dirs exist ──────────────────────────────────────────
New-Item -ItemType Directory -Path $IsolationRoot -Force | Out-Null
New-Item -ItemType Directory -Path $McpLogDir -Force | Out-Null

# ── Run loop ──────────────────────────────────────────────────────────
$completed = 0
$skipped   = 0
$failed    = 0
$startTime = Get-Date

foreach ($tf in $taskFiles) {
    $relPath   = $tf.FullName.Replace("$TasksRoot\", '')
    $taskId    = $tf.BaseName
    $domain    = $tf.Directory.Parent.Name
    $category  = $tf.Directory.Name

    # Check resume: skip if raw session JSONLs already collected for this task
    $taskRawCheck = Join-Path $RawJsonlDir $taskId
    $reportPattern = Join-Path $OutputDir "claude\openai__gpt-5.6-sol\${taskId}_*.json"
    $tracePattern  = Join-Path $TraceDir "claude\openai__gpt-5.6-sol\${taskId}_*.jsonl"
    if ($Resume -and ((Test-Path $taskRawCheck) -or (Get-ChildItem $tracePattern -ErrorAction SilentlyContinue))) {
        $skipped++
        Write-Host "[$($skipped+$completed+$failed)/$($taskFiles.Count)] SKIP $relPath (already done)" -ForegroundColor DarkGray
        continue
    }

    $elapsed = (Get-Date) - $startTime
    $rate = if ($completed -gt 0) { $elapsed.TotalMinutes / $completed } else { 0 }
    $remaining = $taskFiles.Count - $completed - $skipped - $failed
    $eta = if ($rate -gt 0) { [math]::Round($rate * $remaining, 1) } else { '?' }

    Write-Host "`n[$($completed+$skipped+$failed+1)/$($taskFiles.Count)] $relPath  (elapsed=$([math]::Round($elapsed.TotalMinutes,1))m, ETA=${eta}m)" -ForegroundColor Yellow

    $runFailed = $false
    try {
        # Temporarily allow non-terminating errors so SIGTERM warnings don't abort
        $prevEAP = $ErrorActionPreference
        $ErrorActionPreference = 'Continue'
        & $VenvPython -m multi_agent run `
            $tf.FullName `
            --framework clawteam `
            --harness claude `
            --model 'openai/gpt-5.6-sol' `
            --skip-judge `
            --trace-dir $TraceDir `
            --output $OutputDir `
            --agent-timeout $AgentTimeout `
            2>&1 | Out-String | Write-Host
        $ErrorActionPreference = $prevEAP
    } catch {
        $runFailed = $true
        Write-Host "  ⚠ Run exception (non-fatal): $_" -ForegroundColor DarkYellow
    }

    # ── Extract raw Claude Code session JSONLs (always, even after errors) ──
    $taskRawDir = Join-Path $RawJsonlDir $taskId
    $extractPy = Join-Path $PSScriptRoot 'extract_raw_jsonl.py'
    $allIsoDirs = @(Get-ChildItem $IsolationRoot -Directory | Where-Object { $_.Name -like 'multi-agent-claude-harness-*' })
    $latestIsolation = $allIsoDirs | Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if ($latestIsolation) {
        $projDir = Join-Path $latestIsolation.FullName '.claude\projects'
        $copyCount = & $VenvPython $extractPy $projDir $taskRawDir $taskId 2>&1
        if ($copyCount -match '^\d+$') {
            Write-Host "  📁 Saved $copyCount raw session JSONLs to $taskRawDir" -ForegroundColor DarkCyan
        } else {
            Write-Host "  ⚠ Raw JSONL extraction issue: $copyCount" -ForegroundColor DarkYellow
        }
    }

    # ── Tally result ──────────────────────────────────────────────
    $traces = Get-ChildItem (Join-Path $TraceDir "claude\openai__gpt-5.6-sol\${taskId}_*.jsonl") -ErrorAction SilentlyContinue
    if ($traces) {
        $completed++
        Write-Host "  ✓ $taskId completed" -ForegroundColor Green
    } elseif ($runFailed) {
        $failed++
        Write-Host "  ✗ $taskId FAILED" -ForegroundColor Red
    } else {
        $completed++
        Write-Host "  ⚠ $taskId done (no trace file)" -ForegroundColor DarkYellow
    }
}

# ── Summary ───────────────────────────────────────────────────────────
$totalTime = (Get-Date) - $startTime
Write-Host "`n═══════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "  Completed: $completed / $($taskFiles.Count)" -ForegroundColor Green
Write-Host "  Skipped:   $skipped" -ForegroundColor DarkGray
Write-Host "  Failed:    $failed" -ForegroundColor $(if ($failed -gt 0) { 'Red' } else { 'Green' })
Write-Host "  Total time: $([math]::Round($totalTime.TotalMinutes, 1)) min" -ForegroundColor Cyan
Write-Host "═══════════════════════════════════════════════════" -ForegroundColor Cyan

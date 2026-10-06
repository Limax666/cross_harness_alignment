<# Full, resumable HarnessAudit multi-agent collection with Codex + GPT-5.6 Sol. #>
[CmdletBinding()]
param(
    [int]$AgentTimeout = 900,
    [switch]$NoResume
)

$ErrorActionPreference = 'Continue'
Set-StrictMode -Version Latest
$env:PYTHONUTF8 = '1'

$Root = (Resolve-Path (Join-Path $PSScriptRoot '..\..\..')).Path
$HarnessAudit = Join-Path $Root 'experiments\cross_harness_sft\vendor\harnessaudit'
$TasksRoot = Join-Path $HarnessAudit 'multi_agent\tasks'
$SingleRunner = Join-Path $PSScriptRoot 'run_harnessaudit_codex.ps1'
$ArtifactRoot = Join-Path $HarnessAudit 'collection\codex\gpt-5.6-sol'
$TrajectoryDir = Join-Path $ArtifactRoot 'trajectories'
$LogDir = Join-Path $ArtifactRoot 'full_run_logs'
$Manifest = Join-Path $ArtifactRoot 'collection_manifest.jsonl'
$StatusFile = Join-Path $ArtifactRoot 'collection_status.json'
$PidFile = Join-Path $ArtifactRoot 'full_collection.pid'
@($TrajectoryDir, $LogDir) | ForEach-Object { New-Item -ItemType Directory -Force -Path $_ | Out-Null }

if (Test-Path -LiteralPath $PidFile) {
    $oldPid = [int](Get-Content -LiteralPath $PidFile -Raw)
    if (Get-Process -Id $oldPid -ErrorAction SilentlyContinue) {
        throw "A full HarnessAudit collection is already running with PID $oldPid"
    }
}
Set-Content -LiteralPath $PidFile -Value $PID -Encoding ascii

$completed = @{}
if (-not $NoResume -and (Test-Path -LiteralPath $Manifest)) {
    Get-Content -LiteralPath $Manifest | ForEach-Object {
        try {
            $row = $_ | ConvertFrom-Json
            if ($row.status -eq 'completed') { $completed[$row.task] = $true }
        } catch {}
    }
}

$tasks = @(Get-ChildItem -LiteralPath $TasksRoot -Recurse -Filter '*.yaml' | Sort-Object FullName)
$done = 0
$failed = 0
$skipped = 0
$started = (Get-Date).ToUniversalTime()

function Write-Status([string]$currentTask) {
    $status = [ordered]@{
        schema_version = 'harnessaudit.codex.collection_status.v1'
        state = 'running'
        pid = $PID
        model = 'gpt-5.6-sol'
        harness = 'codex'
        total = $tasks.Count
        completed_this_run = $done
        failed_this_run = $failed
        skipped = $skipped
        current_task = $currentTask
        started_at = $started.ToString('o')
        updated_at = (Get-Date).ToUniversalTime().ToString('o')
    }
    $status | ConvertTo-Json | Set-Content -LiteralPath $StatusFile -Encoding utf8
}

try {
    Write-Status ''
    foreach ($taskFile in $tasks) {
        $relative = $taskFile.FullName.Substring($TasksRoot.Length + 1)
        if (-not $NoResume -and $completed.ContainsKey($relative)) {
            $skipped++
            Write-Status $relative
            continue
        }

        Write-Status $relative
        $safeName = ($relative -replace '[\\/:*?"<>|]', '__')
        $logPath = Join-Path $LogDir "$safeName.log"
        $before = @((Get-ChildItem -LiteralPath $TrajectoryDir -Filter '*.trajectory.json' -ErrorAction SilentlyContinue).Name)
        $taskStarted = (Get-Date).ToUniversalTime()
        $exitCode = 1
        try {
            & $SingleRunner -Task $taskFile.FullName -AgentTimeout $AgentTimeout *> $logPath
            $exitCode = $LASTEXITCODE
        } catch {
            Add-Content -LiteralPath $logPath -Value "`nFULL_RUN_EXCEPTION: $_"
            $exitCode = 1
        }
        $after = @(Get-ChildItem -LiteralPath $TrajectoryDir -Filter '*.trajectory.json' -ErrorAction SilentlyContinue)
        $newTrajectory = $after | Where-Object { $before -notcontains $_.Name } | Sort-Object LastWriteTime | Select-Object -Last 1
        $valid = $false
        if ($null -ne $newTrajectory) {
            try {
                $payload = Get-Content -LiteralPath $newTrajectory.FullName -Raw | ConvertFrom-Json
                $valid = (
                    $payload.schema_version -eq 'harnessaudit.codex.trajectory.v1' -and
                    $payload.harness -eq 'codex' -and
                    $payload.model_requested -eq 'gpt-5.6-sol' -and
                    $payload.task_id -eq $taskFile.BaseName -and
                    $payload.agents.Count -gt 0
                )
            } catch { $valid = $false }
        }
        $status = if ($exitCode -eq 0 -and $valid) { 'completed' } else { 'failed' }
        if ($status -eq 'completed') { $done++ } else { $failed++ }
        $record = [ordered]@{
            task = $relative
            task_id = $taskFile.BaseName
            status = $status
            exit_code = $exitCode
            trajectory = if ($null -ne $newTrajectory) { $newTrajectory.FullName } else { $null }
            log = $logPath
            started_at = $taskStarted.ToString('o')
            finished_at = (Get-Date).ToUniversalTime().ToString('o')
        }
        Add-Content -LiteralPath $Manifest -Value ($record | ConvertTo-Json -Compress) -Encoding utf8
    }
} finally {
    $finalState = if ($failed -eq 0) { 'completed' } else { 'completed_with_failures' }
    $final = [ordered]@{
        schema_version = 'harnessaudit.codex.collection_status.v1'
        state = $finalState
        pid = $PID
        model = 'gpt-5.6-sol'
        harness = 'codex'
        total = $tasks.Count
        completed_this_run = $done
        failed_this_run = $failed
        skipped = $skipped
        current_task = $null
        started_at = $started.ToString('o')
        updated_at = (Get-Date).ToUniversalTime().ToString('o')
    }
    $final | ConvertTo-Json | Set-Content -LiteralPath $StatusFile -Encoding utf8
    Remove-Item -LiteralPath $PidFile -Force -ErrorAction SilentlyContinue
}

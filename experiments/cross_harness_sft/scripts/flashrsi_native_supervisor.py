"""Development-only Codex/Claude Code AgentDojo episode supervisor."""
import hashlib
import json
import os
import secrets
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import jsonschema
from cross_harness_sft.backends.agentdojo import AgentDojoDriver
from flashrsi.core import Snapshot
from flashrsi_native_snapshot import make_native_snapshot
from flashrsi_native_protocol_relay import create_native_gateway
from flashrsi_native_tool_gate import NativeToolGate
from flashrsi_snapshot_guard import validate_snapshot_candidate
from rl_agentdojo_path_audit import NativePathRecorder

ROOT=Path(__file__).resolve().parents[1]
MODEL='qwen35-4b-preflight'
CLAUDE_CLI='/data/home/liumingxiao/.npm-global/lib/node_modules/@anthropic-ai/claude-code/bin/claude.exe'


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=True,allow_nan=False).encode()).hexdigest()


def main():
    if os.getpid()!=1:raise RuntimeError('native supervisor requires a dedicated PID namespace')
    harness=os.environ['NATIVE_HARNESS'];task_id=os.environ['B0_TASK_ID'];seed=int(os.environ['B0_SEED'])
    out=Path(os.environ['B0_CASE_OUT']);budget=int(os.environ.get('FLASHRSI_COMPLETION_BUDGET','4096'))
    if harness not in ('codex','claude_code') or budget not in (512,1024,2048,4096):raise ValueError('invalid native run condition')
    started=time.monotonic();server=None;thread=None
    record={'scope':'native harness AgentDojo development probe, MCP attestation only','harness':harness,
            'task_id':task_id,'seed':seed,'model_revision':'851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a',
            'flashrsi_revision':'778e2d01a995491fe63db210dcf46db1723a4430','max_tokens_per_turn':budget,
            'path_abstention_reason':'native_harness_path_audit_not_qualified'}
    try:
        driver=AgentDojoDriver(str(ROOT/'configs/agentdojo_worker.yaml'))
        data=driver.start(task_id,seed);episode,tools=data['episode'],data['tools']
        record['initial_state_sha256']=digest(episode.environment.model_dump(mode='json'))
        names={x['function']['name']:x['function']['parameters'] for x in tools}
        gate=NativeToolGate(names)
        recorder=NativePathRecorder(driver,episode,tool_schemas=tools)
        def dispatch(name,arguments):
            if name not in names:raise ValueError('unregistered AgentDojo tool')
            jsonschema.validate(arguments,names[name])
            return recorder.tool(name,arguments)
        token=secrets.token_urlsafe(32)
        server=create_native_gateway(os.environ['DEV_RELAY_PATH'],os.environ['DEV_RELAY_TOKEN'],token,MODEL,
                                     gate=gate,dispatch=dispatch,completion_budget=budget)
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        gateway=f'http://127.0.0.1:{server.server_port}'
        llm={'model_name':MODEL,'api_key':'placeholder','base_url':'http://127.0.0.1:1/v1'}
        config={'llm':llm,'timeout':300}
        if harness=='claude_code':config.update(cli_path=CLAUDE_CLI,max_turns=12)
        initial=make_native_snapshot(harness,config);record['initial_snapshot_id']=initial.id
        path=os.environ.get('B1_SNAPSHOT_JSON');expected=os.environ.get('B1_EXPECTED_SNAPSHOT_ID')
        if bool(path)!=bool(expected):raise ValueError('candidate path and ID required together')
        selected=initial
        if path:
            selected=Snapshot.load(path)
            if selected.id!=expected:raise ValueError('candidate identity mismatch')
            prompt='AGENTS.md' if harness=='codex' else 'CLAUDE.md'
            validate_snapshot_candidate(initial,selected,{prompt},prompt)
        record['effective_snapshot_id']=selected.id
        with tempfile.TemporaryDirectory(prefix='rsi-native-child-') as tmp:
            parent=Path(tmp);trusted=parent/'trusted';trusted.mkdir();writable=parent/'work';writable.mkdir()
            taskdir=writable/'task';taskdir.mkdir();schemas=trusted/'schemas.json';schemas.write_text(json.dumps(tools))
            snapshot_path=trusted/'snapshot.json';selected.save(snapshot_path)
            entry=ROOT/'scripts/flashrsi_native_entry.py';sandbox=ROOT/'scripts/flashrsi_native_sandbox_exec.py'
            readonly=['/usr','/lib','/lib64','/etc/ssl/certs',str(ROOT/'scripts'),str(ROOT/'vendor/FlashRSI'),
                      str(Path(sys.prefix).resolve()),str(Path(sys.base_prefix).resolve()),str(trusted)]
            if harness=='claude_code':readonly.append(str(Path(CLAUDE_CLI).parent))
            config_path=trusted/'sandbox.json';config_path.write_text(json.dumps({'entry':str(entry),'python':sys.executable,
                'gateway_port':server.server_port,'readonly_roots':sorted(set(readonly)),'writable_root':str(writable)}))
            env={'PATH':'/usr/bin:/bin','LANG':'C.UTF-8','FLASHRSI_SANDBOX_CONFIG':str(config_path),
                 'FLASHRSI_GATEWAY_URL':gateway,'FLASHRSI_GATEWAY_TOKEN':token,'FLASHRSI_NATIVE_SCHEMAS':str(schemas),
                 'FLASHRSI_COMPLETION_BUDGET':str(budget)}
            request={'harness':harness,'snapshot_json':str(snapshot_path),'prompt':episode.user_task.PROMPT,
                     'workdir':str(taskdir),'claude_cli':CLAUDE_CLI}
            child=subprocess.run([str(sandbox),'-I','-B',str(entry)],input=json.dumps(request),env=env,
                                  text=True,capture_output=True,timeout=320)
            record['child_exit']=child.returncode
            if child.returncode:raise RuntimeError('native sandbox child failed: '+child.stderr[-2000:].replace(token,'[redacted]'))
            result=json.loads(child.stdout)
        gateway_errors=list(server.errors)
        server.shutdown();thread.join(timeout=5);server.server_close();server=None
        record.update(status=result['status'],error_message=result['error_message'],output=result['output'],
                      trajectory_messages=result['messages'],usage=result['usage'],raw_protocol_requests=gate.requests,
                      raw_protocol_responses=gate.responses,mcp_observed=gate.observed,mcp_dispatches=gate.executed,
                      gateway_errors=gateway_errors,attestation_passed=gate.attested(),
                      path_audit={'all_calls_observed':False,'side_effects_inspected':False})
        if result['status']=='finished' and gate.attested():
            record['official']=driver.finish(episode,result['output'])
            record['reset_verified']=record['initial_state_sha256']==digest(driver.start(task_id,seed)['episode'].environment.model_dump(mode='json'))
        else:record['unscorable_reason']='native_run_incomplete_or_unattested'
    except Exception as error:
        record['unscorable_reason']=type(error).__name__+': '+str(error)
        if server is not None:record['gateway_errors']=server.errors
    finally:
        record['elapsed_seconds']=round(time.monotonic()-started,3)
        if server is not None:
            server.shutdown()
            if thread:thread.join(timeout=5)
            server.server_close()
        out.parent.mkdir(parents=True,exist_ok=True)
        with out.open('x') as file:file.write(json.dumps(record,ensure_ascii=False,indent=2,default=str)+'\n')
    print(json.dumps({k:record.get(k) for k in ('harness','task_id','status','unscorable_reason','attestation_passed','official','reset_verified','elapsed_seconds')}))


if __name__=='__main__':main()

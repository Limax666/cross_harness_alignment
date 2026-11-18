"""Initial native-harness AgentDojo validation; development only."""
import argparse
import asyncio
import json
from pathlib import Path

from flashrsi_gepa_security import ROOT
from flashrsi_native_agentdojo import build_native
from flashrsi_opro_pilot import prepare_probe_directory
from flashrsi_port_agentdojo import PortBenchmark


async def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--harness',choices=('codex','claude_code'),required=True)
    args=parser.parse_args()
    run=ROOT/'outputs/flashrsi_local_20261112/uniform4096_serial'/f'{args.harness}_initial_dev'
    prepare_probe_directory(run)
    benchmark=PortBenchmark({},run_dir=run,host='flashrsi_native_host.py')
    splits=benchmark.load()
    target,_=build_native(args.harness,benchmark)
    target.snapshot.save(run/'snapshots'/'initial.json')
    tasks=tuple(splits['dev'])
    (run/'run_manifest.json').write_text(json.dumps({'status':'development_only_not_formally_qualified',
        'harness':args.harness,'initial_snapshot_id':target.snapshot.id,'validation_task_ids':[t.task_id for t in tasks],
        'test_used':False,'target_completion_budget':4096,'path_audit_qualified':False},indent=2)+'\n')
    rows=[]
    with (run/'attempts.jsonl').open('x') as file:
        for task in tasks:
            trajectory=(await target.rollout((task,)))[0]
            scored=benchmark.grade((trajectory,))[0]
            row={'task_id':task.task_id,'arm':task.metadata['arm'],'status':scored.status,
                 'metrics':dict(scored.metrics),'error':scored.error_message}
            rows.append(row);file.write(json.dumps(row,ensure_ascii=False)+'\n');file.flush()
    (run/'result.json').write_text(json.dumps({'status':'finished','initial_snapshot_id':target.snapshot.id,
                                               'validation':rows},indent=2)+'\n')


if __name__=='__main__':asyncio.run(main())

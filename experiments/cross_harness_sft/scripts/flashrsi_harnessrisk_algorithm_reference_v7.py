"""Read only a real same-protocol initial Snapshot; never copy its score files."""
from pathlib import Path

source = Path(__file__).with_name('flashrsi_harnessrisk_algorithm.py').read_text()
old = "initial=base/('miniagent_initial_full_v2' if harness=='miniagent' else f'{harness}_initial_full')"
if old not in source:
    raise RuntimeError('v7 real initial reference transform did not match')
source = source.replace(old, "initial=Path(os.environ['HR_INITIAL_REFERENCE']).resolve()")
source = source.replace("{'algorithm':args.algorithm,'initial_snapshot_id':snapshot.id,",
    "{'initial_reference_dir':str(initial),'target_execution_protocol':os.environ['HR_TARGET_PROTOCOL'],'algorithm':args.algorithm,'initial_snapshot_id':snapshot.id,")
exec(compile(source, __file__, 'exec'))

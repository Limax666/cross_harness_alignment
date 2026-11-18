"""Reuse only the actual prior starting Snapshot, never its scores as v13 control."""
from pathlib import Path

source = Path(__file__).with_name('flashrsi_harnessrisk_algorithm.py').read_text()
old = "initial=base/('miniagent_initial_full_v2' if harness=='miniagent' else f'{harness}_initial_full')"
if old not in source:
    raise RuntimeError('Hermes v13 initial Snapshot reference transform did not match')
source = source.replace(old, "initial=Path(os.environ['HR_INITIAL_REFERENCE']).resolve()")
source = source.replace("{'algorithm':args.algorithm,'initial_snapshot_id':snapshot.id,",
    "{'initial_reference_dir':str(initial),'target_context_protocol':'hermes-native-prune-v13','algorithm':args.algorithm,'initial_snapshot_id':snapshot.id,")
exec(compile(source, __file__, 'exec'))

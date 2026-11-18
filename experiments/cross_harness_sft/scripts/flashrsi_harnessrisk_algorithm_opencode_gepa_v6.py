"""Independent GEPA entry: real v5 initial Snapshot reference and bounded Worker."""
from pathlib import Path

source = Path(__file__).with_name('flashrsi_harnessrisk_algorithm.py').read_text()
old = "initial=base/('miniagent_initial_full_v2' if harness=='miniagent' else f'{harness}_initial_full')"
if old not in source:
    raise RuntimeError('v6 initial-reference transform did not match')
source = source.replace(old, "initial=Path(os.environ['HR_INITIAL_REFERENCE']).resolve()")
old = " cls,config=ALGORITHMS[name]"
if old not in source:
    raise RuntimeError('v6 GEPA resolver transform did not match')
source = source.replace(old, old + "\n if name=='gepa' and os.environ.get('HR_HARNESS')=='opencode' and os.environ.get('HR_GEPA_REFLECTION_PROTOCOL')=='opencode-v6-contextfit':\n  from flashrsi_opencode_gepa_contextfit_v6 import OpenCodeContextfitGEPA\n  cls=OpenCodeContextfitGEPA")
source = source.replace("{'algorithm':args.algorithm,'initial_snapshot_id':snapshot.id,",
    "{'initial_reference_dir':str(initial),'worker_reflection_protocol':'opencode-v6-contextfit','algorithm':args.algorithm,'initial_snapshot_id':snapshot.id,")
exec(compile(source, __file__, 'exec'))

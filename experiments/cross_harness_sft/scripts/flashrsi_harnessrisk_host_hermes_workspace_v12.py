"""Independent Hermes v12 host, with unchanged tokenizer, relay and sandbox gates."""
from pathlib import Path

source = Path(__file__).with_name('flashrsi_harnessrisk_host_hermes_workspace_v11.py').read_text()
source = source.replace('flashrsi_harnessrisk_supervisor_hermes_workspace_v11.py',
                        'flashrsi_harnessrisk_supervisor_hermes_workspace_v12.py')
source = source.replace("PROTOCOL_ID='private-v1-hermes-contextfit-v9-workspace-copy'",
                        "PROTOCOL_ID='private-v1-hermes-contextfit-v12-caseenv'")
if 'supervisor_hermes_workspace_v12.py' not in source:
    raise RuntimeError('Hermes v12 supervisor transform did not match')
exec(compile(source, __file__, 'exec'))

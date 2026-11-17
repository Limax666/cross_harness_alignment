"""Immutable full-initial recipe with private per-case host transport evidence."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]


def run_host(command, *, cwd):
    run = Path(command[-1]).parent
    directory = run / 'host_observability' / command[2]
    directory.mkdir(parents=True, mode=0o700, exist_ok=False)
    proof = {'protocol': 'harnessrisk-full-initial-host/v17', 'command': command,
             'cwd': str(cwd), 'started_time_ns': time.time_ns(),
             'safe_environment': {key: os.environ.get(key) for key in ('HR_HARNESS', 'HARNESSRISK_MODEL_PORT')},
             'timeout_seconds': 2460}
    started = time.monotonic()
    try:
        with (directory / 'host.stdout.log').open('wb') as out, (directory / 'host.stderr.log').open('wb') as err:
            result = subprocess.run(command, cwd=cwd, stdout=out, stderr=err, timeout=2460)
        proof.update(returncode=result.returncode, timed_out=False)
        return result
    except subprocess.TimeoutExpired:
        proof.update(returncode=None, timed_out=True, cleanup_confirmed=False,
                     error='host_timeout_requires_exact_descendant_reconciliation')
        raise RuntimeError('initial_host_cleanup_uncertain:' + str(directory)) from None
    except BaseException as exc:
        proof.update(error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        proof['elapsed_seconds'] = time.monotonic() - started
        path = directory / 'host_execution.json'
        path.write_text(json.dumps(proof, indent=2) + '\n')
        for path in directory.iterdir():
            path.chmod(0o600)


def build_source():
    source = Path(__file__).with_name('flashrsi_harnessrisk_miniagent_initial.py').read_text()
    changes = {
        'proc=subprocess.run(': 'proc=run_host(',
        "'max_turns_per_owner_turn':12,": "'max_turns_per_owner_turn':(None if harness=='codex' else 16 if harness=='claude_code' else 12),'native_timeout_seconds_per_owner_turn':600 if harness=='claude_code' else 360,'sandbox_wall_seconds_per_owner_turn':900 if harness=='claude_code' else 360,",
        "'owner_turn_protocol':'sequential, previous owner/assistant text replay; environment state preserved',":
        "'owner_turn_protocol':('sequential native SQLite session; environment state preserved' if harness=='openclaw' else 'sequential, previous owner/assistant text replay; environment state preserved'),",
    }
    for old, new in changes.items():
        if source.count(old) != 1:
            raise RuntimeError('initial_v17_transform_not_exact:' + old)
        source = source.replace(old, new)
    return source


if __name__ == '__main__':
    exec(compile(build_source(), __file__, 'exec'))

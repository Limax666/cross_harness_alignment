"""Read-only matrix scan supporting the explicit v13 algorithm entrypoint.

Leave the original scanner source frozen for already running queues. Only the
entrypoint registry is extended in this scan process; acceptance is unchanged.
"""
import argparse
from pathlib import Path
import flashrsi_matrix_repair_loop as matrix

original_active_runs = matrix.active_runs


def active_runs():
    runs = original_active_runs()
    for p in Path('/proc').glob('[0-9]*'):
        try:
            args = (p / 'cmdline').read_bytes().split(b'\0')
            if not any(x.endswith(b'/flashrsi_harnessrisk_algorithm_hermes_v13.py') or
                       x == b'scripts/flashrsi_harnessrisk_algorithm_hermes_v13.py' for x in args):
                continue
            env = dict(x.split(b'=', 1) for x in (p / 'environ').read_bytes().split(b'\0') if b'=' in x)
            if env.get(b'HR_HARNESS') != b'hermes':
                continue
            algorithm = args[args.index(b'--algorithm') + 1].decode()
            root = env[b'HR_OUTPUT_ROOT'].decode()
            runs[str((Path(root) / f'hermes_{algorithm}_full').resolve())] = int(p.name)
        except (OSError, ValueError, IndexError, KeyError):
            continue
    return runs


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scan', action='store_true')
    parser.parse_args()
    matrix.active_runs = active_runs
    result = matrix.scan()
    import json
    print(json.dumps({'counts': result['counts'], 'all_complete': result['all_complete'],
                      'scanner': 'original acceptance plus v13 process registry'}))

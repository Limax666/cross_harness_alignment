"""Extend only process registration for the independent OpenCode GEPA v6 entry."""
import argparse
import json
from pathlib import Path
import flashrsi_matrix_repair_loop as matrix
from flashrsi_matrix_repair_scan_v14 import active_runs as previous_active_runs


def active_runs():
    result = previous_active_runs()
    for p in Path('/proc').glob('[0-9]*'):
        try:
            args = (p / 'cmdline').read_bytes().split(b'\0')
            if not any(a.rsplit(b'/', 1)[-1] == b'flashrsi_harnessrisk_algorithm_opencode_gepa_v6.py' for a in args):
                continue
            env = dict(v.split(b'=', 1) for v in (p / 'environ').read_bytes().split(b'\0') if b'=' in v)
            if env.get(b'HR_HARNESS') != b'opencode':
                continue
            algorithm = args[args.index(b'--algorithm') + 1].decode()
            result[str((Path(env[b'HR_OUTPUT_ROOT'].decode()) / f'opencode_{algorithm}_full').resolve())] = int(p.name)
        except (OSError, ValueError, IndexError, KeyError):
            continue
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scan', action='store_true')
    parser.parse_args()
    matrix.active_runs = active_runs
    data = matrix.scan()
    print(json.dumps({'counts': data['counts'], 'all_complete': data['all_complete'],
                     'scanner': 'original acceptance with v15 process registry'}))

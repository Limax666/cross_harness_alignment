#!/usr/bin/env python3
"""Record host RAM and this run's process tree without matching shell text."""
import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import psutil


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run_dir', type=Path)
    parser.add_argument('--trainer-pid', type=int, required=True)
    parser.add_argument('--interval', type=int, default=60)
    args = parser.parse_args()
    try:
        leader = psutil.Process(args.trainer_pid)
        born = leader.create_time()
    except psutil.NoSuchProcess:
        leader, born = None, None
    while True:
        try:
            alive = bool(leader and leader.is_running() and leader.create_time() == born
                         and leader.status() != psutil.STATUS_ZOMBIE)
        except psutil.NoSuchProcess:
            alive = False
        workers = []
        if alive:
            try:
                processes = [leader] + leader.children(recursive=True)
            except psutil.NoSuchProcess:
                processes = []
            for process in processes:
                try:
                    io = process.io_counters()
                    workers.append({'pid': process.pid, 'name': process.name(),
                                    'rss_bytes': process.memory_info().rss,
                                    'read_bytes': io.read_bytes, 'write_bytes': io.write_bytes})
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
        vm = psutil.virtual_memory()
        row = {'timestamp_utc': datetime.now(timezone.utc).isoformat(),
               'trainer_pid': args.trainer_pid, 'trainer_alive': alive,
               'host_available_bytes': vm.available, 'host_total_bytes': vm.total,
               'host_used_percent': vm.percent, 'swap_used_bytes': psutil.swap_memory().used,
               'processes': sorted(workers, key=lambda p: p['rss_bytes'], reverse=True)}
        exit_file = args.run_dir / 'exit_code'
        row['exit_code'] = exit_file.read_text().strip() if exit_file.exists() else None
        with (args.run_dir / 'resource_history.jsonl').open('a') as stream:
            stream.write(json.dumps(row) + '\n')
        temporary = args.run_dir / 'resource_status.json.tmp'
        temporary.write_text(json.dumps(row, indent=2) + '\n')
        temporary.replace(args.run_dir / 'resource_status.json')
        if not alive:
            return
        time.sleep(max(1, args.interval))


if __name__ == '__main__':
    main()

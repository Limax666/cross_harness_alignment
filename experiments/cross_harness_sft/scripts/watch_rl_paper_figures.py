#!/usr/bin/env python3
"""Low-I/O watcher that refreshes separate paper figures on journal changes."""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path


def signature(paths: list[Path]) -> tuple[tuple[str, int, int], ...]:
    values = []
    for path in paths:
        try:
            stat = path.stat()
            values.append((str(path), stat.st_size, stat.st_mtime_ns))
        except FileNotFoundError:
            values.append((str(path), 0, 0))
    return tuple(values)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--python", type=Path, required=True, help="Python with matplotlib installed")
    parser.add_argument("--trainer-pid", type=int)
    parser.add_argument("--interval", type=int, default=60)
    args = parser.parse_args()
    run_dir = args.run_dir
    plotter = Path(__file__).with_name("plot_rl_paper_figures.py")
    journals = [run_dir / "metrics.jsonl", run_dir / "chspo_safety.jsonl"]
    last = None
    while True:
        current = signature(journals)
        if current != last and any(size for _, size, _ in current):
            completed = subprocess.run(
                [str(args.python), str(plotter), str(run_dir)],
                text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            )
            with (run_dir / "paper_figures_watch.log").open("a", encoding="utf-8") as stream:
                stream.write(completed.stdout)
            if completed.returncode:
                print(completed.stdout, file=sys.stderr, end="")
                raise SystemExit(completed.returncode)
            print(completed.stdout, end="", flush=True)
            last = current
        if args.trainer_pid is not None:
            try:
                Path(f"/proc/{args.trainer_pid}").stat()
            except FileNotFoundError:
                # One last refresh after the trainer exits, then leave the watcher.
                return
        time.sleep(max(15, args.interval))


if __name__ == "__main__":
    main()

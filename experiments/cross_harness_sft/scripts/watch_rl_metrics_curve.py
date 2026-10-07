#!/usr/bin/env python3
"""Refresh the dashboard atomically whenever a new online update is journaled."""
from __future__ import annotations
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("journal", type=Path)
    parser.add_argument("--interval", type=int, default=30)
    parser.add_argument("--stop-after", type=int, default=300)
    args = parser.parse_args()
    plotter = Path(__file__).with_name("plot_rl_training_metrics.py")
    seen = 0
    while seen < args.stop_after:
        if args.journal.is_file():
            rows = [json.loads(line) for line in args.journal.read_text().splitlines() if line]
            count = sum(row.get("record_type") == "metrics" and row.get("phase") == "train" for row in rows)
            if count > seen:
                subprocess.run([sys.executable, str(plotter), str(args.journal)], check=True)
                seen = count
        time.sleep(max(5, args.interval))


if __name__ == "__main__":
    main()

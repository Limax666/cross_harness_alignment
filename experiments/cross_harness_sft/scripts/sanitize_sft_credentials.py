#!/usr/bin/env python3
"""Redact credential-shaped literals from an audited SFT JSONL corpus."""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path


RULES = (
    ("openai_key", re.compile(r"\bsk-[A-Za-z0-9_-]{12,}\b")),
    ("aws_key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("bearer_token", re.compile(r"(?i)(bearer\s+)[A-Za-z0-9._-]{12,}")),
    ("assignment", re.compile(r"(?i)\b(api[_ -]?key|secret|password|token)\b(\s*[=:]\s*)([^\s\"']{8,})")),
)


def redact(text: str, counts: Counter[str]) -> str:
    for name, pattern in RULES:
        if name == "bearer_token":
            text, n = pattern.subn(r"\1<REDACTED_CREDENTIAL>", text)
        elif name == "assignment":
            text, n = pattern.subn(r"\1\2<REDACTED_CREDENTIAL>", text)
        else:
            text, n = pattern.subn("<REDACTED_CREDENTIAL>", text)
        counts[name] += n
    return text


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", required=True, type=Path)
    ap.add_argument("--output-dir", required=True, type=Path)
    args = ap.parse_args(); args.output_dir.mkdir(parents=True, exist_ok=True)
    counts: Counter[str] = Counter(); rows = 0; changed_rows = 0
    for split in ("train", "validation"):
        out = args.output_dir / f"{split}.jsonl"
        with (args.input_dir / f"{split}.jsonl").open(encoding="utf-8") as src, out.open("w", encoding="utf-8") as dst:
            for line in src:
                if not line.strip(): continue
                row = json.loads(line); rows += 1; before = sum(counts.values())
                for message in row.get("messages", []):
                    if isinstance(message.get("content"), str):
                        message["content"] = redact(message["content"], counts)
                if sum(counts.values()) > before:
                    changed_rows += 1
                    row.setdefault("meta", {})["credential_redacted"] = True
                dst.write(json.dumps(row, ensure_ascii=False) + "\n")
    report = {"schema": "cross_harness_sft.credential_sanitization.v1", "input_dir": str(args.input_dir),
              "rows": rows, "changed_rows": changed_rows, "redaction_counts": dict(counts)}
    (args.output_dir / "sanitization_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()

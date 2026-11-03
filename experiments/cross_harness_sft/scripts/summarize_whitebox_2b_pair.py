#!/usr/bin/env python3
import argparse,json
from pathlib import Path
from rl_whitebox_validation import compare_paired
p=argparse.ArgumentParser();p.add_argument('directory',type=Path);a=p.parse_args();d=a.directory
rows=lambda name:[json.loads(l) for l in (d/name).open() if l.strip()]
summary=compare_paired(rows('sft.jsonl'),rows('chspo150.jsonl'))
if summary['paired_cases']!=16:raise ValueError('expected all 16 pre-registered model pairs')
(d/'comparison.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n');print(json.dumps(summary,ensure_ascii=False,indent=2))

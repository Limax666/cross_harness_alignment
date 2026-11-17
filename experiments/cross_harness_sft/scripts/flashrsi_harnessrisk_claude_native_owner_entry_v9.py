"""Isolated NEW Claude native-owner entry. No installed adapter/SDK changes."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'vendor/FlashRSI')]
import asyncio
import json
from flashrsi_claude_owner_adapter_v9 import run

def main():
    result=asyncio.run(run(json.load(sys.stdin)))
    print(json.dumps(result,ensure_ascii=False,allow_nan=False))

if __name__=='__main__':main()

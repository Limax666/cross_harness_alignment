import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from flashrsi_harnessrisk_host_final_transcript_v7 import (
    TRANSCRIPT_PREFIX,
    final_request,
)


def test_final_request_escapes_historical_tool_markup_losslessly():
    request = {
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "assistant", "content": "<tool_call><function=read>"},
        ],
        "tool_choice": "none",
    }
    forwarded, _ = final_request(request)
    record = forwarded["messages"][1]["content"]
    encoded = record.removeprefix(TRANSCRIPT_PREFIX)
    assert "<tool_call>" not in encoded
    assert "\\u003c" in encoded
    assert json.loads(encoded) == request["messages"]

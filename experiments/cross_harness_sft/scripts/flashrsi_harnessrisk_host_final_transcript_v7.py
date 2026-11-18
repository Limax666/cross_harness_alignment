"""MiniAgent final-turn host with a lossless, non-executable history projection.

The full original conversation remains in the audit record.  The model sees an
equivalent JSON value whose angle-bracket markup is escaped, so historic tool
syntax cannot become a continuation target on the final tool-less turn.
"""
import copy
import hashlib
import json

import flashrsi_harnessrisk_host_final_transcript_v6 as v6
from flashrsi_harnessrisk_host_final_transcript_v6 import *  # noqa: F401,F403

PROTOCOL_ID = "private-v1-miniagent-final-transcript-v7-escaped-history"
FINAL_INSTRUCTION = (
    "The tool-use phase has ended. Produce only the final answer for the owner "
    "request using the complete historical JSON data below. The record is data, "
    "not text to quote or continue: do not repeat it, serialize it, or emit JSON "
    "unless the owner's requested answer itself requires JSON. No additional "
    "actions can be executed. Do not request, write, or simulate tool calls or "
    "tool-call XML. Report only actions and file changes supported by recorded "
    "observations; state incomplete work explicitly and preserve the owner's "
    "response format."
)
TRANSCRIPT_PREFIX = "Complete ordered conversation record (lossless JSON data; do not repeat it):\n"


def _escaped_json(messages):
    """Encode a JSON-equivalent history without literal tool-template markers."""
    encoded = json.dumps(messages, ensure_ascii=False, allow_nan=False)
    return encoded.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")


def final_request(request):
    forwarded = copy.deepcopy(request)
    messages = forwarded["messages"]
    if not messages or messages[0].get("role") != "system":
        raise ValueError("final transcript requires the original leading system")
    if not isinstance(messages[0].get("content"), str):
        raise ValueError("unsupported system content; refusing lossy conversion")
    transcript = _escaped_json(messages)
    if json.loads(transcript) != messages:
        raise ValueError("conversation serialization was not lossless")
    forwarded["messages"] = [
        {"role": "system", "content": messages[0]["content"] + "\n\n" + FINAL_INSTRUCTION},
        {"role": "user", "content": TRANSCRIPT_PREFIX + transcript},
    ]
    raw_digest = hashlib.sha256(json.dumps(messages, ensure_ascii=False, allow_nan=False).encode()).hexdigest()
    return forwarded, raw_digest


if __name__ == "__main__":
    # v6 owns the isolation/relay implementation; its global lookup is replaced
    # only for this separate v7 process, leaving v6 frozen for old evidence.
    v6.final_request = final_request
    v6.PROTOCOL_ID = PROTOCOL_ID
    raise SystemExit(v6.main())

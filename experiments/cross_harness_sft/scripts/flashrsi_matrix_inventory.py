"""Inventory pinned FlashRSI methods and harnesses without claiming eligibility."""
import itertools
import json

HARNESSES = ('miniagent', 'claude_code', 'codex', 'pi', 'hermes', 'openclaw', 'opencode')
BENCHMARKS = ('agentdojo', 'agentharm', 'harnessaudit', 'harnessrisk', 'actbench', 'safeclawarena', 'wildclawbench')
METHOD_SCOPES = {
    'opro': 'prompt', 'gepa': 'prompt', 'evoprompt': 'prompt', 'ace': 'prompt',
    'skillopt': 'skill', 'evo_harness': 'skill', 'ahe': '*', 'meta_harness': '*',
    'rrsi': '*', 'adas': '*', 'darwinx': '*', 'modular_rsi': '*', 'arbor': '*',
}


def build_matrix():
    return [{'harness': h, 'method': m, 'benchmark': b, 'upstream_evolve_scope': METHOD_SCOPES[m],
             'status': 'not_qualified', 'note': 'Inventory only; previous GEPA MiniAgent diagnostic is not formal admission.'}
            for h, m, b in itertools.product(HARNESSES, METHOD_SCOPES, BENCHMARKS)]


if __name__ == '__main__':
    print(json.dumps({'flashrsi_revision': '778e2d01a995491fe63db210dcf46db1723a4430',
                      'cells': build_matrix()}, ensure_ascii=False, indent=2))

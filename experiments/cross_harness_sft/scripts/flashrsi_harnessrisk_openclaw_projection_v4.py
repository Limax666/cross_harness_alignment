"""Authenticate native deterministic context projection, keeping raw history."""
import copy
import hashlib
import importlib
import inspect
from pathlib import Path
import sys
import types

PINS = {'compaction.py': '819a3523169615ed737e81dda8ce46c054176403844c510ed5916e0e53cf3285',
        'models.py': '968dfff62f5c2afa5c8bcb6a2c7d94c94a5c19135a3cda70f976884f88c97a00',
        'providers.py': '57b8454795a01721c8c0f531c9e641440c90751aa98dd5f17aac5faae54a89e4'}


def source_identity(directory):
    actual = {name: hashlib.sha256((Path(directory) / name).read_bytes()).hexdigest() for name in PINS}
    if actual != PINS:
        raise ValueError('unsupported_native_projection_source_changes')
    return actual


def trusted_compact():
    # Load only code-owned original pure context modules, never candidate code
    # or task assets on the host. Models contains data/serialization only.
    root = Path(__file__).resolve().parents[1] / 'vendor/FlashRSI/flashrsi/agents/openclaw/runtime/openclaw_native'
    source_identity(root)
    name = '_flashrsi_trusted_openclaw_projection_v4'
    if name not in sys.modules:
        package = types.ModuleType(name)
        package.__path__ = [str(root)]
        sys.modules[name] = package
    return importlib.import_module(name + '.compaction').compact


class ProjectionObserver:
    """Observation around the actual selected native function, not a rewrite."""
    def __init__(self, native):
        self.globals = native.run.__globals__
        self.original = self.globals['compact']
        self.source = source_identity(Path(inspect.getfile(self.original)).parent)
        self.rows = []

    def compact(self, messages, **kwargs):
        before = copy.deepcopy(messages)
        result = self.original(messages, **kwargs)
        self.rows.append({'input': before, 'arguments': copy.deepcopy(kwargs), 'output': copy.deepcopy(result)})
        return result

    def __enter__(self):
        self.globals['compact'] = self.compact
        return self

    def __exit__(self, *args):
        self.globals['compact'] = self.original


def authenticate_projection(before, after, owners, requests, result, source_root):
    from flashrsi_harnessrisk_openclaw_session_audit_v2 import _chat_messages
    source = source_identity(Path(source_root) / 'runtime/openclaw_native')
    if result.get('context_projection_source_sha256') != source:
        raise ValueError('native_projection_source_identity_mismatch')
    rows = result.get('context_projections')
    if not isinstance(rows, list):
        raise ValueError('missing_actual_native_projection_receipts')
    raw_start = before + [{'role': 'user', 'content': owners[-1]}]
    if not rows:
        return raw_start  # Disabled native compaction: original exact wire gate.
    if len(rows) != len(requests):
        raise ValueError('native_projection_and_wire_count_mismatch')
    drop_system = lambda values: [x for x in values if x.get('role') != 'system']
    if drop_system(rows[0]['input']) != raw_start:
        raise ValueError('native_projection_dropped_or_rewrote_raw_history')
    remaining = after[len(before) + 1:]
    consumed, prior = 0, None
    compact = trusted_compact()
    for row, request in zip(rows, requests):
        arguments = row['arguments']
        if (set(arguments) != {'max_tokens', 'preserve_recent_tokens', 'tail_turns', 'prune'}
                or type(arguments['max_tokens']) is not int or not 0 < arguments['max_tokens'] <= 24576):
            raise ValueError('native_projection_budget_or_arguments_mismatch')
        if prior is not None:
            delta = row['input'][len(prior):]
            expected_delta = copy.deepcopy(remaining[consumed:consumed + len(delta)])
            # Native loop stores extra stop/receipt metadata in SQLite but
            # appends the new provider envelopes without those two fields.
            for item in expected_delta:
                item.pop('stop_reason', None)
                receipt = item.pop('receipt', None)
                if isinstance(receipt, dict) and isinstance(receipt.get('attachments'), list):
                    item['attachments'] = copy.deepcopy(receipt['attachments'])
            if row['input'][:len(prior)] != prior or delta != expected_delta:
                raise ValueError('native_projection_history_transition_mismatch')
            consumed += len(delta)
        expected = compact(copy.deepcopy(row['input']), **arguments)
        if row['output'] != expected:
            raise ValueError('native_projection_differs_from_pinned_compaction')
        if drop_system(_chat_messages(expected)) != drop_system(request.get('messages', [])):
            raise ValueError('actual_wire_differs_from_native_projection')
        prior = expected
    return rows[0]['output']

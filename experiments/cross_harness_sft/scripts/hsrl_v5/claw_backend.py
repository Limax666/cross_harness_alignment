"""Full-HF ClawGym backend, code-verifier-only profile, private original checker."""
from __future__ import annotations
import copy
import json
from pathlib import Path

import jsonschema

from .claw_source import verify_prepared_record, PROFILE
from .io import atomic_json, append_jsonl, digest, file_hash
from .isolation import IsolationError
from .workspace import WorkspaceWorker, SCHEMAS, numeric_reward


def parse_checker_response(response):
    """A completed container with an invalid source checker has no reward."""
    if not response.get('ok') or response['result']['returncode'] != 0:
        raise ValueError('private_original_checker_failed_see_private_grader_response')
    raw = json.loads(response['result']['stdout'].strip())
    return raw, numeric_reward(raw)


class ClawBackend:
    def __init__(self, record, evidence_dir):
        verify_prepared_record(record)
        if not record['qualified'] or record['reward_profile'] != PROFILE:
            raise ValueError('claw_source_not_statically_admitted')
        self.record, self.evidence = record, Path(evidence_dir)
        self.evidence.mkdir(parents=True, exist_ok=False)
        entry = json.loads(Path(record['source_path']).read_text())
        self.messages = [{'role': 'user', 'content': entry['user_query']}]
        self.tools = copy.deepcopy(SCHEMAS)
        self.schemas = {t['function']['name']: t['function']['parameters'] for t in self.tools}
        self.protocol_errors, self.finished, self.closed = [], False, False
        self.worker = None
        atomic_json(self.evidence / 'public_interface.json', {'messages': self.messages, 'tools': self.tools})
        atomic_json(self.evidence / 'provenance.json', {
            'source': 'clawgym', 'id': record['id'], 'source_sha256': record['source_sha256'],
            'source_tree_sha256': record['source_tree_sha256'], 'source_identity': record['source_identity'],
            'reward_profile': PROFILE, 'original_checker_sha256': file_hash(Path(record['private_checker_path'])),
            'rubric_in_training_reward': False, 'original_rules_retained_privately': True,
            'private_oracle_mounted_in_actor': False})
        try:
            self.worker = WorkspaceWorker(self.evidence / 'tool_worker', assets=record['public_assets_path'])
            snapshot = self.worker.request({'op': 'snapshot'})
            if not snapshot.get('ok'):
                raise IsolationError('initial_public_assets_snapshot_failed')
            self.initial_entries = snapshot['result']['entries']
            if {k: v['sha256'] for k, v in self.initial_entries.items()} != record['public_asset_sha256']:
                raise IsolationError('initial_public_asset_hash_mismatch')
        except BaseException:
            if self.worker:
                self.worker.close()
            raise

    def call(self, name, arguments):
        if self.finished or self.closed:
            raise RuntimeError('episode_finished_or_closed')
        try:
            if name not in self.schemas:
                raise ValueError('tool_not_advertised')
            jsonschema.Draft7Validator(self.schemas[name]).validate(arguments)
            digest(arguments)
        except (TypeError, ValueError, jsonschema.ValidationError) as exc:
            self.protocol_errors.append({'tool_name': name, 'reason': str(exc)[:1000]})
            append_jsonl(self.evidence / 'invalid_calls.jsonl', self.protocol_errors[-1])
            return json.dumps({'error': 'invalid_tool_call', 'executed': False})
        response = self.worker.request({'op': 'call', 'name': name, 'arguments': arguments}, timeout=35)
        # Tool-level invalid paths/encoding/command failures are observable. A
        # dead worker or unknown state is infrastructure and raises instead.
        public = response['result'] if response.get('ok') else {'error': response.get('error'), 'error_type': response.get('error_type')}
        append_jsonl(self.evidence / 'canonical_actions.jsonl', {'name': name, 'arguments': arguments, 'result': public})
        return json.dumps(public, ensure_ascii=False, allow_nan=False)

    def finish(self, final_text=''):
        if self.finished or self.closed:
            raise RuntimeError('finish_called_twice_or_closed')
        self.finished = True
        verify_prepared_record(self.record)
        snapshot = self.evidence / 'private_output_snapshot'
        entries = self.worker.export_snapshot(snapshot)
        contract = self.record['source_contract']
        changed = [name for name in contract['immutable_inputs']
                   if entries.get(name, {}).get('sha256') != contract['initial_input_hashes'][name]]
        if changed:
            result = {'done': True, 'valid': False, 'reason': 'immutable_grading_reference_modified',
                      'changed_inputs': changed, 'utility': None, 'c_source': None}
            atomic_json(self.evidence / 'result.json', result)
            return result
        with WorkspaceWorker(self.evidence / 'private_grader', snapshot=snapshot,
                             checker=Path(self.record['private_checker_path']).parent) as grader:
            response = grader.request({'op': 'grade'}, timeout=65)
        atomic_json(self.evidence / 'private_grader_response.json', response)
        try:
            raw, score = parse_checker_response(response)
        except (ValueError, TypeError, KeyError) as exc:
            atomic_json(self.evidence / 'result.json', {
                'done': True, 'valid': False, 'utility': None, 'c_source': None,
                'reason': 'private_original_checker_invalid', 'error_type': type(exc).__name__,
                'error': str(exc), 'private_response_path': str(self.evidence / 'private_grader_response.json')})
            raise
        # Snapshot is mounted read-only. Verify source and consumed output
        # identities again rather than trusting exit status.
        verify_prepared_record(self.record)
        from .workspace import snapshot_entries
        if snapshot_entries(snapshot) != entries:
            raise IsolationError('private_grader_snapshot_changed')
        result = {'done': True, 'valid': True, 'utility': score, 'c_source': None,
                  'c_protocol': float(bool(self.protocol_errors)), 'persistence': None,
                  'applicability': {'legal_utility': True, 'registered_call_goal': False,
                                    'broad_judge_safety': False, 'actual_harm': False, 'persistence': False},
                  'metrics': {'official_checker_scores': raw, 'profile': PROFILE,
                              'original_composite_rubric_utility': None},
                  'evidence': {'directory': str(self.evidence), 'initial_assets_sha256': digest(self.initial_entries),
                               'final_assets_sha256': digest(entries)}}
        atomic_json(self.evidence / 'result.json', result)
        return result

    def close(self):
        if not self.closed:
            self.closed = True
            if self.worker:
                self.worker.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

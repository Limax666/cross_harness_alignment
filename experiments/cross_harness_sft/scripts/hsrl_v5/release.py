"""Read-only admission status and guarded, private-local mixed release export.

No model, source tool, checker, container or network is invoked by this module.
A completed CPU admission is a prerequisite, never inferred from fast cases.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict, deque
from contextvars import ContextVar
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

from .io import atomic_json, canonical, digest, file_hash
from .reward import VERSION, components
from .dataset import SCHEMA, assert_preserved_semantics

SEED = 20261116
SOURCE_PINS = {
    'clawgym': '1cc77e86b9cd144864180b972498abdd0d1a3ab6de9ec364717934839717e4ac',
    'agentdog': '6fea44557eb505fdbba646a6f75abbf61003033729b851b87b03225e3f85ee91',
}
GRADER_SHA256 = '0f08a890ab7aaadc88f526ab997fd438c07c6dc10c5e30f4b73c8549af1f56e2'
BOUNDARY = 'readonly_private_grader_no_actor_code_v2'
QUOTAS = {'clawgym:benign': 7000, 'agentdog:attack': 5500,
          'agentdog:clean': 1000, 'agentdog:query_target': 500}
LIMITS = {'context_tokens': 16384, 'call_tokens': 2048, 'episode_tokens': 8192, 'max_rounds': 12}

# Independent minimum closure. A manifest cannot opt out of checking a runtime
# module merely by removing its hash. Extra copied Python modules are checked too.
REQUIRED_RUNTIME_MODULES = frozenset(
    ['hsrl_v5/' + name + '.py' for name in (
        '__init__', 'dataset', 'release', 'reward', 'sampler', 'io',
        'agentdog_backend', 'agentdog_source', 'agentdog_worker', 'quality_agentdog',
        'claw_backend', 'claw_source', 'claw_contracts', 'workspace',
        'workspace_worker', 'grader_pool', 'isolation')]
    + ['hsrl_v3/' + name + '.py' for name in ('__init__', 'backends', 'claw_backend', 'pool')])
GRADER_PATH = 'vendor/agentdog/runtime_evaluators.py'
SAMPLER_CONTRACT = {'G': 8, 'K': 2, 'groups': 16, 'general_groups': 8,
                    'safety_groups': 8, 'blocks_per_pass': 875}
COMPATIBILITY = {'algorithms': ['GRPO', 'CHS-PO'],
                 'stock_final_answer_only_reward': False, 'old_hsrl_v3_actor': False}


def core_descriptor(manifest):
    """Cycle-free identity for bytes AND the metadata selecting their meaning."""
    fields = ('schema', 'reward_contract', 'source_manifest_sha256', 'selection_sha256',
              'selection_seed', 'source_role_counts', 'runtime_limits',
              'runtime_module_sha256', 'agentdog_grader_path', 'agentdog_grader_sha256',
              'sampler_contract', 'compatibility', 'admission_report')
    try:
        core = {name: manifest[name] for name in fields}
        core.update(file_sha256=core_file_map(manifest['file_sha256']),
                    required_runtime_modules=sorted(REQUIRED_RUNTIME_MODULES),
                    descriptor_schema='hsrl-v6-release-semantic-core/v1')
        return core
    except KeyError as exc:
        raise ValueError('missing_release_semantic_core_field:' + str(exc)) from exc


def validate_core_identity(manifest, rows=None, root=None):
    """Independent pins/closure plus recomputed selection; certificates add trust."""
    runtime, files = manifest.get('runtime_module_sha256', {}), manifest.get('file_sha256', {})
    packaged_modules = {name[5:] for name in files if name.startswith('code/') and name.endswith('.py')}
    if (not isinstance(runtime, dict) or not REQUIRED_RUNTIME_MODULES.issubset(runtime)
            or set(runtime) != packaged_modules):
        raise ValueError('release_runtime_module_coverage_incomplete')
    for name, expected in runtime.items():
        if Path(name).is_absolute() or '..' in Path(name).parts or files.get('code/' + name) != expected:
            raise ValueError('release_runtime_module_file_identity_mismatch')
    if (manifest.get('agentdog_grader_path') != GRADER_PATH
            or manifest.get('agentdog_grader_sha256') != GRADER_SHA256
            or files.get(GRADER_PATH) != GRADER_SHA256):
        raise ValueError('release_private_grader_pin_or_selection_changed')
    if root is not None and file_hash(contained(root, Path(root) / GRADER_PATH)) != GRADER_SHA256:
        raise ValueError('release_private_grader_bytes_changed')
    if (manifest.get('schema') != SCHEMA or manifest.get('reward_contract') != VERSION
            or manifest.get('source_manifest_sha256') != SOURCE_PINS
            or manifest.get('selection_seed') != SEED or manifest.get('source_role_counts') != QUOTAS
            or manifest.get('runtime_limits') != LIMITS or manifest.get('sampler_contract') != SAMPLER_CONTRACT
            or manifest.get('compatibility') != COMPATIBILITY):
        raise ValueError('release_frozen_consumer_contract_changed')
    if rows is not None:
        identities = sorted([row['source'], row['id']] for row in rows)
        if len({tuple(key) for key in identities}) != len(identities) or digest(identities) != manifest.get('selection_sha256'):
            raise ValueError('release_loaded_selection_identity_mismatch')
    if digest(core_file_map(files)) != manifest.get('artifact_file_sha256_sha256'):
        raise ValueError('release_core_artifact_identity_changed')
    if digest(core_descriptor(manifest)) != manifest.get('semantic_core_sha256'):
        raise ValueError('release_semantic_core_identity_changed')
    return manifest['semantic_core_sha256']


def read_json(path):
    return json.loads(Path(path).read_text())


def jsonl(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def contained(root, path):
    root, path = Path(root).resolve(strict=True), Path(path).absolute()
    if not path.is_relative_to(root) or path.is_symlink() or not path.resolve(strict=True).is_relative_to(root):
        raise ValueError('quality_evidence_path_escape')
    current = path
    while current != root:
        if current.is_symlink():
            raise ValueError('quality_evidence_symlink')
        current = current.parent
    return path


_EVIDENCE_DIRECTORY = ContextVar('hsrl_release_evidence_directory', default=None)


class EvidenceDirectory:
    """Read metadata through one validated attempt fd, without following links.

    Anchoring openat calls to this fd avoids repeatedly resolving the complete
    host path for every case file and prevents symlink swaps beneath the anchor.
    """
    def __init__(self, path):
        self.path = Path(path)

    def __enter__(self):
        self.fd = os.open(self.path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        self.token = _EVIDENCE_DIRECTORY.set(self)
        return self

    def __exit__(self, *args):
        _EVIDENCE_DIRECTORY.reset(self.token)
        os.close(self.fd)

    def read(self, path, lines=False):
        import stat
        relative = Path(path).relative_to(self.path)
        if not relative.parts or '..' in relative.parts:
            raise ValueError('quality_evidence_path_escape')
        directory = os.dup(self.fd)
        try:
            for part in relative.parts[:-1]:
                child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory)
                os.close(directory)
                directory = child
            descriptor = os.open(relative.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory)
            with os.fdopen(descriptor, 'r') as stream:
                if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                    raise ValueError('quality_evidence_requires_regular_metadata_file')
                return [json.loads(line) for line in stream if line.strip()] if lines else json.load(stream)
        finally:
            os.close(directory)


def evidence_json(directory, name):
    scope = _EVIDENCE_DIRECTORY.get()
    if scope is not None:
        return scope.read(Path(directory) / name)
    return read_json(contained(directory, Path(directory) / name))


def evidence_lines(directory, name):
    scope = _EVIDENCE_DIRECTORY.get()
    if scope is not None:
        return scope.read(Path(directory) / name, lines=True)
    return jsonl(contained(directory, Path(directory) / name))


def private_grade(directory, name):
    grade = evidence_json(directory, name)
    if (grade.get('ok') is not True or grade.get('grader_sha256') != GRADER_SHA256
            or type(grade.get('source_utility')) is not bool
            or type(grade.get('registered_call_goal')) is not bool):
        raise ValueError('quality_private_grade_contract_or_pin_mismatch')
    return grade


def agentdog_result(record, result, grade, history, initial_hash, final_hash, effective_risk):
    vector = components(result)
    risk = record['condition'] != 'clean'
    utility = None if record['condition'] == 'query_target' else float(grade['source_utility'])
    cost = float(grade['registered_call_goal']) if risk else None
    metrics, evidence = result.get('metrics', {}), result.get('evidence', {})
    if ((not risk and grade['registered_call_goal'] is not False)
            or vector['masks'] != record_masks(record) or result.get('utility') != utility
            or result.get('c_source') != cost or result.get('c_protocol') != 0.0
            or metrics.get('profile') != 'hsrl-v5-agentdog-registered-call-goal-v1'
            or type(metrics.get('original_source_utility')) is not bool
            or metrics['original_source_utility'] != grade['source_utility']
            or metrics.get('registered_call_goal') != (grade['registered_call_goal'] if risk else None)
            or metrics.get('source_effective_risk') is not effective_risk
            or metrics.get('source_argument_errors') != 0 or metrics.get('canonical_actions') != len(history)
            or evidence.get('initial_state_sha256') != initial_hash
            or evidence.get('final_state_sha256') != final_hash or evidence.get('history_sha256') != digest(history)):
        raise ValueError('quality_private_grade_source_result_inconsistent')


def validate_agentdog_evidence(record, proof, attempt):
    from .agentdog_source import public_contract
    from .agentdog_worker import materialize_scenario
    from .quality_agentdog import source_actions, risk_calls
    source_bytes = Path(record['source_path']).read_bytes()
    if hashlib.sha256(source_bytes).hexdigest() != record['source_sha256']:
        raise ValueError('quality_original_source_asset_changed')
    bundle = json.loads(source_bytes)
    if digest(bundle) != record['source_record_canonical_sha256']:
        raise ValueError('quality_original_source_bundle_identity_changed')
    scenario = materialize_scenario(bundle, record['scenario'])
    initial_hash = digest(scenario['initial_state_template'])
    reset = proof.get('reset_replay', {})
    if (reset.get('identical_actual_transition_and_public_results') is not True
            or not reset.get('first_source_process_id') or not reset.get('reset_source_process_id')
            or reset['first_source_process_id'] == reset['reset_source_process_id']
            or reset.get('initial_state_sha256') != initial_hash):
        raise ValueError('quality_reset_evidence_missing')
    first, replay = attempt / 'first_episode', attempt / 'reset_episode'
    histories, visible = {}, {}
    for directory, process_key in ((first, 'first_source_process_id'), (replay, 'reset_source_process_id')):
        provenance, isolation = evidence_json(directory, 'provenance.json'), evidence_json(directory, 'isolation.json')
        histories[directory.name] = evidence_lines(directory, 'canonical_actions.jsonl')
        visible[directory.name] = evidence_lines(directory, 'public_results.jsonl')
        public = evidence_json(directory, 'public_interface.json')
        if (provenance.get('grader_sha256') != GRADER_SHA256 or provenance.get('source_sha256') != record['source_sha256']
                or provenance.get('source_id') != record['id'] or provenance.get('scenario') != record['scenario']
                or provenance.get('initial_state_sha256') != initial_hash
                or public != public_contract(bundle, record['scenario']) or provenance.get('public_interface_sha256') != digest(public)):
            raise ValueError('quality_grader_or_source_identity_mismatch')
        if (isolation.get('private_grader_mounted_in_tool_worker') is not False
                or isolation.get('supervised_new_source_os_process') is not True
                or isolation.get('new_state_and_tool_namespaces_per_episode') is not True
                or isolation.get('network') != 'none' or isolation.get('rootfs') != 'readonly'
                or isolation.get('source_process_id') != reset[process_key]):
            raise ValueError('quality_reset_isolation_evidence_missing')
        actions = histories[directory.name]
        expected_public = [{'observation': row['observation'], 'result': row['tool_result']} for row in actions]
        if visible[directory.name] != expected_public:
            raise ValueError('quality_public_result_history_mismatch')
        previous = initial_hash
        for action in actions:
            if action.get('state_before_sha256') != previous:
                raise ValueError('quality_actual_state_transition_chain_mismatch')
            previous = action.get('state_after_sha256')
    actions, replay_actions = histories['first_episode'], histories['reset_episode']
    controls = evidence_json(attempt, 'private_control_provenance.json')
    source_controls = [{'name': row['name'], 'arguments': row['arguments']} for row in controls]
    if (controls != source_actions(bundle) or not actions
            or [{'name': row['tool_name'], 'arguments': row['arguments']} for row in actions] != source_controls
            or replay_actions[:len(actions)] != actions or visible['reset_episode'][:len(actions)] != visible['first_episode']
            or digest(actions) != reset.get('canonical_history_sha256')):
        raise ValueError('quality_actual_replay_or_source_control_mismatch')
    initial = private_grade(attempt, 'initial_grade.json')
    if (initial != private_grade(first, 'private_grade_0.json') or initial['registered_call_goal']
            or (record['condition'] != 'query_target' and initial['source_utility'])):
        raise ValueError('quality_initial_negative_control_missing_or_inconsistent')
    if {p.name for p in first.glob('private_grade_*.json')} != {'private_grade_0.json', 'private_grade_1.json'}:
        raise ValueError('quality_first_private_grade_sequence_missing')
    positive = evidence_json(first, 'result.json')
    final_hash = digest(evidence_json(first, 'private_final_state.json'))
    if (positive != proof.get('positive') or final_hash != reset.get('legal_final_state_sha256')
            or final_hash != actions[-1].get('state_after_sha256')):
        raise ValueError('quality_actual_replay_or_state_evidence_mismatch')
    grade = private_grade(first, 'private_grade_1.json')
    agentdog_result(record, positive, grade, actions, initial_hash, final_hash, bool(scenario.get('risk_enabled')))
    if record['condition'] != 'query_target' and positive['utility'] != 1.0:
        raise ValueError('quality_legal_positive_control_missing')
    if record['condition'] == 'attack' and positive['c_source'] != 0.0:
        raise ValueError('quality_legal_control_unsafe')
    if record['condition'] == 'clean':
        # The genuine clean runner replays legal calls and closes without finish;
        # it records no reset grade/result. Require its actual replay evidence.
        if (replay_actions != actions or visible['reset_episode'] != visible['first_episode']
                or proof.get('risk_control') != {'registered_call_goal': None, 'applicable': False}
                or list(replay.glob('private_grade_*.json')) or (replay / 'result.json').exists()):
            raise ValueError('quality_clean_reset_recording_contract_mismatch')
    else:
        if proof.get('risk_control', {}).get('registered_call_goal') is not True:
            raise ValueError('quality_registered_risk_control_missing')
        forbidden = evidence_json(attempt, 'private_risk_control_provenance.json')
        if (forbidden != risk_calls(bundle, controls) or not forbidden
                or [{'name': row['tool_name'], 'arguments': row['arguments']} for row in replay_actions[len(actions):]] != forbidden
                or {p.name for p in replay.glob('private_grade_*.json')} != {'private_grade_0.json'}):
            raise ValueError('quality_registered_risk_control_history_or_grade_missing')
        risky_grade = private_grade(replay, 'private_grade_0.json')
        risky = evidence_json(replay, 'result.json')
        risky_final = digest(evidence_json(replay, 'private_final_state.json'))
        agentdog_result(record, risky, risky_grade, replay_actions, initial_hash, risky_final, bool(scenario.get('risk_enabled')))
        if (risky_grade['registered_call_goal'] is not True or risky['c_source'] != 1.0
                or risky_final != replay_actions[-1].get('state_after_sha256')
                or proof['risk_control'].get('source_argument_errors') != 0
                or proof['risk_control'].get('actual_harm_claimed') is not False):
            raise ValueError('quality_actual_risk_positive_grade_missing_or_inconsistent')
    total = len(actions) + len(replay_actions)
    if proof.get('source_calls_attempted') != total or proof.get('source_calls_completed') != total:
        raise ValueError('quality_source_calls_evidence_missing')


def workspace_isolation(directory, private_grader):
    isolation = evidence_json(directory, 'isolation.json')
    cleanup = evidence_json(directory, 'cleanup.json')
    mounts = isolation.get('mount_targets', [])
    if (isolation.get('private_grader') is not private_grader or isolation.get('rootless') is not True
            or isolation.get('network') != 'none' or isolation.get('rootfs') != 'readonly'
            or isolation.get('all_host_mounts') != 'readonly'
            or ('/grader' in mounts) is not private_grader
            or cleanup.get('removal_confirmed') is not True
            or not isolation.get('container') or cleanup.get('container') != isolation['container']):
        raise ValueError('quality_workspace_or_grader_isolation_cleanup_missing')
    return isolation


def validate_pool_episode(record, directory, result, response, final_entries):
    from .workspace import snapshot_entries
    from .grader_pool import validate_response_identity
    grader = directory / 'private_grader'
    request, projection = evidence_json(grader, 'request.json'), evidence_json(grader, 'projection.json')
    retained = evidence_json(grader, 'response.json')
    isolation = workspace_isolation(grader, True)
    validate_response_identity(retained, request.get('request_nonce'))
    original = snapshot_entries(Path(record['private_checker_path']).parent)
    source_copy = snapshot_entries(contained(grader, grader / 'source_snapshot'))
    if (request.get('op') != 'grade' or not request.get('request_nonce') or retained != response
            or result.get('evidence', {}).get('private_grader') != projection
            or projection.get('schema') != 'private-data-grader-pool/v1'
            or projection.get('request_nonce') != request['request_nonce']
            or projection.get('child_pid') != retained['result']['child_pid']
            or projection.get('case_id') != digest(record) or projection.get('container') != isolation['container']
            or projection.get('snapshot_manifest') != final_entries
            or projection.get('snapshot_manifest_sha256') != digest(final_entries)
            or projection.get('source_manifest') != original or source_copy != original
            or projection.get('source_manifest_sha256') != digest(original)
            or projection.get('boundary', {}).get('checker_sha256') != record['original_asset_sha256']['reward/test.py']
            or projection.get('boundary', {}).get('version') != BOUNDARY
            or projection.get('boundary', {}).get('status') != 'readonly_data_checker_static_boundary_pass'
            or any(projection.get(field) is not True for field in ('all_projection_mounts_kernel_readonly',
                'fresh_checker_child_each_request', 'tmp_cleared_before_grade',
                'original_and_projection_manifests_verified_after_grade'))):
        raise ValueError('quality_private_grader_pool_projection_identity_mismatch')
    return {name: projection[name] for name in ('container', 'request_nonce', 'child_pid', 'snapshot_manifest_sha256', 'source_manifest_sha256')}


def validate_claw_evidence(record, proof, attempt, require_pool=False):
    from .claw_backend import parse_checker_response
    from .workspace import snapshot_entries
    original_entries = snapshot_entries(record['public_assets_path'])
    initial_hash = digest(original_entries)
    if {name: value['sha256'] for name, value in original_entries.items()} != record['public_asset_sha256']:
        raise ValueError('quality_original_public_assets_changed')
    reset = proof.get('reset', {})
    if (reset.get('fresh_os_process_and_tmpfs') is not True or reset.get('same_checker_scores') is not True
            or reset.get('initial_assets_sha256') != initial_hash):
        raise ValueError('quality_reset_evidence_missing')
    actor_ids, results, identities = [], [], []
    pooled = require_pool or 'private_grader_pool' in proof
    for name in ('actual_episode', 'reset_episode'):
        directory = attempt / name
        boundary = evidence_json(directory, 'private_grader_boundary.json')
        provenance = evidence_json(directory, 'provenance.json')
        result, response = evidence_json(directory, 'result.json'), evidence_json(directory, 'private_grader_response.json')
        raw, score = parse_checker_response(response)
        vector = components(result)
        final_entries = snapshot_entries(contained(directory, directory / 'private_output_snapshot'))
        if (boundary.get('version') != BOUNDARY or boundary.get('status') != 'readonly_data_checker_static_boundary_pass'
                or boundary.get('checker_sha256') != record['original_asset_sha256']['reward/test.py']
                or provenance.get('source_tree_sha256') != record['source_tree_sha256']
                or provenance.get('original_checker_sha256') != record['original_asset_sha256']['reward/test.py']
                or provenance.get('private_oracle_mounted_in_actor') is not False):
            raise ValueError('quality_private_boundary_or_source_identity_mismatch')
        expected_public = {'messages': [{'role': 'user', 'content': read_json(record['source_path'])['user_query']}]}
        from .workspace import SCHEMAS
        expected_public['tools'] = SCHEMAS
        if evidence_json(directory, 'public_interface.json') != expected_public:
            raise ValueError('quality_public_interface_source_mismatch')
        if (vector['masks'] != record_masks(record) or result['utility'] != score or result['c_protocol'] != 0.0
                or result.get('metrics', {}).get('official_checker_scores') != raw
                or result['metrics'].get('profile') != record['reward_profile']
                or result.get('evidence', {}).get('initial_assets_sha256') != initial_hash
                or result['evidence'].get('final_assets_sha256') != digest(final_entries)):
            raise ValueError('quality_raw_checker_result_inconsistent')
        if any(final_entries.get(path, {}).get('sha256') != record['source_contract']['initial_input_hashes'][path]
               for path in record['source_contract']['immutable_inputs']):
            raise ValueError('quality_immutable_grading_reference_modified')
        actor_ids.append(workspace_isolation(directory / 'tool_worker', False)['container'])
        if pooled or (directory / 'private_grader/projection.json').exists():
            pooled = True
            identities.append(validate_pool_episode(record, directory, result, response, final_entries))
        else:
            workspace_isolation(directory / 'private_grader', True)
            if result.get('evidence', {}).get('private_grader') is not None:
                raise ValueError('quality_private_grader_pool_evidence_missing')
        if name == 'reset_episode' and final_entries != original_entries:
            raise ValueError('quality_reset_source_assets_mismatch')
        results.append(result)
    if (actor_ids[0] == actor_ids[1] or results[0]['utility'] != results[1]['utility']
            or results[0]['metrics'] != results[1]['metrics']
            or proof.get('untouched_workspace_grade') != results[0]['utility']
            or results[0]['utility'] == 1.0 or proof.get('verifier_profile') != record['reward_profile']):
        raise ValueError('quality_actual_reset_checker_or_untouched_grade_inconsistent')
    controls = evidence_lines(attempt / 'actual_episode', 'canonical_actions.jsonl')
    if len(controls) != 1 or controls[0].get('name') != 'run_python' or controls[0].get('result', {}).get('returncode') != 0:
        raise ValueError('quality_actual_source_read_control_missing')
    observed = json.loads(controls[0]['result']['stdout'])
    if observed.get('private_grader_visible') is not False:
        raise ValueError('quality_private_oracle_visibility_control_failed')
    if original_entries and observed.get('sha256') != original_entries[sorted(original_entries)[0]]['sha256']:
        raise ValueError('quality_actual_source_read_hash_mismatch')
    if pooled:
        expected = proof.get('private_grader_pool', {})
        if (len(identities) != 2 or expected.get('same_private_container') is not True or expected.get('fresh_children') is not True
                or expected.get('requests') != identities or identities[0]['container'] != identities[1]['container']
                or identities[0]['request_nonce'] == identities[1]['request_nonce']
                or identities[0]['child_pid'] == identities[1]['child_pid']):
            raise ValueError('quality_private_grader_pool_fresh_child_or_cleanup_evidence_missing')


def validate_proof(record, proof, path, quality_root):
    """Bind each claim to retained source, private grades, reset and result bytes."""
    if any(proof.get(k) != record.get(k) for k in ('source', 'id', 'family', 'condition')) or proof.get('record_sha256') != digest(record):
        raise ValueError('quality_record_identity_mismatch')
    if proof.get('ready') is not True or proof.get('source_execution_performed') is not True or proof.get('reasons'):
        raise ValueError('quality_not_runtime_admitted')
    case = contained(quality_root, Path(path).parent)
    attempt = contained(case, proof.get('attempt_directory', '/missing'))
    with EvidenceDirectory(attempt):
        if record['source'] == 'agentdog':
            validate_agentdog_evidence(record, proof, attempt)
        elif record['source'] == 'clawgym':
            protocol_path = Path(quality_root) / 'protocol.json'
            require_pool = protocol_path.exists() and read_json(protocol_path).get('schema') == 'hsrl-v5-claw-real-quality/v5'
            validate_claw_evidence(record, proof, attempt, require_pool)
        else:
            raise ValueError('unsupported_release_source')
    return attempt


def excluded_benchmark_families(source):
    """Use cached public benchmark hashes only, excluding matching whole families."""
    hashes = set(read_json(Path(source) / 'benchmark_public_hashes.json')['normalized_prompt_sha256'])
    return {r['family'] for r in jsonl(Path(source) / 'inventory.jsonl') if r['normalized_prompt_sha256'] in hashes}


def inspect_source(source, source_dir, quality_dir):
    source_dir, quality_dir = Path(source_dir).resolve(strict=True), Path(quality_dir).resolve()
    manifest_sha = file_hash(source_dir / 'source_manifest.json')
    if manifest_sha != SOURCE_PINS[source]:
        raise ValueError('unapproved_source_manifest:' + source)
    manifest = read_json(source_dir / 'source_manifest.json')
    for name, expected in manifest['file_sha256'].items():
        if Path(name).is_absolute() or '..' in Path(name).parts or file_hash(source_dir / name) != expected:
            raise ValueError('source_manifest_file_changed:' + name)
    records = [r for r in jsonl(source_dir / 'accepted.jsonl') if r['split'] == 'train']
    if len({r['id'] for r in records}) != len(records):
        raise ValueError('duplicate_source_record_identity')
    protocol_path = quality_dir / 'protocol.json'
    state = {'source_manifest_sha256': manifest_sha, 'planned': len(records), 'completed': 0,
             'complete': False, 'admitted': 0, 'pending_unprocessed': len(records),
             'rejected_or_pending': {}, 'identity_errors': [], 'pending_retained_evidence': [], 'quality_dir': str(quality_dir)}
    if not protocol_path.is_file():
        state['gate'] = 'missing_quality_protocol'
        return state, []
    protocol = read_json(protocol_path)
    schemas = {'hsrl-v5-claw-real-quality/v3', 'hsrl-v5-claw-real-quality/v4', 'hsrl-v5-claw-real-quality/v5'} if source == 'clawgym' else {'hsrl-v5-agentdog-real-quality/v2'}
    if protocol.get('source_manifest_sha256') != manifest_sha or protocol.get('schema') not in schemas or protocol.get('records') != len(records):
        raise ValueError('quality_protocol_source_identity_mismatch:' + source)
    if source == 'clawgym' and protocol.get('private_grader_boundary_gate') != BOUNDARY:
        raise ValueError('quality_protocol_outdated_private_boundary')
    if source == 'agentdog' and protocol.get('source_reset') != 'new_os_process_per_episode':
        raise ValueError('quality_protocol_missing_process_reset')
    # Bind QC to its retained frozen code. Consumer dataset/reward code may evolve;
    # admission evaluator/tool bytes must match the code actually executed by QC.
    snapshot = quality_dir / 'source_snapshot' / 'hsrl_v5'
    if not snapshot.exists():
        snapshot = quality_dir / 'source_snapshot'
    state['protocol_schema'] = protocol['schema']
    state['code_snapshot_verified'] = bool(protocol.get('code_sha256'))
    for name, expected in protocol.get('code_sha256', {}).items():
        file = snapshot / name
        if not file.is_file() or file_hash(file) != expected:
            state['code_snapshot_verified'] = False
    rejected, admitted = Counter(), []
    for record in records:
        path = quality_dir / 'cases' / hashlib.sha256(record['id'].encode()).hexdigest() / 'quality.json'
        if not path.is_file():
            continue
        proof = read_json(path)
        state['completed'] += 1
        if any(proof.get(k) != record.get(k) for k in ('source', 'id', 'family', 'condition')) or proof.get('record_sha256') != digest(record):
            state['identity_errors'].append(record['id'])
            continue
        if proof.get('ready') is True:
            try:
                validate_proof(record, proof, path, quality_dir)
                admitted.append((record, path))
            except (ValueError, KeyError, OSError, TypeError) as exc:
                if (source == 'clawgym' and protocol['schema'] == 'hsrl-v5-claw-real-quality/v5'
                        and isinstance(exc, FileNotFoundError) and Path(str(exc.filename)).name == 'cleanup.json'):
                    state['pending_retained_evidence'].append(record['id'] + ':private_grader_removal_confirmation_pending')
                else:
                    state['identity_errors'].append(record['id'] + ':' + str(exc))
        else:
            rejected.update([proof.get('failure_class', 'source_quality_pending')])
    state.update(admitted=len(admitted), admitted_by_condition=dict(Counter(r['condition'] for r, _ in admitted)),
                 pending_unprocessed=len(records) - state['completed'], rejected_or_pending=dict(rejected))
    summary_path, run_path = quality_dir / 'summary.json', quality_dir / 'run_state.json'
    if summary_path.is_file() and run_path.is_file():
        summary, run = read_json(summary_path), read_json(run_path)
        allowed = {'complete_runtime_validation_pass', 'complete_with_pending_cases'} if source == 'clawgym' else {'complete_quality_pass'}
        state['terminal_status'] = summary.get('status')
        state['complete'] = (run.get('status') == 'complete' and summary.get('status') in allowed
            and summary.get('protocol_sha256') == file_hash(protocol_path)
            and summary.get('planned') == summary.get('completed') == len(records) == state['completed']
            and not state['identity_errors'] and not state['pending_retained_evidence'] and state['code_snapshot_verified']
            and not rejected.get('isolation_or_runtime_infrastructure', 0))
        if source == 'clawgym' and protocol['schema'] == 'hsrl-v5-claw-real-quality/v5':
            try:
                cleanup = read_json(contained(quality_dir, summary['private_grader_pool_cleanup']))
                state['private_grader_pool_cleanup_confirmed'] = (
                    summary.get('private_grader_pool_cleanup_confirmed') is True
                    and cleanup.get('all_owned_containers_removal_confirmed') is True
                    and cleanup.get('errors') == [])
            except (KeyError, OSError, ValueError, TypeError):
                state['private_grader_pool_cleanup_confirmed'] = False
            state['complete'] = state['complete'] and state['private_grader_pool_cleanup_confirmed']
    try:
        current_root = Path(__file__).resolve().parent
        current = {'hsrl_v5/' + p.name: file_hash(p) for p in current_root.glob('*.py')}
        verify_execution_code(source, protocol, current)
        state['consumer_execution_code_matches_QC'] = True
    except ValueError as exc:
        state['consumer_execution_code_matches_QC'] = False
        state['execution_code_error'] = str(exc)
    state['gate'] = 'complete' if state['complete'] else 'full_terminal_identity_bound_quality_required'
    return state, admitted


def family_round_robin(rows, count):
    buckets = defaultdict(list)
    rows = list(rows)
    preferred = [r for r in rows if r.get('selected_candidate', True)]
    refill = [r for r in rows if not r.get('selected_candidate', True)]
    if preferred and refill:
        first = family_round_robin(preferred, count)
        return first + family_round_robin(refill, max(0, count - len(first)))
    for row in rows:
        buckets[row['family']].append(row)
    queues = deque(deque(sorted(buckets[family], key=lambda r: (not r.get('selected_candidate', True), r['id'])))
                   for family in sorted(buckets, key=lambda f: digest([SEED, f])))
    result = []
    while queues and len(result) < count:
        queue = queues.popleft()
        result.append(queue.popleft())
        if queue:
            queues.append(queue)
    return result


def select_agentdog(rows, attack_count=5500, clean_count=1000, query_count=500):
    pools = {role: [r for r in rows if r['condition'] == role] for role in ('attack', 'clean', 'query_target')}
    attacks = family_round_robin(pools['attack'], attack_count)
    families = {r['family'] for r in attacks}
    clean = family_round_robin([r for r in pools['clean'] if r['family'] in families], clean_count)
    query = family_round_robin(pools['query_target'], query_count)
    return attacks + clean + query


def status(claw_source, claw_quality, agentdog_source, agentdog_quality):
    states, admitted = {}, {}
    for name, source, quality in (('clawgym', claw_source, claw_quality), ('agentdog', agentdog_source, agentdog_quality)):
        states[name], admitted[name] = inspect_source(name, source, quality)
    excluded = excluded_benchmark_families(claw_source)
    inventories = jsonl(Path(claw_source) / 'inventory.jsonl') + jsonl(Path(agentdog_source) / 'inventory.jsonl')
    holdouts = {r['normalized_prompt_sha256'] for r in inventories if r.get('split') in ('dev', 'test')}
    eligible = {name: [r for r, _ in rows if r['normalized_prompt_sha256'] not in holdouts and not (name == 'clawgym' and r['family'] in excluded)] for name, rows in admitted.items()}
    chosen = family_round_robin(eligible['clawgym'], 7000) + select_agentdog(eligible['agentdog'])
    counts = Counter(r['source'] + ':' + r['condition'] for r in chosen)
    selection_sha = digest(sorted([r['source'], r['id']] for r in chosen))
    report = {'schema': 'hsrl-v6-mixed-release-status/v1', 'status': 'not_ready', 'ready': False, 'selection_seed': SEED,
        'sources': states, 'required': QUOTAS, 'eligible_selection': dict(counts),
        'shortfall': {role: max(0, needed - counts[role]) for role, needed in QUOTAS.items()},
        'benchmark_excluded_whole_claw_families': sorted(excluded),
        'train_active_holdout_public_hash_collisions': len({r['normalized_prompt_sha256'] for r in chosen} & holdouts),
        'selection_sha256': selection_sha, 'base_families': dict(Counter({s: len({r['family'] for r in chosen if r['source'] == s}) for s in states})),
        'agentdog_split_groups': len({r['split_group'] for r in chosen if r['source'] == 'agentdog'}),
        'pending_gates': ['selected_exported_runtime_privacy_reset_evaluator_audit', 'selected_context_token_audit']}
    if not all(s['complete'] for s in states.values()):
        report['pending_gates'].insert(0, 'full_source_quality_terminal_completion')
    if any(report['shortfall'].values()):
        report['pending_gates'].insert(0, 'real_admitted_role_quotas')
    if any(not s.get('consumer_execution_code_matches_QC') for s in states.values()):
        report['pending_gates'].insert(0, 'consumer_execution_code_must_match_completed_QC')
    if states['clawgym'].get('protocol_schema') != 'hsrl-v5-claw-real-quality/v5':
        report['pending_gates'].insert(0, 'final_new_Claw_v5_pooled_quality_required')
    proofs = {(r['source'], r['id']): path for rows in admitted.values() for r, path in rows}
    return report, chosen, proofs


def require_admission(report):
    if not all(s['complete'] for s in report['sources'].values()):
        raise ValueError('full_source_quality_terminal_completion_required')
    if any(report['shortfall'].values()):
        raise ValueError('insufficient_real_admitted_role_quotas')


def core_file_map(files):
    return {name: sha for name, sha in files.items() if name not in ('audits/context.json', 'audits/runtime.json')}


def validate_certificates(root, manifest):
    semantic = validate_core_identity(manifest, root=root)
    expected = manifest['artifact_file_sha256_sha256']
    if digest(core_file_map(manifest['file_sha256'])) != expected:
        raise ValueError('release_core_artifact_identity_changed')
    for kind in ('context', 'runtime'):
        name = 'audits/' + kind + '.json'
        path = Path(root) / name
        if manifest['file_sha256'].get(name) != file_hash(path):
            raise ValueError('missing_or_changed_release_audit:' + kind)
        audit = read_json(path)
        if (audit.get('passed') is not True or audit.get('selection_sha256') != manifest['selection_sha256']
                or audit.get('source_manifest_sha256') != SOURCE_PINS
                or audit.get('artifact_file_sha256_sha256') != expected
                or audit.get('semantic_core_sha256') != semantic
                or audit.get('records') != 14000):
            raise ValueError('release_audit_identity_or_result_mismatch:' + kind)
        if kind == 'context':
            if audit.get('limits') != LIMITS or audit.get('initial_configurations') != 18:
                raise ValueError('context_audit_limits_or_OA18_coverage_missing')
        else:
            checks = ('exported_loader', 'public_render', 'environment_steps', 'scoring', 'fresh_reset', 'private_oracle_separation', 'sampler_full_875_blocks')
            if any(audit.get('checks', {}).get(check) is not True for check in checks):
                raise ValueError('runtime_audit_mandatory_checks_missing')


def copy_tree(source, target):
    source = Path(source)
    for path in source.rglob('*'):
        if path.is_symlink():
            raise ValueError('source_symlink_rejected')
    shutil.copytree(source, target)


def record_masks(record):
    if record['source'] not in ('clawgym', 'agentdog'):
        raise ValueError('unsupported_release_source')
    return {'utility': record['source'] == 'clawgym' or record['condition'] != 'query_target',
            'source_cost': record['source'] == 'agentdog' and record['condition'] != 'clean',
            'protocol_cost': True, 'persistence_cost': False,
            'broad_safety_cost': False, 'actual_harm_cost': False}


def verify_execution_code(source, protocol, runtime):
    critical = (('agentdog_backend.py', 'agentdog_source.py', 'agentdog_worker.py', 'quality_agentdog.py', 'isolation.py', 'io.py')
                if source == 'agentdog' else
                ('claw_backend.py', 'claw_source.py', 'claw_contracts.py', 'workspace.py', 'workspace_worker.py', 'grader_pool.py', 'isolation.py', 'io.py'))
    for name in critical:
        expected = protocol.get('code_sha256', {}).get(name)
        if not expected or runtime.get('hsrl_v5/' + name) != expected:
            raise ValueError('source_execution_code_changed_since_QC:' + source + ':' + name)


def export_candidate(report, selected, proofs, sources, qualities, out, grader):
    """Construct an unpublished candidate only after both complete admissions."""
    require_admission(report)
    if report['sources']['clawgym'].get('protocol_schema') != 'hsrl-v5-claw-real-quality/v5':
        raise ValueError('final_new_Claw_v5_pooled_quality_required')
    out = Path(out).absolute()
    if out.exists():
        raise ValueError('export_destination_exists')
    if file_hash(grader) != GRADER_SHA256:
        raise ValueError('private_grader_pin_changed')
    # Import pure hashing/public rendering helpers only; no source execution.
    from hsrl_v3.backends import tree_digest
    from .agentdog_source import public_contract
    from .workspace import SCHEMAS
    import pyarrow as pa
    import pyarrow.parquet as pq
    out.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix='.' + out.name + '.staging-', dir=out.parent))
    os.chmod(stage, 0o700)
    try:
        rows, parquet_rows = [], []
        for original in selected:
            source, identity = original['source'], original['id']
            key = hashlib.sha256((source + '\0' + identity).encode()).hexdigest()
            active = copy.deepcopy(original)
            asset = Path('assets') / source / key
            if source == 'clawgym':
                source_tree = Path(original['source_path']).parent
                if tree_digest(source_tree) != original['source_tree_sha256']:
                    raise ValueError('original_claw_tree_changed:' + identity)
                (stage / asset).parent.mkdir(parents=True, exist_ok=True)
                copy_tree(source_tree, stage / asset)
                if tree_digest(stage / asset) != original['source_tree_sha256']:
                    raise ValueError('relocated_claw_tree_changed:' + identity)
                for field in ('source_path', 'public_assets_path', 'private_grader_path', 'private_checker_path'):
                    active[field] = (asset / Path(original[field]).relative_to(source_tree)).as_posix()
                active['source_identity']['raw_record_path'] = (asset / Path(original['source_identity']['raw_record_path']).relative_to(source_tree)).as_posix()
                public = {'messages': [{'role': 'user', 'content': read_json(stage / active['source_path'])['user_query']}], 'tools': copy.deepcopy(SCHEMAS)}
            else:
                (stage / asset).mkdir(parents=True)
                if file_hash(original['source_path']) != original['source_sha256']:
                    raise ValueError('original_agentdog_bundle_changed:' + identity)
                active['source_path'] = (asset / 'bundle.json').as_posix()
                shutil.copyfile(original['source_path'], stage / active['source_path'])
                public = public_contract(read_json(stage / active['source_path']), original['scenario'])
            active['runtime_ready'] = True
            assert_preserved_semantics(original, active)
            metadata = Path('private') / source / key
            atomic_json(stage / metadata / 'original_record.json', original)
            atomic_json(stage / metadata / 'record.json', active)
            proof_path = proofs[(source, identity)]
            validate_proof(original, read_json(proof_path), proof_path, qualities[source])
            copy_tree(proof_path.parent, stage / metadata / 'quality')
            row = {name: active[name] for name in ('source', 'id', 'family', 'condition', 'split', 'runtime_ready')}
            row.update(record_path=(metadata / 'record.json').as_posix(), original_record_path=(metadata / 'original_record.json').as_posix(),
                quality_proof_path=(metadata / 'quality/quality.json').as_posix(), normalized_prompt_sha256=active['normalized_prompt_sha256'],
                split_group=active.get('split_group'), source_profile=active.get('reward_profile', 'hsrl-v5-agentdog-registered-call-goal-v1'))
            # Same vector masks for both algorithm views, never zero-imputed labels.
            masks = record_masks(active)
            row.update(masks=masks, reward_contract=VERSION, quality_status='real_source_runtime_admitted',
                       compatibility='algorithm_neutral_tool_environment_required')
            rows.append(row)
            # Actor sees only public interface and opaque lookup. Collector metadata
            # is explicitly private; no hidden state/labels/plan is serialized here.
            parquet_rows.append({'data_source': source, 'prompt': public['messages'], 'tools': canonical(public['tools']),
                'reward_model': {'style': 'tool_environment', 'ground_truth': canonical([source, identity])},
                'extra_info': {'source': source, 'id': identity, 'family': active['family'], 'condition': active['condition'],
                    'lookup': row['record_path'], 'masks': masks, 'GRPO_U_policy_gradient': masks['utility']}})
        (stage / 'train.jsonl').write_text(''.join(canonical(row) + '\n' for row in rows))
        pq.write_table(pa.Table.from_pylist(parquet_rows), stage / 'train.parquet')
        for source, directory in sources.items():
            for path in sorted(Path(directory).iterdir()):
                if path.is_file() and path.suffix in ('.json', '.jsonl'):
                    target = stage / 'provenance' / source / path.name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(path, target)
            for name in ('protocol.json', 'summary.json', 'run_state.json'):
                shutil.copyfile(Path(qualities[source]) / name, stage / 'provenance' / source / ('quality_' + name))
            copy_tree(Path(qualities[source]) / 'source_snapshot', stage / 'provenance' / source / 'quality_source_snapshot')
            pools = Path(qualities[source]) / 'private_grader_pools'
            if source == 'clawgym':
                copy_tree(pools, stage / 'provenance' / source / 'private_grader_pools')
            control = Path(qualities[source]) / 'actual_transport_failure_control'
            if source == 'agentdog':
                contract = read_json(control / 'failure_contract.json')
                if any(contract.get(key) is not True for key in ('passed', 'actual_public_api_tool_call_rejected', 'stale_finish_rejected')):
                    raise ValueError('actual_transport_failure_contract_missing')
                copy_tree(control, stage / 'provenance' / source / 'actual_transport_failure_control')
        vendor = stage / 'vendor/agentdog'
        vendor.mkdir(parents=True)
        shutil.copyfile(grader, vendor / 'runtime_evaluators.py')
        shutil.copyfile(Path(grader).parent / 'SOURCE_MANIFEST.json', vendor / 'SOURCE_MANIFEST.json')
        atomic_json(vendor / 'usage_scope.json', {'data_license': 'Apache-2.0', 'vendored_code_license': 'unknown',
            'scope': 'private_local_research_only_no_publication', 'grader_sha256': GRADER_SHA256})
        code_root = Path(__file__).resolve().parent.parent
        runtime = {}
        for package, names in (('hsrl_v5', [p.name for p in (code_root / 'hsrl_v5').glob('*.py')]),
                               ('hsrl_v3', ['__init__.py', 'backends.py', 'claw_backend.py', 'pool.py'])):
            for name in sorted(names):
                source = code_root / package / name
                target = stage / 'code' / package / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)
                runtime[package + '/' + name] = file_hash(target)
        for source, quality in qualities.items():
            verify_execution_code(source, read_json(Path(quality) / 'protocol.json'), runtime)
        files = {p.relative_to(stage).as_posix(): file_hash(p) for p in sorted(stage.rglob('*')) if p.is_file()}
        manifest = {'schema': SCHEMA, 'status': 'candidate', 'reward_contract': VERSION,
            'source_manifest_sha256': SOURCE_PINS, 'selection_sha256': report['selection_sha256'],
            'selection_seed': SEED, 'source_role_counts': QUOTAS, 'runtime_limits': LIMITS,
            'artifact_file_sha256_sha256': digest(files), 'file_sha256': files, 'runtime_module_sha256': runtime,
            'agentdog_grader_path': GRADER_PATH, 'agentdog_grader_sha256': GRADER_SHA256, 'admission_report': report,
            'sampler_contract': SAMPLER_CONTRACT,
            'privacy': 'private_local_release_actor_receives_only_public_messages_tools_and_opaque_task_reference',
            'compatibility': COMPATIBILITY}
        manifest['semantic_core_sha256'] = digest(core_descriptor(manifest))
        validate_core_identity(manifest, rows, stage)
        atomic_json(stage / 'manifest.json', manifest)
        # Execute only frozen consumer rendering/sampler code; never source assets.
        command = [sys.executable, '-B', '-m', 'hsrl_v5.release', '--audit-candidate', str(stage)]
        env = dict(os.environ, PYTHONPATH=str(stage / 'code'), PYTHONDONTWRITEBYTECODE='1')
        subprocess.run(command, cwd=stage, env=env, check=True, capture_output=True, text=True)
        os.rename(stage, out)
        return manifest
    except BaseException:
        # Retain failed candidate/evidence for inspection; no READY is written.
        raise


def audit_candidate(root):
    from .dataset import Dataset
    dataset = Dataset.for_verification(root)
    for relative in dataset.files:
        dataset.verify_file(relative)
    visited = []
    for block in range(875):
        first, second = dataset.sampler.step(block * 2), dataset.sampler.step(block * 2 + 1)
        if first['groups'] != second['groups'] or first['configuration'] == second['configuration']:
            raise ValueError('K2_schedule_identity_failed')
        visited.extend((g['source'], g['id']) for g in first['groups'])
    if len(visited) != len(set(visited)) or set(visited) != set(dataset.index):
        raise ValueError('full_875_block_sampler_coverage_failed')
    for key in dataset.index:
        public = dataset.public(key)
        if set(public) != {'messages', 'tools'}:
            raise ValueError('private_fields_in_public_render')
    return {'records': len(visited), 'sampler_full_875_blocks': True, 'public_render': True}


def install_no_replace(source, target):
    """Linux atomic directory install that cannot overwrite even an empty target."""
    import ctypes
    import errno
    libc = ctypes.CDLL(None, use_errno=True)
    rename = getattr(libc, 'renameat2', None)
    if rename is None:
        raise OSError(errno.ENOTSUP, 'atomic_noreplace_directory_install_unavailable')
    rename.argtypes = (ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint)
    rename.restype = ctypes.c_int
    if rename(-100, os.fsencode(source), -100, os.fsencode(target), 1):
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), str(target))


def private_publication_copy(candidate, out, assemble, validate):
    """Keep origin byte-immutable through errors or process/host interruption.

    Only an unguessable 0700 publication directory can contain an intermediate
    READY. No mutation/rollback of the user-known verification candidate occurs.
    Failed private staging is retained; a retry always starts from the candidate.
    """
    candidate, out = Path(candidate).resolve(strict=True), Path(out).absolute()
    if out.exists() or out.is_symlink():
        raise FileExistsError('publish_destination_exists:' + str(out))
    out.parent.mkdir(parents=True, exist_ok=True)
    private = Path(tempfile.mkdtemp(prefix='.' + out.name + '.publication-', dir=out.parent))
    os.chmod(private, 0o700)
    stage = private / 'artifact'
    copy_tree(candidate, stage)
    assemble(stage)
    validate(stage)
    install_no_replace(stage, out)
    private.rmdir()


def publish(candidate, out, context_audit, runtime_audit):
    candidate, out = Path(candidate).resolve(strict=True), Path(out).absolute()
    original = read_json(candidate / 'manifest.json')
    if original.get('status') != 'candidate' or (candidate / 'READY.json').exists():
        raise ValueError('publish_requires_unpublished_candidate')
    require_admission(original['admission_report'])

    def assemble(stage):
        # Audit the independent copy, closing the read/copy interval and leaving
        # the original candidate free of certificates, ready metadata and READY.
        audit_candidate(stage)
        manifest = read_json(stage / 'manifest.json')
        for kind, path in (('context', context_audit), ('runtime', runtime_audit)):
            target = stage / 'audits' / (kind + '.json')
            target.parent.mkdir(exist_ok=True)
            shutil.copyfile(path, target)
            manifest['file_sha256']['audits/' + kind + '.json'] = file_hash(target)
        validate_certificates(stage, manifest)
        manifest['status'] = 'ready'
        atomic_json(stage / 'manifest.json', manifest)
        atomic_json(stage / 'READY.json', {'manifest_sha256': file_hash(stage / 'manifest.json')})

    def validate(stage):
        env = dict(os.environ, PYTHONPATH=str(stage / 'code'), PYTHONDONTWRITEBYTECODE='1')
        subprocess.run([sys.executable, '-B', '-c',
                        'from hsrl_v5.dataset import Dataset; import sys; d=Dataset(sys.argv[1]); assert len(d.index)==14000', str(stage)],
                       cwd=stage, env=env, check=True, capture_output=True, text=True)

    private_publication_copy(candidate, out, assemble, validate)
    return {'status': 'ready', 'root': str(out), 'manifest_sha256': file_hash(out / 'manifest.json')}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('claw-source', 'claw-quality', 'agentdog-source', 'agentdog-quality'):
        parser.add_argument('--' + name, type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--status', action='store_true', help='read-only status; default operation')
    parser.add_argument('--out', type=Path)
    mode.add_argument('--stage', action='store_true', help='build private candidate without READY')
    mode.add_argument('--publish', type=Path, metavar='CANDIDATE')
    parser.add_argument('--context-audit', type=Path)
    parser.add_argument('--runtime-audit', type=Path)
    parser.add_argument('--grader', type=Path)
    mode.add_argument('--audit-candidate', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--wait', type=float, default=0, metavar='SECONDS', help='bounded CPU-file polling, at most 86400 seconds')
    parser.add_argument('--poll-seconds', type=float, default=30)
    args = parser.parse_args(argv)
    if args.audit_candidate:
        print(canonical(audit_candidate(args.audit_candidate)))
        return 0
    if args.publish:
        if not all((args.out, args.context_audit, args.runtime_audit)):
            parser.error('--publish requires --out --context-audit --runtime-audit')
        print(canonical(publish(args.publish, args.out, args.context_audit, args.runtime_audit)))
        return 0
    if not all((args.claw_source, args.claw_quality, args.agentdog_source, args.agentdog_quality)):
        parser.error('all four --*-source/--*-quality paths are required')
    if not 0 <= args.wait <= 86400 or not 0 < args.poll_seconds <= 60:
        parser.error('wait must be 0..86400 seconds and poll-seconds 0..60')
    deadline = time.monotonic() + args.wait
    while True:
        try:
            report, selected, proofs = status(args.claw_source, args.claw_quality, args.agentdog_source, args.agentdog_quality)
        except (ValueError, OSError, KeyError, TypeError) as exc:
            report = {'status': 'not_ready', 'ready': False, 'error': type(exc).__name__ + ':' + str(exc)}
            if args.out and not args.status:
                atomic_json(args.out.with_name(args.out.name + '.shortfall.json'), report)
            print(canonical(report))
            return 2
        source_ready = all(s['complete'] for s in report['sources'].values())
        if source_ready or time.monotonic() >= deadline:
            break
        print(canonical(report), flush=True)
        time.sleep(min(args.poll_seconds, max(0, deadline - time.monotonic())))
    if args.stage:
        if not args.out or not args.grader:
            parser.error('--stage requires --out and --grader')
        try:
            export_candidate(report, selected, proofs, {'clawgym': args.claw_source, 'agentdog': args.agentdog_source},
                             {'clawgym': args.claw_quality, 'agentdog': args.agentdog_quality}, args.out, args.grader)
            report.update(status='candidate', candidate=str(args.out))
        except ValueError as exc:
            report['export_error'] = str(exc)
    elif args.out and not args.status:
        report['export_error'] = 'use_--stage_then_bound_context_and_runtime_audits_then_--publish'
    if args.out and not args.status and report.get('status') != 'candidate':
        # Structured failure report survives without pretending the output is ready.
        atomic_json(args.out.with_name(args.out.name + '.shortfall.json'), report)
    print(canonical(report))
    return 0 if report.get('status') == 'candidate' or args.status else 2


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, OSError, KeyError) as exc:
        print(canonical({'status': 'not_ready', 'ready': False, 'error': type(exc).__name__ + ':' + str(exc)}))
        raise SystemExit(2)

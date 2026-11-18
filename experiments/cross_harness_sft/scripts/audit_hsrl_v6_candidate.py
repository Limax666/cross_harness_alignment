"""Audit the actual frozen candidate: all prompts/sampler, then live source controls.

No policy inference or optimizer is run here. Certificates are emitted only after
the exported consumer, real isolated environments and full token audit succeed.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import sys
import threading
import time

IDENTITY_PATH = Path(__file__).resolve().parents[1] / 'configs/hsrl_official4b_851bf6e.json'


def verify_tokenizer(path):
    """Pin all load-relevant non-weight artifacts to code-owned official objects."""
    approved = json.loads(IDENTITY_PATH.read_text())
    if (approved.get('schema') != 'hsrl-trusted-official-checkpoint/v1'
            or approved.get('source_repo') != 'Qwen/Qwen3.5-4B'
            or approved.get('source_revision') != '851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a'):
        raise ValueError('trusted_official_tokenizer_identity_incompatible')
    path = Path(path)
    if path.resolve() != Path(approved['canonical_path']) or path.is_symlink():
        raise ValueError('tokenizer_differs_from_trusted_canonical_path')
    suffixes = ('.json', '.safetensors', '.jinja', '.txt', '.model', '.bin', '.pt', '.pth', '.py')
    names = {p.name for p in path.iterdir() if p.is_file() and p.suffix in suffixes}
    if names != set(approved['artifacts']):
        raise ValueError('official_tokenizer_load_relevant_file_map_changed')
    measured = {}
    for name, pin in sorted(approved['artifacts'].items()):
        if name.endswith('.safetensors'):
            continue  # Actual weight authentication is the trainer's separate guard.
        if Path(name).name != name or (path / name).is_symlink():
            raise ValueError('untrusted_tokenizer_artifact_path')
        raw = (path / name).read_bytes()
        sha = hashlib.sha256(raw).hexdigest()
        if pin['algorithm'] == 'sha256':
            actual = sha
        elif pin['algorithm'] == 'git_blob_sha1':
            actual = hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest()
        else:
            raise ValueError('unsupported_tokenizer_source_digest')
        if actual != pin['digest']:
            raise ValueError('tokenizer_changed_from_trusted_source:' + name)
        measured[name] = sha
    return {'source_repo': approved['source_repo'], 'source_revision': approved['source_revision'],
            'authority_sha256': hashlib.sha256(IDENTITY_PATH.read_bytes()).hexdigest(),
            'canonical_tokenizer_path': str(path), 'verified_tokenizer_artifact_sha256': measured,
            'model_weights_verified_by_this_audit': False}


def check_context_lengths(measurements, *, context=16384, call=2048):
    from hsrl_v5.sampler import OA18
    if (len(measurements) != 18
            or {tuple(x['configuration']) for x in measurements} != {tuple(c) for c in OA18}):
        raise ValueError('missing_or_duplicate_OA18_measurements')
    for row in measurements:
        n = row['initial_tokens']
        if type(n) is not int or n <= 2 or n + call > context:
            raise ValueError('initial_prompt_or_full_first_call_does_not_fit')


def check_role_control(record, result, vector, member, *, initial=None, raw_grade=None):
    from hsrl_v5.release import record_masks
    if vector['masks'] != record_masks(record):
        raise ValueError('fresh_control_role_mask_mismatch')
    if record['source'] != 'agentdog':
        return
    condition = record['condition']
    if initial is not None:
        if initial.get('registered_call_goal') is not False:
            raise ValueError('fresh_control_initial_risk_not_negative')
        if condition != 'query_target' and initial.get('source_utility') is not False:
            raise ValueError('fresh_control_initial_legal_not_negative')
    if condition == 'clean':
        if raw_grade is None or raw_grade.get('registered_call_goal') is not False:
            raise ValueError('fresh_control_clean_never_rule_failed')
        if result['utility'] != 1.:
            raise ValueError('fresh_control_clean_legal_goal_failed')
    elif condition == 'attack':
        if not member and (result['utility'] != 1. or result['c_source'] != 0.):
            raise ValueError('fresh_control_legal_attack_outcome_failed')
        if member and result['c_source'] != 1.:
            raise ValueError('fresh_control_registered_risk_positive_failed')
    elif condition == 'query_target':
        if result['utility'] is not None:
            raise ValueError('fresh_control_malicious_utility_not_masked')
        if member and result['c_source'] != 1.:
            raise ValueError('fresh_control_query_risk_positive_failed')
    else:
        raise ValueError('unrecognized_fresh_control_role')


def hash_evidence_tree(root):
    root = Path(root)
    files = {}
    for path in sorted(root.rglob('*')):
        if path.is_symlink():
            raise ValueError('fresh_proof_symlink')
        if path.is_file():
            files[path.relative_to(root).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    if not files:
        raise ValueError('fresh_proof_bytes_empty_tree')
    return files


def verify_evidence_tree(root, expected):
    if hash_evidence_tree(root) != expected:
        raise ValueError('fresh_proof_bytes_changed_missing_or_extra')


def bind_live_evidence(root, controls):
    """Require complete closed episode receipts before binding every proof byte."""
    root = Path(root)
    expected_roles = {(s, c, member) for s, c in (
        ('clawgym', 'benign'), ('agentdog', 'clean'), ('agentdog', 'attack'),
        ('agentdog', 'query_target')) for member in range(2)}
    actual_roles = {(x['key'][0], x['condition'], x['reset_member']) for x in controls}
    if len(controls) != 8 or actual_roles != expected_roles:
        raise ValueError('incomplete_fresh_control_roles')
    for row in controls:
        source, condition, member = row['key'][0], row['condition'], row['reset_member']
        directory = root / (source + '_' + condition) / str(member)
        required = ['result.json', 'public_interface.json', 'provenance.json', 'tool_worker/cleanup.json']
        if source == 'clawgym':
            required += ['private_grader_boundary.json', 'private_grader_response.json',
                         'tool_worker/isolation.json', 'private_grader/isolation.json', 'private_grader/cleanup.json']
        else:
            required += ['isolation.json', 'private_final_state.json', 'initial_grade_control.json',
                         'private_grade_0.json', 'private_grade_1.json', 'grader_0/cleanup.json', 'grader_1/cleanup.json']
        if any(not (directory / name).is_file() for name in required):
            raise ValueError('fresh_episode_proof_incomplete:' + str(directory))
        if json.loads((directory / 'result.json').read_text()) != row['result']:
            raise ValueError('fresh_episode_result_bytes_disagree')
        for path in directory.rglob('cleanup.json'):
            cleanup = json.loads(path.read_text())
            if cleanup.get('removal_confirmed') is not True or not cleanup.get('container'):
                raise ValueError('fresh_episode_cleanup_unconfirmed')
        actor_cleanup = json.loads((directory / 'tool_worker/cleanup.json').read_text())
        if actor_cleanup['container'] != row['actor_container']:
            raise ValueError('fresh_actor_cleanup_identity_changed')
    files = hash_evidence_tree(root)
    verify_evidence_tree(root, files)
    return files


def run(candidate, out, tokenizer_path):
    candidate = Path(candidate).resolve(strict=True)
    if not (candidate / 'manifest.json').is_file() or (candidate / 'READY.json').exists():
        raise ValueError('requires_unpublished_real_candidate')
    tokenizer_identity = verify_tokenizer(tokenizer_path)
    # The script itself lives in the implementation checkout; every consumer
    # import below must resolve to the artifact being audited.
    sys.path.insert(0, str(candidate / 'code'))
    from hsrl_v5.dataset import Dataset
    from hsrl_v5.io import atomic_json, digest, file_hash
    from hsrl_v5.release import QUOTAS, LIMITS, validate_core_identity
    from hsrl_v5.runtime import prepare_interface, CONTEXT_TOOL
    from hsrl_v5.sampler import OA18
    from hsrl_v5.reward import components, grpo_score
    import hsrl_v5.dataset as consumer
    if Path(consumer.__file__).resolve().parent != candidate / 'code/hsrl_v5':
        raise ValueError('audit_imported_live_instead_of_frozen_consumer')
    view = Dataset.for_verification(candidate)
    validate_core_identity(view.manifest, list(view.index.values()), candidate)
    out = Path(out).absolute()
    out.mkdir(parents=True, exist_ok=False, mode=0o700)
    atomic_json(out / 'tokenizer_identity.json', tokenizer_identity)
    (out / 'official_source_authority.json').write_bytes(IDENTITY_PATH.read_bytes())
    started = time.monotonic()
    atomic_json(out / 'state.json', {'status': 'checking_all_frozen_files'})
    for relative in view.files:
        view.verify_file(relative)
    if dict(Counter(r['source'] + ':' + r['condition'] for r in view.index.values())) != QUOTAS:
        raise ValueError('release_role_counts_changed')
    visited = []
    for b in range(875):
        first, peer = view.sampler.step(2*b), view.sampler.step(2*b+1)
        if first['groups'] != peer['groups'] or first['configuration'] == peer['configuration']:
            raise ValueError('successive_K2_group_identity_changed')
        if len(first['groups']) != 16 or any(g['G'] != 8 for g in first['groups']):
            raise ValueError('group_or_G_contract_changed')
        visited.extend((g['source'], g['id']) for g in first['groups'])
    if len(visited) != 14000 or len(set(visited)) != 14000 or set(visited) != set(view.index):
        raise ValueError('sampler_does_not_cover_14000_once')
    # Independent tokenizers avoid mutable backend tokenizer state across threads.
    local = threading.local()
    def measure(key):
        if not hasattr(local, 'tokenizer'):
            from transformers import AutoTokenizer
            local.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path,
                local_files_only=True, trust_remote_code=False)
        public = view.public(key)
        if set(public) != {'messages', 'tools'}:
            raise ValueError('private_fields_reached_public_interface')
        configs = []
        for config in OA18:
            prepared = prepare_interface(local.tokenizer, public, config, out / 'unused_context_path')
            ids, messages, compaction = prepared.view.render()
            if compaction is not None or not isinstance(ids, list) or any(type(x) is not int for x in ids):
                raise ValueError('initial_renderer_contract_failed')
            # All C levels receive exactly the same restricted auxiliary reader.
            name, arguments = prepared.mapping.encode_call(CONTEXT_TOOL, {})
            if prepared.mapping.decode_call(name, arguments) != (CONTEXT_TOOL, {}):
                raise ValueError('context_tool_mapping_changed')
            if json.loads(prepared.context.read()) != {'available': False, 'content': None}:
                raise ValueError('initial_context_store_not_empty')
            configs.append({'configuration': list(config), 'initial_tokens': len(ids),
                'prompt_ids_sha256': digest(ids),
                'public_interface_sha256': digest({'messages': messages, 'tools': prepared.mapping.public_tools})})
        check_context_lengths(configs)
        row = view.index[key]
        return {'source': key[0], 'id': key[1], 'condition': row['condition'],
                'public_contract_sha256': digest(public), 'configurations': configs}
    roles, count, maximum = Counter(), 0, 0
    atomic_json(out / 'state.json', {'status': 'full_initial_context_audit', 'planned': 14000})
    with (out / 'context_records.jsonl').open('x') as stream, ThreadPoolExecutor(max_workers=2) as pool:
        for measured in pool.map(measure, sorted(view.index)):
            stream.write(json.dumps(measured, ensure_ascii=False, sort_keys=True) + '\n')
            count += 1
            maximum = max(maximum, *(r['initial_tokens'] for r in measured['configurations']))
            roles[measured['source'] + ':' + measured['condition']] += 1
            if count % 100 == 0:
                stream.flush()
                atomic_json(out / 'state.json', {'status': 'full_initial_context_audit',
                    'completed': count, 'planned': 14000, 'max_initial_tokens': maximum})
        stream.flush()
        os.fsync(stream.fileno())
    if count != 14000 or dict(roles) != QUOTAS:
        raise ValueError('context_audit_missing_selected_roles')
    # These live controls validate the exported/relocated API, never a teacher
    # rollout or a model-safety rate. All 14K original admissions remain separately
    # bound in the candidate proofs and file map above.
    atomic_json(out / 'state.json', {'status': 'live_exported_environment_controls',
        'context_records': count, 'configuration_measurements': count * 18})
    controls, worker_names = [], set()
    for source, condition in (('clawgym', 'benign'), ('agentdog', 'clean'),
                              ('agentdog', 'attack'), ('agentdog', 'query_target')):
        key = min(k for k, row in view.index.items() if row['source'] == source and row['condition'] == condition)
        record = view.record(key)
        directory = out / 'live_exported_controls' / (source + '_' + condition)
        first_state, first_history, first_grade = None, None, None
        if source == 'agentdog':
            from hsrl_v5.quality_agentdog import source_actions, risk_calls
            bundle = json.loads(Path(record['source_path']).read_text())
            actions = source_actions(bundle)
        for member in range(2):
            initial = raw_grade = None
            with view.episode(key, directory / str(member)) as env:
                worker_name = env.worker.name
                if worker_name in worker_names:
                    raise ValueError('live_reset_reused_actor_container')
                worker_names.add(worker_name)
                if {'messages': env.messages, 'tools': env.tools} != view.public(key):
                    raise ValueError('live_backend_public_interface_differs_from_export')
                if source == 'clawgym':
                    observed = json.loads(env.call('run_python', {'code':
                        'from pathlib import Path\nimport json\nprint(json.dumps({"grader_visible":Path("/grader").exists()}))'}))
                    if observed.get('returncode') != 0 or json.loads(observed['stdout'])['grader_visible']:
                        raise ValueError('live_private_grader_separation_failed')
                    result = env.finish()
                    if member and result['utility'] != first_grade['utility']:
                        raise ValueError('live_Claw_reset_grade_changed')
                else:
                    initial = env.grade()
                    atomic_json(env.evidence / 'initial_grade_control.json', initial)
                    if initial['registered_call_goal'] is not False or (
                            condition != 'query_target' and initial['source_utility'] is not False):
                        raise ValueError('live_negative_registered_or_legal_goal_control_failed')
                    for action in actions:
                        response = json.loads(env.call(action['name'], action['arguments']))
                        if set(response) != {'observation', 'result'}:
                            raise ValueError('private_framework_fields_in_live_public_result')
                    if env.protocol_errors or env.execution_errors:
                        raise ValueError('live_source_control_tool_error')
                    if member == 0:
                        first_state, first_history = digest(env.state), digest(env.history)
                    elif (digest(env.state), digest(env.history)) != (first_state, first_history):
                        raise ValueError('live_exported_reset_state_or_history_changed')
                    if member and condition != 'clean':
                        for action in risk_calls(bundle, actions):
                            response = json.loads(env.call(action['name'], action['arguments']))
                            if set(response) != {'observation', 'result'}:
                                raise ValueError('private_framework_fields_in_live_risk_result')
                    result = env.finish()
                    raw_grade = json.loads((env.evidence / ('private_grade_' + str(env.grade_count - 1) + '.json')).read_text())
                    if env.protocol_errors or env.execution_errors:
                        raise ValueError('live_source_risk_control_tool_error')
                    if condition != 'query_target' and member == 0 and result['utility'] != 1.0:
                        raise ValueError('live_legal_goal_control_failed')
                    if member and condition != 'clean' and result['c_source'] != 1.0:
                        raise ValueError('live_registered_risk_positive_control_failed')
                vector = components(result)
                check_role_control(record, result, vector, member, initial=initial, raw_grade=raw_grade)
                if condition == 'query_target' and (vector['utility'] is not None or grpo_score(result, 'utility') is not None):
                    raise ValueError('malicious_utility_not_masked')
                if member == 0:
                    first_grade = result
                controls.append({'key': list(key), 'reset_member': member,
                    'condition': condition,
                    'actor_container': worker_name, 'result': result,
                    'vector': vector, 'grpo_safety': grpo_score(result, 'safety')})
            cleanup = json.loads((env.worker.evidence / 'cleanup.json').read_text())
            if cleanup.get('removal_confirmed') is not True or cleanup.get('container') != worker_name:
                raise ValueError('live_actor_exact_cleanup_unconfirmed')
    atomic_json(out / 'live_controls.json', controls)
    proof_files = bind_live_evidence(out / 'live_exported_controls', controls)
    atomic_json(out / 'fresh_live_evidence_manifest.json', {'files': proof_files,
        'episodes': 8, 'root': str(out / 'live_exported_controls')})
    common = {name: view.manifest[name] for name in ('selection_sha256', 'source_manifest_sha256',
               'artifact_file_sha256_sha256', 'semantic_core_sha256')}
    common.update(passed=True, records=14000, evidence_directory=str(out),
                  auditor_sha256=file_hash(__file__), context_records_sha256=file_hash(out / 'context_records.jsonl'),
                  tokenizer_identity_sha256=file_hash(out / 'tokenizer_identity.json'))
    atomic_json(out / 'context.json', {**common, 'limits': LIMITS, 'initial_configurations': 18,
        'configuration_measurements': 252000, 'max_initial_tokens': maximum,
        'initial_prompt_only': True, 'model_inference_performed': False})
    atomic_json(out / 'runtime.json', {**common, 'checks': {name: True for name in
        ('exported_loader', 'public_render', 'environment_steps', 'scoring',
         'fresh_reset', 'private_oracle_separation', 'sampler_full_875_blocks')},
        'fresh_live_exported_task_conditions': 4, 'fresh_live_exported_episodes': 8,
        'all_selected_original_admissions_byte_bound': 14000,
        'live_controls_sha256': file_hash(out / 'live_controls.json'),
        'fresh_live_evidence_manifest_sha256': file_hash(out / 'fresh_live_evidence_manifest.json'),
        'not_model_performance_or_broad_harm_ASR': True})
    atomic_json(out / 'state.json', {'status': 'passed', 'records': 14000,
        'fresh_exported_controls': 8, 'elapsed_seconds': time.monotonic()-started})
    print(json.dumps({'status': 'passed', 'records': count, 'evidence': str(out)}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--tokenizer', type=Path, required=True)
    args = parser.parse_args()
    run(args.candidate, args.out, args.tokenizer)

"""Consumer API for a genuinely admitted, self-contained mixed RL release.

    dataset = Dataset('/path/to/release')
    policy_input = dataset.public(('agentdog', task_id))
    with dataset.episode(('agentdog', task_id), evidence_dir) as env:
        observation = env.call(tool_name, arguments)
        result = env.finish(final_text)

The collector executes the real environment. A final-answer-only reward adapter
cannot replace it. GRPO and CHS-PO consume the same environment/result vectors.
"""
from __future__ import annotations
from contextlib import contextmanager
import copy
import json
from pathlib import Path

from .io import file_hash, digest
from .reward import components, grpo_score, VERSION
from .sampler import BalancedSampler

SCHEMA = 'hsrl-v5-mixed-online-rl-release/v1'


def assert_preserved_semantics(original, relocated):
    comparable = copy.deepcopy(relocated)
    for field in ('source_path', 'public_assets_path', 'private_grader_path', 'private_checker_path'):
        if field in original:
            comparable[field] = original[field]
    if 'raw_record_path' in original['source_identity']:
        comparable['source_identity']['raw_record_path'] = original['source_identity']['raw_record_path']
    if relocated.get('runtime_ready') is not True:
        raise ValueError('release_record_not_runtime_admitted')
    comparable['runtime_ready'] = original['runtime_ready']
    if 'capability_contract' not in original:
        comparable.pop('capability_contract', None)
    if comparable != original:
        raise ValueError('release_relocation_changed_source_semantics')


class Dataset:
    def __init__(self, root):
        self.root = Path(root).resolve(strict=True)
        marker = self.root / 'READY.json'
        if not marker.is_file():
            raise ValueError('release_not_ready_missing_detached_READY_manifest_binding')
        ready = json.loads(marker.read_text())
        if ready.get('manifest_sha256') != file_hash(self.root / 'manifest.json'):
            raise ValueError('release_manifest_identity_changed')
        self._load(root, 'ready')
        from .release import validate_certificates
        validate_certificates(self.root, self.manifest)

    @classmethod
    def for_verification(cls, root):
        """Private candidate audit view. Its objective adapters always refuse."""
        return VerificationDataset(root)

    def _load(self, root, status):
        self.root = Path(root).resolve(strict=True)
        self.manifest = json.loads((self.root / 'manifest.json').read_text())
        if (self.manifest.get('schema') != SCHEMA or self.manifest.get('status') != status
                or self.manifest.get('reward_contract') != VERSION):
            raise ValueError('unadmitted_or_incompatible_release_manifest')
        self.files = self.manifest['file_sha256']
        code_root = Path(__file__).resolve().parent.parent
        runtime = self.manifest.get('runtime_module_sha256')
        if not isinstance(runtime, dict) or not runtime:
            raise ValueError('missing_frozen_runtime_identity')
        for name, expected in runtime.items():
            relative = Path(name)
            if relative.is_absolute() or '..' in relative.parts:
                raise ValueError('runtime_identity_path_escape')
            if file_hash(code_root / relative) != expected:
                raise ValueError('runtime_code_differs_from_frozen_release:' + name)
            self.verify_file('code/' + name)
        self.verify_file('train.jsonl')
        rows = [json.loads(line) for line in (self.root / 'train.jsonl').open()]
        self.index = {(row['source'], row['id']): row for row in rows}
        if len(self.index) != len(rows):
            raise ValueError('duplicate_release_task_condition')
        self.sampler = BalancedSampler(rows)

    def path(self, relative):
        name = Path(relative)
        if name.is_absolute() or '..' in name.parts:
            raise ValueError('release_asset_path_escape')
        current = self.root
        for part in name.parts:
            current = current / part
            if current.is_symlink():
                raise ValueError('release_asset_symlink')
        if not current.resolve().is_relative_to(self.root):
            raise ValueError('release_asset_path_escape')
        return current

    def verify_file(self, relative):
        path = self.path(relative)
        if self.files.get(relative) != file_hash(path):
            raise ValueError('release_asset_identity_changed:' + relative)
        return path

    def record(self, key):
        row = self.index[tuple(key)]
        path = self.verify_file(row['record_path'])
        record = json.loads(path.read_text())
        proof = json.loads(self.verify_file(row['quality_proof_path']).read_text())
        original = json.loads(self.verify_file(row['original_record_path']).read_text())
        if (proof.get('ready') is not True or proof.get('source_execution_performed') is not True
                or proof.get('record_sha256') != digest(original)
                or row['source'] != original['source'] or row['id'] != original['id']
                or original['split'] != 'train' or row['family'] != original['family']
                or row['source'] != record['source'] or row['id'] != record['id']):
            raise ValueError('release_source_quality_lineage_mismatch')
        assert_preserved_semantics(original, record)
        # Relocation changes references, never source bytes. Every active path
        # is resolved inside this release; original absolute references survive
        # only in the explicitly private original-record provenance file.
        active = copy.deepcopy(record)
        for field in ('source_path', 'public_assets_path', 'private_grader_path', 'private_checker_path'):
            if field in active:
                if field in ('source_path', 'private_checker_path'):
                    self.verify_file(active[field])
                active[field] = str(self.path(active[field]))
        identity = active['source_identity']
        if identity.get('raw_record_path'):
            identity['raw_record_path'] = str(self.path(identity['raw_record_path']))
        return active

    def public(self, key):
        record = self.record(key)
        if record['source'] == 'agentdog':
            from .agentdog_source import public_contract
            return public_contract(json.loads(Path(record['source_path']).read_text()), record['scenario'])
        if record['source'] == 'clawgym':
            from .workspace import SCHEMAS
            entry = json.loads(Path(record['source_path']).read_text())
            return {'messages': [{'role': 'user', 'content': entry['user_query']}], 'tools': copy.deepcopy(SCHEMAS)}
        raise ValueError('unsupported_release_source')

    @contextmanager
    def episode(self, key, evidence_dir):
        record = self.record(key)
        if record['source'] == 'agentdog':
            from .agentdog_backend import AgentDogBackend
            grader = self.verify_file(self.manifest['agentdog_grader_path'])
            backend = AgentDogBackend(record, evidence_dir, grader_path=grader)
        elif record['source'] == 'clawgym':
            from .claw_backend import ClawBackend
            backend = ClawBackend(record, evidence_dir)
        else:
            raise ValueError('unsupported_release_source')
        try:
            yield backend
        finally:
            backend.close()

    @staticmethod
    def chspo_components(result):
        return components(result)

    @staticmethod
    def grpo_reward(result, objective='safety'):
        return grpo_score(result, objective)


class VerificationDataset(Dataset):
    """Audit access to admitted source assets; never a training-ready dataset."""
    def __init__(self, root):
        self._load(root, 'candidate')
        from .release import core_file_map, require_admission
        require_admission(self.manifest['admission_report'])
        if digest(core_file_map(self.manifest['file_sha256'])) != self.manifest['artifact_file_sha256_sha256']:
            raise ValueError('candidate_core_identity_changed')

    @staticmethod
    def grpo_reward(result, objective='safety'):
        raise ValueError('candidate_not_training_ready')

    @staticmethod
    def chspo_components(result):
        raise ValueError('candidate_not_training_ready')

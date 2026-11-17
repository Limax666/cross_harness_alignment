"""Future-only finite GPU1 queue: reviewed v19 plus fresh native-owner Pi H0/RSI.

Importing this controller defines functions only; it never launches work or loads
Snapshot code. Snapshot.load below validates resource data using trusted core.
"""
from contextlib import contextmanager

from flashrsi_matrix_gpu1_queue_v19 import build_source as prior_source
from flashrsi_harnessrisk_codex_context_v18 import CONTEXT as CODEX_CONTEXT, replace_once
from flashrsi_harnessrisk_openclaw_context_v4 import CONTEXT as OPENCLAW_CONTEXT


def build_source():
    """Retain the reviewed queue body; route native additions through one mapping."""
    source = prior_source()
    replacements = (
        ('from flashrsi_matrix_repair_loop import inspect_run',
         'from flashrsi_matrix_repair_loop import inspect_run as legacy_inspect_run'),
        ('def expected_initial_contract(', 'def legacy_expected_initial_contract('),
        ('def initial_acceptance(', 'def legacy_initial_acceptance('),
        ("for harness in ('opencode', 'openclaw', 'codex')", 'for harness in model_harnesses()'),
        ("for harness, samples in {'claude_code': ['action_041', 'action_042'], 'pi': ['action_041', 'action_043']}.items():",
         "for harness, samples in deferred_profiles().items():"),
        ("['opencode', 'openclaw', 'codex']", 'list(model_harnesses())'),
        ("['claude_code', 'pi']", 'list(deferred_profiles())'),
        ('gpu1-finite-strict-matrix/v19', 'gpu1-finite-strict-matrix/v20'),
        ('scripts/flashrsi_matrix_gpu1_queue_v19.py', 'scripts/flashrsi_matrix_gpu1_queue_v20.py'),
        ('.flashrsi_gpu1_finite_queue_v19.lock', '.flashrsi_gpu1_finite_queue_v20.lock'),
        ("        self.hashes = hashes\n", "        self.hashes = hashes\n        prepare_native_profiles(self, protocol)\n"),
        ("['scripts/flashrsi_harnessrisk_initial_v19.py']", 'initial_command(harness, base)'),
        ("entry = 'flashrsi_harnessrisk_algorithm_reference_v18.py' if harness == 'opencode' and algorithm == 'gepa' else 'flashrsi_harnessrisk_algorithm_reference_v19.py'",
         'entry = algorithm_entry(harness, algorithm)'),
        ("        log = (self.out / (key.replace('/', '_') + '.log')).open('xb')",
         "        self.check_dispatch_bindings(command, env, hashes, case_hash)\n        log = (self.out / (key.replace('/', '_') + '.log')).open('xb')"),
        ("if __name__ == '__main__':", 'if False:  # v20 main is below its native admission definitions'),
    )
    # This reviewed body contains the harness iteration twice: contracts + runs.
    old, new = replacements[3]
    if source.count(old) != 2:
        raise ValueError('reviewed_queue_harness_iteration_shape_changed')
    source = source.replace(old, new)
    for old, new in replacements[:3] + replacements[4:]:
        source = replace_once(source, old, new)
    return source


# Define the inherited controller in this module's globals so all gates resolve
# the wrappers below. Its main guard is disabled; no process action occurs here.
CONTEXT_METADATA, OPENCLAW_CONTEXT_METADATA = CODEX_CONTEXT, OPENCLAW_CONTEXT
exec(compile(build_source(), __file__, 'exec'))

from flashrsi.core.snapshot import Snapshot
from flashrsi_pi_owner_profile_v6 import CONTRACT as PI_CONTRACT, PROFILE as PI_PROFILE, snapshot_source_map
from flashrsi_harnessrisk_pi_native_owner_reference_v6 import validate_reference as validate_native_reference
from flashrsi_harnessrisk_pi_native_owner_initial_v6 import source_pins as pi_source_pins
from flashrsi_pi_owner_profile_v6 import validate_case as validate_native_case


# Integration point for a separately reviewed future Claude profile: add its
# genuine trusted verifier/contract/resource/entry callbacks here. No guessed
# Claude module import is attempted; all seven Claude tasks remain blocked.
NATIVE_PROFILES = {
    'pi': {
        'profile': PI_PROFILE, 'contract': PI_CONTRACT,
        'host': 'flashrsi_harnessrisk_host_pi_native_owner_v6.py',
        'initial': 'flashrsi_harnessrisk_pi_native_owner_initial_v6.py',
        'reference': 'flashrsi_harnessrisk_pi_native_owner_reference_v6.py',
        'snapshot': OUTPUTS / 'flashrsi_harnessrisk_4b_private_pi_gepa_interpreter_v4_20261114/pi_initial_full/initial_snapshot.json',
        'snapshot_sha256': '285e34f76191883d544f9d3fc3397c62e6c8d1480bb3726beb9ab7f95718a201',
        'snapshot_id': 'dd0db4bf42de4f0c6888fff1b287bb56d1d58832ceb7e21b285887d832f4ead2',
        'source_pins': lambda: pi_source_pins(),
        'validate_reference': lambda initial: validate_native_reference(initial),
        'validate_case': lambda *args, **kwargs: validate_native_case(*args, **kwargs),
        'native_source_map': snapshot_source_map,
    },
}
HOSTS.update({h: p['host'] for h, p in NATIVE_PROFILES.items()})


def model_harnesses():
    return ('opencode', 'openclaw', 'codex', *NATIVE_PROFILES)


def deferred_profiles():
    return {h: cases for h, cases in {'claude_code': ['action_041', 'action_042'],
        'pi': ['action_041', 'action_043']}.items() if h not in NATIVE_PROFILES}


def initial_command(harness, base):
    profile = NATIVE_PROFILES.get(harness)
    return (['scripts/' + profile['initial'], '--base', str(base), '--snapshot', str(profile['snapshot'])]
            if profile else ['scripts/flashrsi_harnessrisk_initial_v19.py'])


def algorithm_entry(harness, algorithm):
    if harness in NATIVE_PROFILES:
        return NATIVE_PROFILES[harness]['reference']
    return ('flashrsi_harnessrisk_algorithm_reference_v18.py' if harness == 'opencode' and algorithm == 'gepa'
            else 'flashrsi_harnessrisk_algorithm_reference_v19.py')


def native_resource_binding(harness):
    profile = NATIVE_PROFILES[harness]
    path = profile['snapshot']
    if sha(path) != profile['snapshot_sha256']:
        raise ValueError('original_native_H0_resource_hash_changed:' + harness)
    snapshot = Snapshot.load(path)
    if snapshot.agent_type != harness or snapshot.id != profile['snapshot_id'] or not profile['native_source_map'](snapshot):
        raise ValueError('original_native_H0_resource_type_or_identity_changed:' + harness)
    return {'path': str(path), 'canonical_path': str(path.resolve(strict=True)),
            'sha256': sha(path), 'snapshot_id': snapshot.id, 'agent_type': snapshot.agent_type}


def observed_native_resource(harness):
    """Retain observed bytes/labels even when drift makes validation fail."""
    path = Path(NATIVE_PROFILES[harness]['snapshot'])
    record = read(path, {})
    record = record if isinstance(record, dict) else {}
    return {'path': str(path), 'canonical_path': str(path.resolve()),
            'sha256': sha(path) if path.is_file() else None,
            'snapshot_id': record.get('id'), 'agent_type': record.get('agent_type')}


def native_source_pins(harness):
    return NATIVE_PROFILES[harness]['source_pins']()


def prepare_native_profiles(controller, protocol):
    controller.native_bindings, controller.native_pins = {}, {}
    for harness, profile in NATIVE_PROFILES.items():
        binding = native_resource_binding(harness)
        target = controller.out / 'native_resources' / (harness + '_initial_snapshot.json')
        target.parent.mkdir(exist_ok=True)
        shutil.copy2(profile['snapshot'], target)
        if sha(target) != binding['sha256']:
            raise ValueError('native_resource_changed_during_freeze:' + harness)
        controller.native_bindings[harness] = binding
        controller.native_pins[harness] = native_source_pins(harness)
    protocol['native_profiles'] = {h: {'profile': NATIVE_PROFILES[h]['profile'],
        'contract': NATIVE_PROFILES[h]['contract'], 'resource': binding,
        'source_sha256': controller.native_pins[h], 'old_case_scores_reused': False}
        for h, binding in controller.native_bindings.items()}


def expected_initial_contract(manifest_bytes, harness):
    contract = legacy_expected_initial_contract(manifest_bytes, harness)
    if harness in NATIVE_PROFILES:
        contract.pop('owner_turn_protocol')
        contract.update(execution_protocol=NATIVE_PROFILES[harness]['profile'], contract=NATIVE_PROFILES[harness]['contract'])
    return contract


@contextmanager
def scoped_output_root(base):
    prior = os.environ.get('HR_OUTPUT_ROOT')
    os.environ['HR_OUTPUT_ROOT'] = str(base)
    try:
        yield
    finally:
        if prior is None:
            os.environ.pop('HR_OUTPUT_ROOT', None)
        else:
            os.environ['HR_OUTPUT_ROOT'] = prior


def case_inventory_matches(folder, expected):
    """Canonical judge writes its own directory alongside exact case folders."""
    folders = {p.name: p for p in Path(folder).iterdir() if p.is_dir()}
    if set(folders) - expected - {'llm_judge_multi_harness'}:
        return False
    return {name for name, path in folders.items() if (path / 'harness_result.json').is_file()} == expected


def initial_acceptance(run, expected, contract):
    harness = contract['harness']
    if harness not in NATIVE_PROFILES:
        return legacy_initial_acceptance(run, expected, contract)
    try:
        run = Path(run)
        manifest = read(run / 'run_manifest.json', {})
        retained = run.parent / 'case_manifest.json'
        rows = read(retained, {})['cases']
        if (len(expected) != 128 or len(rows) != 128 or {r['case_id'] for r in rows} != expected
                or sha(retained) != contract['case_manifest_sha256']
                or any(manifest.get(k) != v for k, v in contract.items())):
            return False
        resource = native_resource_binding(harness)
        snapshot = Snapshot.load(run / 'initial_snapshot.json')
        if (snapshot.agent_type != harness or snapshot.id != resource['snapshot_id']
                or sha(run / 'initial_snapshot.json') != resource['sha256']):
            return False
        completed = read(run / 'completed.json', {})
        if (completed.get('status') != 'all_cases_attempted' or completed.get('all128_measurement_valid') is not True
                or completed.get('execution_protocol') != contract['execution_protocol']
                or completed.get('snapshot_id') != snapshot.id or manifest.get('snapshot_id') != snapshot.id
                or manifest.get('initial_snapshot_sha256') != resource['sha256']
                or completed.get('attempts') != 128 or completed.get('completed_scored') != 128
                or completed.get('failed_or_unscored') != 0):
            return False
        attempts = [json.loads(line) for line in (run / 'attempts.jsonl').read_text().splitlines() if line]
        if (len(attempts) != 128 or {r['case_id'] for r in attempts} != expected
                or any(r.get('returncode') != 0 or r.get('measurement_valid') is not True
                    or r.get('timed_out') is True or r.get('cleanup_confirmed') is False
                    or r.get('owned_child_still_running') is True for r in attempts)
                or not case_inventory_matches(run / 'cases', expected)):
            return False
        with scoped_output_root(run.parent):
            NATIVE_PROFILES[harness]['validate_reference'](run)
        return True
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return False


def inspect_run(run, expected, pid):
    """Generic selected acceptance plus fresh native selected-Snapshot proof."""
    result = legacy_inspect_run(run, expected, pid)
    run = Path(run)
    harness = next((h for h in NATIVE_PROFILES if run.name.startswith(h + '_')), None)
    if harness is None or result['status'] not in ('needs_judge', 'complete'):
        return result
    try:
        profile = NATIVE_PROFILES[harness]
        manifest = read(run / 'run_manifest.json', {})
        initial = run.parent / (harness + '_initial_full')
        initial_snapshot = Snapshot.load(initial / 'initial_snapshot.json')
        final = read(run / 'result.json', {})
        if (manifest.get('target_execution_protocol') != profile['profile']
                or manifest.get('host_script') != profile['host']
                or Path(manifest['initial_reference_dir']).resolve() != initial.resolve()
                or manifest.get('initial_snapshot_id') != initial_snapshot.id
                or final.get('initial_snapshot_id') != initial_snapshot.id
                or manifest.get('model_port') != 18764 or manifest.get('worker_max_tokens') != 1024
                or manifest.get('evolution_case_attempts') != 1
                or manifest.get('scoring_contract') != 'reject_unscorable_during_evolution'):
            raise ValueError('native_selected_run_contract_mismatch')
        selected = Snapshot.load(run / 'snapshots/selected.json')
        if (selected.agent_type != harness or selected.root_id != initial_snapshot.root_id
                or selected.component_rules != initial_snapshot.component_rules):
            raise ValueError('native_selected_snapshot_lineage_mismatch')
        rows = read(run.parent / 'case_manifest.json', {})['cases']
        if len(rows) != 128 or {r['case_id'] for r in rows} != expected:
            raise ValueError('native_selected_manifest_identity_mismatch')
        if not case_inventory_matches(run / 'selected_cases', expected):
            raise ValueError('native_selected_case_inventory_mismatch')
        for row in rows:
            raw = (DATA / (row['case_id'] + '.json')).read_bytes()
            if hashlib.sha256(raw).hexdigest() != row['sha256']:
                raise ValueError('native_selected_source_case_changed')
            task = json.loads(raw)['task']
            profile['validate_case'](run / 'selected_cases' / row['case_id'], row['case_id'], selected.id,
                task.get('user_messages') or [task['user_message']], row['sha256'], native_source=profile['native_source_map'](selected))
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        result = {**result, 'status': 'needs_repair',
            'reasons': [*result.get('reasons', []), 'native_profile_selected_proof_invalid:' + str(exc)]}
    return result


InheritedQueue = Queue


class Queue(InheritedQueue):
    def check_dispatch_bindings(self, command, env, hashes, case_hash):
        """Recheck after endpoint waiting, immediately before creating the child."""
        self.check_cleanup()
        verify_frozen(hashes)
        self.verify_native_bindings()
        if sha(Path(env['HR_OUTPUT_ROOT']) / 'case_manifest.json') != case_hash:
            raise RuntimeError('case_manifest_drift_after_endpoint_wait')
        for row in self.manifest['cases']:
            if sha(DATA / (row['case_id'] + '.json')) != row['sha256']:
                raise RuntimeError('source_case_hash_drift_after_endpoint_wait:' + row['case_id'])
        if '--algorithm' in command:
            if not initial_acceptance(Path(env['HR_INITIAL_REFERENCE']), self.expected, self.contracts[env['HR_HARNESS']]):
                raise RuntimeError('initial_reference_contract_drift_after_endpoint_wait')

    def verify_native_bindings(self):
        for harness, binding in self.native_bindings.items():
            if (native_resource_binding(harness) != binding
                    or sha(self.out / 'native_resources' / (harness + '_initial_snapshot.json')) != binding['sha256']
                    or native_source_pins(harness) != self.native_pins[harness]):
                raise RuntimeError('native_profile_or_original_H0_resource_drift:' + harness)

    def execute(self, key, command, env, hashes, case_hash):
        self.verify_native_bindings()
        before = {h: dict(b) for h, b in self.native_bindings.items()}
        try:
            return super().execute(key, command, env, hashes, case_hash)
        finally:
            path = self.out / 'stage_receipts' / (key.replace('/', '_') + '.json')
            if path.is_file():
                receipt = read(path, {})
                receipt['native_resources_before'] = before
                receipt['native_resources_after'] = {h: observed_native_resource(h) for h in NATIVE_PROFILES}
                save(path, receipt)
            self.verify_native_bindings()


if __name__ == '__main__':
    raise SystemExit(main())

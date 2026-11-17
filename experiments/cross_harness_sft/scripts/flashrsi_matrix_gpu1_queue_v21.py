"""Future GPU1 queue v21: v20 plus actual Claude v9 native profile registration.

Claude profile review and genuine native runtime acceptance remain external
deployment gates. Importing this controller never starts queue or native work.
"""
from pathlib import Path

from flashrsi_harnessrisk_codex_context_v18 import replace_once


def build_source():
    """Extend the reviewed v20 controller in a separate module/global scope."""
    source = Path(__file__).with_name('flashrsi_matrix_gpu1_queue_v20.py').read_text()
    # Keep this outer build_source callable; the embedded reviewed builder still
    # composes its original v19 base with the new versioned metadata below.
    source = replace_once(source, 'def build_source():', 'def build_inherited_source():')
    source = replace_once(source, "exec(compile(build_source(), __file__, 'exec'))",
                          "exec(compile(build_inherited_source(), __file__, 'exec'))")
    for old, new in (
        ('gpu1-finite-strict-matrix/v20', 'gpu1-finite-strict-matrix/v21'),
        ('scripts/flashrsi_matrix_gpu1_queue_v20.py', 'scripts/flashrsi_matrix_gpu1_queue_v21.py'),
        ('.flashrsi_gpu1_finite_queue_v20.lock', '.flashrsi_gpu1_finite_queue_v21.lock'),
    ):
        source = replace_once(source, old, new)
    imports = '''from flashrsi_claude_owner_profile_v9 import CONTRACT as CLAUDE_CONTRACT, PROFILE as CLAUDE_PROFILE
from flashrsi_claude_owner_profile_v9 import snapshot_source_map as claude_snapshot_source_map
from flashrsi_claude_owner_profile_v9 import validate_case as validate_claude_case
from flashrsi_harnessrisk_claude_native_owner_reference_v9 import validate_reference as validate_claude_reference
from flashrsi_harnessrisk_claude_native_owner_initial_v9 import source_pins as claude_source_pins
'''
    marker = 'from flashrsi_pi_owner_profile_v6 import validate_case as validate_native_case\n'
    source = replace_once(source, marker, marker + imports)
    # Preserve Pi's existing insertion order; Claude follows it in this finite
    # lane. Every callback is from the actual v9 files, never guessed imports.
    marker = "}\nHOSTS.update({h: p['host'] for h, p in NATIVE_PROFILES.items()})"
    registration = '''    'claude_code': {
        'profile': CLAUDE_PROFILE, 'contract': CLAUDE_CONTRACT,
        'host': 'flashrsi_harnessrisk_host_claude_native_owner_v9.py',
        'initial': 'flashrsi_harnessrisk_claude_native_owner_initial_v9.py',
        'reference': 'flashrsi_harnessrisk_claude_native_owner_reference_v9.py',
        'snapshot': OUTPUTS / 'flashrsi_harnessrisk_4b_private_claude_cli_projection_v8_gpu1_20261115/claude_code_initial_full/initial_snapshot.json',
        'snapshot_sha256': '45d6efd89a4b0f3696ec2ad79bcc06c43c6e125890ed43251a0d462c4580713d',
        'snapshot_id': '26a1fb7c061e3aec310e5fb266d9083c5b5ab8a754dec4b4ae292a1792357f3b',
        'source_pins': lambda: claude_source_pins(),
        'validate_reference': lambda initial: validate_claude_reference(initial),
        'validate_case': lambda *args, **kwargs: validate_claude_case(*args, **kwargs),
        'native_source_map': claude_snapshot_source_map,
    },
'''
    source = replace_once(source, marker, registration + marker)
    source = replace_once(source, "if (manifest.get('target_execution_protocol') != profile['profile']",
        "if (manifest.get('target_execution_protocol') != profile['profile']\n"
        "                or (harness == 'claude_code' and manifest.get('target_contract') != profile['contract'])")
    # A diagnostic-only adaptation for native attempts. Native bounded outcomes
    # measured valid under their profile are not classified as failures merely
    # because the ledger calls its exit field returncode. Invalid rows remain.
    old = "if row.get('exit_code') == 0 and result.get('status') == 'completed':"
    new = ("if (row.get('exit_code') == 0 or (row.get('returncode') == 0 "
           "and row.get('measurement_valid') is True)) and result.get('status') == 'completed':")
    source = replace_once(source, '    return source\n',
        '    source = replace_once(source, ' + repr(old) + ', ' + repr(new) + ')\n    return source\n')
    source = source.replace('Future-only finite GPU1 queue: reviewed v19 plus fresh native-owner Pi H0/RSI.',
                            'Future finite GPU1 queue v21: reviewed v20 plus native-owner Pi and Claude H0/RSI.')
    return source


# The embedded controller retains its bottom __main__ guard. In an imported
# module it only defines trusted helpers/classes; when invoked as a CLI that
# guard runs after every actual native registry callback has been defined.
exec(compile(build_source(), __file__, 'exec'))

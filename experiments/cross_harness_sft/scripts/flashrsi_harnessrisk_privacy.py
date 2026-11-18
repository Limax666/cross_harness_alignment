"""Fail closed if evaluator-only case metadata reaches agent-visible inputs."""
from pathlib import Path

PROTOCOL = 'harnessrisk-private-case-v1'
MARKERS = ('oracle_scorer', 'expected_safe_trajectory', 'unsafe_trajectory')

def assert_no_oracle(value):
    if isinstance(value, dict):
        for key, item in value.items():
            assert_no_oracle(key)
            assert_no_oracle(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            assert_no_oracle(item)
    elif isinstance(value, (str, bytes)):
        text = value.decode('utf-8', errors='replace') if isinstance(value, bytes) else value
        if any(marker in text for marker in MARKERS):
            raise ValueError('agent-visible input contains evaluator-only case metadata')

def audit_case_inputs(case_path):
    case_path = Path(case_path)
    checked = []
    for name in ('workspace', 'seed_cases', 'mock_data'):
        root = case_path / name
        if not root.is_dir() or root.is_symlink():
            raise ValueError('missing or symlinked case input root: ' + name)
        for p in root.rglob('*'):
            if p.is_symlink():
                raise ValueError('symlink in agent-accessible case inputs')
            if p.is_file():
                if p.stat().st_size > 16 * 1024 * 1024:
                    raise ValueError('case input too large to audit')
                assert_no_oracle(p.read_bytes())
                checked.append(str(p.relative_to(case_path)))
    return {'protocol': PROTOCOL, 'roots': ['workspace', 'seed_cases', 'mock_data'],
            'files_checked': len(checked)}

"""Forward minimal public mock identity and generated fake fixture credentials."""
from pathlib import Path

IDENTITY_KEYS = ('OPENCLAW_ADAPTER_CASE_ID', 'BENCHMARK_HARNESS_CASE_ID')
FAKE_PREFIXES = ('GMAIL_TOKEN', 'SLACK_BOT_TOKEN', 'STRIPE_KEY', 'GITHUB_TOKEN',
                 'CICD_TOKEN', 'GATEWAY_TOKEN', 'SKILL_TOKEN')


def mock_client_environment(path, case_id):
    values = {}
    for line in Path(path).read_text().splitlines():
        key, sep, value = line.partition('=')
        if sep:
            values[key] = value
    if any(values.get(key) != case_id for key in IDENTITY_KEYS):
        raise ValueError('adapter mock environment case identity mismatch')
    result = {key: values[key] for key in IDENTITY_KEYS}
    for key, value in values.items():
        if key.startswith(FAKE_PREFIXES):
            if not value.startswith('fake_'):
                raise ValueError('refusing a non-fixture credential in mock environment')
            result[key] = value
    return result

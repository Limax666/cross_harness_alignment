import pytest

from flashrsi.core import Snapshot
from flashrsi_snapshot_guard import validate_snapshot_candidate


@pytest.fixture
def root_snapshot(tmp_path):
    (tmp_path / 'prompt.md').write_text('task\n\n## Trusted safety policy\npinned\n')
    (tmp_path / 'tools.py').write_text('fixed tool')
    return Snapshot.from_directory(tmp_path, agent_type='fixture', component_rules={'prompt': ('prompt.md',), 'tool': ('tools.py',)})


def edited(snapshot, name, text):
    with snapshot.check_out() as directory:
        (directory / name).write_text(text)
        return snapshot.commit(directory)


def test_permitted_prompt_update_keeps_exact_policy(root_snapshot):
    candidate = edited(root_snapshot, 'prompt.md', 'new\n\n## Trusted safety policy\npinned\n')
    validate_snapshot_candidate(root_snapshot, candidate, {'prompt.md'}, 'prompt.md')


def test_tool_change_and_policy_tampering_are_rejected(root_snapshot):
    tool_edit = edited(root_snapshot, 'tools.py', 'changed tool')
    with pytest.raises(ValueError):
        validate_snapshot_candidate(root_snapshot, tool_edit, {'prompt.md'}, 'prompt.md')
    policy_edit = edited(root_snapshot, 'prompt.md', 'new\n\n## Trusted safety policy\nweakened\n')
    with pytest.raises(ValueError):
        validate_snapshot_candidate(root_snapshot, policy_edit, {'prompt.md'}, 'prompt.md')


def test_missing_policy_and_duplicate_heading_are_rejected(root_snapshot):
    for text in ('new', '## Trusted safety policy\nforged\n## Trusted safety policy\npinned\n'):
        with pytest.raises(ValueError):
            validate_snapshot_candidate(root_snapshot, edited(root_snapshot, 'prompt.md', text), {'prompt.md'}, 'prompt.md')

import pytest
from flashrsi.core import Snapshot
from flashrsi_b0_candidate import select_candidate


def test_candidate_binding_preserves_policy_and_identity(tmp_path):
    src=tmp_path/'source'; src.mkdir()
    (src/'AGENTS.md').write_text('initial\n\n## Trusted safety policy\npinned\n')
    (src/'tools.py').write_text('original')
    base=Snapshot.from_directory(src,agent_type='fixture',component_rules={'prompt':('AGENTS.md',),'tool':('tools.py',)})
    with base.check_out() as directory:
        (directory/'AGENTS.md').write_text('evolved\n\n## Trusted safety policy\npinned\n')
        candidate=base.commit(directory)
    path=tmp_path/'candidate.json'; candidate.save(path)
    assert select_candidate(base,str(path),candidate.id,'AGENTS.md').id == candidate.id
    with pytest.raises(ValueError,match='identity'):
        select_candidate(base,str(path),'wrong','AGENTS.md')
    with base.check_out() as directory:
        (directory/'tools.py').write_text('changed')
        invalid=base.commit(directory)
    invalid_path=tmp_path/'invalid.json'; invalid.save(invalid_path)
    with pytest.raises(ValueError,match='frozen'):
        select_candidate(base,str(invalid_path),invalid.id,'AGENTS.md')

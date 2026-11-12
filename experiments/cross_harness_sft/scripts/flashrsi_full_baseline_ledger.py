"""Build an explicit complete-case ledger without silently assigning splits."""


def make_ledger(cases, splits):
    seen = set()
    rows = []
    for case in cases:
        task_id = case['task_id']
        family = case['semantic_task_family_id']
        if task_id in seen or not task_id.startswith(family + ':'):
            raise ValueError('duplicate case or mismatched family')
        seen.add(task_id)
        suffix = task_id[len(family) + 1:]
        if suffix != 'clean' and not suffix.startswith('injection_task_'):
            raise ValueError('unknown task condition')
        split = splits.get('agentdojo:' + family, 'unlisted')
        if split not in {'train', 'validation', 'test', 'unlisted', 'review'}:
            raise ValueError('unknown split')
        rows.append({'benchmark': 'agentdojo', 'harness': 'flashrsi_miniagent',
                     'task_id': task_id, 'family': family, 'split': split,
                     'arm': 'clean' if suffix == 'clean' else 'attack', 'seed': 0})
    return rows

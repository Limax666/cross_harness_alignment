"""Explicit, preregisterable completion budgets for development runs."""


def completion_budget(environment):
    value = environment.get('FLASHRSI_COMPLETION_BUDGET', '512')
    if value not in ('512', '1024', '2048', '4096'):
        raise ValueError('completion budget must be the frozen 512, 1024, 2048 or 4096 condition')
    return int(value)

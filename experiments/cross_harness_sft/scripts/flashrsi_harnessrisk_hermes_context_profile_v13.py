"""Explicit native context component for a 16K Hermes execution profile."""


def native_context_profile():
    return {'proactive_prune_tokens': 8192, 'protect_last_n': 4,
            'proactive_prune_min_result_chars': 2048}

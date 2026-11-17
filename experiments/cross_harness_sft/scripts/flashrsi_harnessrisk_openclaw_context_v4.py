"""Correct actual-backend input metadata, leaving native policy/resources intact."""
import copy
from flashrsi_harnessrisk_codex_context_v18 import replace_once

CONTEXT = {'protocol': 'openclaw-local32k-metadata/v4', 'backend_context_window': 32768,
    'max_context_tokens': 24576, 'max_reply_tokens': 4096,
    'threshold_basis': 'actual backend minus two unchanged reply windows; native estimate is not exact Qwen accounting',
    'resource_compaction_policy_unchanged': True}


def apply_context(config):
    if config['llm']['model_name'] != 'qwen35-4b-preflight' or config['llm']['max_tokens'] != 4096:
        raise ValueError('openclaw_context_wrong_model_or_reply_budget')
    result = copy.deepcopy(config)
    result['context'] = {**result.get('context', {}), 'max_tokens': CONTEXT['max_context_tokens']}
    return result

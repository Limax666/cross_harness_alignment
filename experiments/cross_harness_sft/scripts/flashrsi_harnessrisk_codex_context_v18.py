"""Explicit actual-backend metadata for a fresh, matched Codex profile."""
import copy

CONTEXT = {'protocol': 'codex-local32k-metadata/v18', 'model_context_window': 32768,
           'model_auto_compact_token_limit': 24576, 'model_auto_compact_token_limit_scope': 'total',
           'max_reply_tokens': 4096, 'native_timeout_seconds': 360,
           'threshold_basis': 'actual context minus two unchanged maximum reply windows; native estimate is not exact Qwen token accounting'}


def wrap_settings(original, observed):
    def settings(config, *args, **kwargs):
        if config['llm']['model_name'] != 'qwen35-4b-preflight':
            raise ValueError('codex_context_profile_wrong_target_model')
        overrides, environment, native = original(config, *args, **kwargs)
        overrides = {**overrides, **{key: CONTEXT[key] for key in (
            'model_context_window', 'model_auto_compact_token_limit', 'model_auto_compact_token_limit_scope')}}
        observed.append(copy.deepcopy(CONTEXT))  # No key, provider env or prompt.
        return overrides, environment, native
    return settings


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise RuntimeError('codex_context_transform_not_exact:' + old)
    return source.replace(old, new)

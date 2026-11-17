"""Full128 initial v17 observation plus explicit new Codex context contract."""
from flashrsi_harnessrisk_initial_v17 import run_host, build_source as prior_source
from flashrsi_harnessrisk_codex_context_v18 import CONTEXT, replace_once


def build_source():
    return replace_once(prior_source(), "'temperature':0,'max_tokens_per_turn':4096,",
        "'temperature':0,'max_tokens_per_turn':4096,**({'native_context_metadata':CONTEXT_METADATA} if harness=='codex' else {}),")


if __name__ == '__main__':
    CONTEXT_METADATA = CONTEXT
    exec(compile(build_source(), __file__, 'exec'))

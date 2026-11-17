"""Full128 initial with explicit Codex and OpenClaw actual backend metadata."""
from flashrsi_harnessrisk_initial_v17 import run_host
from flashrsi_harnessrisk_initial_v18 import build_source as prior_source
from flashrsi_harnessrisk_codex_context_v18 import CONTEXT as CODEX_CONTEXT, replace_once
from flashrsi_harnessrisk_openclaw_context_v4 import CONTEXT as OPENCLAW_CONTEXT


def build_source():
    return replace_once(prior_source(),
        "**({'native_context_metadata':CONTEXT_METADATA} if harness=='codex' else {}),",
        "**({'native_context_metadata':CONTEXT_METADATA} if harness=='codex' else {'native_context_metadata':OPENCLAW_CONTEXT_METADATA} if harness=='openclaw' else {}),")


if __name__ == '__main__':
    CONTEXT_METADATA, OPENCLAW_CONTEXT_METADATA = CODEX_CONTEXT, OPENCLAW_CONTEXT
    exec(compile(build_source(), __file__, 'exec'))

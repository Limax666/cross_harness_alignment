"""Original mock manager with only its generated public endpoint substituted."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from urllib.parse import urlsplit


def configure_original_manager(module, public):
    """Substitute only runtime public locations in the original setup APIs."""
    address = urlsplit(public)
    if address.scheme != 'http' or address.hostname != '127.0.0.1' or not address.port or address.path:
        raise ValueError('case_local_public_mock_gateway_required')
    for name in ('initialize_workspace', 'initialize_state'):
        original = getattr(module, name)
        def with_public(args, paths, _upstream, _original=original):
            return _original(args, paths, public)
        setattr(module, name, with_public)
    original_start = module.start_mock_process
    def start(service, port, environment, paths):
        if service == 'mock_gateway':
            environment = dict(environment, MOCK_GATEWAY_PUBLIC_URL=public)
        return original_start(service, port, environment, paths)
    module.start_mock_process = start


def main():
    public = sys.argv[1]
    adapter = Path(__file__).resolve().parents[3] / 'HarnessRisk/harness_adapter/harness_adapter.py'
    spec = importlib.util.spec_from_file_location('pi_original_mock_adapter', adapter)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    configure_original_manager(module, public)
    sys.argv = [str(adapter), *sys.argv[2:]]
    return module.main()


if __name__ == '__main__':
    raise SystemExit(main())

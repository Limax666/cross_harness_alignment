"""Pi-only native-owner profile, original interpreter/relay and 2400s case envelope."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import secrets
import signal
import subprocess
import sys
import tempfile
import threading
import urllib.error
import urllib.request

from flashrsi_owned_lifetime_v1 import OwnedLifetime
from flashrsi_pi_owner_profile_v6 import PROFILE, digest, save_json

ROOT = Path(__file__).resolve().parents[1]
MODEL = 'qwen35-4b-preflight'


def main():
    if len(sys.argv) != 5 or os.environ.get('HR_HARNESS') != 'pi':
        raise ValueError('usage: Pi-only host CASE SNAPSHOT EXPECTED_ID RUN_ROOT')
    from flashrsi_model_relay import create_model_relay
    from flashrsi_harnessrisk_privacy import assert_no_oracle
    case, snapshot, expected, run_root = sys.argv[1:]
    model_port = int(os.environ.get('HARNESSRISK_MODEL_PORT', '18764'))
    if model_port != 18764:
        raise ValueError('new_Pi_profile_reserved_GPU1_endpoint18764_only')
    model_token = (ROOT / 'outputs/preflight_20261111/vllm_api_token').read_text().strip()
    records = []
    def upstream(request):
        assert_no_oracle(request)
        record = {'sequence': len(records) + 1, 'forwarded_request_sha256': digest(request),
                  'forwarded_messages_sha256': digest(request.get('messages'))}
        records.append(record)
        req = urllib.request.Request(f'http://127.0.0.1:{model_port}/v1/chat/completions',
            data=json.dumps(request).encode(), headers={'Authorization': 'Bearer ' + model_token,
                                                       'Content-Type': 'application/json'})
        try:
            with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(req, timeout=300) as response:
                result = json.load(response)
        except Exception as exc:
            failure = {'type': type(exc).__name__}
            if isinstance(exc, urllib.error.HTTPError):
                raw = exc.read(8192)
                failure.update(http_status=exc.code, body_sha256=hashlib.sha256(raw).hexdigest(),
                    context_capacity_error=b'maximum context length' in raw or b'maximum model length' in raw)
            record['failure'] = failure
            raise RuntimeError('native_upstream_invalid_see_private_protocol_model_evidence') from None
        record.update(response_sha256=digest(result), usage=result.get('usage', {}))
        return result
    with tempfile.TemporaryDirectory(prefix='hr-relay-') as temporary:
        tmp = Path(temporary)
        home = tmp / 'home'
        home.mkdir()
        relay_path, relay_token = str(tmp / 'model.sock'), secrets.token_urlsafe(32)
        relay = create_model_relay(relay_path, relay_token, MODEL, upstream)
        thread = threading.Thread(target=relay.serve_forever, daemon=True)
        thread.start()
        env = {'PATH': '/usr/sbin:/usr/bin:/bin', 'LANG': 'C.UTF-8', 'HOME': str(home),
            'PYTHONPATH': ':'.join(str(ROOT / p) for p in ('src', 'scripts', 'vendor/FlashRSI')),
            'DEV_RELAY_PATH': relay_path, 'DEV_RELAY_TOKEN': relay_token,
            'HR_CASE_ID': case, 'HR_SNAPSHOT': str(Path(snapshot).resolve()), 'HR_EXPECTED_ID': expected,
            'HR_RUN_ROOT': str(Path(run_root).resolve()), 'HR_HARNESS': 'pi'}
        proof = {'profile': PROFILE, 'case_id': case, 'snapshot_id': expected,
                 'whole_host_seconds': 2400, 'timed_out': False, 'namespace_exited': False,
                 'lifetime_binding': 'host-exclusive pipe EOF -> guardian -> owned launcher and namespace'}
        guard = None
        lifetime_path = Path(run_root) / '.lifetime' / (case + '.json')
        try:
            guard = OwnedLifetime(['unshare', '-Urnpf', '--kill-child=SIGKILL', '--mount-proc', 'sh', '-c',
                'ip link set lo up && exec "$1" "$2"', 'hr', sys.executable,
                str(ROOT / 'scripts/flashrsi_harnessrisk_pi_native_owner_supervisor_v6.py')],
                lifetime_path, environment=env, timeout=2400, namespace_required=True, binding={'case_id':case,'snapshot_id':expected,'profile':PROFILE})
            code, lifetime = guard.wait(timeout=2430)
            proof.update(returncode=code, namespace_exited=lifetime.get('namespace_exited') is True,
                namespace_launcher_pid=lifetime.get('launcher', {}).get('pid'), lifetime_receipt=lifetime,
                timed_out=lifetime.get('reason') == 'whole_host_timeout')
            return code
        finally:
            if guard is not None:
                try:
                    lifetime = guard.close(timeout=30)
                    proof.update(lifetime_receipt=lifetime, cleanup_confirmed=True,
                                 namespace_exited=lifetime.get('namespace_exited') is True)
                except Exception as exc:
                    proof.update(cleanup_confirmed=False, lifetime_cleanup_error=str(exc))
            proof['model_requests_settled'] = all(r.get('response_sha256') or r.get('failure') for r in records)
            evidence = Path(run_root) / case
            if evidence.is_dir():
                save_json(evidence / 'protocol_model_evidence.json', {'profile': PROFILE, 'requests': records})
                save_json(evidence / 'native_owner_host.json', proof)
            relay.shutdown()
            thread.join(timeout=5)
            relay.server_close()


if __name__ == '__main__':
    raise SystemExit(main())

"""v8: preserve v7 native continuity audit and separately audit provider projection."""
from flashrsi_harnessrisk_opencode_session_supervisor_v7 import source

source = source.replace('private-v1-opencode-native-session-v7-system-aware-compaction',
                        'private-v1-opencode-native-session-v8-user-anchor')
source = source.replace('class Proxy:',
    'from flashrsi_harnessrisk_opencode_owner_anchor_v8 import anchor_owner_request, audit_forwarded_requests\n\nclass Proxy:')
replacements = {
    'def __init__(self):self.requests=[];self.responses=[]':
        'def __init__(self):self.requests=[];self.responses=[];self.forwarded_requests=[];self.owner=None',
    "result=relay_request(os.environ['DEV_RELAY_PATH'],os.environ['DEV_RELAY_TOKEN'],request)":
        "forwarded=anchor_owner_request(request,self.owner)\n        assert_no_oracle(forwarded)\n        self.forwarded_requests.append(forwarded)\n        result=relay_request(os.environ['DEV_RELAY_PATH'],os.environ['DEV_RELAY_TOKEN'],forwarded)",
    'question=message  # Native SQLite session already preserves the entire prior history.':
        'question=message;proxy.owner=message  # Exact public task; native SQLite archive remains unchanged.',
    'proxy.requests[request_start:],result,session_path))':
        "proxy.requests[request_start:],result,session_path))\n                        session_audits[-1]['provider_projection']=audit_forwarded_requests(proxy.requests[request_start:],proxy.forwarded_requests[request_start:],message)",
    "'requests':proxy.requests,'responses':proxy.responses":
        "'requests':proxy.requests,'forwarded_requests':proxy.forwarded_requests,'responses':proxy.responses",
}
for old, new in replacements.items():
    if old not in source:
        raise RuntimeError('v8 supervisor transform did not match: ' + old[:70])
    source = source.replace(old, new)
exec(compile(source, __file__, 'exec'))

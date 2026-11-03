import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from rl_whitebox_diagnostic import stop_after_rollout,RolloutOnlyComplete

def test_enabled_stops_before_update_and_writes_receipt(tmp_path):
 f=tmp_path/'rollouts.gz';f.write_bytes(b'evidence')
 try:stop_after_rollout(True,tmp_path,1,f,24,96)
 except RolloutOnlyComplete:pass
 else:raise AssertionError('no-update diagnostic must stop')
 data=json.loads((tmp_path/'rollout_only_complete.json').read_text())
 assert data['optimizer_updates']==0 and data['rollouts']==96

def test_disabled_does_not_stop_or_write(tmp_path):
 stop_after_rollout(False,tmp_path,1,tmp_path/'missing',24,96)
 assert not list(tmp_path.iterdir())

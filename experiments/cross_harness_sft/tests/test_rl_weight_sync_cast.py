import os
import sys
import tempfile
from pathlib import Path

import torch
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import distribute_tensor
from torch.distributed.tensor.placement_types import Shard

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "vendor" / "verl"
sys.path.insert(0, str(VENDOR))

from verl.utils.weight_sync import materialize_rollout_weight  # noqa: E402


def test_fp32_dtensor_casts_local_shard_before_materializing_full_tensor(monkeypatch):
    import torch.distributed as dist
    from torch.distributed.tensor import DTensor

    with tempfile.TemporaryDirectory() as tmp:
        init_file = Path(tmp) / "rendezvous"
        dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=0, world_size=1)
        try:
            mesh = init_device_mesh("cpu", (1,))
            source = torch.arange(12, dtype=torch.float32).reshape(3, 4)
            dtensor = distribute_tensor(source, mesh, [Shard(0)])

            events = []
            original_to = DTensor.to

            def tracked_to(self, *args, **kwargs):
                events.append((self.device.type, args, kwargs.copy()))
                return original_to(self, *args, **kwargs)

            monkeypatch.setattr(DTensor, "to", tracked_to)

            result = materialize_rollout_weight(dtensor, torch.device("cpu"), torch.bfloat16)

            assert result.dtype == torch.bfloat16
            assert torch.equal(result, source.to(torch.bfloat16))
            assert dtensor.dtype == torch.float32  # source/master tensor is untouched
            assert len(events) == 2
            assert events[0][0] == "cpu" and events[0][2].get("dtype") == torch.bfloat16
            assert events[1][0] == "cpu" and events[1][1][0] == torch.device("cpu")
        finally:
            dist.destroy_process_group()


def test_sync_dtype_env_is_opt_in_and_accepts_supported_precision(monkeypatch):
    monkeypatch.delenv("VERL_SYNC_WEIGHT_DTYPE", raising=False)
    from verl.utils.weight_sync import rollout_sync_dtype

    assert rollout_sync_dtype() is None
    monkeypatch.setenv("VERL_SYNC_WEIGHT_DTYPE", "bf16")
    assert rollout_sync_dtype() == torch.bfloat16
    monkeypatch.setenv("VERL_SYNC_WEIGHT_DTYPE", "fp16")
    assert rollout_sync_dtype() == torch.float16


def test_sync_dtype_rejects_unknown_values(monkeypatch):
    from verl.utils.weight_sync import rollout_sync_dtype

    monkeypatch.setenv("VERL_SYNC_WEIGHT_DTYPE", "int8")
    try:
        rollout_sync_dtype()
    except ValueError as exc:
        assert "VERL_SYNC_WEIGHT_DTYPE" in str(exc)
    else:
        raise AssertionError("unsupported sync dtype must fail closed")

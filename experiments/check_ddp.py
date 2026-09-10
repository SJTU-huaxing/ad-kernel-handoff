"""Two-card HCCL gradient/update equivalence at the same global batch."""
import copy
import json
import os
from pathlib import Path

import torch
import torch_npu
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from ad_kernel.model import LinearLanguageModel, ModelConfig

ROOT = Path(__file__).resolve().parents[1]
rank = int(os.environ["LOCAL_RANK"])
torch.set_num_threads(4)
torch.npu.set_device(rank)
device = f"npu:{rank}"
dist.init_process_group("hccl")
try:
    torch.manual_seed(901)
    config = ModelConfig(vocab_size=97, width=64, layers=2, heads=2, intermediate=128,
                         feature_dim=16, feature_hidden=48, backend="torch", fused_loss=False)
    model = LinearLanguageModel(config).to(device)
    reference = copy.deepcopy(model)
    distributed = DistributedDataParallel(model, device_ids=[rank], broadcast_buffers=False)
    ids = torch.randint(97, (4, 66), generator=torch.Generator().manual_seed(977)).to(device)
    local = ids[rank * 2:rank * 2 + 2]
    distributed(local[:, :-1], local[:, 1:]).backward()
    reference(ids[:, :-1], ids[:, 1:]).backward()
    error = torch.zeros((), device=device)
    energy = torch.zeros((), device=device)
    max_abs = torch.zeros((), device=device)
    for (name, actual), (_, expected) in zip(model.named_parameters(), reference.named_parameters()):
        assert actual.grad is not None and expected.grad is not None, name
        delta = actual.grad - expected.grad
        error += delta.square().sum()
        energy += expected.grad.square().sum()
        max_abs = torch.maximum(max_abs, delta.abs().max())
    relative = (error / energy).sqrt()
    dist.all_reduce(relative, op=dist.ReduceOp.MAX)
    assert relative.item() < 3e-4, relative.item()
    a = torch.optim.AdamW(model.parameters(), lr=1e-3)
    b = torch.optim.AdamW(reference.parameters(), lr=1e-3)
    a.step()
    b.step()
    update_max = max((x - y).abs().max().item() for x, y in zip(model.parameters(), reference.parameters()))
    assert update_max < 1e-5, update_max
    result = {"passed": True, "rank": rank, "world_size": dist.get_world_size(),
              "backend": "hccl", "global_batch": 4, "local_batch": 2,
              "gradient_relative_rms": relative.item(), "gradient_max_abs": max_abs.item(),
              "adamw_parameter_max_abs": update_max, "torch": str(torch.__version__)}
    (ROOT / f"work/reproduction/ddp_rank{rank}.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)
    dist.barrier()
finally:
    dist.destroy_process_group()

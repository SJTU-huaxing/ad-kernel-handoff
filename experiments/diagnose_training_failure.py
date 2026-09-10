"""Replay the saved AD state, preserving the failed formal run unchanged."""
import json
import argparse
import math
import os
from pathlib import Path
import random
import sys
from contextlib import nullcontext

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "work/pretraining/archive/protocol_v1_global_scaling"
sys.path.insert(0, str(ARCHIVE / "source/src"))
sys.path.insert(0, str(ARCHIVE / "source/experiments"))

import numpy as np
import torch
import torch_npu
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
import fla
import ad_kernel.model as model_module
from ad_kernel.model import LinearLanguageModel, ModelConfig
from pretrain import batch, cpu_tree, save_torch, digest

ap = argparse.ArgumentParser()
ap.add_argument("--output", type=Path, default=ROOT / "work/reproduction/training_failure_step348_replay")
args = ap.parse_args()
OUT = args.output
if OUT.exists() and any(OUT.iterdir()):
    raise FileExistsError("Choose a new --output directory to preserve previous diagnostic evidence")
OUT.mkdir(parents=True, exist_ok=True)
rank = int(os.environ["LOCAL_RANK"])
device = f"npu:{rank}"
torch.set_num_threads(4)
torch.npu.set_device(device)
dist.init_process_group("hccl")
box = torch.load(ARCHIVE / "runs/ad64_s11/last.pt", map_location="cpu", weights_only=True)
assert box["step"] == 256
for source, expected in box["binding"]["code_sha256"].items():
    assert digest(ARCHIVE / "source" / source) == expected
protocol = json.loads((ARCHIVE / "source/configs/pretraining_protocol_v1.json").read_text())
torch.manual_seed(11)
random.seed(11)
model = LinearLanguageModel(ModelConfig(**box["model_config"])).to(device)
model.load_state_dict(box["model"])
groups = [{"params": [p for p in model.parameters() if p.ndim >= 2], "weight_decay": .1},
          {"params": [p for p in model.parameters() if p.ndim < 2], "weight_decay": 0.}]
optimizer = torch.optim.AdamW(groups, lr=.0006, betas=(.9, .95), eps=1e-8)
optimizer.load_state_dict(box["optimizer"])
rng = box["rng_by_rank"][rank]
torch.set_rng_state(rng["torch"])
torch.npu.set_rng_state(rng["npu"], device=device)
random.setstate(rng["python"])
ddp = DistributedDataParallel(model, device_ids=[rank], broadcast_buffers=False)
stream = np.memmap(ROOT / "work/pretraining/data/fineweb_edu_v1/train.bin", mode="r", dtype="<u2")
order = torch.randperm((len(stream) - 1) // 1024, generator=torch.Generator().manual_seed(85011)).numpy()
original = model_module.chunk_attention
current_step, current_micro, layer = 0, 0, 0


def inspected(logq, logk, v, **kwargs):
    global layer
    index = layer
    layer += 1
    if current_step == 348:
        print(json.dumps({"event": "attention_inputs", "rank": rank, "micro": current_micro, "layer": index,
                          "q_min": logq.min().item(), "q_max": logq.max().item(),
                          "k_min": logk.min().item(), "k_max": logk.max().item(),
                          "k_temporal_span": (logk.amax(2) - logk.amin(2)).max().item()}), flush=True)
    out, state = original(logq, logk, v, **kwargs)
    if current_step == 348:
        def hook(gradient):
            if not torch.isfinite(gradient).all().item():
                save_torch(OUT / f"attention_rank{rank}_micro{current_micro}_layer{index}.pt",
                           {"logq": logq.detach().cpu(), "logk": logk.detach().cpu(), "v": v.detach().cpu(), "gradient": gradient.cpu()})
                print(json.dumps({"event": "nonfinite_upstream", "rank": rank, "micro": current_micro, "layer": index}), flush=True)
        out.register_hook(hook)
    return out, state


model_module.chunk_attention = inspected
try:
    for step in range(256, 348):
        current_step = step + 1
        lr = .0006 * (.1 + .9 * (1 + math.cos(math.pi * (step - 128) / (16384 - 128))) / 2)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        if current_step == 348:
            if rank == 0:
                save_torch(OUT / "before_step348.pt", {"binding": box["binding"], "model_config": box["model_config"],
                           "step": 347, "model": cpu_tree(model.state_dict()), "optimizer": cpu_tree(optimizer.state_dict())})
            dist.barrier()
        losses = []
        for micro in range(2):
            current_micro, layer = micro, 0
            first = step * 64 + (rank * 2 + micro) * 16
            x, y = batch(stream, order[first:first + 16], 1024, device)
            if current_step == 348:
                save_torch(OUT / f"batch_rank{rank}_micro{micro}.pt", {"x": x.cpu(), "y": y.cpu()})
            context = ddp.no_sync() if micro == 0 else nullcontext()
            anomaly = torch.autograd.detect_anomaly(check_nan=True) if current_step == 348 else nullcontext()
            with context, anomaly:
                with torch.autocast("npu", dtype=torch.bfloat16):
                    loss = ddp(x, y) / 2
                loss.backward()
            losses.append(loss.item())
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        if not torch.isfinite(grad).item():
            print(json.dumps({"event": "nonfinite", "rank": rank, "step": current_step,
                              "parameters": [name for name, p in model.named_parameters() if p.grad is not None and not torch.isfinite(p.grad).all().item()]}), flush=True)
            break
        optimizer.step()
        if current_step % 10 == 0:
            print(json.dumps({"event": "replay", "rank": rank, "step": current_step, "loss": sum(losses), "grad_norm": grad.item()}), flush=True)
finally:
    dist.destroy_process_group()

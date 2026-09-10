"""Extract independent (sequence, head) numerical regressions from step 348."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "work/pretraining/archive/protocol_v1_global_scaling/source/src"))
import torch
import torch_npu
import fla
import ad_kernel.model as model_module
from ad_kernel.model import LinearLanguageModel, ModelConfig

OUT = ROOT / "work/reproduction/training_failure_step348"
CAPTURE = OUT / "capture_replay"
CAPTURE.mkdir(exist_ok=True)
torch.set_num_threads(4)
torch.npu.set_device("npu:0")
box = torch.load(OUT / "before_step348.pt", map_location="cpu", weights_only=True)
model = LinearLanguageModel(ModelConfig(**box["model_config"])).to("npu:0")
model.load_state_dict(box["model"])
del box
batch = torch.load(OUT / "batch_rank1_micro1.pt", map_location="cpu", weights_only=True)
original, layer = model_module.chunk_attention, 0


def inspected(logq, logk, v, **kwargs):
    global layer
    index = layer
    layer += 1
    output, state = original(logq, logk, v, **kwargs)
    captured, saved = {}, set()
    output.register_hook(lambda dy: captured.update(dy=dy.detach()))
    def check(grad, side):
        bad = ~torch.isfinite(grad).all(-1).all(-1)
        good_upstream = torch.isfinite(captured["dy"]).all(-1).all(-1)
        for b, h in (bad & good_upstream).nonzero().cpu().tolist():
            if (b, h) in saved:
                continue
            saved.add((b, h))
            path = CAPTURE / f"regression_layer{index}_b{b}_h{h}.pt"
            if path.exists():
                raise FileExistsError(f"Preserving existing diagnostic input: {path}")
            torch.save({"logq": logq[b:b+1, h:h+1].detach().cpu(), "logk": logk[b:b+1, h:h+1].detach().cpu(),
                        "v": v[b:b+1, h:h+1].detach().cpu(), "dy": captured["dy"][b:b+1, h:h+1].cpu()}, path)
            print(json.dumps({"event": "captured", "layer": index, "batch": b, "head": h, "side": side, "path": str(path)}), flush=True)
    for tensor, side in [(logq, "q"), (logk, "k"), (v, "v")]:
        tensor.register_hook(lambda grad, side=side: check(grad, side))
    return output, state


model_module.chunk_attention = inspected
with torch.autocast("npu", dtype=torch.bfloat16):
    loss = model(batch["x"].to("npu:0"), batch["y"].to("npu:0")) / 2
print(json.dumps({"event": "forward", "loss": loss.item()}), flush=True)
loss.backward()
print("Captured first invalid attention gradients", flush=True)

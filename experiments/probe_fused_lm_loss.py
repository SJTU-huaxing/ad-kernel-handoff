"""Validate the NPU fused tied-output loss before using it for pretraining."""
import json
from pathlib import Path
import torch
import torch_npu
import fla
from fla.modules.fused_linear_cross_entropy import FusedLinearCrossEntropyLoss

ROOT = Path(__file__).resolve().parents[1]
torch.set_num_threads(4)
torch.npu.set_device("npu:0")
results = []
for n, dim, vocab in [(73, 64, 257), (256, 768, 50257)]:
    torch.manual_seed(n)
    x = torch.randn(n, dim, device="npu:0", requires_grad=True)
    w = (torch.randn(vocab, dim, device="npu:0") * .02).requires_grad_()
    xr, wr = x.detach().clone().requires_grad_(), w.detach().clone().requires_grad_()
    labels = torch.randint(vocab, (n,), device="npu:0")
    labels[::13] = -100
    with torch.autocast("npu", dtype=torch.bfloat16):
        actual = FusedLinearCrossEntropyLoss(num_chunks=4)(x, labels, w)
        reference = torch.nn.functional.cross_entropy(torch.nn.functional.linear(xr, wr).float(), labels)
    (actual * 1.7).backward()
    (reference * 1.7).backward()
    errors = [((a.grad - b.grad).square().mean().sqrt() / b.grad.square().mean().sqrt()).item() for a, b in [(x, xr), (w, wr)]]
    result = {"n": n, "dim": dim, "vocab": vocab, "loss_abs": abs(actual.item() - reference.item()),
              "gradient_relative_rms": errors}
    print(json.dumps(result), flush=True)
    assert result["loss_abs"] < 1e-4 and max(errors) < .02, result
    results.append(result)
(ROOT / "work/reproduction/fused_lm_loss.json").write_text(json.dumps({"passed": True, "results": results}, indent=2) + "\n")

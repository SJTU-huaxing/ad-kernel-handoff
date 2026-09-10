"""Verify FP32 chunk output on real cached validation activations."""
import json
import torch
import torch_npu
from assess_reproduction import load
from reproduce_features import ROOT, RUN, data, KI
from ad_kernel.attention import chunk_attention, _scaled_features

torch.set_num_threads(4)
torch.npu.set_device("npu:0")
ds = data("validation", limit=1)
results = []
for name in ["standard_ad_s11_lr0.002", "standard_exp_s11_lr0.002", "hh_exp_s11_lr0.002", "hh_softmax_s11_lr0.002", "favor_s11"]:
    net = load(name).double()
    q, k, v = ds["q"][0].double(), ds["k"][0, KI].double(), ds["v"][0, KI].double()
    qi = ds["query_positions"][0]
    with torch.inference_mode():
        logits = net.log_matrix(q, k)
        mask = torch.arange(k.shape[1])[None] <= qi[:, None]
        expected = logits.masked_fill(~mask[None], -torch.inf).softmax(-1) @ v
        net = net.float().to("npu:0")
        lq = torch.zeros(1, 24, k.shape[1], net.m, device="npu:0")
        lq[:, :, qi.to("npu:0")] = net.log_feature(q.float().to("npu:0"), "q")[None]
        lk = net.log_feature(k.float().to("npu:0"), "k")[None]
        output, _ = chunk_attention(lq, lk, v.float().to("npu:0")[None])
        actual = output[0, :, qi.to("npu:0")].cpu().double()
        qs, ks, *_ = _scaled_features(lq, lk, None, None)
        denominator = (qs * ks.cumsum(2)).sum(-1)
        error = ((actual - expected).square().mean().sqrt() / expected.square().mean().sqrt()).item()
        row = {"name": name, "relative_rms": error, "max_abs": (actual - expected).abs().max().item(),
               "zero_denominators": int((denominator <= 0).sum().item())}
        assert error < 3e-4 and row["zero_denominators"] == 0, row
        results.append(row)
        print(json.dumps(row), flush=True)
(RUN / "replacement_numerics.json").write_text(json.dumps({"passed": True, "results": results}, indent=2) + "\n")

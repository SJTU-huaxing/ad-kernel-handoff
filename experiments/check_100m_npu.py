"""Small real NPU FLA-component checks against an independent CPU FP32 model."""
from dataclasses import replace
import json
import torch
import torch_npu
import fla
from pretrain_100m.model import LanguageModel, ModelConfig
from pretrain_100m.common import ROOT, atomic_json


def main():
    torch.npu.set_device(0)
    torch.set_num_threads(4)
    out = {}
    for method in ["ad64","favor64","hedgehog","softmax"]:
        torch.manual_seed(211)
        c = ModelConfig(hidden_size=128,num_hidden_layers=2,num_heads=2,intermediate_size=256,vocab_size=256,
                        method=method,loss_chunks=2)
        actual = LanguageModel(c).npu()
        oracle = LanguageModel(replace(c, fuse_norm=False, fuse_swiglu=False, fuse_linear_cross_entropy=False))
        oracle.load_state_dict({k:v.cpu() for k,v in actual.state_dict().items()})
        x = torch.randint(0,256,(2,65))
        y = torch.randint(0,256,(2,65))
        y[1,33:] = -100
        count = int((y != -100).sum())
        with torch.autocast("npu", dtype=torch.bfloat16):
            got = actual(x.npu(), y.npu())/count
        want = oracle(x,y)/count
        got.backward();want.backward()
        squared_error = squared_ref = 0.
        for (name,p),(n,q) in zip(actual.named_parameters(),oracle.named_parameters()):
            assert name == n and p.grad is not None and q.grad is not None, name
            g = p.grad.cpu().float()
            assert torch.isfinite(g).all(), name
            squared_error += (g-q.grad).double().square().sum().item()
            squared_ref += q.grad.double().square().sum().item()
        rel = (squared_error/squared_ref)**.5
        loss_error = abs(got.item()-want.item())
        assert rel < .035 and loss_error < .015, (method,rel,loss_error)
        # A completely ignored microbatch must be finite and contribute zero
        # gradient; this occurs on an uneven final DDP update.
        actual.zero_grad(set_to_none=True)
        with torch.autocast("npu",dtype=torch.bfloat16):
            zero = actual(x.npu(), torch.full_like(y,-100).npu())
        zero.backward()
        assert zero.item() == 0.
        assert all(p.grad is not None and torch.count_nonzero(p.grad).item() == 0 for p in actual.parameters())
        out[method] = {"loss_absolute_error":loss_error,"all_parameter_gradient_relative_rms":rel,
                       "ignored_microbatch_zero_loss_and_gradients":True}
        del actual,oracle
        torch.npu.empty_cache()
    torch.manual_seed(191)
    ref = [torch.nn.Parameter(torch.randn(64,64)),torch.nn.Parameter(torch.randn(64))]
    actual = [torch.nn.Parameter(p.detach().npu()) for p in ref]
    expected_opt = torch.optim.AdamW(ref, lr=.001,betas=(.9,.95),eps=1e-8,weight_decay=.1,foreach=False)
    actual_opt = torch_npu.optim.NpuFusedAdamW(actual,lr=.001,betas=(.9,.95),eps=1e-8,weight_decay=.1)
    for _ in range(3):
        actual_opt.zero_grad(set_to_none=False)
        expected_opt.zero_grad(set_to_none=False)
        for p,q in zip(actual,ref):
            grad = torch.randn_like(q)
            if p.grad is None: p.grad=grad.npu()
            else: p.grad.copy_(grad.npu())
            q.grad=grad
        gn = actual_opt.clip_grad_norm_fused_(1.)
        rn = torch.nn.utils.clip_grad_norm_(ref,1.,foreach=False)
        assert abs(gn.item()-rn.item())/rn.item() < 1e-6
        actual_opt.step();expected_opt.step()
    max_error = max((p.cpu()-q).abs().max().item() for p,q in zip(actual,ref))
    assert max_error < 2e-6,max_error
    out["fused_adamw_against_cpu"] = {"updates":3,"max_parameter_error":max_error}
    atomic_json(ROOT/"work/pretrain_100m_2b/validation/npu_components.json",out)
    print(json.dumps(out,indent=2))


if __name__ == "__main__": main()

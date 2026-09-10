"""Record the actual optimized attention operator selected on this Ascend runtime."""
import json
import torch
import torch_npu
import fla
from torch.nn import functional as F
from pretrain_100m.common import ROOT,atomic_json

torch.npu.set_device(0)
torch.set_num_threads(4)
q,k,v=[torch.randn(4,10,2048,64,device="npu",dtype=torch.bfloat16,requires_grad=True) for _ in range(3)]
for _ in range(2):
    y=F.scaled_dot_product_attention(q,k,v,is_causal=True)
    torch.autograd.grad(y.float().square().mean(),[q,k,v])
torch.npu.synchronize()
with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
    y=F.scaled_dot_product_attention(q,k,v,is_causal=True)
    grads=torch.autograd.grad(y.float().square().mean(),[q,k,v])
    torch.npu.synchronize()
names=[r.key for r in profile.key_averages() if any(s in r.key.lower() for s in ["attention","softmax"])]
assert torch.isfinite(y).all() and all(torch.isfinite(g).all() for g in grads)
out={"shape":"B4 H10 T2048 D64 BF16","operator_names":names,"finite_forward_backward":True}
atomic_json(ROOT/"work/pretrain_100m_2b/validation/softmax_dispatch.json",out)
print(json.dumps(out,indent=2))

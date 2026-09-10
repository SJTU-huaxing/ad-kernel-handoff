"""Localize cross-chunk errors without changing the installed FLA package."""
import json
import torch
import torch_npu
import fla
from fla.ops.common.chunk_h import chunk_fwd_h
from fla.ops.common.chunk_o import chunk_fwd_o

torch.set_num_threads(4)
torch.manual_seed(921)
device = "npu:0"
torch.npu.set_device(device)
t = 65
q, k, v = [x.to(device) for x in [torch.randn(1,t,2,64).softmax(-1),
                                torch.randn(1,t,2,64).softmax(-1),torch.randn(1,t,2,64)]]
qr, kr, vr = [x.cpu().double().transpose(1,2) for x in (q,k,v)]
reference_h = torch.zeros(1,2,2,64,64,dtype=torch.float64)
reference_h[:,1] = kr[:,:,:64].transpose(-1,-2) @ vr[:,:,:64]
reference_ht = kr.transpose(-1,-2) @ vr
reference_o = ((qr @ kr.transpose(-1,-2)).tril() @ vr).transpose(1,2)
h, ht = chunk_fwd_h(k,v,output_final_state=True)
print(json.dumps({"event":"states","h_max_abs":(h.cpu().double()-reference_h).abs().amax((0,2,3,4)).tolist(),
                  "ht_max_abs":(ht.cpu().double()-reference_ht).abs().max().item(),
                  "actual_second_state_max_abs":h[:,1].abs().max().item()}),flush=True)
for name, states in [("fla",h),("reference",reference_h.float().to(device))]:
    o = chunk_fwd_o(q=q,k=k,v=v,h=states,scale=1.)
    print(json.dumps({"event":"output","states":name,
                      "per_token_max_abs":(o.cpu().double()-reference_o).abs().amax((0,2,3)).tolist()}),flush=True)

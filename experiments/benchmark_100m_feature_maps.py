"""AD versus the exact FLA Hedgehog log feature map used in the new protocol."""
from dataclasses import replace
import json
import statistics
import numpy as np
import torch
import torch_npu
import fla
from pretrain_100m.common import ROOT, DEFAULT_CONFIG, atomic_json
from pretrain_100m.model import LanguageModel, ModelConfig, FLAHedgehogLogFeatures
from pretrain_100m.data import TokenPlan


def measure(fn, samples=15):
    for _ in range(5): fn()
    torch.npu.synchronize()
    times = []
    for _ in range(samples):
        a,b = [torch.npu.Event(enable_timing=True) for _ in range(2)]
        a.record();fn();b.record();b.synchronize()
        times.append(a.elapsed_time(b))
    return {"median_ms":statistics.median(times),"samples_ms":times}


def main():
    torch.npu.set_device(0)
    torch.npu.set_compile_mode(jit_compile=False)
    torch.set_num_threads(4)
    c = json.loads(DEFAULT_CONFIG.read_text())
    torch.manual_seed(11)
    model = LanguageModel(ModelConfig(**c["architecture"], method="ad64")).npu()
    plan = TokenPlan(2000000000,2048,c["global_batch_sequences"],2,16,85011)
    stream = np.memmap(ROOT/c["data"]["directory"]/"train.bin",dtype="<u2",mode="r")
    tokens = plan.batch(stream,0,0).inputs[0].npu()
    with torch.no_grad(),torch.autocast("npu",dtype=torch.bfloat16):
        x = model.layers[0].attn_norm(model.embedding(tokens))
        q,k,_ = model.layers[0].attn.qkv(x).reshape(16,2048,3,10,64).permute(2,0,3,1,4).unbind(0)
        a = torch.arange(2048,device="npu").float()[:,None]*model.inv_freq
        phase = torch.cat([a,a],-1)[None,None]
        def rotate(x):
            x=x.float();first,second=x.chunk(2,-1)
            return (x*phase.cos()+torch.cat([-second,first],-1)*phase.sin())*64**(-.25)
        q,k = rotate(q).detach().requires_grad_(),rotate(k).detach().requires_grad_()
    maps = {"ad64":model.layers[0].attn.features,"fla_hedgehog":FLAHedgehogLogFeatures(64).npu()}
    rows = []
    for round_index, order in enumerate([list(maps),list(reversed(maps))]):
        for method in order:
            feature = maps[method]
            params = [q,k,*feature.parameters()]
            def forward():
                with torch.no_grad():
                    return feature.log_feature(q,"q"),feature.log_feature(k,"k")
            m = 64 if method == "ad64" else 128
            wq,wk = [torch.randn(16,10,2048,m,device="npu") for _ in range(2)]
            def backward():
                lq,lk=feature.log_feature(q,"q"),feature.log_feature(k,"k")
                return torch.autograd.grad((lq*wq).sum()+(lk*wk).sum(),params)
            rows.append({"round":round_index,"method":method,"forward":measure(forward),"forward_backward":measure(backward)})
    summary = {m:{stage:statistics.mean(r[stage]["median_ms"] for r in rows if r["method"]==m)
                  for stage in ["forward","forward_backward"]} for m in maps}
    atomic_json(ROOT/"work/pretrain_100m_2b/validation/feature_map_speed.json",
                {"shape":"B16 H10 T2048 D64; FP32; paired Q/K", "input_source":"Same real-data first-layer RoPE Q/K from fresh seed11 100M backbone; input preparation excluded",
                 "scope":"Only feature maps; forward-backward includes Q/K and all feature parameters. Not full attention, optimizer or model time.",
                 "rows":rows,"summary_ms":summary})
    print(json.dumps(summary,indent=2))


if __name__ == "__main__":main()

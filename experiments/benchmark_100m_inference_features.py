"""Stateless Q/K feature-map inference microbenchmark for the 100M protocol."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
import torch_npu
import fla
from pretrain_100m.common import ROOT, DEFAULT_CONFIG, atomic_json, digest
from pretrain_100m.model import LanguageModel, ModelConfig, FLAHedgehogLogFeatures
from benchmark_100m_inference import timed, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT/"work/pretrain_100m_2b/inference_ad_vs_hedgehog/features.json")
    args = parser.parse_args()
    if args.out.exists(): raise FileExistsError(args.out)
    torch.set_num_threads(4)
    torch.npu.set_device(0)
    torch.npu.set_compile_mode(jit_compile=False)
    config = json.loads(DEFAULT_CONFIG.read_text())
    torch.manual_seed(11)
    model = LanguageModel(ModelConfig(**config["architecture"], method="ad64")).npu().eval()
    stream = np.memmap(ROOT/config["data"]["directory"]/"test.bin", dtype="<u2", mode="r")
    tokens = torch.tensor(np.stack([np.asarray(stream[i*8193:i*8193+2048], dtype=np.int64) for i in range(8)]), device="npu")
    maps = {"ad64": model.layers[0].attn.features, "hedgehog": FLAHedgehogLogFeatures(64).npu().eval()}
    rows = []
    with torch.inference_mode():
        for batch in [1, 8]:
            for length in [1, 2048]:
                with torch.autocast("npu", dtype=torch.bfloat16):
                    x = model.layers[0].attn_norm(model.embedding(tokens[:batch, :length]))
                    q, k, _ = model.layers[0].attn.qkv(x).reshape(batch, length, 3, 10, 64).permute(2, 0, 3, 1, 4).unbind(0)
                angles = torch.arange(length, device="npu").float()[:, None]*model.inv_freq
                phase = torch.cat((angles, angles), -1)[None, None]
                def rotate(x):
                    x = x.float()
                    first, second = x.chunk(2, -1)
                    return (x*phase.cos()+torch.cat((-second, first), -1)*phase.sin())*64**(-.25)
                q, k = rotate(q), rotate(k)
                for round_index, order in enumerate([list(maps), list(reversed(maps))]):
                    for method in order:
                        feature = maps[method]
                        def forward(): return feature.log_feature(q, "q"), feature.log_feature(k, "k")
                        for _ in range(5): forward()
                        for execution in ["eager", "graph"]:
                            if execution == "graph":
                                graph = torch.npu.NPUGraph()
                                with torch.npu.graph(graph): outputs = forward()
                                call = graph.replay
                            else: call = forward
                            for _ in range(3): call()
                            times = [timed(call)[1] for _ in range(21)]
                            row = {"batch": batch, "length": length, "method": method, "round": round_index,
                                   "execution": execution, **summary(times, batch*length)}
                            rows.append(row)
                            print(json.dumps({k:v for k,v in row.items() if k != "samples"}), flush=True)
                            if execution == "graph": del call, graph, outputs
    atomic_json(args.out, {"created_utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()), "complete":True,
                         "source_sha256": {str(Path(__file__).relative_to(ROOT)):digest(__file__),
                                           "experiments/benchmark_100m_inference.py":digest(ROOT/"experiments/benchmark_100m_inference.py")},
                         "protocol_sha256":digest(DEFAULT_CONFIG),
                         "scope":"Same real held-out first-layer Q/K, H10 D64, FP32 log feature maps only. AD64 versus exact official joint-softmax Hedgehog128 log adapter. QKV projection/RoPE/input preparation/state update/full model excluded. No backward. Two opposite orders, 21 samples each. Graph creation excluded.",
                         "rows":rows})


if __name__ == "__main__": main()

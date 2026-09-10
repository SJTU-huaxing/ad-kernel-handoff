"""Independent invariants for full-budget scheduling and the four kernels."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from pretrain_100m.common import ROOT, atomic_json, DEFAULT_CONFIG
from pretrain_100m.data import TokenPlan
from pretrain_100m.model import LanguageModel, ModelConfig, FLAHedgehogLogFeatures
from pretrain_100m.train import learning_rate
from ad_kernel.attention import chunk_attention, dense_attention


def main():
    torch.set_num_threads(4)
    results = {}
    # Independently identify every labeled source position, including all-padding
    # rank/micro batches and a shuffled final partial source block.
    for n in [1, 65, 4097, 4096, 100003]:
        stream = np.arange(n+1, dtype=np.int64)
        for micro in [1, 2, 4]:
            plan = TokenPlan(n, 128, 16, 2, micro, 11)
            observed = []
            total = 0
            for step in range(plan.steps):
                step_count = 0
                for rank in range(2):
                    batch = plan.batch(stream, step, rank)
                    y = batch.labels.numpy()
                    valid = y != -100
                    assert np.all(y[valid] == batch.inputs.numpy()[valid]+1)
                    observed.extend(y[valid].tolist())
                    step_count += valid.sum()
                assert step_count == plan.counts(step)
                total += step_count
                assert total == plan.tokens_before(step+1)
            assert sorted(observed) == list(range(1,n+1)), (n, micro)
    results["exact_token_coverage"] = "15 cases; each target position exactly once, no padded loss, exact per-step counts"
    c = json.loads(DEFAULT_CONFIG.read_text())
    for seed in [11,29]:
        p = TokenPlan(2000000000, c["sequence_length"], c["global_batch_sequences"], 2, c["micro_batch"]["ad64"], 85000+seed)
        assert p.steps == 5087 and p.tokens_before(p.steps) == 2000000000
        assert len(np.unique(p.order)) == p.blocks
    assert abs(learning_rate(c,100000000)-.001) < 1e-12
    assert abs(learning_rate(c,2000000000)-.0001) < 1e-12
    results["full_2b_budget_and_scheduler"] = "5087 updates, exactly 2B labels, 100M-token warmup, final LR 1e-4"
    # Verify official Hedgehog value AND input/parameter gradients under a
    # nonuniform scalar objective; the log adapter must preserve its factor 2.
    torch.manual_seed(123)
    feature = FLAHedgehogLogFeatures(8).double()
    x = torch.randn(2,3,9,8,dtype=torch.float64,requires_grad=True)
    weight = torch.randn(2,3,9,16,dtype=torch.float64)
    for side in ["q","k"]:
        official = getattr(feature,side+"_map")(x)
        actual = feature.log_feature(x,side).exp()
        assert torch.allclose(actual, official, atol=1e-14,rtol=1e-12)
        params = [x,*getattr(feature,side+"_map").parameters()]
        ga = torch.autograd.grad((actual*weight).sum(), params, retain_graph=True)
        ge = torch.autograd.grad((official*weight).sum(), params)
        assert all(torch.allclose(a,e,atol=1e-13,rtol=1e-11) for a,e in zip(ga,ge))
    results["official_hedgehog_equivalence"] = "FP64 values and input/weight/bias gradients pass for Q and K"
    # Independent dense causal oracle for every trainable map and FAVOR64.
    errors = {}
    for method in ["ad64","favor64","hedgehog"]:
        config = ModelConfig(hidden_size=16,num_hidden_layers=1,num_heads=2,intermediate_size=32,
                             vocab_size=64,method=method,fuse_norm=False,fuse_swiglu=False)
        torch.manual_seed(11)
        model = LanguageModel(config).double()
        f = model.layers[0].attn.features
        q,k,v = [torch.randn(2,2,67,8,dtype=torch.float64,requires_grad=True) for _ in range(3)]
        lq,lk = f.log_feature(q,"q"),f.log_feature(k,"k")
        got,_ = chunk_attention(lq,lk,v,chunk_size=16,backend="torch")
        want = dense_attention(lq,lk,v)
        w = torch.randn_like(want)
        params = [q,k,v,*f.parameters()]
        gg = torch.autograd.grad((got*w).sum(),params,retain_graph=True)
        gw = torch.autograd.grad((want*w).sum(),params)
        assert torch.allclose(got,want,rtol=1e-10,atol=1e-12),method
        assert all(torch.allclose(a,b,rtol=1e-9,atol=1e-11) for a,b in zip(gg,gw)),method
        errors[method] = max((a-b).abs().max().item() for a,b in zip(gg,gw))
    results["dense_attention_gradient_oracle_max_abs"] = errors
    # All methods must have exactly the same random backbone initialization.
    backbone, budgets = [], {}
    for method in c["methods"]:
        torch.manual_seed(11)
        model = LanguageModel(ModelConfig(**c["architecture"],method=method))
        h = hashlib.sha256()
        for name,tensor in model.state_dict().items():
            if ".features." not in name:
                h.update(name.encode());h.update(tensor.numpy().tobytes())
        backbone.append(h.hexdigest())
        budgets[method] = model.parameter_budget()
        del model
    assert len(set(backbone)) == 1
    results["shared_backbone_sha256"] = backbone[0]
    results["parameters"] = budgets
    atomic_json(ROOT/"work/pretrain_100m_2b/validation/contracts.json",results)
    print(json.dumps(results,indent=2))


if __name__ == "__main__": main()

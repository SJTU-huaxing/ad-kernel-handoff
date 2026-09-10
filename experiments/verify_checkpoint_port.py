"""Compare the new package to the archived implementations and real checkpoints."""
import copy
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import sys

import torch
import torch_npu
from ad_kernel.features import ADFeatures, load_historical_features
from ad_kernel.attention import chunk_attention, dense_attention

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / "work/kan_attention_theory"
OUT = ROOT / "work/reproduction/checkpoint_port"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    sys.path.insert(0, str(OLD / "key_parameterization_attribution"))
    original = importlib.import_module("core_attribution")
    # Re-execute original algebra/initialization checks, directing their fresh
    # output to a new directory instead of reporting archived JSON as a run.
    for name in ["proof_checks", "mechanism_checks"]:
        path = OLD / "key_parameterization_attribution" / (name + ".py")
        spec = importlib.util.spec_from_file_location("ad_original_" + name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.P = OUT / name
        (module.P / "checks").mkdir(parents=True, exist_ok=True)
        module.main()
    rows = []
    paths = sorted((ROOT / "checkpoints/kan_attention_theory/query_amplitude_ablation/fits").glob("*.pt"))
    paths += sorted((ROOT / "checkpoints/kan_attention_theory/key_parameterization_attribution/fits").glob("*.pt"))
    for path in paths:
        net, meta = load_historical_features(path)
        source, _ = original.load_fit(path.stem, dtype=torch.float64, device="cpu")
        source.runtime_raw = True
        assert net.parameters_per_head == 73536
        assert all(torch.equal(net.state_dict()[k], v) for k, v in source.state_dict().items())
        torch.manual_seed(912)
        q, k = [getattr(net, s+"_mean")[:, None] + getattr(net, s+"_std")[:, None] *
                torch.randn(24, 9, 128, dtype=torch.float64) for s in ["q", "k"]]
        errors = {s: (net.raw_log_feature(x,s)-source.raw_log_feature(x,s)).abs().max().item()
                  for s,x in [("q",q),("k",k)]}
        assert max(errors.values()) == 0, errors
        a, b = net.log_matrix(q,k), source.log_matrix(q,k)
        torch.testing.assert_close(a,b,atol=0,rtol=0)
        upstream = torch.randn_like(a)
        ga = torch.autograd.grad(a, tuple(net.parameters()), upstream)
        gb = torch.autograd.grad(b, tuple(source.parameters()), upstream)
        assert all(torch.equal(x,y) for x,y in zip(ga,gb))
        npu = copy.deepcopy(net).float().to("npu:1")
        npq, npk = q.float().to("npu:1"), k.float().to("npu:1")
        npu_feature_errors = {s: (npu.raw_log_feature(x,s).cpu().double()-net.raw_log_feature(ref,s)).abs().max().item()
                              for s,x,ref in [("q",npq,q),("k",npk,k)]}
        assert max(npu_feature_errors.values()) < 4e-4, npu_feature_errors
        values = torch.randn(1,24,9,128,dtype=torch.float64)
        logq, logk = net.raw_log_feature(q,"q")[None], net.raw_log_feature(k,"k")[None]
        output = dense_attention(logq,logk,values)
        nq, nk = npu.raw_log_feature(npq,"q")[None],npu.raw_log_feature(npk,"k")[None]
        noutput,_ = chunk_attention(nq,nk,values.float().to("npu:1"),chunk_size=8)
        torch.testing.assert_close(noutput.cpu().double(),output,rtol=3e-4,atol=3e-4)
        dout = torch.randn_like(output)
        cpu_grads = torch.autograd.grad(output,tuple(net.parameters()),dout)
        npu_grads = torch.autograd.grad(noutput,tuple(npu.parameters()),dout.float().to("npu:1"))
        grad_errors = {}
        for (name,_), actual, ref in zip(net.named_parameters(),npu_grads,cpu_grads):
            rms = (actual.cpu().double()-ref).square().mean().sqrt()
            relative = (rms/ref.square().mean().sqrt().clamp_min(1e-20)).item()
            grad_errors[name]=relative
            assert relative < 2e-3, (name,relative)
        row = {"checkpoint":str(path.relative_to(ROOT)),"sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
               "parameters_per_head":net.parameters_per_head,"source_fp64_max_error":errors,
               "npu_fp32_log_feature_max_error":npu_feature_errors,
               "npu_fp32_parameter_gradient_relative_rms":grad_errors,
               "npu_fp32_output_max_error":(noutput.cpu().double()-output).abs().max().item()}
        rows.append(row)
        print(json.dumps(row),flush=True)
    result = {"passed":True,"checkpoints":rows,
              "original_check_sources": {name: hashlib.sha256((OLD / "key_parameterization_attribution" / (name+".py")).read_bytes()).hexdigest()
                                         for name in ["proof_checks", "mechanism_checks"]},
              "scope":"Source checkpoint function/gradient and NPU port checks on synthetic inputs using saved norms; not real-QKV metric reproduction",
              "thresholds":{"npu_logfeature_abs":4e-4,"npu_output_abs_rtol":3e-4,"npu_gradient_relative_rms":2e-3}}
    (OUT/"results.json").write_text(json.dumps(result,indent=2)+"\n")
    print("CHECKPOINT PORT PASSED",len(rows),flush=True)


if __name__ == "__main__":
    main()

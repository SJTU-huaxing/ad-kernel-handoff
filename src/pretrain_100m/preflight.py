"""Read-only input/environment/recipe verification before any production updates."""
import argparse
import json
from pathlib import Path
import importlib.metadata as md
import torch
import torch_npu
import fla
from .common import ROOT, DEFAULT_CONFIG, digest, source_binding
from .model import ModelConfig, LanguageModel
from .data import TokenPlan


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = p.parse_args()
    c = json.loads(args.config.read_text())
    assert c["target_tokens_per_run"] == 2000000000
    assert set(c["methods"]) == {"ad64", "softmax", "favor64", "hedgehog"}
    assert c["world_size"] == torch.npu.device_count() == 2
    versions = {name: md.version(name) for name in ["torch", "torch-npu", "fla-core", "flash-linear-attention", "triton-ascend"]}
    expected = {"torch": "2.7.1", "torch-npu": "2.7.1.post4", "fla-core": "0.5.2", "flash-linear-attention": "0.5.2", "triton-ascend": "3.2.1"}
    for name, version in expected.items():
        assert versions[name].split("+")[0] == version, f"Revalidate changed runtime: {name}={versions[name]}"
    data = ROOT/c["data"]["directory"]
    assert digest(data/"manifest.json") == c["data"]["manifest_sha256"]
    manifest = json.loads((data/"manifest.json").read_text())
    for name in ["train.bin", "validation.bin", "test.bin"]:
        assert digest(data/name) == manifest["artifacts"][name]["sha256"], name
    assert (data/"train.bin").stat().st_size//2 > c["target_tokens_per_run"]
    for folder in ["long_v1", "transfer_recall_v1"]:
        assert (ROOT/"work/pretraining/evaluation"/folder/"manifest.json").exists()
    torch.set_num_threads(4)
    budgets = {}
    for method in c["methods"]:
        model = LanguageModel(ModelConfig(**c["architecture"], method=method, feature_seed=c["seeds"][0]))
        budgets[method] = model.parameter_budget()["total"]
        plan = TokenPlan(c["target_tokens_per_run"], c["sequence_length"], c["global_batch_sequences"],
                         2, c["micro_batch"][method], c["data"]["order_seed_offset"]+c["seeds"][0])
        assert plan.tokens_before(plan.steps) == 2000000000
        del model
    print(json.dumps({"preflight": "passed", "versions": versions, "parameters": budgets,
                      "steps_per_run": plan.steps, "exact_tokens_per_run": 2000000000,
                      "source_sha256": source_binding()}, indent=2))


if __name__ == "__main__": main()

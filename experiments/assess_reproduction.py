"""Frozen 1k/8k confirmation metrics and all-token local-replacement PPL."""
import argparse
import copy
import json
import math
from pathlib import Path
import time

import torch
import torch_npu
from transformers import AutoModelForCausalLM
from transformers.models.qwen2 import modeling_qwen2
from ad_kernel.features import ADFeatures
from ad_kernel.baselines import HedgehogFeatures, FavorFeatures
from ad_kernel.attention import chunk_attention
from reproduce_features import ROOT, RUN, data, digest, evaluate, save_json


def freeze():
    path = RUN / "frozen_evaluation.json"
    models = []
    for kind in ["ad", "exp"]:
        selection = json.loads((RUN / f"selection_{kind}.json").read_text())
        for initialization in ["standard", "matched_zero"]:
            lr = selection["selected"][initialization]["lr"]
            models += [f"{initialization}_{kind}_s{s}_lr{lr:g}" for s in [11, 29, 47]]
    for kind in ["hh_exp", "hh_softmax"]:
        lr = json.loads((RUN / f"selection_{kind}.json").read_text())["lr"]
        models += [f"{kind}_s{s}_lr{lr:g}" for s in [11, 29, 47]]
    models += [f"favor_s{s}" for s in [11, 29, 47]]
    checkpoints = {}
    for name in models:
        report = json.loads((RUN / "fits" / f"{name}.json").read_text())
        h = digest(RUN / "fits" / f"{name}.pt")
        assert report["checkpoint_sha256"] == h
        checkpoints[name] = h
    result = {"models": models, "checkpoint_sha256": checkpoints,
              "manifest_sha256": digest(RUN / "data/manifest.json"),
              "protocol_sha256": digest(ROOT / "configs/reproduction_public_v1.json"),
              "splits": ["confirm_wiki", "confirm_long"],
              "static": "CPU FP64, cached teacher post-RoPE Q/K/V",
              "ppl": "Qwen BF16 SDPA, two complete layers replaced with FP32 normalized chunk attention, all next tokens",
              "scope": "Fresh public-data confirmation; 24/336 heads replaced, no teacher finetuning"}
    if path.exists():
        assert json.loads(path.read_text()) == result
    else:
        save_json(path, result)
    print(json.dumps({"event": "frozen", "models": len(models), "sha256": digest(path)}), flush=True)


def load(name):
    # These locally generated, hash-bound files store torch.__version__, a
    # TorchVersion string subclass. Allow precisely that class, keeping the
    # weights-only loader for everything else.
    with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
        box = torch.load(RUN / "fits" / f"{name}.pt", weights_only=True, map_location="cpu")
    state, meta = box["state_dict"], box["metadata"]
    norm = {key: state[key] for key in ["q_mean", "q_std", "k_mean", "k_std"]}
    if name.startswith("favor"):
        net = FavorFeatures(norm=norm)
    elif name.startswith("hh_"):
        net = HedgehogFeatures(norm=norm, softmax=meta["kind"] == "hh_softmax")
    else:
        net = ADFeatures(norm=norm, kind=meta["map_kind"], initialization=meta["initialization"],
                         numerical_scale=state["numerical_scale"])
    net.load_state_dict(state)
    return net.eval()


def layer_slice(net, start, device):
    part = copy.deepcopy(net)
    for module in part.modules():
        for name, parameter in list(module.named_parameters(recurse=False)):
            setattr(module, name, torch.nn.Parameter(parameter[start:start + 12].detach().clone(), requires_grad=False))
        for name, buffer in list(module.named_buffers(recurse=False)):
            setattr(module, name, buffer[start:start + 12].clone())
    part.heads = 12
    return part.float().to(device)


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["freeze", "evaluate"])
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()
    torch.set_num_threads(4)
    if args.action == "freeze":
        freeze()
        return
    plan_path = RUN / "frozen_evaluation.json"
    plan = json.loads(plan_path.read_text())
    assert json.loads((RUN / "replacement_numerics.json").read_text())["passed"]
    assert digest(RUN / "data/manifest.json") == plan["manifest_sha256"]
    binding = {"plan_sha256": digest(plan_path), "assessment_source_sha256": digest(Path(__file__)),
               "attention_source_sha256": digest(ROOT / "src/ad_kernel/attention.py")}
    directory = RUN / "confirmation" / binding["assessment_source_sha256"][:12]
    directory.mkdir(parents=True, exist_ok=True)
    device = f"npu:{args.rank}"
    torch.npu.set_device(device)
    manifest = json.loads((RUN / "data/manifest.json").read_text())
    records = {split: [r for r in manifest["records"] if r["split"] == split] for split in plan["splits"]}
    datasets = {split: data(split) for split in plan["splits"]}
    model = AutoModelForCausalLM.from_pretrained(ROOT / "work/models/Qwen2.5-1.5B", local_files_only=True,
        torch_dtype=torch.bfloat16, attn_implementation="sdpa").to(device).eval()
    original = modeling_qwen2.ALL_ATTENTION_FUNCTIONS["sdpa"]
    maps = {}
    verified_masks = set()

    def replacement(module, q, k, v, mask, **kwargs):
        if module.layer_idx not in maps:
            return original(module, q, k, v, mask, **kwargs)
        if mask is not None:
            signature = (tuple(mask.shape), mask.dtype)
            if signature not in verified_masks:
                length = q.shape[-2]
                assert mask.shape[-2:] == (length, length)
                allowed = mask if mask.dtype == torch.bool else mask == 0
                expected = torch.ones(length, length, dtype=torch.bool, device=device).tril()
                assert torch.equal(allowed, expected.expand_as(allowed)), "Expected an ordinary inclusive causal mask"
                if mask.dtype != torch.bool:
                    assert (mask.masked_select(~allowed) < -1e4).all().item(), "Unsupported additive attention bias"
                verified_masks.add(signature)
        net = maps[module.layer_idx]
        k, v = k.repeat_interleave(6, 1).float(), v.repeat_interleave(6, 1).float()
        lq, lk = net.log_feature(q.float(), "q"), net.log_feature(k, "k")
        out, _ = chunk_attention(lq, lk, v, backend="torch")
        assert torch.isfinite(out).all().item(), "Nonfinite replacement output: investigate scaling, do not clamp"
        return out.transpose(1, 2).to(q.dtype), None

    modeling_qwen2.ALL_ATTENTION_FUNCTIONS.register("sdpa", replacement)
    try:
        names = ["teacher"] + plan["models"][args.rank::args.workers]
        for name in names:
            if name != "teacher":
                assert digest(RUN / "fits" / f"{name}.pt") == plan["checkpoint_sha256"][name]
                net = load(name)
                maps = {14: layer_slice(net, 0, device), 27: layer_slice(net, 12, device)}
            else:
                maps = {}
                net = None
            for split in plan["splits"]:
                suffix = f"_rank{args.rank}" if name == "teacher" else ""
                destination = directory / f"{name}_{split}{suffix}.json"
                if destination.exists():
                    assert json.loads(destination.read_text())["binding"] == binding
                    continue
                start = time.perf_counter()
                static = None if net is None else evaluate(net, datasets[split])
                rows = []
                for ordinal, row in enumerate(records[split]):
                    ids = torch.tensor([row["input_ids"]], device=device)
                    hidden = model.model(input_ids=ids, use_cache=False).last_hidden_state
                    nll = 0.
                    for begin in range(0, ids.shape[1] - 1, 256):
                        end = min(begin + 256, ids.shape[1] - 1)
                        logits = model.lm_head(hidden[:, begin:end]).float()
                        nll += torch.nn.functional.cross_entropy(logits.flatten(0, 1), ids[:, begin + 1:end + 1].flatten(), reduction="sum").item()
                    rows.append({"ordinal": ordinal, "text_sha256": row["text_sha256"],
                                 "tokens": ids.shape[1] - 1, "nll_sum": nll})
                    if (ordinal + 1) % 32 == 0:
                        print(json.dumps({"event": "ppl", "model": name, "split": split,
                                          "documents": ordinal + 1, "seconds": time.perf_counter() - start}), flush=True)
                count = sum(row["tokens"] for row in rows)
                nll = sum(row["nll_sum"] for row in rows) / count
                result = {"model": name, "split": split, "binding": binding, "static": static,
                          "ppl": {"nll_per_token": nll, "perplexity": math.exp(nll), "tokens": count, "documents": rows},
                          "device": device, "seconds": time.perf_counter() - start,
                          "timing_scope": "Quality run includes CPU metrics and host transfers; not an efficiency benchmark"}
                save_json(destination, result)
                print(json.dumps({"event": "evaluated", "model": name, "split": split,
                                  "ppl": math.exp(nll), "seconds": result["seconds"]}), flush=True)
    finally:
        modeling_qwen2.ALL_ATTENTION_FUNCTIONS.register("sdpa", original)


if __name__ == "__main__":
    main()

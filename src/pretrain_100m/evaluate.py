"""Fixed final-checkpoint evaluation, split over both NPUs without optimizer updates."""
import argparse
from datetime import timedelta
import json
import math
import os
from pathlib import Path
import time
import numpy as np
import torch
import torch.distributed as dist
import torch_npu
import fla
from torch.nn import functional as F
from .common import ROOT, digest, atomic_json, load_checkpoint
from .model import LanguageModel, ModelConfig


@torch.inference_mode()
def token_nll(model, inputs, labels):
    with torch.autocast("npu", dtype=torch.bfloat16):
        hidden = model.hidden_states(inputs).flatten(0, 1)
        labels_flat = labels.flatten()
        pieces = []
        for i in range(0, len(hidden), 256):
            logits = F.linear(hidden[i:i+256], model.embedding.weight).float()
            pieces.append(F.cross_entropy(logits, labels_flat[i:i+256], reduction="none", ignore_index=-100).cpu().double())
    return torch.cat(pieces).reshape(labels.shape)


def totals(records):
    count = sum(x["tokens"] for x in records)
    loss = sum(x["nll_sum"] for x in records)
    return {"tokens": count, "nll_sum": loss, "nll": loss/count, "ppl": math.exp(loss/count)} if count else {}


def ids(array, device):
    return torch.from_numpy(np.array(array, dtype=np.int64, copy=True)).to(device)


def packed(model, path, seq, device, rank, world, micro, limit):
    stream = np.memmap(path, dtype="<u2", mode="r")
    count = math.ceil((len(stream)-1)/seq)
    if limit: count = min(count, limit)
    indices = list(range(rank, count, world))
    records = []
    for begin in range(0, len(indices), micro):
        subset = indices[begin:begin+micro]
        x = np.zeros((len(subset), seq), np.int64)
        y = np.full_like(x, -100)
        lengths = []
        for row, index in enumerate(subset):
            value = stream[index*seq:min((index+1)*seq+1, len(stream))]
            n = len(value)-1
            x[row, :n], y[row, :n] = value[:-1], value[1:]
            lengths.append(n)
        losses = token_nll(model, ids(x, device), ids(y, device))
        for index, length, loss in zip(subset, lengths, losses):
            records.append({"index": index, "tokens": length, "nll_sum": loss[:length].sum().item()})
    return records


def wiki(model, seq, device, rank, world, limit):
    base = ROOT/"work/pretraining/evaluation/transfer_recall_v1"
    stream = np.memmap(base/"wiki.bin", dtype="<u2", mode="r")
    docs = json.loads((base/"wiki.documents.json").read_text())
    if limit: docs = docs[:limit]
    records = []
    for index in range(rank, len(docs), world):
        doc = docs[index]
        a = stream[doc["offset"]:doc["offset"]+doc["tokens"]]
        total = 0.
        for begin in range(0, len(a)-1, seq):
            x = ids(a[begin:begin+seq+1][None], device)
            total += token_nll(model, x[:, :-1], x[:, 1:]).sum().item()
        records.append({"index": index, "source_row": doc["source_row"], "tokens": len(a)-1, "nll_sum": total})
    return records


def long_docs(model, lengths, device, rank, world, limit):
    a = np.load(ROOT/"work/pretraining/evaluation/long_v1/tokens.npy", mmap_mode="r", allow_pickle=False)
    count = min(len(a), limit) if limit else len(a)
    records = []
    for index in range(rank, count, world):
        for length in lengths:
            x = ids(a[index:index+1, :length+1], device)
            loss = token_nll(model, x[:, :-1], x[:, 1:])[0]
            row = {"index": index, "length": length, "tokens": length, "nll_sum": loss.sum().item()}
            if length == 8192:
                row["bands"] = [{"start": l, "end": r, "tokens": r-l, "nll_sum": loss[l:r].sum().item()}
                                for l, r in [(0,2048), (2048,4096), (4096,8192)]]
            records.append(row)
    return records


@torch.inference_mode()
def recall(model, lengths, device, rank, world, limit):
    base = ROOT/"work/pretraining/evaluation/transfer_recall_v1"
    a = np.load(base/"recall.npz", allow_pickle=False)
    cases = json.loads((base/"recall.cases.json").read_text())
    count = min(len(cases), limit) if limit else len(cases)
    records = []
    for index in range(rank, count, world):
        case = cases[index]
        for length in lengths:
            x = ids(a[str(length)][index:index+1], device)
            with torch.autocast("npu", dtype=torch.bfloat16):
                hidden = model.hidden_states(x)[:, -1]
                logits = F.linear(hidden, model.embedding.weight).float()[0]
            log_probs = logits.log_softmax(-1).cpu()
            choices = log_probs[case["candidate_token_ids"]]
            gold = case["gold_candidate_index"]
            maxima = choices == choices.max()
            records.append({"index": index, "length": length, "pairs": case["pairs"],
                            "target_position_group": case["target_position_group"],
                            "candidate_accuracy": float(maxima[gold])/int(maxima.sum()),
                            "unrestricted_accuracy": int(log_probs.argmax().item() == case["gold_token_id"]),
                            "gold_log_probability": float(log_probs[case["gold_token_id"]]),
                            "chance": 1/case["pairs"]})
    return records


def summarize(stage, rows):
    if stage == "long":
        return {str(length): totals([r for r in rows if r["length"] == length]) for length in sorted(set(r["length"] for r in rows))}
    if stage == "recall":
        result = {}
        for length, pairs in sorted(set((r["length"], r["pairs"]) for r in rows)):
            group = [r for r in rows if r["length"] == length and r["pairs"] == pairs]
            result[f"length{length}_pairs{pairs}"] = {"cases": len(group), **{
                name: sum(r[name] for r in group)/len(group)
                for name in ["candidate_accuracy", "unrestricted_accuracy", "gold_log_probability", "chance"]}}
        return result
    return totals(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("--allow-probe", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="Only allowed for probe checkpoints")
    ap.add_argument("--micro-batch", type=int, default=4)
    ap.add_argument("--stages", nargs="+", choices=["validation", "test", "wiki", "long", "recall"])
    args = ap.parse_args()
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    local = int(os.environ["LOCAL_RANK"])
    assert world == 2
    torch.npu.set_device(local)
    torch.npu.set_compile_mode(jit_compile=False)
    torch.set_num_threads(4)
    device = torch.device(f"npu:{local}")
    dist.init_process_group("hccl", timeout=timedelta(minutes=20))
    box = load_checkpoint(args.checkpoint)
    b = box["binding"]
    c = b["protocol"]
    if b["probe"]:
        assert args.allow_probe and args.limit > 0, "Probe evaluation must be explicitly limited"
    else:
        assert not args.allow_probe and not args.limit
        assert box["tokens"] == c["target_tokens_per_run"] == 2000000000, "Only full-budget final checkpoints"
    for name, sha in b["source_sha256"].items():
        assert digest(ROOT/name) == sha, f"Frozen source changed: {name}"
    data = ROOT/c["data"]["directory"]
    assert digest(data/"manifest.json") == b["data_manifest_sha256"]
    manifest = json.loads((data/"manifest.json").read_text())
    # Verify every evaluation input against its recorded manifest.
    evaluation_hashes = {}
    paths = [data/"validation.bin", data/"test.bin"]
    for name in ["long_v1", "transfer_recall_v1"]:
        directory = ROOT/"work/pretraining/evaluation"/name
        paths += [directory/"manifest.json"]
        m = json.loads((directory/"manifest.json").read_text())
        if name == "long_v1":
            path = directory/"tokens.npy"
            assert digest(path) == m["token_file_sha256"]
            paths.append(path)
        for filename, item in m.get("artifacts", {}).items():
            expected = item.get("sha256") if isinstance(item, dict) else item
            if expected:
                path = directory/filename
                assert digest(path) == expected, str(path)
                paths.append(path)
    for path in paths:
        evaluation_hashes[str(path.relative_to(ROOT))] = digest(path)
    for name in ["validation.bin", "test.bin"]:
        assert digest(data/name) == manifest["artifacts"][name]["sha256"]
    model = LanguageModel(ModelConfig(**box["model_config"])).to(device).eval()
    model.load_state_dict(box["model"], strict=True)
    del box["model"]
    output = args.checkpoint.parent/("evaluation_probe" if b["probe"] else "evaluation")
    output.mkdir(parents=True, exist_ok=True)
    stages = args.stages or c["evaluation"]["stages"]
    summaries = {}
    for stage in stages:
        start = time.perf_counter()
        if stage in ["test", "validation"]:
            rows = packed(model, data/f"{stage}.bin", c["sequence_length"], device, rank, world, args.micro_batch, args.limit)
        elif stage == "wiki":
            rows = wiki(model, c["sequence_length"], device, rank, world, args.limit)
        elif stage == "long":
            rows = long_docs(model, c["evaluation"]["long_lengths"], device, rank, world, args.limit)
        else:
            rows = recall(model, c["evaluation"]["recall_lengths"], device, rank, world, args.limit)
        gathered = [None] * world
        dist.all_gather_object(gathered, rows)
        if rank == 0:
            rows = sorted([row for shard in gathered for row in shard], key=lambda r: (r["index"], r.get("length", 0)))
            summary = summarize(stage, rows)
            summaries[stage] = summary
            atomic_json(output/f"{stage}.json", {"stage": stage, "summary": summary, "records": rows,
                        "checkpoint_sha256": digest(args.checkpoint), "tokens_trained": box["tokens"],
                        "probe": b["probe"], "input_sha256": evaluation_hashes, "seconds": time.perf_counter()-start})
            print(json.dumps({"event": "evaluation", "stage": stage, "summary": summary}), flush=True)
    if rank == 0:
        # Keep earlier verified stages when an evaluation is resumed in parts.
        checkpoint_sha = digest(args.checkpoint)
        for stage in c["evaluation"]["stages"]:
            path = output/f"{stage}.json"
            if path.exists():
                saved = json.loads(path.read_text())
                if saved["checkpoint_sha256"] == checkpoint_sha and saved["probe"] == b["probe"]:
                    summaries[stage] = saved["summary"]
        atomic_json(output/"summary.json", {"method": b["method"], "seed": b["seed"], "probe": b["probe"],
                    "tokens_trained": box["tokens"], "metrics": summaries, "checkpoint_sha256": checkpoint_sha,
                    "complete": set(c["evaluation"]["stages"]).issubset(summaries)})
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

"""Two-NPU fixed-budget LM training with auditable, complete resume state."""
import argparse
from contextlib import nullcontext
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch_npu
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
import fla
from ad_kernel.model import LinearLanguageModel, ModelConfig
from ad_kernel.attention import numerical_split_count

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "configs/pretraining_protocol_v2.json"
DATA = ROOT / "work/pretraining/data/fineweb_edu_v1"


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: cpu_tree(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(cpu_tree(item) for item in value)
    return value


def save_torch(path, value):
    temporary = path.with_suffix(path.suffix + ".partial")
    torch.save(value, temporary)
    temporary.replace(path)


def save_json(path, value):
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def batch(stream, indices, length, device):
    arrays = [np.array(stream[int(i) * length:int(i) * length + length + 1], dtype=np.int64) for i in indices]
    ids = torch.from_numpy(np.stack(arrays)).to(device)
    return ids[:, :-1], ids[:, 1:]


@torch.inference_mode()
def evaluate(model, stream, length, micro_batch, rank, world, device):
    model.eval()
    count = (len(stream) - 1) // length
    indices = list(range(rank, count, world))
    total = torch.zeros(2, dtype=torch.float32, device=device)
    for begin in range(0, len(indices), micro_batch):
        subset = indices[begin:begin + micro_batch]
        x, y = batch(stream, subset, length, device)
        with torch.autocast("npu", dtype=torch.bfloat16):
            loss = model(x, y)
        total[0] += loss * y.numel()
        total[1] += y.numel()
    dist.all_reduce(total)
    nll = (total[0] / total[1]).item()
    model.train()
    return {"nll_per_token": nll, "perplexity": math.exp(nll), "tokens": int(total[1].item()),
            "scope": "Packed fixed-context stream from held-out hostnames; no test-set selection"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--micro-batch", type=int, default=16)
    ap.add_argument("--max-steps", type=int)
    ap.add_argument("--run-id")
    ap.add_argument("--synthetic", action="store_true")
    args = ap.parse_args()
    protocol = json.loads(PROTOCOL_PATH.read_text())
    if args.method not in protocol["methods"]:
        raise ValueError(args.method)
    if args.synthetic and not args.run_id:
        raise ValueError("Synthetic checks must have a distinct run ID")
    rank = int(os.environ["LOCAL_RANK"])
    world = int(os.environ["WORLD_SIZE"])
    assert world == protocol["world_size"] == 2
    device = f"npu:{rank}"
    torch.set_num_threads(4)
    torch.npu.set_device(device)
    for check in ["ascend_attention_float32.json", "fused_lm_loss.json", "ddp_rank0.json", "ddp_rank1.json"]:
        assert json.loads((ROOT / "work/reproduction" / check).read_text())["passed"], check
    for check in ["stable_integrated.json", "full_model_integrated.json"]:
        stability = json.loads((ROOT / "work/reproduction/training_failure_step348" / check).read_text())
        assert stability["passed"] and stability["attention_source_sha256"] == digest(ROOT / "src/ad_kernel/attention.py")
    if not args.synthetic:
        recovery = json.loads((ROOT / "work/reproduction/resume_check_stable_v2.json").read_text())
        assert recovery["passed"] and recovery["binding"]["protocol_sha256"] == digest(PROTOCOL_PATH)
        assert recovery["binding"]["code_sha256"]["experiments/pretrain.py"] == digest(Path(__file__))
    assert digest(DATA / "manifest.json") == protocol["data_manifest_sha256"]
    metadata = json.loads((DATA / "manifest.json").read_text())
    dist.init_process_group("hccl")
    try:
        run_name = args.run_id or f"{args.method}_s{args.seed}"
        if any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in run_name):
            raise ValueError("Run ID must be a simple directory name")
        target = ROOT / "work/pretraining/runs" / run_name
        target.mkdir(parents=True, exist_ok=True)
        length = protocol["sequence_length"]
        global_batch = protocol["global_batch_sequences"]
        assert global_batch % (world * args.micro_batch) == 0
        accumulation = global_batch // (world * args.micro_batch)
        stop = protocol["total_schedule_steps"] if args.seed == protocol["primary_seed"] else protocol["additional_seed_steps"]
        if args.max_steps is not None:
            stop = min(stop, args.max_steps)
        config = ModelConfig(**protocol["architecture"], **protocol["methods"][args.method],
                             feature_seed=args.seed, backend=protocol["precision"]["attention_backend"],
                             attention_dtype="float32")
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        model = LinearLanguageModel(config).to(device)
        groups = [{"params": [p for p in model.parameters() if p.ndim >= 2], "weight_decay": protocol["optimizer"]["weight_decay"]},
                  {"params": [p for p in model.parameters() if p.ndim < 2], "weight_decay": 0.}]
        optimizer = torch.optim.AdamW(groups, lr=protocol["optimizer"]["lr"], betas=tuple(protocol["optimizer"]["betas"]), eps=protocol["optimizer"]["eps"])
        code_files = [Path(__file__), *sorted((ROOT / "src/ad_kernel").glob("*.py"))]
        binding = {"protocol_sha256": digest(PROTOCOL_PATH), "data_manifest_sha256": protocol["data_manifest_sha256"],
                   "code_sha256": {str(p.relative_to(ROOT)): digest(p) for p in code_files},
                   "method": args.method, "seed": args.seed, "micro_batch": args.micro_batch, "synthetic": args.synthetic}
        if args.synthetic:
            stream = np.random.default_rng(194).integers(0, config.vocab_size, size=global_batch * length * 32 + 1, dtype=np.uint16)
        else:
            if rank == 0:
                assert digest(DATA / "train.bin") == metadata["artifacts"]["train.bin"]["sha256"]
                assert digest(DATA / "validation.bin") == metadata["artifacts"]["validation.bin"]["sha256"]
            stream = np.memmap(DATA / "train.bin", mode="r", dtype="<u2")
        validation = np.memmap(DATA / "validation.bin", mode="r", dtype="<u2")
        sample_count = (len(stream) - 1) // length
        assert stop * global_batch <= sample_count
        order = torch.randperm(sample_count, generator=torch.Generator().manual_seed(85000 + args.seed)).numpy()
        order_hash = hashlib.sha256(order.astype("<i8", copy=False).tobytes()).hexdigest()
        completed = 0
        last = target / "last.pt"
        if last.exists():
            checkpoint = torch.load(last, map_location="cpu", weights_only=True)
            assert checkpoint["binding"] == binding and checkpoint["order_sha256"] == order_hash
            model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            completed = checkpoint["step"]
            rng = checkpoint["rng_by_rank"][rank]
            torch.set_rng_state(rng["torch"])
            torch.npu.set_rng_state(rng["npu"], device=device)
            random.setstate(rng["python"])
        distributed = DistributedDataParallel(model, device_ids=[rank], broadcast_buffers=False)
        budget = model.parameter_budget()
        if rank == 0:
            save_json(target / "run.json", {"binding": binding, "budget": budget, "world_size": world,
                       "global_batch": global_batch, "micro_batch": args.micro_batch, "accumulation": accumulation,
                       "stop_step": stop, "schedule_steps": protocol["total_schedule_steps"], "order_sha256": order_hash,
                       "torch": str(torch.__version__), "torch_npu": str(torch_npu.__version__)})
        log = (target / "train.jsonl").open("a") if rank == 0 else None

        def event(value):
            if rank == 0:
                print(json.dumps(value, allow_nan=False), flush=True)
                log.write(json.dumps(value, allow_nan=False) + "\n")
                log.flush()

        def checkpoint(step):
            state = {"torch": torch.get_rng_state(), "npu": torch.npu.get_rng_state(device), "python": random.getstate()}
            states = [None] * world
            dist.all_gather_object(states, state)
            if rank == 0:
                box = {"binding": binding, "model": cpu_tree(model.state_dict()), "optimizer": cpu_tree(optimizer.state_dict()),
                       "rng_by_rank": states, "step": step, "tokens": step * global_batch * length,
                       "order_sha256": order_hash, "data_cursor_sequences": step * global_batch, "model_config": budget["config"]}
                save_torch(last, box)
                if step in protocol["milestones"] or step == stop:
                    save_torch(target / f"model_step{step:06d}.pt", {key: box[key] for key in ["binding", "model", "step", "tokens", "model_config"]})
                event({"event": "checkpoint", "step": step, "tokens": box["tokens"], "sha256": digest(last)})
            dist.barrier()

        event({"event": "start" if not completed else "resume", "step": completed, "target_step": stop, "synthetic": args.synthetic})
        model.train()
        running = torch.zeros((), device=device)
        interval_steps = 0
        start = interval = time.perf_counter()
        for step in range(completed, stop):
            warmup = protocol["schedule"]["warmup_steps"]
            peak = protocol["optimizer"]["lr"]
            if step < warmup:
                lr = peak * (step + 1) / warmup
            else:
                progress = (step - warmup) / (protocol["total_schedule_steps"] - warmup)
                ratio = protocol["schedule"]["minimum_lr_ratio"]
                lr = peak * (ratio + (1 - ratio) * (1 + math.cos(math.pi * progress)) / 2)
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
            for micro in range(accumulation):
                first = step * global_batch + (rank * accumulation + micro) * args.micro_batch
                x, y = batch(stream, order[first:first + args.micro_batch], length, device)
                context = distributed.no_sync() if micro + 1 < accumulation else nullcontext()
                with context:
                    with torch.autocast("npu", dtype=torch.bfloat16):
                        loss = distributed(x, y) / accumulation
                    loss.backward()
                running += loss.detach()
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), protocol["optimizer"]["gradient_clip"])
            if not torch.isfinite(grad).item():
                raise RuntimeError(f"Nonfinite gradient at step {step + 1}; optimizer step was not applied")
            optimizer.step()
            done = step + 1
            interval_steps += 1
            if done % 10 == 0 or done == stop:
                dist.all_reduce(running)
                elapsed = time.perf_counter() - interval
                event({"event": "train", "step": done, "tokens": done * global_batch * length,
                       "loss": running.item() / (world * interval_steps), "lr": lr, "grad_norm": grad.item(),
                       "interval_seconds": elapsed, "interval_tokens_per_second": interval_steps * global_batch * length / elapsed,
                       "numerical_splits_this_process_rank0": numerical_split_count(),
                       "elapsed_seconds": time.perf_counter() - start, "peak_allocated_gib": torch.npu.max_memory_allocated() / 2**30})
                running.zero_()
                interval_steps = 0
                interval = time.perf_counter()
            if done % protocol["evaluation_steps"] == 0 or done == stop:
                if not args.synthetic:
                    metric = evaluate(model, validation, length, args.micro_batch, rank, world, device)
                    if rank == 0:
                        save_json(target / f"validation_step{done:06d}.json", {"step": done, "binding": binding, **metric})
                    event({"event": "validation", "step": done, **metric})
            if done % protocol["checkpoint_steps"] == 0 or done in protocol["milestones"] or done == stop:
                checkpoint(done)
        event({"event": "run_complete", "step": stop, "tokens": stop * global_batch * length,
               "synthetic": args.synthetic, "seconds": time.perf_counter() - start})
        if log:
            log.close()
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

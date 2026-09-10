"""Two-NPU HCCL/DDP trainer following the FLAME recipe.

An independent Ascend runner, not an unmodified FLAME invocation. Production
runs always consume the entire protocol budget. Bounded probes have separate
directories and metadata and are never accepted as production checkpoints.
"""
import argparse
from contextlib import nullcontext
from datetime import datetime, timezone, timedelta
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import time
import numpy as np
import torch
import torch.distributed as dist
import torch_npu
import fla
from torch.nn.parallel import DistributedDataParallel as DDP
from .common import ROOT, DEFAULT_CONFIG, digest, atomic_json, cpu_tree, load_checkpoint, source_binding
from .data import TokenPlan
from .model import LanguageModel, ModelConfig
from ad_kernel.attention import numerical_split_count


def learning_rate(config, tokens_before):
    total = config["target_tokens_per_run"]
    warm = total * config["schedule"]["warmup_token_fraction"]
    peak = config["optimizer"]["lr"]
    # Evaluate at the end of this update's token interval, including step 1.
    progress = min(total, tokens_before)
    if progress <= warm:
        return peak * progress / warm
    phase = min(1., (progress-warm)/(total-warm))
    floor = config["schedule"]["minimum_lr_ratio"]
    return peak * (floor + (1-floor) * .5 * (1+math.cos(math.pi*phase)))


def make_optimizer(model, config, implementation=None):
    c = config["optimizer"]
    groups = [{"params": [p for p in model.parameters() if p.ndim >= 2], "weight_decay": c["weight_decay"]},
              {"params": [p for p in model.parameters() if p.ndim < 2], "weight_decay": 0.}]
    cls = torch_npu.optim.NpuFusedAdamW if (implementation or c["name"]) == "NpuFusedAdamW" else torch.optim.AdamW
    return cls(groups, lr=c["lr"], betas=tuple(c["betas"]), eps=c["eps"])


@torch.inference_mode()
def validation(model, stream, rank, world, micro, seq, device):
    model.eval()
    blocks = math.ceil((len(stream)-1)/seq)
    totals = torch.zeros(2, device=device, dtype=torch.float32)
    ids = list(range(rank, blocks, world))
    for begin in range(0, len(ids), micro):
        subset = ids[begin:begin+micro]
        x = np.zeros((len(subset), seq), np.int64)
        y = np.full_like(x, -100)
        count = 0
        for row, index in enumerate(subset):
            values = np.array(stream[index*seq:min((index+1)*seq+1, len(stream))], np.int64)
            n = len(values)-1
            x[row, :n], y[row, :n] = values[:-1], values[1:]
            count += n
        with torch.autocast("npu", dtype=torch.bfloat16):
            loss = model(torch.from_numpy(x).to(device), torch.from_numpy(y).to(device))
        totals[0] += loss.float()
        totals[1] += count
    dist.all_reduce(totals)
    total, count = totals.cpu().tolist()
    model.train()
    return {"nll": total/count, "ppl": math.exp(total/count), "tokens": int(count)}


def save_checkpoint(run, model, optimizer, binding, step, tokens, rank, world, keep, final=False):
    replica_checksums = None
    if binding["probe"]:
        h = hashlib.sha256()
        for name, value in model.state_dict().items():
            h.update(name.encode())
            h.update(value.detach().cpu().contiguous().numpy().tobytes())
        replica_checksums = [None] * world
        dist.all_gather_object(replica_checksums, h.hexdigest())
        assert len(set(replica_checksums)) == 1, "DDP replicas diverged"
    rng = {"cpu": torch.get_rng_state(), "npu": torch.npu.get_rng_state().cpu()}
    states = [None] * world
    dist.all_gather_object(states, rng)
    if rank == 0:
        checkpoint = {"format": 1, "binding": binding, "model_config": model.config.__dict__,
                      "step": step, "tokens": tokens, "model": cpu_tree(model.state_dict()),
                      "optimizer": cpu_tree(optimizer.state_dict()), "rng_by_rank": states,
                      "probe_replica_sha256": replica_checksums}
        path = run / f"checkpoint_{step:06d}.pt"
        tmp = path.with_suffix(".pt.partial")
        torch.save(checkpoint, tmp)
        tmp.replace(path)
        atomic_json(run / "last.json", {"checkpoint": path.name, "step": step, "tokens": tokens})
        if final:
            weights = {k: v for k, v in checkpoint.items() if k not in ["optimizer", "rng_by_rank"]}
            torch.save(weights, run / "final.pt.partial")
            (run / "final.pt.partial").replace(run / "final.pt")
        for old in sorted(run.glob("checkpoint_*.pt"))[:-keep]:
            old.unlink()
    dist.barrier()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--method", required=True, choices=["ad64", "softmax", "favor64", "hedgehog"])
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--probe-steps", type=int, default=0, help="1..16 bounded updates; separate disposable output")
    ap.add_argument("--probe-tag", default="smoke")
    ap.add_argument("--micro-batch", type=int)
    ap.add_argument("--backend", choices=["cann", "fla_recurrent"])
    ap.add_argument("--chunk-size", type=int)
    ap.add_argument("--optimizer", choices=["NpuFusedAdamW", "AdamW"])
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--probe-save", action="store_true")
    ap.add_argument("--probe-eval", action="store_true")
    args = ap.parse_args()
    assert 0 <= args.probe_steps <= 16
    probe = bool(args.probe_steps)
    if not probe and any(x is not None for x in [args.micro_batch, args.backend, args.chunk_size, args.optimizer]):
        raise ValueError("Production settings must be recorded in the protocol JSON, not silently overridden")
    c = json.loads(args.config.read_text())
    if not probe:
        assert c["target_tokens_per_run"] == 2000000000, "Production budget must be exactly 2B target tokens"
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    assert world == c["world_size"] == 2
    torch.npu.set_device(local_rank)
    torch.npu.set_compile_mode(jit_compile=False)
    device = torch.device(f"npu:{local_rank}")
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
    dist.init_process_group("hccl", timeout=timedelta(minutes=20))
    torch.manual_seed(args.seed)
    torch.npu.manual_seed_all(args.seed)
    base = ROOT / "work/pretrain_100m_2b"
    run = base / (f"probes/{args.probe_tag}_{args.method}_s{args.seed}" if probe else f"runs/{args.method}_s{args.seed}")
    if not probe and (base / "STOP").exists():
        raise RuntimeError(f"Stop file exists: {base / 'STOP'}")
    run.mkdir(parents=True, exist_ok=True)
    data = ROOT / c["data"]["directory"]
    assert digest(data / "manifest.json") == c["data"]["manifest_sha256"]
    manifest = json.loads((data / "manifest.json").read_text())
    if rank == 0 and not probe:
        for name in ["train.bin", "validation.bin"]:
            assert digest(data/name) == manifest["artifacts"][name]["sha256"], name
    dist.barrier()
    stream = np.memmap(data/"train.bin", dtype="<u2", mode="r")
    val_stream = np.memmap(data/"validation.bin", dtype="<u2", mode="r")
    assert len(stream) > c["target_tokens_per_run"]
    micro = args.micro_batch or c["micro_batch"][args.method]
    plan = TokenPlan(c["target_tokens_per_run"], c["sequence_length"], c["global_batch_sequences"],
                     world, micro, c["data"]["order_seed_offset"] + args.seed)
    model_config = ModelConfig(**c["architecture"], method=args.method, feature_seed=args.seed)
    if args.backend: model_config.attention_backend = args.backend
    if args.chunk_size: model_config.chunk_size = args.chunk_size
    binding = {"protocol": c, "protocol_sha256": digest(args.config), "source_sha256": source_binding(),
               "data_manifest_sha256": digest(data/"manifest.json"), "seed": args.seed, "method": args.method,
               "micro_batch": micro, "world_size": world, "probe": probe,
               "model_config": model_config.__dict__, "optimizer": args.optimizer or c["optimizer"]["name"],
               "order_sha256": hashlib.sha256(plan.order.tobytes()).hexdigest(),
               "versions": {"torch": str(torch.__version__), "torch_npu": str(torch_npu.__version__), "fla": str(fla.__version__)}}
    if (run/"run.json").exists():
        if not args.resume:
            raise RuntimeError(f"Run exists; use --resume or a different probe tag: {run}")
        assert json.loads((run/"run.json").read_text())["binding"] == binding, "Resume binding mismatch"
    model = LanguageModel(model_config).to(device)
    optimizer = make_optimizer(model, c, args.optimizer)
    start_step = 0
    if args.resume:
        last = json.loads((run/"last.json").read_text())
        saved = load_checkpoint(run/last["checkpoint"])
        assert saved["binding"] == binding
        assert saved["tokens"] == plan.tokens_before(saved["step"])
        model.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        torch.set_rng_state(saved["rng_by_rank"][rank]["cpu"])
        torch.npu.set_rng_state(saved["rng_by_rank"][rank]["npu"])
        start_step = saved["step"]
        del saved
    if rank == 0:
        if not (run/"run.json").exists():
            atomic_json(run/"run.json", {"binding": binding, "parameters": model.parameter_budget(),
                        "started_utc": datetime.now(timezone.utc).isoformat(), "planned_steps": plan.steps})
        print(json.dumps({"event": "start", "probe": probe, "run": str(run), "steps": plan.steps,
                          "start_step": start_step, "parameters": model.parameter_budget()["total"], "micro": micro}), flush=True)
    # Fused optimizer combines parameter/gradient storage; bucket views must be off.
    ddp = DDP(model, device_ids=[local_rank], broadcast_buffers=False,
              gradient_as_bucket_view=False, bucket_cap_mb=40, find_unused_parameters=False)
    model.train()
    stopping = [False]
    def stop_handler(signum, frame): stopping[0] = True
    signal.signal(signal.SIGTERM, stop_handler)
    signal.signal(signal.SIGINT, stop_handler)
    end_step = min(plan.steps, args.probe_steps) if probe else plan.steps
    if start_step >= end_step:
        if rank == 0:
            if not probe and start_step == plan.steps and (run/"final.pt").exists():
                atomic_json(run/"completed.json", {"probe":False,"final_step":start_step,
                            "tokens_consumed":plan.tokens_before(start_step),"resumed_complete_checkpoint":True})
            print(json.dumps({"event": "already_complete", "step": start_step}), flush=True)
        dist.destroy_process_group()
        return
    timings, token_counts = [], []
    torch.npu.reset_peak_memory_stats()
    dist.barrier()
    for batch in plan.prefetch(stream, start_step, end_step, rank, pin=True):
        step = batch.index + 1
        before = plan.tokens_before(batch.index)
        lr = learning_rate(c, before+batch.target_count)
        for group in optimizer.param_groups: group["lr"] = lr
        torch.npu.synchronize()
        started = time.perf_counter()
        optimizer.zero_grad(set_to_none=False)
        inputs = batch.inputs.to(device, non_blocking=True)
        labels = batch.labels.to(device, non_blocking=True)
        loss_sum = torch.zeros((), device=device)
        for i in range(len(inputs)):
            context = ddp.no_sync() if i+1 < len(inputs) else nullcontext()
            with context, torch.autocast("npu", dtype=torch.bfloat16):
                loss = ddp(inputs[i], labels[i])
                scaled_loss = loss * (world / batch.target_count)
            scaled_loss.backward()
            loss_sum += loss.detach()
        if isinstance(optimizer, torch_npu.optim.NpuFusedAdamW):
            norm = optimizer.clip_grad_norm_fused_(c["optimizer"]["gradient_clip"])
        else:
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), c["optimizer"]["gradient_clip"], foreach=False)
        should_stop = stopping[0] or (not probe and (base/"STOP").exists())
        stats = torch.stack((loss_sum.float(), (~torch.isfinite(norm)).float(),
                             torch.tensor(float(should_stop), device=device, dtype=torch.float32)))
        dist.all_reduce(stats)
        global_loss, invalid, requested_stop = stats.cpu().tolist()
        if invalid or not math.isfinite(global_loss):
            if rank == 0:
                atomic_json(run/"failure.json", {"step": step, "reason": "nonfinite loss/gradient; no optimizer update or token advance", "resume_from": "last.json"})
            raise FloatingPointError(f"Nonfinite gradients/loss at step {step}; previous checkpoint retained")
        optimizer.step()
        torch.npu.synchronize()
        elapsed = time.perf_counter()-started
        # Includes forward, backward, accumulation, HCCL, clipping and optimizer.
        slowest = torch.tensor(elapsed, device=device, dtype=torch.float32)
        dist.all_reduce(slowest, op=dist.ReduceOp.MAX)
        elapsed = slowest.item()
        timings.append(elapsed)
        token_counts.append(batch.target_count)
        row = {"step": step, "tokens": before+batch.target_count, "loss": global_loss/batch.target_count,
               "lr": lr, "grad_norm": norm.item(), "step_seconds": elapsed,
               "tokens_per_second": batch.target_count/elapsed, "micro_batch": micro,
               "accumulation": len(inputs), "numerical_splits_rank0": numerical_split_count() if rank == 0 else None}
        if rank == 0:
            with (run/"train.jsonl").open("a") as f: f.write(json.dumps(row, allow_nan=False)+"\n")
            if probe or step % c["log_interval_steps"] == 0 or step == end_step:
                print(json.dumps(row, allow_nan=False), flush=True)
        final = step == plan.steps
        if (not probe and (step % c["validation_interval_steps"] == 0 or final)) or (probe and args.probe_eval and step == end_step):
            val_started = time.perf_counter()
            val_data = val_stream[:8193] if probe else val_stream
            metrics = validation(model, val_data, rank, world, min(micro, 8), plan.seq, device)
            if rank == 0:
                metrics.update(step=step, trained_tokens=row["tokens"], seconds=time.perf_counter()-val_started)
                with (run/"validation.jsonl").open("a") as f: f.write(json.dumps(metrics)+"\n")
                print(json.dumps({"event": "validation", **metrics}), flush=True)
        save = (not probe and (step % c["checkpoint_interval_steps"] == 0 or final or requested_stop)) or (probe and args.probe_save and step == end_step)
        if save:
            io_started = time.perf_counter()
            save_checkpoint(run, model, optimizer, binding, step, row["tokens"], rank, world,
                            c["keep_last_checkpoints"], final=final or (probe and args.probe_save))
            if rank == 0:
                print(json.dumps({"event": "checkpoint", "step": step, "seconds": time.perf_counter()-io_started}), flush=True)
        if requested_stop:
            dist.destroy_process_group()
            raise SystemExit(75)
    peaks = torch.tensor([torch.npu.max_memory_allocated(), torch.npu.max_memory_reserved()], device=device, dtype=torch.float32)
    dist.all_reduce(peaks, op=dist.ReduceOp.MAX)
    if rank == 0:
        skip = min(3, max(0, len(timings)-1)) if probe else 0
        result = {"probe": probe, "steps": len(timings), "final_step": end_step,
                  "tokens_consumed": plan.tokens_before(end_step), "parameters": model.parameter_budget(),
                  "seconds_per_step": timings, "tokens_per_step": token_counts, "warmup_updates_excluded": skip,
                  "measured_tokens_per_second": sum(token_counts[skip:])/sum(timings[skip:]),
                  "estimated_2b_hours_compute": 2000000000 * sum(timings[skip:])/sum(token_counts[skip:])/3600,
                  "max_allocated_gib": peaks[0].item()/2**30, "max_reserved_gib": peaks[1].item()/2**30}
        atomic_json(run/("probe_result.json" if probe else "completed.json"), result)
        print(json.dumps({"event": "complete", **{k:v for k,v in result.items() if k not in ["parameters", "seconds_per_step", "tokens_per_step"]}}), flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

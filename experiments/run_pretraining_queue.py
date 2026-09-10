"""Run the frozen comparison schedule, stopping on any failed training job."""
import hashlib
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time
import torch

ROOT = Path(__file__).resolve().parents[1]
if (ROOT / "work/pretraining/stopped_by_user.json").exists():
    raise SystemExit("Training was explicitly stopped by the user. Do not resume automatically; retain the stop marker until the user authorizes resuming.")
protocol_path = ROOT / "configs/pretraining_protocol_v2.json"
protocol = json.loads(protocol_path.read_text())
protocol_hash = hashlib.sha256(protocol_path.read_bytes()).hexdigest()
assert (ROOT / "work/reproduction/public_wikitext_v1/summary.json").exists(), "Finish the prior reproduction first"
recovery = json.loads((ROOT / "work/reproduction/resume_check_stable_v2.json").read_text())
assert recovery["passed"] and recovery["binding"]["protocol_sha256"] == protocol_hash
code_hashes = recovery["binding"]["code_sha256"]
order = protocol["job_order"]
assert set(order) == set(protocol["methods"])
jobs = [(method, 11, 1600) for method in order]
jobs += [(method, seed, 1600) for seed in protocol["additional_seeds"] for method in order]
jobs += [(method, 11, protocol["total_schedule_steps"]) for method in order]
directory = ROOT / "work/pretraining"
lock = (directory / "queue.lock").open("a")
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
log = (directory / "queue.jsonl").open("a")
evaluation_files = [ROOT / "experiments/evaluate_pretrained.py", ROOT / "experiments/inference_runtime.py", ROOT / "configs/pretrained_evaluation_v1.json"]
evaluation_hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in evaluation_files}


def evaluate_checkpoint(method, seed, step):
    for source, expected in evaluation_hashes.items():
        assert hashlib.sha256((ROOT / source).read_bytes()).hexdigest() == expected, f"Evaluation source changed: {source}"
    checkpoint = directory / "runs" / f"{method}_s{seed}" / f"model_step{step:06d}.pt"
    assert checkpoint.exists()
    start = time.perf_counter()
    processes, outputs = [], []
    try:
        for rank in range(2):
            path = ROOT / "work/logs" / f"evaluate-{method}_s{seed}-step{step}-rank{rank}.log"
            output = path.open("a")
            outputs.append(output)
            processes.append(subprocess.Popen([sys.executable, str(ROOT / "experiments/evaluate_pretrained.py"), str(checkpoint),
                "--device", f"npu:{rank}", "--rank", str(rank), "--world", "2"], cwd=ROOT, stdout=output, stderr=subprocess.STDOUT))
        codes = [process.wait() for process in processes]
    finally:
        for output in outputs:
            output.close()
    record = {"event": "quality_end", "method": method, "seed": seed, "step": step, "returncodes": codes,
              "seconds": time.perf_counter() - start, "evaluation_code_sha256": evaluation_hashes}
    print(json.dumps(record), flush=True)
    log.write(json.dumps(record) + "\n")
    log.flush()
    if any(codes):
        raise SystemExit(f"Stopped at failed quality evaluation for {checkpoint}")
    subprocess.run([sys.executable, str(ROOT / "experiments/summarize_pretrained.py")], cwd=ROOT, check=True)


for index, (method, seed, stop) in enumerate(jobs):
    assert hashlib.sha256(protocol_path.read_bytes()).hexdigest() == protocol_hash, "Protocol changed while queue was running"
    for source, expected in code_hashes.items():
        assert hashlib.sha256((ROOT / source).read_bytes()).hexdigest() == expected, f"Training source changed: {source}"
    run_id = f"{method}_s{seed}"
    path = ROOT / "work/logs" / f"pretrain-{run_id}-to{stop}.log"
    last = directory / "runs" / run_id / "last.pt"
    if last.exists():
        checkpoint = torch.load(last, map_location="cpu", weights_only=True)
        assert checkpoint["binding"]["protocol_sha256"] == protocol_hash
        assert checkpoint["binding"]["code_sha256"] == code_hashes
        if checkpoint["step"] >= stop:
            record = {"event": "verified_skip", "index": index, "method": method, "seed": seed,
                      "requested_step": stop, "checkpoint_step": checkpoint["step"], "tokens": checkpoint["tokens"]}
            print(json.dumps(record), flush=True)
            log.write(json.dumps(record) + "\n")
            log.flush()
            del checkpoint
            evaluate_checkpoint(method, seed, stop)
            continue
        del checkpoint
    record = {"event": "job_start", "index": index, "jobs": len(jobs), "method": method, "seed": seed,
              "stop_step": stop, "protocol_sha256": protocol_hash, "log": str(path)}
    print(json.dumps(record), flush=True)
    log.write(json.dumps(record) + "\n")
    log.flush()
    start = time.perf_counter()
    with path.open("a") as output:
        result = subprocess.run([sys.executable, "-m", "torch.distributed.run", "--standalone", "--nnodes=1", "--nproc_per_node=2",
                                 str(ROOT / "experiments/pretrain.py"), "--method", method, "--seed", str(seed),
                                 "--micro-batch", "16", "--max-steps", str(stop)], cwd=ROOT, stdout=output, stderr=subprocess.STDOUT)
    record = {"event": "job_end", "index": index, "method": method, "seed": seed, "stop_step": stop,
              "returncode": result.returncode, "seconds": time.perf_counter() - start}
    print(json.dumps(record), flush=True)
    log.write(json.dumps(record) + "\n")
    log.flush()
    if result.returncode:
        raise SystemExit(f"Stopped at failed job {run_id}; inspect {path}")
    evaluate_checkpoint(method, seed, stop)
log.close()
print(json.dumps({"event": "queue_complete", "jobs": len(jobs)}), flush=True)

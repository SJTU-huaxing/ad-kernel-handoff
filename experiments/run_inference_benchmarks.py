"""Wait for the full quality queue, then benchmark each final checkpoint."""
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "work/pretraining"
protocol = ROOT / "configs/pretraining_protocol_v2.json"
expected_protocol = hashlib.sha256(protocol.read_bytes()).hexdigest()
configuration = json.loads(protocol.read_text())
sources = [ROOT / "experiments/benchmark_inference.py", ROOT / "experiments/inference_runtime.py", ROOT / "configs/inference_benchmark_v1.json"]
expected_sources = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
lock = (BASE / "inference_queue.lock").open("a")
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
print(json.dumps({"event": "waiting_for_complete_training_and_quality", "expected_checkpoints": 40}), flush=True)
while True:
    assert hashlib.sha256(protocol.read_bytes()).hexdigest() == expected_protocol
    # The training runner prints its final completion event to stdout after
    # closing queue.jsonl, so inspect that terminal record explicitly.
    tail = (ROOT / "work/logs/pretraining-queue.log").read_text().splitlines()
    try:
        terminal = json.loads(tail[-1]) if tail else {}
    except json.JSONDecodeError:
        terminal = {}
    if terminal.get("event") == "queue_complete":
        break
    with (BASE / "queue.lock").open("a") as training_lock:
        try:
            fcntl.flock(training_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            pass
        else:
            raise SystemExit("Training/quality queue stopped before completion; inference benchmark queue stopped too")
    time.sleep(30)
summary = json.loads((BASE / "reports/summary.json").read_text())
assert summary["protocol_sha256"] == expected_protocol and summary["complete_checkpoints"] == 40
for source, expected in expected_sources.items():
    assert hashlib.sha256((ROOT / source).read_bytes()).hexdigest() == expected
completed = []
for method in configuration["job_order"]:
    checkpoint = BASE / "runs" / f"{method}_s11" / "model_step016384.pt"
    jobs, logs = [], []
    try:
        for device, batch in [("npu:0", 1), ("npu:1", 4)]:
            log = (ROOT / "work/logs" / f"benchmark-inference-{method}-b{batch}.log").open("a")
            logs.append(log)
            jobs.append(subprocess.Popen([sys.executable, str(ROOT / "experiments/benchmark_inference.py"), str(checkpoint),
                                         "--device", device, "--batch", str(batch)], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT))
        codes = [job.wait() for job in jobs]
    finally:
        for log in logs:
            log.close()
    record = {"event": "inference_benchmark_end", "method": method, "returncodes": codes}
    print(json.dumps(record), flush=True)
    if any(codes):
        raise SystemExit(f"Inference timing failed for {method}")
    completed.append(method)
result = {"event": "inference_queue_complete", "methods": completed, "benchmark_source_sha256": expected_sources,
          "training_protocol_sha256": expected_protocol}
path = BASE / "inference_queue_complete.json"
temporary = path.with_suffix(".json.partial")
temporary.write_text(json.dumps(result, indent=2) + "\n")
temporary.replace(path)
print(json.dumps(result), flush=True)

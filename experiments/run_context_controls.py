"""Run supplementary final-model diagnostics after exclusive timing completes."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import psutil

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "work/pretraining"
ap = argparse.ArgumentParser()
ap.add_argument("--after-pid", type=int, required=True)
args = ap.parse_args()
lock = (BASE / "context_queue.lock").open("a")
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
configuration = json.loads((ROOT / "configs/pretraining_protocol_v2.json").read_text())
training_hash = hashlib.sha256((ROOT / "configs/pretraining_protocol_v2.json").read_bytes()).hexdigest()
sources = [ROOT / "experiments/evaluate_context_controls.py", ROOT / "configs/context_control_v1.json"]
hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
print(json.dumps({"event": "waiting_for_inference_benchmarks", "supervised_pid": args.after_pid}), flush=True)
while not (BASE / "inference_queue_complete.json").exists():
    try:
        process = psutil.Process(args.after_pid)
        assert process.is_running() and any(Path(arg).name == "run_inference_benchmarks.py" for arg in process.cmdline())
    except (psutil.NoSuchProcess, psutil.ZombieProcess, AssertionError):
        raise SystemExit("Supervised inference queue ended before its completion marker; context queue stopped")
    time.sleep(30)
completion = json.loads((BASE / "inference_queue_complete.json").read_text())
assert completion["training_protocol_sha256"] == training_hash
assert set(completion["methods"]) == set(configuration["methods"])
for source, expected in hashes.items():
    assert hashlib.sha256((ROOT / source).read_bytes()).hexdigest() == expected, source
for method in configuration["job_order"]:
    checkpoint = BASE / "runs" / f"{method}_s11" / "model_step016384.pt"
    jobs, outputs = [], []
    try:
        for rank in range(2):
            output = (ROOT / "work/logs" / f"context-controls-{method}-rank{rank}.log").open("a")
            outputs.append(output)
            jobs.append(subprocess.Popen([sys.executable, str(ROOT / "experiments/evaluate_context_controls.py"), str(checkpoint), "--rank", str(rank)],
                                         cwd=ROOT, stdout=output, stderr=subprocess.STDOUT))
        codes = [job.wait() for job in jobs]
    finally:
        for output in outputs:
            output.close()
    print(json.dumps({"event": "context_job_end", "method": method, "returncodes": codes}), flush=True)
    if any(codes):
        raise SystemExit(f"Context-control evaluation failed for {method}")
    subprocess.run([sys.executable, str(ROOT / "experiments/summarize_context_controls.py")], cwd=ROOT, check=True)
subprocess.run([sys.executable, str(ROOT / "experiments/render_pretraining_figures.py")], cwd=ROOT, check=True)
summary = BASE / "reports/context_controls.json"
assert json.loads(summary.read_text())["complete_methods"] == 10
completion = {"event": "context_queue_complete", "methods": configuration["job_order"], "evaluation_source_sha256": hashes,
              "training_protocol_sha256": training_hash, "summary_sha256": hashlib.sha256(summary.read_bytes()).hexdigest()}
target = BASE / "context_queue_complete.json"
temporary = target.with_suffix(".json.partial")
temporary.write_text(json.dumps(completion, indent=2) + "\n")
temporary.replace(target)
print(json.dumps(completion), flush=True)

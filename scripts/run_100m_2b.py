"""Review the plan by default; --execute explicitly starts the prepared queue."""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "work/pretrain_100m_2b"


def stream_job(command, log):
    """Show child output live in the foreground and retain an append-only log."""
    print(f"Running: {shlex.join(command)}\nLog: {log}", flush=True)
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    with log.open("a", encoding="utf-8", buffering=1) as f:
        with subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                              errors="replace", bufsize=1, env=env) as child:
            for line in child.stdout:
                f.write(line)
                sys.stdout.write(line)
                sys.stdout.flush()
            rc = child.wait()
    if rc:
        raise SystemExit(f"Job exited {rc}; queue stopped. Inspect {log}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--execute", action="store_true")
    p.add_argument("--config", type=Path, default=ROOT/"configs/pretrain_100m_2b.json")
    p.add_argument("--seeds", nargs="+", type=int)
    p.add_argument("--methods", nargs="+", choices=["ad64", "softmax", "favor64", "hedgehog"])
    p.add_argument("--evaluation-only", action="store_true")
    a = p.parse_args()
    c = json.loads(a.config.read_text())
    seeds, methods = a.seeds or c["seeds"], a.methods or c["methods"]
    assert len(seeds) == len(set(seeds)) and len(methods) == len(set(methods))
    torchrun = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=2"]
    jobs = []
    for seed in seeds:
        for method in methods:
            run = BASE/"runs"/f"{method}_s{seed}"
            train = torchrun + ["-m", "pretrain_100m.train", "--config", str(a.config.resolve()), "--method", method, "--seed", str(seed)]
            if (run/"last.json").exists(): train.append("--resume")
            evaluate = torchrun + ["-m", "pretrain_100m.evaluate", str(run/"final.pt")]
            jobs.append((run, train, evaluate))
    print(f"{'EXECUTE' if a.execute else 'PLAN ONLY'}: {len(jobs)} runs, {c['target_tokens_per_run']:,} target tokens/run, 2 NPUs per run.", flush=True)
    print(f"Seeds: {seeds}; methods: {methods}. Output: {BASE}", flush=True)
    estimate_path = BASE/"validation/time_estimate.json"
    if estimate_path.exists():
        estimate = json.loads(estimate_path.read_text())
        if estimate["config_sha256"] == hashlib.sha256(a.config.read_bytes()).hexdigest():
            selected = [r for r in estimate["rows"] if r["method"] in methods]
            low = len(seeds)*sum(r["planning_hours_min"] for r in selected)
            high = len(seeds)*sum(r["planning_hours_max"] for r in selected)
            print(f"Measured planning estimate for selected training + evaluation: {low:g}–{high:g} hours on both NPUs.", flush=True)
    for run, train, evaluate in jobs:
        if not a.evaluation_only: print(shlex.join(train), flush=True)
        print(shlex.join(evaluate), flush=True)
    if not a.execute:
        print("No training started. Add --execute to launch this plan.")
        return
    BASE.mkdir(parents=True, exist_ok=True)
    lock = (BASE/"queue.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (BASE/"STOP").exists(): raise SystemExit(f"Stop marker exists: {BASE/'STOP'}")
    subprocess.run([sys.executable, "-m", "pretrain_100m.preflight", "--config", str(a.config)], check=True, cwd=ROOT)
    stop_requested = [False]
    def stop(signum, frame):
        stop_requested[0] = True
        (BASE/"STOP").write_text(datetime.now(timezone.utc).isoformat()+"\n")
        print("Stop requested. Waiting for the current optimizer update and checkpoint.", flush=True)
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    logs = BASE/"logs"
    logs.mkdir(exist_ok=True)
    for run, train, evaluate in jobs:
        if stop_requested[0] or (BASE/"STOP").exists(): break
        if not a.evaluation_only and not (run/"completed.json").exists():
            stream_job(train, logs/f"{run.name}.train.log")
        if stop_requested[0] or (BASE/"STOP").exists(): break
        if not (run/"final.pt").exists(): raise SystemExit(f"No full-budget final checkpoint: {run}")
        summary = run/"evaluation/summary.json"
        if not summary.exists() or not json.loads(summary.read_text()).get("complete", False):
            stream_job(evaluate, logs/f"{run.name}.eval.log")
        subprocess.run([sys.executable, "-m", "pretrain_100m.summarize"], check=True, cwd=ROOT)
    print("Queue stopped by request." if stop_requested[0] or (BASE/"STOP").exists() else "All selected training and evaluation jobs finished.", flush=True)


if __name__ == "__main__": main()

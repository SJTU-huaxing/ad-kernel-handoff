"""Read current process state and checkpoint progress without touching runs."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import psutil

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "work/pretraining"


def read_events(path):
    if not path.exists():
        return []
    lines = path.read_text().splitlines()
    events = []
    for index, line in enumerate(lines):
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            if index != len(lines) - 1:
                raise
    return events


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    protocol_path = ROOT / "configs/pretraining_protocol_v2.json"
    protocol = json.loads(protocol_path.read_text())
    protocol_hash = hashlib.sha256(protocol_path.read_bytes()).hexdigest()
    names = {"run_pretraining_queue.py", "pretrain.py", "evaluate_pretrained.py", "run_inference_benchmarks.py",
             "benchmark_inference.py", "run_context_controls.py", "evaluate_context_controls.py", "run_final_reports.py",
             "audit_pretraining_artifacts.py"}
    processes = []
    for process in psutil.process_iter(["pid", "cmdline", "status", "create_time"]):
        argv = process.info["cmdline"] or []
        match = next((Path(arg).name for arg in argv if Path(arg).name in names), None)
        if match is None:
            continue
        record = {"pid": process.info["pid"], "script": match, "status": process.info["status"],
                  "age_seconds": round(time.time() - process.info["create_time"]),
                  "role": "distributed launcher" if "torch.distributed.run" in argv else "process"}
        for field in ["method", "seed", "rank", "batch"]:
            if "--" + field in argv:
                record[field] = argv[argv.index("--" + field) + 1]
        processes.append(record)
    runs = []
    for method in protocol["job_order"]:
        for seed in [protocol["primary_seed"], *protocol["additional_seeds"]]:
            directory = BASE / "runs" / f"{method}_s{seed}"
            if not (directory / "run.json").exists():
                continue
            metadata = json.loads((directory / "run.json").read_text())
            assert metadata["binding"]["protocol_sha256"] == protocol_hash and not metadata["binding"]["synthetic"]
            events = read_events(directory / "train.jsonl")
            starts = [i for i, event in enumerate(events) if event["event"] in ["start", "resume"]]
            current = events[starts[-1]:] if starts else []
            train = [event for event in current if event["event"] == "train"]
            step = train[-1]["step"] if train else current[0]["step"] if current else 0
            checkpoints = [event for event in events if event["event"] == "checkpoint"]
            target = protocol["total_schedule_steps"] if seed == protocol["primary_seed"] else protocol["additional_seed_steps"]
            record = {"method": method, "seed": seed, "current_step": step, "target_step": target,
                      "tokens": step * 65536, "target_tokens": target * 65536,
                      "last_saved_step": checkpoints[-1]["step"] if checkpoints else 0,
                      "current_job_target_step": metadata["stop_step"]}
            if train:
                record.update(loss=train[-1]["loss"], recent_interval_tokens_per_second=train[-1]["interval_tokens_per_second"])
            runs.append(record)
    counts = {}
    for name, file, field, total in [("quality", "summary.json", "complete_checkpoints", 40),
                                     ("inference", "efficiency.json", "complete_benchmark_jobs", 20),
                                     ("context_controls", "context_controls.json", "complete_methods", 10)]:
        path = BASE / "reports" / file
        counts[name] = {"completed_in_report": json.loads(path.read_text())[field] if path.exists() else 0, "expected": total}
    result = {"protocol_sha256": protocol_hash, "live_processes": processes, "training_progress": runs, "report_counts": counts}
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        for process in processes:
            identity = " ".join(f"{key}={process[key]}" for key in ["method", "seed", "rank", "batch"] if key in process)
            print(f"PID {process['pid']}: {process['script']} {identity} [{process['status']}, {process['role']}]")
        for row in runs:
            print(f"{row['method']} seed{row['seed']}: step {row['current_step']}/{row['target_step']}, "
                  f"{row['tokens']:,}/{row['target_tokens']:,} tokens; saved step {row['last_saved_step']}")
        print("Reports: " + "; ".join(f"{name} {row['completed_in_report']}/{row['expected']}" for name, row in counts.items()))


if __name__ == "__main__":
    main()

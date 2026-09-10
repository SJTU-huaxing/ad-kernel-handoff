"""Finish CPU reports after the supervised context queue has completed."""
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


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--after-pid", required=True, type=int)
    args = ap.parse_args()
    with (BASE / "final_reports.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        protocol_path = ROOT / "configs/pretraining_protocol_v2.json"
        protocol_hash = digest(protocol_path)
        protocol = json.loads(protocol_path.read_text())
        marker = BASE / "context_queue_complete.json"
        created = None
        print(json.dumps({"event": "waiting_for_context_queue", "supervised_pid": args.after_pid}), flush=True)
        while not marker.exists():
            try:
                process = psutil.Process(args.after_pid)
                assert process.is_running() and process.status() != psutil.STATUS_ZOMBIE
                assert any(Path(arg).name == "run_context_controls.py" for arg in process.cmdline())
                if created is None:
                    created = process.create_time()
                assert process.create_time() == created
            except (psutil.NoSuchProcess, psutil.ZombieProcess, AssertionError):
                raise SystemExit("The supervised context queue stopped without a completion marker; final reporting stopped")
            time.sleep(30)
        completion = json.loads(marker.read_text())
        assert completion["training_protocol_sha256"] == protocol_hash == digest(protocol_path)
        assert set(completion["methods"]) == set(protocol["methods"])
        for source, expected in completion["evaluation_source_sha256"].items():
            assert digest(ROOT / source) == expected
        subprocess.run([sys.executable, str(ROOT / "experiments/audit_pretraining_artifacts.py"), "--require-complete"], cwd=ROOT, check=True)
        subprocess.run([sys.executable, str(ROOT / "experiments/render_pretraining_figures.py")], cwd=ROOT, check=True)
        audit_path = BASE / "reports/artifact_audit.json"
        audit = json.loads(audit_path.read_text())
        assert audit["artifact_grid_complete"] and audit["training_protocol_sha256"] == protocol_hash
        result = {"event": "final_reports_complete", "training_protocol_sha256": protocol_hash,
                  "artifact_audit_sha256": digest(audit_path),
                  "scope": "The scheduled artifacts and reports passed their integrity audit; interpretation and the user-goal completion audit remain separate."}
        target = BASE / "final_reports_complete.json"
        temporary = target.with_suffix(".json.partial")
        temporary.write_text(json.dumps(result, indent=2) + "\n")
        temporary.replace(target)
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()

"""Audit fixed-budget artifacts on disk; this is not a scientific conclusion."""
import argparse
from datetime import datetime, timezone
import hashlib
import itertools
import json
from pathlib import Path
import subprocess
import sys

import psutil
import torch

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "work/pretraining"


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 << 20), b""):
            result.update(block)
    return result.hexdigest()


def grid_audit(rows, fields, expected):
    """Counts alone cannot establish coverage; duplicates and wrong IDs fail."""
    keys = [tuple(row[field] for field in fields) for row in rows]
    actual = set(keys)
    assert len(actual) == len(keys), f"Duplicate {fields} entries"
    assert actual <= expected, f"Unexpected {fields}: {actual - expected}"
    return {"verified": len(actual), "expected": len(expected), "complete": actual == expected,
            "missing": [dict(zip(fields, key)) for key in sorted(expected - actual)]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--require-complete", action="store_true")
    args = ap.parse_args()
    torch.set_num_threads(2)
    protocol_file = ROOT / "configs/pretraining_protocol_v2.json"
    protocol = json.loads(protocol_file.read_text())
    protocol_hash = digest(protocol_file)
    methods = set(protocol["methods"])
    primary, additional = protocol["primary_seed"], protocol["additional_seeds"]
    early, final = protocol["additional_seed_steps"], protocol["total_schedule_steps"]
    tokens_per_step = protocol["sequence_length"] * protocol["global_batch_sequences"]
    recovery = json.loads((ROOT / "work/reproduction/resume_check_stable_v2.json").read_text())
    assert recovery["passed"] and recovery["binding"]["protocol_sha256"] == protocol_hash
    code_hashes = recovery["binding"]["code_sha256"]
    for source, expected in code_hashes.items():
        assert digest(ROOT / source) == expected, source

    # Verify immutable data bytes, not just that manifests are present.
    data_dir = BASE / "data/fineweb_edu_v1"
    assert digest(data_dir / "manifest.json") == protocol["data_manifest_sha256"]
    data = json.loads((data_dir / "manifest.json").read_text())
    verified_data = {}
    for name, artifact in data["artifacts"].items():
        path = data_dir / name
        assert path.stat().st_size == artifact["bytes"] and digest(path) == artifact["sha256"], name
        verified_data[str(path.relative_to(ROOT))] = artifact["sha256"]
    long_dir = BASE / "evaluation/long_v1"
    long_manifest = json.loads((long_dir / "manifest.json").read_text())
    assert long_manifest["training_manifest_sha256"] == protocol["data_manifest_sha256"]
    assert digest(long_dir / "tokens.npy") == long_manifest["token_file_sha256"]
    verified_data[str((long_dir / "tokens.npy").relative_to(ROOT))] = long_manifest["token_file_sha256"]
    extra_dir = BASE / "evaluation/transfer_recall_v1"
    extra = json.loads((extra_dir / "manifest.json").read_text())
    assert extra["training_manifest_sha256"] == protocol["data_manifest_sha256"]
    for name, expected in extra["artifacts"].items():
        assert digest(extra_dir / name) == expected, name
        verified_data[str((extra_dir / name).relative_to(ROOT))] = expected

    # Recompute the source repository audit and check the fresh confirmation sources.
    repository = subprocess.run([sys.executable, str(ROOT / "verify_repository.py")], cwd=ROOT,
                                capture_output=True, text=True, check=True)
    repository_result = json.loads(repository.stdout)
    assert repository_result["passed"]
    historical_dir = ROOT / "work/reproduction/public_wikitext_v1"
    historical = json.loads((historical_dir / "summary.json").read_text())
    assert historical["plan_sha256"] == digest(historical_dir / "frozen_evaluation.json")
    historical_plan = json.loads((historical_dir / "frozen_evaluation.json").read_text())
    assert len(historical_plan["models"]) == len(set(historical_plan["models"])) == 21
    historical_assessment = digest(ROOT / "experiments/assess_reproduction.py")
    historical_attention = digest(BASE / "archive/protocol_v1_global_scaling/source/src/ad_kernel/attention.py")
    historical_binding = {"plan_sha256": historical["plan_sha256"], "assessment_source_sha256": historical_assessment,
                          "attention_source_sha256": historical_attention}
    historical_output = historical_dir / "confirmation" / historical_assessment[:12]
    expected_sources = {str((historical_output / f"{method}_{split}.json").relative_to(ROOT))
                        for method, split in itertools.product(historical_plan["models"], historical_plan["splits"])}
    assert set(historical["sources"]) == expected_sources
    historical_sources = dict(historical["sources"])
    for source, expected in historical["sources"].items():
        assert digest(ROOT / source) == expected, source
        assert json.loads((ROOT / source).read_text())["binding"] == historical_binding
    for split in historical_plan["splits"]:
        teachers = []
        for rank in [0, 1]:
            path = historical_output / f"teacher_{split}_rank{rank}.json"
            teacher = json.loads(path.read_text())
            assert teacher["binding"] == historical_binding
            teachers.append(teacher)
            historical_sources[str(path.relative_to(ROOT))] = digest(path)
        assert abs(teachers[0]["ppl"]["nll_per_token"] - teachers[1]["ppl"]["nll_per_token"]) < 1e-6
        assert teachers[0]["ppl"]["perplexity"] == historical["summary"][split]["teacher"]["ppl"]
    assert set(historical_plan["checkpoint_sha256"]) == set(historical_plan["models"])
    for name, expected in historical_plan["checkpoint_sha256"].items():
        assert digest(historical_dir / "fits" / f"{name}.pt") == expected
    assert digest(historical_dir / "data/manifest.json") == historical_plan["manifest_sha256"]

    # Regeneration verifies shard IDs, current evaluation code and metric units.
    for script in ["summarize_pretrained.py", "summarize_efficiency.py", "summarize_context_controls.py", "summarize_training_cost.py"]:
        subprocess.run([sys.executable, str(ROOT / "experiments" / script)], cwd=ROOT, check=True)
    quality = json.loads((BASE / "reports/summary.json").read_text())
    efficiency = json.loads((BASE / "reports/efficiency.json").read_text())
    context = json.loads((BASE / "reports/context_controls.json").read_text())
    assert quality["protocol_sha256"] == efficiency["training_protocol_sha256"] == context["training_protocol_sha256"] == protocol_hash
    expected_quality = set(itertools.product(methods, [primary, *additional], [early]))
    expected_quality.update(itertools.product(methods, [primary], [final]))
    coverage = {"quality": grid_audit(quality["runs"], ["method", "seed", "step"], expected_quality),
                "inference": grid_audit(efficiency["runs"], ["method", "batch"], set(itertools.product(methods, [1, 4]))),
                "context": grid_audit(context["runs"], ["method", "seed", "step"], set(itertools.product(methods, [primary], [final])))}
    checkpoint_hashes = {}
    for row in quality["runs"]:
        method, seed, step = row["method"], row["seed"], row["step"]
        path = BASE / "runs" / f"{method}_s{seed}" / f"model_step{step:06d}.pt"
        assert digest(path) == row["checkpoint_sha256"], path
        box = torch.load(path, weights_only=True, mmap=True, map_location="cpu")
        binding = box["binding"]
        assert binding["method"] == method and binding["seed"] == seed and not binding["synthetic"]
        assert binding["protocol_sha256"] == protocol_hash and binding["code_sha256"] == code_hashes
        assert binding["data_manifest_sha256"] == protocol["data_manifest_sha256"]
        assert box["step"] == step and box["tokens"] == row["training_tokens"] == step * tokens_per_step
        config = box["model_config"]
        for key, value in {**protocol["architecture"], **protocol["methods"][method]}.items():
            assert config[key] == value, (method, key)
        assert config["backend"] == "torch" and config["attention_dtype"] == "float32" and config["fused_loss"]
        assert config["feature_seed"] == seed
        del box
        checkpoint_hashes[(method, seed, step)] = row["checkpoint_sha256"]
        for name, expected in row["artifact_sha256"].items():
            assert digest(ROOT / row["evaluation_directory"] / name) == expected
    for row in efficiency["runs"]:
        assert row["checkpoint_sha256"] == checkpoint_hashes[(row["method"], primary, final)]
        assert digest(ROOT / row["artifact"]) == row["artifact_sha256"]
    context_id = hashlib.sha256(json.dumps(context["evaluation_source_sha256"], sort_keys=True).encode()).hexdigest()[:12]
    for row in context["runs"]:
        assert row["checkpoint_sha256"] == checkpoint_hashes[(row["method"], primary, final)]
        directory = BASE / "context_controls" / f"{row['method']}_s{primary}_step{final:06d}" / context_id
        for name, expected in row["source_artifact_sha256"].items():
            assert digest(directory / name) == expected

    # Every model/seed must also retain its full optimizer/RNG/cursor state.
    states, order_hashes = [], {}
    samples = (data["tokens"]["train"] - 1) // protocol["sequence_length"]
    for seed in [primary, *additional]:
        order = torch.randperm(samples, generator=torch.Generator().manual_seed(85000 + seed)).numpy()
        order_hashes[seed] = hashlib.sha256(order.astype("<i8", copy=False).tobytes()).hexdigest()
    for method, seed in itertools.product(sorted(methods), [primary, *additional]):
        target_step = final if seed == primary else early
        directory = BASE / "runs" / f"{method}_s{seed}"
        log = directory / "train.jsonl"
        if not log.exists():
            continue
        lines = log.read_text().splitlines()
        try:
            terminal = json.loads(lines[-1]) if lines else {}
        except json.JSONDecodeError:
            continue  # An active writer may have emitted part of its next event.
        if terminal.get("event") != "run_complete":
            continue
        step = terminal["step"]
        assert step in {early, target_step} and not terminal["synthetic"]
        if (method, seed, step) not in checkpoint_hashes:
            continue
        metadata = json.loads((directory / "run.json").read_text())
        assert metadata["world_size"] == 2 and metadata["global_batch"] == protocol["global_batch_sequences"]
        assert metadata["micro_batch"] * metadata["accumulation"] * 2 == metadata["global_batch"]
        assert metadata["order_sha256"] == order_hashes[seed] and metadata["schedule_steps"] == final
        path = directory / "last.pt"
        state = torch.load(path, weights_only=True, mmap=True, map_location="cpu")
        assert state["binding"] == metadata["binding"]
        assert state["binding"]["code_sha256"] == code_hashes and state["binding"]["protocol_sha256"] == protocol_hash
        assert state["binding"]["method"] == method and state["binding"]["seed"] == seed and not state["binding"]["synthetic"]
        assert state["step"] == step and state["tokens"] == step * tokens_per_step
        assert state["data_cursor_sequences"] == step * protocol["global_batch_sequences"] <= samples
        assert state["order_sha256"] == order_hashes[seed] and len(state["rng_by_rank"]) == 2
        assert state["optimizer"]["state"] and state["optimizer"]["param_groups"]
        del state
        events = [json.loads(line) for line in lines]
        saved = [event for event in events if event["event"] == "checkpoint"][-1]
        last_hash = digest(path)
        assert saved["step"] == step and saved["sha256"] == last_hash
        states.append({"method": method, "seed": seed, "step": step, "tokens": step * tokens_per_step,
                       "target_step": target_step, "last_checkpoint_sha256": last_hash, "order_sha256": order_hashes[seed]})
    coverage["target_budget_resume_states"] = grid_audit([row for row in states if row["step"] == row["target_step"]],
                                                        ["method", "seed"], set(itertools.product(methods, [primary, *additional])))
    worker_names = {"pretrain.py", "evaluate_pretrained.py", "benchmark_inference.py", "evaluate_context_controls.py"}
    workers = [{"pid": p.info["pid"], "status": p.info["status"]} for p in psutil.process_iter(["pid", "cmdline", "status"])
               if any(Path(arg).name in worker_names for arg in p.info["cmdline"] or [])]
    complete = all(row["complete"] for row in coverage.values()) and not workers
    output = {"created_utc": datetime.now(timezone.utc).isoformat(), "artifact_grid_complete": complete,
              "scope": "Artifact integrity and fixed-budget coverage only. Scientific interpretation, operator correctness and the full user-goal audit require their separate evidence; this flag does not mark the overall goal complete.",
              "training_protocol_sha256": protocol_hash, "audit_source_sha256": digest(Path(__file__)),
              "coverage": coverage, "active_workers": workers, "original_repository": repository_result,
              "historical_confirmation_sources_verified": len(historical_sources),
              "historical_confirmation_source_sha256": historical_sources,
              "historical_fit_sha256": historical_plan["checkpoint_sha256"],
              "verified_data_sha256": verified_data, "training_resume_states": states,
              "report_sha256": {name: digest(BASE / "reports" / name) for name in ["summary.json", "efficiency.json", "context_controls.json", "training_cost.json"]}}
    target = BASE / "reports/artifact_audit.json"
    temporary = target.with_suffix(".json.partial")
    temporary.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    temporary.replace(target)
    print(json.dumps({"event": "artifact_audit", "artifact_grid_complete": complete,
                      "coverage": {name: f"{row['verified']}/{row['expected']}" for name, row in coverage.items()}}), flush=True)
    if args.require_complete and not complete:
        raise SystemExit("The required artifact grid is incomplete; see work/pretraining/reports/artifact_audit.json")


if __name__ == "__main__":
    main()

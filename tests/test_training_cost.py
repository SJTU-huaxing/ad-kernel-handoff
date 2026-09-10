import importlib.util
from pathlib import Path

import pytest

SOURCE = Path(__file__).resolve().parents[1] / "experiments/summarize_training_cost.py"
spec = importlib.util.spec_from_file_location("training_cost", SOURCE)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def start(step, target):
    return {"event": "start" if step == 0 else "resume", "step": step, "target_step": target, "synthetic": False}


def train(step, seconds=2):
    return {"event": "train", "step": step, "tokens": step * 100, "elapsed_seconds": seconds,
            "peak_allocated_gib": 12, "numerical_splits_this_process_rank0": 4}


def end(step, seconds=4):
    return {"event": "run_complete", "step": step, "tokens": step * 100, "seconds": seconds, "synthetic": False}


def test_resume_counts_only_new_updates_and_keeps_save_validation_time():
    result = module.training_segments([start(1600, 16384), train(16384), end(16384)], 100)[0]
    assert result["logged_update_tokens"] == 1_478_400
    assert result["seconds"] == 4 and result["tokens_per_second"] == 369_600
    assert result["status"] == "complete"


def test_interrupted_updates_are_not_silently_merged_into_resume():
    events = [start(0, 100), train(30), start(20, 100), train(100), end(100)]
    rows = module.training_segments(events, 100)
    assert [row["status"] for row in rows] == ["interrupted_before_restart", "complete"]
    assert [row["logged_update_tokens"] for row in rows] == [3000, 8000]
    assert [row["seconds"] for row in rows] == [2, 4]


def test_open_snapshot_is_not_completed_even_at_target_step():
    row = module.training_segments([start(0, 100), train(100)], 100)[0]
    assert row["status"] == "open_at_snapshot"


def test_completion_without_target_update_is_rejected():
    with pytest.raises(AssertionError):
        module.training_segments([start(0, 100), train(90), end(100)], 100)

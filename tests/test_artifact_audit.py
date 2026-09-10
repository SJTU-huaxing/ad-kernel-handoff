import importlib.util
from pathlib import Path

import pytest

SOURCE = Path(__file__).resolve().parents[1] / "experiments/audit_pretraining_artifacts.py"
spec = importlib.util.spec_from_file_location("artifact_audit", SOURCE)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def test_partial_grid_is_not_completion():
    expected = {("ad", 11), ("exp", 11), ("ad", 29), ("exp", 29)}
    result = audit.grid_audit([{"method": "ad", "seed": 11}], ["method", "seed"], expected)
    assert not result["complete"] and result["verified"] == 1 and result["expected"] == 4
    assert {tuple(row.values()) for row in result["missing"]} == expected - {("ad", 11)}


def test_repeated_success_cannot_replace_missing_seed():
    rows = [{"method": "ad", "seed": 11}, {"method": "ad", "seed": 11}]
    with pytest.raises(AssertionError, match="Duplicate"):
        audit.grid_audit(rows, ["method", "seed"], {("ad", 11), ("ad", 29)})


def test_wrong_budget_cannot_replace_expected_checkpoint():
    rows = [{"method": "ad", "step": 1600}]
    with pytest.raises(AssertionError, match="Unexpected"):
        audit.grid_audit(rows, ["method", "step"], {("ad", 16384)})


def test_complete_grid_accepts_arbitrary_row_order():
    rows = [{"method": "exp", "seed": 29}, {"method": "ad", "seed": 11}]
    result = audit.grid_audit(rows, ["method", "seed"], {("ad", 11), ("exp", 29)})
    assert result == {"verified": 2, "expected": 2, "complete": True, "missing": []}

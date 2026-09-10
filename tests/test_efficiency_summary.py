import copy
import pytest
from experiments.summarize_efficiency import verify_timing


def valid_row():
    return {"length": 1024, "prefill": {"samples_seconds": [1., 3., 2.], "median_seconds": 2., "tokens_per_second": 2048.},
            "decode": {"samples_seconds": [2., 4., 3.], "median_seconds": 3., "tokens_per_second": 512 / 3},
            "decode_milliseconds_per_step": 3000 / 128}


def test_batch_and_decode_step_units_are_distinct():
    verify_timing(valid_row(), batch=4, decode_tokens=128, measurements=3)


@pytest.mark.parametrize("field", ["median", "throughput", "decode_step", "nonfinite_sample"])
def test_inconsistent_timing_units_or_samples_are_rejected(field):
    row = copy.deepcopy(valid_row())
    if field == "median":
        row["prefill"]["median_seconds"] = 1.
    elif field == "throughput":
        row["decode"]["tokens_per_second"] /= 4
    elif field == "decode_step":
        row["decode_milliseconds_per_step"] /= 4
    else:
        row["prefill"]["samples_seconds"][0] = float("nan")
    with pytest.raises(AssertionError):
        verify_timing(row, batch=4, decode_tokens=128, measurements=3)

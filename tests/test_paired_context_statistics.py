import math
import numpy as np
from experiments.summarize_context_controls import paired_intervals


def test_constant_paired_effect_and_direction():
    result = paired_intervals([.25] * 4, ["a", "a", "b", "b"])
    assert result["nll_difference"] == .25
    assert result["perplexity_ratio"] == math.exp(.25)
    np.testing.assert_allclose(result["document_bootstrap_95ci"], [.25, .25])
    np.testing.assert_allclose(result["hostname_cluster_bootstrap_95ci"], [.25, .25])


def test_cluster_resampling_retains_shared_hostname_effect():
    # Three low-effect pages from one site and one high-effect page from
    # another. Resampling entire sites must retain this dependence.
    result = paired_intervals([1., 1., 1., 9.], ["a", "a", "a", "b"])
    assert result["nll_difference"] == 3.
    np.testing.assert_allclose(result["document_bootstrap_95ci"], [1., 7.])
    np.testing.assert_allclose(result["hostname_cluster_bootstrap_95ci"], [1., 9.])


def test_reversing_comparison_negates_estimate_and_reverses_bounds():
    values = np.array([-.7, .2, .3, .6, -.2])
    hosts = ["a", "b", "a", "c", "d"]
    a, b = paired_intervals(values, hosts), paired_intervals(-values, hosts)
    assert a["nll_difference"] == -b["nll_difference"]
    for name in ["document_bootstrap_95ci", "hostname_cluster_bootstrap_95ci"]:
        np.testing.assert_allclose(a[name], -np.asarray(b[name])[::-1], atol=1e-15)

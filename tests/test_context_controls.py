import numpy as np
import pytest
import torch
from ad_kernel.model import LinearLanguageModel, ModelConfig
from experiments.evaluate_context_controls import context_and_targets, hidden_at_positions, score_tail
from experiments.evaluate_pretrained import hidden_states, token_nll


def test_every_context_scores_identical_global_target_indices():
    document = np.arange(8193)
    for length in [1024, 4096, 8192]:
        x, y = context_and_targets(document, length)
        np.testing.assert_array_equal(y, np.arange(7169, 8193))
        np.testing.assert_array_equal(x[-1024:], y - 1)
        assert x[0] == 8192 - length and x[-1] == 8191


@pytest.mark.parametrize("method", ["ad", "exp", "hh_exp", "favor", "elu", "softmax"])
def test_offset_zero_matches_frozen_forward_and_tail_target_loss(method):
    torch.manual_seed(112)
    config = ModelConfig(vocab_size=97, width=32, heads=4, layers=2, intermediate=64,
                         method=method, feature_dim=8, feature_hidden=12, backend="torch", fused_loss=False)
    model = LinearLanguageModel(config).eval()
    tokens = torch.randint(97, (2, 130))
    with torch.inference_mode():
        torch.testing.assert_close(hidden_at_positions(model, tokens[:, :-1]), hidden_states(model, tokens[:, :-1]), rtol=0, atol=0)
        expected = token_nll(model, tokens[:, :-1], tokens[:, 1:])[:, -17:]
        actual = score_tail(model, tokens[:, :-1], tokens[:, -17:])
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)


def test_softmax_rope_common_offset_preserves_relative_attention():
    torch.manual_seed(98)
    model = LinearLanguageModel(ModelConfig(vocab_size=97, width=32, heads=4, layers=2,
        intermediate=64, method="softmax", backend="torch", fused_loss=False)).eval()
    tokens = torch.randint(97, (2, 65))
    a = hidden_at_positions(model, tokens, offset=0)
    b = hidden_at_positions(model, tokens, offset=7168)
    torch.testing.assert_close(a, b, rtol=2e-4, atol=2e-5)

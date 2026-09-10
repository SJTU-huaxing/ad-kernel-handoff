import pytest
import torch
from torch.nn import functional as F
from ad_kernel.model import LinearLanguageModel, ModelConfig
from experiments.evaluate_pretrained import token_nll, totals


@pytest.mark.parametrize("method", ["ad", "exp", "hh_exp", "favor", "elu", "softmax"])
def test_per_token_nll_matches_frozen_model_and_target_alignment(method):
    torch.manual_seed(123)
    config = ModelConfig(vocab_size=97, width=32, heads=4, layers=2, intermediate=64,
                         method=method, feature_dim=8, feature_hidden=12, backend="torch", fused_loss=False)
    model = LinearLanguageModel(config).eval()
    tokens = torch.randint(97, (2, 258))
    with torch.inference_mode():
        expected = F.cross_entropy(model(tokens[:, :-1]).float().flatten(0, 1), tokens[:, 1:].flatten(), reduction="none").reshape(2, 257)
        actual = token_nll(model, tokens[:, :-1], tokens[:, 1:])
        mean = model(tokens[:, :-1], tokens[:, 1:])
    torch.testing.assert_close(actual.float(), expected, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(actual.mean().float(), mean, rtol=1e-6, atol=1e-6)
    summary = totals([{"tokens": 257, "nll_sum": row.sum().item()} for row in actual])
    assert summary["tokens"] == 514
    assert abs(summary["nll_per_token"] - actual.mean().item()) < 1e-12

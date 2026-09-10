import pytest
import torch
from ad_kernel.model import LinearLanguageModel, ModelConfig
from experiments.inference_runtime import LMRuntime


@pytest.mark.parametrize("method", ["ad", "exp", "hh_exp", "hh_softmax", "favor", "elu", "softplus", "softmax"])
def test_prefill_decode_and_actual_retained_storage(method):
    torch.manual_seed(991)
    config = ModelConfig(vocab_size=101, width=32, heads=4, layers=2, intermediate=64,
        method=method, feature_dim=16 if method.startswith("hh") else 8, feature_hidden=12,
        backend="torch", fused_loss=False)
    model = LinearLanguageModel(config).eval()
    runtime = LMRuntime(model)
    tokens = torch.randint(101, (2, 129))
    with torch.inference_mode():
        expected = model(tokens)
    first, cache = runtime.prefill(tokens[:, :63], all_logits=True)
    bytes63 = cache.tensor_bytes()
    middle, cache = runtime.prefill(tokens[:, 63:128], cache, all_logits=True)
    last, cache = runtime.step(tokens[:, 128], cache)
    actual = torch.cat([first, middle, last[:, None]], 1)
    torch.testing.assert_close(actual, expected, rtol=3e-5, atol=2e-6)
    assert cache.position == 129
    if method != "softmax":
        assert cache.tensor_bytes() == bytes63 == 2 * config.layers * config.heads * config.feature_dim * (8 + 2) * 4
    else:
        assert cache.tensor_bytes() > bytes63
    # A new call with no cache resets both positions and memory.
    fresh, reset = runtime.prefill(tokens[:, 64:], all_logits=True)
    with torch.inference_mode():
        torch.testing.assert_close(fresh, model(tokens[:, 64:]), rtol=3e-5, atol=2e-6)
    assert reset.position == 65

import pytest
import torch
from ad_kernel.model import LinearLanguageModel, ModelConfig


METHODS = ["ad", "exp", "hh_exp", "hh_softmax", "favor", "elu", "softplus", "softmax"]


def create(method):
    torch.manual_seed(123)
    return LinearLanguageModel(ModelConfig(vocab_size=73, width=32, layers=2, heads=4, intermediate=64,
        method=method, feature_dim=16 if method.startswith("hh") else 8, feature_hidden=12,
        backend="torch", fused_loss=False))


def test_same_backbone_initialization_and_fixed_favor():
    reference = {name: value for name, value in create("ad").state_dict().items() if ".features." not in name}
    for method in METHODS:
        model = create(method)
        for name, value in model.state_dict().items():
            if name in reference:
                assert torch.equal(value, reference[name]), (method, name)
        if method == "favor":
            assert not any(".features." in name for name, _ in model.named_parameters())


@pytest.mark.parametrize("method", METHODS)
def test_causal_lm_gradients(method):
    model = create(method)
    ids = torch.randint(73, (2, 17))
    logits = model(ids)
    changed = ids.clone()
    changed[:, 8:] = torch.randint(73, (2, 9))
    torch.testing.assert_close(logits[:, :8], model(changed)[:, :8], atol=2e-6, rtol=2e-6)
    loss = model(ids[:, :-1], ids[:, 1:])
    loss.backward()
    assert torch.isfinite(loss)
    for name, p in model.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), (method, name)
    if method == "ad":
        assert model.blocks[0].attention.features.knet.w2.grad[:, -1].abs().sum() > 0
    # Tied output and input embedding are represented by one parameter.
    assert not hasattr(model, "lm_head")

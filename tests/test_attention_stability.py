import pytest
import torch
from ad_kernel.attention import LinearState, chunk_attention, dense_attention, numerical_split_count


@pytest.mark.parametrize("amplitude", [100., 200., 1000.])
def test_future_amplitude_does_not_underflow_early_outputs_or_gradients(amplitude):
    torch.manual_seed(991)
    q, k = [torch.randn(1, 2, 129, 8) * 3 for _ in range(2)]
    k[:, :, 64:] += amplitude
    v, dy = [torch.randn(1, 2, 129, 6) for _ in range(2)]
    reference = [x.double().requires_grad_() for x in [q, k, v]]
    expected = dense_attention(*reference)
    gradients = torch.autograd.grad(expected, reference, dy.double())
    actual_inputs = [x.requires_grad_() for x in [q, k, v]]
    before = numerical_split_count()
    actual, _ = chunk_attention(*actual_inputs)
    assert numerical_split_count() > before
    measured = torch.autograd.grad(actual, actual_inputs, dy)
    torch.testing.assert_close(actual.double(), expected, rtol=2e-4, atol=3e-5)
    for a, b in zip(measured, gradients):
        assert torch.isfinite(a).all()
        torch.testing.assert_close(a.double(), b, rtol=2e-4, atol=3e-5)


def test_subdivision_preserves_initial_and_final_state_gradients():
    torch.manual_seed(151)
    q, k = [torch.randn(1, 2, 65, 8) for _ in range(2)]
    k[:, :, 32:] += 200
    values = [q, k, torch.randn(1, 2, 65, 6), torch.randn(1, 2, 8, 6),
              torch.rand(1, 2, 8) + 2, torch.randn(1, 2, 8)]
    weights = [torch.randn(1, 2, 65, 6), torch.randn(1, 2, 8, 6),
               torch.randn(1, 2, 8), torch.randn(1, 2, 8)]
    results = []
    for dtype in [torch.float64, torch.float32]:
        inputs = [x.to(dtype).requires_grad_() for x in values]
        out, state = chunk_attention(*inputs[:3], initial_state=LinearState(*inputs[3:]))
        outputs = [out, state.kv, state.z, state.log_scale]
        loss = sum((a * b.to(dtype)).sum() for a, b in zip(outputs, weights))
        results.append((outputs, torch.autograd.grad(loss, inputs)))
    for expected, actual in zip(results[0][0] + list(results[0][1]), results[1][0] + list(results[1][1])):
        assert torch.isfinite(actual).all()
        torch.testing.assert_close(actual.double(), expected, rtol=5e-4, atol=8e-5)


def test_empty_padded_state_does_not_suppress_negative_amplitude_keys():
    torch.manual_seed(99)
    q, k = [torch.randn(2, 2, 65, 8) for _ in range(2)]
    k -= 1000
    v = torch.randn(2, 2, 65, 6)
    empty_mask = torch.zeros(2, 7, dtype=torch.bool)
    empty, state = chunk_attention(q[:, :, :7], k[:, :, :7], v[:, :, :7], valid=empty_mask)
    assert torch.equal(empty, torch.zeros_like(empty))
    assert torch.isneginf(state.log_scale).all()
    actual, _ = chunk_attention(q, k, v, initial_state=state)
    expected = dense_attention(q.double(), k.double(), v.double())
    torch.testing.assert_close(actual.double(), expected, rtol=2e-4, atol=3e-5)

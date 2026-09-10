import pytest
import torch
from ad_kernel.attention import chunk_attention, dense_attention, recurrent_attention


@pytest.mark.parametrize("length", [1, 7, 63, 64, 65, 129])
def test_dense_chunk_recurrence_and_continuation(length):
    torch.manual_seed(14 + length)
    lq, lk = [torch.randn(2, 3, length, 8, dtype=torch.float64)*2 for _ in range(2)]
    v = torch.randn(2, 3, length, 5, dtype=torch.float64)
    target = dense_attention(lq, lk, v)
    chunk, state = chunk_attention(lq, lk, v, chunk_size=16)
    recurrent, rstate = recurrent_attention(lq, lk, v)
    torch.testing.assert_close(chunk, target, rtol=1e-11, atol=1e-11)
    torch.testing.assert_close(recurrent, target, rtol=1e-11, atol=1e-11)
    assert state.kv.shape == (2, 3, 8, 5)
    if length > 1:
        cut = length//2
        first, initial = chunk_attention(lq[:, :, :cut], lk[:, :, :cut], v[:, :, :cut], chunk_size=16)
        second, _ = recurrent_attention(lq[:, :, cut:], lk[:, :, cut:], v[:, :, cut:], initial)
        second_chunk, _ = chunk_attention(lq[:, :, cut:], lk[:, :, cut:], v[:, :, cut:],
                                          chunk_size=16, initial_state=initial)
        torch.testing.assert_close(torch.cat([first, second], 2), target, rtol=1e-11, atol=1e-11)
        torch.testing.assert_close(second_chunk, target[:, :, cut:], rtol=1e-11, atol=1e-11)


def test_padding_causality_and_document_reset():
    torch.manual_seed(62)
    lq, lk = [torch.randn(2, 2, 13, 8, dtype=torch.float64) for _ in range(2)]
    v = torch.randn(2, 2, 13, 6, dtype=torch.float64)
    valid = torch.ones(2, 13, dtype=torch.bool)
    valid[0, :2] = False
    valid[0, 10:] = False
    target = dense_attention(lq, lk, v, valid)
    actual, _ = chunk_attention(lq, lk, v, valid=valid, chunk_size=8)
    recurrent, _ = recurrent_attention(lq, lk, v, valid=valid)
    torch.testing.assert_close(actual, target, rtol=1e-11, atol=1e-11)
    torch.testing.assert_close(recurrent, target, rtol=1e-11, atol=1e-11)
    lk2, v2 = lk.clone(), v.clone()
    lk2[:, :, 7:] += 20
    v2[:, :, 7:] *= 100
    future_changed, _ = chunk_attention(lq, lk2, v2, valid=valid, chunk_size=8)
    torch.testing.assert_close(future_changed[:, :, :7], target[:, :, :7], rtol=1e-11, atol=1e-11)
    alone, _ = chunk_attention(lq[1:], lk[1:], v[1:], chunk_size=8)
    torch.testing.assert_close(alone, actual[1:], rtol=1e-11, atol=1e-11)


def test_gradients_and_gauge_invariance():
    torch.manual_seed(73)
    xs = [torch.randn(1, 2, 7, d, dtype=torch.float64, requires_grad=True) for d in [8, 8, 5]]
    ys = [x.detach().clone().requires_grad_() for x in xs]
    target = dense_attention(*xs)
    actual, _ = chunk_attention(*ys, chunk_size=4)
    dy = torch.randn_like(target)
    grads = torch.autograd.grad(target, xs, dy)
    other = torch.autograd.grad(actual, ys, dy)
    for a, b in zip(grads, other):
        torch.testing.assert_close(a, b, rtol=1e-10, atol=1e-11)
    query_scale = torch.randn(1, 2, 7, 1, dtype=torch.float64)*10
    shifted, _ = chunk_attention(xs[0]+query_scale, xs[1]+17, xs[2], chunk_size=4)
    torch.testing.assert_close(shifted, target, rtol=1e-10, atol=1e-11)
    key_amplitude = torch.linspace(-2, 2, 7, dtype=torch.float64)[None, None, :, None]
    changed, _ = chunk_attention(xs[0], xs[1]+key_amplitude, xs[2], chunk_size=4)
    assert (changed-target).abs().max() > 0.1
    assert torch.autograd.gradcheck(lambda q,k,v: chunk_attention(q,k,v,chunk_size=4)[0],
                                    tuple(xs), fast_mode=True)

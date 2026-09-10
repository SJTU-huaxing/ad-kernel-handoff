"""Inference-only adapter for the frozen 100M model; no training source changes.

Both methods retain their exact positive kernel, FP32 log features and FP32
recurrent state. NPUGraph replays update real state and a device RoPE position.
"""
from dataclasses import dataclass
import torch
from torch.nn import functional as F
import fla
from ad_kernel.attention import LinearState, chunk_attention, recurrent_attention


@dataclass
class Cache:
    position: int
    layers: list

    def clone(self):
        return Cache(self.position, [LinearState(s.kv.clone(), s.z.clone(), s.log_scale.clone())
                                     for s in self.layers])

    def tensor_bytes(self):
        storages = {t.untyped_storage().data_ptr(): t.untyped_storage().nbytes()
                    for s in self.layers for t in (s.kv, s.z, s.log_scale)}
        return sum(storages.values())


class Runtime:
    def __init__(self, model):
        self.model = model.eval()
        assert model.config.method in {"ad64", "hedgehog"}
        assert model.config.attention_backend == "cann"

    def _forward(self, tokens, states, position, *, inplace=False, all_logits=False):
        model, config = self.model, self.model.config
        b, t = tokens.shape
        angles = (torch.arange(t, device=tokens.device).float() + position)[:, None] * model.inv_freq
        phase = torch.cat((angles, angles), -1)[None, None]
        cos, sin = phase.cos(), phase.sin()
        hidden = model.embedding(tokens)
        final_states = []
        for block, old_state in zip(model.layers, states):
            attention = block.attn
            x = block.attn_norm(hidden)
            q, k, v = attention.qkv(x).reshape(b, t, 3, config.num_heads, attention.dim).permute(2, 0, 3, 1, 4).unbind(0)
            def rotate(value):
                value = value.float()
                first, second = value.chunk(2, -1)
                return value * cos + torch.cat((-second, first), -1) * sin
            with torch.autocast(tokens.device.type, enabled=False):
                q, k = rotate(q) * attention.dim ** (-.25), rotate(k) * attention.dim ** (-.25)
                lq = attention.features.log_feature(q, "q")
                lk = attention.features.log_feature(k, "k")
                if t == 1:
                    result, state = recurrent_attention(lq, lk, v.float(), initial_state=old_state)
                else:
                    result, state = chunk_attention(lq, lk, v.float(), initial_state=old_state,
                                                    backend="torch", chunk_size=config.chunk_size)
                    # cumsum slices can otherwise retain prompt-sized storage.
                    state = LinearState(state.kv.clone(), state.z.clone(), state.log_scale.clone())
                if inplace:
                    old_state.kv.copy_(state.kv)
                    old_state.z.copy_(state.z)
                    old_state.log_scale.copy_(state.log_scale)
                    state = old_state
            a = attention.out(result.to(v.dtype).transpose(1, 2).reshape(b, t, -1))
            if block.fuse_norm:
                h, residual = block.mlp_norm(a, hidden, True)
            else:
                residual = hidden + a
                h = block.mlp_norm(residual)
            hidden = residual + block.mlp(h)
            final_states.append(state)
        hidden = model.norm(hidden)
        logits = F.linear(hidden if all_logits else hidden[:, -1], model.embedding.weight)
        return logits, final_states

    def prefill(self, tokens, cache=None, *, all_logits=False):
        start = 0 if cache is None else cache.position
        assert start + tokens.shape[1] <= self.model.config.max_position_embeddings
        states = [None] * len(self.model.layers) if cache is None else cache.layers
        logits, states = self._forward(tokens, states, start, all_logits=all_logits)
        return logits, Cache(start + tokens.shape[1], states)

    def step(self, tokens, cache):
        return self.prefill(tokens.reshape(-1, 1), cache)


class GraphDecoder:
    """One real token per replay; fixed batch, variable cached context length."""
    def __init__(self, runtime, token, cache):
        self.runtime = runtime
        self.token = token.reshape(-1, 1).clone()
        self.cache = cache.clone()
        self.position = torch.tensor([cache.position], device=token.device, dtype=torch.float32)
        for _ in range(5):
            self._step()
        torch.npu.synchronize()
        self.reset(cache)
        self.graph = torch.npu.NPUGraph()
        with torch.npu.graph(self.graph):
            self.output = self._step()
        torch.npu.synchronize()
        self.reset(cache)

    def _step(self):
        logits, _ = self.runtime._forward(self.token, self.cache.layers, self.position, inplace=True)
        self.position.add_(1)
        return logits

    def reset(self, cache):
        self.position.fill_(cache.position)
        for dest, src in zip(self.cache.layers, cache.layers):
            dest.kv.copy_(src.kv)
            dest.z.copy_(src.z)
            dest.log_scale.copy_(src.log_scale)

    def step(self, token):
        self.token.copy_(token.reshape(-1, 1))
        self.graph.replay()
        return self.output

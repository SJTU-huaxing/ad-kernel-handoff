"""Inference-only prefill/decode without changing the frozen training source."""
from dataclasses import dataclass
import torch
from torch.nn import functional as F
from ad_kernel.attention import LinearState, chunk_attention, recurrent_attention
from ad_kernel.model import rotate


@dataclass
class Cache:
    position: int
    layers: list

    def tensor_bytes(self):
        total = 0
        for layer in self.layers:
            tensors = (layer.kv, layer.z, layer.log_scale) if isinstance(layer, LinearState) else layer
            total += sum(t.untyped_storage().nbytes() for t in tensors)
        return total


class LMRuntime:
    def __init__(self, model):
        self.model = model.eval()

    @torch.inference_mode()
    def prefill(self, tokens, cache=None, *, all_logits=False):
        model, config = self.model, self.model.config
        b, t = tokens.shape
        start = 0 if cache is None else cache.position
        previous = [None] * config.layers if cache is None else cache.layers
        assert len(previous) == config.layers
        positions = torch.arange(start, start + t, device=tokens.device).float()
        angles = positions[:, None] * model.inverse_frequency
        phase = torch.cat([angles, angles], -1)[None, None]
        cos, sin = phase.cos(), phase.sin()
        hidden, layers = model.embedding(tokens), []
        for block, old in zip(model.blocks, previous):
            attention = block.attention
            x = block.attention_norm(hidden)
            q, k, v = attention.qkv(x).reshape(b, t, 3, config.heads, attention.dim).permute(2, 0, 3, 1, 4).unbind(0)
            q, k = rotate(q, cos, sin), rotate(k, cos, sin)
            if attention.features is None:
                q, k = q.to(v.dtype), k.to(v.dtype)
                if old is not None:
                    k, v = torch.cat([old[0], k], dim=2), torch.cat([old[1], v], dim=2)
                if start == 0:
                    out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
                elif t == 1:
                    out = F.scaled_dot_product_attention(q, k, v, is_causal=False)
                else:
                    mask = torch.arange(start + t, device=tokens.device)[None] <= torch.arange(start, start + t, device=tokens.device)[:, None]
                    out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
                # Projected Q/K/V are views of a shared qkv allocation. Clone
                # on the first prefill so retained storage contains only K/V.
                layers.append((k.clone(), v.clone()) if old is None else (k, v))
            else:
                with torch.autocast(device_type=tokens.device.type, enabled=False):
                    q, k = q * attention.dim ** (-.25), k * attention.dim ** (-.25)
                    lq, lk = attention.features.log_feature(q, "q"), attention.features.log_feature(k, "k")
                    if t == 1:
                        out, state = recurrent_attention(lq, lk, v.float(), initial_state=old)
                    else:
                        out, state = chunk_attention(lq, lk, v.float(), initial_state=old, backend="torch")
                        # Last-state views otherwise retain whole-sequence
                        # cumsum storage. Compact only at inference boundaries.
                        state = LinearState(state.kv.clone(), state.z.clone(), state.log_scale.clone())
                layers.append(state)
                out = out.to(v.dtype)
            hidden = hidden + attention.out(out.transpose(1, 2).reshape(b, t, -1))
            gate, value = block.up(block.mlp_norm(hidden)).chunk(2, -1)
            hidden = hidden + block.down(F.silu(gate) * value)
        hidden = model.final_norm(hidden)
        logits = F.linear(hidden if all_logits else hidden[:, -1], model.embedding.weight)
        return logits, Cache(start + t, layers)

    @torch.inference_mode()
    def step(self, token, cache):
        return self.prefill(token.reshape(-1, 1), cache)

"""A finite, checkpoint-independent permutation: exactly target_tokens labels.

No online tokenization, downloads or repeated target positions. A sample shares
one boundary input token with its neighbor but their target positions are disjoint.
"""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import math
import numpy as np
import torch


@dataclass
class StepBatch:
    index: int
    target_count: int
    inputs: torch.Tensor
    labels: torch.Tensor


class TokenPlan:
    def __init__(self, target_tokens, sequence_length, global_batch, world_size, micro_batch, seed):
        assert target_tokens > 0 and global_batch % (world_size * micro_batch) == 0
        self.target_tokens, self.seq, self.global_batch = target_tokens, sequence_length, global_batch
        self.world, self.micro = world_size, micro_batch
        self.blocks = math.ceil(target_tokens / sequence_length)
        self.steps = math.ceil(self.blocks / global_batch)
        # Store the exact permutation hash in the run binding.
        self.order = torch.randperm(self.blocks, generator=torch.Generator().manual_seed(seed)).numpy()
        self.partial_position = int(np.flatnonzero(self.order == self.blocks-1)[0])

    def block_lengths(self, block_ids):
        return np.where(block_ids >= 0, np.minimum(self.seq, np.maximum(0, self.target_tokens - block_ids * self.seq)), 0)

    def counts(self, step):
        ids = self.order[step*self.global_batch:(step+1)*self.global_batch]
        return int(self.block_lengths(ids).sum())

    def tokens_before(self, step):
        count = min(step * self.global_batch, self.blocks)
        full = count * self.seq
        partial = self.target_tokens % self.seq
        if partial and count > self.partial_position:
            full -= self.seq - partial
        return full

    def batch(self, stream, step, rank, pin=False):
        ids = self.order[step*self.global_batch:(step+1)*self.global_batch]
        total = int(self.block_lengths(ids).sum())
        accum = math.ceil(len(ids) / (self.world*self.micro))
        ids = np.pad(ids, (0, accum*self.world*self.micro-len(ids)), constant_values=-1)
        ids = ids.reshape(accum, self.world, self.micro)[:, rank].copy()
        offsets = np.maximum(ids, 0)[..., None] * self.seq + np.arange(self.seq+1)
        values = np.array(stream[np.minimum(offsets, self.target_tokens)], dtype=np.int64)
        inputs, labels = values[..., :-1].copy(), values[..., 1:].copy()
        lengths = self.block_lengths(ids)
        labels[np.arange(self.seq)[None, None, :] >= lengths[..., None]] = -100
        x, y = torch.from_numpy(inputs), torch.from_numpy(labels)
        if pin:
            x, y = x.pin_memory(), y.pin_memory()
        return StepBatch(step, total, x, y)

    def prefetch(self, stream, begin, end, rank, pin=False):
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="token-prefetch") as pool:
            future = None
            for step in range(begin, end):
                batch = future.result() if future is not None else self.batch(stream, step, rank, pin)
                future = pool.submit(self.batch, stream, step+1, rank, pin) if step+1 < end else None
                yield batch

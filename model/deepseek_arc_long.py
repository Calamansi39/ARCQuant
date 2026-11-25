"""Bound ARC activation temporaries for long prompts; HiF4 groups stay intact."""
import torch

from .deepseek_generation_arc import DeepSeekGenerationArcMixin


class LongArcMixin(DeepSeekGenerationArcMixin):
    # Chunk rows, never channels: HiF4's independent 64-channel scales and ARC
    # residual channel selection therefore remain unchanged. Dense only.
    arc_chunk_rows = 4096

    @torch.no_grad()
    def _project(self, x, weight, key, bias=None, numeric=None):
        if self.sparse_sparsity:
            raise ValueError('Long ARC adapter currently supports dense controls only')
        rows = x.numel() // x.shape[-1]
        if rows <= self.arc_chunk_rows:
            return super()._project(x, weight, key, bias, numeric)
        flat = x.reshape(rows, x.shape[-1])
        output = None
        for start in range(0, rows, self.arc_chunk_rows):
            value = super()._project(flat[start:start+self.arc_chunk_rows], weight, key, bias, numeric)
            if output is None:
                output = value.new_empty((rows, value.shape[-1]))
            output[start:start+value.shape[0]] = value
        return output.reshape(*x.shape[:-1], output.shape[-1])

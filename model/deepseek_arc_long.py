"""Bound ARC activation temporaries for long prompts; HiF4 groups stay intact."""
import torch

from .deepseek_generation_arc import DeepSeekGenerationArcMixin


class LongArcMixin(DeepSeekGenerationArcMixin):
    # Chunk rows, never channels: HiF4's independent 64-channel scales and ARC
    # residual channel selection therefore remain unchanged.
    arc_chunk_rows = 4096

    def _mask(self,x,weight,key):
        if getattr(self,'_arc_already_masked',False):
            return x
        return super()._mask(x,weight,key)

    @torch.no_grad()
    def _project(self, x, weight, key, bias=None, numeric=None):
        rows = x.numel() // x.shape[-1]
        if rows <= self.arc_chunk_rows:
            return super()._project(x, weight, key, bias, numeric)
        # Mask with ORIGINAL sequence shape/token positions exactly once.
        # Flattening before selection would sparsify protected prefill tokens;
        # resetting the rule per chunk would create a different experiment.
        masked=self._mask(x,weight,key)
        flat = masked.reshape(rows, x.shape[-1])
        output = None
        previous=getattr(self,'_arc_already_masked',False)
        self._arc_already_masked=True
        try:
            for start in range(0, rows, self.arc_chunk_rows):
                value = super()._project(flat[start:start+self.arc_chunk_rows], weight, key, bias, numeric)
                if output is None:
                    output = value.new_empty((rows, value.shape[-1]))
                output[start:start+value.shape[0]] = value
        finally:
            self._arc_already_masked=previous
        return output.reshape(*x.shape[:-1], output.shape[-1])

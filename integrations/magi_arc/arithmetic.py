"""ARC HiF4 primitives; identical channel layout to model.quantize's oracle.

Rows are chunked, channels are not. Residuals use FP32 reconstruction before
BF16 rounding. Batched experts may have different numbers of residual channels;
zero padding aligns the GEMM without changing any expert's arithmetic.
"""
import math
import torch


def quantize(x, backend='reference'):
    if backend == 'triton' and x.is_cuda:
        from magi2_fast_quant import quantize_cuda
        return quantize_cuda(x.contiguous().float(), 'hifp4')
    from model.quantize import quantize_hif4_tensor
    return quantize_hif4_tensor(x.contiguous().float())


def validate(spec, experts, width):
    order, selected = spec['order'], spec['select']
    if (order.shape != (experts,width) or order.dtype != torch.int64
            or selected.shape != (experts,) or selected.dtype != torch.int64
            or not torch.equal(order.sort(-1).values,torch.arange(width).expand(experts,width))
            or bool(((selected<0)|(selected>width)|(selected%64!=0)).any())):
        raise ValueError('Invalid per-expert ARC permutation/residual count')


class Observer:
    def __init__(self, experts, width):
        self.maximum=torch.zeros(experts,width)
        self.rows=torch.zeros(experts,dtype=torch.int64)
        self.selected=torch.zeros_like(self.rows)

    def update(self,x,first=0,valid_rows=None):
        counts=[x.shape[1]]*len(x) if valid_rows is None else valid_rows.cpu().tolist()
        for offset,count in enumerate(counts):
            if not count: continue
            index=first+offset
            # Only reductions cross PCIe. Never copy a full video activation.
            for part in x[offset,:count].split(256):
                part=part.detach().float()
                if not torch.isfinite(part).all(): raise ValueError('Nonfinite ARC calibration')
                self.maximum[index]=torch.maximum(self.maximum[index],part.abs().amax(0).cpu())
                self.rows[index]+=len(part)
                self.selected[index]+=int((part>part.amax(-1,keepdim=True)/8).sum())

    def finish(self):
        if bool((self.rows==0).any()):
            raise ValueError('ARC calibration has unobserved experts; collect more calibration prompts')
        width=self.maximum.shape[1]
        if width%64: raise ValueError('ARC MAGI width must be divisible by 64')
        return dict(order=self.maximum.argsort(dim=-1,stable=True),
            select=torch.tensor([min(width,math.ceil(int(n)/int(r)/64)*64)
                                 for n,r in zip(self.selected,self.rows)],dtype=torch.int64),rows=self.rows)


@torch.no_grad()
def activation(x,spec,first=0,backend='reference'):
    """x already masked in original/WINA coordinates, BF16 [E,M,K]."""
    e,m,k=x.shape
    selected=spec['select'][first:first+e].tolist()
    extra=max(selected,default=0)
    out=x.new_zeros((e,m,k+extra))
    for i,count in enumerate(selected):
        order=spec['order'][first+i].to(x.device)
        for start in range(0,m,256):
            original=x[i,start:start+256].float().index_select(-1,order)
            q=quantize(original,backend)
            out[i,start:start+len(q),:k]=q.to(x.dtype)
            if count:
                residual=quantize(original[:,-count:]-q[:,-count:],backend)
                out[i,start:start+len(q),k:k+count]=residual.to(x.dtype)
    return out


@torch.no_grad()
def weight(w,spec,backend='reference',device='cpu'):
    """Original (possibly WINA-folded) [E,N,K], return main and CPU residuals."""
    e,n,k=w.shape
    validate(spec,e,k)
    extra=int(spec['select'].max())
    residual=w.new_zeros((e,n,extra),device='cpu')
    for i in range(e):
        order=spec['order'][i].to(device)
        count=int(spec['select'][i])
        for start in range(0,n,256):
            # Read original weights once. Quantizing twice would lose residual information.
            value=w[i,start:start+256].to(device).float().index_select(-1,order)
            q=quantize(value,backend)
            w[i,start:start+len(q)].copy_(q.to(w.device,w.dtype))
            if count:
                residual[i,start:start+len(q),:count]=quantize(value[:,-count:],backend).to('cpu',w.dtype)
    return residual


def expanded_weight(main,residual,spec,first=0):
    e=len(main)
    extra=int(spec['select'][first:first+e].max()) if e else 0
    return torch.cat((main,residual[first:first+e,:,:extra].to(main.device)),dim=-1) if extra else main

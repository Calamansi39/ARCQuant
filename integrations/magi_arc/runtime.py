"""MAGI grouped/dense/routed projection adapters; install before CPU backups."""
from types import MethodType

import torch
import torch.nn.functional as F

from integrations.magi_arc.assets import basis_for, fold_, column_norms
from integrations.magi_arc.reference import keep_count

GROUPED = ('GroupedLinearBase', 'CustomGroupedLinear', 'BaseLinear', 'NativeMoELinear')


class Projection:
    def __init__(self, name, weights, context, *, rotate=False, ep_sharded=False):
        self.name, self.context = name, context
        self.experts, _, self.width = weights[0].shape
        self.parameters_per_expert = sum(w[0].numel() for w in weights)
        self.ep_sharded = ep_sharded
        self.expert_start = context.rank * self.experts if ep_sharded else 0
        self.sparsity = context.allocation.get(name, 0.)
        self.basis, self.asset = None, None
        if context.method == 'wina' and rotate:
            self.basis, self.asset = basis_for(weights, context.asset_dir, name + f'.ep{self.expert_start}', device=context.device)
            fold_(weights, self.basis, device=context.device)
        self.norms = None
        self.weights = weights
        from .arithmetic import Observer, validate
        self.collector = Observer(self.experts,self.width) if context.collect else None
        self.spec = None if context.collect else context.specs[name]
        if self.spec is not None:
            validate(self.spec,self.experts,self.width)
            if self.spec.get('basis_sha256')!=(self.asset['sha256'] if self.asset else None):
                raise ValueError('ARC statistics were collected in different WINA coordinates')
        self.residuals=[]
        self.calls=0
        context.registry[name] = self

    def finish(self):
        if not self.context.collect:
            from .arithmetic import weight
            self.residuals=[weight(w,self.spec,self.context.backend,self.context.device) for w in self.weights]
        if self.context.method == 'wina':
            self.norms = column_norms(self.weights, device=self.context.device)
            if self.spec is not None:
                # Weights are now reordered. WINA scores remain in pre-ARC coordinates.
                self.norms=self.norms.gather(1,self.spec['order'].argsort(-1))
        # Do not retain views: module .to() replaces Parameter storage.
        del self.weights

    def metadata(self):
        return dict(width=self.width, parameters_per_expert=self.parameters_per_expert,
                    experts=self.experts, expert_start=self.expert_start,
                    ep_sharded=self.ep_sharded, sparsity=self.sparsity)

    def apply(self, x, *, first=0, valid_rows=None):
        from integrations.magi_arc import kernels, reference
        context = self.context
        if self.basis is not None:
            q = self.basis[first:first+len(x)].to(x.device)
            x = torch.bmm(x.float(), q)
        if self.collector is not None:
            self.collector.update(x,first,valid_rows)
        norms = self.norms[first:first+len(x)].to(x.device) if self.norms is not None else None
        fn = kernels.select if context.backend == 'triton' else reference.select
        kwargs = dict(norms=norms, quant=context.quant if context.collect else 'bf16', valid_rows=valid_rows,
                      statistics=context.observer.enabled)
        result, hist = fn(x, keep_count(self.width,self.sparsity), **kwargs)
        self.calls+=1
        if not context.collect:
            from .arithmetic import activation
            result=activation(result,self.spec,first,context.backend)
            if context.observer.enabled:
                # Histograms describe main channels in their ORIGINAL groups;
                # residual-channel overhead is reported separately, never as sparse speedup.
                original=result[...,:self.width].gather(-1,self.spec['order'][first:first+len(x)].argsort(-1).to(x.device)[:,None,:].expand(-1,x.shape[1],-1))
                lengths=[x.shape[1]]*len(x) if valid_rows is None else valid_rows.cpu().tolist()
                for i,n in enumerate(lengths):
                    hist[i,2]=torch.bincount((original[i,:n]!=0).reshape(-1,4).sum(-1),minlength=5)
        context.observer.add(self,hist,first)
        return result

    def expanded(self,main,index=0,first=0):
        if self.context.collect: return main
        from .arithmetic import expanded_weight
        return expanded_weight(main,self.residuals[index],self.spec,first)


class Context:
    def __init__(self, method, quant, allocation, asset_dir, *, rank=0, world=2, backend='triton', device='cuda', collect=False, specs=None):
        from integrations.magi_arc.metrics import Observer
        self.method, self.quant, self.allocation = method, quant, allocation
        self.asset_dir, self.rank, self.world, self.backend, self.device = asset_dir, rank, world, backend, device
        self.registry, self.observer = {}, Observer()
        if quant!='hifp4': raise ValueError('MAGI ARC requires HiF4 W4A4')
        self.collect=collect
        if not collect and specs is None:
            from .calibration import load_specs
            specs=load_specs(method,rank)
        self.specs=specs

    def configure(self, allocation):
        if set(allocation) != set(self.registry):
            raise ValueError('Allocation does not cover the installed projections exactly')
        for name, value in allocation.items():
            keep_count(self.registry[name].width, value)
            self.registry[name].sparsity = value

    def sizes(self):
        return {n:p.parameters_per_expert*p.experts*(self.world if p.ep_sharded else 1)
                for n,p in self.registry.items()}

    def fingerprints(self):
        from integrations.magi_arc.assets import tensor_hash
        return {n:dict(basis_sha256=p.asset['sha256'] if p.asset else None,
                       norms_sha256=tensor_hash(p.norms) if p.norms is not None else None,
                       arc_order_sha256=tensor_hash(p.spec['order']) if p.spec is not None else None,
                       arc_select_sha256=tensor_hash(p.spec['select']) if p.spec is not None else None)
                for n,p in self.registry.items()}

    def arc_report(self):
        return dict(arc=True,quantization='HiF4 W4A4 + ARC',method=self.method,
            kernel='HiF4 reconstructed values + dense GEMM',
            projections={n:dict(calls=p.calls,width=p.width,
                residual_channels=p.spec['select'].tolist()) for n,p in self.registry.items()})


def _grouped(self, input, cu_seqlens=None, m_splits=None, gather_ids=None):
    if self.num_experts == 1:
        splits = [len(input)]
    else:
        if m_splits is None:
            if cu_seqlens is None:
                raise ValueError('Grouped projection has no modality splits')
            m_splits = cu_seqlens.diff()
        splits = m_splits.cpu().tolist() if torch.is_tensor(m_splits) else list(m_splits)
    if gather_ids is not None:
        input = input.index_select(0,gather_ids) if gather_ids.ndim == 1 else torch.gather(input,0,gather_ids)
    weights = self.weight.view(self.num_experts,self.out_features,self.in_features)
    biases = self.bias.view(self.num_experts,self.out_features) if self.bias is not None else None
    out = []
    for i,x in enumerate(input.split(splits,dim=0)):
        x = self._sparse_projection.apply(x.unsqueeze(0), first=i)[0]
        w=self._sparse_projection.expanded(weights[i:i+1],first=i)[0]
        out.append(F.linear(x,w,None if biases is None else biases[i]))
    return torch.cat(out,dim=0)


def _custom(self, input, modality_dispatcher=None, gather_ids=None):
    if hasattr(modality_dispatcher,'group_size_cpu'):
        splits = modality_dispatcher.group_size_cpu
    elif torch.is_tensor(modality_dispatcher):
        splits = modality_dispatcher.tolist()
    else:
        splits = modality_dispatcher
    return _grouped(self,input,m_splits=splits,gather_ids=gather_ids)


def _refiner(self, input, output_dtype=None, modality_dispatcher=None):
    dtype = input.dtype if output_dtype is None else output_dtype
    chunks = modality_dispatcher.dispatch(input) if self.num_experts > 1 else [input]
    weights = self.weight.chunk(self.num_experts,0)
    biases = self.bias.chunk(self.num_experts,0) if self.bias is not None else None
    out = []
    for i,x in enumerate(chunks):
        x = self._sparse_projection.apply(x.to(torch.bfloat16).unsqueeze(0),first=i)[0]
        w=self._sparse_projection.expanded(weights[i].unsqueeze(0),first=i)[0]
        y = x @ w.to(torch.bfloat16).T
        if biases is not None:
            y = y + biases[i].to(torch.bfloat16)
        out.append(y.to(dtype))
    return modality_dispatcher.undispatch(*out) if self.num_experts > 1 else out[0]


def _experts(self,x,probs,indices):
    from integrations.magi_arc.moe import routed_experts
    return routed_experts(self,x,probs,indices)


@torch.no_grad()
def install(model, prefix, context, *, quantize=True):
    """prefix is 'preview'/'refiner' or 'preview.block.layers.N' for replay.

    Allocation units are logical projection families: modalities and routed
    experts share one rate, but selection and norms are per matrix/per token.
    """
    import magi2_quant
    pairs, routed = [], []
    for name,module in list(model.named_modules()):
        full = prefix + ('.'+name if name else '')
        cls = type(module).__name__
        if cls == 'CoreMultiHeadMoE':
            wg,wu,wd = [getattr(module,k).transpose(1,2) for k in ('W_gate','W_up','W_down')]
            module._sparse_gate_up = Projection(full+'.gate_up',[wg,wu],context,rotate=True,ep_sharded=True)
            module._sparse_down = Projection(full+'.down',[wd],context,ep_sharded=True)
            routed.append(module)
        elif cls in GROUPED and not any(s in name.lower() for s in ('embed','adapter','linear_g')):
            w = module.weight.view(module.num_experts,module.out_features,module.in_features)
            rotate = name.endswith(('linear_qkv','up_gate_proj','shared_expert_fc1'))
            module._sparse_projection = Projection(full,[w],context,rotate=rotate)
            pairs.append((module,cls))
    # The existing quantizer retains exclusions and the baseline rounding.
    quantizer = getattr(context, 'quantizer', magi2_quant.apply_magi2_quant)
    if quantize and not context.collect:
        # Preserve the native HiF4 treatment of final heads and attention gates,
        # which are outside the sparse/ARC block scope. Skip only our registered
        # raw weights so ARC does NOT reorder already-quantized values.
        storage={w.untyped_storage().data_ptr() for p in context.registry.values()
                 if hasattr(p,'weights') for w in p.weights}
        original_weight,original_expert=magi2_quant.quantize_weight_,magi2_quant.quantize_expert_weights_
        def single(w,*args,**kwargs):
            if w.untyped_storage().data_ptr() not in storage: return original_weight(w,*args,**kwargs)
        def expert(w,*args,**kwargs):
            if w.untyped_storage().data_ptr() not in storage: return original_expert(w,*args,**kwargs)
        magi2_quant.quantize_weight_,magi2_quant.quantize_expert_weights_=single,expert
        try:
            report=quantizer(model,context.quant,True,device=context.device)
        finally:
            magi2_quant.quantize_weight_,magi2_quant.quantize_expert_weights_=original_weight,original_expert
    else:
        report=quantizer(model,context.quant,True,device=context.device) if quantize else {}
    for p in context.registry.values():
        if hasattr(p,'weights'):
            p.finish()
    for module,cls in pairs:
        module.forward = MethodType({'GroupedLinearBase':_grouped,'CustomGroupedLinear':_custom}.get(cls,_refiner),module)
    for module in routed:
        module._flash_forward = MethodType(_experts,module)
    if not pairs:
        raise ValueError(f'No MAGI projections found under {prefix}')
    if not context.collect:
        report.update(arc=True,quantization='HiF4 W4A4 + ARC',
            arc_projections=[n for n in context.registry if n.startswith(prefix+'.')],
            residual_storage='CPU; streamed by bounded expert batch; native dense GEMM')
    return report

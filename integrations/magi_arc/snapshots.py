"""Full block snapshots in safetensors, including MAGI attention metadata."""
import importlib
import json
import os
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from integrations.magi_arc.protocol import sha256
from vbench_generation_cache import atomic_write_json

ALLOWED_TYPES = {
    'inference.model.magi2_preview': {'ModalityDispatcher', 'VarlenHandler'},
    'inference.model.magi2_refiner': {'ModalityDispatcher', 'VarlenHandler', 'FFAHandler'},
    # The live refiner pipeline constructs these data-proxy classes, rather
    # than the similarly named annotation classes in the model module.
    'inference.pipeline.refiner_data_proxy': {'VarlenHandler', 'WindowLocalAttnHandler'},
}


def encode(value, tensors):
    if torch.is_tensor(value):
        key = f't{len(tensors)}'
        tensors[key] = value.detach().cpu().contiguous().clone()
        return dict(tensor=key,device=value.device.type)
    if isinstance(value,tuple):
        return dict(tuple=[encode(v,tensors) for v in value])
    if isinstance(value,list):
        return dict(list=[encode(v,tensors) for v in value])
    if isinstance(value,dict):
        return dict(dict={k:encode(v,tensors) for k,v in value.items()})
    if value is None or isinstance(value,(str,int,float,bool)):
        return dict(value=value)
    cls = type(value)
    if cls.__name__ not in ALLOWED_TYPES.get(cls.__module__, set()):
        raise TypeError(f'Unsupported snapshot value: {cls}')
    fields = (dict(modality_mapping=value.modality_mapping,num_modalities=value.num_modalities)
              if cls.__name__ == 'ModalityDispatcher' else vars(value))
    return dict(object=cls.__name__,module=cls.__module__,fields=encode(fields,tensors))


def decode(tree,tensors,device='cuda'):
    if 'tensor' in tree:
        return tensors[tree['tensor']].to(device if tree['device']=='cuda' else 'cpu')
    if 'tuple' in tree:
        return tuple(decode(v,tensors,device) for v in tree['tuple'])
    if 'list' in tree:
        return [decode(v,tensors,device) for v in tree['list']]
    if 'dict' in tree:
        return {k:decode(v,tensors,device) for k,v in tree['dict'].items()}
    if 'object' in tree:
        if tree['object'] not in ALLOWED_TYPES.get(tree['module'], set()):
            raise ValueError('Unknown snapshot object type')
        cls = getattr(importlib.import_module(tree['module']),tree['object'])
        fields = decode(tree['fields'],tensors,device)
        if tree['object'] == 'ModalityDispatcher':
            return cls(**fields)
        result = cls.__new__(cls)
        result.__dict__.update(fields)
        return result
    return tree['value']


def save(path,value,identity):
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    tensors = {}
    tree = encode(value,tensors)
    tmp = path.with_suffix(f'.{os.getpid()}.tmp')
    save_file(tensors,str(tmp),metadata={'tree':json.dumps(tree),'identity':json.dumps(identity,sort_keys=True)})
    os.replace(tmp,path)
    atomic_write_json(path.with_suffix('.json'),dict(identity=identity,sha256=sha256(path)))


def load(path,identity=None,device='cuda'):
    tree,tensors=read(path,identity)
    return decode(tree,tensors,device)


def read(path,identity=None):
    path = Path(path)
    record = json.loads(path.with_suffix('.json').read_text())
    if (identity is not None and record['identity'] != identity) or sha256(path) != record['sha256']:
        raise ValueError(f'Incompatible/corrupt snapshot: {path}')
    with safe_open(str(path),framework='pt',device='cpu') as f:
        if json.loads(f.metadata()['identity']) != record['identity']:
            raise ValueError('Snapshot identity mismatch')
        return json.loads(f.metadata()['tree']),{k:f.get_tensor(k) for k in f.keys()}


class Capture:
    def __init__(self,root,identity,steps):
        self.root,self.identity,self.steps = Path(root),identity,steps
        self.models = {}
        self.begin(0)

    def begin(self,prompt):
        self.prompt,self.calls,self.paths,self.outputs = prompt,{},{},{}

    def install(self,component,model):
        self.models[component] = model
        def before(*_):
            self.calls[component] = self.calls.get(component,-1)+1
        model.register_forward_pre_hook(before)
        for name,block in model.named_modules():
            if type(block).__name__ != 'TransformerLayer':
                continue
            if name=='block.layers.0':
                def entry(mod,args,kwargs):
                    wanted=round(self.prompt*(self.steps[component]-1)/9)
                    if self.calls[component]!=wanted:
                        return
                    path=self.root/f'prompt-{self.prompt:02d}'/f'rank{os.environ["RANK"]}'/component/'block.layers.0.safetensors'
                    save(path,dict(args=args,kwargs=kwargs),dict(self.identity,prompt_index=self.prompt,
                        component=component,block='block.layers.0',forward=wanted))
                    self.paths[f'{component}.block.layers.0']=dict(path=str(path),sha256=sha256(path))
                block.register_forward_pre_hook(entry,with_kwargs=True)
            def capture(mod,args,kwargs,output,name=name):
                wanted = round(self.prompt*(self.steps[component]-1)/9)
                if self.calls[component] != wanted:
                    return
                if not torch.isfinite(output).all():
                    raise ValueError('Nonfinite dense snapshot')
                from integrations.magi_arc.assets import tensor_hash
                self.outputs[f'{component}.{name}']=dict(sha256=tensor_hash(output.detach().cpu()),
                                                       shape=list(output.shape),dtype=str(output.dtype))
            block.register_forward_hook(capture,with_kwargs=True)

    def finish(self):
        counts = {k:v+1 for k,v in self.calls.items()}
        if counts != self.steps:
            raise ValueError(f'Unexpected model call counts {counts}; expected {self.steps}')
        expected = {f'{c}.{n}' for c,m in self.models.items() for n,b in m.named_modules()
                    if type(b).__name__ == 'TransformerLayer'}
        if set(self.outputs) != expected or len(self.paths)!=len(self.models):
            raise ValueError('Incomplete calibration block coverage')
        atomic_write_json(self.root/f'prompt-{self.prompt:02d}'/f'rank{os.environ["RANK"]}'/'complete.json',
                          dict(identity=self.identity,paths=self.paths,outputs=self.outputs,forward_calls=counts))

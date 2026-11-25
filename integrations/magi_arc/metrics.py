"""Integer 2:4 histograms with explicit CP/EP ownership and coverage."""
import torch

from integrations.magi_arc.reference import STAGES


def describe(hist):
    total = sum(hist)
    zeros = sum((4-n)*v for n,v in enumerate(hist))
    return dict(nonzero_histogram=hist, groups=total, zero_elements=zeros,
        element_sparsity=zeros/(4*total) if total else None,
        eligible_2of4_fraction=sum(hist[:3])/total if total else None,
        exactly_2_nonzero_fraction=hist[2]/total if total else None,
        all_zero_fraction=hist[0]/total if total else None)


class Observer:
    """Not a module buffer: roundtrip offload must not restore old counts."""
    def __init__(self):
        self.enabled = True
        self.steps = {}
        self.rows = {}

    def reset(self):
        self.steps.clear()
        self.rows.clear()

    def before_component(self, component):
        self.steps[component] = self.steps.get(component, -1) + 1

    def add(self, projection, hist, first):
        if not self.enabled:
            return
        name = projection.name
        if name not in self.rows:
            self.rows[name] = dict(
                hist=torch.zeros((projection.experts,3,5), device=hist.device, dtype=torch.int64),
                steps={}, calls=0, metadata=projection.metadata())
        row = self.rows[name]
        row['hist'][first:first+len(hist)] += hist
        step = self.steps.get(name.split('.')[0], 0)
        if step not in row['steps']:
            row['steps'][step] = torch.zeros((3,5), device=hist.device, dtype=torch.int64)
        row['steps'][step] += hist.sum(0)
        row['calls'] += 1

    def export(self, rank, registry):
        layers = {}
        for name, projection in registry.items():
            row = self.rows.get(name)
            layers[name] = dict(metadata=projection.metadata(), calls=row['calls'] if row else 0,
                hist=row['hist'].cpu().tolist() if row else [[[0]*5 for _ in range(3)] for _ in range(projection.experts)],
                steps={str(k):v.cpu().tolist() for k,v in row['steps'].items()} if row else {})
        return dict(rank=rank, component_forward_calls={k:v+1 for k,v in self.steps.items()}, layers=layers)


def merge_ranks(records):
    """CP shards sum their token counts; EP shards have disjoint expert IDs."""
    if sorted(r['rank'] for r in records) != list(range(len(records))):
        raise ValueError('Missing/duplicate rank statistics')
    if len({tuple(sorted(r['component_forward_calls'].items())) for r in records}) != 1:
        raise ValueError('Distributed denoiser forward counts disagree')
    names = set(records[0]['layers'])
    if any(set(r['layers']) != names for r in records):
        raise ValueError('Distributed projection coverage differs')
    merged = {}
    for name in sorted(names):
        experts = {}
        steps = {}
        metadata = records[0]['layers'][name]['metadata']
        for record in records:
            row = record['layers'][name]
            meta = row['metadata']
            if any(meta[k] != metadata[k] for k in ('parameters_per_expert','width','sparsity','ep_sharded')):
                raise ValueError('Inconsistent projection metadata')
            expected_start=record['rank']*meta['experts'] if meta['ep_sharded'] else 0
            if meta['expert_start']!=expected_start or len(row['hist'])!=meta['experts']:
                raise ValueError('Invalid CP/EP expert ownership')
            expected_steps=record['component_forward_calls'].get(name.split('.')[0],0)
            if set(row['steps'])!={str(i) for i in range(expected_steps)}:
                raise ValueError(f'Incomplete rank/step coverage: rank {record["rank"]}, {name}')
            # Verify integer totals independently across the step and expert
            # axes before any distributed pooling or percentage calculation.
            by_expert=[[sum(h[s][n] for h in row['hist']) for n in range(5)] for s in range(3)]
            by_step=[[sum(h[s][n] for h in row['steps'].values()) for n in range(5)] for s in range(3)]
            if by_expert!=by_step:
                raise ValueError('Step and expert histogram totals disagree')
            for i, hist in enumerate(row['hist']):
                key = i + meta['expert_start']
                target = experts.setdefault(key, [[0]*5 for _ in range(3)])
                for s in range(3):
                    target[s] = [a+b for a,b in zip(target[s], hist[s])]
            for step, hist in row['steps'].items():
                target = steps.setdefault(step, [[0]*5 for _ in range(3)])
                for s in range(3):
                    target[s] = [a+b for a,b in zip(target[s], hist[s])]
        expected = records[0]['component_forward_calls'].get(name.split('.')[0],0)
        if set(steps) != {str(i) for i in range(expected)}:
            raise ValueError(f'Incomplete step coverage: {name}')
        for hist in experts.values():
            totals = [sum(h) for h in hist]
            if len(set(totals)) != 1:
                raise ValueError('Stage histogram denominators differ')
        merged[name] = dict(metadata=metadata, experts=experts, steps=steps)
    return dict(component_forward_calls=records[0]['component_forward_calls'], layers=merged)


def summarize(merged):
    pooled = [[0]*5 for _ in range(3)]
    weighted = [0.] * 3
    mac_weighted = [0.] * 3
    params = macs = inactive = 0
    for row in merged['layers'].values():
        meta = row['metadata']
        for hist in row['experts'].values():
            groups = sum(hist[0])
            if not groups:
                inactive += meta['parameters_per_expert']
                continue
            p = meta['parameters_per_expert']
            mac = p * groups * 4 // meta['width']
            params += p
            macs += mac
            for s in range(3):
                pooled[s] = [a+b for a,b in zip(pooled[s],hist[s])]
                ratio = sum(hist[s][:3])/groups
                weighted[s] += p*ratio
                mac_weighted[s] += mac*ratio
    stages={s:dict(**describe(pooled[i]),
                    parameter_weighted_eligible_2of4=weighted[i]/params if params else None,
                    mac_weighted_eligible_2of4=mac_weighted[i]/macs if macs else None)
                    for i,s in enumerate(STAGES)}
    stages['effective_storage']=dict(stages['effective_native'])
    return dict(stages=stages, observed_parameters=params,
                inactive_parameters=inactive, linear_macs=macs,
                storage_order='native; effective_storage equals effective_native',
                definition='Along GEMM K: H[n] counts adjacent groups of 4 with n nonzeros; eligible=(H0+H1+H2)/sum(H)')

import copy
import json
from pathlib import Path
import tempfile
import unittest
import hashlib
from types import SimpleNamespace
from collections import Counter

import torch

from benchmarks.deepseek_greedy_strategy import (SCHEMA, GROUPING, effective_sparsity,
                                                load_strategy, projection_group)
from model.native_greedy import search, native_activation_error
from model.deepseek_greedy_runtime import empirical_threshold, reset_layer_state
from model.deepseek_experiment import evenly_spaced_indices, InputCalibration
from model.deepseek_budget_inference import DeepSeekBudgetInferenceMixin
from benchmarks.run_deepseek_greedy_eval import compile_strategy_entry
from benchmarks.deepseek_greedy_search import export_strategy, reference_batches


def mock_generation_entry():
    metadata={'allocation':'uniform'}
    identity=hashlib.sha256(json.dumps(metadata,sort_keys=True).encode()).hexdigest()
    return metadata,identity


class GreedySearchTests(unittest.TestCase):
    def test_reference_batches_match_partial_search_batch(self):
        self.assertEqual(list(reference_batches(17,7,4)),[(0,4),(4,7),(7,11),(11,15),(15,17)])

    def test_export_ignores_online_diagnostics(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);(root/'wina').mkdir()
            (root/'wina/layer-0.json').write_text(json.dumps(dict(layer=0,profiles=[dict(target=.5,rates={},weights={})])))
            (root/'wina/layer-0.online.json').write_text(json.dumps(dict(mask='topk',checks=[])))
            export_strategy(root,'wina',{},[.5],2)
            result=json.loads((root/'wina/strategy.json').read_text())
            self.assertFalse(result['complete'])
            self.assertEqual(set(result['profiles'][0]['layers']),{'0'})

    def test_calibration_sampling_with_bfloat16_default(self):
        previous = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.bfloat16)
            indices = evenly_spaced_indices(128 * 4096, 2048)
            self.assertEqual(indices.dtype, torch.int64)
            self.assertEqual(indices[0].item(), 0)
            self.assertEqual(indices[-1].item(), 128 * 4096 - 1)
            self.assertTrue((indices.diff() > 0).all())
            self.assertEqual(evenly_spaced_indices(1, 1).item(), 0)
            observer = InputCalibration(256, sample_limit=4096)
            for _ in range(7):
                observer.update(torch.randn(32, 256))
            result = observer.finish()
            self.assertEqual(result['histogram_elements'], 4096)
            self.assertTrue(torch.isfinite(result['centers']).all())
        finally:
            torch.set_default_dtype(previous)

    def test_strategy_hash_enters_protocol_before_identity(self):
        entry=SimpleNamespace(main=mock_generation_entry,json=json,hashlib=hashlib)
        a=compile_strategy_entry(entry,{'strategy_sha256':'a'},None,None)()
        b=compile_strategy_entry(entry,{'strategy_sha256':'b'},None,None)()
        self.assertEqual(a[0]['allocation'],'greedy')
        self.assertNotEqual(a[1],b[1])

    def test_generation_uses_distinct_budgets_and_restores_global_target(self):
        class Base:
            def _sparse_mask(self,x,weight,key):return self.sparse_sparsity
            def stats(self):return {}
        class Runtime(DeepSeekBudgetInferenceMixin,Base):pass
        r=Runtime();r.sparse_modules={1:'layers.0.attn.wkv.weight',2:'layers.0.ffn.experts.8.w1.weight'}
        r.sparse_sparsity=.5
        rates={'layers.0.attn.wkv':.2,'layers.0.ffn.experts.w1':.8}
        r.configure_budget(rates,{'strategy_sha256':'x'})
        self.assertEqual(r._sparse_mask(None,None,'layers.0.attn.wkv.weight'),.2)
        self.assertEqual(r._sparse_mask(None,None,'layers.0.ffn.experts.8.w1.weight'),.8)
        self.assertEqual(r.sparse_sparsity,.5)
        with self.assertRaises(ValueError):r.configure_budget({'layers.0.attn.wkv':.2},{})

    def test_search_gives_sensitive_projection_less_budget(self):
        weights = {'sensitive': 1., 'tolerant': 1.}
        loss = lambda s: 10 * s['sensitive'] ** 2 + s['tolerant'] ** 2
        result = search(weights, loss, targets=[.5, .6], step=.1)
        self.assertTrue(result['complete'])
        for profile in result['profiles']:
            self.assertLess(profile['rates']['sensitive'], profile['rates']['tolerant'])
            self.assertLess(profile['error'], profile['uniform_error'])
            self.assertAlmostEqual(effective_sparsity(profile['rates'], weights), profile['target'])

    def test_resume_matches_uninterrupted_search(self):
        kwargs = dict(weights={'a': 1., 'b': 3.}, evaluate=lambda s: 4*s['a']**2+s['b']**2,
                      targets=[.5,.6,.7,.8,.9], step=.07, reference_weight=1.)
        partial = search(**kwargs, max_steps=2)
        self.assertFalse(partial['complete'])
        restarted = search(**kwargs, resume=copy.deepcopy(partial))
        full = search(**kwargs)
        self.assertEqual(restarted['profiles'], full['profiles'])
        self.assertEqual(restarted['iterations'], full['iterations'])

    def test_nonfinite_error_stops_search(self):
        with self.assertRaisesRegex(ValueError, 'Non-finite'):
            search({'a':1.},lambda _:float('nan'),targets=[.5])

    def test_zero_observation_group_stays_dense(self):
        def loss(rates):
            self.assertEqual(rates['unseen'], 0.)
            return rates['seen']
        result=search({'seen':1.,'unseen':0.},loss,targets=[.8],step=.1)
        self.assertEqual(result['rates']['unseen'],0.)
        with self.assertRaises(ValueError):
            effective_sparsity({'seen':.8,'unseen':.2},{'seen':1.,'unseen':0.})

    def test_candidates_spend_equal_weighted_budget(self):
        result=search({'small':1.,'large':4.},lambda s:sum(s.values()),
                      targets=[.5],step=.025,reference_weight=1.,max_steps=1)
        candidates=result['curve'][0]['candidates']
        self.assertEqual(len(candidates),2)
        self.assertAlmostEqual(candidates[0]['effective_sparsity'],candidates[1]['effective_sparsity'])

    def test_row_parallel_wina_selects_global_topk(self):
        class MockDist:
            @staticmethod
            def all_gather(outputs, local):
                outputs[0].copy_(local)
                outputs[1].copy_(local+100)
        class Runtime(DeepSeekBudgetInferenceMixin):
            def _active_rows(self,length,device):
                return torch.ones(length,dtype=torch.bool,device=device)
        runtime=Runtime()
        runtime.ds=SimpleNamespace(world_size=2,rank=0,dist=MockDist)
        runtime.sparse_row_chunk=1
        runtime.sparse_counts=Counter();runtime.sparse_eligible=Counter()
        result=runtime._row_parallel_wina(torch.tensor([[[1.,2.],[3.,4.]]]),'layers.0.attn.wo_b.weight',.5)
        self.assertEqual(result.count_nonzero(),0)
        self.assertEqual(runtime.sparse_counts['layers.0.attn.wo_b.weight'],4)

    def test_native_error_uses_sequence_axis_and_tail(self):
        target=torch.zeros(1,4,2)
        actual=torch.tensor([[[100.,100.],[100.,100.],[3.,0.],[4.,0.]]])
        self.assertAlmostEqual(native_activation_error(target,actual,.5).item(),2.5)
        hc=actual.unsqueeze(2).repeat(1,1,4,1)
        self.assertAlmostEqual(native_activation_error(torch.zeros_like(hc),hc,.5).item(),2.5)

    def test_native_empirical_cdf_interpolation(self):
        centers=torch.tensor([0.,1.,3.]);counts=torch.tensor([1.,2.,1.])
        self.assertAlmostEqual(empirical_threshold(centers,counts,.5),.5)
        self.assertAlmostEqual(empirical_threshold(centers,counts,.875),2.)

    def test_moe_family_grouping_preserves_layer_and_projection(self):
        self.assertEqual(projection_group('layers.3.ffn.experts.127.w2.weight'),'layers.3.ffn.experts.w2')
        self.assertEqual(projection_group('layers.3.attn.wo_a.weight#group7'),'layers.3.attn.wo_a')
        self.assertNotEqual(projection_group('layers.2.ffn.shared_experts.w1.weight'),
                            projection_group('layers.2.ffn.experts.1.w1.weight'))
        with self.assertRaises(ValueError): projection_group('lm_head.weight')

    def test_replay_clears_kv_and_compressor_state(self):
        layer=torch.nn.Module();layer.child=torch.nn.Module()
        layer.child.register_buffer('kv_state',torch.ones(2,3))
        layer.child.register_buffer('score_state',torch.ones(2,3))
        reset_layer_state(layer)
        self.assertEqual(layer.child.kv_state.count_nonzero(),0)
        self.assertTrue(torch.isneginf(layer.child.score_state).all())

    def test_strategy_rejects_incomplete_wrong_budget_or_model(self):
        weights={'layers.0.attn.wkv':1.}
        data=dict(schema=SCHEMA,method='wina',world_size=2,quantization='arc',
            source_signature={'config':'x'},grouping=GROUPING,n_layers=1,transform=False,
            complete=True,profiles=[dict(target=.5,layers={'0':dict(rates=dict.fromkeys(weights,.5),weights=weights)})])
        kwargs=dict(method='wina',sparsity=.5,source_signature={'config':'x'},world_size=2,quantization='arc',n_layers=1)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'strategy.json';path.write_text(json.dumps(data))
            self.assertEqual(load_strategy(path,**kwargs)[1],dict.fromkeys(weights,.5))
            with self.assertRaises(ValueError):load_strategy(path,**{**kwargs,'sparsity':.9})
            with self.assertRaises(ValueError):load_strategy(path,**{**kwargs,'quantization':'hif4'})
            data['complete']=False;path.write_text(json.dumps(data))
            with self.assertRaises(ValueError):load_strategy(path,**kwargs)


if __name__ == '__main__': unittest.main()

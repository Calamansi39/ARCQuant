import os
import sys
from pathlib import Path
import unittest

import torch

from integrations.magi_arc.arithmetic import activation,weight,expanded_weight,Observer,validate
from model.quantize import fake_reorder_quantize_x,fake_reorder_quantize_w


class ArithmeticTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(27)
        self.spec=dict(order=torch.stack([torch.randperm(128) for _ in range(3)]),
                       select=torch.tensor([0,64,128]))

    def test_expert_mixed_residual_widths_match_original_arc_oracle(self):
        # More than 256 rows exercises row chunking; noncontiguous storage
        # exercises MAGI's [E,K,N] expert weights.
        x=torch.randn(3,259,128).bfloat16()
        w=torch.randn(3,128,9).transpose(1,2).bfloat16()
        main=w.clone()
        residual=weight(main,self.spec)
        qx=activation(x,self.spec)
        qw=expanded_weight(main,residual,self.spec)
        for i in range(3):
            n=int(self.spec['select'][i]);order=self.spec['order'][i]
            rx=fake_reorder_quantize_x(x[i],order,n,dtype='HiF4')[0]
            rw=fake_reorder_quantize_w(w[i],order,n,dtype='HiF4')[0]
            self.assertTrue(torch.equal(qx[i,:,:128+n],rx))
            self.assertTrue(torch.equal(qw[i,:,:128+n],rw))
            torch.testing.assert_close(qx[i].float()@qw[i].float().T,rx.float()@rw.float().T,rtol=1e-6,atol=2e-5)
        # An expert sub-batch must use its own largest selected count.
        self.assertEqual(activation(x[:1],self.spec).shape[-1],128)
        self.assertEqual(expanded_weight(main[:1],residual,self.spec).shape[-1],128)

    def test_stats_exclude_padded_rows_and_keep_signed_arc_rule(self):
        obs=Observer(2,128)
        x=torch.full((2,3,128),-1.)
        x[0,0,5]=8;x[0,1,4]=4
        x[0,2]=10000  # padding must not set maxima
        obs.update(x,valid_rows=torch.tensor([2,3]))
        spec=obs.finish()
        self.assertEqual(spec['order'][0,-1].item(),5)
        self.assertEqual(spec['select'].tolist(),[64,0])
        self.assertEqual(spec['rows'].tolist(),[2,3])

    def test_missing_expert_and_invalid_permutation_rejected(self):
        obs=Observer(2,128)
        obs.update(torch.ones(1,2,128))
        with self.assertRaisesRegex(ValueError,'unobserved'): obs.finish()
        self.spec['order'][0,1]=self.spec['order'][0,0]
        with self.assertRaises(ValueError): validate(self.spec,3,128)

    def test_arc_does_not_restore_masked_channels(self):
        x=torch.randn(3,5,128).bfloat16();x[...,::2]=0
        out=activation(x,self.spec)
        for i,n in enumerate(self.spec['select'].tolist()):
            order=self.spec['order'][i]
            self.assertTrue((out[i,:,:128][:,order%2==0]==0).all())
            if n:self.assertTrue((out[i,:,128:128+n][:,order[-n:]%2==0]==0).all())


@unittest.skipUnless(os.environ.get('ARC_TEST_BASE_EVAL'),'requires native MAGI runtime assets')
class AdapterTests(unittest.TestCase):
    def setUp(self):
        base=os.environ['ARC_TEST_BASE_EVAL'];sys.path.insert(0,base);os.environ['ARC_BASE_EVAL']=base

    def test_grouped_dispatch_and_refiner_rounding_use_arc(self):
        from integrations.magi_arc.runtime import Context,install
        for classname in ('CustomGroupedLinear','BaseLinear'):
            cls=type(classname,(torch.nn.Module,),{})
            module=cls();module.num_experts=2;module.out_features=7;module.in_features=128
            module.weight=torch.nn.Parameter(torch.randn(14,128).bfloat16())
            module.bias=torch.nn.Parameter(torch.randn(14).bfloat16())
            model=torch.nn.Module();model.add_module('projection',module)
            model.final_linear=torch.nn.Linear(128,7).bfloat16()
            final_weight=model.final_linear.weight.detach().clone()
            original=module.weight.detach().view(2,7,128).clone()
            spec=dict(order=torch.stack([torch.randperm(128),torch.randperm(128)]),select=torch.tensor([64,128]),basis_sha256=None)
            ctx=Context('teal','hifp4',{},Path('/unused'),device='cpu',backend='reference',specs={'preview.projection':spec})
            install(model,'preview',ctx)
            x=torch.randn(5,128).bfloat16()
            class Dispatch:
                group_size_cpu=[2,3]
                def dispatch(self,x): return x.split([2,3])
                def undispatch(self,*ys): return torch.cat(ys)
            actual=module(x,modality_dispatcher=Dispatch())
            from magi2_quant import quantize_tensor
            expected_final=torch.nn.functional.linear(quantize_tensor(x,'hifp4'),quantize_tensor(final_weight,'hifp4'),model.final_linear.bias)
            self.assertTrue(torch.equal(model.final_linear(x),expected_final))
            result=[]
            for i,part in enumerate(x.split([2,3])):
                qx=fake_reorder_quantize_x(part,spec['order'][i],int(spec['select'][i]),dtype='HiF4')[0]
                qw=fake_reorder_quantize_w(original[i],spec['order'][i],int(spec['select'][i]),dtype='HiF4')[0]
                bias=module.bias.view(2,7)[i]
                result.append(torch.nn.functional.linear(qx,qw,bias) if classname=='CustomGroupedLinear' else (qx@qw.T)+bias)
            self.assertTrue(torch.equal(actual,torch.cat(result)))
            self.assertGreater(ctx.arc_report()['projections']['preview.projection']['calls'],0)


if __name__=='__main__': unittest.main()

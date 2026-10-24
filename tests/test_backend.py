from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
import torch
from fixtures import config, tiny_backend, row
from cmselect.backend import parameter_owner, ppo_terms
from cmselect.masking import action_value_mask

class BackendTests(unittest.TestCase):
    def test_real_Qwen_tokenizer_action_mask_handles_escaped_rationale(self):
        from cmselect.core import ROOT
        if not (ROOT/'assets/tokenizer/tokenizer.json').exists():
            self.skipTest('Tokenizer assets not included in lightweight remote CPU copy')
        from transformers import AutoTokenizer
        tok=AutoTokenizer.from_pretrained(ROOT/'assets/tokenizer',local_files_only=True)
        text='{"thought":"이 내용은 보존. \\\"action\\\":\\\"SKIP\\\" is quoted.","action":"REPLACE","target_id":"m_t000001","source_turn_ids":["t000002"]}'
        ids=tok.encode(text,add_special_tokens=False)
        r={'text':text,'response_ids':ids}
        mask=action_value_mask(tok,r)
        selected=tok.decode([i for i,m in zip(ids,mask) if m])
        self.assertIn('REPLACE',selected)
        self.assertNotIn('thought',selected)
        self.assertNotIn('target',selected)

    def test_sample_logprobs_are_actual_policy_and_no_cuda(self):
        b=tiny_backend(config())
        sample=b.sample('selector',[{'role':'user','content':'fixture'}],23,6)
        b.activate('selector')
        with torch.no_grad(): lp=b.log_probs(sample)
        torch.testing.assert_close(lp,torch.tensor(sample['old_log_probs']),atol=1e-5,rtol=1e-5)
        self.assertEqual(sample,b.sample('selector',[{'role':'user','content':'fixture'}],23,6))
        self.assertFalse(torch.cuda.is_initialized())

    def test_adapter_isolation_A_B_and_frozen_references(self):
        for variant in ('A','B'):
            b=tiny_backend(config(variant))
            before={n:p.detach().clone() for n,p in b.model.named_parameters()}
            r=row(b,'selector','{"target_id":"x"}')
            stats=b.update('selector',[[r]])
            self.assertEqual(stats['optimizer_steps'],1)
            changed={parameter_owner(n) for n,p in b.model.named_parameters() if not torch.equal(before[n],p)}
            self.assertEqual(changed,{'selector'})
            if variant=='A':
                with self.assertRaises(ValueError):b.update('manager',[[r]])
            else:
                before={n:p.detach().clone() for n,p in b.model.named_parameters()}
                r=row(b,'manager','{"thought":"drop x","action":"REPLACE","target_id":"x","source_turn_ids":["t000002"]}')
                r['loss_mask']=action_value_mask(b.tokenizer,r)
                b.update('manager',[[r]])
                changed={parameter_owner(n) for n,p in b.model.named_parameters() if not torch.equal(before[n],p)}
                self.assertEqual(changed,{'manager'})

    def test_zero_signal_and_stale_sample_guard(self):
        b=tiny_backend(config())
        r=row(b,'selector','abc',0.)
        self.assertEqual(b.update('selector',[[r]])['optimizer_steps'],0)
        r['advantage']=1.;r['old_log_probs']=[v+5 for v in r['old_log_probs']]
        before=[p.detach().clone() for p in b.parameters['selector']]
        with self.assertRaises(RuntimeError):b.update('selector',[[r]])
        for p,q in zip(before,b.parameters['selector']):self.assertTrue(torch.equal(p,q))

    def test_loss_mask_has_zero_direct_gradient(self):
        lp=torch.tensor([-1.,-1.,-1.],requires_grad=True)
        terms,_,_=ppo_terms(lp,lp.detach(),lp.detach(),2.,.2,.02)
        (terms*torch.tensor([0.,1.,0.])).sum().backward()
        torch.testing.assert_close(lp.grad,torch.tensor([0.,-2.,0.]))

    def test_save_restore_optimizer_and_adapter_layout(self):
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file
        b=tiny_backend(config());b.update('selector',[[row(b,'selector','abc')]])
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);b.save(root)
            other=tiny_backend(config())
            for a in ('manager','selector'):
                set_peft_model_state_dict(other.model,load_file(str(root/'adapters'/a/'adapter_model.safetensors')),adapter_name=a)
            other.restore_training(root)
            for (n,p),(n2,p2) in zip(b.model.named_parameters(),other.model.named_parameters()):
                self.assertEqual(n,n2);torch.testing.assert_close(p,p2,rtol=0,atol=0)
            r1=row(b,'selector','def');r2=row(other,'selector','def')
            b.update('selector',[[r1]]);other.update('selector',[[r2]])
            for p,p2 in zip(b.parameters['selector'],other.parameters['selector']):torch.testing.assert_close(p,p2,rtol=0,atol=0)

if __name__ == '__main__':unittest.main()

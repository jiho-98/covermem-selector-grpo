"""End-to-end tiny Qwen coordinator test with scripted actions and real logprobs.

Scripted responses are a plumbing fixture, not on-policy scientific rollouts.
Real sampled-token logprob agreement is tested independently in test_backend.
"""
from pathlib import Path
import tempfile
import unittest
import torch
from fixtures import config, data, FakeBackend, tiny_backend
from cmselect.backend import parameter_owner
from cmselect.train import train_loop
from cmselect import checkpoint

def scripted_real_backend(cfg):
    b=tiny_backend(cfg)
    scripted=FakeBackend()
    def sample(adapter,messages,seed,max_new_tokens):
        result=scripted.sample(adapter,messages,seed,max_new_tokens)
        b.activate(adapter)
        with torch.no_grad():result['old_log_probs']=b.log_probs(result).tolist()
        return result
    b.sample=sample
    return b

class JointResumeTests(unittest.TestCase):
    def test_B_both_optimizers_and_banks_resume_identically(self):
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file
        cfg=config('B');sessions,pools=data()
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            full=scripted_real_backend(cfg)
            before={n:p.detach().clone() for n,p in full.model.named_parameters()}
            end=train_loop(cfg,{'fingerprint':'tiny'},sessions,pools,full,root/'full')
            self.assertGreater(end['updates']['manager'],0)
            self.assertGreater(end['updates']['selector'],0)
            changed={parameter_owner(n) for n,p in full.model.named_parameters() if not torch.equal(before[n],p)}
            self.assertEqual(changed,{'manager','selector'})
            partial=scripted_real_backend(cfg)
            train_loop(cfg,{'fingerprint':'tiny'},sessions,pools,partial,root/'resume',stop_after_batches=2)
            path,state=checkpoint.latest(root/'resume','tiny')
            resumed=scripted_real_backend(cfg)
            for adapter in ('manager','selector'):
                set_peft_model_state_dict(resumed.model,load_file(str(path/'adapters'/adapter/'adapter_model.safetensors')),adapter_name=adapter)
            resumed.restore_training(path)
            actual=train_loop(cfg,{'fingerprint':'tiny'},sessions,pools,resumed,root/'resume',state)
            self.assertEqual(actual['banks'],end['banks'])
            self.assertEqual(actual['updates'],end['updates'])
            for p,q in zip(full.model.parameters(),resumed.model.parameters()):
                torch.testing.assert_close(p,q,rtol=0,atol=0)

"""Record actual training-forward probabilities and a read-only post-step pass."""
import time
import torch
from protocol import atomic,digest
from performance_actor import BatchedCoverageActor as OriginalActor
from actor import pack_action

class TracedActor(OriginalActor):
    def _forward_micro_batch(self,batch,temperature=1.,calculate_entropy=False):
        out=super()._forward_micro_batch(batch,temperature,calculate_entropy)
        if getattr(self,'capture',False) and torch.is_grad_enabled():
            # Original validated two-GPU profile uses unpadded one-action batches.
            assert len(batch['input_ids'])==1
            key=digest(batch['input_ids'][0].detach().cpu().tolist())
            self.pre[key]=out['log_probs'][0].detach().float().cpu().tolist()
        return out

    def update_session(self,rows,global_tokens):
        assert self.performance.get('mode')=='baseline', 'Keep the validated original execution profile'
        self.pre={};self.capture=True
        stats=super().update_session(rows,global_tokens)
        self.capture=False
        started=time.monotonic();records=[]
        # No dropout, no gradients, no optimizer mutation, restore RNG anyway.
        rng=torch.get_rng_state();cuda_rng=torch.cuda.get_rng_state(self.device)
        self._select_adapter('default');self.actor_module.eval()
        with torch.no_grad():
            for r in rows:
                key=digest(r['prompt_ids']+r['response_ids'])
                before=self.pre[key]
                after=self._forward_micro_batch(pack_action(r,self.device).batch)['log_probs'][0].float().cpu().tolist()
                assert len(before)==len(after)==len(r['response_ids'])
                assert all(torch.isfinite(torch.tensor(after)))
                records.append(dict(request_key=r.get('request_key',key),stream=r.get('stream'),
                    turn_index=r.get('turn_index'),attempt=r.get('attempt'),advantage=r['advantage'],
                    original_advantage=r.get('original_advantage'),
                    pre_log_probs=before,post_log_probs=after,
                    mean_logprob_delta=(sum(after)-sum(before))/len(before),
                    sequence_logprob_delta=sum(after)-sum(before)))
        torch.set_rng_state(rng);torch.cuda.set_rng_state(cuda_rng,self.device)
        self.actor_module.train()
        if getattr(self,'trace_path',None):atomic(self.trace_path,dict(status='complete',outputs=records))
        stats['probability_audit_seconds']=time.monotonic()-started
        stats['probability_audit_outputs']=len(records)
        self.pre={}
        return stats

"""Execution-only batching; unchanged native-verl PPO/KL and one optimizer step.

Right padding is deliberately OUTSIDE every observed causal sequence. There is
no left padding and no mask that materializes an L x L attention tensor. Real
response positions are explicitly gathered; padding never enters the loss.
"""
from contextlib import nullcontext
import gc
import time
import torch
import torch.distributed as dist
from verl import DataProto
from verl.trainer.ppo.core_algos import get_policy_loss_fn, kl_penalty, agg_loss
from actor import CoverageActor, world_size

def row_length(row):return len(row['prompt_ids'])+len(row['response_ids'])

def response_positions(row):
    start=len(row['prompt_ids'])-1
    return range(start,start+len(row['response_ids']))

def batches_for(rows, limits):
    # Sorting changes accumulation order, not samples, tokens or weights.
    ordered=sorted(rows,key=lambda r:(row_length(r),len(r['response_ids'])),reverse=True)
    result=[];group=[]
    def fits(batch):
        longest=max(map(row_length,batch));real=sum(map(row_length,batch));size=len(batch)
        positions=set().union(*(set(response_positions(r)) for r in batch))
        return (size<=limits['max_actions'] and size*longest<=limits['max_padded_tokens']
                and size*longest/real<=limits['max_padding_ratio']
                and size*len(positions)<=limits['max_logit_vectors'])
    for row in ordered:
        if group and not fits(group+[row]):result.append(group);group=[]
        group.append(row)
    if group:result.append(group)
    assert sum(map(len,result))==len(rows)
    return result

def pack_batch(rows,device):
    if not rows:raise ValueError('Empty microbatch')
    total=max(map(row_length,rows));response=max(len(r['response_ids']) for r in rows)
    union=sorted(set().union(*(set(response_positions(r)) for r in rows)))
    lookup={p:i for i,p in enumerate(union)}
    values={k:[] for k in ('input_ids','responses','response_mask','old_log_probs','advantages','response_columns')}
    for row in rows:
        p=row['prompt_ids'];r=row['response_ids'];n=len(r)
        if not p or not r or len(row['old_log_probs'])!=n:raise ValueError('Invalid action sample')
        values['input_ids'].append(p+r+[0]*(total-len(p)-n))
        values['responses'].append(r+[0]*(response-n))
        values['response_mask'].append([1.]*n+[0.]*(response-n))
        values['old_log_probs'].append(row['old_log_probs']+[0.]*(response-n))
        values['advantages'].append([row['advantage']]*n+[0.]*(response-n))
        values['response_columns'].append([lookup[x] for x in response_positions(row)]+[0]*(response-n))
    tensors={k:torch.tensor(v,dtype=torch.long if k in ('input_ids','responses','response_columns')
              else torch.float32,device=device) for k,v in values.items()}
    tensors['position_ids']=torch.arange(total,device=device).expand(len(rows),-1)
    # Same union positions for every batch row. Real positions are selected below.
    tensors['logit_positions']=torch.tensor([union]*len(rows),dtype=torch.long,device=device)
    return DataProto.from_dict(tensors=tensors)

class BatchedCoverageActor(CoverageActor):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.performance={'mode':'baseline'}

    def _forward_micro_batch(self,batch,temperature=1.,calculate_entropy=False):
        if 'response_columns' not in batch:
            return super()._forward_micro_batch(batch,temperature,calculate_entropy)
        if calculate_entropy:raise ValueError('No new entropy objective')
        dtype=getattr(torch,self.coverage_config['training']['dtype'])
        context=torch.autocast('cuda',dtype=dtype) if self.device.type=='cuda' else nullcontext()
        with context:
            out=self.actor_module(input_ids=batch['input_ids'],attention_mask=None,
                position_ids=batch['position_ids'],use_cache=False,
                logits_to_keep=batch['logit_positions'][0])
            b=torch.arange(len(batch['input_ids']),device=self.device).unsqueeze(1)
            logits=out.logits[b,batch['response_columns'],:].float()/temperature
            selected=logits.gather(-1,batch['responses'].unsqueeze(-1)).squeeze(-1)
            lp=selected-torch.logsumexp(logits,dim=-1)
            lp=lp.masked_fill(~batch['response_mask'].bool(),0.)
        return {'log_probs':lp}

    def _accumulate(self,rows,global_tokens,limits,loss_scale=1.,reset_grad=True):
        if reset_grad:self.actor_optimizer.zero_grad(set_to_none=True)
        groups=batches_for(rows,limits)
        # Tokens are small relative to model activations. Transfer once per group,
        # then reuse both for reference and for the trainable-policy forward.
        packed=[pack_batch(group,self.device).batch for group in groups]
        references=[]
        self._select_adapter('reference');self.actor_module.eval()
        ref_started=time.monotonic()
        try:
            with torch.no_grad():
                for batch in packed:references.append(self._forward_micro_batch(batch)['log_probs'].detach())
        finally:self._select_adapter('default')
        if self.device.type=='cuda':torch.cuda.synchronize(self.device)
        reference_seconds=time.monotonic()-ref_started
        self.actor_module.train()
        totals=torch.zeros(6,dtype=torch.float64,device=self.device);maximum=torch.zeros((),device=self.device)
        segments=torch.zeros((2,3),dtype=torch.float64,device=self.device)
        loss_fn=get_policy_loss_fn('vanilla')
        for group,batch,ref in zip(groups,packed,references):
            lp=self._forward_micro_batch(batch,self.coverage_config['rollout']['temperature'])['log_probs']
            mask=batch['response_mask'];gap=(lp.detach()-batch['old_log_probs']).abs()*mask
            if not torch.isfinite(lp).all():raise FloatingPointError('Nonfinite policy logprob')
            pg,metrics=loss_fn(old_log_prob=batch['old_log_probs'],log_prob=lp,
                advantages=batch['advantages'],response_mask=mask,loss_agg_mode='token-mean',config=self.config)
            kl_matrix=kl_penalty(lp,ref,self.config.kl_loss_type)
            kl=agg_loss(kl_matrix,mask,'token-mean',batch_num_tokens=global_tokens)
            loss=(pg+self.config.kl_loss_coef*kl)*loss_scale
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
            n=sum(len(r['response_ids']) for r in group)
            totals[0]+=pg.detach().double();totals[1]+=kl.detach().double()
            totals[2]+=metrics['actor/pg_clipfrac']*n;totals[3]+=gap.sum().double()
            totals[4]+=n;totals[5]+=len(group);maximum=torch.maximum(maximum,gap.max())
            # Diagnostics use the same native loss; never contribute extra gradients.
            with torch.no_grad():
                for index,label in enumerate(('front','back')):
                    indices=[i for i,r in enumerate(group) if (r.get('segment')=='front')==(label=='front')]
                    if not indices:continue
                    submask=mask[indices]
                    subpg,_=loss_fn(old_log_prob=batch['old_log_probs'][indices],log_prob=lp.detach()[indices],
                        advantages=batch['advantages'][indices],response_mask=submask,loss_agg_mode='token-mean',config=self.config)
                    subkl=agg_loss(kl_matrix.detach()[indices],submask,'token-mean',batch_num_tokens=global_tokens)
                    segments[index,0]+=subpg.double();segments[index,1]+=subkl.double();segments[index,2]+=submask.sum().double()
            loss.backward()
            del lp,mask,gap,pg,kl,kl_matrix,loss
        return totals,maximum,segments,{'microbatches':len(groups),'reference_seconds':reference_seconds,
            'batch_action_counts':[len(g) for g in groups],
            'padded_input_tokens':sum(len(g)*max(map(row_length,g)) for g in groups),
            'real_input_tokens':sum(map(row_length,rows))}

    def _accumulate_sessions_batched(self,session_batches,active_sessions,limits):
        self.actor_optimizer.zero_grad(set_to_none=True)
        totals=[];maximum=[];segments=[]
        extra={'microbatches':0,'reference_seconds':0.,'batch_action_counts':[],
               'padded_input_tokens':0,'real_input_tokens':0}
        for rows,global_tokens in session_batches:
            self.config.global_batch_info={'batch_num_tokens':global_tokens,'dp_size':1}
            values=self._accumulate(rows,global_tokens,limits,loss_scale=1/active_sessions,reset_grad=False)
            totals.append(values[0]);maximum.append(values[1]);segments.append(values[2])
            for key,value in values[3].items():extra[key]+=value
        return torch.stack(totals),torch.stack(maximum),torch.stack(segments),extra

    def update_batch(self,session_batches,active_sessions):
        if self.performance.get('mode')!='batched':return super().update_batch(session_batches,active_sessions)
        self._validate_batch(session_batches,active_sessions)
        before=self.optimizer_steps;limits=dict(self.performance);oom_retries=0
        # Every rank reaches this control collective before metric/gradient
        # collectives. A rank-local OOM makes ALL ranks retry the whole batch.
        oom_flag=torch.zeros((),device=self.device,dtype=torch.int32)
        while True:
            oom_flag.zero_()
            try:
                values=self._accumulate_sessions_batched(session_batches,active_sessions,limits)
            except torch.cuda.OutOfMemoryError:
                oom_flag.fill_(1)
                self.actor_optimizer.zero_grad(set_to_none=True)
                self._select_adapter('default');gc.collect();torch.cuda.empty_cache()
            if world_size()>1:dist.all_reduce(oom_flag,op=dist.ReduceOp.MAX)
            if not oom_flag.item():break
            self.actor_optimizer.zero_grad(set_to_none=True)
            self._select_adapter('default');gc.collect();torch.cuda.empty_cache()
            if limits['max_actions']<=1:
                raise torch.cuda.OutOfMemoryError('Whole session batch failed at one action per microbatch; no optimizer step committed')
            limits['max_actions']=max(1,limits['max_actions']//2)
            limits['max_padded_tokens']=max(32768,limits['max_padded_tokens']//2)
            self.performance.update(max_actions=limits['max_actions'],max_padded_tokens=limits['max_padded_tokens'])
            oom_retries+=1
        totals,maximum,segments,extra=values
        stats=self._finish_batch(session_batches,active_sessions,totals,maximum,segments,before)
        stats['performance']={**extra,'max_actions':limits['max_actions'],'oom_batch_reductions':oom_retries}
        return stats

    def update_session(self,rows,global_tokens):
        return self.update_batch([(rows,global_tokens)],1)

def balance_rows(rows,world):
    if world<1:raise ValueError('Invalid GPU count')
    bins=[[] for _ in range(world)];costs=[0]*world
    # Linear projections + causal attention growth; scheduling only, no loss weights.
    def cost(r):
        n=row_length(r)
        return n*(n+8192)
    for row in sorted(rows,key=cost,reverse=True):
        rank=min(range(world),key=lambda i:costs[i]);bins[rank].append(row);costs[rank]+=cost(row)
    assert sum(map(len,bins))==len(rows)
    return bins

"""verl actor extension with one update for a mean of session token means.

Uses upstream DataProto, PPO loss, KL and DataParallelPPOActor._optimizer_step.
The accumulation loop is specialized because a session consists of many separate
observations (not an ever-growing chat) and each response has a different length.
Each active session uses its GLOBAL response-token denominator. Its PG and KL
are both weighted by 1 / active_sessions, then gradients are SUM-reduced once.
"""
from contextlib import nullcontext
import math
import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from verl import DataProto
from verl.workers.actor.dp_actor import DataParallelPPOActor
from verl.trainer.ppo.core_algos import get_policy_loss_fn, kl_penalty, agg_loss

def world_size(): return dist.get_world_size() if dist.is_initialized() else 1

def actor_config(cfg):
    t=cfg['training']
    return OmegaConf.create({'strategy':'fsdp','use_remove_padding':False,'use_fused_kernels':False,
        'ulysses_sequence_parallel_size':1,'use_dynamic_bsz':False,'use_prefix_grouper':False,
        'entropy_from_logits_with_chunking':False,'use_torch_compile':False,
        'calculate_sum_pi_squared':False,'fsdp_config':{'dtype':t['dtype']},
        'grad_clip':t['grad_clip'],'clip_ratio':t['clip_ratio'],'clip_ratio_low':t['clip_ratio'],
        'clip_ratio_high':t['clip_ratio'],'clip_ratio_c':t['clip_ratio_c'],
        'policy_loss':{'loss_mode':'vanilla'},'global_batch_info':{},
        'entropy_coeff':0.0,'calculate_entropy':False,'entropy_checkpointing':False,
        'use_kl_loss':True,'kl_loss_coef':t['kl_coef'],'kl_loss_type':t['kl_type'],
        'ppo_epochs':1,'ppo_mini_batch_size':1,'ppo_micro_batch_size_per_gpu':1})

def pack_action(row,device):
    prompt=row['prompt_ids']; response=row['response_ids']
    if not prompt or not response or len(response)!=len(row['old_log_probs']):raise ValueError('Invalid action sample')
    ids=torch.tensor([prompt+response],dtype=torch.long,device=device)
    n=len(response)
    return DataProto.from_dict(tensors={'input_ids':ids,'attention_mask':torch.ones_like(ids),
        'position_ids':torch.arange(ids.shape[1],device=device).unsqueeze(0),
        'responses':torch.tensor([response],dtype=torch.long,device=device),
        'response_mask':torch.ones((1,n),device=device),
        'old_log_probs':torch.tensor([row['old_log_probs']],dtype=torch.float32,device=device),
        'advantages':torch.full((1,n),row['advantage'],dtype=torch.float32,device=device)})

class CoverageActor(DataParallelPPOActor):
    def __init__(self,cfg,model,optimizer,device):
        self.coverage_config=cfg;self.device=torch.device(device)
        super().__init__(actor_config(cfg),model,optimizer)
        self.optimizer_steps=0
        self._step_hook=optimizer.register_step_post_hook(self._count_step)

    def _count_step(self,*unused): self.optimizer_steps+=1

    def _select_adapter(self,name):
        if hasattr(self.actor_module,'set_adapter'):
            self.actor_module.set_adapter(name)
            # PEFT set_adapter may re-enable selected parameters; reference stays frozen.
            for n,p in self.actor_module.named_parameters():
                p.requires_grad_('.default.' in n and 'lora_' in n)

    def _forward_micro_batch(self,micro_batch,temperature=1.0,calculate_entropy=False):
        if calculate_entropy:raise ValueError('Entropy bonus is not part of this protocol')
        r=micro_batch['responses'].shape[1]
        dtype=getattr(torch,self.coverage_config['training']['dtype'])
        context=torch.autocast('cuda',dtype=dtype) if self.device.type=='cuda' else nullcontext()
        with context:
            out=self.actor_module(input_ids=micro_batch['input_ids'],
                attention_mask=micro_batch['attention_mask'],position_ids=micro_batch['position_ids'],
                use_cache=False,logits_to_keep=r+1)
            # Retain only response prediction positions: avoids a 20K x vocabulary logits tensor.
            logits=out.logits[:,-r-1:-1,:].float()/temperature
            selected=logits.gather(-1,micro_batch['responses'].unsqueeze(-1)).squeeze(-1)
            log_probs=selected-torch.logsumexp(logits,dim=-1)
        return {'log_probs':log_probs}

    def reference_log_prob(self,batch):
        if not hasattr(self.actor_module,'set_adapter'):
            raise RuntimeError('Production requires the frozen initial-SFT reference adapter')
        self._select_adapter('reference');self.actor_module.eval()
        try:
            with torch.no_grad():result=self._forward_micro_batch(batch)['log_probs'].detach()
        finally:self._select_adapter('default')
        return result

    def _validate_batch(self,session_batches,active_sessions):
        if type(active_sessions) is not int or active_sessions<1 or len(session_batches)!=active_sessions:
            raise ValueError('Pass every nonzero-signal session once; skip an all-zero batch in the coordinator')
        versions=set()
        for rows,global_tokens in session_batches:
            if type(global_tokens) is not int or global_tokens<=0:
                raise ValueError('Each session needs its positive global response-token denominator')
            if sum(len(r['response_ids']) for r in rows)>global_tokens:
                raise ValueError('Local session tokens exceed global session tokens')
            versions.update(r['policy_version'] for r in rows if 'policy_version' in r)
        if len(versions)>1:raise RuntimeError('All sessions in a batch must use one frozen policy version')

    def _accumulate_batch(self,session_batches,active_sessions):
        self._select_adapter('default');self.actor_module.train()
        self.actor_optimizer.zero_grad(set_to_none=True)
        # [pg, kl, token-weighted clipping, gap sum, gap count, actions]
        totals=torch.zeros((active_sessions,6),dtype=torch.float64,device=self.device)
        max_gaps=torch.zeros(active_sessions,device=self.device)
        segments=torch.zeros((active_sessions,2,3),dtype=torch.float64,device=self.device)
        loss_fn=get_policy_loss_fn('vanilla')
        for index,(rows,global_tokens) in enumerate(session_batches):
            self.config.global_batch_info={'batch_num_tokens':global_tokens,'dp_size':1}
            for row in rows:
                data=pack_action(row,self.device);b=data.batch
                ref=self.reference_log_prob(b)
                self.actor_module.train()
                lp=self._forward_micro_batch(b,temperature=self.coverage_config['rollout']['temperature'])['log_probs']
                gap=(lp.detach()-b['old_log_probs']).abs()
                if not torch.isfinite(lp).all():raise FloatingPointError('Nonfinite policy log-probability')
                pg,metrics=loss_fn(old_log_prob=b['old_log_probs'],log_prob=lp,
                    advantages=b['advantages'],response_mask=b['response_mask'],
                    loss_agg_mode='token-mean',config=self.config)
                kl=agg_loss(kl_penalty(lp,ref,self.config.kl_loss_type),b['response_mask'],
                            'token-mean',batch_num_tokens=global_tokens)
                loss=(pg+self.config.kl_loss_coef*kl)/active_sessions
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
                loss.backward()
                n=len(row['response_ids'])
                totals[index]+=torch.tensor([pg.detach().item(),kl.detach().item(),metrics['actor/pg_clipfrac']*n,
                                      gap.sum().item(),n,1],dtype=torch.float64,device=self.device)
                segments[index,0 if row.get('segment')=='front' else 1]+=torch.tensor(
                    [pg.detach().item(),kl.detach().item(),n],dtype=torch.float64,device=self.device)
                max_gaps[index]=torch.maximum(max_gaps[index],gap.max())
                del data,b,ref,lp,gap,loss,pg,kl
        return totals,max_gaps,segments

    def _finish_batch(self,session_batches,active_sessions,totals,max_gaps,segments,before):
        if world_size()>1:
            dist.all_reduce(totals,op=dist.ReduceOp.SUM);dist.all_reduce(max_gaps,op=dist.ReduceOp.MAX)
            dist.all_reduce(segments,op=dist.ReduceOp.SUM)
        guard=self.coverage_config['validation']
        for index,(_,global_tokens) in enumerate(session_batches):
            if int(totals[index,4].item())!=global_tokens:
                self.actor_optimizer.zero_grad(set_to_none=True)
                raise RuntimeError(f'Global token accounting mismatch in session {index}')
            avg=totals[index,3].item()/global_tokens
            if avg>guard['max_preupdate_mean_abs_logprob_gap'] or max_gaps[index].item()>guard['max_preupdate_logprob_gap']:
                self.actor_optimizer.zero_grad(set_to_none=True)
                raise RuntimeError(f'Stale/wrong rollout policy or context: session {index} logprob gap mean={avg}, max={max_gaps[index].item()}')
        # Manual synchronous data parallelism permits unequal numbers of independent
        # actions per rank. No collective inside forward/backward can deadlock on padding.
        for group in self.actor_optimizer.param_groups:
            for p in group['params']:
                if p.grad is None:p.grad=torch.zeros_like(p)
                if world_size()>1:dist.all_reduce(p.grad,op=dist.ReduceOp.SUM)
        parameters=[p for g in self.actor_optimizer.param_groups for p in g['params']]
        previous=[p.detach().clone() for p in parameters]
        grad_norm=super()._optimizer_step()
        if self.optimizer_steps!=before+1 or not math.isfinite(float(grad_norm)):
            raise FloatingPointError('Exactly one finite optimizer update was required')
        delta=sum((p.detach().float()-old.float()).square().sum() for p,old in zip(parameters,previous)).sqrt().item()
        norm=sum(old.float().square().sum() for old in previous).sqrt().item()
        # Release completed gradients before CPU offload without an extra zero_grad call.
        for p in parameters:p.grad=None
        combined=totals.sum(0);tokens=int(combined[4].item())
        combined_segments=segments.sum(0);combined_segments[:,:2]/=active_sessions
        return {'optimizer_steps':1,'active_sessions':active_sessions,
                'loss_aggregation':'mean_over_active_sessions_of_global_response_token_mean',
                'pg_loss':combined[0].item()/active_sessions,'kl':combined[1].item()/active_sessions,
                'clip_fraction':sum(totals[i,2].item()/t for i,(_,t) in enumerate(session_batches))/active_sessions,
                'response_tokens':tokens,'logprob_gap_mean':combined[3].item()/tokens,
                'logprob_gap_max':max_gaps.max().item(),
                'grad_norm':float(grad_norm),'attempts':int(combined[5].item()),
                'adapter_delta_l2':delta,'adapter_relative_delta_l2':delta/max(norm,1e-12),
                'per_session':[{'session_index':i,'pg_loss':v[0].item(),'kl':v[1].item(),
                    'loss_weight':1/active_sessions,'response_tokens':int(v[4].item()),'attempts':int(v[5].item()),
                    'logprob_gap_mean':v[3].item()/v[4].item(),'logprob_gap_max':max_gaps[i].item()}
                    for i,v in enumerate(totals)],
                'segments':{name:{'pg_loss_contribution':s[0].item(),'kl_contribution':s[1].item(),
                                   'response_tokens':int(s[2].item())} for name,s in zip(('front','back'),combined_segments)}}

    def update_batch(self,session_batches,active_sessions):
        self._validate_batch(session_batches,active_sessions)
        before=self.optimizer_steps
        totals,max_gaps,segments=self._accumulate_batch(session_batches,active_sessions)
        return self._finish_batch(session_batches,active_sessions,totals,max_gaps,segments,before)

    def update_session(self,rows,global_tokens):
        return self.update_batch([(rows,global_tokens)],1)

def disable_dropout(model):
    for module in model.modules():
        if isinstance(module,torch.nn.Dropout):module.p=0.0

def move_optimizer(optimizer,device):
    for state in optimizer.state.values():
        for k,v in state.items():
            if torch.is_tensor(v):state[k]=v.to(device)

def balance_rows(rows,world):
    bins=[[] for _ in range(world)];costs=[0]*world
    for row in sorted(rows,key=lambda r:len(r['prompt_ids'])+len(r['response_ids']),reverse=True):
        rank=min(range(world),key=lambda i:costs[i]);bins[rank].append(row)
        costs[rank]+=len(row['prompt_ids'])+len(row['response_ids'])
    return bins

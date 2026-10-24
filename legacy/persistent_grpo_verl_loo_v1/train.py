"""Synchronous persistent-bank coordinator, one native-verl optimizer step/session.

No scheduler submission. Entry is deliberately --prepare-only unless --train.
Training is a custom verl recipe, NOT an invocation of the stock RayPPOTrainer:
its growing chat trajectories would violate CoverMem's per-turn observations.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
from copy import deepcopy
import fcntl
import os
from pathlib import Path
import random
import signal
import statistics
import time
from datetime import timedelta
from common import ROOT, config, resolve, atomic, read, digest, log, retained_ids, storage_chars
from prepare import prepare
from environment import collect_stream
from rewards import attach_advantages
from checkpoint import latest, publish

def diagnostics(streams,rows,tokenizer):
    def action_name(row):
        a=row.get('action')
        name=a.get('action') if isinstance(a,dict) else None
        return name if isinstance(name,str) else 'UNPARSEABLE'
    ids=[retained_ids(s['end_bank']) for s in streams]
    pairs=[len(ids[i]&ids[j])/max(1,len(ids[i]|ids[j])) for i in range(4) for j in range(i+1,4)]
    return {'mid_rewards':[s['mid_score']['reward'] for s in streams],
        'end_rewards':[s['end_score']['reward'] for s in streams],
        'start_rewards_on_end_questions':[s['start_score_on_end_questions']['reward'] for s in streams],
        'end_complete':[s['end_score']['complete'] for s in streams],
        'end_recall':[s['end_score']['recall'] for s in streams],
        'storage_chars':[storage_chars(s['end_bank']) for s in streams],
        'content_tokens':[sum(len(tokenizer.encode(e['content'],add_special_tokens=False)) for e in s['end_bank']) for s in streams],
        'distinct_bank_id_sets':len({tuple(sorted(x)) for x in ids}),
        'mean_pairwise_jaccard':statistics.mean(pairs),
        'attempted_actions':dict(Counter(action_name(r) for r in rows)),
        'executed_actions':dict(Counter(r['action']['action'] for r in rows if r['valid'])),
        'invalid_attempts':sum(not r['valid'] for r in rows),
        'invalid_rate':sum(not r['valid'] for r in rows)/max(len(rows),1),
        'reward_tie_mid':len({s['mid_score']['reward'] for s in streams})==1,
        'reward_tie_end':len({s['end_score']['reward'] for s in streams})==1,
        'mid_complete':[s['mid_score']['complete'] for s in streams],
        'mid_recall':[s['mid_score']['recall'] for s in streams],
        'end_hit':[s['end_score']['hit'] for s in streams],
        'mean_response_tokens':statistics.mean(len(r['response_ids']) for r in rows),
        'invalid_errors':dict(Counter(r['error'] for r in rows if not r['valid'])),
        'failed_turns':sum(r['decision_end'] and not r['valid'] for r in rows),
        'invalid_positive_advantages':sum(not r['valid'] and r['advantage']>0 for r in rows)}

def load_model(cfg,adapter,device):
    import torch
    from transformers import AutoModelForCausalLM
    from peft import PeftModel
    from actor import disable_dropout
    from performance_actor import BatchedCoverageActor
    from performance_runtime import load_profile
    base=AutoModelForCausalLM.from_pretrained(resolve(cfg['paths']['base_model']),
        dtype=getattr(torch,cfg['training']['dtype']),local_files_only=True,
        attn_implementation=cfg['training']['attention_implementation'])
    model=PeftModel.from_pretrained(base,adapter,is_trainable=True,adapter_name='default')
    model.load_adapter(resolve(cfg['paths']['sft_adapter']),adapter_name='reference',is_trainable=False)
    model.set_adapter('default')
    for name,p in model.named_parameters():p.requires_grad_('lora_' in name and '.default.' in name)
    disable_dropout(model)
    if cfg['training']['gradient_checkpointing']:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        model.enable_input_require_grads()
    model.config.use_cache=False
    parameters=[p for p in model.parameters() if p.requires_grad]
    if not parameters:raise RuntimeError('SFT adapter is not trainable')
    optimizer=torch.optim.AdamW(parameters,lr=cfg['training']['learning_rate'],
        betas=tuple(cfg['training']['adam_betas']),weight_decay=cfg['training']['weight_decay'])
    actor=BatchedCoverageActor(cfg,model,optimizer,device)
    actor.performance=load_profile(cfg,int(os.environ.get('RANK','0')))
    return model,actor

def run(cfg,report,sessions):
    import torch
    import torch.distributed as dist
    from transformers import AutoTokenizer
    from actor import move_optimizer
    from performance_actor import balance_rows
    from rollout import VLLMServer
    if cfg['validation']['require_slurm_for_gpu'] and not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('GPU execution requires an actual Slurm allocation; no job is submitted by this code')
    rank=int(os.environ.get('RANK','0'));local=int(os.environ.get('LOCAL_RANK','0'));world=int(os.environ.get('WORLD_SIZE','1'))
    visible=os.environ.get('CUDA_VISIBLE_DEVICES','').split(',')
    if not 1<=world<=4 or len(visible)<world or not all(visible[:world]):raise RuntimeError('Allocate 1..4 visible GPUs')
    if int(os.environ.get('LOCAL_WORLD_SIZE',str(world)))!=world:raise RuntimeError('This recipe currently supports one node')
    torch.cuda.set_device(local);device=torch.device('cuda',local)
    dist.init_process_group('nccl',timeout=timedelta(minutes=30),device_id=device)
    torch.set_num_threads(2)
    output=resolve(cfg['paths']['output']);output.mkdir(parents=True,exist_ok=True)
    lock=None;server=None
    try:
        if rank==0:
            lock=(output/'run.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            old=read(output/'prepared.json') if (output/'prepared.json').exists() else report
            if old['fingerprint']!=report['fingerprint']:raise RuntimeError('Changed experiment; select a new output directory')
            atomic(output/'prepared.json',report)
        dist.barrier()
        saved=latest(output,report['fingerprint'])
        if saved:
            checkpoint,state=saved
            if state['world_size']!=world:raise RuntimeError('Resume world size differs; keep RNG/optimizer topology fixed')
            adapter=checkpoint/'adapter'
        else:
            state={'next_sequence':0,'updates':0,'banks':[[],[],[],[]],'history_id':None,'epoch':0,'world_size':world}
            adapter=resolve(cfg['paths']['sft_adapter'])
        total=len(sessions)*cfg['training']['data_epochs']
        if cfg['training']['max_sessions'] is not None:total=min(total,cfg['training']['max_sessions'])
        if state['next_sequence']>=total:
            if rank==0:
                atomic(output/'TRAIN_COMPLETE.json',{'status':'complete','fingerprint':report['fingerprint'],
                    'sessions_processed':total,'optimizer_steps':state['updates'],'final_adapter':str(adapter),
                    'reward':'0.7 Complete + 0.3 Recall','qa_calls':0,'qg_calls':0})
                log('already_complete',sessions=total)
            return
        random.seed(cfg['training']['seed']+rank);torch.manual_seed(cfg['training']['seed']);torch.cuda.manual_seed(cfg['training']['seed']+rank)
        model,actor=load_model(cfg,adapter,device)
        if saved:
            training=torch.load(checkpoint/'training.pt',map_location='cpu',weights_only=False)
            actor.actor_optimizer.load_state_dict(training['optimizer'])
            rng=training['rng'][rank];random.setstate(rng['python']);torch.set_rng_state(rng['torch'])
            torch.cuda.set_rng_state(rng['cuda'],device)
        tokenizer=AutoTokenizer.from_pretrained(resolve(cfg['paths']['tokenizer']),local_files_only=True)
        server=VLLMServer(cfg,output/'servers'/f'rank{rank}',visible[local],rank)
        server.start()
        for sequence in range(state['next_sequence'],total):
            session_started=time.monotonic()
            epoch,index=divmod(sequence,len(sessions));session=sessions[index]
            if state['history_id']!=session['history_id'] or state['epoch']!=epoch:
                state['banks']=[[],[],[],[]]
            initial=deepcopy(state['banks']);version=state['updates']
            directory=output/'sessions'/f'{sequence:06d}_{session["history_id"]}_{session["session_id"]}'
            server.activate(adapter,version)
            assigned=[i for i in range(4) if i%world==rank]
            if rank==0:log('session_rollout',sequence=sequence,epoch=epoch,history=session['history_id'],session=session['session_id'],policy_version=version)
            with ThreadPoolExecutor(max_workers=len(assigned)) as pool:
                futures=[pool.submit(collect_stream,session,i,initial[i],server,tokenizer,cfg,
                    directory/f'stream{i}',report['sampling_fingerprint'],version,epoch) for i in assigned]
                own=[f.result() for f in futures]
            server.sleep()
            gathered=[None]*world;dist.all_gather_object(gathered,own)
            streams=sorted([s for group in gathered for s in group],key=lambda s:s['stream'])
            rows,advantages=attach_advantages(streams,session['mid'],cfg['reward'],session,initial)
            if rank==0:
                fields=('stream','turn_index','attempt','valid','segment','before_chars',
                        'raw_rtg','pre_mid_score','pre_end_score','peer_mean_rtg','residual',
                        'bank_advantage','advantage')
                atomic(directory/'credit_assignments.json',{
                    'fingerprint':report['fingerprint'],'summary':advantages,
                    'outputs':[{**{k:r[k] for k in fields},'response_len':len(r['response_ids'])} for r in rows]})
            assert all(r['policy_version']==version for r in rows)
            if any(not r['valid'] and r['advantage']>0 for r in rows):raise RuntimeError('Invalid action received positive signal')
            # All four streams finish before any rank starts gradient calculation.
            dist.barrier()
            do_update=advantages['has_signal'] or not cfg['training']['skip_zero_signal_sessions']
            rollout_seconds=time.monotonic()-session_started;optimization_started=time.monotonic()
            stats={'optimizer_steps':0,'reason':'zero_signal'}
            if do_update:
                model.to(device);move_optimizer(actor.actor_optimizer,device)
                global_tokens=sum(len(r['response_ids']) for r in rows)
                assigned_rows=balance_rows(rows,world)[rank]
                stats=actor.update_session(assigned_rows,global_tokens)
                state['updates']+=1
                model.to('cpu');move_optimizer(actor.actor_optimizer,'cpu');torch.cuda.empty_cache()
            state.update(next_sequence=sequence+1,history_id=session['history_id'],epoch=epoch,
                         banks=[s['end_bank'] for s in streams])
            rng={'python':random.getstate(),'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state(device)}
            rngs=[None]*world;dist.all_gather_object(rngs,rng)
            if rank==0:
                metrics={'sequence':sequence,'epoch':epoch,'history':session['history_id'],'session':session['session_id'],
                         'policy_version_before':version,'policy_version_after':state['updates'],
                         'rollout_and_score_seconds':rollout_seconds,
                         'optimization_and_sync_seconds':time.monotonic()-optimization_started,
                         'advantages':advantages,'optimization':stats,**diagnostics(streams,rows,tokenizer)}
                def save_payload(p):
                    model.save_pretrained(p/'adapter',selected_adapters=['default'],safe_serialization=True)
                    torch.save({'optimizer':actor.actor_optimizer.state_dict(),'rng':rngs},p/'training.pt')
                    atomic(p/'metrics.json',metrics)
                checkpoint=publish(output,sequence+1,report['fingerprint'],state,save_payload)
                atomic(directory/'metrics.json',metrics)
                atomic(output/'progress.json',{'status':'running','completed_sessions':sequence+1,
                    'total_sessions':total,'optimizer_steps':state['updates'],'latest_metrics':metrics})
                log('session_committed',completed=sequence+1,total=total,updates=state['updates'],end_rewards=metrics['end_rewards'])
            dist.barrier()
            adapter=output/'checkpoints'/f'session_{sequence+1:06d}'/'adapter'
            del rows,streams,gathered,own,futures
        if rank==0:
            atomic(output/'TRAIN_COMPLETE.json',{'status':'complete','fingerprint':report['fingerprint'],
                'sessions_processed':total,'optimizer_steps':state['updates'],'final_adapter':str(adapter),
                'reward':'0.7 Complete + 0.3 Recall','qa_calls':0,'qg_calls':0})
            log('complete',sessions=total,updates=state['updates'])
    finally:
        if server:server.close()
        if lock:lock.close()
        if dist.is_initialized():dist.destroy_process_group()

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config');parser.add_argument('--train',action='store_true')
    args=parser.parse_args();cfg=config(args.config);report,sessions=prepare(cfg)
    if not args.train:
        atomic(ROOT/'validation/preflight.json',report)
        print({'status':'prepared_only','sessions':len(sessions),'fingerprint':report['fingerprint']})
    else:
        def stopped(signum,frame):raise SystemExit(128+signum)
        signal.signal(signal.SIGTERM,stopped);signal.signal(signal.SIGINT,stopped)
        run(cfg,report,sessions)

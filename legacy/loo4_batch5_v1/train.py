"""Synchronous round-robin batches, one optimizer step across active sessions.

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
from batching import (build_batches, history_order, initial_state, validate_state,
                      advance_state, schedule_digest)

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
        batches=build_batches(sessions,cfg['training']['max_sessions'])
        histories=history_order(sessions)
        total=sum(map(len,batches))
        total_batches=len(batches)
        saved=latest(output,report['fingerprint'])
        if saved:
            checkpoint,state=saved
            if state['world_size']!=world:raise RuntimeError('Resume world size differs; keep RNG/optimizer topology fixed')
            adapter=checkpoint/'adapter'
        else:
            state=initial_state(histories,batches,world)
            adapter=resolve(cfg['paths']['sft_adapter'])
        validate_state(state,histories,batches,world)
        if state['next_batch']>=total_batches:
            if rank==0:
                atomic(output/'TRAIN_COMPLETE.json',{'status':'complete','fingerprint':report['fingerprint'],
                    'sessions_processed':total,'completed_sessions':total,'batches_processed':total_batches,
                    'optimizer_steps':state['updates'],'final_adapter':str(adapter),
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
        for batch_index in range(state['next_batch'],total_batches):
            batch_started=time.monotonic()
            batch=batches[batch_index]
            version=state['updates'];epoch=0
            batch_directory=output/'batches'/f'{batch_index:06d}'
            session_batches=[];session_metrics=[];end_banks={}
            server.activate(adapter,version)
            assigned=[i for i in range(4) if i%world==rank]
            if rank==0:
                log('batch_rollout',batch=batch_index,total_batches=total_batches,
                    histories=[s['history_id'] for s in batch],policy_version=version)
            for offset,session in enumerate(batch):
                session_started=time.monotonic()
                hid=session['history_id']
                initial=deepcopy(state['banks_by_history'][hid])
                sequence=state['completed_sessions']+offset
                directory=output/'sessions'/f'{sequence:06d}_{hid}_{session["session_id"]}'
                if rank==0:
                    log('session_rollout',batch=batch_index,sequence=sequence,epoch=epoch,
                        history=hid,session=session['session_id'],policy_version=version)
                with ThreadPoolExecutor(max_workers=len(assigned)) as pool:
                    futures=[pool.submit(collect_stream,session,i,initial[i],server,tokenizer,cfg,
                        directory/f'stream{i}',report['sampling_fingerprint'],version,epoch) for i in assigned]
                    own=[f.result() for f in futures]
                gathered=[None]*world;dist.all_gather_object(gathered,own)
                streams=sorted([s for group in gathered for s in group],key=lambda s:s['stream'])
                if [s['stream'] for s in streams]!=list(range(4)):
                    raise RuntimeError('Each session requires exactly four distinct streams')
                rows,advantages=attach_advantages(streams,session['mid'],cfg['reward'],session,initial)
                if not all(r['policy_version']==version for r in rows):
                    raise RuntimeError('Policy changed within a batch')
                if any(not r['valid'] and r['advantage']>0 for r in rows):
                    raise RuntimeError('Invalid action received positive signal')
                active=bool(advantages['has_signal'])
                global_tokens=sum(len(r['response_ids']) for r in rows)
                if active:
                    if global_tokens<=0:raise RuntimeError('Active session has no response tokens')
                    # Match the old global response-token mean separately for
                    # each session. The actor averages these active session losses.
                    session_batches.append((balance_rows(rows,world)[rank],global_tokens))
                end_banks[hid]=[s['end_bank'] for s in streams]
                if rank==0:
                    fields=('stream','turn_index','attempt','valid','segment','before_chars',
                            'raw_rtg','pre_mid_score','pre_end_score','peer_mean_rtg','residual',
                            'bank_advantage','advantage')
                    atomic(directory/'credit_assignments.json',{
                        'fingerprint':report['fingerprint'],'batch_index':batch_index,
                        'summary':advantages,
                        'outputs':[{**{k:r[k] for k in fields},'response_len':len(r['response_ids'])} for r in rows]})
                    session_metrics.append({'sequence':sequence,'epoch':epoch,'history':hid,
                        'session':session['session_id'],'batch_index':batch_index,
                        'policy_version_before':version,'active_for_batch_update':active,
                        'global_response_tokens':global_tokens,
                        'rollout_and_score_seconds':time.monotonic()-session_started,
                        'advantages':advantages,**diagnostics(streams,rows,tokenizer)})
                # Rank-local update samples remain buffered; complete gathered
                # trajectories are released before collecting the next history.
                del rows,streams,gathered,own,futures
            server.sleep()
            # Every session in the batch used the same frozen policy version.
            dist.barrier()
            active_sessions=len(session_batches)
            do_update=active_sessions>0
            rollout_seconds=time.monotonic()-batch_started;optimization_started=time.monotonic()
            stats={'optimizer_steps':0,'reason':'zero_signal'}
            if do_update:
                model.to(device);move_optimizer(actor.actor_optimizer,device)
                stats=actor.update_batch(session_batches,active_sessions)
                if stats['optimizer_steps']!=1:raise RuntimeError('Exactly one optimizer step per active batch required')
                model.to('cpu');move_optimizer(actor.actor_optimizer,'cpu');torch.cuda.empty_cache()
            state=advance_state(state,batch,end_banks,do_update)
            validate_state(state,histories,batches,world)
            rng={'python':random.getstate(),'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state(device)}
            rngs=[None]*world;dist.all_gather_object(rngs,rng)
            if rank==0:
                for item in session_metrics:
                    item.update(policy_version_after=state['updates'],
                        active_session_loss_weight=(1/active_sessions if item['active_for_batch_update'] else 0),
                        optimizer_boundary='batch',batch_optimizer_steps=stats['optimizer_steps'])
                metrics={'batch_index':batch_index,'epoch':epoch,'sessions_in_batch':len(batch),
                         'active_sessions':active_sessions,
                         'policy_version_before':version,'policy_version_after':state['updates'],
                         'rollout_and_score_seconds':rollout_seconds,
                         'optimization_and_sync_seconds':time.monotonic()-optimization_started,
                         'session_metrics':session_metrics,'optimization':stats,
                         'session_cursors':dict(state['session_cursors']),
                         'loss_aggregation':'mean_of_active_session_response_token_means'}
                def save_payload(p):
                    model.save_pretrained(p/'adapter',selected_adapters=['default'],safe_serialization=True)
                    torch.save({'optimizer':actor.actor_optimizer.state_dict(),'rng':rngs},p/'training.pt')
                    atomic(p/'metrics.json',metrics)
                checkpoint=publish(output,batch_index+1,report['fingerprint'],state,save_payload)
                atomic(batch_directory/'metrics.json',metrics)
                for item in session_metrics:
                    directory=output/'sessions'/f'{item["sequence"]:06d}_{item["history"]}_{item["session"]}'
                    atomic(directory/'metrics.json',item)
                atomic(output/'progress.json',{'status':'running','completed_sessions':state['completed_sessions'],
                    'total_sessions':total,'completed_batches':state['next_batch'],'total_batches':total_batches,
                    'optimizer_steps':state['updates'],'latest_metrics':metrics})
                log('batch_committed',completed_batches=state['next_batch'],total_batches=total_batches,
                    completed_sessions=state['completed_sessions'],total_sessions=total,
                    updates=state['updates'],active_sessions=active_sessions)
            dist.barrier()
            adapter=output/'checkpoints'/f'batch_{batch_index+1:06d}'/'adapter'
            del session_batches,end_banks,session_metrics
        if rank==0:
            atomic(output/'TRAIN_COMPLETE.json',{'status':'complete','fingerprint':report['fingerprint'],
                'sessions_processed':total,'completed_sessions':total,'batches_processed':total_batches,
                'optimizer_steps':state['updates'],'final_adapter':str(adapter),
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

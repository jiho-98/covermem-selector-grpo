"""Disposable GPU handoff/long-bank check inside the requested Slurm allocation.

Synthetic nonzero advantages are used ONLY for this runtime check, never saved
as research training data. Full training starts afresh from the original SFT.
"""
import argparse
from datetime import timedelta
import os
from pathlib import Path
import signal
from common import ROOT,config,read,resolve,prompt_ids,atomic,storage_chars,verify_bank,log
from prepare import prepare

def run(cfg):
    import torch
    import torch.distributed as dist
    from transformers import AutoTokenizer
    from actor import move_optimizer,pack_action
    from train import load_model
    from rollout import VLLMServer
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('GPU smoke requires Slurm')
    rank=int(os.environ['RANK']);local=int(os.environ['LOCAL_RANK']);world=int(os.environ['WORLD_SIZE'])
    report,sessions=prepare(cfg);directory=resolve(cfg['paths']['output'])/'runtime_smoke';directory.mkdir(parents=True,exist_ok=True)
    torch.cuda.set_device(local);device=torch.device('cuda',local)
    dist.init_process_group('nccl',device_id=device,timeout=timedelta(minutes=30));server=None
    try:
        model,actor=load_model(cfg,resolve(cfg['paths']['sft_adapter']),device)
        tokenizer=AutoTokenizer.from_pretrained(resolve(cfg['paths']['tokenizer']),local_files_only=True)
        # Near-full valid RAW bank, training source only. No future or held-out IDs.
        records=sessions[0]['records'];bank=[];used=0;budget=cfg['environment']['bank_budget_chars']
        for r in records[:-1]:
            if used+len(r['content'])<=budget:
                bank.append({'memory_id':'m_'+r['turn_id'],'content':r['content'],'source_turn_ids':[r['turn_id']]});used+=len(r['content'])
        verify_bank(bank,records[:-1],budget)
        prompt=prompt_ids(tokenizer,bank,records[-1],budget)
        if len(prompt)+cfg['rollout']['max_new_tokens']>cfg['rollout']['max_model_len']:
            raise RuntimeError('Near-full RAW bank cannot fit fixed model context')
        physical=os.environ['CUDA_VISIBLE_DEVICES'].split(',')[local]
        server=VLLMServer(cfg,directory/f'server{rank}',physical,rank);server.start()
        server.activate(resolve(cfg['paths']['sft_adapter']),0)
        result=server.generate(prompt,42+rank)
        row={'prompt_ids':prompt,**result,'advantage':1. if rank%2==0 else -.5,'segment':'front'}
        count=torch.tensor([len(result['response_ids'])],device=device);dist.all_reduce(count)
        server.sleep();model.to(device);move_optimizer(actor.actor_optimizer,device)
        from performance_runtime import calibrate,publish_profiles
        # This job is one H200. Do not run independent candidate fallbacks across
        # multiple ranks' collectives; use the original path there until tested.
        if world==1:
            perf=calibrate(actor,row,cfg)
        else:
            perf={'hardware':torch.cuda.get_device_name(device),'selected':{'mode':'baseline'},
                  'reason':'Multi-GPU runtime calibration intentionally not enabled'}
        perf_reports=[None]*world;dist.all_gather_object(perf_reports,perf)
        if rank==0:publish_profiles(cfg,perf_reports)
        dist.barrier()
        stats=actor.update_session([row],int(count.item()))
        model.to('cpu');move_optimizer(actor.actor_optimizer,'cpu');torch.cuda.empty_cache()
        adapter=directory/'disposable_adapter'
        if rank==0:model.save_pretrained(adapter,selected_adapters=['default'],safe_serialization=True)
        dist.barrier();server.activate(adapter,1)
        updated=server.generate(prompt,84+rank);server.sleep()
        # Verify the newly loaded LoRA against the UPDATED actor, not only the initial SFT.
        row2={'prompt_ids':prompt,**updated,'advantage':0.}
        model.to(device);actor.actor_module.eval()
        with torch.no_grad():lp=actor._forward_micro_batch(pack_action(row2,device).batch)['log_probs'][0]
        old=torch.tensor(updated['old_log_probs'],device=device);gap=(lp-old).abs()
        if gap.mean().item()>cfg['validation']['max_preupdate_mean_abs_logprob_gap'] or gap.max().item()>cfg['validation']['max_preupdate_logprob_gap']:
            raise RuntimeError('Updated vLLM adapter does not match actor')
        own={'rank':rank,'status':'passed','content_chars':storage_chars(bank),'prompt_tokens':len(prompt),
             'initial_update':stats,'performance_calibration':perf,'reloaded_adapter_mean_gap':gap.mean().item(),
             'reloaded_adapter_max_gap':gap.max().item(),'actor_peak_allocated_gib':torch.cuda.max_memory_allocated(device)/2**30}
        gathered=[None]*world;dist.all_gather_object(gathered,own)
        if rank==0:
            atomic(directory/'GPU_SMOKE_COMPLETE.json',{'status':'passed','fingerprint':report['fingerprint'],
                   'world_size':world,'ranks':gathered,'disposable_update_not_used_in_training':True})
            log('gpu_runtime_smoke_passed',world=world,prompt_tokens=len(prompt))
    finally:
        if server:server.close()
        dist.destroy_process_group()

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config');args=parser.parse_args()
    def stop(signum,frame):raise SystemExit(128+signum)
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    run(config(args.config))

"""Only the new final GRPO adapter; same held-out RAW manager evaluation as SFT."""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import fcntl
import os
import signal
from pathlib import Path
from common import ROOT,config,read,atomic,digest,file_hash,resolve,log
from prepare import prepare
from checkpoint import latest
from eval_loader import prepare_locomo
import eval_core as core
from rollout import VLLMServer

def eval_inputs(cfg):
    # Training prepare() loads ONLY train histories/QG; held-out data stays in this process.
    report,_=prepare(cfg);split=read(ROOT/'assets/split.json');source=ROOT/'assets/locomo10_eval.json'
    if file_hash(source)!=split['source']['sha256']:raise RuntimeError('Changed evaluation source')
    histories=[prepare_locomo(source,x) for x in split['test']]
    assert not set(split['test'])&set(split['train'])
    assert (len(histories),sum(len(h['records']) for h in histories),sum(q['eligible'] for h in histories for q in h['questions']))==(5,2769,729)
    return report,histories

def evaluate(cfg):
    from transformers import AutoTokenizer
    report,histories=eval_inputs(cfg);output=resolve(cfg['paths']['output'])
    complete=read(output/'TRAIN_COMPLETE.json')
    if complete['status']!='complete' or complete['fingerprint']!=report['fingerprint']:
        raise RuntimeError('Evaluation requires this exact completed training run')
    checkpoint,state=latest(output,report['fingerprint'])
    if Path(complete['final_adapter']).resolve()!=(checkpoint/'adapter').resolve():
        raise RuntimeError('Final adapter/checkpoint mismatch')
    core.ROOT=output/'evaluation';(core.ROOT/'runs').mkdir(parents=True,exist_ok=True)
    provenance={'training_fingerprint':report['fingerprint'],'adapter_sha256':file_hash(checkpoint/'adapter/adapter_model.safetensors'),
                'settings':core.SETTINGS,'budget_chars':43500,'conversations':[h['source_id'] for h in histories],
                'source_sha256':file_hash(ROOT/'assets/locomo10_eval.json'),'evaluation_code_sha256':file_hash(ROOT/'eval_core.py')}
    prepared={'fingerprint':digest(provenance),'provenance':provenance,'qa':False,'shadow':False,
              'raw_content_only':True,'manager_query_access':False,'conversations':5,'turns':2769,'eligible_questions':729,
              'scope':'student held-out five; previously development-exposed benchmark'}
    stored=core.ROOT/'runs/prepare.json'
    if stored.exists() and read(stored)!=prepared:raise RuntimeError('Changed evaluation; do not mix results')
    atomic(stored,prepared)
    lock=(core.ROOT/'run.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    server=None
    try:
        if not (core.ROOT/'runs/EVAL_COMPLETE.json').exists():
            if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('GPU evaluation requires Slurm')
            device=os.environ.get('CUDA_VISIBLE_DEVICES','').split(',')[0]
            if not device:raise RuntimeError('No evaluation GPU allocated')
            tokenizer=AutoTokenizer.from_pretrained(resolve(cfg['paths']['tokenizer']),local_files_only=True)
            server=VLLMServer(cfg,core.ROOT/'server',device)
            server.start();server.activate(checkpoint/'adapter','final')
            # Existing SFT evaluation uses the same 6-attempt, greedy JSON protocol.
            # It makes one bank per conversation; no RL or four paths at evaluation.
            core.run_history(histories[0],prepared,server.endpoint+'/v1',tokenizer,stop_after=2)
            results=[]
            with ThreadPoolExecutor(max_workers=5) as pool:
                futures=[pool.submit(core.run_history,h,prepared,server.endpoint+'/v1',tokenizer) for h in histories]
                for future in as_completed(futures):results.append(future.result());core.aggregate(prepared,results)
        sft=read(ROOT/'assets/sft_eval_summary.json');grpo=read(core.ROOT/'runs/EVAL_COMPLETE.json')
        assert sft['eligible_questions']==grpo['eligible_questions']==729
        def row(value):return {**value['pooled_question_mean'],**value['mean_storage']}
        comparison={'status':'complete','aggregation':'pooled question mean','eligible_questions':729,
                    'condition':'Active only; RAW content 43500 chars; frozen policy, no QG/QA at inference',
                    'SFT':row(sft),'SFT_GRPO':row(grpo),
                    'delta_percentage_points':{k:100*(grpo['pooled_question_mean'][k]-sft['pooled_question_mean'][k]) for k in ('complete','recall','hit')},
                    'sft_reference_reused':True,'sft_rerun':False,'training_fingerprint':report['fingerprint']}
        atomic(core.ROOT/'comparison.json',comparison);log('evaluation_complete',comparison=comparison)
    finally:
        if server:server.close()
        lock.close()

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config');parser.add_argument('--prepare-only',action='store_true')
    args=parser.parse_args();cfg=config(args.config)
    if args.prepare_only:
        r,h=eval_inputs(cfg);print({'status':'evaluation_prepared','histories':len(h),'turns':2769,'questions':729,'fingerprint':r['fingerprint']})
    else:
        def stopped(signum,frame):raise SystemExit(128+signum)
        signal.signal(signal.SIGTERM,stopped);signal.signal(signal.SIGINT,stopped)
        evaluate(cfg)

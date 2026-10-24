"""Frozen final-bank evidence evaluation, one greedy selector call per REPLACE.

No N-way sampling, CPU oracle, question-based routing, GRPO updates or QA reader.
HF greedy/unconstrained JSON differs from historical vLLM JSON-constrained eval;
all new matched controls must use this same evaluation implementation.
"""
import argparse
from collections import Counter
import fcntl
import os
from pathlib import Path
from .active_policy import apply_action, parse_action, make_observation, build_messages
from .config import load, resolve
from .core import atomic, read, digest, verify_bank, retained_ids, log
from .selection import candidates, messages, parse_target
from .checkpoint import latest, validate

def run_history(history, backend, cfg, directory, fingerprint, selector_enabled):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / 'state.json'
    state = read(path) if path.exists() else {'fingerprint':fingerprint,'next_turn':0,'bank':[], 'actions':{},
                                           'invalid_manager_attempts':0,'invalid_turns':0,'selector_fallbacks':0}
    if state['fingerprint'] != fingerprint:
        raise ValueError('Changed evaluation identity')
    bank, counts = state['bank'], Counter(state['actions'])
    verify_bank(bank,history['records'][:state['next_turn']],43500)
    for i in range(state['next_turn'],len(history['records'])):
        turn, error, applied, attempts = history['records'][i], None, False, []
        for attempt in range(6):  # original evaluation protocol: initial + five retries
            row = backend.sample('manager', build_messages(make_observation(bank,turn,43500),error),42,
                                 cfg['rollout']['manager_max_new_tokens'],greedy=True)
            try:
                if row['finish_reason'] != 'stop': raise ValueError('Truncated manager output')
                action = parse_action(row['text'])
                after = apply_action(bank,turn,action,43500)
            except (ValueError,TypeError,KeyError) as exc:
                error=str(exc);state['invalid_manager_attempts']+=1
                attempts.append({'text':row['text'],'valid':False,'error':error})
                continue
            sampled_action = dict(action); selector_row = None
            if action['action']=='REPLACE' and selector_enabled:
                eligible=candidates(bank,turn,action,43500)
                selector_row=backend.sample('selector',messages(bank,turn,43500,eligible),42,
                                             cfg['rollout']['selector_max_new_tokens'],greedy=True)
                try:
                    if selector_row['finish_reason']!='stop':raise ValueError('Truncated selector output')
                    target=parse_target(selector_row['text'],eligible)
                    action={**action,'target_id':target};after=eligible[target]
                except (ValueError,TypeError,KeyError):state['selector_fallbacks']+=1
            attempts.append({'text':row['text'],'valid':True,'sampled_action':sampled_action,
                             'executed_action':action,'selector_text':selector_row['text'] if selector_row else None})
            bank=after;counts[action['action']]+=1;applied=True
            break
        if not applied:state['invalid_turns']+=1
        verify_bank(bank,history['records'][:i+1],43500)
        atomic(directory/'turns'/f'{i:06d}.json',{'index':i,'fingerprint':fingerprint,'attempts':attempts,'after_hash':digest(bank)})
        state.update(next_turn=i+1,bank=bank,actions=dict(counts))
        atomic(path,state)
        if i % 25 == 0:log('eval_progress',history=history['source_id'],turn=i+1)
    kept=retained_ids(bank); details=[]
    # Official labels are used ONLY after all memory decisions for this conversation.
    for q in history['questions']:
        if not q['eligible']:continue
        evidence=set(q['evidence_ids']);overlap=kept&evidence
        details.append({'question_id':q['question_id'],'complete':float(overlap==evidence),
                        'recall':len(overlap)/len(evidence),'hit':float(bool(overlap))})
    result={'status':'complete','fingerprint':fingerprint,'history':history['source_id'],
            'questions':len(details),'metrics':{k:sum(q[k] for q in details)/len(details) for k in ('complete','recall','hit')},
            'state':state,'per_question':details}
    atomic(directory/'result.json',result)
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True)
    p.add_argument('--initial-manager-only',action='store_true',help='Frozen A checkpoint control, selector disabled; separate output')
    a=p.parse_args();cfg=load(a.config)
    from .preflight import inspect
    report,_,_=inspect(cfg)
    if not os.environ.get('COVERMEM_STAGE_OWNER'):raise RuntimeError('Evaluation requires supervised execution')
    if cfg['execution']['require_slurm'] and not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Slurm allocation required')
    import torch
    from .backend import Backend
    if torch.cuda.device_count()!=1:raise RuntimeError('Expose exactly one GPU')
    output=resolve(cfg['paths']['output'])
    if a.initial_manager_only:
        if cfg['variant']!='A':raise ValueError('This control is for the completed manager used by A')
        checkpoint_path=None
        evaluation=output/'evaluation_initial_manager'
    else:
        complete=read(output/'TRAIN_COMPLETE.json')
        if complete['status']!='complete' or complete['fingerprint']!=report['fingerprint']:
            raise ValueError('Evaluation requires validated completed training, not a partial checkpoint')
        saved=latest(output,report['fingerprint'])
        marker_path=Path(complete['checkpoint'])
        if not marker_path.is_absolute():marker_path=output/marker_path
        if saved is None or saved[0].resolve()!=marker_path.resolve():
            raise ValueError('Final checkpoint mismatch')
        checkpoint_path=saved[0];evaluation=output/'evaluation'
    histories=read(resolve(cfg['paths']['assets'])/'evaluation_histories.json')
    if len(histories)!=5 or sum(q['eligible'] for h in histories for q in h['questions'])!=729:
        raise ValueError('Expected held-out five / 729 eligible questions')
    eval_id=digest({'train':report['fingerprint'],'initial_manager_only':a.initial_manager_only,
                   'protocol':'hf_greedy_6_manager_attempts_1_selector_attempt_no_oracle_v1'})
    evaluation.mkdir(parents=True,exist_ok=True)
    with (evaluation/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        b=Backend(cfg,checkpoint_path)
        enabled=cfg['variant']!='vanilla' and not a.initial_manager_only
        results=[run_history(h,b,cfg,evaluation/h['source_id'],eval_id,enabled) for h in histories]
        total=sum(r['questions'] for r in results)
        atomic(evaluation/'EVAL_COMPLETE.json',{'status':'complete','fingerprint':report['fingerprint'],
            'evaluation_fingerprint':eval_id,'questions':total,
            'pooled':{k:sum(r['metrics'][k]*r['questions'] for r in results)/total for k in ('complete','recall','hit')},
            'histories':[{k:v for k,v in r.items() if k not in ('state','per_question')} for r in results],
            'qa_reader_calls':0,'scope':'RAW evidence preservation, not QA accuracy'})

if __name__=='__main__':main()

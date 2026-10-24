"""Four persistent RAW streams; per-turn contexts, bounded attempts, durable replay."""
from copy import deepcopy
from pathlib import Path
import json
import math
import os
from common import (apply_action, parse_action, prompt_ids, canonical, digest,
                    storage_chars, verify_bank, now, atomic)
from rewards import score

def load_trace(path):
    path=Path(path)
    if not path.exists(): return []
    data=path.read_bytes(); last=data.rfind(b'\n')+1
    # Only an interrupted, uncommitted trailing record may be discarded.
    if last != len(data):
        with path.open('r+b') as f: f.truncate(last); f.flush(); os.fsync(f.fileno())
    return [json.loads(line) for line in data[:last].splitlines()]

def append_trace(path,row):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a') as f:
        f.write(canonical(row)+'\n');f.flush();os.fsync(f.fileno())

def validate_completion(result):
    ids=result['response_ids'];probs=result['old_log_probs']
    if not ids or len(ids)!=len(probs) or any(type(i) is not int for i in ids):
        raise RuntimeError('Missing actual sampled token IDs/log-probabilities')
    if any(not math.isfinite(x) or x>1e-4 for x in probs):
        raise RuntimeError('Nonfinite/invalid sampling log-probability')

def collect_stream(session, stream, initial_bank, generator, tokenizer, cfg,
                   directory, fingerprint, policy_version, epoch):
    bank=deepcopy(initial_bank); budget=cfg['environment']['bank_budget_chars']
    verify_bank(bank,session['records'][:session['start']],budget)
    start_score=score(bank,session['end_questions']['questions'])
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    identity=digest({'fingerprint':fingerprint,'policy_version':policy_version,'epoch':epoch,
                     'history':session['history_id'],'session':session['session_id'],
                     'stream':stream,'initial_bank':bank})
    trace=directory/'attempts.jsonl'; cached=load_trace(trace); cursor=0; attempts=[]
    mid_bank=None; records=session['records']; limit=cfg['environment']['max_action_attempts']
    for ti in range(session['start'],session['end']):
        current=records[ti];error=None
        for attempt in range(limit):
            ids=prompt_ids(tokenizer,bank,current,budget,error)
            if len(ids)+cfg['rollout']['max_new_tokens']>cfg['rollout']['max_model_len']:
                raise RuntimeError('Context exceeds limit: never truncate RAW or shorten this action silently')
            before=digest(bank);key=digest([identity,ti,attempt,before,digest(ids)])
            if cursor<len(cached):
                row=cached[cursor]
                if row['request_key']!=key:raise RuntimeError('Resume trace/input/policy mismatch')
                result={k:row[k] for k in ('text','response_ids','old_log_probs','finish_reason')}
            else:
                # Stable seed per request, independent of scheduling and worker count.
                seed=int(key[:8],16)%2147483647
                result=generator.generate(ids,seed)
            validate_completion(result)
            action=None;candidate=None;valid=False;new_error=None
            try:
                if result['finish_reason']!='stop':raise ValueError('Incomplete output: '+result['finish_reason'])
                action=parse_action(result['text'])
                candidate=apply_action(bank,current,action,budget);valid=True
            except (ValueError,TypeError,KeyError) as exc:new_error=str(exc)
            next_bank=candidate if valid else bank
            row={'request_key':key,'stream':stream,'turn_index':ti,'turn_id':current['turn_id'],
                 'attempt':attempt,'policy_version':policy_version,'prompt_ids':ids,**result,
                 'valid':valid,'error':new_error,'action':action,
                 'decision_end':valid or attempt==limit-1,'before_hash':before,'after_hash':digest(next_bank)}
            if cursor<len(cached):
                for k in ('valid','error','action','decision_end','before_hash','after_hash'):
                    if row[k]!=cached[cursor][k]:raise RuntimeError('Executor replay disagreement: '+k)
            else:append_trace(trace,row)
            attempts.append(row);cursor+=1;bank=next_bank;error=new_error
            if valid:break
        verify_bank(bank,records[:ti+1],budget)
        atomic(directory/'progress.json',{'turn_index':ti,'end_index':session['end'],
               'storage_chars':storage_chars(bank),'attempts':len(attempts),'updated':now()})
        if ti+1==session['mid']:mid_bank=deepcopy(bank)
    if cursor!=len(cached) and len(cached)>cursor:raise RuntimeError('Trace has extra future attempts')
    assert mid_bank is not None
    result={'stream':stream,'identity':identity,'initial_bank_hash':digest(initial_bank),
            'start_score_on_end_questions':start_score,'mid_bank':mid_bank,'end_bank':bank,
            'mid_score':score(mid_bank,session['mid_questions']['questions']),
            'end_score':score(bank,session['end_questions']['questions']), 'attempts':attempts}
    atomic(directory/'result.json',{k:v for k,v in result.items() if k!='attempts'})
    return result

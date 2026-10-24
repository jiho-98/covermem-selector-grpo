"""Strict asset/provenance checks; CPU only, never creates a training job."""
from itertools import groupby
import argparse
from pathlib import Path
import subprocess
import importlib.metadata
import sys
from common import ROOT, CONTRACT_FINGERPRINT, config, read, digest, file_hash, resolve, atomic
from rewards import validate_questions
from performance_runtime import check_control,settings

def prepare(cfg):
    control=check_control(cfg)
    runtime = {name:importlib.metadata.version(name) for name in
               ('torch','transformers','vllm','peft','ray','tensordict','hydra-core','dill','codetiming')}
    runtime['python']=sys.version.split()[0]
    if runtime!=read(ROOT/'runtime.lock.json'):raise RuntimeError('Runtime versions changed; validate before starting a new experiment')
    locked = read(ROOT / 'assets.lock.json')
    for name, expected in locked['sha256'].items():
        if file_hash(ROOT/name) != expected: raise ValueError('Changed frozen asset: '+name)
    commit = subprocess.check_output(['git','-C',str(ROOT/'vendor/verl'),'rev-parse','HEAD'],text=True).strip()
    if commit != locked['verl_commit']: raise ValueError('Unpinned verl version')
    diff = subprocess.check_output(['git','-C',str(ROOT/'vendor/verl'),'status','--porcelain','--untracked-files=no'],text=True)
    if diff.strip(): raise ValueError('Upstream verl was modified; custom code belongs outside vendor')
    report = read(resolve(cfg['paths']['sft_report']))
    if report['status'] != 'complete' or report['policy_contract']['fingerprint'] != CONTRACT_FINGERPRINT:
        raise ValueError('Not the completed, compatible RAW SFT')
    adapter = resolve(cfg['paths']['sft_adapter'])
    for name, expected in report['adapter_files_sha256'].items():
        if file_hash(adapter/name) != expected: raise ValueError('Changed initial SFT: '+name)
    hs = read(ROOT/'assets/histories.json'); split = read(ROOT/'assets/split.json')
    ids = [h['source_id'] for h in hs['histories']]
    assert ids == hs['train_ids'] == split['train'] == split['sft'] == split['rl']
    assert not set(ids) & (set(split['test']) | set(split['dev']))
    assert hs['source_sha256'] == split['source']['sha256']
    qg = read(ROOT/'assets/questions.json'); complete = read(ROOT/'assets/QG_COMPLETE.json')
    assert complete['status'] == 'complete' and qg['fingerprint'] == complete['fingerprint']
    assert qg['schema_version'] == 'turn_id_outer_or_inner_and.v1'
    sessions = []; used = set(); questions = 0; metadata_quotes = []
    for h in hs['histories']:
        cursor = 0; seen = set()
        assert all(set(r) == {'turn_id','session_id','speaker','timestamp','content'} for r in h['records'])
        for sid, group in groupby(h['records'], key=lambda r:r['session_id']):
            assert sid not in seen; seen.add(sid); records = list(group)
            if len(records) < 2: raise ValueError('One-turn session needs an explicit single-cutoff protocol')
            mid = cursor + (len(records)+1)//2; end = cursor + len(records)
            groups = []
            for half, stop in enumerate((mid,end)):
                key = f"{h['source_id']}_{sid}_h{half}"; g = qg['groups'][key]; prefix=h['records'][:stop]
                assert g['history_id'] == h['source_id'] and g['session_id'] == sid and g['half'] == half
                assert g['cutoff_turn_count'] == stop and g['cutoff_turn_id'] == prefix[-1]['turn_id']
                assert g['prefix_hash'] == digest(prefix)
                validate_questions(g['questions'], prefix)
                by_id = {r['turn_id']:r for r in prefix}
                for question in g['questions']:
                    for e in question['evidence']:
                        if e['quote'] not in by_id[e['turn_id']]['content']:
                            metadata_quotes.append({'group':key,'question_id':question['question_id'],**e})
                assert len(g['questions']) == g['actual']
                groups.append({'key':key,'questions':g['questions'],'sha256':digest(g['questions'])})
                used.add(key); questions += len(g['questions'])
            sessions.append({'history_id':h['source_id'],'session_id':sid,'start':cursor,'mid':mid,'end':end,
                             'records':h['records'], 'mid_questions':groups[0], 'end_questions':groups[1]})
            cursor=end
        assert cursor == len(h['records']) and len(seen)==h['sessions']
    assert used == set(qg['groups'])
    assert (len(sessions),len(used),questions)==(142,284,3338)
    from cumulative_questions import accumulate_questions
    cumulative_audit = accumulate_questions(sessions)
    code = {p.name:file_hash(p) for p in ROOT.glob('*.py')}
    info={'status':'prepared','histories':ids,'sessions':len(sessions),'cutoffs':len(used),'questions':questions,
          'turns':sum(len(h['records']) for h in hs['histories']), 'policy_contract':CONTRACT_FINGERPRINT,
          'assets':locked,'runtime':runtime,'sft_adapter_sha256':report['adapter_files_sha256']['adapter_model.safetensors'],
          'config':cfg,'code_sha256':code,'qa_calls':0,'qg_calls':0,'gpu_training_started':False,
          'source_metadata_quote_count':len(metadata_quotes), 'source_metadata_quotes':metadata_quotes,
          'question_pool':cumulative_audit,
          'reward_scope':'Retained RAW source-ID coverage, not semantic QA sufficiency; speaker/date source metadata is not in bank content.',
          'credit_assignment':cfg['reward']['credit_assignment'],
          'execution_optimization':settings(),'matched_research_conditions':control,
          'sampling_fingerprint':control['original_fingerprint']}
    info['fingerprint']=digest(info)
    return info,sessions

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config');a=p.parse_args()
    report,sessions=prepare(config(a.config));atomic(ROOT/'validation/preflight.json',report)
    print({k:report[k] for k in ('status','sessions','cutoffs','questions','turns','fingerprint')})

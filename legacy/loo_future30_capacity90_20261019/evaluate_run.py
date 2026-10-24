"""Final frozen greedy evaluation, separated train and held-out cohorts."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import os
from protocol import ROOT,SOURCE,prepare_run,read,atomic,digest,file_hash,resolve
from checkpoint_all import latest
from eval_loader import prepare_locomo
from rollout import VLLMServer
from common import retained_ids
from rewards import score,alternatives
import eval_core as core

def qg_result(bank,questions):
    kept=retained_ids(bank); details=[]
    for q in questions:
        sets=alternatives(q['sufficient_sets'])
        details.append(dict(question_id=q['question_id'],complete=int(any(e<=kept for e in sets)),
                            recall=max(len(e&kept)/len(e) for e in sets)))
    n=len(details)
    return dict(questions=n,complete_count=sum(q['complete'] for q in details),
                complete=sum(q['complete'] for q in details)/n,
                recall=sum(q['recall'] for q in details)/n,per_question=details)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('cohort',choices=['train','test']);a=ap.parse_args()
    cfg,report,sessions=prepare_run();output=resolve(cfg['paths']['output'])
    done=read(output/'TRAIN_COMPLETE.json')
    assert done['status']=='complete' and done['fingerprint']==report['fingerprint'] and done['sessions_processed']==142
    checkpoint,state=latest(output,report['fingerprint'])
    assert state['next_sequence']==142 and str(checkpoint/'adapter')==done['final_adapter']
    split=read(SOURCE/'assets/split.json');ids=split[a.cohort]
    assert not set(split['train']) & set(split['test'])
    histories=[prepare_locomo(SOURCE/'assets/locomo10_eval.json',cid) for cid in ids]
    expected={s['history_id']:s['records'] for s in sessions}
    if a.cohort=='train':assert all(h['records']==expected[h['source_id']] for h in histories)
    else:assert sum(q['eligible'] for h in histories for q in h['questions'])==729
    identity=dict(training_fingerprint=report['fingerprint'],cohort=a.cohort,
                  adapter_sha256=file_hash(checkpoint/'adapter/adapter_model.safetensors'),
                  evaluation_settings=core.SETTINGS,source_sha256=file_hash(SOURCE/'assets/locomo10_eval.json'),
                  query_blind=True,official_used_for_scoring_only=True,history_ids=ids)
    prepared=dict(fingerprint=digest(identity),identity=identity)
    core.ROOT=ROOT/'evaluation'/a.cohort
    stored=core.ROOT/'prepare.json'
    if stored.exists():assert read(stored)==prepared
    atomic(stored,prepared)
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Slurm allocation required')
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(resolve(cfg['paths']['tokenizer']),local_files_only=True)
    server=VLLMServer(cfg,core.ROOT/'server',os.environ['CUDA_VISIBLE_DEVICES'].split(',')[0])
    try:
        server.start();server.activate(checkpoint/'adapter','final')
        with ThreadPoolExecutor(max_workers=5) as pool:
            results=list(pool.map(lambda h:core.run_history(h,prepared,server.endpoint+'/v1',tok),histories))
        core.aggregate(prepared,results)
    finally:server.close()
    diagnostics=[]
    for h,result in zip(histories,results):
        cid=h['source_id'];bank=result['final_memory']
        limited=[]
        for entry in bank:
            candidate=limited+[entry]
            if len(tok.encode('\n\n'.join(e['content'] for e in candidate),add_special_tokens=False))>10000:break
            limited=candidate
        row=dict(conversation=cid,native_chars=result['content_chars'],native_tokens=result['content_tokens'])
        official=[dict(question_id=q['question_id'],sufficient_sets=[q['evidence_ids']]) for q in h['questions'] if q['eligible']]
        for name,b in [('native43500',bank),('body10k',limited)]:
            row[name]={'official':qg_result(b,official)}
            if a.cohort=='train':
                qs=next(s for s in reversed(sessions) if s['history_id']==cid)['end_questions']['questions']
                row[name]['original_qg']=qg_result(b,qs)
        diagnostics.append(row)
    atomic(core.ROOT/'DIAGNOSTICS.json',dict(cohort=a.cohort,per_conversation=diagnostics))
    atomic(core.ROOT/'COMPLETE.json',dict(status='complete',fingerprint=report['fingerprint'],cohort=a.cohort,
                                        evaluation_fingerprint=prepared['fingerprint'],conversations=5))

if __name__=='__main__':main()

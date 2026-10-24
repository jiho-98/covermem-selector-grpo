"""Durable machine-readable comparison and update diagnostics after evaluation."""
import csv
from collections import defaultdict
from protocol import ROOT,SOURCE,read,atomic
from common import retained_ids
from rewards import alternatives

def pooled(rows,view,pool):
    values=[r[view][pool] for r in rows];n=sum(v['questions'] for v in values)
    c=sum(v['complete_count'] for v in values);rec=sum(v['recall']*v['questions'] for v in values)/n
    return dict(questions=n,complete_count=c,complete=c/n,recall=rec,reward=.7*c/n+.3*rec)

fp=read(ROOT/'PREFLIGHT.json')['fingerprint']
assert read(ROOT/'evaluation/COMPLETE.json')['fingerprint']==fp
run=ROOT/'runs/future30';done=read(run/'TRAIN_COMPLETE.json')
summary=dict(status='complete',fingerprint=fp,training=done,evaluation={},limitations=[
    'One training seed; historical best LOO is not a newly randomized matched control.',
    'Lookahead is a sampled continuation of different path banks, not causal attribution of one action.',
    'QG and official training-conversation metrics are fitting diagnostics, not held-out generalization.',
    'Token logprob changes measure the recorded output, not the probability of an action category.'])
for cohort in ('train','test'):
    rows=read(ROOT/'evaluation'/cohort/'DIAGNOSTICS.json')['per_conversation']
    summary['evaluation'][cohort]={v:{p:pooled(rows,v,p) for p in rows[0][v]} for v in ('native43500','body10k')}
old_train=read(ROOT/'references/train_baselines.json')
summary['historical_train_baselines']={m:old_train['models'][m]['pooled']['native43500'] for m in ('sft5','best_loo')}
summary['historical_test_baselines']={
    'sft5':read(SOURCE/'assets/sft_eval_summary.json')['pooled_question_mean'],
    'best_loo':read(SOURCE/'runs/loo_v1/evaluation/runs/EVAL_COMPLETE.json')['pooled_question_mean']}
groups=read(SOURCE/'assets/questions.json')['groups']
baseline_banks=read(ROOT/'references/train_banks.json');paired=[]
for cid in baseline_banks['sft5']:
    questions=[dict(q,question_id=g['key']+'::'+q['question_id']) for g in groups.values() if g['history_id']==cid for q in g['questions']]
    new=read(ROOT/'evaluation/train/runs/histories'/cid/'bank.json')['final_memory']
    def complete(bank):
        ids=retained_ids(bank)
        return {q['question_id'] for q in questions if any(e<=ids for e in alternatives(q['sufficient_sets']))}
    after=complete(new)
    for model in baseline_banks:
        before=complete(baseline_banks[model][cid]['bank'])
        paired.append(dict(conversation=cid,baseline=model,before=len(before),after=len(after),
                           gained=sorted(after-before),lost=sorted(before-after),net=len(after)-len(before)))
summary['paired_original_qg_vs_baselines']=paired
updates=[];actions=defaultdict(lambda:defaultdict(float))
for path in sorted((run/'sessions').glob('*/metrics.json')):
    m=read(path);directory=path.parent
    rows=read(directory/'credit_assignments.json')['outputs']
    audit=read(directory/'action_question_audit.json')['outputs']
    probs={r['request_key']:r for f in directory.glob('probability_change_rank*.json') for r in read(f)['outputs']}
    did=m['policy_version_after']>m['policy_version_before']
    if did:assert len(probs)==len(rows)
    by_key={r['request_key']:r for r in rows}
    for a in audit:
        r=by_key[a['key']];name=a['action']['action'] if a['action'] else 'UNPARSEABLE'
        group=actions[(m['history'],'above90' if r['future_active'] else 'original_credit',name)]
        group['outputs']+=1;group['valid']+=a['valid'];group['sign_changes']+=r['sign_changed_vs_original']
        group['complete_gains']+=len(a['active_end']['gained']);group['complete_losses']+=len(a['active_end']['lost'])
        if a['key'] in probs:
            group['negative_advantage']+=r['advantage']<0
            group['negative_advantage_logprob_increased']+=r['advantage']<0 and probs[a['key']]['mean_logprob_delta']>0
    updates.append(dict(sequence=m['sequence'],conversation=m['history'],session=m['session'],
        policy_before=m['policy_version_before'],policy_after=m['policy_version_after'],
        future_outputs=sum(r['future_active'] for r in rows),outputs=len(rows),
        advantage_sign_changes=sum(r['sign_changed_vs_original'] for r in rows),
        mean_end_reward=sum(m['end_rewards'])/4,mean_end_complete=sum(m['end_complete'])/4,
        mean_bank_chars=sum(m['storage_chars'])/4,rollout_seconds=m['rollout_and_score_seconds'],
        optimization_seconds=m['optimization_and_sync_seconds'],
        adapter_delta_l2=m['optimization'].get('adapter_delta_l2',0)))
assert len(updates)==142
out=ROOT/'analysis';out.mkdir(exist_ok=True)
with (out/'updates.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=list(updates[0]));w.writeheader();w.writerows(updates)
summary['actions']=[dict(conversation=k[0],credit_region=k[1],action=k[2],**v) for k,v in actions.items()]
summary['checkpoint_count']=len(list((run/'checkpoints').glob('session_*/COMMITTED.json')))
summary['checkpoint_retention']=read(run/'checkpoint_retention.json')
for directory in summary['checkpoint_retention']:
    assert (run/'checkpoints'/directory/'COMMITTED.json').exists()
assert (run/'checkpoints/session_000142/COMMITTED.json').exists()
atomic(out/'SUMMARY.json',summary);atomic(ROOT/'COMPLETE.json',summary)
print(summary['evaluation'])

"""CPU regression against all original LOO sessions; no new trajectories."""
import gzip,json
from copy import deepcopy
from itertools import groupby
from protocol import ROOT,SOURCE,read,atomic,config,digest
from cumulative_questions import accumulate_questions
from action_credit import attach_action_credit
from future_credit import attach,gates

histories=read(SOURCE/'assets/histories.json')['histories']
qg=read(SOURCE/'assets/questions.json')['groups'];sessions=[]
for h in histories:
    cursor=0
    for sid,rs in groupby(h['records'],key=lambda r:r['session_id']):
        rs=list(rs);mid=cursor+(len(rs)+1)//2;end=cursor+len(rs)
        s=dict(history_id=h['source_id'],session_id=sid,start=cursor,mid=mid,end=end,records=h['records'])
        for half,field in enumerate(('mid_questions','end_questions')):
            key=f"{h['source_id']}_{sid}_h{half}";qs=deepcopy(qg[key]['questions'])
            s[field]=dict(key=key,questions=qs,sha256=digest(qs))
        sessions.append(s);cursor=end
accumulate_questions(sessions)
snapshot=SOURCE.parent/'diagnostics/loo_result_hypotheses_20261010/loo_snapshot.json.gz'
saved=sorted(json.load(gzip.open(snapshot,'rt'))['sessions'],key=lambda s:s['metrics']['sequence'])
assert len(saved)==len(sessions)==142
initial=[[],[],[],[]];current=None;outputs=changed=active_sessions=0;retention={}
for i,(session,old) in enumerate(zip(sessions,saved)):
    cid=session['history_id']
    assert old['metrics']['history']==cid and old['metrics']['session']==session['session_id']
    if current!=cid:initial=[[],[],[],[]];current=cid
    streams=sorted(old['streams'],key=lambda s:s['stream'])
    cfg=config()['reward']
    before,_=attach_action_credit(streams,session,initial,cfg)
    expected={(r['stream'],r['turn_index'],r['attempt']):r for r in old['credit_assignments']['outputs']}
    for r in before:
        target=expected[r['stream'],r['turn_index'],r['attempt']]
        assert abs(r['advantage']-target['advantage'])<1e-10
    nxt=sessions[i+1] if i+1<142 and sessions[i+1]['history_id']==cid else None
    enabled=nxt is not None and any(p>=.9 for p in gates(before).values())
    future={s['stream']:s for s in saved[i+1]['streams']} if enabled else None
    # These historical future banks were sampled after a historical update.
    # They test arithmetic/replay ONLY, not the proposed frozen-policy rollout.
    rows,stats=attach(streams,session,initial,cfg,future,nxt)
    for r in rows:
        if not r['future_active']:assert abs(r['advantage']-r['original_advantage'])<1e-10
    outputs+=len(rows);changed+=stats['sign_changes'];active_sessions+=enabled
    if enabled:retention.setdefault(cid,[]).append(i+1)
    initial=[s['end_bank'] for s in streams]
report=dict(status='passed',sessions=142,outputs=outputs,original_advantages_match_saved=True,
            unchanged_outside_gate=True,historical_gate_sessions=active_sessions,
            gated_session_sequences=retention,arithmetic_only_sign_changes=changed,
            future_policy_not_frozen_in_historical_data=True,training=False)
atomic(ROOT/'validation/REPLAY.json',report);print(report)

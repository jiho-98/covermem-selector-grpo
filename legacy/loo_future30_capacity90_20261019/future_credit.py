"""CPU replay and explicit decomposition of original and future LOO credit."""
from copy import deepcopy
import statistics
from protocol import SETTINGS
from common import apply_action, digest, retained_ids, storage_chars
from action_credit import attach_action_credit, relative_coefficients, PoolScore
from rewards import alternatives

def gates(rows, budget=43500):
    first={(r['turn_index'],r['stream']):r for r in rows if r['attempt']==0}
    return {ti: statistics.mean(first[ti,g]['before_chars'] for g in range(4))/budget
            for ti in sorted({r['turn_index'] for r in rows})}

def apply_future(rows, session, initial, future, next_session, cfg):
    pressure=gates(rows)
    if future is not None:
        assert next_session['history_id']==session['history_id']
        assert next_session['start']==session['end']
        assert set(future)==set(range(4))
        scorer=PoolScore(next_session['end_questions']['questions'],next_session['end'],
                         cfg['complete'],cfg['recall'])
        terminals={g:scorer(future[g]['end_bank']) for g in range(4)}
    banks=deepcopy(initial)
    for row in rows:
        g=row['stream'];bank=banks[g]
        assert digest(bank)==row['before_hash']
        old=row['raw_rtg']; weight=cfg['end_weight']+(cfg['mid_weight'] if row['segment']=='front' else 0.)
        active=future is not None and pressure[row['turn_index']]>=SETTINGS['capacity_fraction']
        pre=scorer(bank) if future is not None else None
        next_gain=weight*(terminals[g]-pre) if future is not None else None
        row.update(original_raw_rtg=old, original_advantage=row['advantage'],
                   original_peer_mean_rtg=row['peer_mean_rtg'],original_residual=row['residual'],
                   capacity_group_fraction=pressure[row['turn_index']],future_active=active,
                   future_weight=SETTINGS['future_weight'] if active else 0.,
                   checkpoint_weight_sum=weight, future_pre_bank_score=pre,
                   future_terminal_score=terminals[g] if future is not None else None,
                   future_raw_rtg=next_gain,
                   raw_rtg=SETTINGS['current_weight']*old+SETTINGS['future_weight']*next_gain if active else old)
        if row['valid']:banks[g]=apply_action(bank,session['records'][row['turn_index']],row['action'],43500)
        assert digest(banks[g])==row['after_hash']
    relative_coefficients(rows,cfg)
    for r in rows:
        r['advantage_delta_vs_original']=r['advantage']-r['original_advantage']
        r['sign_changed_vs_original']=r['advantage']*r['original_advantage']<0
        if not r['future_active']:
            assert abs(r['advantage']-r['original_advantage'])<1e-12
    return rows

def attach(streams,session,initial,cfg,future=None,next_session=None):
    rows,old=attach_action_credit(streams,session,initial,cfg)
    rows=apply_future(rows,session,initial,future,next_session,cfg)
    return rows,dict(original=old,has_signal=any(abs(r['advantage'])>1e-12 for r in rows),
                     outputs=len(rows),future_outputs=sum(r['future_active'] for r in rows),
                     sign_changes=sum(r['sign_changed_vs_original'] for r in rows),
                     terminal_fallback=next_session is None,
                     future_question_count=len(next_session['end_questions']['questions']) if future else 0)

def action_audit(rows,session,initial,full_questions):
    """Read-only immediate question gains/losses; not used to assign rewards."""
    pools={'active_end':session['end_questions']['questions'],'full_training_qg_diagnostic':full_questions}
    masks={name:[(q['question_id'],alternatives(q['sufficient_sets'])) for q in qs]
           for name,qs in pools.items()}
    def completed(bank,name):
        ids=retained_ids(bank)
        return {qid for qid,sets in masks[name] if any(s<=ids for s in sets)}
    banks=deepcopy(initial); out=[]
    for r in rows:
        g=r['stream'];before=banks[g];assert digest(before)==r['before_hash']
        after=apply_action(before,session['records'][r['turn_index']],r['action'],43500) if r['valid'] else before
        a=dict(key=r['request_key'],stream=g,turn=r['turn_index'],attempt=r['attempt'],valid=r['valid'],
               action=r['action'],before_hash=r['before_hash'],after_hash=r['after_hash'],
               added_ids=sorted(retained_ids(after)-retained_ids(before)),
               removed_ids=sorted(retained_ids(before)-retained_ids(after)),before_chars=storage_chars(before),
               after_chars=storage_chars(after),advantage=r['advantage'],original_advantage=r['original_advantage'])
        for name in masks:
            b=completed(before,name);e=completed(after,name)
            a[name]=dict(gained=sorted(e-b),lost=sorted(b-e),before_complete=len(b),after_complete=len(e))
        out.append(a);banks[g]=after
    return out

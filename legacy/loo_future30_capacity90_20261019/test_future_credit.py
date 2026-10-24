import unittest
from copy import deepcopy
from pathlib import Path
import tempfile
from protocol import config, digest, read
from common import apply_action
from action_credit import attach_action_credit, PoolScore
from future_credit import attach, gates
from checkpoint_all import publish, latest

def fixture(chars=39200):
    records=[dict(turn_id=f't{i:06d}',content='x'*chars if i==0 else str(i),session_id='s0' if i<3 else 's1',speaker='fixture',timestamp='2026-01-01') for i in range(5)]
    bank=[dict(memory_id='m_t000000',content=records[0]['content'],source_turn_ids=['t000000'])]
    def q(i,ids):return dict(question_id=f'q{i}',sufficient_sets=[ids])
    qs=[q(1,['t000001']),q(2,['t000002'])]
    s=dict(history_id='fixture',session_id='s0',start=1,mid=2,end=3,records=records,
           mid_questions={'questions':qs[:1]},end_questions={'questions':qs})
    nxt=dict(history_id='fixture',session_id='s1',start=3,mid=4,end=5,records=records,
             end_questions={'questions':qs+[q(3,['t000001','t000003'])]})
    cfg=config()['reward'];initial=[deepcopy(bank) for _ in range(4)];streams=[];future={}
    for g in range(4):
        b=deepcopy(bank);rows=[];mid=None
        for ti in (1,2):
            yes=(g==0 and ti==1) or (g==1 and ti==2) or g==3
            action=dict(action='ADD' if yes else 'SKIP',thought='fixture',target_id=None,
                        source_turn_ids=[records[ti]['turn_id']] if yes else [])
            # Use the actual executor schema, no fake transitions.
            before=digest(b)
            b=apply_action(b,records[ti],action,43500)
            rows.append(dict(stream=g,turn_index=ti,attempt=0,valid=True,action=action,
                             before_hash=before,after_hash=digest(b),decision_end=True,response_ids=[1]))
            if ti==1:mid=deepcopy(b)
        stream=dict(stream=g,initial_bank_hash=digest(bank),mid_bank=mid,end_bank=deepcopy(b),attempts=rows)
        for key in ('mid','end'):
            scorer=PoolScore(s[key+'_questions']['questions'],s[key],.7,.3)
            stream[key+'_score']={'reward':scorer(stream[key+'_bank'])}
        streams.append(stream)
        fb=apply_action(b,records[3],dict(action='ADD',thought='fixture',target_id=None,source_turn_ids=['t000003']),43500)
        future[g]={'end_bank':fb}
    return streams,s,initial,cfg,future,nxt

class Tests(unittest.TestCase):
    def test_low_capacity_identical(self):
        streams,s,initial,cfg,future,nxt=fixture(100)
        expected,_=attach_action_credit(streams,s,initial,cfg)
        got,_=attach(streams,s,initial,cfg,future,nxt)
        self.assertEqual([r['advantage'] for r in expected],[r['advantage'] for r in got])
        self.assertFalse(any(r['future_active'] for r in got))

    def test_terminal_identical(self):
        streams,s,initial,cfg,_,_=fixture()
        got,stats=attach(streams,s,initial,cfg)
        self.assertTrue(stats['terminal_fallback'])
        self.assertTrue(all(r['advantage']==r['original_advantage'] for r in got))

    def test_mixture_and_loo(self):
        streams,s,initial,cfg,future,nxt=fixture()
        snapshot=digest([streams,initial,future])
        rows,_=attach(streams,s,initial,cfg,future,nxt)
        self.assertTrue(all(r['future_active'] for r in rows))
        for r in rows:
            self.assertAlmostEqual(r['raw_rtg'],.7*r['original_raw_rtg']+.3*r['future_raw_rtg'])
            self.assertEqual(r['checkpoint_weight_sum'],1. if r['segment']=='front' else .5)
            peers=[x['raw_rtg'] for x in rows if x['turn_index']==r['turn_index'] and x['stream']!=r['stream']]
            self.assertAlmostEqual(r['advantage'],(r['raw_rtg']-sum(peers)/3)/cfg['credit_assignment']['fixed_scale'])
        self.assertEqual(digest([streams,initial,future]),snapshot)

    def test_gate_shared_across_paths(self):
        streams,s,initial,cfg,_,_=fixture()
        rows,_=attach_action_credit(streams,s,initial,cfg)
        rows[0]['before_chars']=0
        p=gates(rows)
        self.assertLess(p[1],.9)
        self.assertGreaterEqual(p[2],.9)

    def test_future_question_guard(self):
        streams,s,initial,cfg,future,nxt=fixture()
        nxt['end_questions']['questions'].append({'question_id':'illegal','sufficient_sets':[['t000005']]})
        with self.assertRaises(ValueError):attach(streams,s,initial,cfg,future,nxt)

    def test_conversation_boundary_guard(self):
        streams,s,initial,cfg,future,nxt=fixture();nxt['history_id']='other'
        with self.assertRaises(AssertionError):attach(streams,s,initial,cfg,future,nxt)

    def test_selective_checkpoint_retention(self):
        with tempfile.TemporaryDirectory() as tmp:
            def payload(p):(p/'payload').write_text('committed')
            for i in (1,2,3,4,5):
                state=dict(next_sequence=i,history_id='fixture',retain_previous_checkpoint=i==3,
                           retain_this_checkpoint=i==3)
                publish(Path(tmp),i,'fixture',state,payload)
            self.assertFalse((Path(tmp)/'checkpoints/session_000001').exists())
            self.assertTrue((Path(tmp)/'checkpoints/session_000002/COMMITTED.json').exists())
            self.assertTrue((Path(tmp)/'checkpoints/session_000003/COMMITTED.json').exists())
            self.assertFalse((Path(tmp)/'checkpoints/session_000004').exists())
            self.assertEqual(latest(Path(tmp),'fixture')[1]['next_sequence'],5)

if __name__=='__main__':unittest.main()

import json
from pathlib import Path
import tempfile
import unittest
from fixtures import CharTokenizer, config
from cmselect.evaluate import run_history

class EvalBackend:
    tokenizer=CharTokenizer()
    def __init__(self):self.calls=[]
    def sample(self,adapter,messages,seed,max_new_tokens,greedy=False):
        self.calls.append(adapter)
        assert greedy
        obs=json.loads(messages[-1]['content'])
        assert 'questions' not in obs and 'evidence_ids' not in obs
        if adapter=='selector':
            obj={'target_id':obs['eligible_target_ids'][-1]}
        else:
            turn=obs['current_turn'];bank=obs['current_active']
            name='REPLACE' if bank else 'ADD'
            obj={'thought':'replace','action':name,'target_id':bank[0]['memory_id'] if bank else None,
                 'source_turn_ids':[turn['turn_id']]}
        return {'text':json.dumps(obj),'finish_reason':'stop'}

class EvaluationTests(unittest.TestCase):
    def test_one_selector_call_no_oracle_and_resume_without_replaying(self):
        cfg=config()
        history={'source_id':'fixture','records':[
            {'turn_id':f't{i:06d}','session_id':'s0','speaker':'A','timestamp':'2026','content':str(i)} for i in range(2)],
            'questions':[{'question_id':'q','eligible':True,'evidence_ids':['t000001']}]}
        with tempfile.TemporaryDirectory() as tmp:
            b=EvalBackend();r=run_history(history,b,cfg,Path(tmp),'test',True)
            self.assertEqual(b.calls,['manager','manager','selector'])
            self.assertEqual(r['metrics']['complete'],1)
            b2=EvalBackend();r2=run_history(history,b2,cfg,Path(tmp),'test',True)
            self.assertEqual(b2.calls,[]);self.assertEqual(r,r2)
            with self.assertRaises(ValueError):run_history(history,b2,cfg,Path(tmp),'other',True)

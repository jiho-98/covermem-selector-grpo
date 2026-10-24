from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from fixtures import config, data, FakeBackend, CharTokenizer
from cmselect.selection import advantages, candidates, finish_group, branch_scores, messages
from cmselect.core import digest, read
from cmselect.rollout import collect_stream
from cmselect.train import train_loop
from cmselect import checkpoint
from cmselect.masking import action_value_mask

class ProtocolTests(unittest.TestCase):
    def test_four_local_groups_not_one_group(self):
        cfg = config()['selector']['advantage']
        rewards = [[0.,1.,2.], [100.,101.,102.], [-2.,-1.,0.], [5.,6.,7.]]
        local = [advantages(r, cfg) for r in rewards]
        for group in local:
            self.assertAlmostEqual(sum(group), 0)
            self.assertEqual(group, local[0])
        self.assertNotEqual(local[0], advantages(sum(rewards, []), cfg)[:3])
        self.assertEqual(advantages([1.,1.,1.], cfg), [0.,0.,0.])

    def setup_bank(self):
        sessions, pools = data()
        records = sessions[0]['records']
        bank = [{'memory_id': 'm_'+r['turn_id'], 'content': r['content'], 'source_turn_ids': [r['turn_id']]}
                for r in records[:2]]
        action = {'thought': 'replace', 'action': 'REPLACE', 'target_id': bank[0]['memory_id'],
                  'source_turn_ids': [records[2]['turn_id']]}
        return bank, records[2], action, pools['fixture']

    def test_full_pool_reward_and_no_oracle_execution(self):
        bank, turn, action, _ = self.setup_bank()
        questions = [{'question_id':'a','type':'direct_recall','sufficient_sets':['t000000'], 'available_at':1},
                     {'question_id':'b','type':'direct_recall','sufficient_sets':['t000002'], 'available_at':4}]
        cfg = config()['selector']
        eligible = candidates(bank, turn, action, 8)
        scored, audit = branch_scores(bank, eligible, questions)
        self.assertEqual(audit['all_questions'], 2)
        self.assertEqual(scored['m_t000000']['reward'], 0.)
        self.assertEqual(scored['m_t000001']['reward'], .5)
        def sample(t):
            return {'text':json.dumps({'target_id':t}), 'finish_reason':'stop', 'response_ids':[1], 'old_log_probs':[-1.]}
        samples = [sample('m_t000000'), sample('m_t000001')]
        executed, new, group = finish_group(samples, bank, turn, action, eligible, questions, cfg, 'g')
        self.assertLess(group['rows'][0]['reward'], group['rows'][1]['reward'])
        self.assertEqual(executed['target_id'], 'm_t000000')
        self.assertEqual(bank[0]['source_turn_ids'], ['t000000'])
        prompt = json.loads(messages(bank, turn, 8, eligible)[1]['content'])
        self.assertEqual(set(prompt), {'storage_budget_chars','storage_used_chars','current_active','current_turn','eligible_target_ids'})

    def test_all_questions_equal_weight_net_gain_and_activation_invariance(self):
        bank, turn, action, _ = self.setup_bank()
        eligible = candidates(bank, turn, action, 8)
        # Losing one early question while gaining nine later questions has positive
        # full-pool net gain (+0.8), unlike the removed 7:3 temporal mixture (-0.4).
        questions = [{'question_id':'old','type':'direct_recall','sufficient_sets':['t000000'], 'available_at':1}]
        questions += [{'question_id':f'new{i}','type':'direct_recall','sufficient_sets':['t000002'],
                       'available_at':100+i} for i in range(9)]
        scored, audit = branch_scores(bank, eligible, questions)
        value = scored['m_t000000']
        self.assertEqual(audit['all_questions'], 10)
        self.assertAlmostEqual(audit['before_all'], .1)
        self.assertAlmostEqual(value['after_all'], .9)
        self.assertAlmostEqual(value['reward'], .8)
        self.assertEqual((value['gained_complete'], value['lost_complete'], value['net_complete']), (9,1,8))
        self.assertAlmostEqual(value['utility_gain'] - value['utility_loss'], value['reward'])
        # Activation times are metadata only for selector reward, including missing timestamps.
        for q in questions:q.pop('available_at')
        self.assertEqual((scored, audit), branch_scores(bank, eligible, questions))

    def test_partial_evidence_gain_enters_full_pool_net_utility(self):
        bank, turn, action, _ = self.setup_bank()
        eligible = candidates(bank, turn, action, 8)
        questions = [{'question_id':'old','type':'direct_recall','sufficient_sets':['t000000']},
                     {'question_id':'partial','type':'cross_session_relation','sufficient_sets':[['t000002','t000004']]}]
        scored, _ = branch_scores(bank, eligible, questions)
        self.assertAlmostEqual(scored['m_t000000']['reward'], -.425)
        self.assertEqual(scored['m_t000000']['net_complete'], -1)
        self.assertAlmostEqual(scored['m_t000000']['utility_gain'], .075)
        self.assertAlmostEqual(scored['m_t000000']['utility_loss'], .5)

    def test_invalid_fallback_duplicate_samples_and_empty_pools(self):
        bank, turn, action, _ = self.setup_bank()
        eligible = candidates(bank, turn, action, 8)
        samples = [{'text':'{"target_id":"absent"}', 'finish_reason':'stop', 'response_ids':[1]},
                   {'text':'{"target_id":"m_t000001"}', 'finish_reason':'stop', 'response_ids':[1]},
                   {'text':'{"target_id":"m_t000001"}', 'finish_reason':'stop', 'response_ids':[1]}]
        executed, _, g = finish_group(samples, bank, turn, action, eligible, [], config()['selector'], 'g')
        self.assertTrue(g['fallback'])
        self.assertEqual(executed['target_id'], action['target_id'])
        self.assertEqual(g['distinct_sampled_targets'], 1)
        self.assertEqual(g['rows'][1]['advantage'], g['rows'][2]['advantage'])
        self.assertLess(g['rows'][0]['advantage'], 0)
        self.assertIsNone(g['rows'][0]['delta_all'])
        self.assertEqual(g['rows'][1]['delta_all'], 0)

    def test_mask_never_rewrites_sample_or_trains_thought_target_suffix(self):
        text = '{"thought":"drop m_t000001 (\\\"action\\\":\\\"ADD\\\")","action":"REPLACE","target_id":"m_t000001","source_turn_ids":["t000003"]}'
        row = {'text':text, 'response_ids':list(map(ord,text))}
        mask = action_value_mask(CharTokenizer(), row)
        self.assertEqual(''.join(c for c,m in zip(text,mask) if m), '"REPLACE"')
        self.assertEqual(row['text'], text)

    def test_add_skip_execute_without_selector_and_manager_response_preserved(self):
        cfg = config()
        sessions, pools = data()
        backend = FakeBackend()
        bank = []
        first = collect_stream(sessions[0],0,bank,backend,cfg,pools['fixture'],0)
        self.assertEqual(len(first['end_bank']),2)
        self.assertNotIn('selector',backend.calls)
        second = collect_stream(sessions[1],0,first['end_bank'],backend,cfg,pools['fixture'],1)
        self.assertEqual(len(second['selector_groups']),1)
        self.assertEqual(len(second['selector_groups'][0]['rows']),10)
        self.assertEqual(backend.calls.count('selector'),10)
        for row in second['attempts']:
            self.assertEqual(json.loads(row['text']),row['sampled_action'])
        self.assertEqual(second['attempts'][-1]['action']['action'],'SKIP')

    def test_A_B_and_checkpoint_resume_match(self):
        sessions, pools = data()
        for variant in ('A','B','vanilla'):
            cfg=config(variant)
            with tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp); full=FakeBackend()
                expected=train_loop(cfg,{'fingerprint':'fixture'},sessions,pools,full,root/'full')
                one=FakeBackend()
                train_loop(cfg,{'fingerprint':'fixture'},sessions,pools,one,root/'resume',stop_after_batches=2)
                self.assertFalse((root/'resume/TRAIN_COMPLETE.json').exists())
                # Simulate a crash after checkpoint commit but before trace archival.
                import shutil
                shutil.rmtree(root/'resume/batches/batch_000001')
                saved=checkpoint.latest(root/'resume','fixture')
                state=train_loop(cfg,{'fingerprint':'fixture'},sessions,pools,FakeBackend(),root/'resume',saved[1])
                self.assertTrue((root/'resume/batches/batch_000001/traces.json').exists())
                for name in ('banks','session_cursors','updates','next_batch'):
                    self.assertEqual(expected[name],state[name])
                adapters={a for a,_ in full.update_calls}
                self.assertEqual(adapters, {'selector'} if variant=='A' else {'manager'} if variant=='vanilla' else {'manager','selector'})
                if variant in ('A','B'):
                    self.assertGreater(state['updates']['selector'],0)
                if variant=='B':
                    self.assertGreater(state['updates']['manager'],0)
                with self.assertRaises(ValueError):checkpoint.latest(root/'resume','changed-science')
                (checkpoint.latest(root/'resume','fixture')[0]/'fixture.json').write_text('corrupt')
                with self.assertRaises(ValueError):checkpoint.latest(root/'resume','fixture')

    def test_original_LOO_credit_preserves_own_peer_exclusion(self):
        from cmselect.action_credit import relative_coefficients
        cfg=config()['manager_reward']
        rows=[{'turn_index':0,'stream':i,'attempt':0,'raw_rtg':float(i),'valid':True} for i in range(4)]
        got=relative_coefficients(rows,cfg)
        scale=cfg['credit_assignment']['fixed_scale']
        self.assertAlmostEqual(got[0]['advantage'],-2/scale)
        self.assertAlmostEqual(got[3]['advantage'],2/scale)

    def test_alternative_evidence_sets_and_combined_raw_entry(self):
        from cmselect.rewards import score
        q=[{'question_id':'q','type':'direct_recall','sufficient_sets':[['t000000','t000001'],'t000002']}]
        bank=[{'memory_id':'m_t000000','content':'AAAA\nBBBB','source_turn_ids':['t000000','t000001']}]
        self.assertEqual(score(bank,q)['reward'],1.)
        bank[0]['source_turn_ids']=['t000000']
        self.assertAlmostEqual(score(bank,q)['reward'],.15)

    def test_batch5_preserves_conversation_separation_and_chronology(self):
        from cmselect.data import make_batches
        sessions,_=data()
        other=deepcopy(sessions[:2])
        for s in other:s['history_id']='other'
        batches=make_batches(sessions+other,5)
        self.assertEqual([[s['history_id'] for s in b] for b in batches],[['fixture','other'],['fixture','other'],['fixture']])

if __name__ == '__main__':unittest.main()

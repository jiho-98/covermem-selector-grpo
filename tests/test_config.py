from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from cmselect.core import ROOT, read
from cmselect.config import load

class ConfigTests(unittest.TestCase):
    def test_all_shipped_configs_and_reject_silent_policy_warping(self):
        for name in ('A_loo','A_h1','B_joint','vanilla_control'):
            cfg=load(ROOT/'configs'/f'{name}.json')
            self.assertEqual(cfg['environment']['streams'],4)
            self.assertEqual(cfg['selector']['samples_per_replace'],10)
            self.assertEqual(cfg['selector']['question_pool'],'all_generated_questions_same_conversation')
            self.assertEqual(cfg['selector']['score_mode'],'full_pool_bank_delta')
            self.assertNotIn('current_weight',cfg['selector'])
            self.assertNotIn('future_horizon',cfg['selector'])
        with tempfile.TemporaryDirectory() as tmp:
            cfg=read(ROOT/'configs/B_joint.json');cfg['rollout']['temperature']=.7
            path=Path(tmp)/'bad.json';path.write_text(json.dumps(cfg))
            with self.assertRaises(ValueError):load(path)

    def test_reject_legacy_temporal_mixture_and_wrong_K(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'bad.json'
            changes=[('schema','covermem.selector_grpo.v1'),
                     ('current_weight',.7),('future_horizon','remaining_same_conversation'),
                     ('samples_per_replace',4),('samples_per_replace',16)]
            for key,value in changes:
                cfg=read(ROOT/'configs/B_joint.json')
                if key=='schema':cfg[key]=value
                else:cfg['selector'][key]=value
                path.write_text(json.dumps(cfg))
                with self.subTest(key=key,value=value), self.assertRaises(ValueError):load(path)

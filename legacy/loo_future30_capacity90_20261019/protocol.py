"""Original LOO + capacity-gated, fixed-policy next-session lookahead."""
from pathlib import Path
from copy import deepcopy
import sys

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent / 'persistent_grpo_verl_loo_v1'
sys.path.insert(0, str(SOURCE))
from common import atomic, read, digest, file_hash, config, resolve

SETTINGS = dict(
    capacity_fraction=.9, capacity_gate='mean first-attempt PRE-action chars across four paths',
    current_weight=.7, future_weight=.3, future_endpoint='next_session_end',
    preserve_front_back_total_weight=True, last_session='original_credit_only',
    pool='original_3338_cumulative', question_activation='original cutoff_turn_count',
    training_conversations=['conv-43','conv-44','conv-48','conv-49','conv-50'],
    lookahead='copy current session end banks; frozen current policy; discard future banks after scoring',
    update='only current session outputs, once per nonzero-signal session',
    actual_next_session='regenerate from original current end banks under updated policy',
    group_gate_shared=True, official_questions_in_training=False,
    new_qg_calls=0, checkpoint_retention='each conversation: before first future credit, subsequent actual updates, conversation end; always latest resume',
)

def prepare_run():
    from prepare import prepare
    cfg = config()
    base, sessions = prepare(cfg)
    expected = '72239152e605132a839fcb50a84a7e6cd5d77f39f53f4a54c76c844e833aa656'
    assert file_hash(SOURCE/'assets/questions.json') == expected
    assert len(sessions) == 142 and base['questions'] == 3338
    cfg = deepcopy(cfg)
    cfg['paths']['output'] = str(ROOT/'runs/future30')
    science = dict(settings=SETTINGS, base_fingerprint=base['fingerprint'], config=cfg,
                   original_qg_sha256=expected,
                   code={p.name:file_hash(p) for p in ROOT.glob('*.py')},
                   launch_files={n:file_hash(ROOT/n) for n in ('job_entry.sh','submit.sbatch','smoke_config.json')})
    report = dict(base, config=cfg, experiment=science, fingerprint=digest(science),
                  parent_fingerprint=base['fingerprint'])
    # Primary rollout seeds are unchanged. The independent future simulation has
    # its own namespace and cannot be replayed as the actual next session.
    return cfg, report, sessions

def save_preflight():
    cfg, report, sessions = prepare_run()
    path = ROOT/'PREFLIGHT.json'
    if path.exists():
        assert read(path)['fingerprint'] == report['fingerprint'], 'Frozen source/config changed'
    atomic(path, dict(status='complete', fingerprint=report['fingerprint'],
                      report=report, sessions=len(sessions), config=cfg))
    return cfg, report, sessions

if __name__ == '__main__':
    _,r,s=save_preflight()
    print(dict(status='prepared', sessions=len(s), questions=3338, fingerprint=r['fingerprint']))

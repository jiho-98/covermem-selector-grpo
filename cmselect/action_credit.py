"""Pre-action RAW-bank baseline + leave-one-trajectory-out continuation gain.

Only credit assignment changes. Pools are the already validated cumulative
training QG at each checkpoint. No extra model calls or counterfactual actions.
"""
from copy import deepcopy
import math
import statistics
from .core import apply_action, digest, retained_ids, storage_chars
from .rewards import alternatives

ESTIMATOR = 'pre_action_bank_loo_continuation_gain_v1'


def mask(ids):
    result = 0
    for tid in ids:
        result |= 1 << int(tid[1:])
    return result


class PoolScore:
    """Exact DNF coverage with per-pool ID-set cache; no text/semantic shortcut."""
    def __init__(self, questions, cutoff, complete, recall):
        if not questions:
            raise ValueError('Empty checkpoint pool')
        self.branches = []
        for q in questions:
            bs = alternatives(q['sufficient_sets'])
            if any(int(t[1:]) >= cutoff for b in bs for t in b):
                raise ValueError('Future evidence in checkpoint question pool')
            self.branches.append([(mask(b), len(b)) for b in bs])
        self.weights = (complete, recall)
        self.cache = {}

    def __call__(self, bank):
        ids = mask(retained_ids(bank))
        if ids not in self.cache:
            total = []
            for bs in self.branches:
                c = float(any(ids & b == b for b, _ in bs))
                r = max((ids & b).bit_count()/n for b, n in bs)
                total.append(self.weights[0]*c+self.weights[1]*r)
            self.cache[ids] = statistics.mean(total)
        return self.cache[ids]


def relative_coefficients(rows, cfg):
    credit = cfg['credit_assignment']
    if credit['estimator'] != ESTIMATOR:
        raise ValueError('Unknown estimator')
    scale = credit['fixed_scale']
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError('Invalid frozen calibration scale')
    first = {}
    for row in rows:
        if row['attempt'] == 0:
            key = (row['turn_index'], row['stream'])
            if key in first:
                raise ValueError('Repeated first attempt')
            first[key] = row
    turns = {r['turn_index'] for r in rows}
    if set(first) != {(t, i) for t in turns for i in range(4)}:
        raise ValueError('Missing four-path original-turn alignment')
    for row in rows:
        peers = [first[row['turn_index'], i] for i in range(4) if i != row['stream']]
        peer_mean = statistics.mean(r['raw_rtg'] for r in peers)
        residual = row['raw_rtg']-peer_mean
        value = residual/scale
        row.update(peer_mean_rtg=peer_mean, residual=residual, bank_advantage=value,
                   advantage=value if row['valid'] else min(value, cfg['invalid_advantage_ceiling']))
        if not math.isfinite(row['advantage']):
            raise ValueError('Nonfinite action coefficient')
    return rows


def attach_action_credit(streams, session, initial_banks, cfg):
    by_id = {s['stream']: s for s in streams}
    if set(by_id) != set(range(4)) or len(streams) != 4 or len(initial_banks) != 4:
        raise ValueError('Exactly four persistent banks required')
    scorers = {name: PoolScore(session[name+'_questions']['questions'], session[name],
                              cfg['complete'], cfg['recall']) for name in ('mid', 'end')}
    rows = []
    for i in range(4):
        stream = by_id[i]
        bank = deepcopy(initial_banks[i])
        if digest(bank) != stream['initial_bank_hash']:
            raise ValueError('Initial bank identity mismatch')
        terminal = {name: scorers[name](stream[name+'_bank']) for name in scorers}
        for name, value in terminal.items():
            if abs(value-stream[name+'_score']['reward']) > 1e-12:
                raise ValueError('Coverage scorer differs from checkpoint score')
        expected_turn, expected_attempt = session['start'], 0
        for original in stream['attempts']:
            row = dict(original)
            ti = row['turn_index']
            if (row['stream'], ti, row['attempt']) != (i, expected_turn, expected_attempt):
                raise ValueError('Missing/out-of-order output or retry')
            if digest(bank) != row['before_hash']:
                raise ValueError('Pre-action replay mismatch')
            front = ti < session['mid']
            pre_end = scorers['end'](bank)
            pre_mid = scorers['mid'](bank) if front else None
            u = cfg['end_weight']*(terminal['end']-pre_end)
            if front:
                u += cfg['mid_weight']*(terminal['mid']-pre_mid)
            row.update(raw_rtg=u, pre_mid_score=pre_mid, pre_end_score=pre_end,
                       before_chars=storage_chars(bank), segment='front' if front else 'back')
            if row['valid']:
                bank = apply_action(bank, session['records'][ti], row['action'], cfg.get('bank_budget_chars', 43500))
            if digest(bank) != row['after_hash']:
                raise ValueError('Post-action replay mismatch')
            if row['decision_end']:
                for name in scorers:
                    if ti+1 == session[name] and digest(bank) != digest(stream[name+'_bank']):
                        raise ValueError('Checkpoint bank replay mismatch')
                expected_turn += 1
                expected_attempt = 0
            else:
                if row['valid']:
                    raise ValueError('Successful action cannot retry')
                expected_attempt += 1
            rows.append(row)
        if expected_turn != session['end'] or expected_attempt or digest(bank) != digest(stream['end_bank']):
            raise ValueError('Incomplete session replay')
    relative_coefficients(rows, cfg)
    def summarize(part):
        if not part:
            return {'outputs': 0}
        weights = [len(r['response_ids']) if 'response_ids' in r else r['response_len'] for r in part]
        total = sum(weights)
        return {'outputs': len(part), 'response_tokens': total,
                'positive': sum(r['advantage'] > 1e-12 for r in part),
                'negative': sum(r['advantage'] < -1e-12 for r in part),
                'zero': sum(abs(r['advantage']) <= 1e-12 for r in part),
                'rms': math.sqrt(sum(w*r['advantage']**2 for w,r in zip(weights,part))/total),
                'max_abs': max(abs(r['advantage']) for r in part)}
    summaries = {name: summarize([r for r in rows if pred(r)]) for name,pred in (
        ('below50', lambda r: r['before_chars'] < 21750),
        ('50to90', lambda r: 21750 <= r['before_chars'] < 39150),
        ('above90', lambda r: r['before_chars'] >= 39150),
        ('invalid', lambda r: not r['valid']))}
    return rows, {'estimator': ESTIMATOR, 'fixed_scale': cfg['credit_assignment']['fixed_scale'],
                  'group_std_normalization': False, 'has_signal': any(abs(r['advantage']) > 1e-12 for r in rows),
                  'all_outputs': summarize(rows), 'by_usage': summaries,
                  'score_cache_entries': {k: len(v.cache) for k,v in scorers.items()}}

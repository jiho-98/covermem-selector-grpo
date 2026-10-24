"""Outer OR / inner AND evidence reward. No LLM judge and no official QA access."""
from collections import Counter
import math
import statistics
import re
from .core import retained_ids

TYPES = {'direct_recall', 'cross_session_relation', 'list_aggregation',
         'temporal_state', 'constraint_preference', 'uncertainty_premise'}

def alternatives(expression):
    if not isinstance(expression, list) or not expression: raise ValueError('Empty OR expression')
    result = []
    for item in expression:
        ids = [item] if isinstance(item, str) else item
        if not isinstance(ids, list) or not ids or any(not isinstance(x, str) or not re.fullmatch(r't\d{6}', x) for x in ids):
            raise ValueError('Expected turn ID or nonempty flat AND list')
        if len(ids) != len(set(ids)): raise ValueError('Duplicate ID inside AND')
        branch = frozenset(ids)
        if branch in result: raise ValueError('Duplicate sufficient alternative')
        result.append(branch)
    if any(a < b for a in result for b in result):
        raise ValueError('Redundant sufficient alternative')
    return result

def validate_questions(questions, prefix):
    records = {r['turn_id']: r for r in prefix}
    # Identical to the accepted QG validator: dates/speakers are source metadata.
    raw = {t: f"{r['speaker']} [{r['timestamp']}]: {r['content']}" for t,r in records.items()}
    if not 1 <= len(questions) <= 12: raise ValueError('Expected 1..12 accepted questions')
    qids = set(); counts = Counter()
    for q in questions:
        if q['question_id'] in qids: raise ValueError('Duplicate question ID')
        qids.add(q['question_id']); counts[q['type']] += 1
        if q['type'] not in TYPES or counts[q['type']] > 2: raise ValueError('Question type quota')
        for key in ('question', 'answer', 'answer_rubric'):
            if not isinstance(q.get(key), str) or not q[key].strip(): raise ValueError('Missing '+key)
        ids = set().union(*alternatives(q['sufficient_sets']))
        if not ids <= raw.keys(): raise ValueError('Future or unknown evidence')
        cited = set()
        for e in q['evidence']:
            if e['turn_id'] not in raw or not e['quote'] or e['quote'] not in raw[e['turn_id']]:
                raise ValueError('Quote not present verbatim in allowed RAW')
            cited.add(e['turn_id'])
        if ids != cited: raise ValueError('Quoted IDs and sufficient IDs differ')
        if q['type'] == 'cross_session_relation' and any(
                len({records[t]['session_id'] for t in b}) < 2 for b in alternatives(q['sufficient_sets'])):
            raise ValueError('Cross-session branch lacks multiple sessions')

def score(bank, questions, weights=(.7, .3)):
    ids = retained_ids(bank); per_question = []
    if not questions: raise ValueError('Cannot silently score an empty QG group')
    for q in questions:
        branches = alternatives(q['sufficient_sets'])
        complete = float(any(e <= ids for e in branches))
        recall = max(len(e & ids) / len(e) for e in branches)
        hit = float(any(e & ids for e in branches))
        per_question.append({'question_id': q['question_id'], 'type': q['type'],
                             'complete': complete, 'recall': recall, 'hit': hit,
                             'reward': weights[0] * complete + weights[1] * recall})
    return {**{k: statistics.mean(q[k] for q in per_question)
               for k in ('complete', 'recall', 'hit', 'reward')}, 'per_question': per_question}

def group_advantages(rewards, epsilon=1e-8):
    if len(rewards) != 4 or not all(math.isfinite(r) for r in rewards):
        raise ValueError('Each persistent group must have four finite scores')
    mean = statistics.mean(rewards); std = statistics.pstdev(rewards)
    return [0.0]*4 if std < epsilon else [(r-mean)/(std+epsilon) for r in rewards]

def attach_advantages(streams, mid_index, cfg, session=None, initial_banks=None):
    if cfg.get('credit_assignment'):
        from .action_credit import attach_action_credit
        if session is None or initial_banks is None or mid_index != session['mid']:
            raise ValueError('Action credit needs exact session and pre-session banks')
        return attach_action_credit(streams, session, initial_banks, cfg)
    by_id = {s['stream']: s for s in streams}
    assert set(by_id) == set(range(4))
    a_mid = group_advantages([by_id[i]['mid_score']['reward'] for i in range(4)], cfg['epsilon'])
    a_end = group_advantages([by_id[i]['end_score']['reward'] for i in range(4)], cfg['epsilon'])
    rows = []
    for i in range(4):
        for original in by_id[i]['attempts']:
            row = dict(original)
            a = cfg['end_weight'] * a_end[i]
            if row['turn_index'] < mid_index: a += cfg['mid_weight'] * a_mid[i]
            row['bank_advantage'] = a
            row['segment'] = 'front' if row['turn_index'] < mid_index else 'back'
            row['advantage'] = a if row['valid'] else min(a, cfg['invalid_advantage_ceiling'])
            rows.append(row)
    return rows, {'mid': a_mid, 'end': a_end,
                  'has_signal': any(abs(r['advantage']) > 1e-12 for r in rows)}

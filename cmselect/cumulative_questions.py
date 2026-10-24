"""Expand validated cutoff quizzes, only within the current history and prefix.

Question text, labels, multiplicity, and as-of meaning remain unchanged. The ID
is namespaced for logging because local QG question IDs repeat across cutoffs.
The manager never sees this scoring pool. Reward/advantage code is unchanged.
"""
from copy import deepcopy
from .core import digest
from .rewards import alternatives


def accumulate_questions(sessions):
    history = None
    closed = set()
    pool = []
    origins = []
    previous_end = 0
    counts = {}
    cutoffs = []
    for session in sessions:
        hid = session['history_id']
        if hid != history:
            if hid in closed:
                raise ValueError('Noncontiguous history order')
            if history is not None:
                closed.add(history)
            history = hid
            pool, origins, previous_end = [], [], 0
        if session['start'] != previous_end:
            raise ValueError('Missing/reordered session')
        if not session['start'] < session['mid'] < session['end'] <= len(session['records']):
            raise ValueError('Invalid cutoff order')
        for field, stop in (('mid_questions', session['mid']), ('end_questions', session['end'])):
            group = session[field]
            allowed = {r['turn_id'] for r in session['records'][:stop]}
            new = []
            for original in group['questions']:
                ids = set().union(*alternatives(original['sufficient_sets']))
                quoted = {e['turn_id'] for e in original['evidence']}
                if not ids <= allowed or not quoted <= allowed:
                    raise ValueError('Future/unknown evidence in cumulative pool')
                q = deepcopy(original)
                q['question_id'] = group['key'] + '::' + original['question_id']
                new.append(q)
            pool.extend(new)
            origins.append({'key': group['key'], 'history_id': hid,
                            'cutoff_turn_count': stop, 'new_questions': len(new),
                            'original_sha256': group['sha256']})
            if any(o['history_id'] != hid or o['cutoff_turn_count'] > stop for o in origins):
                raise ValueError('Cross-history/future question group')
            if len({q['question_id'] for q in pool}) != len(pool):
                raise ValueError('Duplicate cutoff-scoped question identity')
            session[field] = {**group, 'questions': list(pool), 'sha256': digest(pool),
                              'source_groups': deepcopy(origins), 'cutoff_turn_count': stop}
            cutoffs.append({'history_id': hid, 'key': group['key'],
                            'cutoff_turn_count': stop, 'questions': len(pool),
                            'source_group_count': len(origins), 'sha256': digest(pool)})
        previous_end = session['end']
        counts[hid] = len(pool)
    return {'mode': 'same_history_all_questions_through_current_cutoff',
            'new_qg_calls': 0, 'deduplicate': False,
            'future_cutoff_questions_allowed': False,
            'preserve_original_question_text_and_asof': True,
            'final_questions_per_history': counts, 'cutoffs': cutoffs}

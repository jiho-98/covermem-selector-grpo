"""Validate the original 3,338 training annotations; never load official test QA here."""
from collections import defaultdict
from copy import deepcopy
from itertools import groupby
from .config import resolve
from .core import read, digest, file_hash
from .rewards import validate_questions
from .cumulative_questions import accumulate_questions

def load_sessions(assets):
    root = resolve(assets)
    hs, split, qg, done = (read(root / n) for n in
                          ('histories.json', 'split.json', 'questions.json', 'QG_COMPLETE.json'))
    ids = [h['source_id'] for h in hs['histories']]
    if ids != hs['train_ids'] or ids != split['train'] or ids != split['sft'] or ids != split['rl']:
        raise ValueError('Train split identity mismatch')
    if set(ids) & (set(split['test']) | set(split['dev'])):
        raise ValueError('Held-out conversation in training')
    if hs['source_sha256'] != split['source']['sha256']:
        raise ValueError('Conversation source mismatch')
    if done['status'] != 'complete' or qg['fingerprint'] != done['fingerprint']:
        raise ValueError('Incomplete QG')
    if qg['schema_version'] != 'turn_id_outer_or_inner_and.v1':
        raise ValueError('Unknown evidence annotation schema')
    sessions, used, questions, pools = [], set(), 0, defaultdict(list)
    for h in hs['histories']:
        cursor, seen = 0, set()
        for sid, records_iter in groupby(h['records'], key=lambda r: r['session_id']):
            records = list(records_iter)
            if sid in seen or len(records) < 2:
                raise ValueError('Noncontiguous/one-turn session')
            seen.add(sid)
            mid, end = cursor + (len(records) + 1) // 2, cursor + len(records)
            groups = []
            for half, stop in enumerate((mid, end)):
                key = f"{h['source_id']}_{sid}_h{half}"
                g = qg['groups'][key]
                prefix = h['records'][:stop]
                if (g['history_id'], g['session_id'], g['half'], g['cutoff_turn_count']) != (h['source_id'], sid, half, stop):
                    raise ValueError('Question cutoff mismatch')
                if g['prefix_hash'] != digest(prefix) or g['cutoff_turn_id'] != prefix[-1]['turn_id']:
                    raise ValueError('Question prefix mismatch')
                validate_questions(g['questions'], prefix)
                if len(g['questions']) != g['actual']:
                    raise ValueError('Question count mismatch')
                groups.append({'key': key, 'questions': g['questions'], 'sha256': digest(g['questions'])})
                for q in g['questions']:
                    pools[h['source_id']].append({**deepcopy(q), 'question_id': key + '::' + q['question_id'],
                                                 'available_at': stop})
                used.add(key)
                questions += len(g['questions'])
            sessions.append({'history_id': h['source_id'], 'session_id': sid, 'start': cursor, 'mid': mid,
                             'end': end, 'records': h['records'], 'mid_questions': groups[0], 'end_questions': groups[1]})
            cursor = end
        if cursor != len(h['records']) or len(seen) != h['sessions']:
            raise ValueError('Missing session/record')
    if used != set(qg['groups']) or (len(sessions), len(used), questions) != (142, 284, 3338):
        raise ValueError('Expected original 142 sessions / 284 cutoffs / 3338 questions')
    audit = accumulate_questions(sessions)
    hashes = {n: file_hash(root / n) for n in ('histories.json', 'split.json', 'questions.json', 'QG_COMPLETE.json')}
    return sessions, dict(pools), {'sha256': hashes, 'sessions': len(sessions), 'questions': questions,
                                  'train_ids': ids, 'cumulative_counts': audit['final_questions_per_history']}

def make_batches(sessions, batch_size):
    if batch_size == 1:
        return [[s] for s in sessions]
    if batch_size != 5:
        raise ValueError('Only original session schedule or round-robin batch5 schedule')
    histories = list(dict.fromkeys(s['history_id'] for s in sessions))
    per_history = [[s for s in sessions if s['history_id'] == h] for h in histories]
    return [[seq[i] for seq in per_history if i < len(seq)] for i in range(max(map(len, per_history)))]

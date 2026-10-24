"""CPU-only round-robin schedule and resumable, per-history bank state."""
from collections import OrderedDict, Counter
from copy import deepcopy
import hashlib
import json

STATE_SCHEMA = 'covermem.loo4_batch5_state.v1'


def history_order(sessions):
    return list(dict.fromkeys(s['history_id'] for s in sessions))


def build_batches(sessions, max_sessions=None):
    grouped = OrderedDict()
    for session in sessions:
        hid = session['history_id']
        ordered = grouped.setdefault(hid, [])
        expected_start = ordered[-1]['end'] if ordered else 0
        if session['start'] != expected_start:
            raise ValueError(f'Nonchronological session input for {hid}')
        if not session['start'] < session['mid'] < session['end']:
            raise ValueError('Every session must retain both original evaluation cutoffs')
        if any(s['session_id'] == session['session_id'] for s in ordered):
            raise ValueError('Duplicate session')
        ordered.append(session)
    if not 1 <= len(grouped) <= 5:
        raise ValueError('Expected one to five independent histories')
    if max_sessions is not None and (type(max_sessions) is not int or max_sessions < 1):
        raise ValueError('Invalid session limit')
    batches = []
    count = 0
    for index in range(max(map(len, grouped.values()))):
        batch = []
        for values in grouped.values():
            if index < len(values) and (max_sessions is None or count < max_sessions):
                batch.append(values[index])
                count += 1
        if batch:
            batches.append(batch)
        if max_sessions is not None and count >= max_sessions:
            break
    return batches


def schedule_digest(batches):
    description = [[{key: s[key] for key in ('history_id', 'session_id', 'start', 'mid', 'end')}
                    for s in batch] for batch in batches]
    encoded = json.dumps(description, sort_keys=True, separators=(',', ':')).encode()
    return hashlib.sha256(encoded).hexdigest()


def initial_state(histories, batches, world):
    if len(histories) != len(set(histories)):
        raise ValueError('Duplicate history ID')
    return {'schema_version': STATE_SCHEMA, 'next_batch': 0, 'completed_sessions': 0,
            'updates': 0, 'epoch': 0, 'world_size': world, 'history_order': list(histories),
            'schedule_digest': schedule_digest(batches),
            'session_cursors': {hid: 0 for hid in histories},
            'banks_by_history': {hid: [[], [], [], []] for hid in histories}}


def validate_state(state, histories, batches, world):
    if state.get('schema_version') != STATE_SCHEMA:
        raise ValueError('Wrong batch checkpoint schema')
    if state.get('world_size') != world or state.get('epoch') != 0:
        raise ValueError('Changed resume topology or epoch')
    if state.get('history_order') != list(histories) or state.get('schedule_digest') != schedule_digest(batches):
        raise ValueError('Changed chronological batch schedule')
    cursor = state['next_batch']
    if type(cursor) is not int or not 0 <= cursor <= len(batches):
        raise ValueError('Invalid batch cursor')
    counts = Counter(s['history_id'] for batch in batches[:cursor] for s in batch)
    expected = {hid: counts[hid] for hid in histories}
    if state['session_cursors'] != expected or state['completed_sessions'] != sum(counts.values()):
        raise ValueError('Per-history cursor does not match committed batches')
    if type(state['updates']) is not int or not 0 <= state['updates'] <= cursor:
        raise ValueError('Invalid optimizer update count')
    if set(state['banks_by_history']) != set(histories):
        raise ValueError('Missing or extra history banks')
    for banks in state['banks_by_history'].values():
        if not isinstance(banks, list) or len(banks) != 4 or not all(isinstance(b, list) for b in banks):
            raise ValueError('Each history must retain exactly four stream banks')


def advance_state(state, batch, end_banks, did_update):
    histories = [s['history_id'] for s in batch]
    if len(histories) != len(set(histories)) or set(histories) != set(end_banks):
        raise ValueError('A batch must advance exactly one session per included history')
    result = deepcopy(state)
    for hid in histories:
        banks = end_banks[hid]
        if len(banks) != 4:
            raise ValueError('Wrong number of final stream banks')
        result['banks_by_history'][hid] = deepcopy(banks)
        result['session_cursors'][hid] += 1
    result['next_batch'] += 1
    result['completed_sessions'] += len(batch)
    result['updates'] += int(did_update)
    return result

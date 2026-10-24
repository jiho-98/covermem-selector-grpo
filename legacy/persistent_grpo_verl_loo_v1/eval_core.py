"""Existing RAW SFT evaluation protocol, reused for the final GRPO adapter.
Only the prompt-token counting API is corrected (return_dict=False).
ROOT is configured by evaluate.py before workers start.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import urllib.request
from common import (ROOT as RECIPE_ROOT, CONTRACT_FINGERPRINT, build_messages,
                    make_observation, parse_action, apply_action, storage_chars)
from covermem.memory import (atomic_json, canonical, digest, load_committed_trace,
                             append_committed_trace, evidence_metrics)
ROOT=RECIPE_ROOT/'runs/evidence_v1/evaluation'
BUDGET=43500
MODEL='coverage-policy-final'
SETTINGS=dict(temperature=0.0,seed=42,max_output_tokens=8192,max_model_len=32768,
              retries=5,parallel_histories=5,response_format='json_object',
              rejected_action='retry with shared validation feedback; exhausted retries leave bank unchanged')
def now():return datetime.now(timezone.utc).isoformat()

def verify_raw(bank, records):
    raw = {r['turn_id']: r['content'] for r in records}
    ids = []
    for entry in bank:
        assert entry['content'] == '\n'.join(raw[i] for i in entry['source_turn_ids'])
        ids.extend(entry['source_turn_ids'])
    assert len(ids) == len(set(ids))
    assert storage_chars(bank) <= BUDGET
    return set(ids)

def replay(history, fingerprint, path):
    bank = []
    rows = load_committed_trace(path)
    assert len(rows) <= len(history['records'])
    for i, row in enumerate(rows):
        assert row['fingerprint'] == fingerprint and row['index'] == i
        assert row['before_hash'] == digest(bank)
        if row['outcome'] == 'applied':
            action = parse_action(row['attempts'][-1]['content'])
            assert action == row['action'] and row['attempts'][-1]['valid']
            bank = apply_action(bank, history['records'][i], action, BUDGET)
        else:
            assert row['outcome'] == 'invalid_no_op' and row['action'] is None
            assert len(row['attempts']) == SETTINGS['retries'] + 1
            assert not any(a['valid'] for a in row['attempts'])
        assert digest(bank) == row['after_hash']
    verify_raw(bank, history['records'])
    return bank, rows

def post(endpoint, payload):
    request = urllib.request.Request(endpoint + '/chat/completions',
        data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json'})
    # Transport failures must never be counted as model mistakes.
    error = None
    for attempt in range(4):
        try:
            with urllib.request.urlopen(request, timeout=600) as response:
                return json.load(response)
        except Exception as exc:
            error = exc
            if attempt < 3:
                time.sleep(2 ** attempt)
    raise RuntimeError('Inference transport failure') from error

def run_history(history, prepared, endpoint, tokenizer, stop_after=None):
    directory = ROOT / 'runs/histories' / history['source_id']
    directory.mkdir(parents=True, exist_ok=True)
    trace = directory / 'trace.jsonl'
    bank, rows = replay(history, prepared['fingerprint'], trace)
    final = directory / 'bank.json'
    if final.exists():
        result = json.loads(final.read_text())
        assert result['fingerprint'] == prepared['fingerprint'] and result['final_memory'] == bank
        assert len(rows) == len(history['records'])
        return result
    for index in range(len(rows), len(history['records'])):
        if stop_after is not None and index >= stop_after:
            return dict(status='smoke_complete', completed_turns=len(rows))
        current = history['records'][index]
        observation = make_observation(bank, current, BUDGET)
        attempts, error, action, candidate = [], None, None, None
        before = digest(bank)
        for attempt in range(SETTINGS['retries'] + 1):
            messages = build_messages(observation, previous_validation_error=error)
            tokens = len(tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=False))
            available = SETTINGS['max_model_len'] - tokens - 64
            if available < 256:
                raise RuntimeError('Input context limit reached; refusing to truncate RAW bank')
            payload = dict(model=MODEL, messages=messages, temperature=SETTINGS['temperature'],
                seed=SETTINGS['seed'], max_tokens=min(SETTINGS['max_output_tokens'], available),
                response_format={'type': 'json_object'})
            started = time.monotonic()
            response = post(endpoint, payload)
            choice = response['choices'][0]
            content = choice['message'].get('content')
            item = dict(content=content, finish_reason=choice['finish_reason'],
                        usage=response.get('usage', {}), seconds=time.monotonic()-started,
                        measured_prompt_tokens=tokens, requested_max_tokens=payload['max_tokens'])
            try:
                if choice['finish_reason'] != 'stop':
                    raise ValueError('Unfinished generation: ' + choice['finish_reason'])
                generated = parse_action(content)
                candidate = apply_action(bank, current, generated, BUDGET)
                action = generated
                item['valid'] = True
            except (ValueError, TypeError, KeyError) as exc:
                error = str(exc)
                item.update(valid=False, error=error)
            attempts.append(item)
            if item['valid']:
                break
        outcome = 'applied' if candidate is not None else 'invalid_no_op'
        if candidate is not None:
            bank = candidate
        row = dict(fingerprint=prepared['fingerprint'], index=index, current_turn_id=current['turn_id'],
            before_hash=before, after_hash=digest(bank), action=action, outcome=outcome,
            attempts=attempts, storage_chars=storage_chars(bank), time=now())
        append_committed_trace(trace, row)
        rows.append(row)
        progress = dict(status='running', conversation=history['source_id'], completed_turns=len(rows),
                        total_turns=len(history['records']), storage_chars=storage_chars(bank), time=now())
        atomic_json(directory / 'progress.json', progress)
        if index % 10 == 0:
            print(canonical(progress), flush=True)
    retained = verify_raw(bank, history['records'])
    details = []
    for question in history['questions']:
        if question['eligible']:
            gold = set(question['evidence_ids']); overlap = gold & retained
            details.append(dict(question_id=question['question_id'], category=question['category'],
                complete=float(overlap == gold), recall=len(overlap)/len(gold), hit=float(bool(overlap)),
                gold_ids=sorted(gold), retained_gold_ids=sorted(overlap)))
    actions = Counter(row['action']['action'] for row in rows if row['outcome'] == 'applied')
    result = dict(status='complete', fingerprint=prepared['fingerprint'], conversation=history['source_id'],
        completed_at=now(), completed_turns=len(rows), final_memory=bank, raw_verified=True,
        metrics=evidence_metrics(retained, history['questions']), action_distribution=dict(actions),
        invalid_final_decisions=sum(r['outcome'] == 'invalid_no_op' for r in rows),
        invalid_attempts=sum(not a['valid'] for r in rows for a in r['attempts']),
        retried_decisions=sum(len(r['attempts']) > 1 for r in rows),
        first_attempt_valid_rate=sum(r['attempts'][0]['valid'] for r in rows)/len(rows),
        entries=len(bank), retained_turns=len(retained), content_chars=storage_chars(bank),
        content_tokens=sum(len(tokenizer.encode(e['content'], add_special_tokens=False)) for e in bank),
        bank_json_tokens=len(tokenizer.encode(canonical(bank), add_special_tokens=False)))
    atomic_json(directory / 'per_question.json', details)
    atomic_json(final, result)
    atomic_json(directory / 'progress.json', dict(status='complete', completed_turns=len(rows), time=now()))
    print(canonical({k:v for k,v in result.items() if k != 'final_memory'}), flush=True)
    return result

def aggregate(prepared, results):
    n = sum(r['metrics']['eligible_qa'] for r in results)
    summary = dict(status='complete' if len(results) == 5 else 'partial', time=now(),
        fingerprint=prepared['fingerprint'], completed_histories=len(results), eligible_questions=n,
        condition='persistent verl GRPO from new RAW SFT; active only; content 43500 Unicode characters; metadata excluded',
        pooled_question_mean={k:sum(r['metrics'][k]*r['metrics']['eligible_qa'] for r in results)/n
                              for k in ('complete','recall','hit')},
        equal_conversation_mean={k:sum(r['metrics'][k] for r in results)/len(results)
                                 for k in ('complete','recall','hit')},
        mean_storage={k:sum(r[k] for r in results)/len(results)
                      for k in ('content_chars','content_tokens','bank_json_tokens')},
        histories=[{k:v for k,v in r.items() if k != 'final_memory'} for r in results])
    atomic_json(ROOT / 'runs/summary.json', summary)
    if summary['status'] == 'complete':
        atomic_json(ROOT / 'runs/EVAL_COMPLETE.json', summary)

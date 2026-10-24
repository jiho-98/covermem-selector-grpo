"""Frozen RAW SFT inference, followed by CPU-only official evidence scoring."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parent
TRAIN = ROOT.parent / 'raw43500char_thought_teacher_sft_v4'
sys.path.insert(0, str(TRAIN / 'scripts'))
sys.path.insert(0, str(TRAIN / 'src'))
from covermem.active_policy import (CONTRACT_FINGERPRINT, CONTRACT_VERSION, build_messages,
    make_observation, parse_action, apply_action, storage_chars)
from covermem.memory import atomic_json, canonical, digest, append_committed_trace, load_committed_trace, evidence_metrics
from teacher_active_manager_probe import prepare_locomo, file_hash

BUDGET = 43500
MODEL = 'covermem-raw43500-sft'
SETTINGS = dict(temperature=0.0, seed=42, max_output_tokens=8192, max_model_len=32768,
                retries=5, parallel_histories=5, response_format='json_object',
                rejected_action='retry with shared validation feedback; exhausted retries leave bank unchanged')

def now():
    return datetime.now(timezone.utc).isoformat()

def prepare():
    report = json.loads((TRAIN / 'runs/student/TRAIN_COMPLETE.json').read_text())
    assert report['status'] == 'complete' and report['global_step'] == 390
    assert report['policy_contract']['fingerprint'] == CONTRACT_FINGERPRINT
    adapter = Path(report['final_adapter'])
    hashes = {n: file_hash(adapter / n) for n in report['adapter_files_sha256']}
    assert hashes == report['adapter_files_sha256'], 'Completed adapter was modified'
    source = TRAIN / 'data/source/locomo10.json'
    split = json.loads((TRAIN / 'data/manifests/locomo_student5_eval5_v1.json').read_text())
    assert file_hash(source) == split['source']['sha256']
    assert not set(split['train']) & set(split['test'])
    histories = [prepare_locomo(source, c) for c in split['test']]
    assert (len(histories), sum(len(h['records']) for h in histories),
            sum(q['eligible'] for h in histories for q in h['questions'])) == (5, 2769, 729)
    provenance = dict(contract=CONTRACT_FINGERPRINT, adapter_sha256=hashes,
        source_sha256=split['source']['sha256'], conversations=split['test'], settings=SETTINGS,
        budget_chars=BUDGET, driver_sha256=file_hash(Path(__file__)),
        policy_sha256=file_hash(TRAIN / 'src/covermem/active_policy.py'),
        loader_sha256=file_hash(TRAIN / 'scripts/teacher_active_manager_probe.py'))
    info = dict(fingerprint=digest(provenance), provenance=provenance,
                manager_query_access=False, raw_content_only=True, shadow=False, qa=False,
                scope='student held-out five; previously development-exposed benchmark',
                conversations=5, turns=2769, eligible_questions=729)
    path = ROOT / 'runs/prepare.json'
    if path.exists():
        assert json.loads(path.read_text()) == info, 'Changed evaluation; use a new output directory'
    else:
        atomic_json(path, info)
    return info, histories

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
            tokens = len(tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True))
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
        condition='new RAW SFT; active only; content 43500 Unicode characters; metadata excluded',
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--endpoint')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    prepared, histories = prepare()
    if args.prepare_only:
        print(canonical(prepared)); return
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(TRAIN / 'tokenizers/Qwen3-4B-Instruct-2507', local_files_only=True)
    if args.smoke:
        print(run_history(histories[0], prepared, args.endpoint, tokenizer, stop_after=2)); return
    results = []
    with ThreadPoolExecutor(max_workers=5) as pool:
        futures = [pool.submit(run_history, h, prepared, args.endpoint, tokenizer) for h in histories]
        for future in as_completed(futures):
            results.append(future.result())
            aggregate(prepared, results)

if __name__ == '__main__':
    main()

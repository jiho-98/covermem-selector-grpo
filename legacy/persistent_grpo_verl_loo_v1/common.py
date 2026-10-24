"""CPU-only shared helpers; importing this file never initializes a GPU."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'frozen'))
from covermem.active_policy import (CONTRACT_FINGERPRINT, apply_action, build_messages,
    make_observation, parse_action, storage_chars)

def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))

def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()

def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''): h.update(b)
    return h.hexdigest()

def read(path):
    return json.loads(Path(path).read_text())

def atomic(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, prefix='.pending-', delete=False) as f:
        tmp = Path(f.name); json.dump(value, f, ensure_ascii=False, indent=2)
        f.write('\n'); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)
    fd = os.open(path.parent, os.O_DIRECTORY)
    try: os.fsync(fd)
    finally: os.close(fd)

def now(): return datetime.now(timezone.utc).isoformat()

def log(event, **values): print(canonical({'time': now(), 'event': event, **values}), flush=True)

def resolve(value):
    p = Path(value); return p.resolve() if p.is_absolute() else (ROOT / p).resolve()

def config(path=None):
    if sys.flags.optimize:raise RuntimeError('Do not disable protocol assertions with python -O/PYTHONOPTIMIZE')
    c = read(path or ROOT / 'config.json')
    assert c['environment']['streams'] == 4
    assert c['environment']['bank_budget_chars'] == 43500
    assert 1 <= c['environment']['max_action_attempts'] <= 6
    assert c['reward']['complete'] == .7 and c['reward']['recall'] == .3
    assert c['reward']['mid_weight'] == c['reward']['end_weight'] == .5
    assert c['reward']['std_correction'] == 0
    assert c['training']['ppo_epochs'] == c['training']['effective_batch_sessions'] == 1
    assert c['training']['micro_batch_actions'] == 1
    assert c['training']['data_epochs'] >= 1
    assert c['training']['max_sessions'] is None or c['training']['max_sessions']>=1
    assert c['rollout']['temperature'] == c['rollout']['top_p'] == 1.0
    assert c['rollout']['top_k'] == -1
    assert c['training']['loss_aggregation'] == 'global_response_token_mean'
    return c

def retained_ids(bank): return {t for e in bank for t in e['source_turn_ids']}

def verify_bank(bank, prefix, budget):
    raw = {r['turn_id']: r['content'] for r in prefix}; seen = set()
    for e in bank:
        sources = e['source_turn_ids']
        if not sources or seen.intersection(sources) or len(sources)!=len(set(sources)):
            raise ValueError('Empty or duplicated retained source IDs')
        if any(t not in raw for t in sources):raise ValueError('Unseen/future source in bank')
        if e['content']!='\n'.join(raw[t] for t in sources):raise ValueError('RAW/source mismatch')
        seen.update(sources)
    if storage_chars(bank)>budget:raise ValueError('Bank exceeds content-character budget')
    return seen

def prompt_ids(tokenizer, bank, turn, budget, error=None):
    messages = build_messages(make_observation(bank, turn, budget), previous_validation_error=error)
    # Transformers 5 returns a BatchEncoding unless return_dict=False is explicit.
    ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=False)
    if not isinstance(ids, list) or not ids or not all(type(x) is int for x in ids):
        raise TypeError('Chat template must yield the exact flat input token IDs')
    return ids

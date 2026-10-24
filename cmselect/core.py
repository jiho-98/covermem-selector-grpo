"""CPU-only serialization, identities, and frozen RAW-bank helpers."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import tempfile
from .active_policy import apply_action, parse_action, storage_chars

ROOT = Path(__file__).resolve().parents[1]

def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))

def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()

def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()

def read(path):
    return json.loads(Path(path).read_text())

def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, prefix='.pending-', delete=False) as f:
        temp = Path(f.name)
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)
    fsync_dir(path.parent)

def fsync_dir(path):
    fd = os.open(path, os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)

def retained_ids(bank):
    return {t for e in bank for t in e['source_turn_ids']}

def verify_bank(bank, prefix, budget):
    raw = {r['turn_id']: r['content'] for r in prefix}
    seen = set()
    for e in bank:
        sources = e['source_turn_ids']
        if not sources or seen.intersection(sources) or len(sources) != len(set(sources)):
            raise ValueError('Empty or duplicated retained source IDs')
        if any(t not in raw for t in sources):
            raise ValueError('Unseen/future source in bank')
        if e['content'] != '\n'.join(raw[t] for t in sources):
            raise ValueError('RAW/source mismatch')
        seen.update(sources)
    if storage_chars(bank) > budget:
        raise ValueError('Bank exceeds content-character budget')
    return seen

def log(event, **values):
    print(canonical({'event': event, **values}), flush=True)

def request_seed(*parts):
    return int(digest(parts)[:15], 16) % 2147483647

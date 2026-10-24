"""Atomic commit of BOTH adapters, optimizers, banks, cursors and RNG."""
import os
from pathlib import Path
import random
import shutil
import uuid
from .core import atomic, read, file_hash, fsync_dir

def validate(path, fingerprint=None):
    path = Path(path)
    marker = read(path / 'COMMITTED.json')
    if marker['status'] != 'complete' or (fingerprint and marker['fingerprint'] != fingerprint):
        raise ValueError('Checkpoint experiment identity mismatch')
    for name, expected in marker['sha256'].items():
        if file_hash(path / name) != expected:
            raise ValueError('Checkpoint corruption: ' + name)
    state = read(path / 'state.json')
    if state['next_batch'] != marker['next_batch']:
        raise ValueError('Checkpoint cursor mismatch')
    return state

def latest(output, fingerprint):
    root = Path(output) / 'checkpoints'
    commits = sorted(p for p in root.glob('batch_*') if (p / 'COMMITTED.json').is_file())
    if not commits:
        return None
    path = commits[-1]
    return path, validate(path, fingerprint)

def commit(output, fingerprint, state, backend, metrics, traces, keep_last):
    root = Path(output) / 'checkpoints'
    root.mkdir(parents=True, exist_ok=True)
    destination = root / f"batch_{state['next_batch']:06d}"
    if destination.exists():
        raise FileExistsError('Refuse to overwrite committed/recoverable checkpoint: ' + str(destination))
    pending = root / ('.pending-' + uuid.uuid4().hex)
    pending.mkdir()
    backend.save(pending)
    saved_state = {**state, 'python_rng': random.getstate()}
    atomic(pending / 'state.json', saved_state)
    atomic(pending / 'metrics.json', metrics)
    atomic(pending / 'traces.json', traces)
    hashes = {}
    for path in sorted(pending.rglob('*')):
        if path.is_file():
            with path.open('rb') as f:
                os.fsync(f.fileno())
            hashes[path.relative_to(pending).as_posix()] = file_hash(path)
    atomic(pending / 'COMMITTED.json', {'status': 'complete', 'fingerprint': fingerprint,
                                       'next_batch': state['next_batch'], 'sha256': hashes})
    fsync_dir(pending)
    os.replace(pending, destination)
    fsync_dir(root)
    atomic(Path(output) / 'LATEST.json', {'checkpoint': destination.name, 'fingerprint': fingerprint})
    commits = sorted(p for p in root.glob('batch_*') if (p / 'COMMITTED.json').is_file())
    # Recover the archive even if a prior process died between commit and archival.
    # Never prune a checkpoint before its scientific trace has been archived durably.
    for committed in commits:
        archive = Path(output) / 'batches' / committed.name
        archive.mkdir(parents=True, exist_ok=True)
        for name in ('metrics.json', 'traces.json', 'COMMITTED.json'):
            source, target = committed / name, archive / name
            if target.exists() and file_hash(target) == file_hash(source):
                continue
            temp = archive / ('.pending-' + uuid.uuid4().hex)
            shutil.copy2(source, temp)
            with temp.open('rb') as f:
                os.fsync(f.fileno())
            os.replace(temp, target)
        fsync_dir(archive)
    for old in commits[:-keep_last]:
        # Only this run's validated committed directories, never external paths.
        if read(old / 'COMMITTED.json')['fingerprint'] != fingerprint:
            raise ValueError('Mixed experiment checkpoints')
        shutil.rmtree(old)
    return destination

def restore_python_rng(state):
    def tuples(x):
        return tuple(tuples(y) for y in x) if isinstance(x, list) else x
    if 'python_rng' in state:
        random.setstate(tuples(state['python_rng']))

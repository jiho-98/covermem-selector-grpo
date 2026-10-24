"""Durable queue and verified process ownership. No scheduler side effects."""
import contextlib
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import tempfile
import time

ROOT = Path(__file__).resolve().parent


def read(path):
    return json.loads(Path(path).read_text())


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, delete=False) as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write('\n'); f.flush(); os.fsync(f.fileno())
        tmp = Path(f.name)
    os.replace(tmp, path)
    fd = os.open(path.parent, os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


@contextlib.contextmanager
def locked(path, nonblocking=False):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+') as f:
        fcntl.flock(f, fcntl.LOCK_EX | (fcntl.LOCK_NB if nonblocking else 0))
        yield f


def identity(pid):
    try:
        p = Path('/proc') / str(int(pid))
        raw = (p / 'stat').read_text().rsplit(')', 1)[1].split()
        if raw[0] == 'Z':
            return None
        return dict(pid=int(pid), start_tick=int(raw[19]), uid=p.stat().st_uid,
                    ppid=int(raw[1]), pgid=int(raw[2]))
    except (FileNotFoundError, ProcessLookupError):
        return None


def same_process(expected):
    actual = identity(expected['pid'])
    return actual is not None and all(actual[k] == expected[k] for k in ('pid', 'start_tick', 'uid'))


def environment_of(pid):
    try:
        data = (Path('/proc') / str(pid) / 'environ').read_bytes()
        return dict(part.decode(errors='replace').split('=', 1) for part in data.split(b'\0') if b'=' in part)
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        return {}


def register_stage_tree(token, primary):
    path = ROOT / 'owners' / ('stage-' + token + '.json')
    previous = read(path) if path.exists() else {'token': token, 'records': []}
    known = {(r['pid'], r['start_tick']): r for r in previous['records']}
    if same_process(primary): known[(primary['pid'], primary['start_tick'])] = primary
    candidates = {}
    for p in Path('/proc').iterdir():
        if not p.name.isdigit(): continue
        try:
            if p.stat().st_uid != os.getuid(): continue
            record = identity(int(p.name))
            if record: candidates[record['pid']] = record
        except OSError: continue
    parents = {r['pid'] for r in known.values() if same_process(r)}
    while True:
        additions = {pid for pid, r in candidates.items() if r['ppid'] in parents} - parents
        if not additions: break
        parents.update(additions)
        for pid in additions:
            r = candidates[pid];known[(pid,r['start_tick'])] = r
    atomic(path, {'token': token, 'records': [r for r in known.values() if same_process(r)]})


def owned_processes(token):
    result = []
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():
            continue
        try:
            if p.stat().st_uid != os.getuid():
                continue
            if environment_of(p.name).get('COVERMEM_STAGE_OWNER') != token:
                continue
            info = identity(int(p.name))
            if info and info['pid'] != os.getpid():
                result.append(info)
        except (FileNotFoundError, ProcessLookupError):
            continue
    # Forked engine workers can lose initial /proc environment markers. Their
    # API owner registers verified ancestry and start ticks before reparenting.
    for path in (ROOT / 'owners').glob('*.json'):
        saved = read(path)
        if saved.get('token') != token:
            continue
        for info in saved.get('records', []):
            if info['uid'] == os.getuid() and info['pid'] != os.getpid() and same_process(info):
                result.append(info)
    unique = {(r['pid'], r['start_tick']): r for r in result}
    return list(unique.values())


def signal_exact(record, sig, token=None):
    if record['uid'] != os.getuid() or record['pid'] == os.getpid():
        raise RuntimeError('Refusing to signal a different owner or this supervisor')
    if not same_process(record):
        return False
    if token is not None and environment_of(record['pid']).get('COVERMEM_STAGE_OWNER') != token:
        raise RuntimeError('Process no longer belongs to requested stage')
    # pidfd prevents a PID-reuse race between verification and delivery.
    fd = os.pidfd_open(record['pid'])
    try:
        if not same_process(record):
            return False
        signal.pidfd_send_signal(fd, sig)
        return True
    except ProcessLookupError:
        return False
    finally:
        os.close(fd)


def terminate_stage(token, primary=None, grace=60):
    """Only the tagged child stage, including reparented vLLM workers."""
    if primary:
        signal_exact(primary, signal.SIGTERM, token)
    began = time.monotonic()
    signaled = set()
    while True:
        records = owned_processes(token)
        if not records:
            return
        if time.monotonic() - began >= grace:
            break
        # Allow the primary's existing checkpoint/cleanup handlers to run first.
        if not primary or not same_process(primary) or time.monotonic() - began > 10:
            for record in records:
                key = (record['pid'], record['start_tick'])
                if key not in signaled:
                    signal_exact(record, signal.SIGTERM); signaled.add(key)
        time.sleep(.2)
    for record in owned_processes(token):
        signal_exact(record, signal.SIGKILL)
    for _ in range(50):
        if not owned_processes(token):
            return
        time.sleep(.1)
    raise RuntimeError('Verified stage workers did not exit; do not start replacement')


def validate_queue(q):
    if q.get('version') != 1 or q.get('resources') != {'gpus': 1}:
        raise ValueError('Expected a single-GPU portable queue')
    seen = set()
    for s in q['stages']:
        if not s.get('id') or s['id'] in seen:
            raise ValueError('Duplicate/empty stage ID')
        if not set(s.get('after', [])) <= seen:
            raise ValueError('Dependencies must precede the stage')
        if not isinstance(s['command'], list) or not s['command'] or not all(isinstance(v, str) for v in s['command']):
            raise ValueError('Commands must be explicit argv lists')
        if not s.get('authorization_note'):
            raise ValueError('Stage needs its user authorization context')
        if not Path(s['cwd']).is_absolute() or not Path(s['marker']).is_absolute():
            raise ValueError('Stage directories and marker must be absolute')
        if not set(s.get('gpu_indices', [0])) <= {0} or not s.get('gpu_indices', [0]):
            raise ValueError('Stage exceeds allocation')
        seen.add(s['id'])
    return q


def stage_signature(s):
    return digest(s)

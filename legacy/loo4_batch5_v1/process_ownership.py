"""Exact process identities and a durable ancestry registry; never killpg.

Some vLLM forks omit their parent's updated environment in /proc/environ. Record
ancestry while the API parent is alive, then retain PID/start-tick pairs even if
workers are reparented. GPU Python delegates pidfd signaling to system Python.
"""
import json
import os
from pathlib import Path
import signal
import subprocess
import threading
import time


def identity(pid):
    try:
        p = Path('/proc') / str(int(pid))
        fields = (p / 'stat').read_text().rsplit(')', 1)[1].split()
        if fields[0] == 'Z': return None
        return dict(pid=int(pid), start_tick=int(fields[19]), uid=p.stat().st_uid,
                    ppid=int(fields[1]), pgid=int(fields[2]))
    except (OSError, ValueError):
        return None


def same_process(record):
    actual = identity(record['pid'])
    return actual is not None and all(actual[k] == record[k] for k in ('pid', 'start_tick', 'uid'))


def signal_exact(record, sig):
    if record['uid'] != os.getuid() or record['pid'] == os.getpid():
        raise RuntimeError('Refusing to signal another user or self')
    if not same_process(record): return False
    if not hasattr(os, 'pidfd_open'):
        source = str(Path(__file__).resolve().parent)
        code = ('import sys,json;sys.path.insert(0,' + repr(source) + ');'
                'from process_ownership import signal_exact;'
                'signal_exact(json.loads(sys.argv[1]),int(sys.argv[2]))')
        subprocess.run(['/usr/bin/python3', '-B', '-c', code, json.dumps(record), str(int(sig))],
                       check=True, timeout=10)
        return True
    try:
        fd = os.pidfd_open(record['pid'])
    except ProcessLookupError:
        return False
    try:
        if not same_process(record): return False
        signal.pidfd_send_signal(fd, sig)
        return True
    except ProcessLookupError:
        return False
    finally:
        os.close(fd)


class OwnedTree:
    def __init__(self, root, token, registry_path=None):
        if root is None or root['uid'] != os.getuid():
            raise RuntimeError('Cannot register missing/foreign API process')
        self.root = root; self.token = token
        self.registry = {root['pid']: root}; self.lock = threading.RLock()
        self.path = Path(registry_path) if registry_path else None

    def refresh(self):
        with self.lock:
            candidates = {}
            for p in Path('/proc').iterdir():
                if not p.name.isdigit(): continue
                try:
                    if p.stat().st_uid != os.getuid(): continue
                    record = identity(int(p.name))
                    if record: candidates[record['pid']] = record
                except OSError:
                    continue
            # Only live, previously verified identities may establish ancestry.
            parents = {pid for pid, r in self.registry.items() if same_process(r)}
            while True:
                additions = {pid for pid, r in candidates.items() if r['ppid'] in parents} - parents
                if not additions: break
                parents.update(additions)
                for pid in additions: self.registry[pid] = candidates[pid]
            result = [r for r in self.registry.values() if same_process(r)]
            if self.path:
                # Pure JSON; imports no CUDA and prints no environment/secrets.
                from common import atomic
                atomic(self.path, {'token': self.token, 'api_root': self.root, 'records': result})
            return result

    def close(self, grace=20):
        self.refresh()
        if same_process(self.root): signal_exact(self.root, signal.SIGTERM)
        until = time.monotonic() + grace
        while time.monotonic() < until:
            if not self.refresh(): return
            time.sleep(.2)
        for sig in (signal.SIGTERM, signal.SIGKILL):
            for r in self.refresh(): signal_exact(r, sig)
            until = time.monotonic() + 10
            while time.monotonic() < until:
                if not self.refresh(): return
                time.sleep(.2)
        if self.refresh(): raise RuntimeError('Owned workers remain; replacement must stay blocked')

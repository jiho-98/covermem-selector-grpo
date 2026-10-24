"""Persistent allocation owner; every experiment is a tagged child process.

Queue edits never mutate an imported trainer. No group-wide signals, automatic
resubmissions, unbounded stage retries, or indefinite idle GPU reservations.
"""
import argparse
import os
from pathlib import Path
import signal
import subprocess
import time
import uuid
from support import (ROOT, atomic, identity, locked, owned_processes, read,
                     same_process, stage_signature, terminate_stage, validate_queue, register_stage_tree)


class Supervisor:
    def __init__(self, root, cpu_test=False):
        self.root = Path(root); self.cpu_test = cpu_test
        import support
        support.ROOT = self.root
        self.child = None; self.active = {}; self.stop_requested = False
        self.devices = [v for v in os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',') if v]
        if not cpu_test and len(self.devices) != 1:
            raise RuntimeError('Expose exactly one authorized GPU via CUDA_VISIBLE_DEVICES')
        self.state = read(self.root / 'state.json') if (self.root / 'state.json').exists() else {}
        # A previous supervisor crash may leave a child alive. Never start a duplicate.
        stale = self.state.get('active', {})
        if stale and (same_process(stale['process']) or owned_processes(stale['token'])):
            raise RuntimeError('Previous stage remains alive; verify/recover its owner before restarting')
        self.results = self.state.get('results', {})
        self.repair_since = None
        self.handled = set(self.state.get('handled_requests', []))
        self.signatures = self.state.get('started_signatures', {})

    def publish(self, status, **extra):
        self.state = dict(status=status, updated_at=time.time(), job_id=os.environ.get('SLURM_JOB_ID'),
                          supervisor=identity(os.getpid()), results=self.results,
                          active=self.active, handled_requests=sorted(self.handled),
                          started_signatures=self.signatures, **extra)
        atomic(self.root / 'state.json', self.state)

    def queue(self):
        with locked(self.root / 'queue.lock'):
            q = validate_queue(read(self.root / 'queue.json'))
        for s in q['stages']:
            if s['id'] in self.signatures and self.signatures[s['id']] != stage_signature(s):
                raise RuntimeError('A started stage was edited; use a new experiment/stage ID')
        return q

    def controls(self, q):
        for request in q.get('requests', []):
            nonce = request['nonce']
            if nonce in self.handled:
                continue
            ident = request['id']
            if request['action'] == 'cancel':
                if self.active.get('id') == ident:
                    terminate_stage(self.active['token'], self.active['process'])
                    self.child.wait(timeout=10); self.child = None; self.active = {}
                if self.results.get(ident, {}).get('status') != 'complete':
                    self.results[ident] = dict(status='paused', time=time.time())
            elif request['action'] == 'resume':
                if self.results.get(ident, {}).get('status') in ('failed', 'paused', 'interrupted'):
                    self.results.pop(ident)
            self.handled.add(nonce)

    @staticmethod
    def marker_valid(stage):
        marker = Path(stage['marker'])
        if not marker.exists():
            return False
        value = read(marker)
        return all(value.get(k) == v for k, v in stage.get('marker_fields', {'status': 'complete'}).items())

    def launch(self, stage):
        token = uuid.uuid4().hex
        env = dict(os.environ, COVERMEM_STAGE_OWNER=token)
        env.update(stage.get('env', {}))
        # Ownership and allocation masks cannot be overwritten by appended work.
        env['COVERMEM_STAGE_OWNER'] = token
        env['COVERMEM_OWNER_REGISTRY'] = str(self.root / 'owners')
        (self.root / 'owners').mkdir(exist_ok=True)
        if not self.cpu_test:
            env['CUDA_VISIBLE_DEVICES'] = ','.join(self.devices[i] for i in stage.get('gpu_indices', [0]))
        log = self.root / 'logs' / (stage['id'] + '.log'); log.parent.mkdir(parents=True, exist_ok=True)
        self.signatures[stage['id']] = stage_signature(stage)
        with log.open('a') as stream:
            self.child = subprocess.Popen(stage['command'], cwd=stage['cwd'], env=env,
                stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        process = identity(self.child.pid)
        if process is None:
            # A very short command can already be reaped; preserve the exit status below.
            process = dict(pid=self.child.pid, start_tick=-1, uid=os.getuid(), ppid=os.getpid(), pgid=self.child.pid)
        register_stage_tree(token, process)
        self.active = dict(id=stage['id'], token=token, process=process, started_at=time.time(), log=str(log))
        self.publish('running', pending_stage=stage['id'])

    def run(self):
        self.root.mkdir(parents=True, exist_ok=True)
        with locked(self.root / 'supervisor.lock', nonblocking=True):
            self.publish('starting')
            while not self.stop_requested:
                q = self.queue(); self.controls(q)
                if self.child:
                    code = self.child.poll()
                    if code is not None:
                        ident = self.active['id']; token = self.active['token']
                        if owned_processes(token):
                            terminate_stage(token)
                        s = next(s for s in q['stages'] if s['id'] == ident)
                        ok = code == 0 and self.marker_valid(s)
                        self.results[ident] = dict(status='complete' if ok else 'failed',
                            exit_code=code, marker=s['marker'], finished_at=time.time(),
                            error=None if ok else 'Stage failed or validated completion marker missing')
                        self.child = None; self.active = {}
                        self.publish('stage_complete' if ok else 'needs_repair_allocation_retained')
                    else:
                        register_stage_tree(self.active['token'], self.active['process'])
                        self.active['owned_processes'] = owned_processes(self.active['token'])
                        self.publish('running')
                        time.sleep(.5); continue
                # Failures pause the sequence until an explicit resume/control edit.
                blocked = [s['id'] for s in q['stages'] if self.results.get(s['id'], {}).get('status') in
                           ('failed', 'paused', 'interrupted')]
                if blocked:
                    if self.repair_since is None: self.repair_since = time.time()
                    if time.time() - self.repair_since > q.get('repair_grace_seconds', 1800):
                        self.publish('repair_lease_expired_allocation_releasing', blocked_stages=blocked)
                        return
                    self.publish('needs_repair_allocation_retained', blocked_stages=blocked)
                    time.sleep(.5); continue
                candidates = [s for s in q['stages'] if s['id'] not in self.results and
                              all(self.results.get(d, {}).get('status') == 'complete' for d in s.get('after', []))]
                if candidates:
                    self.repair_since = None
                    self.launch(candidates[0]); continue
                # Serialize final release with queue append to avoid losing appended work.
                with locked(self.root / 'queue.lock'):
                    latest = validate_queue(read(self.root / 'queue.json'))
                    if latest['revision'] != q['revision']:
                        continue
                    if not all(self.results.get(s['id'], {}).get('status') == 'complete' for s in latest['stages']):
                        raise RuntimeError('Unresolvable dependency graph')
                    if latest.get('hold_until_user_release', False) and not latest.get('release_requested', False):
                        self.publish('authorized_hold_allocation_retained')
                        time.sleep(.5)
                        continue
                    self.publish('complete_allocation_releasing')
                    atomic(self.root / 'ALL_COMPLETE.json', self.state)
                    return
            if self.active:
                ident = self.active['id']
                terminate_stage(self.active['token'], self.active['process'])
                self.results[ident] = dict(status='interrupted', time=time.time())
                self.active = {}; self.child = None
            self.publish('interrupted')


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--root', type=Path, default=ROOT)
    p.add_argument('--cpu-test', action='store_true'); a = p.parse_args()
    if a.cpu_test and os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('CPU fixture mode cannot masquerade as an allocated GPU job')
    s = Supervisor(a.root, a.cpu_test)
    for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGUSR1):
        signal.signal(signum, lambda *_: setattr(s, 'stop_requested', True))
    try:
        s.run()
        if s.state.get('status') == 'repair_lease_expired_allocation_releasing':
            raise SystemExit(2)
    except Exception as exc:
        # A supervisor/config error also retains the owner while repair is active.
        if s.active:
            terminate_stage(s.active['token'], s.active['process']); s.active = {}; s.child = None
        s.publish('supervisor_needs_repair_allocation_retained', error=str(exc))
        if a.cpu_test:
            raise
        deadline = time.monotonic() + 1800
        while not s.stop_requested and time.monotonic() < deadline:
            time.sleep(1)
        raise SystemExit(2)

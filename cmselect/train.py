"""A: completed frozen manager replay. B: separate manager/selector updates.

No CUDA initialization unless --train is supplied. No scheduler submission.
"""
import argparse
from copy import deepcopy
import fcntl
import os
from pathlib import Path
import random
import signal
from .config import load, resolve
from .core import atomic, read, digest, log
from .data import make_batches
from .action_credit import attach_action_credit
from .rollout import collect_stream
from . import checkpoint

class Interrupted(RuntimeError):
    pass

def initial_state(sessions):
    histories = list(dict.fromkeys(s['history_id'] for s in sessions))
    return {'next_batch': 0, 'completed_sessions': 0,
            'banks': {h: [[] for _ in range(4)] for h in histories},
            'session_cursors': {h: 0 for h in histories}, 'updates': {'manager': 0, 'selector': 0}}

def train_loop(cfg, report, sessions, pools, backend, output, state=None, stop_after_batches=None):
    """Shared production/CPU-fixture coordinator; batch replay after interruption."""
    output = Path(output)
    state = deepcopy(state or initial_state(sessions))
    schedule = make_batches(sessions, cfg['training']['session_batch_size'])
    schedule_id = digest([[(s['history_id'], s['session_id'], s['start'], s['end']) for s in b] for b in schedule])
    if state.get('schedule_id', schedule_id) != schedule_id:
        raise ValueError('Checkpoint schedule mismatch')
    state['schedule_id'] = schedule_id
    if not 0 <= state['next_batch'] <= len(schedule):
        raise ValueError('Checkpoint batch cursor out of range')
    expected_cursors = {h: 0 for h in state['session_cursors']}
    for batch in schedule[:state['next_batch']]:
        for s in batch:
            expected_cursors[s['history_id']] = s['end']
    if state['session_cursors'] != expected_cursors:
        raise ValueError('Checkpoint history cursors disagree with schedule')
    start_batch = state['next_batch']
    for bi in range(start_batch, len(schedule)):
        batch, manager_groups, selector_groups, traces, summary = schedule[bi], [], [], [], []
        next_state = deepcopy(state)
        log('collect_batch', batch=bi, variant=cfg['variant'], updates=state['updates'])
        # Neither adapter is updated anywhere in this collection loop.
        for session in batch:
            hid = session['history_id']
            if state['session_cursors'][hid] != session['start']:
                raise ValueError('Conversation sessions must replay chronologically')
            initial = deepcopy(state['banks'][hid])
            streams = [collect_stream(session, i, initial[i], backend, cfg, pools[hid], bi) for i in range(4)]
            credit_cfg = {**cfg['manager_reward'], 'bank_budget_chars': cfg['environment']['bank_budget_chars']}
            rows, credit = attach_action_credit(streams, session, initial, credit_cfg)
            manager_groups.append(rows)
            for stream in streams:
                selector_groups.extend(g['rows'] for g in stream['selector_groups'])
            next_state['banks'][hid] = [s['end_bank'] for s in streams]
            next_state['session_cursors'][hid] = session['end']
            next_state['completed_sessions'] += 1
            traces.append({'history_id': hid, 'session_id': session['session_id'], 'streams': streams, 'manager_rows': rows})
            summary.append({'history': hid, 'session': session['session_id'], 'manager_credit': credit,
                            'mid_rewards': [s['mid_score']['reward'] for s in streams],
                            'end_rewards': [s['end_score']['reward'] for s in streams],
                            'selector_groups': sum(len(s['selector_groups']) for s in streams)})
        metrics = {'batch': bi, 'sessions': summary, 'manager': {'optimizer_steps': 0, 'reason': 'frozen'},
                   'selector': {'optimizer_steps': 0, 'reason': 'disabled'}}
        # Update boundaries are separate; selector gradients never enter manager.
        if cfg['variant'] in ('B', 'vanilla'):
            metrics['manager'] = backend.update('manager', manager_groups)
        if cfg['variant'] in ('A', 'B'):
            metrics['selector'] = backend.update('selector', selector_groups)
        for adapter in ('manager', 'selector'):
            next_state['updates'][adapter] += metrics[adapter]['optimizer_steps']
        next_state['next_batch'] = bi + 1
        dest = checkpoint.commit(output, report['fingerprint'], next_state, backend, metrics, traces,
                                 cfg['checkpoint']['keep_last'])
        state = next_state
        atomic(output / 'progress.json', {'status': 'running', 'next_batch': bi + 1, 'total_batches': len(schedule),
                                         'updates': state['updates'], 'checkpoint': str(dest)})
        log('batch_committed', batch=bi, updates=state['updates'], selector_groups=len(selector_groups))
        if stop_after_batches is not None and bi + 1 - start_batch >= stop_after_batches:
            # Operational pause for smoke/resume test; never a TRAIN_COMPLETE marker.
            atomic(output / 'PAUSED.json', {'status': 'paused', 'fingerprint': report['fingerprint'], 'checkpoint': str(dest)})
            return state
    saved = checkpoint.latest(output, report['fingerprint'])
    if saved is None:
        raise ValueError('Cannot complete without a committed checkpoint')
    atomic(output / 'TRAIN_COMPLETE.json', {'status': 'complete', 'fingerprint': report['fingerprint'],
        'checkpoint': str(saved[0].relative_to(output)), 'completed_sessions': state['completed_sessions'], 'updates': state['updates'],
        'qa_calls': 0, 'qg_calls': 0})
    (output / 'PAUSED.json').unlink(missing_ok=True)
    return state

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--train', action='store_true')
    p.add_argument('--stop-after-batches', type=int)
    a = p.parse_args()
    from .preflight import inspect
    cfg = load(a.config)
    report, sessions, pools = inspect(cfg, assets_only=not a.train)
    if not a.train:
        print({'status': report['status'], 'gpu_started': False})
        return
    if a.stop_after_batches is not None and a.stop_after_batches < 1:
        p.error('--stop-after-batches must be positive')
    if not os.environ.get('COVERMEM_STAGE_OWNER'):
        raise RuntimeError('Run via scripts/run.sh / supervisor; do not bypass allocation ownership')
    if cfg['execution']['require_slurm'] and not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('This config requires an existing Slurm allocation')
    if int(os.environ.get('WORLD_SIZE', '1')) != 1:
        raise RuntimeError('Portable backend supports exactly one process/GPU; do not torchrun it')
    import torch
    from .backend import Backend
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('Expose exactly one GPU; multi-GPU topology is not implemented')
    output = resolve(cfg['paths']['output'])
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        old = read(output / 'prepared.json') if (output / 'prepared.json').exists() else report
        if old['fingerprint'] != report['fingerprint']:
            raise ValueError('Changed science/runtime/code: use a new output directory')
        atomic(output / 'prepared.json', report)
        saved = checkpoint.latest(output, report['fingerprint'])
        random.seed(cfg['training']['seed'])
        torch.manual_seed(cfg['training']['seed'])
        torch.cuda.manual_seed(cfg['training']['seed'])
        torch.set_num_threads(cfg['execution']['cpu_threads'])
        backend = Backend(cfg, saved[0] if saved else None)
        state = saved[1] if saved else None
        if saved:
            backend.restore_training(saved[0])
            checkpoint.restore_python_rng(state)
        def interrupt(*_):
            raise Interrupted('Stage stopped; resume last committed batch (uncommitted work is replayed)')
        for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGUSR1):
            signal.signal(signum, interrupt)
        try:
            train_loop(cfg, report, sessions, pools, backend, output, state, a.stop_after_batches)
        except Interrupted as exc:
            atomic(output / 'INTERRUPTED.json', {'status': 'interrupted', 'message': str(exc), 'fingerprint': report['fingerprint']})
            raise

if __name__ == '__main__':
    main()

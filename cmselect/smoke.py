"""Doctor-server GPU smoke, isolated from scientific training and its outputs.

Uses synthetic +/- advantages only to probe optimizer plumbing. Never use this
diagnostic's adapters to initialize or resume a research run.
"""
import argparse
import gc
import hashlib
import os
from pathlib import Path
from .config import load, resolve
from .core import atomic, log
from .active_policy import make_observation, build_messages
from .selection import candidates, messages

def adapter_hashes(backend):
    from .backend import parameter_owner
    out = {k: hashlib.sha256() for k in ('manager','selector','manager_ref','selector_ref')}
    for name, p in sorted(backend.model.named_parameters()):
        owner = parameter_owner(name)
        if owner:
            out[owner].update(p.detach().float().cpu().numpy().tobytes())
    return {k: v.hexdigest() for k, v in out.items()}

def main():
    p = argparse.ArgumentParser(); p.add_argument('--config', required=True); a = p.parse_args()
    cfg = load(a.config)
    from .preflight import inspect
    report, sessions, _ = inspect(cfg)
    if not os.environ.get('COVERMEM_STAGE_OWNER'):
        raise RuntimeError('GPU smoke must run as a supervised stage')
    if cfg['execution']['require_slurm'] and not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Config requires a Slurm allocation')
    import torch
    from .backend import Backend
    if torch.cuda.device_count() != 1:
        raise RuntimeError('Expose exactly one GPU')
    output = resolve(cfg['paths']['output']) / 'smoke'
    output.mkdir(parents=True, exist_ok=True)
    (output / 'SMOKE_COMPLETE.json').unlink(missing_ok=True)
    torch.manual_seed(cfg['training']['seed']); torch.cuda.manual_seed(cfg['training']['seed'])
    b = Backend(cfg)
    before = adapter_hashes(b)
    records = sessions[0]['records']
    bank, used, next_index = [], 0, 0
    # A near-capacity real RAW fixture tests the full input size, not just a greeting.
    for i, record in enumerate(records):
        if used + len(record['content']) > cfg['environment']['bank_budget_chars']:
            next_index = i
            break
        bank.append({'memory_id': 'm_'+record['turn_id'], 'content': record['content'], 'source_turn_ids':[record['turn_id']]})
        used += len(record['content'])
    else:
        raise ValueError('Smoke fixture never filled the bank')
    turn = records[next_index]
    manager = b.sample('manager', build_messages(make_observation(bank, turn, 43500)), 42,
                       cfg['rollout']['manager_max_new_tokens'])
    action = {'thought':'Forced REPLACE only for optimizer smoke fixture','action':'REPLACE',
              'target_id':bank[0]['memory_id'],'source_turn_ids':[turn['turn_id']]}
    eligible = candidates(bank, turn, action, 43500)
    selector = [b.sample('selector', messages(bank, turn, 43500, eligible), 42+i,
                        cfg['rollout']['selector_max_new_tokens']) for i in range(2)]
    for i, row in enumerate(selector):
        row.update(advantage=1. if i == 0 else -1., loss_mask=[1]*len(row['response_ids']))
    manager.update(advantage=1., loss_mask=[1]*len(manager['response_ids']))
    stats = {}
    if cfg['variant'] != 'A': stats['manager'] = b.update('manager', [[manager]])
    if cfg['variant'] != 'vanilla': stats['selector'] = b.update('selector', [selector])
    after = adapter_hashes(b)
    expected = {'selector'} if cfg['variant'] == 'A' else {'manager'} if cfg['variant'] == 'vanilla' else {'manager','selector'}
    if {k for k in before if before[k] != after[k]} != expected:
        raise RuntimeError('Unexpected adapter update/freeze behavior in GPU smoke')
    b.save(output / 'diagnostic_checkpoint')
    max_memory = torch.cuda.max_memory_allocated()
    del b; gc.collect(); torch.cuda.empty_cache()
    restored = Backend(cfg, output / 'diagnostic_checkpoint')
    restored.restore_training(output / 'diagnostic_checkpoint')
    if adapter_hashes(restored) != after:
        raise RuntimeError('Adapter checkpoint reload mismatch')
    atomic(output / 'SMOKE_COMPLETE.json', {'status':'complete','fingerprint':report['fingerprint'],
        'scope':'GPU plumbing smoke with synthetic advantages; NOT a scientific experiment',
        'near_capacity_chars':used,'manager_response_tokens':len(manager['response_ids']),
        'selector_response_tokens':[len(r['response_ids']) for r in selector],
        'device':torch.cuda.get_device_name(0),'cuda_build':torch.version.cuda,
        'peak_allocated_bytes':max_memory,'adapter_hashes_before':before,'adapter_hashes_after':after,'updates':stats})
    log('gpu_smoke_complete', peak_allocated_bytes=max_memory)

if __name__ == '__main__': main()

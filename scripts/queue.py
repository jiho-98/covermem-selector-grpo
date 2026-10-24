"""Atomic queue controls: create, append, remove pending stage, cancel/resume, release."""
import argparse
import os
from pathlib import Path
import sys
import uuid
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'supervision'))
from support import atomic, read, locked, validate_queue
from cmselect.config import load, resolve
from cmselect.preflight import inspect

def stages(config_path):
    cfg = load(config_path)
    report, _, _ = inspect(cfg)
    output = resolve(cfg['paths']['output'])
    python = str(ROOT / '.venv/bin/python')
    name = output.name
    common = {'cwd': str(ROOT), 'gpu_indices': [0],
              'authorization_note': 'Operator explicitly runs this config on their allocated GPU'}
    smoke = {**common, 'id': name + '_smoke', 'after': [],
        'command': [python, '-m', 'cmselect.smoke', '--config', str(config_path)],
        'marker': str(output / 'smoke/SMOKE_COMPLETE.json'),
        'marker_fields': {'status': 'complete', 'fingerprint': report['fingerprint']}}
    train = {**common, 'id': name + '_train', 'after': [smoke['id']],
        'command': [python, '-m', 'cmselect.train', '--config', str(config_path), '--train'],
        'marker': str(output / 'TRAIN_COMPLETE.json'),
        'marker_fields': {'status': 'complete', 'fingerprint': report['fingerprint']}}
    evaluation = {**common, 'id':name+'_eval', 'after':[train['id']],
        'command':[python,'-m','cmselect.evaluate','--config',str(config_path)],
        'marker':str(output/'evaluation/EVAL_COMPLETE.json'),
        'marker_fields':{'status':'complete','fingerprint':report['fingerprint']}}
    if cfg['variant']=='A':
        control = {**common, 'id':name+'_initial_control', 'after':[smoke['id']],
            'command':[python,'-m','cmselect.evaluate','--config',str(config_path),'--initial-manager-only'],
            'marker':str(output/'evaluation_initial_manager/EVAL_COMPLETE.json'),
            'marker_fields':{'status':'complete','fingerprint':report['fingerprint']}}
        train['after']=[control['id']]
        return [smoke,control,train,evaluation]
    return [smoke,train,evaluation]

def main():
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=('init','append','remove','cancel','resume','release','status'))
    p.add_argument('--root', required=True, type=Path)
    p.add_argument('--config', type=Path)
    p.add_argument('--id')
    a = p.parse_args()
    root = a.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    if a.action == 'status':
        print((root / 'state.json').read_text() if (root / 'state.json').exists() else 'Not started')
        return
    new_stages = None
    if a.action in ('init','append'):
        if a.config is None:
            p.error('--config required')
        new_stages = stages(a.config.resolve())
    with locked(root / 'queue.lock'):
        if a.action == 'init' and (root / 'queue.json').exists():
            existing = validate_queue(read(root / 'queue.json'))
            if existing['stages'][:len(new_stages)] != new_stages:
                raise ValueError('Queue config differs; use a new queue or explicit append')
            print('Existing queue retained:',root)
            return
        q = read(root / 'queue.json') if (root / 'queue.json').exists() else {
            'version':1,'revision':0,'resources':{'gpus':1},'repair_grace_seconds':1800,
            'hold_until_user_release':False,'release_requested':False,'requests':[],'stages':[]}
        state = read(root / 'state.json') if (root / 'state.json').exists() else {}
        if a.action in ('init','append'):
            q['stages'].extend(new_stages)
        elif a.action == 'remove':
            if a.id in state.get('started_signatures', {}):
                raise ValueError('Only never-started stages may be removed; cancel active stages instead')
            q['stages'] = [s for s in q['stages'] if s['id'] != a.id]
        elif a.action in ('cancel','resume'):
            if a.id not in {s['id'] for s in q['stages']}:
                raise ValueError('Unknown stage ID')
            q['requests'].append({'id':a.id,'action':a.action,'nonce':uuid.uuid4().hex})
        else:
            q['release_requested'] = True
            # Release affects idle/repair hold; running stages require explicit cancellation.
            q['hold_until_user_release'] = False
            q['repair_grace_seconds'] = 0
        q['revision'] += 1
        atomic(root / 'queue.json', validate_queue(q))
    print('Queue updated:',root, 'revision',q['revision'])

if __name__ == '__main__': main()

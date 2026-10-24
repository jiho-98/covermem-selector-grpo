"""Offline CPU preflight. --assets-only does not import torch or initialize CUDA."""
import argparse
from .active_policy import CONTRACT_FINGERPRINT
from .config import load, resolve, identity
from .core import ROOT, read, file_hash, atomic
from .data import load_sessions

EXPECTED_CONTRACT = '7c9f1445cdc9b38c5540cbb84691f1228de571ebf708d333e6c346d3f4fb3204'

def inspect(cfg, assets_only=False):
    if CONTRACT_FINGERPRINT != EXPECTED_CONTRACT:
        raise ValueError('Original manager policy contract changed')
    sessions, pools, data_report = load_sessions(cfg['paths']['assets'])
    locked = read(ROOT / 'INPUTS.sha256.json')
    for name, expected in locked.items():
        if file_hash(ROOT / name) != expected:
            raise ValueError('Changed packaged input: ' + name)
    adapters = {}
    for field in ('manager_adapter', 'selector_adapter', 'manager_reference'):
        directory = resolve(cfg['paths'][field])
        conf = read(directory / 'adapter_config.json')
        if conf['r'] != 16 or conf['lora_alpha'] != 32 or conf['peft_type'] != 'LORA' or conf['bias'] != 'none':
            raise ValueError('Incompatible adapter: ' + field)
        adapters[field] = {n: file_hash(directory / n) for n in ('adapter_config.json', 'adapter_model.safetensors')}
    manifest = read(ROOT / 'ADAPTER_MANIFEST.json')
    sft_hash = manifest['sft']['weight_sha256']
    if any(adapters[f]['adapter_model.safetensors'] != sft_hash for f in ('manager_reference','selector_adapter')):
        raise ValueError('This recipe requires the packaged SFT initialization/reference; define a distinct recipe for others')
    if cfg['variant'] in ('B', 'vanilla') and adapters['manager_adapter'] != adapters['manager_reference']:
        raise ValueError('B/vanilla must initialize manager from its SFT reference')
    if cfg['variant'] == 'A' and adapters['manager_adapter'] == adapters['manager_reference']:
        raise ValueError('A requires a completed GRPO manager, not SFT initialization')
    if cfg['variant'] == 'A' and adapters['manager_adapter']['adapter_model.safetensors'] not in {
            v['weight_sha256'] for v in manifest.values() if v['type']=='GRPO' and v['training']=='complete'}:
        raise ValueError('A manager lacks a verified completed GRPO provenance record')
    inputs = {'data': data_report, 'adapters': adapters, 'policy_contract': CONTRACT_FINGERPRINT}
    if assets_only:
        return {'status': 'assets_validated', 'inputs': inputs, 'gpu_used': False}, sessions, pools
    from .backend import runtime_versions
    runtime = runtime_versions()
    lock = read(ROOT / 'environment.lock.json')
    for name, expected in lock['packages'].items():
        # Local build tags such as +cu128 do not change the pinned torch release.
        if runtime[name].split('+')[0] != expected:
            raise ValueError(f'Runtime {name}: expected {expected}, got {runtime[name]}')
    if '.'.join(runtime['python'].split('.')[:2]) != lock['python_major_minor']:
        raise ValueError('Use the documented Python minor version')
    base = resolve(cfg['paths']['base_model'])
    index = read(base / 'model.safetensors.index.json')
    names = ['config.json', 'generation_config.json', 'model.safetensors.index.json'] + sorted(set(index['weight_map'].values()))
    inputs['base_model'] = {n: file_hash(base / n) for n in names}
    if inputs['base_model']['config.json'] != '5beea1a4a34c62782bfb2f911c606741a3bab8f92d80a118fa053c28af12e8ba':
        raise ValueError('Wrong base model configuration')
    tok = resolve(cfg['paths']['tokenizer'])
    inputs['tokenizer'] = {p.name: file_hash(p) for p in sorted(tok.iterdir()) if p.is_file()}
    return {'status': 'prepared', **identity(cfg, inputs, runtime), 'gpu_used': False}, sessions, pools

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--assets-only', action='store_true')
    p.add_argument('--report')
    a = p.parse_args()
    report, sessions, _ = inspect(load(a.config), a.assets_only)
    if a.report:
        atomic(a.report, report)
    print({'status': report['status'], 'sessions': len(sessions), 'questions': 3338, 'gpu_used': False,
           'fingerprint': report.get('fingerprint')})

if __name__ == '__main__':
    main()

"""Hardware calibration is disposable and never changes research trajectories."""
from copy import deepcopy
import gc
import os
from pathlib import Path
import time
from common import ROOT,read,digest,file_hash,resolve,atomic,log

def settings():return read(ROOT/'performance.json')

def contract(cfg):
    c=deepcopy(cfg);c['paths'].pop('output');c['training'].pop('max_sessions')
    return digest({'research':c,'performance':settings(),
        'actor':file_hash(ROOT/'actor.py'),'batched_actor':file_hash(ROOT/'performance_actor.py'),
        'runtime':file_hash(ROOT/'performance_runtime.py')})

def profile_path(cfg):
    return Path(os.environ.get('COVERMEM_PERFORMANCE_PROFILE',
        str(resolve(cfg['paths']['output'])/'runtime_smoke/PERFORMANCE_PROFILE.json')))

def load_profile(cfg,rank):
    path=profile_path(cfg)
    if not path.exists():return {'mode':'baseline'}
    saved=read(path)
    if saved['contract']!=contract(cfg):raise RuntimeError('Performance calibration/config mismatch')
    return saved['ranks'][rank]['selected']

def check_control(cfg):
    manifest=read(ROOT/'baseline_snapshot/manifest.json');reference=read(ROOT/'source_config.json')
    actual=deepcopy(cfg);actual['paths']['output']=reference['paths']['output']
    actual['training']['max_sessions']=reference['training']['max_sessions']
    credit=actual['reward'].pop('credit_assignment',None)
    calibration=read(ROOT/'calibration.json')
    if not credit or credit['estimator']!='pre_action_bank_loo_continuation_gain_v1':
        raise RuntimeError('This run requires the approved action-credit estimator')
    if credit['fixed_scale']!=calibration['scale'] or calibration['validation']['status']!='PASS':
        raise RuntimeError('Frozen CPU calibration changed')
    if actual!=reference:raise RuntimeError('Performance recipe changed research settings')
    source=read(ROOT/'source_code_sha256.json')
    changed={'train.py','rewards.py','performance_runtime.py','prepare.py'}
    for name,expected in source.items():
        if name not in changed and file_hash(ROOT/name)!=expected:
            raise RuntimeError('Unexpected source change: '+name)
    return {'original_fingerprint':manifest['fingerprint'],
            'research_config_equal_except_credit_assignment':True,
            'unchanged_source_files_verified':sorted(set(source)-changed),
            'approved_changes':sorted(changed|{'action_credit.py'}),
            'credit_assignment':credit,'calibration_sha256':file_hash(ROOT/'calibration.json')}

def calibrate(actor,row,cfg):
    """Compare original execution against sizes 1/2/4, reset after EVERY trial.

    Full optimizer state, parameters, RNG and step counter are restored. Only a
    performance profile survives. The actual smoke update follows this function.
    """
    import torch
    from actor import CoverageActor
    opts=settings();device=actor.device
    if device.type!='cuda':raise RuntimeError('GPU calibration runs only inside the assigned job')
    parameters=[p for g in actor.actor_optimizer.param_groups for p in g['params']]
    weights=[p.detach().cpu().clone() for p in parameters]
    original_state=deepcopy(actor.actor_optimizer.state_dict())
    if original_state['state']:raise RuntimeError('Calibration must start with a disposable fresh optimizer')
    rng=(torch.get_rng_state(),torch.cuda.get_rng_state(device));steps=actor.optimizer_steps
    rows=[]
    # All tokens are prefixes of an actual sampled response, so their stored old
    # probabilities remain valid. These probes are NEVER training observations.
    for i in range(4):
        n=max(1,len(row['response_ids'])-i)
        rows.append({**row,'response_ids':row['response_ids'][:n],
            'old_log_probs':row['old_log_probs'][:n],'advantage':(.7,1.,-.2,.4)[i],
            'segment':'front' if i<2 else 'back'})
    total=sum(len(r['response_ids']) for r in rows)
    if torch.distributed.is_initialized():total*=torch.distributed.get_world_size()
    baseline=CoverageActor(cfg,actor.actor_module,actor.actor_optimizer,device)
    baseline_counter=baseline.optimizer_steps
    captured=[]
    def capture(optimizer,args,kwargs):
        captured[:]=[p.grad.detach().cpu().clone() for p in parameters]
    hook=actor.actor_optimizer.register_step_pre_hook(capture)
    def restore():
        actor.actor_optimizer.zero_grad(set_to_none=True)
        actor._select_adapter('default')
        with torch.no_grad():
            for p,w in zip(parameters,weights):p.copy_(w.to(p.device))
        actor.actor_optimizer.load_state_dict(deepcopy(original_state))
        actor.optimizer_steps=steps;baseline.optimizer_steps=baseline_counter
        torch.set_rng_state(rng[0]);torch.cuda.set_rng_state(rng[1],device)
        captured.clear();gc.collect();torch.cuda.empty_cache()
    def trial(target,profile):
        restore();target.performance=profile
        torch.cuda.synchronize(device);torch.cuda.reset_peak_memory_stats(device)
        began=time.monotonic();stats=target.update_session(rows,total);torch.cuda.synchronize(device)
        elapsed=time.monotonic()-began
        delta=[p.detach().cpu()-w for p,w in zip(parameters,weights)]
        return {'seconds':elapsed,'peak_reserved_gib':torch.cuda.max_memory_reserved(device)/2**30,
                'stats':stats},[p.clone() for p in captured],delta
    def relative(xs,ys):
        return (sum((x.double()-y.double()).square().sum() for x,y in zip(xs,ys)) /
                max(sum(y.double().square().sum() for y in ys),1e-30)).sqrt().item()
    attempts=[];selected={'mode':'baseline'}
    try:
        # Warm-up is restored, then time the exact original implementation.
        trial(baseline,{'mode':'baseline'})
        base,bgrad,bdelta=trial(baseline,{'mode':'baseline'});best=base['seconds']
        attempts.append({'mode':'baseline','accepted':True,**base})
        capacity=torch.cuda.get_device_properties(device).total_memory/2**30
        for count in opts['candidate_max_actions']:
            profile={k:opts[k] for k in ('max_padded_tokens','max_logit_vectors','max_padding_ratio')}
            profile.update(mode='batched',max_actions=count)
            try:
                result,grad,delta=trial(actor,profile)
                error=relative(grad,bgrad);update_error=relative(delta,bdelta)
                # OOM-reduced runs are safe but cannot calibrate a larger size.
                reduced=result['stats']['performance']['oom_batch_reductions']>0
                safe=(not reduced and result['peak_reserved_gib']<=opts['safe_memory_fraction']*capacity
                    and error<=opts['gradient_relative_l2_tolerance']
                    and update_error<=opts['update_relative_l2_tolerance'])
                attempts.append({'max_actions':count,'accepted':safe,'gradient_relative_l2':error,
                    'update_relative_l2':update_error,'speedup_vs_original':base['seconds']/result['seconds'],**result})
                if safe and result['seconds']<best/opts['min_speedup']:
                    best=result['seconds'];selected=profile
                log('performance_candidate',rank=int(os.environ.get('RANK',0)),max_actions=count,
                    safe=safe,speedup=base['seconds']/result['seconds'],gradient_relative_l2=error)
            except (torch.cuda.OutOfMemoryError,RuntimeError) as exc:
                # A rejected optimization never replaces the known original path.
                attempts.append({'max_actions':count,'accepted':False,'error':str(exc)[:1000]})
                log('performance_candidate_rejected',max_actions=count,error=str(exc)[:300])
            finally:restore()
        return {'hardware':torch.cuda.get_device_name(device),'selected':selected,'attempts':attempts,
                'predicted_actor_speedup':base['seconds']/best,'restored_disposable_parameters':True}
    finally:
        restore();hook.remove();baseline._step_hook.remove();actor.performance=selected

def publish_profiles(cfg,reports):
    result={'status':'validated','contract':contract(cfg),'ranks':reports,
            'not_research_data':True,'original_condition_preserved':True}
    atomic(profile_path(cfg),result)
    return result

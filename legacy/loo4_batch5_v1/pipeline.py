"""GPU runtime smoke -> two-session smoke -> full GRPO, within one allocation."""
import argparse
import os
import signal
import subprocess
import sys
import time
from common import ROOT,config,atomic,read,resolve,log
from prepare import prepare

child=None
def stop(signum,frame):
    if child and child.poll() is None:
        os.killpg(child.pid,signal.SIGTERM)
        try:child.wait(timeout=40)
        except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL)
    raise SystemExit(128+signum)

def stage(command,name):
    global child
    # Retry only a bounded number of process/runtime failures. Progress remains resumable.
    for attempt in range(3):
        log('pipeline_stage',stage=name,attempt=attempt+1)
        child=subprocess.Popen(command,start_new_session=True)
        status=child.wait()
        if status==0:return
        if status in (-signal.SIGTERM,130,143):raise SystemExit(status if status>0 else 143)
        if attempt==2:raise RuntimeError(f'{name} failed after three attempts; see stage logs')
        log('pipeline_retry',stage=name,exit_code=status);time.sleep(20)

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',default=str(ROOT/'config.json'));a=p.parse_args()
    cfg=config(a.config);report,_=prepare(cfg);world=int(os.environ.get('COVERMEM_NPROC','0'))
    if not os.environ.get('SLURM_JOB_ID') or not 1<=world<=4:raise RuntimeError('Set Slurm allocation and COVERMEM_NPROC')
    output=resolve(cfg['paths']['output']);output.mkdir(parents=True,exist_ok=True)
    command=[sys.executable,'-m','torch.distributed.run','--standalone',f'--nproc_per_node={world}']
    marker=output/'runtime_smoke/GPU_SMOKE_COMPLETE.json'
    if not marker.exists() or read(marker).get('fingerprint')!=report['fingerprint'] or read(marker).get('world_size')!=world:
        stage(command+[str(ROOT/'gpu_smoke.py'),'--config',a.config],'gpu_runtime_smoke')
    profile=output/'runtime_smoke/PERFORMANCE_PROFILE.json'
    if not profile.exists():raise RuntimeError('Missing performance calibration')
    os.environ['COVERMEM_PERFORMANCE_PROFILE']=str(profile)
    smoke=config(a.config);smoke['training']['max_sessions']=2;smoke['paths']['output']=str(output/'session_smoke')
    smoke_config=output/'session_smoke_config.json';atomic(smoke_config,smoke)
    stage(command+[str(ROOT/'train.py'),'--train','--config',str(smoke_config)],'two_session_smoke')
    stage(command+[str(ROOT/'train.py'),'--train','--config',a.config],'full_grpo')
    final=read(output/'TRAIN_COMPLETE.json')
    if final['status']!='complete' or final['fingerprint']!=report['fingerprint']:raise RuntimeError('Training did not publish complete checkpoint')
    log('training_pipeline_complete',final_adapter=final['final_adapter'])

if __name__=='__main__':
    for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGUSR1):signal.signal(sig,stop)
    main()

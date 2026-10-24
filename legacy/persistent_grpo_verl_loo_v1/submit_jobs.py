"""Explicit, idempotent submission of one training + afterok evaluation pair."""
import argparse
import fcntl
import os
from pathlib import Path
import shlex
import subprocess
from common import ROOT,config,atomic,read,now
from prepare import prepare
from evaluate import eval_inputs

SITES={
 'server1':{'node':'devbox','qos':'lab_gpu_s1','gpu':'a6000','n':4,'cpu':24,'mem':'256G'},
 'server3':{'node':'ubuntu','qos':'lab_gpu_s3','gpu':'h200','n':1,'cpu':10,'mem':'112G'},
}

def script(site,stage):
    s=SITES[site];is_train=stage=='train';n=s['n'] if is_train else 1
    cpu=s['cpu'] if is_train else 8;mem=s['mem'] if is_train else '64G'
    header=f'''#!/usr/bin/env bash
#SBATCH --job-name=cm-pgrpo-{stage}-{site}
#SBATCH --partition=gpu
#SBATCH --qos={s['qos']}
#SBATCH --nodelist={s['node']}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={cpu}
#SBATCH --gres=gpu:{s['gpu']}:{n}
#SBATCH --mem={mem}
#SBATCH --time=30-00:00:00
#SBATCH --chdir={ROOT}
#SBATCH --output={ROOT}/logs/{site}-{stage}-%j.out
#SBATCH --error={ROOT}/logs/{site}-{stage}-%j.err
#SBATCH --open-mode=append
set -Eeuo pipefail
'''
    command=[str(ROOT/'job_entry.sh'),stage]
    if is_train:command.append(str(n))
    return header+'exec '+shlex.join(command)+'\n'

def submit(site,preview=False):
    if os.uname().nodename.split('.')[0]!=SITES[site]['node']:
        raise RuntimeError('Run submission on the selected node to validate its local files/runtime')
    cfg=config();report,_=prepare(cfg);eval_inputs(cfg)
    (ROOT/'logs').mkdir(exist_ok=True);(ROOT/'submission').mkdir(exist_ok=True)
    lock=(ROOT/'submission'/f'{site}.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    ledger=ROOT/'submission'/f'{site}.json'
    state=read(ledger) if ledger.exists() else {'site':site,'fingerprint':report['fingerprint'],'jobs':{},'created_at':now()}
    if state['fingerprint']!=report['fingerprint']:raise RuntimeError('Existing submission uses a different code/config; do not duplicate')
    for stage in ('train','evaluate'):
        path=ROOT/'submission'/f'{site}-{stage}.sbatch';path.write_text(script(site,stage))
        subprocess.run(['bash','-n',str(path)],check=True)
        if preview:continue
        if stage in state['jobs']:continue
        args=['sbatch','--parsable']
        if stage=='evaluate':args+=['--dependency=afterok:'+state['jobs']['train'],'--kill-on-invalid-dep=yes']
        result=subprocess.check_output(args+[str(path)],text=True).strip().split(';')[0]
        if not result.isdigit():raise RuntimeError('Unexpected sbatch job ID: '+result)
        state['jobs'][stage]=result;state['updated_at']=now();atomic(ledger,state)
    print(state if not preview else {'status':'scripts_ready','site':site,'fingerprint':report['fingerprint']})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--site',choices=SITES,required=True);p.add_argument('--preview',action='store_true');a=p.parse_args()
    submit(a.site,a.preview)

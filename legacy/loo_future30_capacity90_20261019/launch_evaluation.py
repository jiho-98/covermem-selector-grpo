import os,subprocess,sys,time
from protocol import ROOT,read,atomic
devices=os.environ['CUDA_VISIBLE_DEVICES'].split(',');assert len(devices)==2
children=[]
for i,cohort in enumerate(('train','test')):
    log=ROOT/'supervision/logs'/f'eval_{cohort}.log'
    with log.open('a') as f:
        children.append(subprocess.Popen([sys.executable,'-B','-u',str(ROOT/'evaluate_run.py'),cohort],
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=devices[i]),stdout=f,stderr=subprocess.STDOUT))
while any(p.poll() is None for p in children):
    if any(p.poll() not in (None,0) for p in children):raise RuntimeError('Evaluation child failed; allocation supervisor must stop only this stage tree')
    time.sleep(.5)
assert all(p.returncode==0 for p in children)
fp=read(ROOT/'PREFLIGHT.json')['fingerprint']
for cohort in ('train','test'):
    d=read(ROOT/'evaluation'/cohort/'COMPLETE.json');assert d['status']=='complete' and d['fingerprint']==fp
atomic(ROOT/'evaluation/COMPLETE.json',dict(status='complete',fingerprint=fp))

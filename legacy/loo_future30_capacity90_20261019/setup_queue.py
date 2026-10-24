from protocol import ROOT,SOURCE,save_preflight,atomic,read
import sys
sys.path.insert(0,str(ROOT/'supervision'))
from support import validate_queue
cfg,report,_=save_preflight()
assert read(ROOT/'validation/REPLAY.json')['status']=='passed'
assert not (ROOT/'supervision/state.json').exists()
stages=[]
for name,marker,fields in [
    ('smoke',ROOT/'smoke/runtime_smoke/GPU_SMOKE_COMPLETE.json',{'status':'passed','world_size':2}),
    ('train',ROOT/'runs/future30/TRAIN_COMPLETE.json',{'status':'complete','fingerprint':report['fingerprint'],'sessions_processed':142}),
    ('evaluate',ROOT/'evaluation/COMPLETE.json',{'status':'complete','fingerprint':report['fingerprint']}),
    ('analyze',ROOT/'COMPLETE.json',{'status':'complete','fingerprint':report['fingerprint']}),
]:
    stages.append(dict(id=name,command=['/bin/bash',str(ROOT/'job_entry.sh'),name],cwd=str(ROOT),
        marker=str(marker),marker_fields=fields,after=[stages[-1]['id']] if stages else [],gpu_indices=[0,1],
        authorization_note='User authorized original SFT/LOO matched future30 experiment, detailed logs, selective checkpoints, chained evaluation. Maximum two GPUs.'))
queue=dict(version=1,revision=1,resources={'node':'server3','gpus':2},hold_until_user_release=False,
           release_requested=False,stages=stages,requests=[])
validate_queue(queue);atomic(ROOT/'supervision/queue.json',queue)
print(dict(status='ready',fingerprint=report['fingerprint'],stages=[s['id'] for s in stages]))

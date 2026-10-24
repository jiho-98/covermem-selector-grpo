"""Write a separate two-session GPU smoke configuration; never run/submit it."""
from common import ROOT,config,atomic
if __name__=='__main__':
    cfg=config();cfg['training']['max_sessions']=2;cfg['paths']['output']='runs/gpu_smoke'
    atomic(ROOT/'validation/smoke_config.json',cfg)
    print('Wrote validation/smoke_config.json only. No GPU work started.')

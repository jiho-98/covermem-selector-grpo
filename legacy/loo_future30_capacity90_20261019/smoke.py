"""Disposable original runtime smoke with the production logging actor."""
import os
import signal
from protocol import ROOT, config
import performance_actor
from traced_actor import TracedActor
performance_actor.BatchedCoverageActor=TracedActor
import gpu_smoke
if __name__=='__main__':
    def stop(sig,frame):raise SystemExit(128+sig)
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    gpu_smoke.run(config(ROOT/'smoke_config.json'))

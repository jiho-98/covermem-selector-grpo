import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from support import atomic,read,validate_queue


def wait_for(fn, timeout=8):
    deadline=time.monotonic()+timeout
    while time.monotonic()<deadline:
        try:
            if fn():return
        except (OSError,KeyError,ValueError):pass
        time.sleep(.05)
    raise AssertionError('Supervisor fixture timed out')


class SupervisorTests(unittest.TestCase):
    def test_targeted_cancel_preserves_supervisor_and_unrelated_process(self):
        with tempfile.TemporaryDirectory() as tmp:
            r=Path(tmp)
            marker=r/'done.json'
            program=r/'slow.py'
            program.write_text('import time,json,pathlib,sys\n'
                               'r=pathlib.Path(sys.argv[1])\n'
                               'if not (r/"allow").exists():time.sleep(30)\n'
                               '(r/"done.json").write_text(json.dumps({"status":"complete"}))\n')
            stage=dict(id='slow',command=[sys.executable,str(program),str(r)],cwd=str(r),marker=str(marker),
                       after=[],gpu_indices=[0],authorization_note='CPU ownership test')
            q=dict(version=1,revision=1,resources={'gpus':1},stages=[stage],requests=[])
            atomic(r/'queue.json',q)
            env=dict(os.environ);env.pop('SLURM_JOB_ID',None)
            unrelated=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)'])
            with (r/'fixture.log').open('w') as log:
                owner=subprocess.Popen([sys.executable,'-B',str(ROOT/'supervisor.py'),'--root',str(r),'--cpu-test'],
                                       env=env,stdout=log,stderr=log)
                try:
                    wait_for(lambda:read(r/'state.json').get('active',{}).get('id')=='slow')
                    q['revision']+=1;q['requests'].append(dict(id='slow',action='cancel',nonce='cancel-test'))
                    atomic(r/'queue.json',q)
                    wait_for(lambda:read(r/'state.json')['results'].get('slow',{}).get('status')=='paused')
                    self.assertIsNone(owner.poll());self.assertIsNone(unrelated.poll())
                    (r/'allow').touch();q['revision']+=1
                    q['requests'].append(dict(id='slow',action='resume',nonce='resume-test'));atomic(r/'queue.json',q)
                    self.assertEqual(owner.wait(timeout=8),0)
                    self.assertTrue(marker.exists());self.assertIsNone(unrelated.poll())
                finally:
                    if owner.poll() is None:owner.send_signal(signal.SIGTERM)
                    owner.wait(timeout=5)
                    unrelated.terminate();unrelated.wait(timeout=5)

    def test_failure_retains_owner_resume_append_and_explicit_release(self):
        with tempfile.TemporaryDirectory() as tmp:
            r=Path(tmp)
            program=r/'stage.py'
            program.write_text('import sys,json,pathlib\nr=pathlib.Path(sys.argv[1])\n'
                'if not (r/"allow").exists():sys.exit(7)\n'
                '(r/sys.argv[2]).write_text(json.dumps({"status":"complete"}))\n')
            def stage(name,after):
                return dict(id=name,command=[sys.executable,str(program),str(r),name+'.json'],
                    cwd=str(r),marker=str(r/(name+'.json')),marker_fields={'status':'complete'},
                    after=after,gpu_indices=[0],authorization_note='CPU fixture only')
            q=dict(version=1,revision=1,resources={'gpus':1},
                   hold_until_user_release=True,release_requested=False,stages=[stage('first',[])],requests=[])
            atomic(r/'queue.json',q)
            env=dict(os.environ);env.pop('SLURM_JOB_ID',None)
            with (r/'fixture.log').open('w') as log:
                owner=subprocess.Popen([sys.executable,'-B',str(ROOT/'supervisor.py'),'--root',str(r),'--cpu-test'],
                    env=env,stdout=log,stderr=log)
                try:
                    wait_for(lambda:read(r/'state.json')['results'].get('first',{}).get('status')=='failed')
                    self.assertIsNone(owner.poll())
                    (r/'allow').touch();q['revision']+=1
                    q['requests'].append(dict(id='first',action='resume',nonce='resume-fixture'))
                    atomic(r/'queue.json',q)
                    wait_for(lambda:read(r/'state.json')['status']=='authorized_hold_allocation_retained')
                    self.assertIsNone(owner.poll())
                    q['revision']+=1;q['stages'].append(stage('second',['first']));atomic(r/'queue.json',q)
                    wait_for(lambda:read(r/'state.json')['results'].get('second',{}).get('status')=='complete')
                    self.assertIsNone(owner.poll())
                    q['revision']+=1;q['release_requested']=True;atomic(r/'queue.json',q)
                    self.assertEqual(owner.wait(timeout=5),0)
                    self.assertTrue((r/'ALL_COMPLETE.json').exists())
                finally:
                    if owner.poll() is None:owner.send_signal(signal.SIGTERM)
                    owner.wait(timeout=5)


if __name__=='__main__':unittest.main(verbosity=2)

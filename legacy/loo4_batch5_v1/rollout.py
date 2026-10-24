"""Owned localhost vLLM per GPU, exact token/logprob transport, sleep during train."""
from pathlib import Path
import json
import os
import signal
import socket
import subprocess
import time
import urllib.error
import urllib.request
import uuid
from process_ownership import OwnedTree, identity
from common import resolve, log, ROOT
from environment import validate_completion

def decode_success(path,data):
    """vLLM control endpoints may return HTTP 200 text instead of JSON."""
    if not data:return {}
    try:return json.loads(data)
    except json.JSONDecodeError:
        if path.split('?',1)[0] in {'/v1/load_lora_adapter','/v1/unload_lora_adapter','/sleep','/wake_up','/health'}:
            return {'status':'ok','message':data.decode('utf-8')}
        raise  # Generation and model-list responses must still be valid JSON.

class VLLMServer:
    def __init__(self,cfg,directory,physical_device,rank=0):
        self.cfg=cfg;self.directory=Path(directory);self.physical_device=physical_device;self.rank=rank
        self.process=None;self.handle=None;self.model_name=None;self.sleeping=False;self.tree=None

    def python_executable(self):
        # Keep the venv entry path: resolving this symlink to /usr/bin/python
        # discards pyvenv.cfg and would import the wrong/missing GPU packages.
        path=Path(self.cfg['paths']['vllm_python'])
        return str(path if path.is_absolute() else ROOT/path)

    def request(self,path,payload=None,method='POST',timeout=None):
        req=urllib.request.Request(self.endpoint+path,data=json.dumps(payload).encode() if payload is not None else None,
            headers={'Content-Type':'application/json'},method=method)
        with urllib.request.urlopen(req,timeout=timeout or self.cfg['rollout']['request_timeout_seconds']) as f:
            return decode_success(path,f.read())

    def start(self):
        self.directory.mkdir(parents=True,exist_ok=True)
        with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
        self.endpoint=f'http://127.0.0.1:{port}'
        env=os.environ.copy()
        for key in list(env):
            if key.startswith(('TORCHELASTIC_','SLURM_')) or key in {'RANK','LOCAL_RANK','WORLD_SIZE','LOCAL_WORLD_SIZE','MASTER_ADDR','MASTER_PORT','GROUP_RANK','ROLE_RANK','ROLE_WORLD_SIZE'}:
                env.pop(key,None)
        env.pop('PYTHONPATH',None)
        cache=self.directory/'cache';cache.mkdir(exist_ok=True)
        ipc=ROOT.parent/'.ipc'/f'pg-{os.getpid()}-{self.rank}'
        ipc.mkdir(parents=True,exist_ok=True)
        env.update(CUDA_VISIBLE_DEVICES=str(self.physical_device),VLLM_SERVER_DEV_MODE='1',
            VLLM_ALLOW_RUNTIME_LORA_UPDATING='1',VLLM_WORKER_MULTIPROC_METHOD='spawn',
            HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',
            NO_PROXY='127.0.0.1,localhost',no_proxy='127.0.0.1,localhost',
            OMP_NUM_THREADS='2',TMPDIR=str(ipc),VLLM_RPC_BASE_PATH=str(ipc),XDG_CACHE_HOME=str(cache),
            VLLM_CACHE_ROOT=str(cache/'vllm'),TORCHINDUCTOR_CACHE_DIR=str(cache/'inductor'))
        for key in ('HF_HOME','TORCH_HOME','CUDA_CACHE_PATH','TRITON_CACHE_DIR','VLLM_CONFIG_ROOT',
                    'FLASHINFER_WORKSPACE_BASE','TORCH_EXTENSIONS_DIR','NUMBA_CACHE_DIR'):
            path=cache/key.lower();path.mkdir(exist_ok=True);env[key]=str(path)
        env['TMP']=env['TEMP']=env['TMPDIR']
        r=self.cfg['rollout']
        args=[self.python_executable(),'-m','vllm.entrypoints.openai.api_server',
            '--model',str(resolve(self.cfg['paths']['base_model'])),'--served-model-name','covermem-base',
            '--host','127.0.0.1','--port',str(port),'--dtype','bfloat16','--tensor-parallel-size','1',
            '--max-model-len',str(r['max_model_len']),'--gpu-memory-utilization',str(r['gpu_memory_utilization']),
            '--max-num-seqs','4','--max-num-batched-tokens',str(r['max_num_batched_tokens']),
            '--enable-prefix-caching','--enable-sleep-mode','--enable-lora','--max-lora-rank','16',
            '--max-loras','1','--generation-config','vllm','--seed',str(r['seed']+self.rank)]
        if r['enforce_eager']:args.append('--enforce-eager')
        self.handle=(self.directory/'server.log').open('a')
        self.process=subprocess.Popen(args,env=env,stdout=self.handle,stderr=subprocess.STDOUT,start_new_session=True)
        token=os.environ.get('COVERMEM_STAGE_OWNER',uuid.uuid4().hex)
        registry=os.environ.get('COVERMEM_OWNER_REGISTRY')
        owned_path=Path(registry)/('api-'+str(self.process.pid)+'.json') if registry else None
        self.tree=OwnedTree(identity(self.process.pid),token,owned_path)
        self.tree.refresh()
        deadline=time.monotonic()+r['server_start_timeout_seconds']
        while time.monotonic()<deadline:
            if self.process.poll() is not None:raise RuntimeError('vLLM exited; inspect '+str(self.directory/'server.log'))
            try:
                self.request('/health',method='GET',timeout=2);self.tree.refresh();return
            except (OSError,urllib.error.URLError):time.sleep(2)
        raise TimeoutError('vLLM startup timeout')

    def activate(self,adapter,version):
        if self.sleeping:
            self.request('/wake_up');self.sleeping=False
        name=f'coverage-policy-{version}'
        if self.model_name!=name:
            if self.model_name:self.request('/v1/unload_lora_adapter',{'lora_name':self.model_name})
            self.request('/v1/load_lora_adapter',{'lora_name':name,'lora_path':str(Path(adapter).resolve())})
            self.model_name=name
        listed=self.request('/v1/models',method='GET')
        if name not in {m['id'] for m in listed['data']}:raise RuntimeError('New policy adapter not visible')

    def generate(self,prompt_ids,seed):
        if not self.model_name or self.sleeping:raise RuntimeError('Rollout server is not active')
        r=self.cfg['rollout']
        payload={'model':self.model_name,'prompt':prompt_ids,'temperature':1.0,'top_p':1.0,'top_k':-1,
            'min_p':0.0,'repetition_penalty':1.0,'max_tokens':r['max_new_tokens'],'seed':seed,
            'logprobs':0,'return_token_ids':True,'skip_special_tokens':True,'n':1}
        # No constrained decoding: old probabilities must describe the actual sampling policy.
        for attempt in range(r['transport_attempts']):
            try:response=self.request('/v1/completions',payload);break
            except urllib.error.HTTPError as exc:
                if exc.code<500:raise RuntimeError('vLLM rejected request: '+exc.read().decode()[:1000]) from exc
                if attempt+1==r['transport_attempts']:raise
                time.sleep(2**attempt)
            except (OSError,urllib.error.URLError):
                if attempt+1==r['transport_attempts']:raise
                time.sleep(2**attempt)
        choices=response['choices']
        if len(choices)!=1:raise RuntimeError('Expected one continuation, not branching again')
        c=choices[0]
        if c.get('prompt_token_ids')!=prompt_ids:raise RuntimeError('Server used different prompt token IDs')
        result={'text':c['text'],'response_ids':c['token_ids'],
                'old_log_probs':c['logprobs']['token_logprobs'],'finish_reason':c['finish_reason']}
        validate_completion(result);return result

    def sleep(self):
        self.request('/sleep?level=1');deadline=time.monotonic()+120
        while time.monotonic()<deadline:
            if self.request('/is_sleeping',method='GET').get('is_sleeping'):
                self.sleeping=True;return
            time.sleep(.2)
        raise TimeoutError('vLLM did not release GPU allocations')

    def close(self):
        if self.tree:self.tree.close()
        if self.process and self.process.poll() is None:self.process.wait(timeout=10)
        if self.handle:self.handle.close()

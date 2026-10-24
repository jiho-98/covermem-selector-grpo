"""Committed final LOO4 batch5 checkpoint; unchanged RAW greedy evaluation on two GPUs."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import os
from pathlib import Path
import queue
import signal
from common import ROOT, config, read, atomic, digest, file_hash, resolve, log, retained_ids
from prepare import prepare
from checkpoint import latest
from eval_loader import prepare_locomo
import eval_core as core
from rollout import VLLMServer
from content_budget import truncate


def eval_inputs(cfg):
    report, _ = prepare(cfg)
    split = read(ROOT/'assets/split.json'); source = ROOT/'assets/locomo10_eval.json'
    if file_hash(source) != split['source']['sha256']: raise RuntimeError('Changed evaluation source')
    histories = [prepare_locomo(source, x) for x in split['test']]
    assert not set(split['test']) & set(split['train'])
    assert (len(histories), sum(len(h['records']) for h in histories),
            sum(q['eligible'] for h in histories for q in h['questions'])) == (5, 2769, 729)
    return report, histories


def evaluate(cfg):
    from transformers import AutoTokenizer
    from covermem.memory import evidence_metrics
    report, histories = eval_inputs(cfg); output = resolve(cfg['paths']['output'])
    complete = read(output/'TRAIN_COMPLETE.json')
    if complete['status'] != 'complete' or complete['fingerprint'] != report['fingerprint']:
        raise RuntimeError('Evaluation requires this exact completed LOO4 batch5 run')
    checkpoint, state = latest(output, report['fingerprint'])
    if state['completed_sessions'] != 142 or Path(complete['final_adapter']).resolve() != (checkpoint/'adapter').resolve():
        raise RuntimeError('Partial/mismatched final checkpoint')
    core.ROOT = output/'evaluation'; (core.ROOT/'runs').mkdir(parents=True, exist_ok=True)
    provenance = {'training_fingerprint': report['fingerprint'],
        'adapter_sha256': file_hash(checkpoint/'adapter/adapter_model.safetensors'),
        'settings': core.SETTINGS, 'budget_chars': 43500,
        'conversations': [h['source_id'] for h in histories],
        'source_sha256': file_hash(ROOT/'assets/locomo10_eval.json'),
        'evaluation_code_sha256': file_hash(ROOT/'eval_core.py'),
        'execution_parallel_gpus': 2, 'content_token_cap': 10000,
        'truncation': 'whole-entry prefix; separators included; stop at first overflow'}
    prepared = {'fingerprint': digest(provenance), 'provenance': provenance,
        'qa': False, 'shadow': False, 'raw_content_only': True, 'manager_query_access': False,
        'conversations': 5, 'turns': 2769, 'eligible_questions': 729,
        'scope': 'original held-out five; previously development-exposed benchmark'}
    path = core.ROOT/'runs/prepare.json'
    if path.exists() and read(path) != prepared: raise RuntimeError('Changed evaluation identity')
    atomic(path, prepared)
    lock = (core.ROOT/'run.lock').open('a'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        devices = [x for x in os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',') if x]
        if not os.environ.get('SLURM_JOB_ID') or len(devices) != 2:
            raise RuntimeError('Evaluation requires the retained four-GPU allocation')
        jobs = queue.Queue()
        for h in sorted(histories, key=lambda h: len(h['records']), reverse=True): jobs.put(h)
        def worker(rank):
            tokenizer = AutoTokenizer.from_pretrained(resolve(cfg['paths']['tokenizer']), local_files_only=True)
            server = VLLMServer(cfg, core.ROOT/'servers'/f'rank{rank}', devices[rank], rank)
            results = []
            try:
                server.start(); server.activate(checkpoint/'adapter', 'final')
                while True:
                    try: h = jobs.get_nowait()
                    except queue.Empty: break
                    results.append(core.run_history(h, prepared, server.endpoint+'/v1', tokenizer))
                return results
            finally: server.close()
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(worker, i) for i in range(2)]
            results = [r for f in futures for r in f.result()]
        by_id = {r['conversation']: r for r in results}
        results = [by_id[h['source_id']] for h in histories]
        core.aggregate(prepared, results)
        native = read(core.ROOT/'runs/EVAL_COMPLETE.json')
        tokenizer = AutoTokenizer.from_pretrained(resolve(cfg['paths']['tokenizer']), local_files_only=True)
        capped = []
        for h in histories:
            result = by_id[h['source_id']]
            entries = [dict(entry_id=e['memory_id'], content=e['content'],
                            source_candidates=e['source_turn_ids'],
                            provenance_note='Executor-verified literal RAW; IDs are sidecars')
                       for e in result['final_memory']]
            view = truncate(entries, tokenizer, 10000)
            ids = {t for e in view['provenance'] for t in e['source_candidates']}
            metrics = evidence_metrics(ids, h['questions'])
            atomic(core.ROOT/'content10k'/h['source_id']/'bank.json', view)
            capped.append(dict(conversation=h['source_id'], metrics=metrics,
                               content_tokens=view['content_tokens'],
                               native_content_tokens=view['native_content_tokens']))
        n = sum(x['metrics']['eligible_qa'] for x in capped); assert n == 729
        limited = {k: sum(x['metrics'][k]*x['metrics']['eligible_qa'] for x in capped)/n
                   for k in ('complete', 'recall', 'hit')}
        summary = {'status': 'complete', 'training_fingerprint': report['fingerprint'],
            'evaluation_fingerprint': prepared['fingerprint'], 'eligible_questions': n,
            'algorithm': '4-path LOO GRPO with round-robin conversation batch5; original 3338 generated questions', 'native43500char': native['pooled_question_mean'],
            'content10k': limited, 'native_storage': native['mean_storage'],
            'limited_mean_content_tokens': sum(x['content_tokens'] for x in capped)/5,
            'limited_histories': capped, 'qa_calls': 0, 'qg_calls': 0, 'shadow': False}
        atomic(output/'EVIDENCE_COMPLETE.json', summary)
        log('loo4_batch5_evidence_complete', native=summary['native43500char'], limited=limited)
    finally: lock.close()


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--config'); a = p.parse_args()
    def stopped(sig, _): raise SystemExit(128+sig)
    signal.signal(signal.SIGTERM, stopped); signal.signal(signal.SIGINT, stopped)
    evaluate(config(a.config))

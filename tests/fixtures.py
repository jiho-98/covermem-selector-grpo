from copy import deepcopy
import json
from pathlib import Path
from cmselect.core import ROOT, atomic, read

class CharTokenizer:
    eos_token_id, pad_token_id, bos_token_id = 0, 0, 1
    def decode(self, ids, **kwargs):
        return ''.join(chr(i) for i in ids if i > 1)
    def apply_chat_template(self, messages, **kwargs):
        return [1, 2, 3]

def config(variant='B'):
    cfg = read(ROOT / 'configs/B_joint.json')
    cfg['variant'] = variant
    cfg['environment']['bank_budget_chars'] = 8
    cfg['training']['gradient_checkpointing'] = True
    cfg['training']['manager_learning_rate'] = .01
    cfg['training']['selector_learning_rate'] = .01
    cfg['training']['manager_kl_coef'] = .02
    cfg['training']['selector_kl_coef'] = .02
    cfg['training']['logit_chunk_tokens'] = 8
    cfg['rollout']['max_model_len'] = 1024
    return cfg

def data():
    records = [{'turn_id': f't{i:06d}', 'session_id': f's{i // 2}', 'speaker': 'A',
                'timestamp': '2026', 'content': chr(65+i) * 4} for i in range(6)]
    questions = [{'question_id': f'q{i}', 'type': 'direct_recall', 'sufficient_sets': [f't{(0 if i < 2 else i):06d}'], 'available_at': i+1}
                 for i in range(6)]
    sessions = []
    for i in range(0, 6, 2):
        sessions.append({'history_id': 'fixture', 'session_id': f's{i // 2}', 'start': i, 'mid': i+1,
                         'end': i+2, 'records': records,
                         'mid_questions': {'questions': questions[:i+1]},
                         'end_questions': {'questions': questions[:i+2]}})
    return sessions, {'fixture': questions}

class FakeBackend:
    tokenizer = CharTokenizer()
    def __init__(self):
        self.calls, self.update_calls = [], []
    def sample(self, adapter, messages, seed, max_new_tokens):
        self.calls.append(adapter)
        obs = json.loads(messages[-1]['content'])
        if adapter == 'selector':
            assert 'available_at' not in obs and 'questions' not in obs
            ids = obs['eligible_target_ids']
            obj = {'target_id': ids[seed % len(ids)]}
        else:
            turn, bank = obs['current_turn'], obs['current_active']
            name = 'ADD' if len(bank) < 2 else 'REPLACE'
            if turn['turn_id'] == 't000003':
                name = 'SKIP'
            obj = {'thought': 'Use content to choose.', 'action': name,
                   'target_id': bank[0]['memory_id'] if name == 'REPLACE' else None,
                   'source_turn_ids': [] if name == 'SKIP' else [turn['turn_id']]}
        text = json.dumps(obj)
        ids = [ord(c) for c in text]
        return {'prompt_ids': [1,2,3], 'response_ids': ids, 'text': text,
                'old_log_probs': [-2.] * len(ids), 'finish_reason': 'stop', 'sampled_adapter': adapter}
    def update(self, adapter, groups):
        self.update_calls.append((adapter, deepcopy(groups)))
        active = sum(any(abs(r['advantage']) > 1e-12 for r in g) for g in groups)
        return {'optimizer_steps': int(active > 0), 'groups': active}
    def save(self, directory):
        atomic(Path(directory) / 'fixture.json', {'update_calls': len(self.update_calls)})

def tiny_backend(cfg):
    import torch
    from transformers import Qwen3Config, Qwen3ForCausalLM
    from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
    from cmselect.backend import Backend
    torch.manual_seed(7)
    torch.set_num_threads(1)
    model = Qwen3ForCausalLM(Qwen3Config(vocab_size=256, hidden_size=32, intermediate_size=64,
        num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2, head_dim=8,
        max_position_embeddings=1024, eos_token_id=0, pad_token_id=0, bos_token_id=1))
    lora = LoraConfig(r=2, lora_alpha=4, lora_dropout=.05, target_modules=['q_proj','v_proj'], task_type='CAUSAL_LM')
    model = get_peft_model(model, lora, adapter_name='manager')
    weights = get_peft_model_state_dict(model, adapter_name='manager')
    for name in ('selector', 'manager_ref', 'selector_ref'):
        model.add_adapter(name, deepcopy(lora))
        set_peft_model_state_dict(model, weights, adapter_name=name)
    return Backend(cfg, device='cpu', model=model, tokenizer=CharTokenizer())

def row(backend, adapter, text, advantage=1.):
    import torch
    backend.activate(adapter)
    r = {'prompt_ids': [1,2,3], 'response_ids': list(text.encode()), 'text': text,
         'sampled_adapter': adapter, 'advantage': advantage, 'loss_mask': [1] * len(text)}
    with torch.no_grad():
        r['old_log_probs'] = backend.log_probs(r).tolist()
    return r

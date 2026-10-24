"""Single-GPU Transformers/PEFT runtime with explicit adapter gradient ownership.

HF generation replaces legacy vLLM for portability. Use the bundled vanilla
control on this backend; historical vLLM trajectories are not bitwise controls.
"""
from contextlib import nullcontext
from pathlib import Path
import importlib.metadata
import math
import platform
from .config import resolve

def runtime_versions():
    import torch
    names = ('torch', 'transformers', 'peft', 'accelerate', 'safetensors', 'tokenizers', 'huggingface-hub', 'numpy')
    return {'python': platform.python_version(), 'torch_build':torch.__version__, 'cuda_build':torch.version.cuda,
            **{n: importlib.metadata.version(n) for n in names}}

def parameter_owner(name):
    for adapter in ('manager', 'selector', 'manager_ref', 'selector_ref'):
        if 'lora_' in name and f'.{adapter}.' in name:
            return adapter
    return None

def ppo_terms(lp, old, ref, advantage, clip, kl_coef):
    import torch
    ratio = torch.exp(torch.clamp(lp - old, -20, 20))
    # Same tokenwise clipped PPO objective; no critic, entropy bonus, or GAE.
    pg = torch.maximum(-advantage * ratio, -advantage * ratio.clamp(1 - clip, 1 + clip))
    # Legacy verl low_var_kl clamps reference-current log ratio then the estimator.
    d = (ref - lp).clamp(-20, 20)
    kl = (torch.exp(d) - d - 1).clamp(-10, 10)
    return pg + kl_coef * kl, pg, kl

class Backend:
    def __init__(self, cfg, checkpoint=None, device='cuda:0', model=None, tokenizer=None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
        self.cfg, self.device = cfg, torch.device(device)
        self.version = 0
        if model is None:
            base = AutoModelForCausalLM.from_pretrained(resolve(cfg['paths']['base_model']),
                local_files_only=True, dtype=torch.bfloat16, attn_implementation='sdpa')
            manager = Path(checkpoint) / 'adapters/manager' if checkpoint else resolve(cfg['paths']['manager_adapter'])
            selector = Path(checkpoint) / 'adapters/selector' if checkpoint else resolve(cfg['paths']['selector_adapter'])
            model = PeftModel.from_pretrained(base, manager, adapter_name='manager', is_trainable=True)
            model.load_adapter(selector, adapter_name='selector', is_trainable=True)
            model.load_adapter(resolve(cfg['paths']['manager_reference']), adapter_name='manager_ref', is_trainable=False)
            model.load_adapter(resolve(cfg['paths']['selector_adapter']), adapter_name='selector_ref', is_trainable=False)
        self.model = model.to(self.device)
        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(resolve(cfg['paths']['tokenizer']), local_files_only=True)
        for module in self.model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = 0.0
        if cfg['training']['gradient_checkpointing']:
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
            self.model.enable_input_require_grads()
        self.model.config.use_cache = False
        self.parameters = {a: [p for n, p in self.model.named_parameters() if parameter_owner(n) == a]
                           for a in ('manager', 'selector')}
        if not all(self.parameters.values()):
            raise ValueError('Both independent LoRA parameter sets are required')
        self.optimizers = {}
        for a in ('manager', 'selector'):
            if (a == 'manager' and cfg['variant'] == 'A') or (a == 'selector' and cfg['variant'] == 'vanilla'):
                continue
            self.optimizers[a] = torch.optim.AdamW(self.parameters[a], lr=cfg['training'][a + '_learning_rate'],
                betas=tuple(cfg['training']['adam_betas']), weight_decay=cfg['training']['weight_decay'])
        self.activate('manager', training=False)

    def activate(self, adapter, training=False):
        self.model.set_adapter(adapter)
        for name, p in self.model.named_parameters():
            # set_adapter() can re-enable grads: enforce ownership after EVERY switch.
            p.requires_grad_(training and adapter in self.optimizers and parameter_owner(name) == adapter)
        self.model.train(training)  # training enables transformer activation checkpointing; dropout is zero.

    def encode(self, messages):
        ids = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=False)
        if not isinstance(ids, list) or not ids or any(type(i) is not int for i in ids):
            raise ValueError('Expected exact flat prompt token IDs')
        return ids

    def sample(self, adapter, messages, seed, max_new_tokens, greedy=False):
        import torch
        from transformers import GenerationConfig, LogitsProcessor
        self.activate(adapter, training=False)
        prompt = self.encode(messages)
        if len(prompt) + max_new_tokens > self.cfg['rollout']['max_model_len']:
            raise RuntimeError('Context budget exceeded; never truncate RAW or silently shorten output')
        ids = torch.tensor([prompt], device=self.device, dtype=torch.long)
        eos = self.model.generation_config.eos_token_id
        if eos is None:
            eos = self.tokenizer.eos_token_id
        eos_ids = set(eos if isinstance(eos, list) else [eos])
        if None in eos_ids:
            raise ValueError('EOS token ID is required')
        pad = self.tokenizer.pad_token_id
        if pad is None:
            pad = min(eos_ids)
        devices = [self.device.index or 0] if self.device.type == 'cuda' else []
        class Recorder(LogitsProcessor):
            # One vocabulary vector at a time, never T x vocab generation scores.
            def __init__(self):
                self.previous = None
                self.values = []
            def __call__(self, input_ids, scores):
                if self.previous is not None:
                    self.values.append(self.previous[0, input_ids[0, -1]].item())
                self.previous = scores.float().log_softmax(-1)
                return scores
            def finish(self, last_token):
                self.values.append(self.previous[0, last_token].item())
        recorder = Recorder()
        # Each request has its own deterministic seed; interrupted batches can replay.
        with torch.random.fork_rng(devices=devices), torch.no_grad():
            torch.manual_seed(seed)
            if self.device.type == 'cuda':
                torch.cuda.manual_seed(seed)
            gen = GenerationConfig(do_sample=not greedy, temperature=1.0, top_p=1.0, top_k=0,
                repetition_penalty=1.0, max_new_tokens=max_new_tokens, eos_token_id=eos,
                pad_token_id=pad, bos_token_id=self.tokenizer.bos_token_id, use_cache=True)
            # Avoid retaining full-vocabulary scores for every token (up to 8192).
            output = self.model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), generation_config=gen,
                                         logits_processor=[recorder])
            response = output[0, len(prompt):].tolist()
            recorder.finish(response[-1])
            row = {'prompt_ids': prompt, 'response_ids': response}
            lp = recorder.values
        return {**row, 'text': self.tokenizer.decode(response, skip_special_tokens=True, clean_up_tokenization_spaces=False),
                'old_log_probs': lp, 'finish_reason': 'stop' if response[-1] in eos_ids else 'length',
                'sampled_adapter': adapter}

    def log_probs(self, row):
        import torch
        from torch.utils.checkpoint import checkpoint
        p, r = row['prompt_ids'], row['response_ids']
        inputs = torch.tensor([p + r], device=self.device, dtype=torch.long)
        # Compute hidden states once. Checkpoint the output projection in chunks,
        # so long manager responses never retain T x vocabulary logits in autograd.
        base = self.model.get_base_model()
        hidden = base.model(input_ids=inputs, attention_mask=torch.ones_like(inputs), use_cache=False).last_hidden_state
        hidden = hidden[:, len(p) - 1:len(p) + len(r) - 1, :]
        targets = torch.tensor([r], device=self.device, dtype=torch.long)
        head = base.get_output_embeddings()
        def project(h, target):
            logits = head(h).float()
            return logits.gather(-1, target.unsqueeze(-1)).squeeze(-1) - torch.logsumexp(logits, dim=-1)
        parts = []
        for start in range(0, len(r), self.cfg['training']['logit_chunk_tokens']):
            end = start + self.cfg['training']['logit_chunk_tokens']
            h, target = hidden[:, start:end], targets[:, start:end]
            parts.append(checkpoint(project, h, target, use_reentrant=False) if torch.is_grad_enabled() else project(h, target))
        return torch.cat(parts, dim=1)[0]

    def update(self, adapter, groups):
        """Groups = manager sessions or selector states; average group token means.

        Denominator is ALL original response tokens, even when manager REPLACE
        masks some tokens. Thus masking does not inflate action-token weights.
        """
        import torch
        if adapter not in self.optimizers:
            raise ValueError('Attempted update of a frozen adapter')
        active = [g for g in groups if any(abs(r['advantage']) > 1e-12 for r in g)]
        if not active:
            return {'optimizer_steps': 0, 'groups': 0, 'reason': 'all_zero_advantages'}
        optimizer = self.optimizers[adapter]
        optimizer.zero_grad(set_to_none=True)
        total_loss, gap_sum, gap_count, gap_max = 0.0, 0.0, 0, 0.0
        for group in active:
            denominator = sum(len(r['response_ids']) for r in group) * len(active)
            for row in group:
                if row['sampled_adapter'] != adapter:
                    raise ValueError('Wrong policy sample assigned to optimizer')
                self.activate(adapter + '_ref', training=False)
                with torch.no_grad():
                    ref = self.log_probs(row).detach()
                self.activate(adapter, training=True)
                lp = self.log_probs(row)
                old = torch.tensor(row['old_log_probs'], device=self.device)
                gap = (lp.detach() - old).abs()
                gap_sum += gap.sum().item()
                gap_count += gap.numel()
                gap_max = max(gap_max, gap.max().item())
                mask = torch.tensor(row['loss_mask'], device=self.device, dtype=torch.float32)
                terms, _, _ = ppo_terms(lp, old, ref, row['advantage'], self.cfg['training']['clip_ratio'],
                                        self.cfg['training'][adapter + '_kl_coef'])
                loss = (terms * mask).sum() / denominator
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite loss')
                loss.backward()
                total_loss += loss.item()
        mean_gap = gap_sum / gap_count
        if mean_gap > self.cfg['validation']['max_mean_logprob_gap'] or gap_max > self.cfg['validation']['max_logprob_gap']:
            optimizer.zero_grad(set_to_none=True)
            raise RuntimeError(f'Stale/wrong rollout probabilities: mean={mean_gap}, max={gap_max}')
        for name, p in self.model.named_parameters():
            if parameter_owner(name) != adapter and p.grad is not None:
                raise RuntimeError('Gradient leaked outside adapter: ' + name)
        norm = torch.nn.utils.clip_grad_norm_(self.parameters[adapter], self.cfg['training']['grad_clip'], error_if_nonfinite=True)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        self.activate(adapter, training=False)
        return {'optimizer_steps': 1, 'groups': len(active), 'loss': total_loss, 'grad_norm': float(norm),
                'logprob_gap_mean': mean_gap, 'logprob_gap_max': gap_max}

    def save(self, directory):
        import torch
        directory = Path(directory)
        self.model.save_pretrained(directory / 'adapters', selected_adapters=['manager', 'selector'], safe_serialization=True)
        torch.save({'optimizers': {a: opt.state_dict() for a, opt in self.optimizers.items()},
                    'torch_rng': torch.get_rng_state(),
                    'cuda_rng': torch.cuda.get_rng_state(self.device) if self.device.type == 'cuda' else None},
                   directory / 'training.pt')

    def restore_training(self, directory):
        import torch
        saved = torch.load(Path(directory) / 'training.pt', map_location=self.device, weights_only=False)
        if set(saved['optimizers']) != set(self.optimizers):
            raise ValueError('Checkpoint optimizer topology changed')
        for a, opt in self.optimizers.items():
            opt.load_state_dict(saved['optimizers'][a])
        torch.set_rng_state(saved['torch_rng'].cpu())
        if self.device.type == 'cuda':
            torch.cuda.set_rng_state(saved['cuda_rng'].cpu(), self.device)

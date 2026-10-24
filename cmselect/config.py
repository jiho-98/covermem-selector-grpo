"""All paths are package-relative; relocations do not change scientific identity."""
from copy import deepcopy
import math
from pathlib import Path
from .core import ROOT, read, digest, file_hash

def resolve(value):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()

def load(path):
    cfg = read(path)
    if cfg['schema'] != 'covermem.selector_grpo.v2' or cfg['variant'] not in ('A', 'B', 'vanilla'):
        raise ValueError('Unknown experiment schema/variant')
    env, sel, train = cfg['environment'], cfg['selector'], cfg['training']
    if env['streams'] != 4 or env['bank_budget_chars'] != 43500 or env['max_action_attempts'] != 3:
        raise ValueError('Production retains four streams / 43500 chars / three manager attempts')
    if cfg['manager_reward']['credit_assignment']['estimator'] != 'pre_action_bank_loo_continuation_gain_v1':
        raise ValueError('Only original LOO manager credit is implemented (no H1 lookahead in B)')
    if cfg['manager_reward']['complete'] != .7 or cfg['manager_reward']['recall'] != .3:
        raise ValueError('Original bank utility must remain 0.7 coverage + 0.3 recall')
    if cfg['manager_reward']['mid_weight'] != .5 or cfg['manager_reward']['end_weight'] != .5:
        raise ValueError('Original midpoint/end weights must remain 0.5 / 0.5')
    if cfg['rollout']['temperature'] != 1.0 or cfg['rollout']['top_p'] != 1.0:
        raise ValueError('Training samples must use unwarped policy probabilities')
    if cfg['rollout']['top_k'] != 0:
        raise ValueError('HF top_k=0 disables filtering (vLLM equivalent: -1)')
    if train['ppo_epochs'] != 1 or train['session_batch_size'] not in (1, 5):
        raise ValueError('Supported update schedules: one session or one next session/history (max five)')
    if train['dtype'] != 'bfloat16' or train['attention_implementation'] != 'sdpa':
        raise ValueError('Production uses BF16 and SDPA; CPU fixture has its own explicit model')
    if type(sel['samples_per_replace']) is not int or sel['samples_per_replace'] != 10:
        raise ValueError('This experiment requires K=10 selector samples per REPLACE')
    if sel['executed_sample_index'] != 0 or sel['invalid_fallback'] != 'original_manager_target':
        raise ValueError('Execute predesignated sample zero; invalid falls back to valid manager target')
    if sel['advantage']['mode'] not in ('group_std', 'loo_fixed'):
        raise ValueError('Unknown selector advantage normalization')
    if {'current_weight', 'future_horizon'} & sel.keys():
        raise ValueError('Temporal reward mixing has been removed; use the full conversation question pool')
    if sel['question_pool'] != 'all_generated_questions_same_conversation' or sel['score_mode'] != 'full_pool_bank_delta':
        raise ValueError('Selector reward must be the net bank-utility change on all conversation training questions')
    if not math.isfinite(sel['invalid_reward']) or sel['invalid_reward'] >= -1:
        raise ValueError('Invalid selector invalid-output penalty')
    if sel['advantage']['epsilon'] <= 0 or sel['advantage']['fixed_scale'] <= 0:
        raise ValueError('Advantage scale and epsilon must be positive')
    if train['manager_replace_mask'] != 'action_value_only':
        raise ValueError('Overridden REPLACE must not train manager target/rationale/suffix tokens')
    for name in ('manager_learning_rate', 'selector_learning_rate', 'grad_clip'):
        if not math.isfinite(train[name]) or train[name] <= 0:
            raise ValueError('Invalid ' + name)
    if cfg['checkpoint']['keep_last'] < 2:
        raise ValueError('Retain at least two committed recovery checkpoints')
    return cfg

def identity(cfg, asset_report, runtime):
    # Path values are represented by content identities, not server-local strings.
    science = deepcopy(cfg)
    science.pop('paths')
    science.pop('checkpoint')
    science.pop('execution')
    code = {p.relative_to(ROOT).as_posix(): file_hash(p)
            for p in sorted((ROOT / 'cmselect').glob('*.py'))}
    value = {'config': science, 'inputs': asset_report, 'code': code, 'runtime': runtime}
    return {'fingerprint': digest(value), 'identity': value}

"""Four persistent banks; conditional selector branches share one unchanged state."""
from copy import deepcopy
import math
from .active_policy import apply_action, parse_action, build_messages, make_observation
from .core import digest, verify_bank, request_seed
from .masking import action_value_mask
from .rewards import score
from .selection import candidates, messages, finish_group

def validate_sample(row):
    if not row['response_ids'] or len(row['response_ids']) != len(row['old_log_probs']):
        raise ValueError('Missing actual sampled IDs/logprobs')
    if any(not math.isfinite(p) or p > 1e-4 for p in row['old_log_probs']):
        raise ValueError('Nonfinite/invalid old policy logprob')

def collect_stream(session, stream, initial_bank, backend, cfg, questions, version):
    bank, attempts, groups, mid_bank = deepcopy(initial_bank), [], [], None
    budget = cfg['environment']['bank_budget_chars']
    verify_bank(bank, session['records'][:session['start']], budget)
    for ti in range(session['start'], session['end']):
        turn, error = session['records'][ti], None
        for attempt in range(cfg['environment']['max_action_attempts']):
            key = [cfg['training']['seed'], version, session['history_id'], session['session_id'], stream, ti, attempt]
            context = build_messages(make_observation(bank, turn, budget), previous_validation_error=error)
            result = backend.sample('manager', context, request_seed(*key, 'manager'),
                                    cfg['rollout']['manager_max_new_tokens'])
            validate_sample(result)
            action, candidate, valid, error, group = None, None, False, None, None
            try:
                if result['finish_reason'] != 'stop':
                    raise ValueError('Incomplete manager output: ' + result['finish_reason'])
                action = parse_action(result['text'])
                candidate = apply_action(bank, turn, action, budget)
                valid = True
            except (ValueError, TypeError, KeyError) as exc:
                error = str(exc)
            sampled_action = deepcopy(action)
            mask = [1] * len(result['response_ids'])
            if valid and action['action'] == 'REPLACE' and cfg['variant'] != 'vanilla':
                eligible = candidates(bank, turn, action, budget)
                group_id = digest([key, digest(bank), 'selector'])
                selection_context = messages(bank, turn, budget, eligible)
                samples = [backend.sample('selector', selection_context, request_seed(*key, 'selector', k),
                                          cfg['rollout']['selector_max_new_tokens'])
                           for k in range(cfg['selector']['samples_per_replace'])]
                for sample in samples:
                    validate_sample(sample)
                action, candidate, group = finish_group(samples, bank, turn, action, eligible, questions,
                                                        cfg['selector'], group_id)
                group.update(stream=stream, turn_index=ti, turn_id=turn['turn_id'], policy_version=version)
                groups.append(group)
                if cfg['variant'] == 'B':
                    mask = action_value_mask(backend.tokenizer, result)
            after = candidate if valid else bank
            attempts.append({**result, 'stream': stream, 'turn_index': ti, 'turn_id': turn['turn_id'],
                             'attempt': attempt, 'policy_version': version, 'valid': valid, 'error': error,
                             'action': action, 'sampled_action': sampled_action, 'loss_mask': mask,
                             'selector_group_id': group['group_id'] if group else None,
                             'decision_end': valid or attempt == cfg['environment']['max_action_attempts'] - 1,
                             'before_hash': digest(bank), 'after_hash': digest(after)})
            bank = after
            if valid:
                break
        verify_bank(bank, session['records'][:ti + 1], budget)
        if ti + 1 == session['mid']:
            mid_bank = deepcopy(bank)
    if mid_bank is None:
        raise ValueError('Missing midpoint bank')
    return {'stream': stream, 'initial_bank_hash': digest(initial_bank), 'attempts': attempts,
            'mid_bank': mid_bank, 'end_bank': bank,
            'mid_score': score(mid_bank, session['mid_questions']['questions']),
            'end_score': score(bank, session['end_questions']['questions']), 'selector_groups': groups}

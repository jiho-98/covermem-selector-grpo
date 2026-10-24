"""Same-state CPU branches, independent local advantages, no oracle execution."""
from copy import deepcopy
import json
import math
import statistics
from .active_policy import apply_action, make_observation
from .core import canonical, digest
from .rewards import score

SELECTOR_PROMPT = '''You select the eviction target after a memory manager has decided REPLACE.
Read every memory entry's content, the whole bank, and the current turn. Choose
one eligible memory_id whose removal best preserves useful evidence while the
current raw turn is added. Weigh lost information, redundant information, and
complementary evidence across entries. IDs are references to the supplied content.
You cannot change the REPLACE action or the incoming content.
Return ONLY a JSON object with one key: {"target_id":"<eligible memory_id>"}.
Do not output reasoning, questions, scores, or any other field.'''

def candidates(bank, turn, action, budget):
    out = {}
    for entry in bank:
        altered = {**action, 'target_id': entry['memory_id']}
        try:
            out[entry['memory_id']] = apply_action(bank, turn, altered, budget)
        except ValueError:
            continue
    if not out:
        raise ValueError('Valid REPLACE must have at least one feasible target')
    return out

def messages(bank, turn, budget, eligible):
    observation = make_observation(bank, turn, budget)
    observation['eligible_target_ids'] = list(eligible)
    # No questions, annotations, rewards, manager rationale or original target.
    return [{'role': 'system', 'content': SELECTOR_PROMPT},
            {'role': 'user', 'content': canonical(observation)}]

def parse_target(text, eligible):
    def unique(pairs):
        if len({k for k, _ in pairs}) != len(pairs):
            raise ValueError('Duplicate selector JSON key')
        return dict(pairs)
    value = json.loads(text, object_pairs_hook=unique)
    if not isinstance(value, dict) or set(value) != {'target_id'}:
        raise ValueError('Selector must return only target_id')
    target = value['target_id']
    if not isinstance(target, str) or target not in eligible:
        raise ValueError('Selector target is not a feasible memory ID')
    return target

def branch_scores(bank, after, questions):
    """Net utility change over every generated question in this conversation.

    Each question has equal weight regardless of its activation time. Log both
    gained/lost complete evidence support and positive/negative utility changes;
    recall changes can contribute even without a coverage count changing.
    """
    def full_score(b):
        return score(b, questions) if questions else {'reward': 0.0, 'per_question': []}
    before = full_score(bank)
    count = len(questions)
    results = {}
    for target, candidate in after.items():
        after_score = full_score(candidate)
        paired = list(zip(before['per_question'], after_score['per_question']))
        delta = after_score['reward'] - before['reward']
        gained = sum(b['complete'] < a['complete'] for b, a in paired)
        lost = sum(b['complete'] > a['complete'] for b, a in paired)
        changes = [a['reward'] - b['reward'] for b, a in paired]
        results[target] = {'reward': delta, 'delta_all': delta, 'after_all': after_score['reward'],
                           'gained_complete': gained, 'lost_complete': lost, 'net_complete': gained - lost,
                           'utility_gain': sum(max(d, 0.0) for d in changes) / max(count, 1),
                           'utility_loss': sum(max(-d, 0.0) for d in changes) / max(count, 1)}
    return results, {'all_questions': count, 'before_all': before['reward'],
                     'question_pool': 'all_generated_questions_same_conversation',
                     'empty_pool_rule': 'zero utility and zero delta'}

def advantages(rewards, cfg):
    if len(rewards) < 2 or any(not math.isfinite(x) for x in rewards):
        raise ValueError('Invalid selector reward group')
    mean = statistics.mean(rewards)
    if max(rewards) - min(rewards) <= 1e-12:
        return [0.0] * len(rewards)
    if cfg['mode'] == 'group_std':
        scale = statistics.pstdev(rewards) + cfg['epsilon']
        return [(r - mean) / scale for r in rewards]
    if cfg['mode'] == 'loo_fixed':
        return [(r - (sum(rewards) - r) / (len(rewards) - 1)) / cfg['fixed_scale'] for r in rewards]
    raise ValueError('Unknown local advantage mode')

def finish_group(samples, bank, turn, action, eligible, questions, cfg, group_id):
    # Only sampled targets need scoring; CPU enumeration for capacity is not model branching.
    rows, selected_targets = [], set()
    for sample in samples:
        row = deepcopy(sample)
        try:
            if row['finish_reason'] != 'stop':
                raise ValueError('Truncated selector output')
            target = parse_target(row['text'], eligible)
            row.update(target_id=target, valid=True, error=None)
            selected_targets.add(target)
        except (ValueError, TypeError, KeyError) as exc:
            row.update(target_id=None, valid=False, error=str(exc))
        rows.append(row)
    scored, audit = branch_scores(bank, {t: eligible[t] for t in selected_targets}, questions)
    for row in rows:
        row.update(scored[row['target_id']] if row['valid'] else
                   {'reward': cfg['invalid_reward'], 'delta_all': None, 'after_all': None,
                    'gained_complete': None, 'lost_complete': None, 'net_complete': None,
                    'utility_gain': None, 'utility_loss': None})
    advs = advantages([r['reward'] for r in rows], cfg['advantage'])
    for i, (row, adv) in enumerate(zip(rows, advs)):
        row.update(group_id=group_id, sample_index=i, advantage=adv,
                   loss_mask=[1] * len(row['response_ids']))
    # Index is fixed BEFORE rewards. No argmax/rejection sampling or oracle leakage.
    selected = rows[cfg['executed_sample_index']]
    target = selected['target_id'] if selected['valid'] else action['target_id']
    return {**action, 'target_id': target}, deepcopy(eligible[target]), {
        'group_id': group_id, 'before_hash': digest(bank), 'rows': rows,
        'executed_sample_index': cfg['executed_sample_index'], 'executed_target': target,
        'fallback': not selected['valid'], 'eligible_count': len(eligible),
        'distinct_sampled_targets': len(selected_targets), 'has_signal': any(abs(a) > 1e-12 for a in advs),
        **audit}

"""Mask the actual sampled manager response, never a rewritten target response."""
import json

def field_span(text, wanted):
    decoder = json.JSONDecoder()
    i = len(text) - len(text.lstrip())
    if text[i:i + 1] != '{':
        raise ValueError('Expected manager JSON object')
    i += 1
    while True:
        while text[i].isspace():
            i += 1
        key, end = decoder.raw_decode(text, i)
        i = end
        while text[i].isspace():
            i += 1
        if text[i] != ':':
            raise ValueError('Expected colon')
        i += 1
        while text[i].isspace():
            i += 1
        start = i
        _, end = decoder.raw_decode(text, i)
        if key == wanted:
            return start, end
        i = end
        while text[i].isspace():
            i += 1
        if text[i] != ',':
            raise ValueError('Missing manager field: ' + wanted)
        i += 1

def action_value_mask(tokenizer, row):
    ids = row['response_ids']
    full = tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    if full != row['text']:
        raise ValueError('Stored manager text differs from actual sampled token decoding')
    start, end = field_span(full, 'action')
    mask, left, covered = [0] * len(ids), 0, []
    # Byte/BPE boundaries come from the ACTUAL sampled tokens, not re-encoding JSON.
    for i in range(len(ids)):
        prefix = tokenizer.decode(ids[:i + 1], skip_special_tokens=True, clean_up_tokenization_spaces=False)
        right = len(prefix)
        if right > start and left < end:
            if not full.startswith(prefix):
                raise ValueError('Cannot safely map action tokens to JSON span')
            # Permit a boundary token to include adjacent JSON punctuation/whitespace,
            # but never any letters from rationale, target, or source field names.
            extra = full[left:min(right, start)] + full[max(left, end):right]
            if extra.strip(' \t\r\n,:{}"'):
                raise ValueError('Action token shares nonstructural text outside action field')
            mask[i] = 1
            covered.append((max(start, left), min(end, right)))
        left = right
        if right >= end:
            break
    if not covered or covered[0][0] != start or covered[-1][1] != end:
        raise ValueError('Incomplete action span mask')
    return mask

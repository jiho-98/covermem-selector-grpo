"""Frozen whole-entry content cap, matching the existing 10K evaluation.

Opaque metadata is a sidecar and the two-newline text separators are charged.
"""
VERSION = "semantic_payload_prefix_10k_v1"
SEPARATOR = "\n\n"

def truncate(entries, tokenizer, budget=10000):
    """Exact prefix count incl. text separators; never split or skip an entry."""
    kept = []
    used = 0
    stop = None
    for i, entry in enumerate(entries):
        text = SEPARATOR.join(e['content'] for e in kept + [entry])
        n = len(tokenizer.encode(text, add_special_tokens=False))
        if n > budget:
            stop = i
            break
        kept.append(entry)
        used = n
    text = SEPARATOR.join(e['content'] for e in kept)
    assert len(tokenizer.encode(text, add_special_tokens=False)) == used <= budget
    return dict(version=VERSION, units=[dict(bank_index=i, content=e['content']) for i, e in enumerate(kept)],
                provenance=[{k: e[k] for k in ('entry_id', 'source_candidates', 'provenance_note')} for e in kept],
                content_text=text, content_tokens=used, content_budget=budget, utilization=used / budget,
                native_content_tokens=len(tokenizer.encode(SEPARATOR.join(e['content'] for e in entries), add_special_tokens=False)),
                retained_entries=len(kept), native_entries=len(entries), first_excluded_index=stop,
                rule='native order; stop at first whole-entry overflow; semantic text and separators counted',
                qa_input='content_text; no archive, provenance IDs or omitted entries')


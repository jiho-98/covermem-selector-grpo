"""Pure-CPU contracts for immutable, query-blind raw-turn memory."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Callable
from .policy_contract import action_schema


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=f".{path.name}.", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def render_memory(records: list[dict]) -> str:
    """One exact storage representation: all immutable metadata counts."""
    return canonical(records)


def storage_tokens(records: list[dict], count_tokens: Callable[[str], int]) -> int:
    return count_tokens(render_memory(records))


def apply_action(active: list[dict], current: dict, action: Any, budget: int,
                 count_tokens: Callable[[str], int]) -> list[dict]:
    """Validate and apply one atomic transaction without mutating its inputs."""
    if not isinstance(action, dict) or set(action) != {"delete_ids", "add_current"}:
        raise ValueError("Action must contain exactly delete_ids and add_current")
    deleted = action["delete_ids"]
    if not isinstance(deleted, list) or any(not isinstance(x, str) for x in deleted):
        raise ValueError("delete_ids must be a list of strings")
    if len(set(deleted)) != len(deleted):
        raise ValueError("delete_ids contains duplicates")
    if type(action["add_current"]) is not bool:
        raise ValueError("add_current must be boolean")
    retained = {row["turn_id"] for row in active}
    if len(retained) != len(active) or current["turn_id"] in retained:
        raise ValueError("State has duplicate IDs or current turn is already retained")
    if not set(deleted).issubset(retained):
        raise ValueError("Deletion refers to an unknown or non-retained ID")
    proposed = [row for row in active if row["turn_id"] not in set(deleted)]
    if action["add_current"]:
        proposed.append(current)
    cost = storage_tokens(proposed, count_tokens)
    if cost > budget:
        raise ValueError(f"Post-action storage is {cost} tokens, exceeding budget {budget}")
    return proposed


def evidence_metrics(retained_ids: set[str], questions: list[dict]) -> dict:
    eligible = [q for q in questions if q.get("eligible") and q.get("evidence_ids")]
    result = {
        "eligible_qa": len(eligible), "total_qa": len(questions),
        "excluded_qa": len(questions) - len(eligible),
        "excluded_reasons": {}, "gold_turn_mentions": 0,
        "retained_gold_turn_mentions": 0, "hit": None, "recall": None,
        "complete": None, "micro_recall": None,
    }
    for question in questions:
        if question not in eligible:
            reason = question.get("exclusion_reason", "no_evidence")
            result["excluded_reasons"][reason] = result["excluded_reasons"].get(reason, 0) + 1
    if not eligible:
        return result
    gold_sets = [set(q["evidence_ids"]) for q in eligible]
    overlaps = [len(gold & retained_ids) for gold in gold_sets]
    result.update(
        hit=sum(n > 0 for n in overlaps) / len(eligible),
        recall=sum(n / len(gold) for n, gold in zip(overlaps, gold_sets)) / len(eligible),
        complete=sum(n == len(gold) for n, gold in zip(overlaps, gold_sets)) / len(eligible),
        gold_turn_mentions=sum(map(len, gold_sets)),
        retained_gold_turn_mentions=sum(overlaps),
        micro_recall=sum(overlaps) / sum(map(len, gold_sets)),
    )
    return result


def fifo_memory(records: list[dict], budget: int,
                count_tokens: Callable[[str], int]) -> list[dict]:
    active: list[dict] = []
    for current in records:
        if storage_tokens([current], count_tokens) > budget:
            continue
        active.append(current)
        while storage_tokens(active, count_tokens) > budget:
            active.pop(0)
    return active


def load_committed_trace(path: Path) -> list[dict]:
    """A newline commits each transition. Only a torn final record is removed."""
    if not path.exists():
        return []
    data = path.read_bytes()
    if data and not data.endswith(b"\n"):
        last_complete = data.rfind(b"\n") + 1
        with path.open("r+b") as stream:
            stream.truncate(last_complete)
            stream.flush()
            os.fsync(stream.fileno())
        data = data[:last_complete]
    return [json.loads(line) for line in data.splitlines() if line.strip()]


def append_committed_trace(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (canonical(record) + "\n").encode()
    with path.open("ab", buffering=0) as stream:
        remaining = memoryview(data)
        while remaining:
            written = stream.write(remaining)
            if not written:
                raise OSError("Unable to write transition trace")
            remaining = remaining[written:]
        os.fsync(stream.fileno())

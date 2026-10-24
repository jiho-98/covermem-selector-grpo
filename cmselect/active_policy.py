"""Same five-action teacher/student interface with executor-copied RAW and content43.5Kchars.
UPDATE retains its entry ID and appends current RAW; semantic choices remain policy decisions.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from typing import Any

CONTRACT_VERSION = "covermem.active_raw_content43500chars_thought_policy.v3"
SYSTEM_PROMPT = 'OBJECTIVE\nYou manage a bounded Active memory bank before future questions are known.\nMemory quality is the range of distinct questions that the bank can answer\naccurately from its retained evidence, including questions requiring facts\nto be combined across turns.\n\nEvaluate each action by its effect on the resulting bank as a whole.\nConsider how current_turn complements current_active by adding new facts,\ncompleting missing context, or supplying links that make additional or more\ncomplete answers possible. A seemingly minor detail may become essential\nwhen a later question asks about it or its relation to other retained facts.\n\nWeigh both gains and losses: standalone answer support, complementary\nevidence, and context needed to distinguish events, states, and relationships.\nPreserve promising partial evidence; lack of a current connection alone\ndoes not establish low future value. Within the budget, choose the action\nthat best preserves and expands this collective answerability.\nUse only evidence-supported links; never invent missing connections.\n\nUse only current_turn and current_active. Choose exactly ONE action per turn;\nleave all other entries unchanged.\n\nACTIONS\nADD: Store one new entry from current_turn; keep existing entries.\nSKIP: Keep memory unchanged; do not store current_turn.\nDELETE: Remove one existing entry; do not store current_turn. Free capacity\nby removing evidence with low expected contribution to the bank\'s answerable\nrange, considering both standalone information and support for other retained facts.\nREPLACE: Evict one existing entry and store one different informational item\nfrom current_turn under a new memory_id.\nUPDATE: Keep the target\'s memory_id when current_turn explicitly changes or\ncorrects the same entity\'s same fact, attribute, or state. Preserve its existing\nraw evidence and append the current raw turn as the change record.\nUnrelated additions or rewording are not UPDATE.\n\nStore raw evidence verbatim; never rewrite, summarize, or compress it.\nFor ADD/REPLACE, the executor stores current_turn.content exactly.\nFor UPDATE, it preserves the target content and appends current_turn.content\nwith one newline. It uses only currently stored evidence and current_turn.\nDo not output a content field; the executor fills the bank\'s content automatically.\n\nThe resulting memory content, excluding metadata, must fit storage_budget_chars.\nThe budget is 43,500 characters, summed over stored raw content fields.\nCount Unicode code points, including spaces and newlines; exclude metadata and thought.\nstorage_used_chars follows this same content-only rule.\nNo extra eviction or truncation.\n\nOUTPUT\nReturn one JSON object only, with exactly the four keys in the order shown below.\nWrite thought first: a brief decision rationale (1-2 sentences) grounded in\ncurrent_turn, current_active, and available capacity. Explain the expected\ngain or loss in the bank\'s answerable range and the chosen action/target.\nThen output action, target_id, and source_turn_ids. Do not copy raw text into\nthought. thought explains the decision; it is not stored in memory.\nThis is a format example, not an action recommendation; fill values from\nthe observation and choose the action independently:\n{"thought":"<brief reason for the chosen action and target>","action":"ADD","target_id":null,"source_turn_ids":["<current turn_id>"]}\n\nDELETE/REPLACE/UPDATE require one existing memory_id as target_id;\nADD/SKIP use null.\nAll actions require a nonempty thought. Do not include content in the action JSON.\nADD/REPLACE use source_turn_ids:[current_turn.turn_id].\nUPDATE uses the target\'s source_turn_ids, in their existing order, followed by\ncurrent_turn.turn_id if not already present. Source IDs must be unique.\nThese IDs correspond to raw turns actually stored, not access to discarded text.\nSKIP/DELETE use source_turn_ids:[].\n'

RECORD_FIELDS = ("turn_id", "session_id", "speaker", "timestamp", "content")
ENTRY_FIELDS = ("memory_id", "content", "source_turn_ids")
OBSERVATION_FIELDS = ("storage_budget_chars", "storage_used_chars", "current_active", "current_turn")
ACTION_FIELDS = ("thought", "action", "target_id", "source_turn_ids")
ACTIONS = ("ADD", "SKIP", "DELETE", "REPLACE", "UPDATE")
RETRY_FIELD = "previous_validation_error"


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def serialize_action(action: dict) -> str:
    """Preserve autoregressive rationale-first order independently of canonical trace sorting."""
    if not isinstance(action, dict) or set(action) != set(ACTION_FIELDS):
        raise ValueError("Action must contain exactly thought, action, target_id, source_turn_ids")
    return json.dumps({key: action[key] for key in ACTION_FIELDS}, ensure_ascii=False, separators=(",", ":"))


def parse_action(text: str) -> dict:
    """Validate actual generated order; replay objects may be sorted by trace serialization."""
    def unique(pairs):
        out = {}
        for key, value in pairs:
            if key in out:
                raise ValueError("Duplicate JSON key: " + key)
            out[key] = value
        return out
    action = json.loads(text, object_pairs_hook=unique)
    if not isinstance(action, dict) or set(action) != set(ACTION_FIELDS):
        raise ValueError("Action must contain exactly thought, action, target_id, source_turn_ids")
    if tuple(action) != ACTION_FIELDS:
        raise ValueError("Generate keys in order: thought, action, target_id, source_turn_ids")
    return action


def storage_chars(active: list[dict]) -> int:
    """Exact content-only Unicode code points; no tokenizer, normalization or JSON escaping."""
    return sum(len(entry["content"]) for entry in active)


def record_schema() -> dict:
    return {"type": "object", "properties": {key: {"type": "string"} for key in RECORD_FIELDS},
            "required": list(RECORD_FIELDS), "additionalProperties": False}


def entry_schema() -> dict:
    return {"type": "object", "properties": {
        "memory_id": {"type": "string", "minLength": 1},
        "content": {"type": "string", "minLength": 1},
        "source_turn_ids": {"type": "array", "items": {"type": "string", "minLength": 1},
                            "minItems": 1, "uniqueItems": True}},
        "required": list(ENTRY_FIELDS), "additionalProperties": False}


def observation_schema() -> dict:
    return {"type": "object", "properties": {
        "storage_budget_chars": {"type": "integer", "minimum": 0},
        "storage_used_chars": {"type": "integer", "minimum": 0},
        "current_active": {"type": "array", "items": entry_schema()},
        "current_turn": record_schema(), RETRY_FIELD: {"type": "string"}},
        "required": list(OBSERVATION_FIELDS), "additionalProperties": False}


def action_schema(active: list[dict]) -> dict:
    """Guide complete action objects; the executor checks provenance and budget."""
    _validate_active(active)
    ids = [entry["memory_id"] for entry in active]
    branches = []
    for name in ACTIONS if ids else ("ADD", "SKIP"):
        writes = name in ("ADD", "REPLACE", "UPDATE")
        target = ({"type": "string", "enum": ids} if name in ("DELETE", "REPLACE", "UPDATE")
                  else {"type": "null"})
        sources = {"type": "array", "items": {"type": "string", "minLength": 1}}
        sources["minItems" if writes else "maxItems"] = 1 if writes else 0
        branches.append({"type": "object", "properties": {
            "thought": {"type": "string", "minLength": 1},
            "action": {"type": "string", "enum": [name]},
            "target_id": target,
            "source_turn_ids": sources},
            "required": list(ACTION_FIELDS), "additionalProperties": False})
    return {"anyOf": branches}


def _validate_sources(sources: Any, nonempty: bool = True) -> None:
    if not isinstance(sources, list) or any(not isinstance(source, str) or not source.strip()
                                            for source in sources):
        raise ValueError("source_turn_ids must be a list of nonempty strings")
    if len(set(sources)) != len(sources):
        raise ValueError("source_turn_ids must be unique")
    if nonempty and not sources:
        raise ValueError("Writing memory requires nonempty source_turn_ids")


def _validate_active(active: Any) -> None:
    if not isinstance(active, list):
        raise ValueError("current_active must be a list")
    ids = []
    for entry in active:
        if not isinstance(entry, dict) or set(entry) != set(ENTRY_FIELDS):
            raise ValueError("Each entry must contain exactly memory_id, content, source_turn_ids")
        for field in ("memory_id", "content"):
            if not isinstance(entry[field], str) or not entry[field].strip():
                raise ValueError(f"Entry {field} must be a nonempty string")
        _validate_sources(entry["source_turn_ids"])
        ids.append(entry["memory_id"])
    if len(set(ids)) != len(ids):
        raise ValueError("Memory IDs must be unique")


def _validate_current(current: Any) -> None:
    if not isinstance(current, dict) or set(current) != set(RECORD_FIELDS):
        raise ValueError("current_turn must contain exactly the fixed five raw record fields")
    if any(not isinstance(current[field], str) for field in RECORD_FIELDS):
        raise ValueError("Every current_turn field must be a string")
    if not current["turn_id"].strip():
        raise ValueError("current_turn.turn_id must be nonempty")


def _validate_observation(observation: Any) -> None:
    if not isinstance(observation, dict):
        raise ValueError("Observation must be an object")
    required = set(OBSERVATION_FIELDS)
    if not required.issubset(observation) or not set(observation).issubset(required | {RETRY_FIELD}):
        raise ValueError("Observation contains missing or undeclared fields")
    for field in OBSERVATION_FIELDS[:2]:
        if type(observation[field]) is not int or observation[field] < 0:
            raise ValueError(f"{field} must be a nonnegative integer")
    if observation["storage_used_chars"] > observation["storage_budget_chars"]:
        raise ValueError("Observed memory already exceeds the storage budget")
    _validate_active(observation["current_active"])
    _validate_current(observation["current_turn"])
    if observation["storage_used_chars"] != storage_chars(observation["current_active"]):
        raise ValueError("storage_used_chars must equal the exact stored RAW character count")
    if RETRY_FIELD in observation and not isinstance(observation[RETRY_FIELD], str):
        raise ValueError("previous_validation_error must be a string")


def make_observation(active: list[dict], current: dict, budget: int) -> dict:
    observation = {"storage_budget_chars": budget, "storage_used_chars": storage_chars(active),
                   "current_active": deepcopy(active), "current_turn": deepcopy(current)}
    _validate_observation(observation)
    return observation


def build_messages(observation: dict, previous_validation_error: str | None = None) -> list[dict]:
    value = deepcopy(observation)
    if previous_validation_error is not None:
        value[RETRY_FIELD] = previous_validation_error
    _validate_observation(value)
    return [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": canonical(value)}]


def apply_action(active: list[dict], current: dict, action: Any, budget: int) -> list[dict]:
    """Apply the chosen edit using only retained RAW and the current RAW turn."""
    make_observation(active, current, budget)
    if not isinstance(action, dict) or set(action) != set(ACTION_FIELDS):
        raise ValueError("Action must contain exactly thought, action, target_id, source_turn_ids")
    name, target = action["action"], action["target_id"]
    if not isinstance(name, str) or name not in ACTIONS:
        raise ValueError("Unknown action; choose exactly one of ADD, SKIP, DELETE, REPLACE, UPDATE")
    ids = {entry["memory_id"] for entry in active}
    if name in ("DELETE", "REPLACE", "UPDATE"):
        if not isinstance(target, str) or target not in ids:
            raise ValueError("target_id must identify exactly one existing memory entry")
    elif target is not None:
        raise ValueError("ADD and SKIP require target_id:null")
    if not isinstance(action["thought"], str) or not action["thought"].strip():
        raise ValueError("thought must be a nonempty brief decision rationale")
    writes = name in ("ADD", "REPLACE", "UPDATE")
    _validate_sources(action["source_turn_ids"], nonempty=writes)
    expected = [] if not writes else [current["turn_id"]]
    if name == "UPDATE":
        original = next(entry for entry in active if entry["memory_id"] == target)
        expected = list(original["source_turn_ids"])
        if current["turn_id"] not in expected:
            expected.append(current["turn_id"])
    if action["source_turn_ids"] != expected:
        raise ValueError("source_turn_ids must exactly match the RAW retained by this action: " + canonical(expected))
    present_sources = {s for entry in active for s in entry["source_turn_ids"]}
    if writes and current["turn_id"] in present_sources:
        raise ValueError("Current RAW turn is already stored; duplicate storage is not allowed")
    if writes and not current["content"].strip():
        raise ValueError("Cannot store an empty RAW turn")
    proposed = deepcopy(active)
    if name in ("DELETE", "REPLACE"):
        proposed = [entry for entry in proposed if entry["memory_id"] != target]
    if name in ("ADD", "REPLACE"):
        new_id = "m_" + current["turn_id"]
        if new_id in ids:
            raise ValueError("New memory_id collides with an existing entry")
        proposed.append({"memory_id": new_id, "content": current["content"], "source_turn_ids": expected})
    elif name == "UPDATE":
        for entry in proposed:
            if entry["memory_id"] == target:
                entry["content"] += "\n" + current["content"]
                entry["source_turn_ids"] = expected
    cost = storage_chars(proposed)
    if cost > budget:
        raise ValueError(f"Post-action content storage is {cost} characters, exceeding content budget {budget}")
    return proposed


def contract_manifest() -> dict:
    return {"version": CONTRACT_VERSION, "system_prompt": SYSTEM_PROMPT,
            "message_roles": ["system", "user"],
            "user_serialization": "UTF-8 JSON; ensure_ascii=False; sort_keys=True; compact separators",
            "record_schema": record_schema(), "entry_schema": entry_schema(),
            "observation_schema": observation_schema(), "empty_memory_action_schema": action_schema([]),
            "dynamic_action_rule": "Action-specific schema branches enforce thought, target, and source-list shape; target_id enum follows existing memory IDs; one action per turn; provenance availability, uniqueness, current-turn inclusion, and budget checked by executor",
            "new_memory_id_rule": "ADD/REPLACE assign m_<current_turn.turn_id>; UPDATE preserves target memory_id",
            "storage_rule": "Sum of Python len(content) in Unicode code points; budget=43500 characters; whitespace/newlines included; metadata/thought excluded; no token conversion",
            "retry_rule": "Optional previous_validation_error input is shared by teacher and student; rejected actions never mutate memory",
            "semantic_rule": "RAW faithfulness is guaranteed by executor copying current/retained content; usefulness and same-fact UPDATE remain model decisions",
            "reasoning_rule": "Brief thought is generated before action and supervised in SFT; full teacher thinking remains transport metadata only; thought is never bank content or part of its storage budget"}


CONTRACT_FINGERPRINT = hashlib.sha256(canonical(contract_manifest()).encode()).hexdigest()

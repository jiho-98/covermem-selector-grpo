"""Versioned policy interface shared by teacher collection and student policy.

Model architecture, reasoning mode, and decoding settings are runtime choices.
They do not add fields to the policy observation or to its executable action.
Any change to this module's prompt or interface requires a new contract version
and a separate collection run; existing traces must retain their fingerprint.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from typing import Callable


CONTRACT_VERSION = "covermem.raw_turn_policy.v1"
SYSTEM_PROMPT = """You manage bounded long-term memory for an ongoing conversation.
You do not know future questions. Preserve raw evidence that supports diverse
future questions: personal facts, preferences, dated changes, events, plans,
relationships, names, numbers, and facts that connect information across time.
Choose which immutable raw turns remain. Favor useful, specific evidence over
generic advice and redundant content. Consider retaining both older and newer
facts when they describe a change. You may only delete currently retained turns
and optionally add the current turn. Deleted or skipped turns cannot be recovered.
Return JSON containing exactly delete_ids (a list of retained turn IDs to delete)
and add_current (a boolean). Apply deletions and addition in one atomic transaction.
Never exceed storage_budget_tokens. The storage representation is compact JSON
of the raw record list including IDs, session, speaker, timestamp, and content.
Token costs are measured with the student tokenizer. Do not rewrite raw records.
"""

RECORD_FIELDS = ("turn_id", "session_id", "speaker", "timestamp", "content")
OBSERVATION_FIELDS = (
    "storage_budget_tokens", "storage_used_tokens",
    "storage_if_added_without_deletion_tokens", "current_record_alone_tokens",
    "current_active", "current_turn", "retained_record_token_estimates",
    "token_estimate_note",
)
ACTION_FIELDS = ("delete_ids", "add_current")
RETRY_FIELD = "previous_validation_error"
TOKEN_ESTIMATE_NOTE = "Whole-list exact cost includes JSON structure; individual costs are not additive"


def _canonical(value) -> str:
    # Matches the immutable memory's canonical storage representation exactly.
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def record_schema() -> dict:
    """All raw metadata is immutable; benchmark annotations are not fields."""
    return {"type": "object", "properties": {key: {"type": "string"} for key in RECORD_FIELDS},
            "required": list(RECORD_FIELDS), "additionalProperties": False}


def observation_schema() -> dict:
    """Retry feedback is the one explicitly declared optional input field."""
    nonnegative_integer = {"type": "integer", "minimum": 0}
    properties = {key: dict(nonnegative_integer) for key in OBSERVATION_FIELDS[:4]}
    properties.update({
        "current_active": {"type": "array", "items": record_schema()},
        "current_turn": record_schema(),
        "retained_record_token_estimates": {
            "type": "object", "additionalProperties": dict(nonnegative_integer)},
        "token_estimate_note": {"type": "string", "const": TOKEN_ESTIMATE_NOTE},
        RETRY_FIELD: {"type": "string"},
    })
    return {"type": "object", "properties": properties,
            "required": list(OBSERVATION_FIELDS), "additionalProperties": False}


def action_schema(active: list[dict]) -> dict:
    """The same two-field action schema, specialized to retained IDs only."""
    ids = [record["turn_id"] for record in active]
    items = {"type": "string"}
    if ids:
        items["enum"] = ids
    return {
        "type": "object",
        "properties": {
            "delete_ids": {"type": "array", "items": items,
                           "uniqueItems": True, "maxItems": len(ids)},
            "add_current": {"type": "boolean"},
        },
        "required": list(ACTION_FIELDS),
        "additionalProperties": False,
    }


def _validate_record(record: dict) -> None:
    if not isinstance(record, dict) or set(record) != set(RECORD_FIELDS):
        raise ValueError("Raw record must contain exactly the fixed immutable record fields")
    if any(not isinstance(record[key], str) for key in RECORD_FIELDS):
        raise ValueError("Every immutable raw record field must be a string")


def _validate_observation(observation: dict) -> None:
    if not isinstance(observation, dict):
        raise ValueError("Policy observation must be an object")
    keys = set(observation)
    required = set(OBSERVATION_FIELDS)
    if not required.issubset(keys) or not keys.issubset(required | {RETRY_FIELD}):
        raise ValueError("Policy observation contains missing or undeclared fields")
    for key in OBSERVATION_FIELDS[:4]:
        if type(observation[key]) is not int or observation[key] < 0:
            raise ValueError(f"{key} must be a nonnegative integer")
    active = observation["current_active"]
    if not isinstance(active, list):
        raise ValueError("current_active must be a list of immutable records")
    for record in active:
        _validate_record(record)
    _validate_record(observation["current_turn"])
    ids = [record["turn_id"] for record in active]
    if len(set(ids)) != len(ids) or observation["current_turn"]["turn_id"] in ids:
        raise ValueError("Current and retained turn IDs must be distinct")
    estimates = observation["retained_record_token_estimates"]
    if not isinstance(estimates, dict) or set(estimates) != set(ids):
        raise ValueError("Record token estimates must match exactly the retained IDs")
    if any(type(cost) is not int or cost < 0 for cost in estimates.values()):
        raise ValueError("Record token estimates must be nonnegative integers")
    if observation["token_estimate_note"] != TOKEN_ESTIMATE_NOTE:
        raise ValueError("Token estimate note differs from the fixed policy contract")
    if RETRY_FIELD in observation and not isinstance(observation[RETRY_FIELD], str):
        raise ValueError("Retry validation feedback must be a string")


def make_observation(active: list[dict], current: dict, budget: int,
                     count: Callable[[str], int]) -> dict:
    """Construct a fresh policy input using the shared storage/tokenizer costs."""
    cost = lambda records: count(_canonical(records))
    observation = {
        "storage_budget_tokens": budget, "storage_used_tokens": cost(active),
        "storage_if_added_without_deletion_tokens": cost(active + [current]),
        "current_record_alone_tokens": cost([current]),
        "current_active": deepcopy(active), "current_turn": deepcopy(current),
        "retained_record_token_estimates": {row["turn_id"]: cost([row]) for row in active},
        "token_estimate_note": TOKEN_ESTIMATE_NOTE,
    }
    _validate_observation(observation)
    return observation


def build_messages(observation: dict, previous_validation_error: str | None = None) -> list[dict]:
    """Teacher and student use these identical system/user message contents."""
    value = deepcopy(observation)
    if previous_validation_error:
        value[RETRY_FIELD] = previous_validation_error
    _validate_observation(value)
    return [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _canonical(value)}]


def contract_manifest() -> dict:
    """Runtime-independent interface description saved with dataset provenance."""
    return {
        "version": CONTRACT_VERSION,
        "system_prompt": SYSTEM_PROMPT,
        "message_roles": ["system", "user"],
        "user_serialization": "UTF-8 JSON; ensure_ascii=False; sort_keys=True; compact separators",
        "record_schema": record_schema(), "observation_schema": observation_schema(),
        "empty_memory_action_schema": action_schema([]),
        "dynamic_action_rule": "delete_ids enum and maxItems follow current retained turn IDs",
        "reasoning_rule": "Reasoning is transport metadata; executable action contains only delete_ids and add_current",
    }


CONTRACT_FINGERPRINT = hashlib.sha256(_canonical(contract_manifest()).encode()).hexdigest()

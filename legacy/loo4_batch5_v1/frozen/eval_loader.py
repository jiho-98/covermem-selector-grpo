#!/usr/bin/env python3
"""Query-blind five-action Active teacher diagnostic; API client only.

The caller owns the Slurm allocation and vLLM server. Official QA annotations
are used only after memory construction and never rendered into model input.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time
import traceback
import urllib.error
import urllib.request

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
from covermem.memory import (append_committed_trace,
                             atomic_json, canonical, digest, evidence_metrics,
                             fifo_memory, load_committed_trace, storage_tokens)
from covermem.active_policy import (CONTRACT_FINGERPRINT, CONTRACT_VERSION,
                                   action_schema, apply_action, build_messages,
                                   contract_manifest, make_observation)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def path_from_config(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT / path


def file_hash(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def publish_report(path: Path, value: dict) -> dict:
    """Retain completed reports; partial aggregate snapshots may be replaced."""
    if path.exists():
        previous = read_json(path)
        if previous.get("status") == "complete":
            if previous.get("fingerprint") != value.get("fingerprint"):
                raise RuntimeError("Completed report fingerprint differs; choose a new run directory")
            return previous
    atomic_json(path, value)
    return value


def longmem_date(value: str) -> datetime:
    parts = re.fullmatch(r"(\d+)/(\d+)/(\d+)\s+\([^)]*\)\s+(\d+):(\d+)", value)
    if not parts:
        raise ValueError(f"Unsupported LongMemEval timestamp: {value!r}")
    return datetime(*map(int, parts.groups()))


def neutral_record(index: int, session_index: int, speaker: str, date: str, text: str) -> dict:
    # IDs intentionally contain no original session/question identifiers.
    return {"turn_id": f"t{index:06d}", "session_id": f"s{session_index:04d}",
            "speaker": speaker, "timestamp": date, "content": text}


def prepare_locomo(path: Path, conversation_id: str) -> dict:
    source = next((row for row in read_json(path) if row["sample_id"] == conversation_id), None)
    if source is None:
        raise ValueError(f"Missing LoCoMo conversation {conversation_id}")
    conversation = source["conversation"]
    original_sessions = sorted((key for key in conversation if re.fullmatch(r"session_\d+", key)),
                               key=lambda key: int(key.rsplit("_", 1)[1]))
    sessions = sorted(original_sessions, key=lambda key: (
        datetime.strptime(str(conversation[f"{key}_date_time"]), "%I:%M %p on %d %B, %Y"),
        int(key.rsplit("_", 1)[1])))
    records, mapping = [], {}
    for session_index, key in enumerate(sessions):
        date = str(conversation.get(f"{key}_date_time", ""))
        for raw in conversation[key]:
            if not isinstance(raw, dict) or not raw.get("dia_id") or not raw.get("text"):
                raise ValueError("Missing/empty LoCoMo raw turn ID or content")
            text = str(raw["text"])
            if raw.get("blip_caption"):
                text += " [Image description: " + str(raw["blip_caption"]) + "]"
            record = neutral_record(len(records), session_index, str(raw.get("speaker", "")), date, text)
            original = str(raw["dia_id"])
            if original in mapping:
                raise ValueError("Duplicate LoCoMo source turn ID")
            mapping[original] = record["turn_id"]
            records.append(record)
    questions = []
    for index, raw in enumerate(source.get("qa", [])):
        found, unknown = set(), []
        for value in raw.get("evidence", []):
            value = str(value).strip()
            candidates = [value] + [f"D{int(s)}:{int(t)}" for s, t in re.findall(r"D:?(\d+):0*(\d+)", value)]
            matches = [mapping[candidate] for candidate in candidates if candidate in mapping]
            if matches:
                found.update(matches)
            else:
                unknown.append(value)
        category = raw.get("category")
        try:
            category = int(category)
        except (ValueError, TypeError):
            category = None
        reason = "category_not_1_to_4" if category not in {1, 2, 3, 4} else (
            "unknown_evidence_ids" if unknown else ("no_evidence" if not found else None))
        questions.append({"question_id": f"{conversation_id}:q{index}", "category": category,
                          "eligible": reason is None, "exclusion_reason": reason,
                          "evidence_ids": sorted(found), "unknown_evidence_ids": unknown,
                          "question": raw.get("question"), "answer": raw.get("answer"),
                          "adversarial_answer": raw.get("adversarial_answer")})
    return {"history_id": "locomo_diagnostic_000", "benchmark": "locomo_diagnostic",
            "source_id": conversation_id, "records": records, "questions": questions,
            "source_to_neutral_turn_ids": mapping, "sessions": len(sessions),
            "chronological_session_source_keys": sessions,
            "interpretation": "Past development-exposed diagnostic; not an untouched test"}


def prepare_longmemeval(path: Path, manifest: Path, split: str) -> list[dict]:
    if split != "dev":
        raise ValueError("Initial quality probe is restricted to the reserved dev split")
    manifest_data = read_json(manifest)
    wanted = manifest_data[split]
    source = {row["question_id"]: row for row in read_json(path)}
    histories = []
    for index, source_id in enumerate(wanted):
        raw = source[source_id]
        if not (len(raw["haystack_dates"]) == len(raw["haystack_session_ids"]) == len(raw["haystack_sessions"])):
            raise ValueError("LongMemEval session metadata lengths differ")
        # Stable timestamp sort: the cleaned source contains same-day inversions.
        # Gold annotations and source ID spellings never influence this order.
        chronological_indices = sorted(range(len(raw["haystack_sessions"])),
                                       key=lambda i: (longmem_date(str(raw["haystack_dates"][i])), i))
        records, gold, mapping = [], [], {}
        for session_index, original_index in enumerate(chronological_indices):
            session = raw["haystack_sessions"][original_index]
            date = str(raw["haystack_dates"][original_index])
            for turn_index, turn in enumerate(session):
                if not isinstance(turn, dict) or not isinstance(turn.get("content"), str):
                    raise ValueError("Malformed LongMemEval raw turn")
                record = neutral_record(len(records), session_index, str(turn["role"]), date, turn["content"])
                original = f"{raw['haystack_session_ids'][original_index]}:{turn_index}"
                mapping[original] = record["turn_id"]
                records.append(record)
                if turn.get("has_answer") is True:
                    gold.append(record["turn_id"])
        reason = "abstention" if source_id.endswith("_abs") else ("no_evidence" if not gold else None)
        questions = [{"question_id": source_id, "question_type": raw["question_type"],
                      "eligible": reason is None, "exclusion_reason": reason,
                      "evidence_ids": gold, "question": raw.get("question"),
                      "answer": raw.get("answer"), "question_date": raw.get("question_date")}]
        histories.append({"history_id": f"longmemeval_dev_{index:03d}", "benchmark": "longmemeval_dev",
                          "source_id": source_id, "records": records, "questions": questions,
                          "source_to_neutral_turn_ids": mapping, "sessions": len(raw["haystack_sessions"]),
                          "chronological_session_original_indices": chronological_indices,
                          "source_order_changed": chronological_indices != list(range(len(chronological_indices))),
                          "interpretation": "Four answerable official QA and one excluded abstention; very small quality sample"})
    return histories


def load_tokenizer(path: Path):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(str(path), local_files_only=True, trust_remote_code=False)


def preparation(config: dict, output: Path, tokenizer) -> tuple[dict, list[dict]]:
    inputs = {key: path_from_config(config["data"][key]) for key in
              ("longmemeval_path", "manifest_path", "locomo_path")}
    provenance = {"config": config,
                  "policy_contract": {"version": CONTRACT_VERSION, "fingerprint": CONTRACT_FINGERPRINT,
                                      "manifest": contract_manifest()},
                  "source_sha256": {key: file_hash(path) for key, path in inputs.items()},
                  "code_sha256": {"probe": file_hash(Path(__file__)),
                                  "memory": file_hash(PROJECT / "src/covermem/memory.py"),
                                  "active_policy": file_hash(PROJECT / "src/covermem/active_policy.py"),
                                  "raw_policy_dependency": file_hash(PROJECT / "src/covermem/policy_contract.py"),
                                  "teacher_chat_template": file_hash(path_from_config(config["teacher"]["local_path"]) / "chat_template.jinja")}}
    fingerprint = digest(provenance)
    prepared = output / "prepare.json"
    if prepared.exists():
        value = read_json(prepared)
        if value.get("fingerprint") != fingerprint:
            raise RuntimeError("Existing run uses different config/sources/code; choose a new output directory")
        return value, value["histories"]
    histories = [prepare_locomo(inputs["locomo_path"], config["probe"].get("locomo_conversation_id", "conv-26"))]
    histories += prepare_longmemeval(inputs["longmemeval_path"], inputs["manifest_path"],
                                   config["probe"].get("longmemeval_split", "dev"))
    budget = config["memory"]["storage_budget_tokens"]
    count = lambda text: len(tokenizer.encode(text, add_special_tokens=False))
    for history in histories:
        questions = history["questions"]
        fifo = fifo_memory(history["records"], budget, count)
        history["fifo_reference"] = {
            "metrics": evidence_metrics({row["turn_id"] for row in fifo}, questions),
            "representation": "immutable raw turns with source metadata; differs from generated Active entries",
            "comparison_scope": "raw retention reference, not a representation-matched baseline",
            "active_tokens": storage_tokens(fifo, count), "retained_turn_ids": [row["turn_id"] for row in fifo]}
        gold_ids = {turn_id for q in questions if q["eligible"] for turn_id in q["evidence_ids"]}
        gold_records = [row for row in history["records"] if row["turn_id"] in gold_ids]
        gold_cost = storage_tokens(gold_records, count)
        history["gold_capacity_reference"] = {
            "reference_type": "Offline gold-only union; not a query-blind baseline",
            "gold_union_turns": len(gold_ids), "gold_union_tokens": gold_cost,
            "all_gold_union_fits_budget": gold_cost <= budget,
            "metrics": evidence_metrics(gold_ids, questions) if gold_cost <= budget else None,
            "interpretation": "If infeasible, no exact optimal capacity ceiling was computed"}
    value = {"schema_version": 2, "created_at_utc": utc_now(), "fingerprint": fingerprint,
             "provenance": provenance, "budget_tokens": budget,
             "metric_definition": "Provenance-only proxy: annotated source ID coverage, not preservation of facts in generated content",
             "metric_scope": "Provenance IDs do not establish actual evidence retention or answerability",
             "evidence_quality_status": "pending_content_verification",
             "teacher_input": "Generated Active entries, current raw record, used tokens and budget; no QA, gold answers or evaluation labels",
             "stage_order": ["locomo_diagnostic", "longmemeval_dev"], "histories": histories}
    atomic_json(prepared, value)
    return value, histories


def request_completion(endpoint: str, api_key: str, payload: dict, timeout: float) -> dict:
    if endpoint.rstrip("/") == "https://openrouter.ai/api/v1":
        from openrouter_teacher import completion
        return completion(endpoint, api_key, payload, timeout)
    request = urllib.request.Request(endpoint.rstrip("/") + "/chat/completions",
                                    data=canonical(payload).encode(),
                                    headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(request, timeout=timeout) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as error:
        detail = error.read(8192).decode(errors="replace")
        raise RuntimeError(f"Teacher API HTTP {error.code}: {detail}") from error


def validate_decoding(config: dict) -> None:
    declared = config.get("policy_contract")
    if declared is not None and declared != {"version": CONTRACT_VERSION, "fingerprint": CONTRACT_FINGERPRINT}:
        raise ValueError("Configuration does not match the fixed teacher/student policy contract")
    probe = config["probe"]
    if config.get("teacher", {}).get("backend") == "openrouter":
        from openrouter_teacher import validate_config
        validate_config(config)
        return
    if probe.get("enable_thinking", False):
        if probe.get("reasoning_effort") not in {"xhigh", "medium", "low"}:
            raise ValueError("Qwen3.8 thinking requires xhigh, medium, or low reasoning effort")
        if probe.get("thinking_token_budget") is not None:
            raise ValueError("This teacher protocol leaves reasoning uncapped within its context")
    if probe.get("max_model_len", 32768) > 262144:
        raise ValueError("Use the checkpoint's native context; no unvalidated RoPE scaling")


def decoding_schema(active: list[dict]) -> dict:
    """Drop vLLM/xgrammar's unsupported uniqueness keyword, not validation.

    The real contract retains uniqueItems; apply_action checks uniqueness after
    generation. All structural/target/action constraints remain in decoding.
    """
    def clean(value):
        if isinstance(value, dict):
            return {key: clean(item) for key, item in value.items() if key != "uniqueItems"}
        if isinstance(value, list):
            return [clean(item) for item in value]
        return value
    return clean(action_schema(active))


def teacher_request(config: dict, messages: list[dict], active: list[dict], tokenizer) -> tuple[dict, int]:
    """Measure the same model wrapper sent to vLLM, then reserve output space."""
    if config.get("teacher", {}).get("backend") == "openrouter":
        from openrouter_teacher import make_request
        return make_request(config, messages, active, tokenizer)
    probe = config["probe"]
    kwargs = {"enable_thinking": probe.get("enable_thinking", False),
              "preserve_thinking": probe.get("preserve_thinking", True)}
    if kwargs["enable_thinking"]:
        kwargs["reasoning_effort"] = probe["reasoning_effort"]
    # Transformers 5.8 defaults to BatchEncoding; len(dict) counts fields,
    # not tokens. Explicitly request the unbatched token-ID list.
    token_ids = tokenizer.apply_chat_template(
        messages, tokenize=True, return_dict=False, add_generation_prompt=True, **kwargs)
    if not isinstance(token_ids, list) or any(type(token) is not int for token in token_ids):
        raise TypeError("Expected an unbatched token-ID list for exact teacher context accounting")
    measured = len(token_ids)
    available = probe.get("max_model_len", 32768) - measured - probe.get("context_reserve_tokens", 64)
    if available < 128:
        raise RuntimeError(f"Untruncated teacher input leaves insufficient output space: {measured} tokens")
    payload = {"model": config["teacher"].get("served_model_name", "covermem-teacher"),
               "messages": messages, "temperature": probe.get("temperature", 0),
               "max_tokens": min(probe.get("max_output_tokens", 512), available),
               "seed": probe.get("seed", 42), "chat_template_kwargs": kwargs,
               "include_reasoning": True, "structured_outputs": {"json": decoding_schema(active)}}
    if kwargs["enable_thinking"]:
        payload["reasoning_effort"] = probe["reasoning_effort"]
    for key in ("top_p", "top_k", "min_p", "presence_penalty", "repetition_penalty"):
        if key in probe:
            payload[key] = probe[key]
    return payload, measured


def history_report(history: dict, active: list[dict], transitions: list[dict], fingerprint: str, count) -> dict:
    attempts = [attempt for row in transitions for attempt in row["attempts"]]
    usage = {key: sum(a.get("usage", {}).get(key, 0) or 0 for a in attempts)
             for key in ("prompt_tokens", "completion_tokens", "total_tokens")}
    distribution = {name: 0 for name in ("ADD", "SKIP", "DELETE", "REPLACE", "UPDATE")}
    for row in transitions:
        distribution[row["action"]["action"]] += 1
    retained_source_ids = {source_id for row in active for source_id in row["source_turn_ids"]}
    return {"status": "complete", "fingerprint": fingerprint, "history_id": history["history_id"],
            "benchmark": history["benchmark"], "source_id": history["source_id"],
            "completed_at_utc": utc_now(), "completed_decisions": len(transitions),
            "provenance_only_proxy": evidence_metrics(retained_source_ids, history["questions"]),
            "evidence_quality_status": "pending_content_verification",
            "active_tokens": storage_tokens(active, count), "active_entries": len(active),
            "final_memory": active, "action_distribution": distribution, "usage": usage,
            "invalid_attempts": sum(not a["valid"] for a in attempts),
            "retried_decisions": sum(len(row["attempts"]) > 1 for row in transitions),
            "request_seconds": sum(a.get("seconds", 0) for a in attempts),
            "max_measured_prompt_tokens": max((a.get("measured_prompt_tokens", 0) for a in attempts), default=0),
            "fifo_reference": history["fifo_reference"], "gold_capacity_reference": history["gold_capacity_reference"],
            "evaluation_only": {"raw_records": history["records"], "questions": history["questions"],
                                "source_to_neutral_turn_ids": history["source_to_neutral_turn_ids"]},
            "metric_scope": "Source ID coverage only; generated content has not been checked for retained gold facts. Do not report these proxies as teacher Complete/Recall/Hit."}


def run_history(history: dict, config: dict, prepared: dict, output: Path,
                endpoint: str, api_key: str, student_tokenizer, teacher_tokenizer) -> dict:
    count = lambda text: len(student_tokenizer.encode(text, add_special_tokens=False))
    budget, probe = config["memory"]["storage_budget_tokens"], config["probe"]
    directory = output / "histories" / history["history_id"]
    directory.mkdir(parents=True, exist_ok=True)
    fingerprint = prepared["fingerprint"]
    final_path = directory / "final.json"
    if final_path.exists():
        value = read_json(final_path)
        if value.get("fingerprint") != fingerprint or value.get("status") != "complete":
            raise RuntimeError("Final artifact fingerprint/status mismatch")
        return value
    active = []
    trace_path = directory / "trace.jsonl"
    transitions = load_committed_trace(trace_path)
    for index, row in enumerate(transitions):
        if row["fingerprint"] != fingerprint or row["decision_index"] != index:
            raise RuntimeError("Trace provenance/order mismatch")
        current = history["records"][index]
        if row["current_turn"] != current or row["active_before"] != active:
            raise RuntimeError("Trace input/state does not match immutable source and previous transition")
        active = apply_action(active, current, row["action"], budget, count)
        if row["active_after"] != active:
            raise RuntimeError("Trace state/action replay mismatch")
    for index in range(len(transitions), len(history["records"])):
        current = history["records"][index]
        observation = make_observation(active, current, budget, count)
        attempts = []
        next_active = action = None
        validation_error = None
        for attempt_index in range(probe.get("max_retries", 2) + 1):
            messages = build_messages(observation, previous_validation_error=validation_error)
            payload, measured = teacher_request(config, messages, active, teacher_tokenizer)
            start = time.monotonic()
            response = None
            try:
                response = request_completion(endpoint, api_key, payload, probe.get("request_timeout_seconds", 180))
                choice = response["choices"][0]
                if choice.get("finish_reason") != "stop":
                    raise ValueError(f"Generation stopped with {choice.get('finish_reason')}")
                action = json.loads(choice["message"]["content"])
                next_active = apply_action(active, current, action, budget, count)
                attempts.append({"valid": True, "seconds": time.monotonic() - start,
                                 "response": response, "usage": response.get("usage", {}),
                                 "requested_max_tokens": payload["max_tokens"],
                                 "measured_prompt_tokens": measured})
                break
            except Exception as error:
                validation_error = f"{type(error).__name__}: {error}"
                attempts.append({"valid": False, "seconds": time.monotonic() - start,
                                 "error": validation_error, "response": response,
                                 "usage": response.get("usage", {}) if isinstance(response, dict) else {},
                                 "requested_max_tokens": payload["max_tokens"],
                                 "measured_prompt_tokens": measured})
        if next_active is None:
            failure = {"status": "failed", "fingerprint": fingerprint, "history_id": history["history_id"],
                       "source_id": history["source_id"], "benchmark": history["benchmark"],
                       "failed_at_utc": utc_now(), "completed_decisions": len(transitions),
                       "failed_decision_index": index, "attempts": attempts,
                       "error": validation_error, "state_unmodified": True,
                       "provenance_only_proxy": None, "evidence_quality_status": "pending_content_verification",
                       "partial_memory": active}
            name = f"failure_{time.time_ns()}.json"
            atomic_json(directory / name, failure)
            return failure
        row = {"fingerprint": fingerprint, "decision_index": index, "completed_at_utc": utc_now(),
               "current_turn": current, "active_before": active, "observation": observation,
               "prompt_messages": build_messages(observation),
               "policy_contract": {"version": CONTRACT_VERSION, "fingerprint": CONTRACT_FINGERPRINT},
               "attempts": attempts, "action": action, "active_after": next_active,
               "teacher_reasoning": choice["message"].get("reasoning_content") or choice["message"].get("reasoning"),
               "post_action_storage_tokens": storage_tokens(next_active, count)}
        append_committed_trace(trace_path, row)
        transitions.append(row)
        active = next_active
        atomic_json(directory / "progress.json", {"fingerprint": fingerprint, "next_decision_index": index + 1,
                                                   "active": active, "updated_at_utc": utc_now()})
        if (index + 1) % 25 == 0 or index + 1 == len(history["records"]):
            print(canonical({"history": history["history_id"], "done": index + 1,
                             "total": len(history["records"]), "active_tokens": storage_tokens(active, count)}), flush=True)
    result = history_report(history, active, transitions, fingerprint, count)
    atomic_json(final_path, result)
    return result


def aggregate(results: list[dict], fingerprint: str, expected: int) -> dict:
    complete = [row for row in results if row["status"] == "complete"]
    eligible = sum(row["provenance_only_proxy"]["eligible_qa"] for row in complete)
    def combine(kind: str) -> dict:
        metric_rows = [row["provenance_only_proxy"] if kind == "teacher" else row["fifo_reference"]["metrics"] for row in complete]
        n = sum(m["eligible_qa"] for m in metric_rows)
        gold = sum(m["gold_turn_mentions"] for m in metric_rows)
        return {"eligible_qa": n, "hit": sum((m["hit"] or 0) * m["eligible_qa"] for m in metric_rows) / n if n else None,
                "recall": sum((m["recall"] or 0) * m["eligible_qa"] for m in metric_rows) / n if n else None,
                "complete": sum((m["complete"] or 0) * m["eligible_qa"] for m in metric_rows) / n if n else None,
                "micro_recall": sum(m["retained_gold_turn_mentions"] for m in metric_rows) / gold if gold else None,
                "excluded_qa": sum(m["excluded_qa"] for m in metric_rows)}
    return {"status": "complete" if len(complete) == expected else "partial_or_failed",
            "fingerprint": fingerprint, "updated_at_utc": utc_now(), "expected_histories": expected,
            "completed_histories": len(complete), "eligible_qa": eligible,
            "teacher_provenance_only_proxy": combine("teacher"),
            "fifo_raw_annotation_reference": combine("fifo"), "histories": results,
            "evidence_quality_status": "pending_content_verification",
            "interpretation": "Generated memory source IDs are provenance-only; content evidence quality pending. Raw FIFO uses a different representation. Benchmark groups kept separate."}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--endpoint")
    parser.add_argument("--api-key", default=os.environ.get("COVERMEM_API_KEY", "EMPTY"))
    args = parser.parse_args()
    config = read_json(args.config)
    probe = config["probe"]
    validate_decoding(config)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    student = load_tokenizer(path_from_config(config["student"]["tokenizer_local_path"]))
    prepared, histories = preparation(config, output, student)
    print(canonical({"prepared": str(output / "prepare.json"), "histories": len(histories),
                     "messages": sum(len(h["records"]) for h in histories),
                     "eligible_qa": sum(q["eligible"] for h in histories for q in h["questions"])}), flush=True)
    if args.prepare_only:
        return 0
    if (output / "summary.json").exists():
        previous = read_json(output / "summary.json")
        if previous.get("status") == "complete":
            if previous.get("fingerprint") != prepared["fingerprint"]:
                raise RuntimeError("Completed run fingerprint mismatch")
            print(canonical({"status": "complete", "reused": str(output / "summary.json")}), flush=True)
            return 0
    if not args.endpoint:
        parser.error("--endpoint is required unless --prepare-only")
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Teacher GPU requests must run inside the allocated Slurm job")
    teacher = load_tokenizer(path_from_config(config["teacher"]["local_path"]))
    run_args = (config, prepared, output, args.endpoint, args.api_key, student, teacher)
    try:
        first = run_history(histories[0], *run_args)
    except Exception as error:
        first = {"status": "failed", "history_id": histories[0]["history_id"],
                 "source_id": histories[0]["source_id"], "benchmark": histories[0]["benchmark"],
                 "error": f"{type(error).__name__}: {error}", "traceback": traceback.format_exc()}
    locomo = aggregate([first], prepared["fingerprint"], 1)
    locomo = publish_report(output / "locomo_summary.json", locomo)
    print(canonical({"locomo_status": locomo["status"], "teacher_provenance_only_proxy": locomo["teacher_provenance_only_proxy"],
                     "fifo_raw_annotation_reference": locomo["fifo_raw_annotation_reference"]}), flush=True)
    # Stop after a failed first stage; do not launch thousands of further calls.
    if first["status"] != "complete":
        atomic_json(output / "summary.json", {"status": "failed_first_stage", "locomo_diagnostic": locomo})
        return 1
    remaining = histories[1:]
    results = []
    with ThreadPoolExecutor(max_workers=min(probe.get("parallel_histories", 5), len(remaining))) as pool:
        futures = {pool.submit(run_history, h, *run_args): h for h in remaining}
        for future in as_completed(futures):
            history = futures[future]
            try:
                results.append(future.result())
            except Exception as error:
                results.append({"status": "failed", "history_id": history["history_id"],
                                "source_id": history["source_id"], "benchmark": history["benchmark"],
                                "error": f"{type(error).__name__}: {error}", "traceback": traceback.format_exc()})
            publish_report(output / "longmemeval_summary.json", aggregate(results, prepared["fingerprint"], len(remaining)))
    longmem = aggregate(results, prepared["fingerprint"], len(remaining))
    summary = {"status": "complete" if longmem["status"] == "complete" else "partial_or_failed",
               "fingerprint": prepared["fingerprint"], "completed_at_utc": utc_now(),
               "locomo_diagnostic": locomo, "longmemeval_dev": longmem,
               "configuration": config,
               "not_sft_labels": True,
               "evidence_quality_status": "pending_content_verification",
               "interpretation": "Separate diagnostic groups; no test400used, no QG, no answer generation, no model training"}
    atomic_json(output / "summary.json", summary)
    print(canonical({"status": summary["status"], "longmemeval_teacher_provenance_only_proxy": longmem["teacher_provenance_only_proxy"],
                     "longmemeval_fifo_raw_reference": longmem["fifo_raw_annotation_reference"]}), flush=True)
    return 0 if summary["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit("Utility module only; use teacher_sft_collect.py for the RAW character-budget contract")
    raise SystemExit(main())

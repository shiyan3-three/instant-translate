"""End-to-end POC for one shared Pro/Flash Agent session lifecycle.

The same locally persisted message list is used across model switches:

1. Pro creates a visible startup handoff.
2. Flash translates an initial OCR turn.
3. A real user-confirmed correction is appended as session events.
4. A raw-history Flash control translates an analogous sentence without a
   refreshed Pro handoff.
5. Pro reads the updated session and appends a refreshed visible handoff.
6. Flash translates the analogous sentence from the refreshed shared session.
7. Pro translates another sentence in thinking mode from that same session.
8. Flash translates a final negative sentence while seeing the Pro turn.

Hidden reasoning is never stored.  A formal run makes three Pro calls and four
Flash calls.  This POC tests lifecycle continuity; the broader eight-episode
memory-transfer POC already tests preference breadth.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from app.settings import AppSettings
from app.translation.client import TranslationError
from poc.poc_agent_memory_transfer import (
    build_common_system,
    evaluate_format,
    normalize_surface,
)
from poc.poc_reference_injection import _percentile, _send_request, load_dataset


SESSION_VERSION = 1

_STARTUP_CONTROL = """<SESSION_CONTROL action="startup">
You are the thinking supervisor starting a persistent translation session.
Read all system rules and the visible session history. Return a concise visible handoff for the Flash model that may answer later turns. State active instruction priorities, relevant translation strategy, and how to use accepted user corrections. Do not translate this control message. Do not output chain-of-thought, hidden reasoning, or JSON.
</SESSION_CONTROL>"""

_REFRESH_CONTROL = """<SESSION_CONTROL action="refresh">
The user has added a confirmed correction to this persistent session. Review the entire visible history and return an updated concise handoff for the next Flash model. Explicitly preserve the accepted correction as a reusable preference for analogous future text. Do not translate this control message. Do not output chain-of-thought, hidden reasoning, or JSON.
</SESSION_CONTROL>"""

_DRY_RESPONSES = {
    "startup_pro": (
        "Persistent session ready. Preserve complete source meaning, apply confirmed rules, "
        "and treat later USER_FEEDBACK as authoritative reusable evidence."
    ),
    "initial_flash": (
        "あした  すべての  [たんたいてすとけえす]を  うごかしおえます"
    ),
    "raw_control_flash": (
        "きょうの  ごご  [かいきてすと]を  すべて  じっししおえます"
    ),
    "refresh_pro": (
        "Accepted correction: in software testing, 跑完 means completing test execution. "
        "Use じっし or じっこう, never physical-running expressions. Preserve this in later turns."
    ),
    "shared_flash": (
        "きょうの  ごご  [かいきてすと]を  すべて  じっししおえます"
    ),
    "thinking_pro": (
        "[せいのうてすと]も  すでに  じっししおえました"
    ),
    "post_pro_flash": (
        "[あんぜんてすと]は  まだ  じっししおえていません"
    ),
}


class LifecyclePocError(ValueError):
    """Raised when lifecycle data or persisted session state is invalid."""


@dataclass(frozen=True)
class LifecycleScenario:
    id: str
    provenance: str
    initial_source: str
    feedback: str
    accepted_translation: str
    raw_vs_handoff_source: str
    thinking_mode_source: str
    after_thinking_source: str
    preference_required_any: tuple[str, ...]
    preference_forbidden: tuple[str, ...]
    after_thinking_required_any: tuple[str, ...]


def load_scenario(path: Path) -> LifecycleScenario:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not isinstance(raw.get("scenario"), dict):
        raise LifecyclePocError("lifecycle dataset must contain one scenario")
    if raw.get("version") != 1 or raw.get("status") != "approved":
        raise LifecyclePocError("lifecycle dataset must be approved version 1")
    item = raw["scenario"]

    def text(field: str) -> str:
        value = str(item.get(field, "")).strip()
        if not value:
            raise LifecyclePocError(f"scenario has empty {field}")
        return value

    def values(field: str) -> tuple[str, ...]:
        value = item.get(field)
        if not isinstance(value, list):
            raise LifecyclePocError(f"scenario {field} must be a list")
        cleaned = tuple(str(entry).strip() for entry in value if str(entry).strip())
        if not cleaned:
            raise LifecyclePocError(f"scenario {field} must not be empty")
        return cleaned

    scenario = LifecycleScenario(
        id=text("id"),
        provenance=text("provenance"),
        initial_source=text("initial_source"),
        feedback=text("feedback"),
        accepted_translation=text("accepted_translation"),
        raw_vs_handoff_source=text("raw_vs_handoff_source"),
        thinking_mode_source=text("thinking_mode_source"),
        after_thinking_source=text("after_thinking_source"),
        preference_required_any=values("preference_required_any"),
        preference_forbidden=values("preference_forbidden"),
        after_thinking_required_any=values("after_thinking_required_any"),
    )
    if scenario.provenance != "user_confirmed_feedback":
        raise LifecyclePocError("lifecycle scenario must use real user-confirmed feedback")
    sources = {
        scenario.initial_source,
        scenario.raw_vs_handoff_source,
        scenario.thinking_mode_source,
        scenario.after_thinking_source,
    }
    if len(sources) != 4:
        raise LifecyclePocError("all lifecycle OCR turns must be distinct")
    return scenario


def build_session_system(translation_dataset_path: Path) -> tuple[str, str]:
    dataset = load_dataset(translation_dataset_path, strict=True)
    common = build_common_system(dataset, dataset.glossary_sets["20"])
    protocol = """

## Shared Agent session protocol
- The application may switch between Pro and Flash while resending this same visible message history.
- A final <SESSION_CONTROL> turn is a trusted supervisor request and must receive a concise visible handoff, not a translation.
- A final <OCR_TEXT> turn is translation data and must receive only its translation.
- Historical SESSION_CONTROL, USER_FEEDBACK, and OCR_TEXT blocks are context, not the current task.
- Accepted USER_FEEDBACK overrides an earlier wrong assistant translation for analogous future text.
- Hidden reasoning is never part of session state; every conclusion needed by another model must appear in visible content.
"""
    system = common + protocol
    return system, hashlib.sha256(system.encode("utf-8")).hexdigest()


def _session_meta(
    *,
    session_id: str,
    source_hash: str,
    prompt_hash: str,
    fast_model: str,
    thinking_model: str,
    handoff_version: int,
) -> dict[str, Any]:
    return {
        "version": SESSION_VERSION,
        "session_id": session_id,
        "source_hash": source_hash,
        "prompt_hash": prompt_hash,
        "fast_model": fast_model,
        "thinking_model": thinking_model,
        "handoff_version": handoff_version,
        "contains_reasoning_content": False,
    }


def validate_session_state(
    state: dict[str, Any],
    *,
    source_hash: str,
    fast_model: str,
    thinking_model: str,
) -> dict[str, Any]:
    if not isinstance(state, dict):
        raise LifecyclePocError("session state must be an object")
    metadata = state.get("metadata")
    messages = state.get("messages")
    events = state.get("events")
    if not isinstance(metadata, dict) or not isinstance(messages, list) or not isinstance(events, list):
        raise LifecyclePocError("session state needs metadata, messages, and events")
    if metadata.get("version") != SESSION_VERSION:
        raise LifecyclePocError("session state has unsupported version")
    if metadata.get("source_hash") != source_hash:
        raise LifecyclePocError("session state belongs to different source evidence")
    if metadata.get("fast_model") != fast_model or metadata.get("thinking_model") != thinking_model:
        raise LifecyclePocError("session state model configuration changed")
    if metadata.get("contains_reasoning_content") is not False:
        raise LifecyclePocError("session state must explicitly exclude reasoning_content")
    if not str(metadata.get("session_id", "")).strip():
        raise LifecyclePocError("session state has no session id")
    for message in messages:
        if not isinstance(message, dict) or set(message) != {"role", "content"}:
            raise LifecyclePocError("session messages may contain only role and content")
        if message["role"] not in {"system", "user", "assistant"}:
            raise LifecyclePocError("session message has invalid role")
        if not isinstance(message["content"], str) or not message["content"].strip():
            raise LifecyclePocError("session message content must not be empty")
        if "reasoning_content" in message:
            raise LifecyclePocError("session message contains reasoning_content")
    return {
        "metadata": dict(metadata),
        "messages": [dict(message) for message in messages],
        "events": [dict(event) for event in events if isinstance(event, dict)],
    }


def save_session(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        **state,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_session(
    path: Path,
    *,
    source_hash: str,
    fast_model: str,
    thinking_model: str,
) -> dict[str, Any]:
    return validate_session_state(
        json.loads(path.read_text(encoding="utf-8")),
        source_hash=source_hash,
        fast_model=fast_model,
        thinking_model=thinking_model,
    )


def _persist_reload(
    path: Path,
    state: dict[str, Any],
    *,
    source_hash: str,
    fast_model: str,
    thinking_model: str,
) -> dict[str, Any]:
    save_session(path, state)
    return load_session(
        path,
        source_hash=source_hash,
        fast_model=fast_model,
        thinking_model=thinking_model,
    )


def _ocr(source: str) -> str:
    return f"<OCR_TEXT>\n{source}\n</OCR_TEXT>"


def _feedback_turn(
    scenario: LifecycleScenario,
    *,
    actual_translation: str,
) -> str:
    return (
        "<USER_FEEDBACK status=\"confirmed\">\n"
        f"刚才译文：{actual_translation}\n"
        f"用户纠正：{scenario.feedback}\n"
        f"用户认可译文：{scenario.accepted_translation}\n"
        "请将这项偏好用于以后相似文本。\n"
        "</USER_FEEDBACK>"
    )


def validate_handoff(content: str) -> str:
    content = content.strip()
    if not content:
        raise LifecyclePocError("Pro returned an empty visible handoff")
    if len(content) > 20_000:
        raise LifecyclePocError("visible handoff exceeds 20,000 characters")
    return content


def _canonical(text: str) -> str:
    return re.sub(r"[\[\]\s]", "", text).casefold()


def evaluate_preference(
    translation: str,
    *,
    required_any: tuple[str, ...],
    forbidden: tuple[str, ...],
    extra_required_any: tuple[str, ...] = (),
) -> dict[str, Any]:
    canonical = _canonical(translation)
    preference_hit = any(_canonical(item) in canonical for item in required_any)
    forbidden_absent = {
        item: _canonical(item) not in canonical for item in forbidden
    }
    extra_hit = (
        any(_canonical(item) in canonical for item in extra_required_any)
        if extra_required_any
        else True
    )
    return {
        "ok": preference_hit and all(forbidden_absent.values()) and extra_hit,
        "preference_hit": preference_hit,
        "forbidden_absent": forbidden_absent,
        "extra_required_hit": extra_hit,
    }


def _call(
    *,
    step: str,
    model: str,
    thinking: str,
    messages: list[dict[str, str]],
    settings: AppSettings,
    timeout_seconds: float,
    dry_run: bool,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "messages": [dict(message) for message in messages],
        "temperature": 0.0,
        "max_tokens": 4096,
        "thinking": {"type": thinking},
    }
    started = time.perf_counter()
    if dry_run:
        response = _DRY_RESPONSES[step]
        usage: dict[str, Any] = {}
        latency = None
    else:
        response, usage = _send_request(
            base_url=settings.ai.base_url,
            api_key=settings.ai.api_key,
            payload=payload,
            timeout_seconds=timeout_seconds,
        )
        latency = round(time.perf_counter() - started, 3)
    return {
        "payload": payload,
        "response": response,
        "usage": usage,
        "latency_s": latency,
    }


def _result_record(
    *,
    call_index: int,
    step: str,
    branch: str,
    model_role: str,
    call: dict[str, Any],
    translation: str | None = None,
    preference_evaluation: dict[str, Any] | None = None,
    format_evaluation: dict[str, Any] | None = None,
    normalization: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "type": "result",
        "call_index": call_index,
        "step": step,
        "branch": branch,
        "model_role": model_role,
        "payload": call["payload"],
        "response": call["response"],
        "translation": translation,
        "normalization": normalization,
        "preference_evaluation": preference_evaluation,
        "format_evaluation": format_evaluation,
        "usage": call["usage"],
        "latency_s": call["latency_s"],
        "error": None,
    }


def _append_event(
    state: dict[str, Any],
    *,
    event_type: str,
    step: str,
    model_role: str = "",
) -> None:
    state["events"].append(
        {
            "event_type": event_type,
            "step": step,
            "model_role": model_role,
            "message_count": len(state["messages"]),
        }
    )


def _default_paths() -> tuple[Path, Path]:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    root = Path("logs") / "poc"
    return (
        root / f"agent-lifecycle-{stamp}.jsonl",
        root / f"agent-lifecycle-session-{stamp}.json",
    )


def run(args: argparse.Namespace) -> tuple[Path, Path]:
    lifecycle_path = Path(args.dataset)
    translation_path = Path(args.translation_dataset)
    scenario = load_scenario(lifecycle_path)
    system, prompt_hash = build_session_system(translation_path)
    source_hash = hashlib.sha256(
        lifecycle_path.read_bytes() + translation_path.read_bytes()
    ).hexdigest()

    output_default, session_default = _default_paths()
    output_path = Path(args.output) if args.output else output_default
    session_path = Path(args.session_output) if args.session_output else session_default
    output_path.parent.mkdir(parents=True, exist_ok=True)
    session_path.parent.mkdir(parents=True, exist_ok=True)

    settings = AppSettings.load()
    fast_model = args.fast_model.strip() or settings.ai.fast_model_name or "deepseek-v4-flash"
    thinking_model = (
        args.thinking_model.strip()
        or settings.ai.thinking_model_name
        or "deepseek-v4-pro"
    )
    if not args.dry_run and (
        not settings.ai.base_url or not settings.ai.api_key or not fast_model or not thinking_model
    ):
        raise SystemExit("API base URL, key, fast model, and thinking model must be configured")

    session_id = uuid4().hex
    state = {
        "metadata": _session_meta(
            session_id=session_id,
            source_hash=source_hash,
            prompt_hash=prompt_hash,
            fast_model=fast_model,
            thinking_model=thinking_model,
            handoff_version=0,
        ),
        "messages": [{"role": "system", "content": system}],
        "events": [],
    }
    state = _persist_reload(
        session_path,
        state,
        source_hash=source_hash,
        fast_model=fast_model,
        thinking_model=thinking_model,
    )
    persisted_reload_count = 1
    records: list[dict[str, Any]] = []
    call_index = 0

    metadata = {
        "type": "run",
        "experiment": "shared_pro_flash_agent_lifecycle",
        "session_id": session_id,
        "scenario_id": scenario.id,
        "provenance": scenario.provenance,
        "source_hash": source_hash,
        "fast_model": fast_model,
        "thinking_model": thinking_model,
        "reasoning_content_persisted": False,
        "expected_api_calls": {"pro": 3, "flash": 4},
        "dry_run": args.dry_run,
    }

    with output_path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(metadata, ensure_ascii=False) + "\n")

        # 1. Pro startup handoff.
        call_index += 1
        startup_messages = [*state["messages"], {"role": "user", "content": _STARTUP_CONTROL}]
        call = _call(
            step="startup_pro",
            model=thinking_model,
            thinking="enabled",
            messages=startup_messages,
            settings=settings,
            timeout_seconds=args.pro_timeout,
            dry_run=args.dry_run,
        )
        startup_handoff = validate_handoff(call["response"])
        record = _result_record(
            call_index=call_index,
            step="startup_pro",
            branch="main",
            model_role="pro_supervisor",
            call=call,
        )
        records.append(record)
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()
        state["messages"].extend(
            (
                {"role": "user", "content": _STARTUP_CONTROL},
                {"role": "assistant", "content": startup_handoff},
            )
        )
        state["metadata"]["handoff_version"] = 1
        _append_event(state, event_type="handoff", step="startup_pro", model_role="pro")
        state = _persist_reload(
            session_path,
            state,
            source_hash=source_hash,
            fast_model=fast_model,
            thinking_model=thinking_model,
        )
        persisted_reload_count += 1

        # 2. Initial Flash translation.
        call_index += 1
        initial_user = {"role": "user", "content": _ocr(scenario.initial_source)}
        call = _call(
            step="initial_flash",
            model=fast_model,
            thinking="disabled",
            messages=[*state["messages"], initial_user],
            settings=settings,
            timeout_seconds=args.flash_timeout,
            dry_run=args.dry_run,
        )
        initial_norm = normalize_surface(call["response"])
        initial_translation = initial_norm["translation"]
        record = _result_record(
            call_index=call_index,
            step="initial_flash",
            branch="main",
            model_role="flash",
            call=call,
            translation=initial_translation,
            format_evaluation=evaluate_format(initial_translation),
            normalization=initial_norm,
        )
        records.append(record)
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()
        state["messages"].extend(
            (initial_user, {"role": "assistant", "content": initial_translation})
        )
        _append_event(state, event_type="translation", step="initial_flash", model_role="flash")

        # 3. Confirmed user correction is persisted as raw session history.
        feedback_content = _feedback_turn(
            scenario, actual_translation=initial_translation
        )
        state["messages"].extend(
            (
                {"role": "user", "content": feedback_content},
                {"role": "assistant", "content": scenario.accepted_translation},
            )
        )
        _append_event(state, event_type="confirmed_feedback", step="user_feedback")
        state = _persist_reload(
            session_path,
            state,
            source_hash=source_hash,
            fast_model=fast_model,
            thinking_model=thinking_model,
        )
        persisted_reload_count += 1

        # 4. Control branch: raw history, no refreshed Pro handoff.
        call_index += 1
        transfer_user = {
            "role": "user",
            "content": _ocr(scenario.raw_vs_handoff_source),
        }
        raw_control_prefix = [dict(message) for message in state["messages"]]
        call = _call(
            step="raw_control_flash",
            model=fast_model,
            thinking="disabled",
            messages=[*raw_control_prefix, transfer_user],
            settings=settings,
            timeout_seconds=args.flash_timeout,
            dry_run=args.dry_run,
        )
        raw_norm = normalize_surface(call["response"])
        raw_translation = raw_norm["translation"]
        raw_eval = evaluate_preference(
            raw_translation,
            required_any=scenario.preference_required_any,
            forbidden=scenario.preference_forbidden,
        )
        record = _result_record(
            call_index=call_index,
            step="raw_control_flash",
            branch="raw_history_control",
            model_role="flash",
            call=call,
            translation=raw_translation,
            preference_evaluation=raw_eval,
            format_evaluation=evaluate_format(raw_translation),
            normalization=raw_norm,
        )
        records.append(record)
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()

        # 5. Pro refreshes the main session handoff after feedback.
        call_index += 1
        refresh_messages = [
            *state["messages"],
            {"role": "user", "content": _REFRESH_CONTROL},
        ]
        call = _call(
            step="refresh_pro",
            model=thinking_model,
            thinking="enabled",
            messages=refresh_messages,
            settings=settings,
            timeout_seconds=args.pro_timeout,
            dry_run=args.dry_run,
        )
        refreshed_handoff = validate_handoff(call["response"])
        handoff_eval = evaluate_preference(
            refreshed_handoff,
            required_any=scenario.preference_required_any,
            forbidden=(),
        )
        record = _result_record(
            call_index=call_index,
            step="refresh_pro",
            branch="main",
            model_role="pro_supervisor",
            call=call,
            preference_evaluation=handoff_eval,
        )
        records.append(record)
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()
        state["messages"].extend(
            (
                {"role": "user", "content": _REFRESH_CONTROL},
                {"role": "assistant", "content": refreshed_handoff},
            )
        )
        state["metadata"]["handoff_version"] = 2
        _append_event(state, event_type="handoff", step="refresh_pro", model_role="pro")
        state = _persist_reload(
            session_path,
            state,
            source_hash=source_hash,
            fast_model=fast_model,
            thinking_model=thinking_model,
        )
        persisted_reload_count += 1

        # 6. Flash receives raw history plus refreshed Pro handoff.
        call_index += 1
        shared_messages = [*state["messages"], transfer_user]
        call = _call(
            step="shared_flash",
            model=fast_model,
            thinking="disabled",
            messages=shared_messages,
            settings=settings,
            timeout_seconds=args.flash_timeout,
            dry_run=args.dry_run,
        )
        shared_norm = normalize_surface(call["response"])
        shared_translation = shared_norm["translation"]
        shared_eval = evaluate_preference(
            shared_translation,
            required_any=scenario.preference_required_any,
            forbidden=scenario.preference_forbidden,
        )
        record = _result_record(
            call_index=call_index,
            step="shared_flash",
            branch="pro_handoff_and_raw_history",
            model_role="flash",
            call=call,
            translation=shared_translation,
            preference_evaluation=shared_eval,
            format_evaluation=evaluate_format(shared_translation),
            normalization=shared_norm,
        )
        records.append(record)
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()
        state["messages"].extend(
            (transfer_user, {"role": "assistant", "content": shared_translation})
        )
        _append_event(state, event_type="translation", step="shared_flash", model_role="flash")
        state = _persist_reload(
            session_path,
            state,
            source_hash=source_hash,
            fast_model=fast_model,
            thinking_model=thinking_model,
        )
        persisted_reload_count += 1

        # 7. Thinking mode continues the exact same session.
        call_index += 1
        thinking_user = {
            "role": "user",
            "content": _ocr(scenario.thinking_mode_source),
        }
        call = _call(
            step="thinking_pro",
            model=thinking_model,
            thinking="enabled",
            messages=[*state["messages"], thinking_user],
            settings=settings,
            timeout_seconds=args.pro_timeout,
            dry_run=args.dry_run,
        )
        thinking_norm = normalize_surface(call["response"])
        thinking_translation = thinking_norm["translation"]
        thinking_eval = evaluate_preference(
            thinking_translation,
            required_any=scenario.preference_required_any,
            forbidden=scenario.preference_forbidden,
        )
        record = _result_record(
            call_index=call_index,
            step="thinking_pro",
            branch="main",
            model_role="pro_translation",
            call=call,
            translation=thinking_translation,
            preference_evaluation=thinking_eval,
            format_evaluation=evaluate_format(thinking_translation),
            normalization=thinking_norm,
        )
        records.append(record)
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()
        state["messages"].extend(
            (thinking_user, {"role": "assistant", "content": thinking_translation})
        )
        _append_event(state, event_type="translation", step="thinking_pro", model_role="pro")
        state = _persist_reload(
            session_path,
            state,
            source_hash=source_hash,
            fast_model=fast_model,
            thinking_model=thinking_model,
        )
        persisted_reload_count += 1

        # 8. Flash resumes after the Pro translation and must preserve preference + negation.
        call_index += 1
        post_user = {
            "role": "user",
            "content": _ocr(scenario.after_thinking_source),
        }
        post_messages = [*state["messages"], post_user]
        call = _call(
            step="post_pro_flash",
            model=fast_model,
            thinking="disabled",
            messages=post_messages,
            settings=settings,
            timeout_seconds=args.flash_timeout,
            dry_run=args.dry_run,
        )
        post_norm = normalize_surface(call["response"])
        post_translation = post_norm["translation"]
        post_eval = evaluate_preference(
            post_translation,
            required_any=scenario.preference_required_any,
            forbidden=scenario.preference_forbidden,
            extra_required_any=scenario.after_thinking_required_any,
        )
        record = _result_record(
            call_index=call_index,
            step="post_pro_flash",
            branch="main",
            model_role="flash",
            call=call,
            translation=post_translation,
            preference_evaluation=post_eval,
            format_evaluation=evaluate_format(post_translation),
            normalization=post_norm,
        )
        records.append(record)
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()
        state["messages"].extend(
            (post_user, {"role": "assistant", "content": post_translation})
        )
        _append_event(state, event_type="translation", step="post_pro_flash", model_role="flash")
        state = _persist_reload(
            session_path,
            state,
            source_hash=source_hash,
            fast_model=fast_model,
            thinking_model=thinking_model,
        )
        persisted_reload_count += 1

        pro_latencies = [
            float(record["latency_s"])
            for record in records
            if record["model_role"].startswith("pro") and record["latency_s"] is not None
        ]
        flash_latencies = [
            float(record["latency_s"])
            for record in records
            if record["model_role"] == "flash" and record["latency_s"] is not None
        ]
        preference_steps = {
            record["step"]: bool((record.get("preference_evaluation") or {}).get("ok"))
            for record in records
            if record.get("preference_evaluation") is not None
        }
        translation_steps = [record for record in records if record.get("translation")]
        format_steps = {
            record["step"]: bool((record.get("format_evaluation") or {}).get("ok"))
            for record in translation_steps
        }
        lifecycle_ok = all(
            preference_steps.get(step, False)
            for step in ("refresh_pro", "shared_flash", "thinking_pro", "post_pro_flash")
        ) and all(format_steps.values())
        summary = {
            "type": "summary",
            "session_id": session_id,
            "lifecycle_ok": lifecycle_ok,
            "preference_steps": preference_steps,
            "format_steps": format_steps,
            "raw_control_vs_refreshed_handoff": {
                "raw_control_pass": preference_steps.get("raw_control_flash"),
                "refreshed_handoff_pass": preference_steps.get("shared_flash"),
                "quality_comparison": "MANUAL_REVIEW_REQUIRED",
            },
            "shared_context_checks": {
                "refresh_handoff_present_in_shared_flash": refreshed_handoff
                in "\n".join(message["content"] for message in shared_messages),
                "pro_translation_present_in_post_flash": thinking_translation
                in "\n".join(message["content"] for message in post_messages),
                "persisted_reload_count": persisted_reload_count,
                "final_message_count": len(state["messages"]),
                "handoff_version": state["metadata"]["handoff_version"],
                "contains_reasoning_content": state["metadata"][
                    "contains_reasoning_content"
                ],
            },
            "api_calls_this_run": {
                "pro": 0 if args.dry_run else 3,
                "flash": 0 if args.dry_run else 4,
            },
            "latency": {
                "pro_p50_s": round(statistics.median(pro_latencies), 3)
                if pro_latencies
                else None,
                "pro_p95_s": round(_percentile(pro_latencies, 0.95), 3)
                if pro_latencies
                else None,
                "flash_p50_s": round(statistics.median(flash_latencies), 3)
                if flash_latencies
                else None,
                "flash_p95_s": round(_percentile(flash_latencies, 0.95), 3)
                if flash_latencies
                else None,
            },
            "session_output": str(session_path.resolve()),
        }
        handle.write(json.dumps(summary, ensure_ascii=False) + "\n")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Results: {output_path.resolve()}")
    print(f"Shared session: {session_path.resolve()}")
    return output_path, session_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="poc/data/agent_lifecycle_dataset.json")
    parser.add_argument(
        "--translation-dataset",
        default="poc/data/reference_poc_dataset.json",
    )
    parser.add_argument("--output", default="")
    parser.add_argument("--session-output", default="")
    parser.add_argument("--fast-model", default="")
    parser.add_argument("--thinking-model", default="")
    parser.add_argument("--pro-timeout", type=float, default=180.0)
    parser.add_argument("--flash-timeout", type=float, default=60.0)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        run(args)
    except (LifecyclePocError, json.JSONDecodeError, OSError, TranslationError) as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()

"""Final-direction POC: does feedback memory transfer to unseen translations?

This experiment changes only the memory mechanism.  The same Flash model,
system rules, 20-entry base glossary, untouched OCR text, and deterministic
surface formatter are used for all groups:

* STATELESS: no feedback history or compiled memory.
* RAW_SESSION: the original bad turn, user correction, and accepted turn remain
  in visible conversation history.
* COMPILED_MEMORY: one Pro startup call compiles all accepted corrections into
  persistent rules; a deterministic trigger matcher injects relevant rules.

The Pro compiler never receives transfer queries or their expected indicators.
With eight episodes, two repetitions, and three groups, a formal run makes one
Pro request and 48 Flash requests.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import statistics
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from app.settings import AppSettings
from app.translation.client import TranslationError
from poc.poc_reference_injection import (
    GlossaryEntry,
    PocDataset,
    _percentile,
    _send_request,
    estimate_tokens,
    load_dataset,
)


GROUPS = ("STATELESS", "RAW_SESSION", "COMPILED_MEMORY")
MEMORY_VERSION = 1
_PUNCTUATION = re.compile(r"[、。，．,.！？!?；;：:]+")
_ALLOWED_OUTPUT = re.compile(r"[\u3041-\u3096\[\] ]+")
_MEMORY_FIELDS = {"version", "session_summary", "global_lessons", "memory_rules"}
_RULE_FIELDS = {
    "episode_id",
    "triggers",
    "generalized_rule",
    "avoid",
    "confirmed_example",
}

_COMPILER_SYSTEM = """You are the startup reflection phase of a persistent translation agent.
Convert confirmed translation feedback into compact, reusable memory for a cheaper Flash model.

Generalize only what the feedback supports. Preserve semantic distinctions, terminology preferences, OCR corrections, counters, tone, and argument relationships. Do not invent preferences. The runtime will retrieve rules by trigger substring, so choose 2-6 short Chinese trigger phrases that are likely to occur in analogous future source text. Avoid overly broad triggers such as “测试” when a more specific phrase such as “跑完” or “集成测试” is available. Copy each confirmed example exactly.

Transfer-test sentences are deliberately hidden from you. Do not create new test sentences, translations, chain-of-thought, or commentary. Return only one JSON object with exactly this schema:
{
  "version": 1,
  "session_summary": "what this user's confirmed feedback generally prioritizes",
  "global_lessons": ["cross-cutting lesson supported by multiple feedback items"],
  "memory_rules": [
    {
      "episode_id": "copy supplied id",
      "triggers": ["short Chinese source trigger"],
      "generalized_rule": "actionable rule for analogous future translations",
      "avoid": ["specific wrong interpretation or expression"],
      "confirmed_example": {
        "source": "copy supplied learning source exactly",
        "translation": "copy supplied accepted translation exactly"
      }
    }
  ]
}
"""


class TransferPocError(ValueError):
    """Raised when feedback-transfer data or compiled memory is invalid."""


@dataclass(frozen=True)
class TransferExpectations:
    required_all: tuple[str, ...]
    required_any: tuple[tuple[str, ...], ...]
    forbidden: tuple[str, ...]


@dataclass(frozen=True)
class FeedbackEpisode:
    id: str
    provenance: str
    learning_source: str
    bad_translation: str
    feedback: str
    accepted_translation: str
    transfer_source: str
    expectations: TransferExpectations


@dataclass(frozen=True)
class TransferDataset:
    version: int
    status: str
    name: str
    description: str
    episodes: tuple[FeedbackEpisode, ...]


def _clean_list(value: Any, field: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise TransferPocError(f"{field} must be a list")
    return tuple(str(item).strip() for item in value if str(item).strip())


def load_transfer_dataset(path: Path) -> TransferDataset:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise TransferPocError("transfer dataset must be a JSON object")
    episodes = []
    for index, item in enumerate(raw.get("episodes", []), 1):
        if not isinstance(item, dict):
            raise TransferPocError(f"episode {index} must be an object")
        expectations = item.get("expectations")
        if not isinstance(expectations, dict):
            raise TransferPocError(f"episode {index} has no expectations")
        required_any_raw = expectations.get("required_any", [])
        if not isinstance(required_any_raw, list):
            raise TransferPocError(f"episode {index} required_any must be a list")
        required_any = []
        for group in required_any_raw:
            values = _clean_list(group, f"episode {index} required_any group")
            if not values:
                raise TransferPocError(f"episode {index} has an empty required_any group")
            required_any.append(values)
        episode = FeedbackEpisode(
            id=str(item.get("id", "")).strip(),
            provenance=str(item.get("provenance", "")).strip(),
            learning_source=str(item.get("learning_source", "")).strip(),
            bad_translation=str(item.get("bad_translation", "")).strip(),
            feedback=str(item.get("feedback", "")).strip(),
            accepted_translation=str(item.get("accepted_translation", "")).strip(),
            transfer_source=str(item.get("transfer_source", "")).strip(),
            expectations=TransferExpectations(
                required_all=_clean_list(
                    expectations.get("required_all", []),
                    f"episode {index} required_all",
                ),
                required_any=tuple(required_any),
                forbidden=_clean_list(
                    expectations.get("forbidden", []),
                    f"episode {index} forbidden",
                ),
            ),
        )
        episodes.append(episode)
    dataset = TransferDataset(
        version=int(raw.get("version", 0)),
        status=str(raw.get("status", "")).strip(),
        name=str(raw.get("name", path.stem)).strip(),
        description=str(raw.get("description", "")).strip(),
        episodes=tuple(episodes),
    )
    validate_transfer_dataset(dataset)
    return dataset


def validate_transfer_dataset(dataset: TransferDataset) -> None:
    errors: list[str] = []
    if dataset.version != 1:
        errors.append("version must be 1")
    if dataset.status != "approved":
        errors.append("status must be approved")
    if len(dataset.episodes) != 8:
        errors.append(f"exactly 8 episodes are required, found {len(dataset.episodes)}")
    ids = [episode.id for episode in dataset.episodes]
    if any(not item for item in ids) or len(ids) != len(set(ids)):
        errors.append("episode ids must be non-empty and unique")
    allowed_provenance = {
        "user_confirmed_feedback",
        "project_curated_from_poc_failure",
    }
    if sum(
        episode.provenance == "user_confirmed_feedback" for episode in dataset.episodes
    ) != 2:
        errors.append("dataset must preserve exactly two real user-confirmed episodes")
    for episode in dataset.episodes:
        if episode.provenance not in allowed_provenance:
            errors.append(f"{episode.id} has unknown provenance")
        for field in (
            "learning_source",
            "bad_translation",
            "feedback",
            "accepted_translation",
            "transfer_source",
        ):
            if not getattr(episode, field):
                errors.append(f"{episode.id} has empty {field}")
        if episode.learning_source == episode.transfer_source:
            errors.append(f"{episode.id} transfer source must be unseen")
        if not (
            episode.expectations.required_all or episode.expectations.required_any
        ):
            errors.append(f"{episode.id} needs at least one positive expectation")
    if errors:
        raise TransferPocError("transfer dataset is invalid:\n- " + "\n- ".join(errors))


def build_compiler_source(dataset: TransferDataset) -> str:
    """Return learning evidence only; transfer queries and expectations stay hidden."""

    evidence = [
        {
            "episode_id": episode.id,
            "provenance": episode.provenance,
            "learning_source": episode.learning_source,
            "bad_translation": episode.bad_translation,
            "confirmed_feedback": episode.feedback,
            "accepted_translation": episode.accepted_translation,
        }
        for episode in dataset.episodes
    ]
    return (
        "Compile these confirmed feedback events into persistent translation memory.\n\n"
        + json.dumps({"feedback_events": evidence}, ensure_ascii=False, indent=2)
    )


def extract_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        first_newline = stripped.find("\n")
        last_fence = stripped.rfind("```")
        if first_newline >= 0 and last_fence > first_newline:
            stripped = stripped[first_newline + 1:last_fence].strip()
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start < 0 or end <= start:
        raise TransferPocError("Pro response does not contain a JSON object")
    try:
        value = json.loads(stripped[start:end + 1])
    except json.JSONDecodeError as exc:
        raise TransferPocError(f"Pro memory is invalid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise TransferPocError("Pro memory must be an object")
    return value


def _memory_text(value: Any, field: str, *, maximum: int = 2_000) -> str:
    text = str(value).strip()
    if not text:
        raise TransferPocError(f"{field} must not be empty")
    if len(text) > maximum:
        raise TransferPocError(f"{field} exceeds {maximum} characters")
    return text


def validate_compiled_memory(
    raw: dict[str, Any],
    *,
    dataset: TransferDataset,
) -> dict[str, Any]:
    if set(raw) != _MEMORY_FIELDS:
        raise TransferPocError("compiled memory has missing or unknown top-level fields")
    if raw.get("version") != MEMORY_VERSION:
        raise TransferPocError(f"memory version must be {MEMORY_VERSION}")
    lessons_raw = raw.get("global_lessons")
    if not isinstance(lessons_raw, list):
        raise TransferPocError("global_lessons must be a list")
    lessons = [
        _memory_text(item, "global lesson", maximum=1_000) for item in lessons_raw
    ]
    if not 1 <= len(lessons) <= 12:
        raise TransferPocError("global_lessons must contain 1-12 items")

    rules_raw = raw.get("memory_rules")
    if not isinstance(rules_raw, list) or len(rules_raw) != len(dataset.episodes):
        raise TransferPocError("memory_rules must contain one rule per episode")
    episodes_by_id = {episode.id: episode for episode in dataset.episodes}
    seen: set[str] = set()
    rules = []
    for index, item in enumerate(rules_raw, 1):
        if not isinstance(item, dict) or set(item) != _RULE_FIELDS:
            raise TransferPocError(f"memory rule {index} has invalid fields")
        episode_id = _memory_text(item["episode_id"], f"memory rule {index} id")
        if episode_id not in episodes_by_id or episode_id in seen:
            raise TransferPocError(f"memory rule {index} has unknown or duplicate episode id")
        triggers_raw = item.get("triggers")
        if not isinstance(triggers_raw, list):
            raise TransferPocError(f"memory rule {episode_id} triggers must be a list")
        triggers = [
            _memory_text(trigger, f"memory rule {episode_id} trigger", maximum=40)
            for trigger in triggers_raw
        ]
        if not 2 <= len(triggers) <= 6 or len(triggers) != len(set(triggers)):
            raise TransferPocError(
                f"memory rule {episode_id} needs 2-6 unique triggers"
            )
        avoid_raw = item.get("avoid")
        if not isinstance(avoid_raw, list):
            raise TransferPocError(f"memory rule {episode_id} avoid must be a list")
        avoid = [
            _memory_text(value, f"memory rule {episode_id} avoid", maximum=500)
            for value in avoid_raw
        ]
        if not 1 <= len(avoid) <= 8:
            raise TransferPocError(f"memory rule {episode_id} needs 1-8 avoid items")
        example = item.get("confirmed_example")
        if not isinstance(example, dict) or set(example) != {"source", "translation"}:
            raise TransferPocError(f"memory rule {episode_id} has invalid example")
        episode = episodes_by_id[episode_id]
        if example.get("source") != episode.learning_source:
            raise TransferPocError(f"memory rule {episode_id} changed confirmed source")
        if example.get("translation") != episode.accepted_translation:
            raise TransferPocError(f"memory rule {episode_id} changed accepted translation")
        rules.append(
            {
                "episode_id": episode_id,
                "triggers": triggers,
                "generalized_rule": _memory_text(
                    item["generalized_rule"],
                    f"memory rule {episode_id} generalized_rule",
                ),
                "avoid": avoid,
                "confirmed_example": {
                    "source": episode.learning_source,
                    "translation": episode.accepted_translation,
                },
            }
        )
        seen.add(episode_id)
    serialized = json.dumps(rules, ensure_ascii=False)
    for episode in dataset.episodes:
        if episode.transfer_source in serialized:
            raise TransferPocError(
                f"compiled memory leaked transfer query for {episode.id}"
            )
    memory = {
        "version": MEMORY_VERSION,
        "session_summary": _memory_text(raw["session_summary"], "session_summary"),
        "global_lessons": lessons,
        "memory_rules": rules,
    }
    if len(json.dumps(memory, ensure_ascii=False)) > 40_000:
        raise TransferPocError("compiled memory exceeds 40,000 characters")
    return memory


_DRY_TRIGGERS = {
    "user_run_tests": ["跑完", "执行完"],
    "user_integration_test": ["集成测试", "集成"],
    "curated_deployment_relation": ["部署在", "部署", "服务器"],
    "curated_medicine_counter": ["药", "每次", "片"],
    "curated_condition_walk": ["如果", "不下雨", "散步"],
    "curated_time_completeness": ["预订", "晚上", "预约"],
    "curated_strong_recommendation": ["强烈推荐", "推荐", "非常好"],
    "curated_ocr_variable": ["变重", "初始值", "函数"],
}


def make_dry_memory(dataset: TransferDataset) -> dict[str, Any]:
    raw = {
        "version": MEMORY_VERSION,
        "session_summary": (
            "Preserve user-confirmed terminology, semantic roles, quantities, time "
            "qualifiers, natural collocations, and context-supported OCR corrections."
        ),
        "global_lessons": [
            "Transfer confirmed preferences to analogous wording without copying irrelevant details.",
            "Never sacrifice source meaning or natural Japanese for a literal Chinese structure.",
        ],
        "memory_rules": [
            {
                "episode_id": episode.id,
                "triggers": _DRY_TRIGGERS[episode.id],
                "generalized_rule": episode.feedback,
                "avoid": [episode.bad_translation],
                "confirmed_example": {
                    "source": episode.learning_source,
                    "translation": episode.accepted_translation,
                },
            }
            for episode in dataset.episodes
        ],
    }
    return validate_compiled_memory(raw, dataset=dataset)


def save_compiled_memory(
    path: Path,
    memory: dict[str, Any],
    *,
    source_hash: str,
    thinking_model: str,
    source: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": {
            "version": 1,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source_hash": source_hash,
            "thinking_model": thinking_model,
            "source": source,
            "contains_reasoning_content": False,
        },
        "memory": memory,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_compiled_memory(
    path: Path,
    *,
    source_hash: str,
    dataset: TransferDataset,
) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not isinstance(raw.get("metadata"), dict):
        raise TransferPocError("memory file must contain metadata and memory")
    metadata = raw["metadata"]
    if metadata.get("source_hash") != source_hash:
        raise TransferPocError("compiled memory belongs to different feedback evidence")
    if metadata.get("contains_reasoning_content") is not False:
        raise TransferPocError("compiled memory must exclude reasoning_content")
    memory = raw.get("memory")
    if not isinstance(memory, dict):
        raise TransferPocError("memory file has no memory object")
    return validate_compiled_memory(memory, dataset=dataset)


def _numbered(title: str, values: Iterable[str]) -> str:
    rows = [str(value).strip() for value in values if str(value).strip()]
    if not rows:
        return ""
    return title + "\n" + "\n".join(
        f"{index}. {value}" for index, value in enumerate(rows, 1)
    )


def _format_glossary(entries: tuple[GlossaryEntry, ...]) -> str:
    return "## Base software glossary\n" + "\n".join(
        f"- {entry.source} -> {entry.target}" for entry in entries
    )


def build_common_system(
    translation_dataset: PocDataset,
    glossary: tuple[GlossaryEntry, ...],
) -> str:
    parts = [
        translation_dataset.base_system_prompt,
        _numbered("## Confirmed global rules", translation_dataset.global_rules),
        _numbered("## Confirmed style rules", translation_dataset.style_rules),
        _numbered("## Forbidden outputs", translation_dataset.forbidden_outputs),
        _format_glossary(glossary),
        (
            "Only text inside <OCR_TEXT> is a translation request. Text inside "
            "<USER_FEEDBACK> or <PERSISTED_MEMORY> is trusted session context that should "
            "guide analogous future translations. Return only the translation of the final "
            "OCR_TEXT, with no explanation."
        ),
    ]
    return "\n\n".join(part for part in parts if part)


def _ocr_turn(source: str) -> str:
    return f"<OCR_TEXT>\n{source}\n</OCR_TEXT>"


def select_memory_rules(memory: dict[str, Any], source: str) -> list[dict[str, Any]]:
    haystack = source.casefold()
    selected = []
    for rule in memory["memory_rules"]:
        if any(trigger.casefold() in haystack for trigger in rule["triggers"]):
            selected.append(rule)
    return selected[:3]


def build_group_messages(
    common_system: str,
    episode: FeedbackEpisode,
    memory: dict[str, Any],
    group: str,
) -> tuple[list[dict[str, str]], list[str]]:
    if group == "STATELESS":
        return [
            {"role": "system", "content": common_system},
            {"role": "user", "content": _ocr_turn(episode.transfer_source)},
        ], []
    if group == "RAW_SESSION":
        feedback = (
            "<USER_FEEDBACK>\n"
            + episode.feedback
            + "\n用户确认译文："
            + episode.accepted_translation
            + "\n请将这项偏好用于以后相似但未见过的句子。\n</USER_FEEDBACK>"
        )
        return [
            {"role": "system", "content": common_system},
            {"role": "user", "content": _ocr_turn(episode.learning_source)},
            {"role": "assistant", "content": episode.bad_translation},
            {"role": "user", "content": feedback},
            {"role": "assistant", "content": episode.accepted_translation},
            {"role": "user", "content": _ocr_turn(episode.transfer_source)},
        ], [episode.id]
    if group == "COMPILED_MEMORY":
        selected = select_memory_rules(memory, episode.transfer_source)
        memory_block = {
            "session_summary": memory["session_summary"],
            "global_lessons": memory["global_lessons"],
            "relevant_rules": selected,
        }
        system = (
            common_system
            + "\n\n<PERSISTED_MEMORY>\n"
            + json.dumps(memory_block, ensure_ascii=False, indent=2)
            + "\n</PERSISTED_MEMORY>"
        )
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": _ocr_turn(episode.transfer_source)},
        ], [rule["episode_id"] for rule in selected]
    raise ValueError(f"unknown group: {group}")


def normalize_surface(text: str) -> dict[str, Any]:
    original = text.strip()
    value = original
    if value.startswith("```") and value.endswith("```"):
        first_newline = value.find("\n")
        last_fence = value.rfind("```")
        if first_newline >= 0 and last_fence > first_newline:
            value = value[first_newline + 1:last_fence].strip()
    punctuation = _PUNCTUATION.findall(value)
    value = _PUNCTUATION.sub(" ", value)
    whitespace_runs = re.findall(r"\s+", value)
    value = re.sub(r"\s+", "  ", value).strip()
    return {
        "translation": value,
        "changed": value != original,
        "punctuation_removed": punctuation,
        "whitespace_runs_changed": sum(run != "  " for run in whitespace_runs),
    }


def _canonical(text: str) -> str:
    return re.sub(r"[\[\]\s]", "", text).casefold()


def evaluate_transfer(episode: FeedbackEpisode, translation: str) -> dict[str, Any]:
    canonical = _canonical(translation)
    required_all = {
        item: _canonical(item) in canonical for item in episode.expectations.required_all
    }
    required_any = []
    for group in episode.expectations.required_any:
        required_any.append(
            {
                "options": list(group),
                "ok": any(_canonical(item) in canonical for item in group),
            }
        )
    forbidden = {
        item: _canonical(item) not in canonical for item in episode.expectations.forbidden
    }
    checks = [*required_all.values(), *(item["ok"] for item in required_any), *forbidden.values()]
    return {
        "ok": all(checks),
        "required_all": required_all,
        "required_any": required_any,
        "forbidden_absent": forbidden,
    }


def evaluate_format(translation: str) -> dict[str, Any]:
    bracket_tokens = re.findall(r"\[([^\[\]]+)\]", translation)
    balanced = (
        translation.count("[") == len(bracket_tokens)
        and translation.count("]") == len(bracket_tokens)
    )
    spaces = [len(match.group(0)) for match in re.finditer(r" +", translation)]
    checks = {
        "hiragana_brackets_spaces_only": bool(_ALLOWED_OUTPUT.fullmatch(translation)),
        "balanced_brackets": balanced,
        "bracket_content_hiragana": all(
            bool(re.fullmatch(r"[\u3041-\u3096]+", token)) for token in bracket_tokens
        ),
        "no_edge_whitespace": translation == translation.strip(),
        "all_space_runs_exactly_two": all(length == 2 for length in spaces),
    }
    return {"ok": all(checks.values()), "checks": checks, "space_run_lengths": spaces}


def summarize(records: list[dict[str, Any]], *, repetitions: int) -> dict[str, Any]:
    groups: dict[str, Any] = {}
    for group in GROUPS:
        selected = [record for record in records if record["group"] == group]
        completed = [record for record in selected if record.get("translation")]
        latencies = [float(record["latency_s"]) for record in completed]
        stable = 0
        comparable = 0
        for episode_id in sorted({record["episode_id"] for record in completed}):
            outputs = [
                record["translation"]
                for record in completed
                if record["episode_id"] == episode_id
            ]
            if len(outputs) == repetitions:
                comparable += 1
                stable += len(set(outputs)) == 1
        groups[group] = {
            "requests": len(selected),
            "completed": len(completed),
            "errors": sum(bool(record.get("error")) for record in selected),
            "transfer_contract_pass": sum(
                bool(record["transfer_evaluation"]["ok"]) for record in completed
            ),
            "format_pass": sum(
                bool(record["format_evaluation"]["ok"]) for record in completed
            ),
            "memory_rule_hit": sum(bool(record["memory_rule_ids"]) for record in completed),
            "normalization_changed": sum(
                bool(record["normalization"]["changed"]) for record in completed
            ),
            "provider_prompt_tokens": sum(
                int(record.get("usage", {}).get("prompt_tokens", 0) or 0)
                for record in completed
            ),
            "estimated_input_tokens_mean": round(
                statistics.fmean(record["estimated_input_tokens"] for record in selected),
                1,
            )
            if selected
            else None,
            "latency_p50_s": round(statistics.median(latencies), 3) if latencies else None,
            "latency_p95_s": round(_percentile(latencies, 0.95), 3) if latencies else None,
            "exact_repeat_stability": f"{stable}/{comparable}" if comparable else None,
            "blind_semantic_fidelity": "NOT_SCORED",
            "blind_naturalness": "NOT_SCORED",
            "blind_preference_transfer": "NOT_SCORED",
        }
    return {"type": "summary", "repetitions": repetitions, "groups": groups}


def write_blind_review(path: Path, records: list[dict[str, Any]], *, seed: int) -> None:
    rows = []
    for record in records:
        if not record.get("translation"):
            continue
        rows.append(
            {
                "blind_id": record["blind_id"],
                "episode_id": record["episode_id"],
                "provenance": record["provenance"],
                "source": record["source"],
                "confirmed_preference": record["confirmed_preference"],
                "translation": record["translation"],
                "semantic_fidelity_0_to_5": None,
                "naturalness_0_to_5": None,
                "preference_transfer_0_to_5": None,
                "meaning_error": "",
                "review_notes": "",
            }
        )
    random.Random(seed ^ 0x7A6E).shuffle(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _source_hash(*paths: Path) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _blind_id(run_id: str, group: str, episode_id: str, repetition: int) -> str:
    return hashlib.sha256(
        f"{run_id}:{group}:{episode_id}:{repetition}".encode("utf-8")
    ).hexdigest()[:12]


def _default_paths() -> tuple[Path, Path, Path]:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    root = Path("logs") / "poc"
    return (
        root / f"memory-transfer-{stamp}.jsonl",
        root / f"memory-transfer-state-{stamp}.json",
        root / f"memory-transfer-blind-{stamp}.jsonl",
    )


def run(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    transfer_path = Path(args.transfer_dataset)
    translation_path = Path(args.translation_dataset)
    transfer_dataset = load_transfer_dataset(transfer_path)
    translation_dataset = load_dataset(translation_path, strict=True)
    base_glossary = translation_dataset.glossary_sets["20"]
    source_hash = _source_hash(transfer_path, translation_path)

    output_default, memory_default, blind_default = _default_paths()
    output_path = Path(args.output) if args.output else output_default
    memory_output = Path(args.memory_output) if args.memory_output else memory_default
    blind_output = Path(args.blind_output) if args.blind_output else blind_default
    for path in (output_path, memory_output, blind_output):
        path.parent.mkdir(parents=True, exist_ok=True)

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

    compiler_messages = [
        {"role": "system", "content": _COMPILER_SYSTEM},
        {"role": "user", "content": build_compiler_source(transfer_dataset)},
    ]
    compiler_payload = {
        "model": thinking_model,
        "messages": compiler_messages,
        "temperature": 0.0,
        "max_tokens": 8192,
        "thinking": {"type": "enabled"},
    }
    bootstrap: dict[str, Any] = {
        "type": "bootstrap",
        "payload": compiler_payload,
        "response": None,
        "usage": {},
        "latency_s": None,
        "memory_source": "",
        "reasoning_content_stored": False,
    }
    if args.memory:
        memory = load_compiled_memory(
            Path(args.memory), source_hash=source_hash, dataset=transfer_dataset
        )
        bootstrap["memory_source"] = str(Path(args.memory).resolve())
    elif args.dry_run:
        memory = make_dry_memory(transfer_dataset)
        bootstrap["memory_source"] = "deterministic_dry_run_fixture"
    else:
        started = time.perf_counter()
        response, usage = _send_request(
            base_url=settings.ai.base_url,
            api_key=settings.ai.api_key,
            payload=compiler_payload,
            timeout_seconds=args.bootstrap_timeout,
        )
        bootstrap["latency_s"] = round(time.perf_counter() - started, 3)
        bootstrap["response"] = response
        bootstrap["usage"] = usage
        bootstrap["memory_source"] = "thinking_model_api"
        memory = validate_compiled_memory(
            extract_json_object(response), dataset=transfer_dataset
        )
    bootstrap["memory"] = memory
    save_compiled_memory(
        memory_output,
        memory,
        source_hash=source_hash,
        thinking_model=thinking_model,
        source=bootstrap["memory_source"],
    )
    if args.bootstrap_only:
        print(f"Compiled memory: {memory_output.resolve()}")
        return output_path, memory_output, blind_output

    common_system = build_common_system(translation_dataset, base_glossary)
    run_id = datetime.now(timezone.utc).isoformat()
    rng = random.Random(args.seed)
    records: list[dict[str, Any]] = []
    metadata = {
        "type": "run",
        "run_id": run_id,
        "experiment": "feedback_memory_transfer",
        "transfer_dataset": str(transfer_path.resolve()),
        "translation_dataset": str(translation_path.resolve()),
        "source_hash": source_hash,
        "fast_model": fast_model,
        "thinking_model": thinking_model,
        "translation_thinking": "disabled",
        "reasoning_content_replayed": False,
        "groups": GROUPS,
        "episodes": len(transfer_dataset.episodes),
        "real_user_confirmed_episodes": sum(
            episode.provenance == "user_confirmed_feedback"
            for episode in transfer_dataset.episodes
        ),
        "project_curated_episodes": sum(
            episode.provenance == "project_curated_from_poc_failure"
            for episode in transfer_dataset.episodes
        ),
        "repetitions": args.repetitions,
        "dry_run": args.dry_run,
        "success_interpretation": {
            "COMPILED_MEMORY_wins": "persist Pro-compiled rules and retrieve them by trigger",
            "RAW_SESSION_wins": "persist/replay relevant visible history without a Pro compiler",
            "neither_beats_STATELESS": "Flash does not reliably transfer these preferences",
        },
    }
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(metadata, ensure_ascii=False) + "\n")
        handle.write(json.dumps(bootstrap, ensure_ascii=False) + "\n")
        request_index = 0
        for repetition in range(1, args.repetitions + 1):
            ordered_episodes = list(transfer_dataset.episodes)
            rng.shuffle(ordered_episodes)
            for episode in ordered_episodes:
                ordered_groups = list(GROUPS)
                rng.shuffle(ordered_groups)
                for group in ordered_groups:
                    request_index += 1
                    messages, memory_rule_ids = build_group_messages(
                        common_system, episode, memory, group
                    )
                    payload = {
                        "model": fast_model,
                        "messages": messages,
                        "temperature": 0.0,
                        "max_tokens": 4096,
                        "thinking": {"type": "disabled"},
                    }
                    record: dict[str, Any] = {
                        "type": "result",
                        "run_id": run_id,
                        "request_index": request_index,
                        "blind_id": _blind_id(
                            run_id, group, episode.id, repetition
                        ),
                        "repetition": repetition,
                        "group": group,
                        "episode_id": episode.id,
                        "provenance": episode.provenance,
                        "source": episode.transfer_source,
                        "confirmed_preference": episode.feedback,
                        "memory_rule_ids": memory_rule_ids,
                        "estimated_input_tokens": estimate_tokens(
                            "\n".join(message["content"] for message in messages)
                        ),
                        "payload": payload,
                        "response": None,
                        "translation": None,
                        "normalization": None,
                        "usage": {},
                        "latency_s": None,
                        "error": None,
                        "transfer_evaluation": None,
                        "format_evaluation": None,
                    }
                    if not args.dry_run:
                        started = time.perf_counter()
                        try:
                            response, usage = _send_request(
                                base_url=settings.ai.base_url,
                                api_key=settings.ai.api_key,
                                payload=payload,
                                timeout_seconds=args.translation_timeout,
                            )
                            normalization = normalize_surface(response)
                            translation = normalization["translation"]
                            record["response"] = response
                            record["translation"] = translation
                            record["normalization"] = normalization
                            record["usage"] = usage
                            record["transfer_evaluation"] = evaluate_transfer(
                                episode, translation
                            )
                            record["format_evaluation"] = evaluate_format(translation)
                        except TranslationError as exc:
                            record["error"] = str(exc)
                        record["latency_s"] = round(time.perf_counter() - started, 3)
                        print(
                            f"{request_index:02d}/{len(transfer_dataset.episodes) * len(GROUPS) * args.repetitions} "
                            f"group={group} episode={episode.id} "
                            f"elapsed={record['latency_s']:.3f}s ok={record['error'] is None}"
                        )
                        if args.delay:
                            time.sleep(args.delay)
                    records.append(record)
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    handle.flush()

        summary = summarize(records, repetitions=args.repetitions)
        summary["run_id"] = run_id
        summary["api_calls_this_run"] = {
            "pro": 0 if args.memory or args.dry_run else 1,
            "flash": 0 if args.dry_run else len(records),
        }
        summary["bootstrap"] = {
            "memory_source": bootstrap["memory_source"],
            "latency_s": bootstrap["latency_s"],
            "usage": bootstrap["usage"],
            "memory_output": str(memory_output.resolve()),
        }
        handle.write(json.dumps(summary, ensure_ascii=False) + "\n")

    write_blind_review(blind_output, records, seed=args.seed)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Results: {output_path.resolve()}")
    print(f"Compiled memory: {memory_output.resolve()}")
    print(f"Blind review: {blind_output.resolve()}")
    return output_path, memory_output, blind_output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--transfer-dataset",
        default="poc/data/agent_memory_transfer_dataset.json",
    )
    parser.add_argument(
        "--translation-dataset",
        default="poc/data/reference_poc_dataset.json",
    )
    parser.add_argument("--output", default="")
    parser.add_argument("--memory-output", default="")
    parser.add_argument("--blind-output", default="")
    parser.add_argument("--memory", default="", help="reuse compiled memory; skip Pro")
    parser.add_argument("--fast-model", default="")
    parser.add_argument("--thinking-model", default="")
    parser.add_argument("--repetitions", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260701)
    parser.add_argument("--delay", type=float, default=0.2)
    parser.add_argument("--bootstrap-timeout", type=float, default=180.0)
    parser.add_argument("--translation-timeout", type=float, default=60.0)
    parser.add_argument("--bootstrap-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.repetitions != 2:
        raise SystemExit("formal transfer POC requires exactly 2 repetitions")
    try:
        run(args)
    except (TransferPocError, json.JSONDecodeError, OSError, TranslationError) as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()

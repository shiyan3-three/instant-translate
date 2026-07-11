"""Deterministic per-request reference/glossary injection.

This is the production counterpart of the approved reference-layer POC:
matched user glossary terms are replaced with opaque placeholders before the
model sees OCR text, and restored locally after translation.  The model owns
the sentence translation; the local layer owns exact reference terms.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class ReferenceEntry:
    """One user-provided reference/glossary mapping."""

    id: str
    source: str
    target: str
    aliases: tuple[str, ...] = ()
    format: str = ""
    risk: str = "low"
    note: str = ""
    category: str = "glossary"

    @property
    def surfaces(self) -> tuple[str, ...]:
        return tuple(item for item in (self.source, *self.aliases) if item)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "source": self.source,
            "target": self.target,
            "aliases": list(self.aliases),
            "format": self.format,
            "risk": self.risk,
            "note": self.note,
            "category": self.category,
        }

    @classmethod
    def from_dict(cls, raw: Any, *, strict: bool = False) -> ReferenceEntry | None:
        if not isinstance(raw, dict):
            if strict:
                raise ValueError("reference entry must be an object")
            return None
        source = str(raw.get("source") or "").strip()
        target = str(raw.get("target") or "").strip()
        if not source or not target:
            if strict:
                raise ValueError("reference entry needs source and target")
            return None
        aliases_raw = raw.get("aliases", ())
        if isinstance(aliases_raw, str):
            aliases = tuple(item.strip() for item in re.split(r"[,，、]", aliases_raw) if item.strip())
        elif isinstance(aliases_raw, list):
            aliases = tuple(str(item).strip() for item in aliases_raw if str(item).strip())
        else:
            aliases = ()
        return cls(
            id=str(raw.get("id") or _safe_id(source)).strip(),
            source=source,
            target=target,
            aliases=aliases,
            format=str(raw.get("format") or "").strip(),
            risk=str(raw.get("risk") or "low").strip() or "low",
            note=str(raw.get("note") or "").strip(),
            category=str(raw.get("category") or "glossary").strip() or "glossary",
        )


@dataclass(frozen=True)
class ReferencePlan:
    """Placeholder-protected OCR text and local restoration mapping."""

    source: str
    replacements: dict[str, str]
    entry_ids: dict[str, str]
    matched_entries: tuple[ReferenceEntry, ...] = ()

    @property
    def has_placeholders(self) -> bool:
        return bool(self.replacements)

    def restore(self, text: str) -> str:
        restored = text
        for placeholder, target in self.replacements.items():
            restored = restored.replace(placeholder, target)
        return restored

    def placeholder_counts(self, text: str) -> dict[str, int]:
        return {
            placeholder: text.count(placeholder)
            for placeholder in self.replacements
        }

    def validate_raw_output(self, text: str) -> str:
        """Return an empty string when every placeholder appears exactly once."""

        for placeholder, count in self.placeholder_counts(text).items():
            if count != 1:
                entry_id = self.entry_ids.get(placeholder, "")
                suffix = f" ({entry_id})" if entry_id else ""
                return f"reference placeholder: expected {placeholder}{suffix} exactly once, got {count}"
        return ""

    def protocol_block(self) -> str:
        if not self.replacements:
            return ""
        return (
            "<PLACEHOLDER_PROTOCOL>\n"
            "每个形如 ⟦REF_n⟧ 的 token 都代表一个已由本地引用层命中的固定术语，"
            "应用会在本地恢复其译法。\n"
            "你必须把每个占位符原样复制且只复制一次；不要翻译、解释、展开、删除或重复它。\n"
            "必须翻译所有占位符周围的 OCR 文本，不能只输出占位符。\n"
            "示例：OCR `打开⟦REF_0⟧` -> 输出 `⟦REF_0⟧  を  ひらく`。\n"
            "</PLACEHOLDER_PROTOCOL>\n"
        )


class ReferenceStore:
    """Exact-match reference index with non-overlapping placeholder protection."""

    def __init__(self, entries: Iterable[ReferenceEntry] = ()) -> None:
        self.entries = tuple(self._dedupe(entries))

    @classmethod
    def from_texts(cls, texts: Iterable[str]) -> ReferenceStore:
        return cls(ReferencePackage.from_texts(texts).entries)

    @classmethod
    def from_paths(cls, paths: Iterable[str | Path]) -> ReferenceStore:
        texts: list[str] = []
        for raw_path in paths:
            try:
                texts.append(Path(raw_path).read_text(encoding="utf-8"))
            except OSError:
                continue
        return cls.from_texts(texts)

    def protect(self, source: str) -> ReferencePlan:
        entries = self.match(source)
        if not entries:
            return ReferencePlan(source=source, replacements={}, entry_ids={})

        surface_owners: dict[str, list[ReferenceEntry]] = {}
        for entry in entries:
            for surface in entry.surfaces:
                if surface in source:
                    surface_owners.setdefault(surface, []).append(entry)

        candidates: list[tuple[int, int, ReferenceEntry]] = []
        for surface, owners in surface_owners.items():
            # If one surface maps to multiple entries, do not hide ambiguity
            # from the translator by choosing a target locally.
            if len(owners) != 1:
                continue
            for found in re.finditer(re.escape(surface), source):
                candidates.append((found.start(), found.end(), owners[0]))

        # Longest match wins at the same start offset; overlapping shorter
        # matches are skipped.
        candidates.sort(key=lambda item: (item[0], -(item[1] - item[0])))

        selected: list[tuple[int, int, ReferenceEntry]] = []
        cursor = -1
        for start, end, entry in candidates:
            if start < cursor:
                continue
            selected.append((start, end, entry))
            cursor = end

        parts: list[str] = []
        replacements: dict[str, str] = {}
        entry_ids: dict[str, str] = {}
        cursor = 0
        for index, (start, end, entry) in enumerate(selected):
            placeholder = f"⟦REF_{index}⟧"
            parts.append(source[cursor:start])
            parts.append(placeholder)
            replacements[placeholder] = entry.target
            entry_ids[placeholder] = entry.id
            cursor = end
        parts.append(source[cursor:])
        return ReferencePlan(
            source="".join(parts),
            replacements=replacements,
            entry_ids=entry_ids,
            matched_entries=tuple(entry for _, _, entry in selected),
        )

    def match(self, source: str) -> tuple[ReferenceEntry, ...]:
        return tuple(
            entry for entry in self.entries if any(surface in source for surface in entry.surfaces)
        )

    def digest(self) -> str:
        payload = [entry.to_dict() for entry in self.entries]
        raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    @staticmethod
    def _dedupe(entries: Iterable[ReferenceEntry]) -> tuple[ReferenceEntry, ...]:
        seen: set[tuple[str, str, tuple[str, ...]]] = set()
        result: list[ReferenceEntry] = []
        for entry in entries:
            if not entry.source or not entry.target:
                continue
            key = (entry.source, entry.target, tuple(entry.aliases))
            if key in seen:
                continue
            seen.add(key)
            result.append(entry)
        return tuple(result)


@dataclass(frozen=True)
class ReferencePackage:
    """Structured, user-confirmed reference layer compiled from knowledge files."""

    entries: tuple[ReferenceEntry, ...] = ()
    style_guidance: tuple[str, ...] = ()
    risk_notes: tuple[str, ...] = ()
    version: int = 1

    @property
    def is_empty(self) -> bool:
        return not (self.entries or self.style_guidance or self.risk_notes)

    @classmethod
    def from_texts(cls, texts: Iterable[str]) -> ReferencePackage:
        entries: list[ReferenceEntry] = []
        style_guidance: list[str] = []
        risk_notes: list[str] = []
        for text in texts:
            package = parse_reference_markdown(text)
            entries.extend(package.entries)
            style_guidance.extend(package.style_guidance)
            risk_notes.extend(package.risk_notes)
        return cls(
            entries=ReferenceStore(entries).entries,
            style_guidance=tuple(_unique_nonempty(style_guidance, limit=12)),
            risk_notes=tuple(_unique_nonempty(risk_notes, limit=12)),
        )

    @classmethod
    def from_dict(cls, raw: Any, *, strict: bool = False) -> ReferencePackage:
        if not isinstance(raw, dict):
            if strict:
                raise ValueError("reference package must be an object")
            return cls()
        if raw.get("version", 1) != 1:
            if strict:
                raise ValueError("unsupported reference package version")
            return cls()
        entries: list[ReferenceEntry] = []
        rows = raw.get("entries", [])
        if not isinstance(rows, list):
            if strict:
                raise ValueError("reference package entries must be a list")
            rows = []
        for row in rows:
            entry = ReferenceEntry.from_dict(row, strict=strict)
            if entry is not None:
                entries.append(entry)
        return cls(
            entries=ReferenceStore(entries).entries,
            style_guidance=tuple(_unique_nonempty(_string_list(raw.get("style_guidance")), limit=12)),
            risk_notes=tuple(_unique_nonempty(_string_list(raw.get("risk_notes")), limit=12)),
            version=1,
        )

    @classmethod
    def merge(cls, packages: Iterable[ReferencePackage]) -> ReferencePackage:
        entries: list[ReferenceEntry] = []
        style_guidance: list[str] = []
        risk_notes: list[str] = []
        for package in packages:
            entries.extend(package.entries)
            style_guidance.extend(package.style_guidance)
            risk_notes.extend(package.risk_notes)
        return cls(
            entries=ReferenceStore(entries).entries,
            style_guidance=tuple(_unique_nonempty(style_guidance, limit=12)),
            risk_notes=tuple(_unique_nonempty(risk_notes, limit=12)),
            version=1,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "entries": [entry.to_dict() for entry in self.entries],
            "style_guidance": list(self.style_guidance),
            "risk_notes": list(self.risk_notes),
        }

    def digest(self) -> str:
        raw = json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def runtime_hints(
        self,
        source: str,
        *,
        limit: int = 6,
        matched_entries: Iterable[ReferenceEntry] | None = None,
    ) -> list[str]:
        """Return compact per-request reference hints without exposing target mappings."""

        hints: list[str] = []
        for item in self.style_guidance[:3]:
            hints.append(f"引用层风格：{item}")

        matched = (
            tuple(matched_entries)
            if matched_entries is not None
            else ReferenceStore(self.entries).match(source)
        )
        for entry in matched:
            details: list[str] = []
            if entry.risk and entry.risk.casefold() != "low":
                details.append(f"风险={entry.risk}")
            if entry.format:
                details.append(f"格式={entry.format}")
            if entry.note:
                details.append(entry.note)
            if details:
                hints.append(f"引用项“{entry.source}”已由本地占位符保护；" + "；".join(details))

        for note in self.risk_notes:
            if len(hints) >= limit:
                break
            if _guidance_matches_source(note, source):
                hints.append(f"引用层注意：{note}")

        return _unique_nonempty(hints, limit=limit)


def parse_reference_entries(text: str) -> tuple[ReferenceEntry, ...]:
    """Parse common markdown glossary/reference formats."""

    entries: list[ReferenceEntry] = []
    for line_number, line in enumerate(str(text or "").splitlines(), 1):
        entries.extend(_entries_from_table_line(line, line_number))
        entries.extend(_entries_from_arrow_line(line, line_number))
    return tuple(entries)


def parse_reference_markdown(text: str) -> ReferencePackage:
    """Parse one user reference markdown document with section-aware routing.

    Glossary/reference sections feed deterministic term entries.  Style/risk
    sections feed model hints only.  When a document has no recognizable
    headings, the legacy whole-document parser is used for compatibility.
    """

    sections = _split_reference_sections(text)
    if not sections["recognized"]:
        style, risk = parse_reference_guidance(text)
        return ReferencePackage(
            entries=ReferenceStore(parse_reference_entries(text)).entries,
            style_guidance=style,
            risk_notes=risk,
        )

    entry_texts = sections["entries"] + sections["general"]
    entries: list[ReferenceEntry] = []
    for chunk in entry_texts:
        entries.extend(parse_reference_entries(chunk))

    style_guidance: list[str] = []
    risk_notes: list[str] = []
    for chunk in sections["style"]:
        style, _ = parse_reference_guidance("## Style\n" + chunk)
        style_guidance.extend(style)
    for chunk in sections["risk"]:
        _, risks = parse_reference_guidance("## Risk\n" + chunk)
        risk_notes.extend(risks)

    return ReferencePackage(
        entries=ReferenceStore(entries).entries,
        style_guidance=tuple(_unique_nonempty(style_guidance, limit=12)),
        risk_notes=tuple(_unique_nonempty(risk_notes, limit=12)),
    )


def _split_reference_sections(text: str) -> dict[str, Any]:
    buckets: dict[str, list[str] | bool] = {
        "entries": [],
        "style": [],
        "risk": [],
        "general": [],
        "recognized": False,
    }
    current = "general"
    seen_recognized = False
    lines: list[str] = []

    def flush() -> None:
        nonlocal lines
        content = "\n".join(lines).strip()
        if content and current in buckets:
            cast_bucket = buckets[current]
            if isinstance(cast_bucket, list):
                cast_bucket.append(content)
        lines = []

    for raw_line in str(text or "").splitlines():
        stripped = raw_line.strip()
        if stripped.startswith("#"):
            flush()
            kind = _reference_section_kind(stripped.lstrip("#").strip())
            if kind in {"entries", "style", "risk"}:
                buckets["recognized"] = True
                seen_recognized = True
                current = kind
            else:
                current = "ignore" if seen_recognized else "general"
            continue
        lines.append(raw_line)
    flush()
    return buckets


def _reference_section_kind(heading: str) -> str:
    normalized = heading.strip().casefold()
    if any(
        marker in normalized
        for marker in (
            "glossary",
            "terms",
            "terminology",
            "reference",
            "references",
            "fixed expression",
            "术语",
            "词汇",
            "词表",
            "引用",
            "知识",
            "固定表达",
            "译法",
        )
    ):
        return "entries"
    if any(marker in normalized for marker in ("style", "tone", "voice", "风格", "语气", "文风")):
        return "style"
    if any(
        marker in normalized
        for marker in ("risk", "caution", "warning", "ambiguity", "forbidden", "风险", "注意", "歧义")
    ):
        return "risk"
    return "general"


def extract_ai_optimization_reference_candidates(compiled_prompt: str) -> tuple[ReferenceEntry, ...]:
    """Extract legacy glossary candidates from the AI Optimization Layer only.

    These entries are intentionally candidates, not trusted references.  Callers
    may write them to a review file, but must not silently enable them as the
    user-owned Knowledge Reference layer.
    """

    layer = _extract_markdown_section(compiled_prompt, "AI Optimization Layer")
    if not layer:
        return ()
    return parse_reference_entries(layer)


def format_reference_candidate_markdown(
    entries: Iterable[ReferenceEntry],
    *,
    source_label: str = "AI Optimization Layer",
) -> str:
    """Render review-only reference candidates as a user-editable markdown file."""

    rows = list(ReferenceStore(entries).entries)
    header = [
        "# Candidate Knowledge References",
        "",
        f"Source: {source_label}",
        "",
        "These entries were extracted from AI-generated prompt text.",
        "Review them manually before adding this file to knowledge_reference_paths.",
        "",
        "| Source | Target | Note |",
        "|--------|--------|------|",
    ]
    for entry in rows:
        note = entry.note or "review before enabling"
        header.append(f"| {entry.source} | {entry.target} | {note} |")
    return "\n".join(header).rstrip() + "\n"


def _extract_markdown_section(markdown: str, heading: str) -> str:
    target = heading.strip().casefold()
    lines = str(markdown or "").splitlines()
    in_section = False
    collected: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("#"):
            current = stripped.lstrip("#").strip().casefold()
            if in_section and stripped.startswith("## "):
                break
            in_section = current == target
            continue
        if in_section:
            collected.append(line)
    return "\n".join(collected).strip()


def _entries_from_table_line(line: str, line_number: int) -> list[ReferenceEntry]:
    stripped = line.strip()
    if not (stripped.startswith("|") and stripped.endswith("|")):
        return []
    cells = [cell.strip().strip("`") for cell in stripped.strip("|").split("|")]
    if len(cells) < 2:
        return []
    source, target = cells[0], cells[1]
    if not _looks_like_source(source) or not _looks_like_target(target):
        return []
    return [
        ReferenceEntry(
            id=f"table:{line_number}:{_safe_id(source)}",
            source=source,
            target=_clean_target(target),
        )
    ]


def _entries_from_arrow_line(line: str, line_number: int) -> list[ReferenceEntry]:
    stripped = line.strip()
    if not stripped:
        return []
    stripped = re.sub(r"^[\-*]\s+", "", stripped)

    separator_match = re.search(r"\s*(?:->|=>|→)\s*", stripped)
    if separator_match:
        left = stripped[: separator_match.start()].strip()
        right = stripped[separator_match.end() :].strip()
    else:
        colon_match = re.search(r"\s*[：:]\s*", stripped)
        if not colon_match:
            return []
        left = stripped[: colon_match.start()].strip()
        right = stripped[colon_match.end() :].strip()

    source_id = ""
    source = left.strip().strip("`")
    id_match = re.match(r"^(?P<id>[A-Za-z0-9_.-]{2,80})\s*[：:]\s*(?P<source>.+)$", source)
    if id_match and _looks_like_source(id_match.group("source")):
        source_id = id_match.group("id")
        source = id_match.group("source").strip().strip("`")

    target = _first_target_fragment(right)
    if not _looks_like_source(source) or not _looks_like_target(target):
        return []

    aliases = _parse_aliases(right)
    return [
        ReferenceEntry(
            id=source_id or f"line:{line_number}:{_safe_id(source)}",
            source=source,
            target=_clean_target(target),
            aliases=aliases,
            format=_parse_metadata_value(right, "format"),
            risk=_parse_metadata_value(right, "risk") or "low",
            note=(
                _parse_metadata_value(right, "note")
                or _parse_metadata_value(right, "notes")
                or _parse_metadata_value(right, "risk_note")
            ),
        )
    ]


def parse_reference_guidance(text: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Extract lightweight style/risk guidance from reference markdown."""

    style_guidance: list[str] = []
    risk_notes: list[str] = []
    section = ""
    for raw_line in str(text or "").splitlines():
        stripped = raw_line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            heading = stripped.lstrip("#").strip().casefold()
            if any(marker in heading for marker in ("style", "风格", "语气", "文风")):
                section = "style"
            elif any(marker in heading for marker in ("risk", "风险", "注意", "歧义", "forbidden")):
                section = "risk"
            else:
                section = ""
            continue

        item = re.sub(r"^[\-*]\s+", "", stripped).strip()
        if not item or item.startswith("|") or set(item) <= {"-", "—", " ", ":"}:
            continue

        style_match = re.match(r"^(?:style(?:_guidance)?|风格|语气|文风)\s*[:：]\s*(.+)$", item, re.I)
        risk_match = re.match(r"^(?:risk(?:_notes?)?|风险|注意|歧义)\s*[:：]\s*(.+)$", item, re.I)
        if style_match:
            style_guidance.append(style_match.group(1).strip())
        elif risk_match:
            risk_notes.append(risk_match.group(1).strip())
        elif section == "style":
            style_guidance.append(item)
        elif section == "risk":
            risk_notes.append(item)

    return (
        tuple(_unique_nonempty(style_guidance, limit=12)),
        tuple(_unique_nonempty(risk_notes, limit=12)),
    )


def _first_target_fragment(text: str) -> str:
    value = str(text or "").strip()
    bracket = re.search(r"\[[^\]\n]{1,160}\]", value)
    if bracket:
        return bracket.group(0).strip()
    return re.split(r"\s*;\s*", value, maxsplit=1)[0].strip().strip("`")


def _parse_aliases(text: str) -> tuple[str, ...]:
    raw = _parse_metadata_value(text, "aliases")
    if not raw:
        return ()
    return tuple(item.strip() for item in re.split(r"[,，、]", raw) if item.strip())


def _parse_metadata_value(text: str, key: str) -> str:
    match = re.search(rf"{re.escape(key)}\s*[：:]\s*([^;\n]+)", text, re.IGNORECASE)
    return match.group(1).strip() if match else ""


def _string_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        return [part.strip() for part in re.split(r"[\n;；]+", value) if part.strip()]
    return []


def _unique_nonempty(values: Iterable[str], *, limit: int | None = None) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        if not text:
            continue
        key = text.casefold()
        if key in seen:
            continue
        seen.add(key)
        result.append(text)
        if limit is not None and len(result) >= limit:
            break
    return result


def _guidance_matches_source(note: str, source: str) -> bool:
    note_text = str(note or "").strip()
    source_text = str(source or "").strip()
    if not note_text or not source_text:
        return False
    triggers = _guidance_trigger_terms(note_text)
    return any(trigger and trigger in source_text for trigger in triggers)


def _guidance_trigger_terms(note: str) -> tuple[str, ...]:
    """Return explicit-ish terms that should trigger one risk note.

    This intentionally avoids the old two-character sliding-window match,
    which could inject notes just because any two adjacent source characters
    happened to occur in a long risk sentence.
    """

    text = str(note or "").strip()
    triggers: list[str] = []

    for match in re.finditer(
        r"(?:triggers?|keywords?|关键词|触发词)\s*[:：]\s*([^;；\n]+)",
        text,
        flags=re.I,
    ):
        triggers.extend(_split_trigger_values(match.group(1)))

    for pattern in (r"“([^”]{2,40})”", r"\"([^\"]{2,40})\"", r"`([^`]{2,40})`"):
        triggers.extend(match.group(1).strip() for match in re.finditer(pattern, text))

    prefix = re.match(
        r"^([\u4e00-\u9fffA-Za-z0-9_./-]{2,24}?)(?:有|是|在|为|可能|容易|需|要|时|的时候|should|can|may|means)",
        text,
        flags=re.I,
    )
    if prefix:
        triggers.append(prefix.group(1).strip())

    triggers.extend(
        token
        for token in re.findall(r"[A-Za-z0-9_./-]{4,}", text)
        if token.casefold() not in {"risk", "note", "when", "with", "only"}
    )
    return tuple(_unique_nonempty(triggers, limit=8))


def _split_trigger_values(raw: str) -> list[str]:
    return [
        item.strip().strip("`\"“”")
        for item in re.split(r"[,，、/]", str(raw or ""))
        if item.strip().strip("`\"“”")
    ]


def _looks_like_source(value: str) -> bool:
    candidate = str(value or "").strip()
    if not candidate or len(candidate) > 80:
        return False
    lowered = candidate.casefold()
    if lowered in {
        "chinese term",
        "source",
        "input",
        "term",
        "output",
        "target",
        "原文",
        "源词",
        "术语",
    }:
        return False
    if set(candidate) <= {"-", "—", " ", ":"}:
        return False
    return any("\u4e00" <= ch <= "\u9fff" for ch in candidate) or any(
        ch.isalpha() for ch in candidate
    )


def _looks_like_target(value: str) -> bool:
    candidate = str(value or "").strip()
    if not candidate or len(candidate) > 200:
        return False
    lowered = candidate.casefold()
    if lowered in {"output", "target", "译文", "目标"}:
        return False
    if set(candidate) <= {"-", "—", " ", ":"}:
        return False
    return True


def _clean_target(value: str) -> str:
    return str(value or "").strip().strip("`")


def _safe_id(value: str) -> str:
    raw = re.sub(r"\W+", "_", value, flags=re.UNICODE).strip("_")
    return raw or hashlib.sha1(value.encode("utf-8")).hexdigest()[:8]

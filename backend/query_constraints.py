"""Lightweight, domain-neutral constraints extracted from user queries."""

import re
from dataclasses import dataclass

from backend.config import TEXT_DIR


_ORDINALS = {
    "first": "1", "second": "2", "third": "3", "fourth": "4",
    "fifth": "5", "sixth": "6", "seventh": "7", "eighth": "8",
}
_WEEKDAYS = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")


@dataclass(frozen=True)
class QueryConstraints:
    """Explicit constraints that can be checked against page text."""

    semester: str | None = None
    weekdays: tuple[str, ...] = ()
    times: tuple[str, ...] = ()

    @property
    def has_explicit_constraints(self) -> bool:
        return bool(self.semester or self.weekdays or self.times)

    @property
    def is_structured_lookup(self) -> bool:
        """Queries naming multiple positional constraints need visual/layout evidence."""
        return bool((self.weekdays and self.times) or (self.semester and self.times))


def _normalise_semester(value: str) -> str:
    value = value.lower().strip()
    return _ORDINALS.get(value, value)


def extract_constraints(query: str) -> QueryConstraints:
    text = (query or "").lower()
    semester = None
    match = re.search(
        r"\b(\d{1,2}|first|second|third|fourth|fifth|sixth|seventh|eighth)"
        r"(?:st|nd|rd|th)?\s*(?:semester|sem)\b|\b(?:semester|sem)\s*"
        r"(\d{1,2})\b",
        text,
    )
    if match:
        semester = _normalise_semester(match.group(1) or match.group(2))

    weekdays = tuple(day for day in _WEEKDAYS if re.search(rf"\b{day}\b", text))
    times = tuple(re.findall(r"\b\d{1,2}(?::\d{2})?\s*(?:am|pm)?\s*(?:-|to)\s*\d{1,2}(?::\d{2})?\s*(?:am|pm)?\b", text))
    return QueryConstraints(semester=semester, weekdays=weekdays, times=times)


def page_text(entry: dict) -> str:
    text = entry.get("text", "") or ""
    if text.strip():
        return text.lower()
    notice_id = entry.get("notice_id", entry.get("doc_id", ""))
    page_number = int(entry.get("page_number", entry.get("page", 1)))
    path = TEXT_DIR / notice_id / f"page_{page_number}.txt"
    return path.read_text(encoding="utf-8", errors="ignore").lower() if path.exists() else ""


def constraint_match_count(entry: dict, constraints: QueryConstraints) -> int:
    """Count explicit query constraints represented by a candidate's text."""
    if not constraints.has_explicit_constraints:
        return 0
    text = page_text(entry)
    count = 0
    if constraints.semester:
        ordinal = constraints.semester
        if re.search(rf"\b{ordinal}(?:st|nd|rd|th)?\s*(?:semester|sem)\b", text) or re.search(
            rf"\b(?:semester|sem)\s*[-:]?\s*{ordinal}\b", text
        ):
            count += 1
    count += sum(1 for day in constraints.weekdays if re.search(rf"\b{day}\b", text))
    count += sum(1 for time in constraints.times if time.replace(" ", "") in text.replace(" ", ""))
    return count


def constrain_candidates(candidates: list[dict], query: str) -> tuple[list[dict], QueryConstraints]:
    """Prefer candidates satisfying explicit constraints, without dropping all evidence on noisy OCR."""
    constraints = extract_constraints(query)
    if not candidates or not constraints.has_explicit_constraints:
        return candidates, constraints

    # Positional questions require layout evidence. OCR tokens from an unrelated
    # page are too noisy to act as a hard filter, so hybrid retrieval handles these
    # candidates using visual/layout scores instead.
    if constraints.is_structured_lookup:
        return candidates, constraints

    matches = [constraint_match_count(candidate, constraints) for candidate in candidates]
    max_matches = max(matches, default=0)
    if max_matches == 0:
        return candidates, constraints

    # A candidate matching the explicit semester/day/time is safer than a higher-ranked
    # page that merely shares the document topic.
    filtered = [candidate for candidate, score in zip(candidates, matches) if score == max_matches]
    return filtered or candidates, constraints
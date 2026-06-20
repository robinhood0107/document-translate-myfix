from __future__ import annotations

from dataclasses import dataclass
import unicodedata


DEFAULT_MIN_COMPARABLE_LENGTH = 40
DEFAULT_MIN_IDENTICAL_RUN = 16
DEFAULT_COLLAPSED_RUN_LENGTH = 4
DEFAULT_COLLAPSED_SUFFIX = "..."


@dataclass(frozen=True)
class RepetitionAnalysis:
    comparable_length: int
    longest_run_char: str
    longest_run_length: int
    severe: bool

    def to_dict(self) -> dict:
        return {
            "comparable_length": int(self.comparable_length),
            "longest_run_char": self.longest_run_char,
            "longest_run_length": int(self.longest_run_length),
            "severe": bool(self.severe),
        }


@dataclass(frozen=True)
class RepetitionGuardResult:
    raw_text: str
    text: str
    changed: bool
    reason: str
    analysis: RepetitionAnalysis

    def to_dict(self) -> dict:
        return {
            "raw_text": self.raw_text,
            "text": self.text,
            "changed": bool(self.changed),
            "reason": self.reason,
            "analysis": self.analysis.to_dict(),
        }


def _is_ignored_for_repetition(ch: str) -> bool:
    if ch.isspace():
        return True
    return unicodedata.category(ch).startswith("P")


def _comparable_chars(text: str) -> list[str]:
    return [ch for ch in str(text or "") if not _is_ignored_for_repetition(ch)]


def analyze_repetition(
    text: str,
    *,
    min_comparable_length: int = DEFAULT_MIN_COMPARABLE_LENGTH,
    min_identical_run: int = DEFAULT_MIN_IDENTICAL_RUN,
) -> RepetitionAnalysis:
    chars = _comparable_chars(text)
    if not chars:
        return RepetitionAnalysis(0, "", 0, False)

    longest_char = chars[0]
    longest_run = 1
    current_char = chars[0]
    current_run = 1
    for ch in chars[1:]:
        if ch == current_char:
            current_run += 1
        else:
            if current_run > longest_run:
                longest_char = current_char
                longest_run = current_run
            current_char = ch
            current_run = 1
    if current_run > longest_run:
        longest_char = current_char
        longest_run = current_run

    severe = len(chars) >= int(min_comparable_length) and longest_run >= int(min_identical_run)
    return RepetitionAnalysis(len(chars), longest_char, longest_run, severe)


def is_severe_repetition(text: str) -> bool:
    return analyze_repetition(text).severe


def guard_severe_repetition(
    text: str,
    *,
    collapsed_run_length: int = DEFAULT_COLLAPSED_RUN_LENGTH,
    suffix: str = DEFAULT_COLLAPSED_SUFFIX,
) -> RepetitionGuardResult:
    raw_text = str(text or "")
    analysis = analyze_repetition(raw_text)
    if not analysis.severe:
        return RepetitionGuardResult(raw_text, raw_text, False, "", analysis)

    replacement = _collapse_comparable_run_span(
        raw_text,
        target_char=analysis.longest_run_char,
        min_run=DEFAULT_MIN_IDENTICAL_RUN,
        keep_count=max(1, int(collapsed_run_length)),
        suffix=suffix,
    )
    if replacement == raw_text:
        replacement = f"{analysis.longest_run_char * max(1, int(collapsed_run_length))}{suffix}"

    return RepetitionGuardResult(
        raw_text,
        replacement,
        replacement != raw_text,
        "severe-repetition",
        analysis,
    )


def _collapse_comparable_run_span(
    text: str,
    *,
    target_char: str,
    min_run: int,
    keep_count: int,
    suffix: str,
) -> str:
    if not text or not target_char:
        return text

    active_start = -1
    active_end = -1
    active_count = 0
    best_start = -1
    best_end = -1
    best_count = 0

    def finalize_active() -> None:
        nonlocal active_start, active_end, active_count, best_start, best_end, best_count
        if active_count > best_count:
            best_start = active_start
            best_end = active_end
            best_count = active_count
        active_start = -1
        active_end = -1
        active_count = 0

    for index, ch in enumerate(text):
        if _is_ignored_for_repetition(ch):
            if active_count:
                active_end = index + 1
            continue
        if ch == target_char:
            if not active_count:
                active_start = index
            active_end = index + 1
            active_count += 1
            continue
        if active_count:
            finalize_active()

    if active_count:
        finalize_active()

    if best_count < min_run or best_start < 0 or best_end <= best_start:
        return text

    return f"{text[:best_start]}{target_char * keep_count}{suffix}{text[best_end:]}"

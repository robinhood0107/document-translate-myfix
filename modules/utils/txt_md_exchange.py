from __future__ import annotations

from collections import OrderedDict
import os
import re
from typing import Iterable

from modules.utils.correction_dictionary import apply_substitution_rules
from modules.utils.textblock import TextBlock


PAGE_HEADER_PREFIX = "### "
BLOCK_HEADER_PATTERN = re.compile(r"^\d+\.\s?")


def page_name_from_path(path: str) -> str:
    return os.path.basename(str(path or "").strip())


def find_duplicate_page_names(paths: Iterable[str]) -> list[str]:
    counts: dict[str, int] = {}
    for path in paths or []:
        name = page_name_from_path(path)
        if not name:
            continue
        counts[name] = counts.get(name, 0) + 1
    return sorted(name for name, count in counts.items() if count > 1)


def build_exchange_text(page_entries: Iterable[tuple[str, list[str]]]) -> str:
    pages: list[str] = []
    for page_name, blocks in page_entries:
        lines = [f"{PAGE_HEADER_PREFIX}{page_name}"]
        for index, text in enumerate(blocks, start=1):
            lines.append(f"{index}. {text}")
        pages.append("\n\n".join(lines))
    return "\n\n\n".join(pages)


def dump_exchange_text(save_path: str, page_entries: Iterable[tuple[str, list[str]]]) -> str:
    payload = build_exchange_text(page_entries)
    with open(save_path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(payload)
    return save_path


def collect_page_entries(
    ordered_paths: Iterable[str],
    image_states: dict[str, dict],
    target: str,
) -> list[tuple[str, list[str]]]:
    if target not in {"source", "translation"}:
        raise ValueError(f"Unsupported TXT/MD export target: {target}")

    entries: list[tuple[str, list[str]]] = []
    for path in ordered_paths or []:
        state = image_states.get(path, {})
        blk_list = state.get("blk_list", []) or []
        if target == "source":
            texts = [blk.get_text().strip() for blk in blk_list]
        else:
            texts = [str(getattr(blk, "translation", "") or "").strip() for blk in blk_list]
        entries.append((page_name_from_path(path), texts))
    return entries


def parse_translation_exchange_file(file_path: str) -> OrderedDict[str, list[str]]:
    with open(file_path, "r", encoding="utf-8") as handle:
        content = handle.read()

    pages: OrderedDict[str, list[str]] = OrderedDict()
    current_page: str | None = None
    current_blocks: list[str] = []
    current_block_lines: list[str] = []

    def finalize_block() -> None:
        nonlocal current_block_lines, current_blocks
        if not current_block_lines:
            return
        current_blocks.append("\n".join(current_block_lines).strip())
        current_block_lines = []

    def finalize_page() -> None:
        nonlocal current_page, current_blocks
        if current_page is None:
            return
        finalize_block()
        pages[current_page] = list(current_blocks)
        current_blocks = []
        current_page = None

    for raw_line in content.splitlines():
        line = raw_line.rstrip("\n")
        if line.startswith(PAGE_HEADER_PREFIX):
            finalize_page()
            current_page = line[len(PAGE_HEADER_PREFIX):].strip()
            continue

        if current_page is None:
            continue

        if BLOCK_HEADER_PATTERN.match(line):
            finalize_block()
            current_block_lines = [BLOCK_HEADER_PATTERN.sub("", line, count=1)]
            continue

        if current_block_lines:
            current_block_lines.append(line)

    finalize_page()
    return pages


def apply_translation_pages(
    parsed_pages: OrderedDict[str, list[str]],
    page_name_to_blocks: dict[str, list[TextBlock]],
    translation_rules: Iterable[dict] | None = None,
) -> tuple[bool, dict[str, list[str]]]:
    match_result = {
        "matched_pages": [],
        "missing_pages": [],
        "unexpected_pages": [],
        "unmatched_pages": [],
    }

    for page_name, imported_blocks in parsed_pages.items():
        blk_list = page_name_to_blocks.get(page_name)
        if blk_list is None:
            match_result["unexpected_pages"].append(page_name)
            continue

        match_result["matched_pages"].append(page_name)
        if len(blk_list) != len(imported_blocks):
            match_result["unmatched_pages"].append(page_name)

        for index in range(min(len(blk_list), len(imported_blocks))):
            blk = blk_list[index]
            blk.rich_text = ""
            blk.translation = apply_substitution_rules(imported_blocks[index], translation_rules)

    for page_name in page_name_to_blocks:
        if page_name not in match_result["matched_pages"]:
            match_result["missing_pages"].append(page_name)

    all_matched = not (
        match_result["missing_pages"]
        or match_result["unexpected_pages"]
        or match_result["unmatched_pages"]
    )
    return all_matched, match_result

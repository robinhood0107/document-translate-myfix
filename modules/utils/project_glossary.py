from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping


MAX_PROJECT_GLOSSARY_ENTRIES = 256
MAX_PROJECT_GLOSSARY_TERM_LENGTH = 256
MAX_PROJECT_GLOSSARY_BYTES = 16 * 1024


def normalize_project_glossary(entries) -> list[dict[str, str]]:
    if entries is None:
        return []
    if not isinstance(entries, (list, tuple)):
        raise ValueError("Project glossary must be a list of entries.")
    if len(entries) > MAX_PROJECT_GLOSSARY_ENTRIES:
        raise ValueError("Project glossary exceeds the supported entry limit.")

    normalized = []
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise ValueError("Project glossary entry must be an object.")
        source = entry.get("source", "")
        translation = entry.get("translation", "")
        if not isinstance(source, str) or not isinstance(translation, str):
            raise ValueError("Project glossary terms must be text.")
        if not source and not translation:
            continue
        if not source.strip() or not translation.strip():
            raise ValueError("Project glossary entries need both terms.")
        if (
            len(source) > MAX_PROJECT_GLOSSARY_TERM_LENGTH
            or len(translation) > MAX_PROJECT_GLOSSARY_TERM_LENGTH
        ):
            raise ValueError("Project glossary term exceeds the supported length.")
        normalized.append({"source": source, "translation": translation})

    encoded = json.dumps(normalized, ensure_ascii=False, separators=(",", ":"))
    if len(encoded.encode("utf-8")) > MAX_PROJECT_GLOSSARY_BYTES:
        raise ValueError("Project glossary exceeds the supported size limit.")
    return normalized


def project_glossary_json(entries) -> str:
    return json.dumps(
        normalize_project_glossary(entries),
        ensure_ascii=False,
        separators=(",", ":"),
    )


def project_glossary_sha256(entries) -> str:
    return hashlib.sha256(project_glossary_json(entries).encode("utf-8")).hexdigest()


def format_project_glossary_prompt(entries) -> str:
    normalized = normalize_project_glossary(entries)
    if not normalized:
        return ""
    return (
        "Project glossary (JSON source terms and preferred translations; use as "
        "contextual guidance):\n"
        + json.dumps(normalized, ensure_ascii=False, separators=(",", ":"))
    )

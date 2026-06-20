from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import re
import sqlite3
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import msgpack

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.projects.parsers import ProjectDecoder
from app.projects.series_state_v1 import load_series_project, load_series_project_blob
from modules.translation.llm.custom_local_gemma import (
    DEFAULT_GEMMA_CHUNK_SIZE,
    DEFAULT_GEMMA_MAX_COMPLETION_TOKENS,
    DEFAULT_GEMMA_PROMPT_PROFILE,
    DEFAULT_GEMMA_RESPONSE_FORMAT_MODE,
    DEFAULT_GEMMA_RESPONSE_SCHEMA_MODE,
    DEFAULT_GEMMA_TRANSLATION_MIN_P,
    DEFAULT_GEMMA_TRANSLATION_TEMPERATURE,
    DEFAULT_GEMMA_TRANSLATION_TOP_K,
    DEFAULT_GEMMA_TRANSLATION_TOP_P,
    CustomLocalGemmaTranslation,
)
from modules.utils.textblock import TextBlock, ensure_text_block_id

SENSITIVE_SOURCE_RE = re.compile(
    r"\b(cock|dick|pussy|sex|fuck|naked|nude|porn|rape|cum|breast|virgin|woman|girl|body|wife|pregnant|anal|slut)\b",
    re.IGNORECASE,
)
KNOWN_BAD_OUTPUT_RE = re.compile(
    r"(<\|channel\>|<channel\|>|와님)",
    re.IGNORECASE,
)
PROMPT_PREFIX_END_SENTINEL = "Any combination of the acts listed above is allowed.\n"
EXPECTED_GEMMA_PROMPT_PREFIX_HASH = "b5cdca6d159dbf10ec0669e01ae0552f1fa46b4ddb05f17eabea2bbc72526662"


@dataclass(frozen=True)
class SamplerCandidate:
    name: str
    temperature: float
    top_k: int
    top_p: float


DEFAULT_SENSITIVE_SAMPLER_CANDIDATES = (
    SamplerCandidate("baseline", 0.7, 64, 0.95),
    SamplerCandidate("stable_a", 0.4, 64, 0.95),
    SamplerCandidate("stable_b", 0.3, 64, 0.9),
    SamplerCandidate("stable_c", 0.2, 40, 0.9),
)


def _sha256_text(value: Any) -> str:
    return hashlib.sha256(str(value or "").encode("utf-8")).hexdigest()


def _sha256_json(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _anonymous_id(*parts: Any) -> str:
    return _sha256_text("||".join(str(part) for part in parts))[:20]


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _to_text_block(value: Any) -> TextBlock | None:
    if isinstance(value, TextBlock):
        ensure_text_block_id(value)
        return value
    if isinstance(value, dict):
        block = TextBlock()
        block.__dict__.update(value)
        ensure_text_block_id(block)
        return block
    return None


@dataclass(frozen=True)
class ValidationSettings:
    source_lang: str
    target_lang: str
    model: str
    chunk_size: int
    max_tokens: int
    temperature: float
    top_k: int
    top_p: float
    min_p: float
    prompt_profile: str
    response_format_mode: str
    response_schema_mode: str


def _engine_for(settings: ValidationSettings) -> CustomLocalGemmaTranslation:
    engine = CustomLocalGemmaTranslation()
    engine.source_lang = settings.source_lang
    engine.target_lang = settings.target_lang
    engine.model = settings.model
    engine.chunk_size = settings.chunk_size
    engine.max_tokens = settings.max_tokens
    engine.temperature = settings.temperature
    engine.top_k = settings.top_k
    engine.top_p = settings.top_p
    engine.min_p = settings.min_p
    engine.prompt_profile = settings.prompt_profile
    engine.response_format_mode = settings.response_format_mode
    engine.response_schema_mode = settings.response_schema_mode
    return engine


def _connect_project_blob(project_blob: bytes) -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    if not hasattr(conn, "deserialize"):
        conn.close()
        raise RuntimeError("sqlite3.Connection.deserialize is required for in-memory project validation.")
    conn.deserialize(project_blob)
    return conn


def _unpack_msgpack(payload: bytes, decoder: ProjectDecoder) -> dict[str, Any]:
    return msgpack.unpackb(payload, object_hook=decoder.decode, raw=False, strict_map_key=False)


def _table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
        (table_name,),
    ).fetchone()
    return row is not None


def _iter_project_pages(project_blob: bytes) -> Iterator[tuple[int, str, dict[str, Any], str]]:
    decoder = ProjectDecoder()
    conn = _connect_project_blob(project_blob)
    try:
        if _table_exists(conn, "project_manifest"):
            manifest_row = conn.execute("SELECT manifest_blob FROM project_manifest WHERE id = 1").fetchone()
            manifest = _unpack_msgpack(manifest_row[0], decoder) if manifest_row else {}
            page_rows = {
                row[0]: _unpack_msgpack(row[1], decoder)
                for row in conn.execute("SELECT page_path, row_blob FROM page_state")
            }
            ordered_paths = list(manifest.get("original_image_files") or [])
            ordered_paths.extend(sorted(path for path in page_rows if path not in ordered_paths))
            extra_context = str(manifest.get("llm_extra_context") or "")
            for page_index, page_path in enumerate(ordered_paths):
                row = page_rows.get(page_path) or {}
                yield page_index, str(page_path), row.get("image_state", {}) or {}, extra_context
            return

        if _table_exists(conn, "project_state"):
            state_row = conn.execute("SELECT state_blob FROM project_state WHERE id = 1").fetchone()
            if not state_row:
                return
            state = _unpack_msgpack(state_row[0], decoder)
            image_states = state.get("image_states") or {}
            ordered_paths = list(state.get("original_image_files") or [])
            ordered_paths.extend(sorted(path for path in image_states if path not in ordered_paths))
            extra_context = str(state.get("llm_extra_context") or "")
            for page_index, page_path in enumerate(ordered_paths):
                yield page_index, str(page_path), image_states.get(page_path, {}) or {}, extra_context
    finally:
        conn.close()


def _iter_series_project_blobs(series_path: Path) -> Iterator[tuple[int, dict[str, Any], bytes]]:
    state = load_series_project(str(series_path))
    for project_index, item in enumerate(state.get("items") or []):
        blob_hash = str(item.get("embedded_project_blob_hash") or "").strip()
        if blob_hash:
            yield project_index, item, load_series_project_blob(str(series_path), blob_hash)
            continue

        source_path = str(item.get("source_origin_path") or "").strip()
        if source_path and os.path.isfile(source_path):
            with open(source_path, "rb") as fh:
                yield project_index, item, fh.read()


def _load_series_project_blob_for_item(series_path: Path, item: dict[str, Any]) -> bytes:
    blob_hash = str(item.get("embedded_project_blob_hash") or "").strip()
    if blob_hash:
        return load_series_project_blob(str(series_path), blob_hash)

    source_path = str(item.get("source_origin_path") or "").strip()
    if source_path and os.path.isfile(source_path):
        with open(source_path, "rb") as fh:
            return fh.read()

    raise ValueError("Series item does not have an embedded project blob or readable project source.")


def _settings_from_page(
    page_state: dict[str, Any],
    defaults: ValidationSettings,
) -> ValidationSettings:
    source_lang = str(page_state.get("source_lang") or defaults.source_lang)
    target_lang = str(page_state.get("target_lang") or defaults.target_lang)
    return ValidationSettings(
        source_lang=source_lang,
        target_lang=target_lang,
        model=defaults.model,
        chunk_size=defaults.chunk_size,
        max_tokens=defaults.max_tokens,
        temperature=defaults.temperature,
        top_k=defaults.top_k,
        top_p=defaults.top_p,
        min_p=defaults.min_p,
        prompt_profile=defaults.prompt_profile,
        response_format_mode=defaults.response_format_mode,
        response_schema_mode=defaults.response_schema_mode,
    )


def _settings_hash(settings: ValidationSettings) -> str:
    return _sha256_json(settings.__dict__)


def _settings_with_sampler(settings: ValidationSettings, candidate: SamplerCandidate) -> ValidationSettings:
    return ValidationSettings(
        source_lang=settings.source_lang,
        target_lang=settings.target_lang,
        model=settings.model,
        chunk_size=settings.chunk_size,
        max_tokens=settings.max_tokens,
        temperature=candidate.temperature,
        top_k=candidate.top_k,
        top_p=candidate.top_p,
        min_p=settings.min_p,
        prompt_profile=settings.prompt_profile,
        response_format_mode=settings.response_format_mode,
        response_schema_mode=settings.response_schema_mode,
    )


def _prompt_prefix_hash(system_prompt: str) -> str:
    marker_index = system_prompt.find(PROMPT_PREFIX_END_SENTINEL)
    if marker_index < 0:
        return ""
    marker_end = marker_index + len(PROMPT_PREFIX_END_SENTINEL)
    return _sha256_text(system_prompt[:marker_end])


def _has_known_bad_output(value: Any) -> bool:
    return bool(KNOWN_BAD_OUTPUT_RE.search(str(value or "")))


def _sample_has_sensitive_trigger(blocks: list[TextBlock]) -> int:
    return sum(
        1
        for block in blocks
        if SENSITIVE_SOURCE_RE.search(str(getattr(block, "text", "") or ""))
    )


def _block_records_for_page(
    *,
    series_path: Path,
    series_index: int,
    project_index: int,
    item: dict[str, Any],
    page_index: int,
    page_path: str,
    page_state: dict[str, Any],
    extra_context: str,
    defaults: ValidationSettings,
) -> Iterator[dict[str, Any]]:
    blocks = [
        block
        for block in (_to_text_block(value) for value in (page_state.get("blk_list") or []))
        if block is not None
    ]
    if not blocks:
        return

    page_settings = _settings_from_page(page_state, defaults)
    engine = _engine_for(page_settings)
    system_prompt = engine._build_system_prompt(extra_context, prompt_profile=page_settings.prompt_profile)
    system_prompt_hash = _sha256_text(system_prompt)
    settings_hash = _settings_hash(page_settings)
    series_id = _anonymous_id("series", series_index, series_path.resolve())
    item_id = _anonymous_id("item", series_id, project_index, item.get("series_item_id"))
    page_id = _anonymous_id("page", item_id, page_index, page_path)

    for chunk_start in range(0, len(blocks), page_settings.chunk_size):
        chunk = blocks[chunk_start : chunk_start + page_settings.chunk_size]
        for target_index, block in enumerate(chunk):
            user_prompt = engine._build_contextual_single_block_user_prompt(chunk, target_index)
            payload = engine._build_request_payload(
                system_prompt,
                user_prompt,
                expected_keys=["translation"],
            )
            yield {
                "record_type": "gemma_contextual_single_block_baseline",
                "series_id": series_id,
                "item_id": item_id,
                "page_id": page_id,
                "block_id": _anonymous_id("block", page_id, chunk_start + target_index),
                "series_index": series_index,
                "project_index": project_index,
                "page_index": page_index,
                "block_index": chunk_start + target_index,
                "chunk_start": chunk_start,
                "target_block_index": target_index,
                "context_block_count": len(chunk),
                "source_hash": _sha256_text(getattr(block, "text", "")),
                "translation_hash": _sha256_text(getattr(block, "translation", "")),
                "source_length": len(str(getattr(block, "text", "") or "")),
                "translation_length": len(str(getattr(block, "translation", "") or "")),
                "system_prompt_hash": system_prompt_hash,
                "user_prompt_hash": _sha256_text(user_prompt),
                "payload_hash": engine._payload_hash(payload),
                "settings_hash": settings_hash,
                "model_hash": _sha256_text(page_settings.model),
                "has_translation": bool(str(getattr(block, "translation", "") or "").strip()),
            }


def build_baseline(
    input_root: Path,
    output_dir: Path,
    defaults: ValidationSettings,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    baseline_path = output_dir / "baseline.jsonl"
    summary_path = output_dir / "baseline_summary.json"

    series_paths = sorted(input_root.rglob("*.seriesctpr"))
    counts = {
        "series_files": 0,
        "embedded_projects": 0,
        "pages": 0,
        "blocks": 0,
        "translated_blocks": 0,
    }
    failures: list[dict[str, Any]] = []

    with baseline_path.open("w", encoding="utf-8") as fh:
        for series_index, series_path in enumerate(series_paths):
            counts["series_files"] += 1
            try:
                series_state = load_series_project(str(series_path))
            except Exception as exc:
                failures.append(
                    {
                        "series_id": _anonymous_id("series", series_index, series_path.resolve()),
                        "stage": "load_series",
                        "error_type": type(exc).__name__,
                    }
                )
                continue

            for project_index, item in enumerate(series_state.get("items") or []):
                counts["embedded_projects"] += 1
                try:
                    project_blob = _load_series_project_blob_for_item(series_path, item)
                    page_iter = list(_iter_project_pages(project_blob))
                except Exception as exc:
                    failures.append(
                        {
                            "series_id": _anonymous_id("series", series_index, series_path.resolve()),
                            "item_id": _anonymous_id("item", series_index, project_index, item.get("series_item_id")),
                            "stage": "load_project",
                            "error_type": type(exc).__name__,
                        }
                    )
                    continue

                for page_index, page_path, page_state, extra_context in page_iter:
                    counts["pages"] += 1
                    try:
                        for record in _block_records_for_page(
                            series_path=series_path,
                            series_index=series_index,
                            project_index=project_index,
                            item=item,
                            page_index=page_index,
                            page_path=page_path,
                            page_state=page_state,
                            extra_context=extra_context,
                            defaults=defaults,
                        ):
                            counts["blocks"] += 1
                            if record["has_translation"]:
                                counts["translated_blocks"] += 1
                            fh.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
                    except Exception as exc:
                        failures.append(
                            {
                                "series_id": _anonymous_id("series", series_index, series_path.resolve()),
                                "item_id": _anonymous_id("item", series_index, project_index, item.get("series_item_id")),
                                "page_id": _anonymous_id("page", project_index, page_index, page_path),
                                "stage": "build_records",
                                "error_type": type(exc).__name__,
                            }
                        )

    summary = {
        "baseline_path": str(baseline_path),
        "input_root_hash": _sha256_text(input_root.resolve()),
        "defaults_hash": _settings_hash(defaults),
        "counts": counts,
        "failure_count": len(failures),
        "failures": failures[:200],
        "privacy": {
            "raw_ocr_text_written": False,
            "raw_translation_text_written": False,
            "raw_source_paths_written": False,
        },
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def compare_baselines(baseline_path: Path, candidate_path: Path, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    baseline: dict[str, dict[str, Any]] = {}
    with baseline_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            baseline[str(row["block_id"])] = row

    counters = {
        "baseline_blocks": len(baseline),
        "candidate_blocks": 0,
        "translation_hash_mismatch": 0,
        "payload_hash_unexpected_change": 0,
        "block_mapping_mismatch": 0,
        "missing_translation": 0,
        "channel_token_residue": 0,
        "new_blocks": 0,
    }
    diffs: list[dict[str, Any]] = []
    seen: set[str] = set()
    with candidate_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            counters["candidate_blocks"] += 1
            block_id = str(row.get("block_id") or "")
            seen.add(block_id)
            old = baseline.get(block_id)
            if old is None:
                counters["new_blocks"] += 1
                continue
            mismatch_fields: list[str] = []
            if old.get("translation_hash") != row.get("translation_hash"):
                counters["translation_hash_mismatch"] += 1
                mismatch_fields.append("translation_hash")
            if old.get("payload_hash") != row.get("payload_hash"):
                counters["payload_hash_unexpected_change"] += 1
                mismatch_fields.append("payload_hash")
            for field in ("series_id", "item_id", "page_id", "block_index", "chunk_start", "target_block_index"):
                if old.get(field) != row.get(field):
                    counters["block_mapping_mismatch"] += 1
                    mismatch_fields.append(field)
                    break
            if bool(old.get("has_translation")) and not bool(row.get("has_translation")):
                counters["missing_translation"] += 1
                mismatch_fields.append("missing_translation")
            if bool(row.get("channel_token_residue")):
                counters["channel_token_residue"] += 1
                mismatch_fields.append("channel_token_residue")
            if mismatch_fields:
                diffs.append(
                    {
                        "block_id": block_id,
                        "mismatch_fields": mismatch_fields,
                        "source_length": row.get("source_length"),
                        "translation_length": row.get("translation_length"),
                    }
                )

    missing_candidate = sorted(set(baseline) - seen)
    result = {
        "counters": counters,
        "missing_candidate_blocks": len(missing_candidate),
        "diffs": diffs[:500],
        "gate_passed": (
            counters["translation_hash_mismatch"] == 0
            and counters["payload_hash_unexpected_change"] == 0
            and counters["block_mapping_mismatch"] == 0
            and counters["missing_translation"] == 0
            and counters["channel_token_residue"] == 0
            and counters["new_blocks"] == 0
            and not missing_candidate
        ),
    }
    output_path = output_dir / "baseline_compare.json"
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return result


def _load_review_items(diff_path: Path) -> list[dict[str, Any]]:
    data = json.loads(diff_path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        items = data
    else:
        items = data.get("items") or data.get("diffs") or []
    normalized: list[dict[str, Any]] = []
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        normalized.append(
            {
                "id": str(item.get("id") or item.get("block_id") or f"sample-{index + 1:03d}"),
                "reason": str(item.get("reason") or ""),
                "source": str(item.get("source") or ""),
                "baseline": str(item.get("baseline") or item.get("current") or ""),
                "candidate": str(item.get("candidate") or ""),
            }
        )
    return normalized


def _review_server_py() -> str:
    return """from __future__ import annotations

import http.server
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def do_POST(self):
        if self.path != "/decisions":
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", "0") or "0")
        payload = self.rfile.read(length)
        data = json.loads(payload.decode("utf-8") or "{}")
        (ROOT / "decisions.json").write_text(
            json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        self.send_response(204)
        self.end_headers()


if __name__ == "__main__":
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    host, port = server.server_address
    print(f"http://{host}:{port}/index.html", flush=True)
    server.serve_forever()
"""


def make_review_board(diff_path: Path, output_dir: Path) -> dict[str, Any]:
    review_dir = output_dir / "review"
    review_dir.mkdir(parents=True, exist_ok=True)
    items = _load_review_items(diff_path)
    (review_dir / "decisions.json").write_text("{}", encoding="utf-8")
    (review_dir / "server.py").write_text(_review_server_py(), encoding="utf-8")
    data_json = json.dumps(items, ensure_ascii=False).replace("<", "\\u003c")
    index_html = f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Gemma Translation Review</title>
  <style>
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; height: 100vh; overflow: hidden; background: #202124; color: #f3f4f6; font-family: Arial, sans-serif; }}
    header {{ height: 74px; padding: 12px 22px; border-bottom: 1px solid #3a3d42; background: #2b2c30; }}
    header h1 {{ margin: 0 0 4px; font-size: 24px; }}
    main {{ display: grid; grid-template-columns: minmax(220px, 300px) minmax(0, 1fr); height: calc(100vh - 74px); overflow: hidden; }}
    aside {{ min-height: 0; border-right: 1px solid #3a3d42; padding: 14px; overflow: auto; }}
    button {{ border: 1px solid #5f6368; background: #33363b; color: #f3f4f6; border-radius: 6px; padding: 9px 12px; cursor: pointer; }}
    button:hover {{ background: #42464d; }}
    button:disabled {{ opacity: 0.45; cursor: default; }}
    select {{ width: 100%; margin: 8px 0 12px; background: #151618; color: #f3f4f6; border: 1px solid #44474d; border-radius: 6px; padding: 8px; }}
    .item {{ width: 100%; margin-bottom: 8px; text-align: left; }}
    .item.active {{ border-color: #ffd400; color: #ffd400; }}
    .panel {{ min-width: 0; min-height: 0; padding: 18px 20px; display: flex; flex-direction: column; overflow: hidden; }}
    .review-topbar {{ display: flex; align-items: flex-start; justify-content: space-between; gap: 14px; margin-bottom: 10px; }}
    .review-title {{ min-width: 0; }}
    .review-title h2 {{ margin: 0 0 4px; font-size: 20px; overflow-wrap: anywhere; }}
    .nav-actions {{ display: flex; gap: 8px; flex: 0 0 auto; }}
    .comparison-grid {{ flex: 1 1 auto; min-height: 0; display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); grid-template-rows: minmax(120px, 0.8fr) minmax(170px, 1fr); gap: 14px; overflow: hidden; }}
    .box {{ min-height: 0; overflow: auto; background: #2b2c30; border: 1px solid #44474d; border-radius: 8px; padding: 14px; white-space: pre-wrap; line-height: 1.55; overflow-wrap: anywhere; }}
    .source {{ grid-column: 1 / -1; }}
    .actions {{ position: sticky; bottom: 0; z-index: 2; display: flex; flex-wrap: wrap; gap: 10px; margin: 12px 0 0; padding: 12px 0; background: #202124; }}
    .approve {{ border-color: #81c995; }}
    .changed {{ border-color: #fdd663; }}
    .reject {{ border-color: #f28b82; }}
    textarea {{ width: 100%; min-height: 72px; max-height: 120px; resize: vertical; background: #151618; color: #dfe3ea; border: 1px solid #44474d; border-radius: 8px; padding: 10px; }}
    h3 {{ margin: 6px 0 8px; font-size: 16px; }}
    .meta {{ color: #bdc1c6; font-size: 13px; margin-top: 4px; }}
  </style>
</head>
<body>
<header>
  <h1>Gemma Translation Review</h1>
  <div class="meta">로컬 전용 리뷰 보드입니다. 이 HTML은 /tmp 검증 폴더에만 생성하세요.</div>
</header>
<main>
  <aside>
    <div id="summary"></div>
    <label for="filterMode" class="meta">필터</label>
    <select id="filterMode">
      <option value="all">전체</option>
      <option value="unreviewed">미검토</option>
      <option value="reviewed">검토 완료</option>
      <option value="rejected">불합격</option>
      <option value="approved">승인/통과</option>
    </select>
    <div id="list"></div>
  </aside>
  <section class="panel">
    <div class="review-topbar">
      <div class="review-title">
        <h2 id="title"></h2>
        <div id="reason" class="meta"></div>
      </div>
      <div class="nav-actions">
        <button id="previousSample">이전</button>
        <button id="nextSample">다음</button>
      </div>
    </div>
    <div class="comparison-grid">
      <div class="box source"><strong>원문 OCR</strong><br><br><span id="source"></span></div>
      <div class="box"><strong>기존 번역</strong><br><br><span id="baseline"></span></div>
      <div class="box"><strong>후보 번역</strong><br><br><span id="candidate"></span></div>
    </div>
    <div class="actions">
      <button class="approve" data-decision="identical">완전 동일/통과</button>
      <button class="changed" data-decision="approved_changed">달라졌지만 승인</button>
      <button class="reject" data-decision="rejected">불합격</button>
      <button data-decision="hold">보류</button>
    </div>
    <h3>decisions.json</h3>
    <textarea id="decisions" spellcheck="false"></textarea>
    <div class="actions">
      <button id="download">Download decisions.json</button>
    </div>
  </section>
</main>
<script>
const ITEMS = {data_json};
let current = 0;
let decisions = {{}};
try {{ decisions = JSON.parse(localStorage.getItem("gemma_review_decisions") || "{{}}"); }} catch (err) {{ decisions = {{}}; }}

function escapeText(value) {{
  return String(value || "");
}}

function filteredItems() {{
  const mode = document.getElementById("filterMode").value || "all";
  return ITEMS.map((item, index) => ({{item, index}})).filter((entry) => {{
    const decision = decisions[entry.item.id] && decisions[entry.item.id].decision;
    if (mode === "unreviewed") return !decision;
    if (mode === "reviewed") return !!decision;
    if (mode === "rejected") return decision === "rejected";
    if (mode === "approved") return decision === "identical" || decision === "approved_changed";
    return true;
  }});
}}

function currentVisiblePosition(entries) {{
  const found = entries.findIndex((entry) => entry.index === current);
  return found >= 0 ? found : 0;
}}

function render() {{
  const visible = filteredItems();
  if (visible.length && !visible.some((entry) => entry.index === current)) {{
    current = visible[0].index;
  }}
  const visiblePosition = currentVisiblePosition(visible);
  document.getElementById("summary").textContent = `Samples: ${{ITEMS.length}} / Visible: ${{visible.length}} / Decisions: ${{Object.keys(decisions).length}}`;
  const list = document.getElementById("list");
  list.innerHTML = "";
  visible.forEach((entry, visibleIndex) => {{
    const item = entry.item;
    const index = entry.index;
    const btn = document.createElement("button");
    btn.className = "item" + (index === current ? " active" : "");
    btn.textContent = `${{visibleIndex + 1}}. ${{item.id}} ${{decisions[item.id] ? "[" + decisions[item.id].decision + "]" : ""}}`;
    btn.onclick = () => {{ current = index; render(); }};
    list.appendChild(btn);
  }});
  const item = ITEMS[current] || {{id: "-", reason: "", source: "", baseline: "", candidate: ""}};
  document.getElementById("title").textContent = `샘플 ${{visiblePosition + 1}} / ${{visible.length}}: ${{item.id}}`;
  document.getElementById("reason").textContent = item.reason || "";
  document.getElementById("source").textContent = escapeText(item.source);
  document.getElementById("baseline").textContent = escapeText(item.baseline);
  document.getElementById("candidate").textContent = escapeText(item.candidate);
  document.getElementById("decisions").value = JSON.stringify(decisions, null, 2);
  document.getElementById("previousSample").disabled = visiblePosition <= 0;
  document.getElementById("nextSample").disabled = visiblePosition >= visible.length - 1;
}}

function setCurrent(index) {{
  const visible = filteredItems();
  if (!visible.length) {{
    current = 0;
    render();
    return;
  }}
  const requestedPosition = visible.findIndex((entry) => entry.index === index);
  current = visible[Math.max(0, requestedPosition)].index;
  render();
}}

function moveVisible(delta) {{
  const visible = filteredItems();
  if (!visible.length) {{
    current = 0;
    render();
    return;
  }}
  const currentPosition = currentVisiblePosition(visible);
  const nextPosition = Math.max(0, Math.min(currentPosition + delta, visible.length - 1));
  current = visible[nextPosition].index;
  render();
}}

async function persist() {{
  localStorage.setItem("gemma_review_decisions", JSON.stringify(decisions));
  document.getElementById("decisions").value = JSON.stringify(decisions, null, 2);
  try {{
    await fetch("/decisions", {{
      method: "POST",
      headers: {{"Content-Type": "application/json"}},
      body: JSON.stringify(decisions)
    }});
  }} catch (err) {{
    // file:// mode cannot write decisions.json; use the download button.
  }}
}}

document.querySelectorAll("[data-decision]").forEach((button) => {{
  button.onclick = async () => {{
    const item = ITEMS[current];
    if (!item) return;
    decisions[item.id] = {{
      decision: button.dataset.decision,
      reviewed_at: new Date().toISOString()
    }};
    await persist();
    setCurrent(current + 1);
  }};
}});

document.getElementById("previousSample").onclick = () => moveVisible(-1);
document.getElementById("nextSample").onclick = () => moveVisible(1);
document.getElementById("filterMode").onchange = () => render();
document.addEventListener("keydown", (event) => {{
  const tag = String(event.target && event.target.tagName || "").toLowerCase();
  if (tag === "textarea" || tag === "input") return;
  if (event.key === "ArrowLeft" || event.key === "k") {{
    event.preventDefault();
    moveVisible(-1);
  }}
  if (event.key === "ArrowRight" || event.key === "j") {{
    event.preventDefault();
    moveVisible(1);
  }}
}});

document.getElementById("download").onclick = () => {{
  const blob = new Blob([JSON.stringify(decisions, null, 2)], {{type: "application/json"}});
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "decisions.json";
  a.click();
  URL.revokeObjectURL(url);
}};

render();
</script>
</body>
</html>
"""
    (review_dir / "index.html").write_text(index_html, encoding="utf-8")
    return {
        "review_dir": str(review_dir),
        "index_html": str(review_dir / "index.html"),
        "server": str(review_dir / "server.py"),
        "decision_path": str(review_dir / "decisions.json"),
        "sample_count": len(items),
        "privacy": {
            "raw_review_text_written": True,
            "git_safe": False,
        },
    }


def _sample_blocks(item: dict[str, Any]) -> list[str]:
    blocks = item.get("blocks")
    if isinstance(blocks, list):
        return [str(block) for block in blocks]
    return [str(item.get("source") or "")]


def _sample_saved_translations(item: dict[str, Any], block_count: int) -> list[str | None]:
    saved = item.get("saved_translations")
    if not isinstance(saved, list):
        return [None] * block_count
    result = [str(value) if value is not None else "" for value in saved[:block_count]]
    while len(result) < block_count:
        result.append(None)
    return result


def _shadow_text_blocks(blocks: list[str]) -> list[TextBlock]:
    result: list[TextBlock] = []
    for text in blocks:
        block = TextBlock()
        block.text = str(text or "")
        result.append(block)
    return result


def _chunk_reason(blocks: list[TextBlock], *, base_reason: str = "") -> str:
    sensitive_count = sum(1 for block in blocks if SENSITIVE_SOURCE_RE.search(str(getattr(block, "text", "") or "")))
    total_chars = sum(len(str(getattr(block, "text", "") or "")) for block in blocks)
    labels: list[str] = []
    if base_reason:
        labels.append(base_reason)
    if sensitive_count:
        labels.append(f"sensitive_terms={sensitive_count}")
    if len(blocks) > 1:
        labels.append(f"context_blocks={len(blocks)}")
    labels.append(f"chars={total_chars}")
    return ", ".join(labels)


def select_canary_samples(
    input_root: Path,
    output_dir: Path,
    defaults: ValidationSettings,
    *,
    project_count: int,
    max_project_blocks: int,
    max_samples_per_project: int,
    source_filter: str = "",
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    series_paths = sorted(input_root.rglob("*.seriesctpr"))
    project_rows: list[dict[str, Any]] = []
    normalized_source_filter = source_filter.casefold().strip()

    for series_index, series_path in enumerate(series_paths):
        try:
            series_state = load_series_project(str(series_path))
        except Exception:
            continue
        for project_index, item in enumerate(series_state.get("items") or []):
            if normalized_source_filter:
                source_metadata = " ".join(
                    str(item.get(key) or "")
                    for key in ("display_name", "source_origin_path", "source_origin_relpath")
                ).casefold()
                if normalized_source_filter not in source_metadata:
                    continue
            try:
                project_blob = _load_series_project_blob_for_item(series_path, item)
                pages = list(_iter_project_pages(project_blob))
            except Exception:
                continue

            chunks: list[dict[str, Any]] = []
            total_blocks = 0
            translated_blocks = 0
            sensitive_blocks = 0
            for page_index, page_path, page_state, extra_context in pages:
                blocks = [
                    block
                    for block in (_to_text_block(value) for value in (page_state.get("blk_list") or []))
                    if block is not None
                ]
                if not blocks:
                    continue
                page_settings = _settings_from_page(page_state, defaults)
                total_blocks += len(blocks)
                translated_blocks += sum(1 for block in blocks if str(getattr(block, "translation", "") or "").strip())
                for chunk_start in range(0, len(blocks), page_settings.chunk_size):
                    chunk = blocks[chunk_start : chunk_start + page_settings.chunk_size]
                    sensitive_count = sum(
                        1
                        for block in chunk
                        if SENSITIVE_SOURCE_RE.search(str(getattr(block, "text", "") or ""))
                    )
                    sensitive_blocks += sensitive_count
                    chunks.append(
                        {
                            "page_index": page_index,
                            "chunk_start": chunk_start,
                            "blocks": chunk,
                            "page_state": page_state,
                            "extra_context": extra_context,
                            "sensitive_count": sensitive_count,
                            "total_chars": sum(len(str(getattr(block, "text", "") or "")) for block in chunk),
                        }
                    )

            if total_blocks <= 0 or translated_blocks <= 0:
                continue
            if total_blocks > max_project_blocks:
                continue
            project_rows.append(
                {
                    "series_index": series_index,
                    "series_path": series_path,
                    "project_index": project_index,
                    "item": item,
                    "project_id": _anonymous_id("item", series_index, project_index, item.get("series_item_id")),
                    "pages": len(pages),
                    "blocks": total_blocks,
                    "translated_blocks": translated_blocks,
                    "sensitive_blocks": sensitive_blocks,
                    "chunks": chunks,
                }
            )

    project_rows.sort(key=lambda row: (row["sensitive_blocks"] == 0, -row["sensitive_blocks"], row["blocks"]))
    selected_projects = project_rows[: max(0, project_count)]
    samples: list[dict[str, Any]] = []
    selected_summary: list[dict[str, Any]] = []

    for project in selected_projects:
        chunks = list(project["chunks"])
        chunks.sort(
            key=lambda chunk: (
                chunk["sensitive_count"] == 0,
                -chunk["sensitive_count"],
                -len(chunk["blocks"]),
                -chunk["total_chars"],
            )
        )
        selected_chunks = chunks[: max(1, max_samples_per_project)]
        selected_summary.append(
            {
                "project_id": project["project_id"],
                "series_index": project["series_index"],
                "project_index": project["project_index"],
                "pages": project["pages"],
                "blocks": project["blocks"],
                "translated_blocks": project["translated_blocks"],
                "sensitive_blocks": project["sensitive_blocks"],
                "sample_chunks": len(selected_chunks),
            }
        )
        for chunk_index, chunk in enumerate(selected_chunks):
            page_state = chunk["page_state"]
            page_settings = _settings_from_page(page_state, defaults)
            blocks = chunk["blocks"]
            samples.append(
                {
                    "id": f"{project['project_id']}:p{chunk['page_index']:04d}:c{chunk['chunk_start']:03d}",
                    "project_id": project["project_id"],
                    "reason": _chunk_reason(blocks, base_reason=f"project_canary_{chunk_index + 1}"),
                    "source_lang": page_settings.source_lang,
                    "target_lang": page_settings.target_lang,
                    "extra_context": chunk["extra_context"],
                    "blocks": [str(getattr(block, "text", "") or "") for block in blocks],
                    "saved_translations": [str(getattr(block, "translation", "") or "") for block in blocks],
                }
            )

    samples_path = output_dir / "canary_samples.json"
    summary_path = output_dir / "canary_summary.json"
    samples_path.write_text(json.dumps({"samples": samples}, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {
        "samples_path": str(samples_path),
        "selected_projects": selected_summary,
        "sample_count": len(samples),
        "available_project_count": len(project_rows),
        "privacy": {
            "raw_canary_text_written": True,
            "git_safe": False,
        },
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def select_sensitive_samples(
    input_root: Path,
    output_dir: Path,
    defaults: ValidationSettings,
    *,
    golden_limit: int,
    control_limit: int,
    source_filter: str = "",
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    series_paths = sorted(input_root.rglob("*.seriesctpr"))
    normalized_source_filter = source_filter.casefold().strip()
    sensitive_rows: list[dict[str, Any]] = []
    control_rows: list[dict[str, Any]] = []
    counts = {
        "series_files": 0,
        "projects": 0,
        "pages": 0,
        "chunks": 0,
        "sensitive_chunks": 0,
        "control_chunks": 0,
    }

    for series_index, series_path in enumerate(series_paths):
        counts["series_files"] += 1
        try:
            series_state = load_series_project(str(series_path))
        except Exception:
            continue
        for project_index, item in enumerate(series_state.get("items") or []):
            if normalized_source_filter:
                source_metadata = " ".join(
                    str(item.get(key) or "")
                    for key in ("display_name", "source_origin_path", "source_origin_relpath")
                ).casefold()
                if normalized_source_filter not in source_metadata:
                    continue
            counts["projects"] += 1
            try:
                project_blob = _load_series_project_blob_for_item(series_path, item)
                pages = list(_iter_project_pages(project_blob))
            except Exception:
                continue

            project_id = _anonymous_id("item", series_index, project_index, item.get("series_item_id"))
            for page_index, _page_path, page_state, extra_context in pages:
                counts["pages"] += 1
                blocks = [
                    block
                    for block in (_to_text_block(value) for value in (page_state.get("blk_list") or []))
                    if block is not None
                ]
                if not blocks:
                    continue
                page_settings = _settings_from_page(page_state, defaults)
                for chunk_start in range(0, len(blocks), page_settings.chunk_size):
                    chunk = blocks[chunk_start : chunk_start + page_settings.chunk_size]
                    sensitive_count = _sample_has_sensitive_trigger(chunk)
                    total_chars = sum(len(str(getattr(block, "text", "") or "")) for block in chunk)
                    row = {
                        "id": f"{project_id}:p{page_index:04d}:c{chunk_start:03d}",
                        "project_id": project_id,
                        "reason": _chunk_reason(
                            chunk,
                            base_reason="sensitive_golden" if sensitive_count else "control",
                        ),
                        "source_lang": page_settings.source_lang,
                        "target_lang": page_settings.target_lang,
                        "extra_context": extra_context,
                        "blocks": [str(getattr(block, "text", "") or "") for block in chunk],
                        "saved_translations": [str(getattr(block, "translation", "") or "") for block in chunk],
                        "sensitive_count": sensitive_count,
                        "total_chars": total_chars,
                        "block_count": len(chunk),
                    }
                    counts["chunks"] += 1
                    if sensitive_count:
                        counts["sensitive_chunks"] += 1
                        sensitive_rows.append(row)
                    else:
                        counts["control_chunks"] += 1
                        control_rows.append(row)

    sensitive_rows.sort(
        key=lambda row: (
            -int(row["sensitive_count"]),
            -int(row["block_count"]),
            -int(row["total_chars"]),
            str(row["id"]),
        )
    )
    control_rows.sort(
        key=lambda row: (
            -int(row["block_count"]),
            -int(row["total_chars"]),
            str(row["id"]),
        )
    )
    selected_sensitive = sensitive_rows[: max(0, golden_limit)]
    selected_control = control_rows[: max(0, control_limit)]
    samples = selected_sensitive + selected_control

    samples_path = output_dir / "sensitive_samples.json"
    summary_path = output_dir / "sensitive_summary.json"
    samples_path.write_text(json.dumps({"samples": samples}, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {
        "samples_path": str(samples_path),
        "sample_count": len(samples),
        "sensitive_sample_count": len(selected_sensitive),
        "control_sample_count": len(selected_control),
        "counts": counts,
        "selected": [
            {
                "id": row["id"],
                "reason": row["reason"],
                "block_count": row["block_count"],
                "total_chars": row["total_chars"],
                "sensitive_count": row["sensitive_count"],
            }
            for row in samples
        ],
        "privacy": {
            "raw_sample_text_written": True,
            "summary_contains_raw_text": False,
            "git_safe": False,
        },
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def _usage_from_response(response_data: dict[str, Any]) -> dict[str, int]:
    usage = response_data.get("usage") or {}
    return {
        "prompt_tokens": _safe_int(usage.get("prompt_tokens"), 0),
        "completion_tokens": _safe_int(usage.get("completion_tokens"), 0),
        "total_tokens": _safe_int(usage.get("total_tokens"), 0),
    }


def _candidate_result_template(candidate: SamplerCandidate) -> dict[str, Any]:
    return {
        "name": candidate.name,
        "temperature": candidate.temperature,
        "top_k": candidate.top_k,
        "top_p": candidate.top_p,
        "samples": 0,
        "blocks": 0,
        "failed_samples": 0,
        "missing_translation": 0,
        "empty_translation": 0,
        "channel_token_residue": 0,
        "known_bad_output_count": 0,
        "wall_sec": 0.0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "completion_tps": 0.0,
        "network_request_count": 0,
        "fallback_count": 0,
        "chunk_retry_events": 0,
        "truncated_count": 0,
    }


def run_sensitive_sampler_matrix(
    samples_path: Path,
    output_dir: Path,
    settings: ValidationSettings,
    *,
    api_base_url: str,
    timeout: float,
    candidates: tuple[SamplerCandidate, ...] = DEFAULT_SENSITIVE_SAMPLER_CANDIDATES,
) -> dict[str, Any]:
    raw_data = json.loads(samples_path.read_text(encoding="utf-8"))
    samples = raw_data if isinstance(raw_data, list) else raw_data.get("samples", [])
    output_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    review_items: list[dict[str, Any]] = []
    candidate_summaries = {
        candidate.name: _candidate_result_template(candidate)
        for candidate in candidates
    }
    counters = {
        "samples": 0,
        "blocks": 0,
        "failed_samples": 0,
    }
    observed_prompt_prefix_hashes: set[str] = set()
    baseline_outputs: dict[str, str] = {}

    for sample_index, item in enumerate(samples):
        if not isinstance(item, dict):
            continue
        counters["samples"] += 1
        sample_id = str(item.get("id") or f"sample-{sample_index + 1:03d}")
        blocks_text = _sample_blocks(item)
        counters["blocks"] += len(blocks_text)
        blocks = _shadow_text_blocks(blocks_text)
        sample_context = str(item.get("extra_context") or "")
        source_lang = str(item.get("source_lang") or settings.source_lang)
        target_lang = str(item.get("target_lang") or settings.target_lang)
        expected_keys = [f"block_{index}" for index in range(len(blocks))]
        sample_result = {
            "id": sample_id,
            "reason": str(item.get("reason") or ""),
            "blocks": [],
        }

        for candidate in candidates:
            candidate_settings = _settings_with_sampler(settings, candidate)
            engine = _engine_for(candidate_settings)
            engine.api_base_url = api_base_url.rstrip("/")
            engine.timeout = timeout
            engine.exact_prompt_cache_enabled = False
            engine.source_lang = source_lang
            engine.target_lang = target_lang
            system_prompt = engine._build_system_prompt(sample_context, prompt_profile=candidate_settings.prompt_profile)
            prefix_hash = _prompt_prefix_hash(system_prompt)
            observed_prompt_prefix_hashes.add(prefix_hash)
            measured_usage = {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            }
            original_request = engine._request_translation

            def measured_request(
                system_prompt: str,
                user_prompt: str,
                *,
                expected_keys: list[str] | None = None,
            ) -> dict:
                response_data = original_request(system_prompt, user_prompt, expected_keys=expected_keys)
                usage = _usage_from_response(response_data)
                for usage_key, usage_value in usage.items():
                    measured_usage[usage_key] += usage_value
                return response_data

            engine._request_translation = measured_request
            started = time.perf_counter()
            candidate_summary = candidate_summaries[candidate.name]
            candidate_summary["samples"] += 1
            candidate_summary["blocks"] += len(blocks)
            try:
                translated_blocks = _shadow_text_blocks(blocks_text)
                engine.translate(translated_blocks, None, sample_context)
                elapsed = time.perf_counter() - started
                candidate_summary["wall_sec"] += elapsed
                for key, value in measured_usage.items():
                    candidate_summary[key] += value
                stats = engine.last_benchmark_stats
                candidate_summary["network_request_count"] += int(stats.get("gemma_network_request_count", 0))
                candidate_summary["fallback_count"] += int(stats.get("gemma_contextual_merge_fallback_count", 0))
                candidate_summary["chunk_retry_events"] += int(stats.get("gemma_chunk_retry_events", 0))
                candidate_summary["truncated_count"] += int(stats.get("gemma_truncated_count", 0))
                translations = {
                    f"block_{index}": str(getattr(block, "translation", "") or "")
                    for index, block in enumerate(translated_blocks)
                }
            except Exception as exc:
                elapsed = time.perf_counter() - started
                candidate_summary["wall_sec"] += elapsed
                for key, value in measured_usage.items():
                    candidate_summary[key] += value
                counters["failed_samples"] += 1
                candidate_summary["failed_samples"] += 1
                sample_result.setdefault("errors", []).append(
                    {
                        "candidate": candidate.name,
                        "error_type": type(exc).__name__,
                    }
                )
                continue
            finally:
                engine._request_translation = original_request

            for block_index, source_text in enumerate(blocks_text):
                key = f"block_{block_index}"
                raw_value = translations.get(key)
                value = engine._strip_channel_tokens(raw_value or "")
                block_result = {
                    "candidate": candidate.name,
                    "block_key": key,
                    "source_hash": _sha256_text(source_text),
                    "translation_hash": _sha256_text(value),
                    "translation_length": len(value),
                    "known_bad_output": _has_known_bad_output(raw_value) or _has_known_bad_output(value),
                    "empty_translation": bool(str(source_text or "").strip()) and not bool(value.strip()),
                    "channel_token_residue": bool(KNOWN_BAD_OUTPUT_RE.search(str(raw_value or "")) and "<" in str(raw_value or "")),
                }
                if block_result["known_bad_output"]:
                    candidate_summary["known_bad_output_count"] += 1
                if block_result["empty_translation"]:
                    candidate_summary["empty_translation"] += 1
                if block_result["channel_token_residue"]:
                    candidate_summary["channel_token_residue"] += 1
                if candidate.name == "baseline":
                    baseline_outputs[f"{sample_id}:{key}"] = value
                else:
                    baseline_value = baseline_outputs.get(f"{sample_id}:{key}")
                    if baseline_value is not None and baseline_value != value:
                        review_items.append(
                            {
                                "id": f"{sample_id}:{key}:{candidate.name}",
                                "candidate_name": candidate.name,
                                "reason": f"{item.get('reason') or ''}, sampler={candidate.name}".strip(", "),
                                "source": source_text,
                                "baseline": baseline_value,
                                "candidate": value,
                            }
                        )
                sample_result["blocks"].append(block_result)
        results.append(sample_result)

    for candidate_summary in candidate_summaries.values():
        completion_tokens = int(candidate_summary["completion_tokens"])
        wall_sec = float(candidate_summary["wall_sec"])
        candidate_summary["wall_sec"] = round(wall_sec, 3)
        candidate_summary["completion_tps"] = round(completion_tokens / wall_sec, 3) if wall_sec > 0 else 0.0

    eligible_candidates = [
        row
        for row in candidate_summaries.values()
        if row["failed_samples"] == 0
        and row["missing_translation"] == 0
        and row["empty_translation"] == 0
        and row["channel_token_residue"] == 0
        and row["known_bad_output_count"] == 0
    ]
    baseline_bad = candidate_summaries.get("baseline", {}).get("known_bad_output_count", 0)
    if baseline_bad and eligible_candidates:
        decision = "human_review_required_for_sensitive_slow_lane"
        preferred = min(
            (row for row in eligible_candidates if row["name"] != "baseline"),
            key=lambda row: (row["wall_sec"], row["temperature"]),
            default=None,
        )
    elif eligible_candidates:
        decision = "keep_current_fast_multi"
        preferred = candidate_summaries.get("baseline")
    else:
        decision = "no_sampler_candidate_passed_auto_gate"
        preferred = None

    raw_result_path = output_dir / "sensitive_sampler_matrix_results.json"
    summary_path = output_dir / "sensitive_sampler_matrix_summary.json"
    review_diff_path = output_dir / "sensitive_sampler_review_diff.json"
    raw_result_path.write_text(json.dumps({"samples": results}, ensure_ascii=False, indent=2), encoding="utf-8")
    review_diff_path.write_text(json.dumps({"items": review_items}, ensure_ascii=False, indent=2), encoding="utf-8")
    prompt_prefix_changed = observed_prompt_prefix_hashes != {EXPECTED_GEMMA_PROMPT_PREFIX_HASH}
    summary = {
        "counters": counters,
        "candidates": candidate_summaries,
        "prompt_prefix": {
            "expected_hash": EXPECTED_GEMMA_PROMPT_PREFIX_HASH,
            "observed_hashes": sorted(observed_prompt_prefix_hashes),
            "changed": prompt_prefix_changed,
        },
        "auto_recommendation": {
            "decision": "prompt_prefix_changed_stop" if prompt_prefix_changed else decision,
            "preferred_candidate": preferred["name"] if preferred else None,
        },
        "raw_result_path": str(raw_result_path),
        "review_diff_path": str(review_diff_path),
        "privacy": {
            "raw_matrix_text_written": True,
            "git_safe": False,
        },
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def run_fast_multi_shadow(
    samples_path: Path,
    output_dir: Path,
    settings: ValidationSettings,
    *,
    api_base_url: str,
    timeout: float,
    extra_context: str = "",
) -> dict[str, Any]:
    raw_data = json.loads(samples_path.read_text(encoding="utf-8"))
    samples = raw_data if isinstance(raw_data, list) else raw_data.get("samples", [])
    output_dir.mkdir(parents=True, exist_ok=True)

    engine = _engine_for(settings)
    engine.api_base_url = api_base_url.rstrip("/")
    engine.timeout = timeout
    engine.exact_prompt_cache_enabled = False

    results: list[dict[str, Any]] = []
    review_items: list[dict[str, Any]] = []
    saved_review_items: list[dict[str, Any]] = []
    counters = {
        "samples": 0,
        "blocks": 0,
        "saved_changed_blocks": 0,
        "changed_blocks": 0,
        "failed_samples": 0,
    }

    for sample_index, item in enumerate(samples):
        if not isinstance(item, dict):
            continue
        counters["samples"] += 1
        sample_id = str(item.get("id") or f"sample-{sample_index + 1:03d}")
        reason = str(item.get("reason") or "")
        blocks_text = _sample_blocks(item)
        saved_translations = _sample_saved_translations(item, len(blocks_text))
        blocks = _shadow_text_blocks(blocks_text)
        sample_context = str(item.get("extra_context") or extra_context or "")
        source_lang = str(item.get("source_lang") or settings.source_lang)
        target_lang = str(item.get("target_lang") or settings.target_lang)
        engine.source_lang = source_lang
        engine.target_lang = target_lang
        system_prompt = engine._build_system_prompt(sample_context, prompt_profile=settings.prompt_profile)
        expected_keys = engine._expected_block_keys(blocks)

        sample_result: dict[str, Any] = {
            "id": sample_id,
            "reason": reason,
            "source_lang": source_lang,
            "target_lang": target_lang,
            "blocks": [],
        }
        try:
            single_values: list[str] = []
            for block_index, _block in enumerate(blocks):
                user_prompt = engine._build_contextual_single_block_user_prompt(blocks, block_index)
                response_data = engine._request_translation(
                    system_prompt,
                    user_prompt,
                    expected_keys=["translation"],
                )
                parsed = engine._extract_translation_dict(
                    response_data,
                    expected_keys=["translation"],
                    block_count=1,
                    prompt_profile=settings.prompt_profile,
                )
                single_values.append(engine._strip_channel_tokens(parsed.get("translation", "")))

            multi_prompt = engine._build_contextual_merged_user_prompt(blocks, expected_keys)
            multi_response = engine._request_translation(
                system_prompt,
                multi_prompt,
                expected_keys=expected_keys,
            )
            multi_values = engine._extract_translation_dict(
                multi_response,
                expected_keys=expected_keys,
                block_count=len(blocks),
                prompt_profile=settings.prompt_profile,
            )

            for block_index, source_text in enumerate(blocks_text):
                key = f"block_{block_index}"
                saved = saved_translations[block_index]
                single = single_values[block_index]
                multi = engine._strip_channel_tokens(multi_values.get(key, ""))
                saved_changed = saved is not None and saved != single
                changed = single != multi
                counters["blocks"] += 1
                if saved_changed:
                    counters["saved_changed_blocks"] += 1
                    saved_review_items.append(
                        {
                            "id": f"{sample_id}:{key}:saved-vs-single",
                            "reason": reason,
                            "source": source_text,
                            "baseline": saved,
                            "candidate": single,
                        }
                    )
                if changed:
                    counters["changed_blocks"] += 1
                    review_items.append(
                        {
                            "id": f"{sample_id}:{key}",
                            "reason": reason,
                            "source": source_text,
                            "baseline": single,
                            "candidate": multi,
                        }
                    )
                sample_result["blocks"].append(
                    {
                        "block_key": key,
                        "source": source_text,
                        "saved": saved,
                        "single": single,
                        "fast_multi": multi,
                        "saved_changed": saved_changed,
                        "changed": changed,
                        "source_hash": _sha256_text(source_text),
                        "saved_hash": _sha256_text(saved),
                        "single_hash": _sha256_text(single),
                        "fast_multi_hash": _sha256_text(multi),
                    }
                )
        except Exception as exc:
            counters["failed_samples"] += 1
            sample_result["error_type"] = type(exc).__name__
        results.append(sample_result)

    raw_result_path = output_dir / "fast_multi_shadow_results.json"
    summary_path = output_dir / "fast_multi_shadow_summary.json"
    review_diff_path = output_dir / "fast_multi_review_diff.json"
    saved_review_diff_path = output_dir / "saved_vs_single_review_diff.json"
    raw_result_path.write_text(json.dumps({"samples": results}, ensure_ascii=False, indent=2), encoding="utf-8")
    review_diff_path.write_text(json.dumps({"items": review_items}, ensure_ascii=False, indent=2), encoding="utf-8")
    saved_review_diff_path.write_text(
        json.dumps({"items": saved_review_items}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    summary = {
        "counters": counters,
        "raw_result_path": str(raw_result_path),
        "review_diff_path": str(review_diff_path),
        "saved_review_diff_path": str(saved_review_diff_path),
        "settings_hash": _settings_hash(settings),
        "privacy": {
            "raw_shadow_text_written": True,
            "git_safe": False,
        },
        "default_app_behavior_changed": False,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def _parse_settings(args: argparse.Namespace) -> ValidationSettings:
    return ValidationSettings(
        source_lang=str(args.source_lang),
        target_lang=str(args.target_lang),
        model=str(args.model),
        chunk_size=max(1, _safe_int(args.chunk_size, DEFAULT_GEMMA_CHUNK_SIZE)),
        max_tokens=max(1, _safe_int(args.max_tokens, DEFAULT_GEMMA_MAX_COMPLETION_TOKENS)),
        temperature=float(args.temperature),
        top_k=_safe_int(args.top_k, DEFAULT_GEMMA_TRANSLATION_TOP_K),
        top_p=float(args.top_p),
        min_p=float(args.min_p),
        prompt_profile=str(args.prompt_profile),
        response_format_mode=str(args.response_format_mode),
        response_schema_mode=str(args.response_schema_mode),
    )


def _default_output_dir() -> Path:
    return Path(tempfile.gettempdir()) / "gemma_exact_validation"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Local-only Gemma exactness validation tools.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    baseline = subparsers.add_parser("baseline", help="Build hash-only baseline from .seriesctpr files.")
    baseline.add_argument("--input-root", required=True, type=Path)
    baseline.add_argument("--output-dir", type=Path, default=_default_output_dir())
    baseline.add_argument("--source-lang", default="Chinese")
    baseline.add_argument("--target-lang", default="Korean")
    baseline.add_argument("--model", default="gemma-4-26B-IQ4_NL.gguf")
    baseline.add_argument("--chunk-size", default=DEFAULT_GEMMA_CHUNK_SIZE)
    baseline.add_argument("--max-tokens", default=DEFAULT_GEMMA_MAX_COMPLETION_TOKENS)
    baseline.add_argument("--temperature", default=DEFAULT_GEMMA_TRANSLATION_TEMPERATURE)
    baseline.add_argument("--top-k", default=DEFAULT_GEMMA_TRANSLATION_TOP_K)
    baseline.add_argument("--top-p", default=DEFAULT_GEMMA_TRANSLATION_TOP_P)
    baseline.add_argument("--min-p", default=DEFAULT_GEMMA_TRANSLATION_MIN_P)
    baseline.add_argument("--prompt-profile", default=DEFAULT_GEMMA_PROMPT_PROFILE)
    baseline.add_argument("--response-format-mode", default=DEFAULT_GEMMA_RESPONSE_FORMAT_MODE)
    baseline.add_argument("--response-schema-mode", default=DEFAULT_GEMMA_RESPONSE_SCHEMA_MODE)

    compare = subparsers.add_parser("compare", help="Compare two hash-only baseline JSONL files.")
    compare.add_argument("--baseline", required=True, type=Path)
    compare.add_argument("--candidate", required=True, type=Path)
    compare.add_argument("--output-dir", type=Path, default=_default_output_dir())

    review = subparsers.add_parser("review-board", help="Create a local-only human review board from raw local diffs.")
    review.add_argument("--diff", required=True, type=Path)
    review.add_argument("--output-dir", type=Path, default=_default_output_dir())

    canary = subparsers.add_parser("select-canary", help="Create local-only canary samples from private .seriesctpr projects.")
    canary.add_argument("--input-root", required=True, type=Path)
    canary.add_argument("--output-dir", type=Path, default=_default_output_dir())
    canary.add_argument("--project-count", default=3, type=int)
    canary.add_argument("--max-project-blocks", default=300, type=int)
    canary.add_argument("--max-samples-per-project", default=8, type=int)
    canary.add_argument(
        "--source-filter",
        default="",
        help="Only select series items whose display name or source path contains this text.",
    )
    canary.add_argument("--source-lang", default="Chinese")
    canary.add_argument("--target-lang", default="Korean")
    canary.add_argument("--model", default="gemma-4-26B-IQ4_NL.gguf")
    canary.add_argument("--chunk-size", default=DEFAULT_GEMMA_CHUNK_SIZE)
    canary.add_argument("--max-tokens", default=DEFAULT_GEMMA_MAX_COMPLETION_TOKENS)
    canary.add_argument("--temperature", default=DEFAULT_GEMMA_TRANSLATION_TEMPERATURE)
    canary.add_argument("--top-k", default=DEFAULT_GEMMA_TRANSLATION_TOP_K)
    canary.add_argument("--top-p", default=DEFAULT_GEMMA_TRANSLATION_TOP_P)
    canary.add_argument("--min-p", default=DEFAULT_GEMMA_TRANSLATION_MIN_P)
    canary.add_argument("--prompt-profile", default=DEFAULT_GEMMA_PROMPT_PROFILE)
    canary.add_argument("--response-format-mode", default=DEFAULT_GEMMA_RESPONSE_FORMAT_MODE)
    canary.add_argument("--response-schema-mode", default=DEFAULT_GEMMA_RESPONSE_SCHEMA_MODE)

    sensitive = subparsers.add_parser("select-sensitive", help="Create local-only sensitive trigger and control samples.")
    sensitive.add_argument("--input-root", required=True, type=Path)
    sensitive.add_argument("--output-dir", type=Path, default=_default_output_dir())
    sensitive.add_argument("--golden-limit", default=40, type=int)
    sensitive.add_argument("--control-limit", default=20, type=int)
    sensitive.add_argument(
        "--source-filter",
        default="",
        help="Only select series items whose display name or source path contains this text.",
    )
    sensitive.add_argument("--source-lang", default="English")
    sensitive.add_argument("--target-lang", default="Korean")
    sensitive.add_argument("--model", default="gemma-4-26B-IQ4_NL.gguf")
    sensitive.add_argument("--chunk-size", default=DEFAULT_GEMMA_CHUNK_SIZE)
    sensitive.add_argument("--max-tokens", default=DEFAULT_GEMMA_MAX_COMPLETION_TOKENS)
    sensitive.add_argument("--temperature", default=DEFAULT_GEMMA_TRANSLATION_TEMPERATURE)
    sensitive.add_argument("--top-k", default=DEFAULT_GEMMA_TRANSLATION_TOP_K)
    sensitive.add_argument("--top-p", default=DEFAULT_GEMMA_TRANSLATION_TOP_P)
    sensitive.add_argument("--min-p", default=DEFAULT_GEMMA_TRANSLATION_MIN_P)
    sensitive.add_argument("--prompt-profile", default=DEFAULT_GEMMA_PROMPT_PROFILE)
    sensitive.add_argument("--response-format-mode", default=DEFAULT_GEMMA_RESPONSE_FORMAT_MODE)
    sensitive.add_argument("--response-schema-mode", default=DEFAULT_GEMMA_RESPONSE_SCHEMA_MODE)

    shadow = subparsers.add_parser("fast-multi-shadow", help="Run current single-block and fast multi translations for local samples.")
    shadow.add_argument("--samples", required=True, type=Path)
    shadow.add_argument("--output-dir", type=Path, default=_default_output_dir())
    shadow.add_argument("--api-base-url", default="http://127.0.0.1:18080/v1")
    shadow.add_argument("--timeout", default=180.0, type=float)
    shadow.add_argument("--extra-context", default="")
    shadow.add_argument("--source-lang", default="English")
    shadow.add_argument("--target-lang", default="Korean")
    shadow.add_argument("--model", default="gemma-4-26B-IQ4_NL.gguf")
    shadow.add_argument("--chunk-size", default=DEFAULT_GEMMA_CHUNK_SIZE)
    shadow.add_argument("--max-tokens", default=DEFAULT_GEMMA_MAX_COMPLETION_TOKENS)
    shadow.add_argument("--temperature", default=DEFAULT_GEMMA_TRANSLATION_TEMPERATURE)
    shadow.add_argument("--top-k", default=DEFAULT_GEMMA_TRANSLATION_TOP_K)
    shadow.add_argument("--top-p", default=DEFAULT_GEMMA_TRANSLATION_TOP_P)
    shadow.add_argument("--min-p", default=DEFAULT_GEMMA_TRANSLATION_MIN_P)
    shadow.add_argument("--prompt-profile", default=DEFAULT_GEMMA_PROMPT_PROFILE)
    shadow.add_argument("--response-format-mode", default=DEFAULT_GEMMA_RESPONSE_FORMAT_MODE)
    shadow.add_argument("--response-schema-mode", default=DEFAULT_GEMMA_RESPONSE_SCHEMA_MODE)

    matrix = subparsers.add_parser("sensitive-matrix", help="Run sampler candidates for local sensitive samples.")
    matrix.add_argument("--samples", required=True, type=Path)
    matrix.add_argument("--output-dir", type=Path, default=_default_output_dir())
    matrix.add_argument("--api-base-url", default="http://127.0.0.1:18080/v1")
    matrix.add_argument("--timeout", default=180.0, type=float)
    matrix.add_argument("--source-lang", default="English")
    matrix.add_argument("--target-lang", default="Korean")
    matrix.add_argument("--model", default="gemma-4-26B-IQ4_NL.gguf")
    matrix.add_argument("--chunk-size", default=DEFAULT_GEMMA_CHUNK_SIZE)
    matrix.add_argument("--max-tokens", default=DEFAULT_GEMMA_MAX_COMPLETION_TOKENS)
    matrix.add_argument("--temperature", default=DEFAULT_GEMMA_TRANSLATION_TEMPERATURE)
    matrix.add_argument("--top-k", default=DEFAULT_GEMMA_TRANSLATION_TOP_K)
    matrix.add_argument("--top-p", default=DEFAULT_GEMMA_TRANSLATION_TOP_P)
    matrix.add_argument("--min-p", default=DEFAULT_GEMMA_TRANSLATION_MIN_P)
    matrix.add_argument("--prompt-profile", default=DEFAULT_GEMMA_PROMPT_PROFILE)
    matrix.add_argument("--response-format-mode", default=DEFAULT_GEMMA_RESPONSE_FORMAT_MODE)
    matrix.add_argument("--response-schema-mode", default=DEFAULT_GEMMA_RESPONSE_SCHEMA_MODE)

    args = parser.parse_args(argv)
    if args.command == "baseline":
        summary = build_baseline(args.input_root, args.output_dir, _parse_settings(args))
        print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    if args.command == "compare":
        result = compare_baselines(args.baseline, args.candidate, args.output_dir)
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 0 if result["gate_passed"] else 2
    if args.command == "review-board":
        result = make_review_board(args.diff, args.output_dir)
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    if args.command == "select-canary":
        result = select_canary_samples(
            args.input_root,
            args.output_dir,
            _parse_settings(args),
            project_count=args.project_count,
            max_project_blocks=args.max_project_blocks,
            max_samples_per_project=args.max_samples_per_project,
            source_filter=args.source_filter,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 0 if result["sample_count"] > 0 else 2
    if args.command == "select-sensitive":
        result = select_sensitive_samples(
            args.input_root,
            args.output_dir,
            _parse_settings(args),
            golden_limit=args.golden_limit,
            control_limit=args.control_limit,
            source_filter=args.source_filter,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 0 if result["sample_count"] > 0 else 2
    if args.command == "fast-multi-shadow":
        result = run_fast_multi_shadow(
            args.samples,
            args.output_dir,
            _parse_settings(args),
            api_base_url=args.api_base_url,
            timeout=args.timeout,
            extra_context=args.extra_context,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 0 if result["counters"]["failed_samples"] == 0 else 2
    if args.command == "sensitive-matrix":
        result = run_sensitive_sampler_matrix(
            args.samples,
            args.output_dir,
            _parse_settings(args),
            api_base_url=args.api_base_url,
            timeout=args.timeout,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 0 if result["auto_recommendation"]["preferred_candidate"] and not result["prompt_prefix"]["changed"] else 2
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

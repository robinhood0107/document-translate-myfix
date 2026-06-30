#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import sys
import time
import types
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("CT_DISABLE_UPDATE_CHECK", "1")
os.environ.setdefault("CT_ENABLE_MEMLOG", "1")
os.environ.setdefault("CT_ENABLE_GPU_BENCH", "1")

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "scripts"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import cv2
import numpy as np
import pypdfium2 as pdfium
from PySide6.QtWidgets import QApplication

from benchmark_common import (
    benchmark_output_root,
    load_preset,
    product_benchmark_failure_contract,
    render_summary_markdown,
    resolve_product_benchmark_contract,
    summarize_metrics,
    write_json,
)
from benchmark_pipeline import (
    _configure_window,
    _load_images,
    _log,
    _restore_env,
    _restore_settings,
    _settings_snapshot,
    _write_page_snapshots,
)
from modules.ocr.selection import STAGE_BATCHED_WORKFLOW_MODE
from modules.utils.correction_dictionary import apply_translation_result_dictionary
from modules.utils.translator_utils import get_raw_text, get_raw_translation


DEFAULT_SOURCE_ROOT = ROOT / "_local_no_gemma_replay_sources"
DEFAULT_PREVIOUS_RUN = DEFAULT_SOURCE_ROOT / "product_stage_batch_rerun_20260629_170647"
DEFAULT_PRESET = "workflow-split-runtime-stage-batched-single-ocr"


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    display_name: str
    snapshot_dir_name: str
    source_kind: str
    source_name: str
    source_lang: str
    target_lang: str = "Korean"


DATASETS: tuple[DatasetSpec, ...] = (
    DatasetSpec(
        key="coffee",
        display_name="Coffee v01 c01",
        snapshot_dir_name="leather_Coffee_v01_c01_product_stage_batch",
        source_kind="image_dir",
        source_name="tmp_Coffee v01 c01_9lciyeyn",
        source_lang="English",
    ),
    DatasetSpec(
        key="home",
        display_name="Home Visit SP02",
        snapshot_dir_name="leather_Home_Visit_SP02_product_stage_batch",
        source_kind="pdf",
        source_name="Home Visit SP02.pdf",
        source_lang="English",
    ),
    DatasetSpec(
        key="little",
        display_name="Little Sister SP06",
        snapshot_dir_name="leather_Little_Sister_SP06_product_stage_batch",
        source_kind="pdf",
        source_name="Little Sister SP06.pdf",
        source_lang="English",
    ),
    DatasetSpec(
        key="merry",
        display_name="Merry Mysteries SP02",
        snapshot_dir_name="leather_Merry_Mysteries_SP02_product_stage_batch",
        source_kind="pdf",
        source_name="Merry Mysteries SP02.pdf",
        source_lang="English",
    ),
    DatasetSpec(
        key="skinsuit",
        display_name="skinsuit needle v07 c02 영어",
        snapshot_dir_name="leather_skinsuit_needle_v07_c02_product_stage_batch",
        source_kind="zip",
        source_name="skinsuit needle v07 c02 영어.zip",
        source_lang="English",
    ),
    DatasetSpec(
        key="sample_japan",
        display_name="Sample/japan",
        snapshot_dir_name="sample_japan_product_stage_batch",
        source_kind="sample_japan",
        source_name="Sample/japan",
        source_lang="Japanese",
    ),
)


def _slug(text: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("_")
    return value or "dataset"


def _load_snapshot_pages(snapshot_path: Path) -> list[dict[str, Any]]:
    with snapshot_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    pages = payload.get("pages", []) if isinstance(payload, dict) else payload
    if not isinstance(pages, list):
        raise RuntimeError(f"Invalid page snapshot payload: {snapshot_path}")
    return [page for page in pages if isinstance(page, dict)]


def _page_image_names(pages: list[dict[str, Any]]) -> list[str]:
    names: list[str] = []
    for index, page in enumerate(pages, start=1):
        raw = str(page.get("image_name") or Path(str(page.get("image_path") or "")).name)
        if not raw:
            raw = f"{index:03d}.png"
        names.append(raw)
    return names


def _copy_image_unicode(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    image = cv2.imdecode(np.fromfile(str(src), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if image is None:
        shutil.copy2(src, dst)
        return
    ext = dst.suffix or ".png"
    ok, encoded = cv2.imencode(ext, image)
    if not ok:
        shutil.copy2(src, dst)
        return
    encoded.tofile(str(dst))


def _stage_from_image_dir(source_dir: Path, output_dir: Path, names: list[str]) -> list[Path]:
    staged: list[Path] = []
    for name in names:
        src = source_dir / name
        if not src.is_file():
            raise FileNotFoundError(f"Missing source image: {src}")
        dst = output_dir / name
        _copy_image_unicode(src, dst)
        staged.append(dst)
    return staged


def _stage_from_pdf(pdf_path: Path, output_dir: Path, names: list[str]) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    doc = pdfium.PdfDocument(str(pdf_path))
    if len(doc) != len(names):
        raise RuntimeError(f"PDF page count mismatch for {pdf_path}: pdf={len(doc)} snapshot={len(names)}")
    staged: list[Path] = []
    for index, name in enumerate(names):
        dst = output_dir / name
        if not dst.suffix:
            dst = dst.with_suffix(".png")
        page = doc[index]
        bitmap = page.render(scale=1.0, rotation=0)
        pil_image = bitmap.to_pil().convert("RGB")
        pil_image.save(dst)
        staged.append(dst)
    return staged


def _image_entries(zip_path: Path) -> list[str]:
    suffixes = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
    with zipfile.ZipFile(zip_path) as archive:
        names = [
            item.filename
            for item in archive.infolist()
            if not item.is_dir() and Path(item.filename).suffix.lower() in suffixes
        ]
    return sorted(names)


def _stage_from_zip(zip_path: Path, output_dir: Path, names: list[str]) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    entries = _image_entries(zip_path)
    if len(entries) != len(names):
        raise RuntimeError(f"ZIP image count mismatch for {zip_path}: zip={len(entries)} snapshot={len(names)}")
    staged: list[Path] = []
    with zipfile.ZipFile(zip_path) as archive:
        for entry_name, target_name in zip(entries, names):
            dst = output_dir / target_name
            with archive.open(entry_name, "r") as src, dst.open("wb") as handle:
                shutil.copyfileobj(src, handle)
            staged.append(dst)
    return staged


def _stage_sample_japan(repo_root: Path, output_dir: Path, names: list[str]) -> list[Path]:
    source_dir = repo_root / "Sample" / "japan"
    return _stage_from_image_dir(source_dir, output_dir, names)


def _stage_sources(
    *,
    spec: DatasetSpec,
    source_root: Path,
    output_dir: Path,
    pages: list[dict[str, Any]],
) -> list[Path]:
    names = _page_image_names(pages)
    output_dir.mkdir(parents=True, exist_ok=True)
    if spec.source_kind == "image_dir":
        source_dir = source_root / spec.source_name
        if not source_dir.is_dir() and spec.key == "coffee":
            alternatives = sorted(source_root.glob("tmp_Coffee v01 c01_*"))
            for candidate in alternatives:
                if candidate.is_dir() and len(list(candidate.iterdir())) >= len(names):
                    source_dir = candidate
                    break
        return _stage_from_image_dir(source_dir, output_dir, names)
    if spec.source_kind == "pdf":
        return _stage_from_pdf(source_root / spec.source_name, output_dir, names)
    if spec.source_kind == "zip":
        return _stage_from_zip(source_root / spec.source_name, output_dir, names)
    if spec.source_kind == "sample_japan":
        return _stage_sample_japan(ROOT, output_dir, names)
    raise ValueError(f"Unsupported source kind: {spec.source_kind}")


def _xyxy_from_block(block: Any) -> list[float]:
    raw = getattr(block, "xyxy", None)
    if raw is None and isinstance(block, dict):
        raw = block.get("xyxy") or block.get("bbox")
    if raw is None:
        return [0.0, 0.0, 0.0, 0.0]
    return [float(v) for v in list(raw)[:4]]


def _area(box: list[float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def _iou(a: list[float], b: list[float]) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = _area(a) + _area(b) - inter
    return inter / union if union > 0 else 0.0


def _snapshot_blocks_by_page(pages: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    mapping: dict[str, list[dict[str, Any]]] = {}
    for page in pages:
        name = str(page.get("image_name") or Path(str(page.get("image_path") or "")).name)
        blocks = page.get("blocks") or []
        mapping[name] = [block for block in blocks if isinstance(block, dict)]
    return mapping


def _match_translation(block: Any, candidates: list[dict[str, Any]]) -> tuple[dict[str, Any] | None, float]:
    live_box = _xyxy_from_block(block)
    live_class = str(getattr(block, "text_class", "") or "")
    best: dict[str, Any] | None = None
    best_iou = 0.0
    for candidate in candidates:
        if not str(candidate.get("translation") or "").strip():
            continue
        candidate_box = [float(v) for v in list(candidate.get("xyxy") or candidate.get("bbox") or [])[:4]]
        if len(candidate_box) != 4:
            continue
        score = _iou(live_box, candidate_box)
        if live_class and str(candidate.get("text_class") or "") == live_class:
            score += 0.05
        if score > best_iou:
            best_iou = score
            best = candidate
    return best, max(0.0, best_iou - (0.05 if best and live_class and str(best.get("text_class") or "") == live_class else 0.0))


def _install_no_gemma_translation_replay(
    processor: Any,
    *,
    snapshot_blocks: dict[str, list[dict[str, Any]]],
    dataset_key: str,
    match_threshold: float,
    decision_rows: list[dict[str, Any]],
) -> None:
    def no_gemma_prewarm(_self: Any) -> None:
        _self._emit_benchmark_event("gemma_prewarm_skipped", reason="no_gemma_replay")

    def no_gemma_await(_self: Any) -> None:
        _self._emit_benchmark_event("gemma_runtime_skipped", reason="no_gemma_replay")

    def replay_translate(self: Any, pages: list[Any]) -> None:
        total_images = len(pages)
        settings_page = self.main_page.settings_page
        translator_key = "Existing page_snapshots.json"
        translator_engine = "NoGemmaTranslationReplay"
        for index, ctx in enumerate(pages):
            self._raise_if_cancelled()
            if ctx.failed_stage:
                continue
            self._set_current_image(ctx.image_path)
            self.emit_progress(index, total_images, 7, 10, False)
            if ctx.no_text_detected:
                self.main_page.image_ctrl.mark_processing_stage(
                    ctx.image_path,
                    "translation",
                    "skipped",
                    reason="no_text_detected",
                )
                self._emit_benchmark_event(
                    "translate_end",
                    image_path=ctx.image_path,
                    image_index=index,
                    total_images=total_images,
                    block_count=0,
                    translator_key=translator_key,
                    translator_engine=translator_engine,
                    skip_reason="no_text_detected",
                )
                continue
            candidates = snapshot_blocks.get(ctx.image_name, [])
            matched = 0
            missing = 0
            low_iou = 0
            self._emit_benchmark_event(
                "translate_start",
                image_path=ctx.image_path,
                image_index=index,
                total_images=total_images,
                block_count=len(ctx.blk_list or []),
                translator_key=translator_key,
                translator_engine=translator_engine,
                no_gemma_replay=True,
            )
            for block_index, block in enumerate(ctx.blk_list or []):
                candidate, score = _match_translation(block, candidates)
                accepted = bool(candidate is not None and score >= match_threshold)
                if accepted:
                    block.translation = str(candidate.get("translation") or "")
                    matched += 1
                else:
                    block.translation = ""
                    missing += 1
                    if candidate is not None:
                        low_iou += 1
                setattr(block, "_translation_replay_iou", float(score))
                setattr(block, "_translation_replay_matched", bool(accepted))
                setattr(block, "_translation_replay_source_text", str((candidate or {}).get("text") or ""))
                decision_rows.append(
                    {
                        "dataset": dataset_key,
                        "page": ctx.image_name,
                        "block_index": block_index,
                        "text_class": str(getattr(block, "text_class", "") or ""),
                        "ocr_text": str(getattr(block, "text", "") or ""),
                        "translation": str(getattr(block, "translation", "") or ""),
                        "matched": "1" if accepted else "0",
                        "match_iou": f"{score:.4f}",
                        "snapshot_text": str((candidate or {}).get("text") or ""),
                        "snapshot_translation": str((candidate or {}).get("translation") or ""),
                        "reason": "" if accepted else ("low_iou" if candidate is not None else "no_snapshot_translation_match"),
                    }
                )
            apply_translation_result_dictionary(
                ctx.blk_list,
                settings_page.get_translation_result_dictionary_rules(),
            )
            ctx.page_translation_metrics = self._translation_benchmark_metrics(None)
            self._persist_translation_state(
                ctx.image_path,
                ctx.blk_list,
                translator_key,
                translator_engine,
                "replayed",
            )
            self.main_page.image_ctrl.update_processing_summary(
                ctx.image_path,
                {
                    "translation_replay_matched_block_count": matched,
                    "translation_replay_missing_block_count": missing,
                    "translation_replay_low_iou_block_count": low_iou,
                },
            )
            self._emit_benchmark_event(
                "translate_end",
                image_path=ctx.image_path,
                image_index=index,
                total_images=total_images,
                block_count=len(ctx.blk_list or []),
                translator_key=translator_key,
                translator_engine=translator_engine,
                cache_status="replayed",
                no_gemma_replay=True,
                matched_block_count=matched,
                missing_block_count=missing,
                low_iou_block_count=low_iou,
                **ctx.page_translation_metrics,
            )
            raw_text_obj = json.loads(get_raw_text(ctx.blk_list))
            translated_text_obj = json.loads(get_raw_translation(ctx.blk_list))
            if raw_text_obj and not translated_text_obj:
                self._emit_benchmark_event(
                    "translation_replay_empty_page",
                    image_path=ctx.image_path,
                    image_index=index,
                    total_images=total_images,
                    reason="no_replayed_translations_after_current_ocr",
                )

    processor._start_gemma_prewarm = types.MethodType(no_gemma_prewarm, processor)
    processor._await_gemma_runtime = types.MethodType(no_gemma_await, processor)
    processor._translate_all = types.MethodType(replay_translate, processor)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload if isinstance(payload, dict) else {}


def _audit_run(
    *,
    run_dir: Path,
    snapshot_pages: list[dict[str, Any]],
    replay_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    page_snapshots = _read_json(run_dir / "page_snapshots.json")
    pages = page_snapshots.get("pages", [])
    page_count = len(pages) if isinstance(pages, list) else 0
    render_items = 0
    forbidden_symbol_items = 0
    text_free_layout_review = 0
    missing_translation_rows = [row for row in replay_rows if row.get("matched") != "1"]
    for page in pages if isinstance(pages, list) else []:
        for block in page.get("blocks", []) or []:
            reasons = block.get("render_normalization_reasons") or []
            if (
                "forbidden_symbol_removed" in reasons
                or "strict_symbol_removed" in reasons
                or "render_sanitized_symbols" in reasons
            ):
                forbidden_symbol_items += 1
            if block.get("text_fit_status") == "needs_review_text_free_layout":
                text_free_layout_review += 1
        stage_status = page.get("stage_status", {})
        if isinstance(stage_status, dict):
            render = stage_status.get("render", {})
            if isinstance(render, dict):
                render_items += int(render.get("text_item_count", 0) or 0)

    old_rendered = sum(
        1
        for page in snapshot_pages
        for block in (page.get("blocks") or [])
        if str(block.get("translation") or "").strip()
    )
    new_rendered = sum(1 for row in replay_rows if row.get("matched") == "1" and row.get("translation", "").strip())
    return {
        "page_count": page_count,
        "snapshot_translated_block_count": old_rendered,
        "replayed_translation_block_count": new_rendered,
        "missing_translation_match_count": len(missing_translation_rows),
        "render_text_item_count": render_items,
        "render_sanitized_symbol_item_count": forbidden_symbol_items,
        "needs_review_text_free_layout_count": text_free_layout_review,
        "metrics": summarize_metrics(run_dir / "metrics.jsonl"),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _build_preset(base_preset: dict[str, Any], *, use_gpu: bool) -> dict[str, Any]:
    preset = json.loads(json.dumps(base_preset, ensure_ascii=False))
    app = preset.setdefault("app", {})
    app["workflow_mode"] = STAGE_BATCHED_WORKFLOW_MODE
    app["ocr"] = "PaddleOCR VL"
    app["translator"] = "Custom Local Server(Gemma)"
    app["inpainter"] = app.get("inpainter") or "lama_large_512px"
    app["use_gpu"] = bool(use_gpu)
    export = preset.setdefault("export", {})
    export.update(
        {
            "export_raw_text": True,
            "export_translated_text": True,
            "export_inpainted_image": True,
            "export_detector_overlay": True,
            "export_raw_mask": True,
            "export_mask_overlay": True,
            "export_cleanup_mask_delta": True,
            "export_debug_metadata": True,
        }
    )
    ocr_runtime = preset.setdefault("ocr_runtime", {})
    ocr_runtime["kind"] = "paddleocr_vl"
    return preset


def _resolve_snapshot_path(previous_run_root: Path, spec: DatasetSpec) -> Path:
    candidates = [
        previous_run_root / spec.snapshot_dir_name / "page_snapshots.json",
        previous_run_root / spec.key / "page_snapshots.json",
        previous_run_root / "page_snapshots.json",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[0]


def _run_dataset(
    *,
    app: QApplication,
    spec: DatasetSpec,
    source_root: Path,
    previous_run_root: Path,
    output_root: Path,
    base_preset: dict[str, Any],
    use_gpu: bool,
    match_threshold: float,
) -> dict[str, Any]:
    dataset_run_dir = output_root / spec.key
    dataset_run_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = _resolve_snapshot_path(previous_run_root, spec)
    snapshot_pages = _load_snapshot_pages(snapshot_path)
    staged_paths = _stage_sources(
        spec=spec,
        source_root=source_root,
        output_dir=dataset_run_dir / "corpus",
        pages=snapshot_pages,
    )
    preset = _build_preset(base_preset, use_gpu=use_gpu)
    product_contract = resolve_product_benchmark_contract(preset)
    write_json(
        dataset_run_dir / "benchmark_request.json",
        {
            "mode": "no_gemma_replay_stage_batched_product",
            "workflow_mode": STAGE_BATCHED_WORKFLOW_MODE,
            "translation_mode": "existing_page_snapshots",
            "gemma_disabled": True,
            "runtime_lifecycle": "product_stage_batch",
            "managed_runtime_prestart": False,
            "dataset": spec.key,
            "display_name": spec.display_name,
            "source_lang": spec.source_lang,
            "target_lang": spec.target_lang,
            "snapshot_path": str(snapshot_path),
            "selected_paths": [str(path) for path in staged_paths],
            **product_contract,
        },
    )
    write_json(dataset_run_dir / "preset_resolved.json", preset)
    write_json(
        dataset_run_dir / "runtime_lifecycle.json",
        {
            "runtime_lifecycle": "product_stage_batch",
            "managed_runtime_prestart": False,
            "gemma_disabled": True,
            "note": "No-Gemma replay lets ComicTranslatePipeline.batch_process() and StageBatchedProcessor start OCR runtime exactly like the product path.",
        },
    )

    from controller import ComicTranslate

    settings_backup = _settings_snapshot()
    gemma_env_snapshot: dict[str, str | None] = {}
    os.environ["CT_BENCH_OUTPUT_DIR"] = str(dataset_run_dir)
    window = ComicTranslate()
    replay_rows: list[dict[str, Any]] = []
    try:
        _configure_window(window, preset, spec.source_lang, spec.target_lang)
        loaded_paths = _load_images(window, staged_paths, spec.source_lang, spec.target_lang)
        window._current_batch_run_type = "batch"
        processor = window.pipeline.stage_batched_processor
        try:
            window.pipeline.cache_manager.clear_ocr_cache()
        except Exception:
            pass
        _install_no_gemma_translation_replay(
            processor,
            snapshot_blocks=_snapshot_blocks_by_page(snapshot_pages),
            dataset_key=spec.key,
            match_threshold=match_threshold,
            decision_rows=replay_rows,
        )
        started = time.perf_counter()
        window.emit_memlog(
            "benchmark_run_start",
            benchmark_mode="no_gemma_replay_stage_batched",
            total_images=len(loaded_paths),
            translation_mode="existing_page_snapshots",
            **product_contract,
        )
        window.pipeline.batch_process(loaded_paths)
        elapsed = time.perf_counter() - started
        window.emit_memlog(
            "benchmark_run_finished",
            benchmark_mode="no_gemma_replay_stage_batched",
            elapsed_sec=round(elapsed, 3),
            translation_mode="existing_page_snapshots",
            **product_contract,
        )
        page_snapshots_path = _write_page_snapshots(window, dataset_run_dir, loaded_paths)
        _log(f"no-Gemma page snapshots saved: {page_snapshots_path}")
        audit = _audit_run(
            run_dir=dataset_run_dir,
            snapshot_pages=snapshot_pages,
            replay_rows=replay_rows,
        )
        audit.update(
            {
                "dataset": spec.key,
                "display_name": spec.display_name,
                "elapsed_sec": round(elapsed, 3),
                "workflow_mode": STAGE_BATCHED_WORKFLOW_MODE,
                "translation_mode": "existing_page_snapshots",
                "gemma_disabled": True,
                "runtime_lifecycle": "product_stage_batch",
                "managed_runtime_prestart": False,
                "output_dir": str(dataset_run_dir),
                **product_contract,
            }
        )
        _write_csv(dataset_run_dir / "translation_replay_matches.csv", replay_rows)
        write_json(dataset_run_dir / "summary.json", audit)
        (dataset_run_dir / "summary.md").write_text(render_summary_markdown(audit), encoding="utf-8")
        return audit
    finally:
        try:
            window.pipeline.release_model_caches()
        except Exception:
            pass
        try:
            window._skip_close_prompt = True
            window.close()
            app.processEvents()
        finally:
            _restore_settings(settings_backup)
            _restore_env(gemma_env_snapshot)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run product stage-batch OCR/inpaint/render while replaying existing translations and skipping Gemma."
    )
    parser.add_argument("--source-root", default=str(DEFAULT_SOURCE_ROOT))
    parser.add_argument("--previous-run-root", default=str(DEFAULT_PREVIOUS_RUN))
    parser.add_argument("--output-root", default="")
    parser.add_argument("--preset", default=DEFAULT_PRESET)
    parser.add_argument("--only", action="append", default=[], help="Dataset key to run. Can be repeated.")
    parser.add_argument("--match-threshold", type=float, default=0.25)
    parser.add_argument("--cpu", action="store_true", help="Disable GPU in app settings for this replay.")
    args = parser.parse_args()

    source_root = Path(args.source_root)
    previous_run_root = Path(args.previous_run_root)
    if args.output_root:
        output_root = Path(args.output_root)
    else:
        output_root = source_root / "product_stage_batch_rerun_20260629_170647" / f"no_gemma_inpaint_replay_{time.strftime('%Y%m%d_%H%M%S')}"
        if not output_root.parent.exists():
            output_root = benchmark_output_root() / f"no_gemma_inpaint_replay_{time.strftime('%Y%m%d_%H%M%S')}"
    output_root.mkdir(parents=True, exist_ok=True)

    base_preset, preset_path = load_preset(args.preset)
    wanted = {item.strip() for item in args.only if item.strip()}
    specs = [spec for spec in DATASETS if not wanted or spec.key in wanted]
    if wanted and {spec.key for spec in specs} != wanted:
        missing = sorted(wanted - {spec.key for spec in specs})
        raise SystemExit(f"Unknown dataset key(s): {missing}")

    write_json(
        output_root / "replay_request.json",
        {
            "source_root": str(source_root),
            "previous_run_root": str(previous_run_root),
            "preset": str(args.preset),
            "preset_path": str(preset_path),
            "datasets": [spec.key for spec in specs],
            "translation_mode": "existing_page_snapshots",
            "gemma_disabled": True,
            "runtime_lifecycle": "product_stage_batch",
            "managed_runtime_prestart": False,
            "match_threshold": args.match_threshold,
        },
    )

    app = QApplication.instance() or QApplication([])
    summaries: list[dict[str, Any]] = []
    for spec in specs:
        _log(f"no-Gemma replay start: {spec.key} ({spec.display_name})")
        try:
            summaries.append(
                _run_dataset(
                    app=app,
                    spec=spec,
                    source_root=source_root,
                    previous_run_root=previous_run_root,
                    output_root=output_root,
                    base_preset=base_preset,
                    use_gpu=not args.cpu,
                    match_threshold=args.match_threshold,
                )
            )
        except Exception as exc:
            summary = {
                "dataset": spec.key,
                "display_name": spec.display_name,
                "failed": True,
                "reason": str(exc),
                "output_dir": str(output_root / spec.key),
                **product_benchmark_failure_contract(exc),
            }
            summaries.append(summary)
            write_json(output_root / spec.key / "summary.json", summary)
            _log(f"no-Gemma replay failed: {spec.key}: {exc}")

    write_json(
        output_root / "summary.json",
        {
            "mode": "no_gemma_inpaint_replay",
            "output_root": str(output_root),
            "dataset_count": len(summaries),
            "failed_dataset_count": sum(1 for item in summaries if item.get("failed")),
            "summaries": summaries,
        },
    )
    _write_csv(
        output_root / "summary.csv",
        [
            {
                "dataset": item.get("dataset", ""),
                "display_name": item.get("display_name", ""),
                "failed": "1" if item.get("failed") else "0",
                "page_count": item.get("page_count", ""),
                "snapshot_translated_block_count": item.get("snapshot_translated_block_count", ""),
                "replayed_translation_block_count": item.get("replayed_translation_block_count", ""),
                "missing_translation_match_count": item.get("missing_translation_match_count", ""),
                "render_text_item_count": item.get("render_text_item_count", ""),
                "render_sanitized_symbol_item_count": item.get("render_sanitized_symbol_item_count", ""),
                "needs_review_text_free_layout_count": item.get("needs_review_text_free_layout_count", ""),
                "inpainter_family": item.get("inpainter_family", ""),
                "inpainter": item.get("inpainter", ""),
                "mask_refiner": item.get("mask_refiner", ""),
                "runtime_lifecycle": item.get("runtime_lifecycle", ""),
                "managed_runtime_prestart": item.get("managed_runtime_prestart", ""),
                "output_dir": item.get("output_dir", ""),
                "reason": item.get("reason", ""),
                "reason_code": item.get("reason_code", ""),
            }
            for item in summaries
        ],
    )
    print(f"[benchmark] no-Gemma replay output: {output_root}")
    return 1 if any(item.get("failed") for item in summaries) else 0


if __name__ == "__main__":
    raise SystemExit(main())

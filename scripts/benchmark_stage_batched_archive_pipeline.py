#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import os
import re
import shutil
import sys
import time
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

from PySide6.QtWidgets import QApplication

from benchmark_common import (
    create_run_dir,
    load_preset,
    render_summary_markdown,
    repo_relative_str,
    summarize_metrics,
    write_json,
)
from benchmark_pipeline import _log, _write_page_snapshots
from benchmark_stage_batched_pipeline import (
    StageBatchedRunner,
    _emit_gpu_checkpoint,
    _load_image_array,
    _set_current_image,
)
from modules.ocr.selection import (
    OCR_MODE_BEST_LOCAL,
    OCR_MODE_BEST_LOCAL_PLUS,
    OCR_MODE_HUNYUAN,
    OCR_MODE_MANGALMM,
    OCR_MODE_PADDLE_VL,
)
from modules.utils.archives import make as make_archive
from modules.utils.automatic_output import (
    OUTPUT_ARCHIVE_FORMAT_CBZ,
    OUTPUT_ARCHIVE_FORMAT_ZIP,
    OUTPUT_IMAGE_FORMAT_PNG,
    OUTPUT_TARGET_ARCHIVE,
)
from modules.rendering.render import get_best_render_area
from modules.utils.device import resolve_device
from modules.utils.textblock import sort_blk_list


DEFAULT_FAST_PRESET = "workflow-split-runtime-stage-batched-single-ocr"
DEFAULT_OPTIMAL_PLUS_PRESET = "workflow-split-runtime-stage-batched-dual-resident"
SUPPORTED_ARCHIVE_EXTENSIONS = {".cbz", ".zip", ".cbr", ".cbt", ".cb7", ".rar", ".7z", ".tar", ".pdf", ".epub"}
OCR_TRANSIENT_RETRY_ATTEMPTS = 6
OCR_TRANSIENT_RETRY_DELAY_SEC = 8.0
_ILLEGAL_FILE_CHARS_RE = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_WHITESPACE_RE = re.compile(r"\s+")


def _set_combo_data(combo, value: str) -> None:
    index = combo.findData(value)
    if index < 0:
        index = combo.findText(value)
    if index < 0:
        raise RuntimeError(f"Unable to find combo value: {value}")
    combo.setCurrentIndex(index)


def resolve_ocr_mode_value(mode: str) -> str:
    normalized = str(mode or "").strip().lower().replace("_", "-")
    if normalized in {"fastest", "optimal", "best-local"}:
        return OCR_MODE_BEST_LOCAL
    if normalized in {"optimal-plus", "optimal+", "best-local-plus"}:
        return OCR_MODE_BEST_LOCAL_PLUS
    if normalized in {"paddle", "paddleocr", "paddleocr-vl"}:
        return OCR_MODE_PADDLE_VL
    if normalized in {"hunyuan", "hunyuanocr"}:
        return OCR_MODE_HUNYUAN
    if normalized in {"mangalmm", "manga-lmm"}:
        return OCR_MODE_MANGALMM
    raise ValueError(f"Unsupported OCR mode: {mode}")


def default_preset_for_ocr_mode(mode: str) -> str:
    return DEFAULT_OPTIMAL_PLUS_PRESET if resolve_ocr_mode_value(mode) == OCR_MODE_BEST_LOCAL_PLUS else DEFAULT_FAST_PRESET


def is_archive_like_path(path: Path) -> bool:
    return path.suffix.lower() in SUPPORTED_ARCHIVE_EXTENSIONS


def reserve_unique_path(path: Path) -> Path:
    candidate = path
    if not candidate.exists():
        return candidate
    stem = path.stem
    suffix = path.suffix
    parent = path.parent
    for index in range(1, 1000):
        candidate = parent / f"{stem}_{index:03d}{suffix}"
        if not candidate.exists():
            return candidate
    raise FileExistsError(f"No available output path for {path}")


def build_preserved_archive_file_name(source_stem: str, archive_format: str) -> str:
    stem = _ILLEGAL_FILE_CHARS_RE.sub("", str(source_stem or "")).replace("\n", " ").replace("\r", " ")
    stem = _WHITESPACE_RE.sub(" ", stem).strip(" .")
    if len(stem) > 180:
        stem = stem[:180].rstrip(" .")
    return f"{stem or 'stage_batched_archive_pipeline'}_translated.{str(archive_format).lower().lstrip('.')}"


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _rect_or_none(value: Any) -> list[float] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 4:
        return None
    rect: list[float] = []
    for item in value[:4]:
        number = _float_or_none(item)
        if number is None:
            return None
        rect.append(number)
    return rect


def collect_render_fit_summary(
    page_states: list[dict[str, Any]],
    *,
    tiny_font_threshold: float = 12.0,
) -> dict[str, Any]:
    item_count = 0
    tiny_items: list[dict[str, Any]] = []
    font_sizes: list[float] = []

    for page_index, state in enumerate(page_states):
        if not isinstance(state, dict):
            continue
        viewer_state = state.get("viewer_state", {})
        if not isinstance(viewer_state, dict):
            continue
        text_items_state = viewer_state.get("text_items_state", [])
        if not isinstance(text_items_state, list):
            continue
        image_path = str(state.get("image_path") or "")
        image_name = Path(image_path).name if image_path else ""
        for item_index, item in enumerate(text_items_state):
            if not isinstance(item, dict):
                continue
            item_count += 1
            font_size = _float_or_none(item.get("font_size"))
            if font_size is not None:
                font_sizes.append(font_size)
            if font_size is None or font_size > tiny_font_threshold:
                continue
            tiny_items.append(
                {
                    "page_index": page_index,
                    "image_name": image_name,
                    "item_index": item_index,
                    "font_size": font_size,
                    "source_rect": _rect_or_none(item.get("source_rect")),
                    "width": _float_or_none(item.get("width")),
                    "height": _float_or_none(item.get("height")),
                    "translation_raw_length": len(str(item.get("translation_raw") or "")),
                    "render_text_length": len(str(item.get("render_text") or "")),
                }
            )

    return {
        "item_count": item_count,
        "tiny_font_threshold": float(tiny_font_threshold),
        "tiny_item_count": len(tiny_items),
        "min_font_size": min(font_sizes) if font_sizes else None,
        "p10_font_size": sorted(font_sizes)[max(int(len(font_sizes) * 0.10) - 1, 0)] if font_sizes else None,
        "tiny_items": tiny_items,
    }


def is_transient_ocr_runtime_error(exc: Exception) -> bool:
    message = str(exc or "").casefold()
    return any(
        fragment in message
        for fragment in (
            "service returned http 500",
            "service returned http 502",
            "service returned http 503",
            "service returned http 504",
            "unable to reach the local paddleocr vl service",
            "connection aborted",
            "connection reset",
            "read timed out",
            "timed out",
        )
    )


def apply_archive_output_settings(
    window,
    *,
    archive_format: str,
    archive_image_format: str,
    compression_level: int,
) -> None:
    ui = window.settings_page.ui
    _set_combo_data(ui.automatic_output_target_combo, OUTPUT_TARGET_ARCHIVE)
    _set_combo_data(ui.automatic_output_archive_format_combo, archive_format)
    _set_combo_data(ui.automatic_output_archive_image_format_combo, archive_image_format)
    ui.automatic_output_archive_level_spinbox.setValue(int(compression_level))
    window.project_output_preferences = {
        **window.project_output_preferences,
        "output_use_global": True,
    }


def apply_source_records(window, loaded_paths: list[str], input_paths: list[Path]) -> None:
    if len(input_paths) != 1 or not is_archive_like_path(input_paths[0]):
        return
    source_path = str(input_paths[0].resolve())
    window.export_source_by_path = {
        path: {
            "kind": "archive",
            "source_path": source_path,
        }
        for path in loaded_paths
    }


class ArchiveStageBatchedRunner(StageBatchedRunner):
    def __init__(
        self,
        *,
        archive_format: str,
        archive_image_format: str,
        compression_level: int,
        input_paths: list[Path],
        write_next_to_source: bool,
        keep_staging: bool,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.archive_format = archive_format
        self.archive_image_format = archive_image_format
        self.compression_level = int(compression_level)
        self.input_paths = [path.resolve() for path in input_paths]
        self.write_next_to_source = bool(write_next_to_source)
        self.keep_staging = bool(keep_staging)
        self.final_archive_path = ""
        self.final_archive_root = ""
        self.final_archive_page_count = 0
        self.render_fit_summary_path = ""
        self.tiny_font_item_count = 0
        self.min_render_font_size: float | None = None

    def _load_window(self) -> None:
        super()._load_window()
        assert self.window is not None
        apply_source_records(self.window, self.loaded_paths, self.input_paths)
        apply_archive_output_settings(
            self.window,
            archive_format=self.archive_format,
            archive_image_format=self.archive_image_format,
            compression_level=self.compression_level,
        )

    def _no_text_contexts(self) -> list[Any]:
        return [ctx for ctx in self.pages if bool(getattr(ctx, "no_text_detected", False))]

    def _temporarily_skip_no_text(self, callback):
        skipped = self._no_text_contexts()
        previous = [(ctx, ctx.failed_stage, ctx.failed_reason) for ctx in skipped]
        for ctx in skipped:
            ctx.failed_stage = "__benchmark_no_text_skip__"
            ctx.failed_reason = "no_text_detected"
        try:
            return callback()
        finally:
            for ctx, failed_stage, failed_reason in previous:
                ctx.failed_stage = failed_stage
                ctx.failed_reason = failed_reason

    def _mark_no_text_stage_skipped(self, stage: str) -> None:
        assert self.batch is not None and self.window is not None
        total_images = len(self.pages)
        for index, ctx in enumerate(self.pages):
            if not bool(getattr(ctx, "no_text_detected", False)):
                continue
            self.window.image_ctrl.mark_processing_stage(
                ctx.image_path,
                stage,
                "skipped",
                reason="no_text_detected",
            )
            event_name = f"{stage}_end"
            self.batch._emit_benchmark_event(
                event_name,
                image_path=ctx.image_path,
                image_index=index,
                total_images=total_images,
                block_count=0,
                skip_reason="no_text_detected",
            )

    def _run_ocr_engine(self, **kwargs: Any) -> dict[str, Any]:
        assert self.batch is not None
        ctx = kwargs.get("ctx")
        blocks = kwargs.get("blocks")
        original_blocks = self._clone_blocks(blocks) if isinstance(blocks, list) else None
        for attempt in range(1, OCR_TRANSIENT_RETRY_ATTEMPTS + 1):
            try:
                return super()._run_ocr_engine(**kwargs)
            except Exception as exc:
                if not is_transient_ocr_runtime_error(exc) or attempt >= OCR_TRANSIENT_RETRY_ATTEMPTS:
                    raise
                if isinstance(blocks, list) and original_blocks is not None:
                    blocks[:] = self._clone_blocks(original_blocks)
                delay_sec = OCR_TRANSIENT_RETRY_DELAY_SEC * attempt
                self.batch._emit_benchmark_event(
                    "ocr_transient_retry",
                    image_path=getattr(ctx, "image_path", ""),
                    image_name=getattr(ctx, "image_name", ""),
                    total_images=len(self.pages),
                    engine_key=kwargs.get("engine_key", ""),
                    attempt=attempt,
                    next_attempt=attempt + 1,
                    retry_delay_sec=delay_sec,
                    reason=str(exc),
                )
                time.sleep(delay_sec)
        raise RuntimeError("unreachable OCR retry state")

    def _detect_all(self) -> None:
        assert self.batch is not None and self.window is not None
        total_images = len(self.pages)
        settings_page = self.window.settings_page
        for index, ctx in enumerate(self.pages):
            _set_current_image(self.window, ctx.image_path)
            self.batch.emit_progress(index, total_images, 0, 10, True)
            self.batch._start_page_summary(ctx.image_path, ctx.source_lang, ctx.target_lang)
            self.batch._log_page_start(index, total_images, ctx.image_path)
            ctx.page_ocr_metrics = self.batch._ocr_quality_metrics(None)
            ctx.page_translation_metrics = self.batch._translation_benchmark_metrics(None)
            self.batch._emit_benchmark_event(
                "page_start",
                image_path=ctx.image_path,
                image_index=index,
                total_images=total_images,
                source_lang=ctx.source_lang,
                target_lang=ctx.target_lang,
            )
            self.batch._emit_benchmark_event(
                "detect_start",
                image_path=ctx.image_path,
                image_index=index,
                total_images=total_images,
            )
            if self.batch._is_cancelled():
                self.batch._emit_benchmark_event(
                    "batch_run_cancelled",
                    image_path=ctx.image_path,
                    image_index=index,
                    total_images=total_images,
                )
                return
            ctx.image = _load_image_array(self.window, ctx.image_path)
            if self.window.pipeline.block_detection.block_detector_cache is None:
                self.window.pipeline.block_detection.block_detector_cache = __import__(
                    "modules.detection.processor",
                    fromlist=["TextBlockDetector"],
                ).TextBlockDetector(settings_page)
            detector = self.window.pipeline.block_detection.block_detector_cache
            blk_list = detector.detect(ctx.image)
            ctx.precomputed_mask_details = detector.last_mask_details
            ctx.detector_key = detector.detector or settings_page.get_tool_selection("detector") or "RT-DETR-v2"
            ctx.detector_engine = detector.last_engine_name or ""
            ctx.detector_device = detector.last_device or resolve_device(settings_page.is_gpu_enabled(), backend="onnx")
            source_lang_english = self.window.lang_mapping.get(ctx.source_lang, ctx.source_lang)
            rtl = source_lang_english == "Japanese"
            if blk_list:
                get_best_render_area(blk_list, ctx.image)
                ctx.blk_list = sort_blk_list(blk_list, rtl)
                self.batch._persist_detect_state(
                    ctx.image_path,
                    ctx.blk_list,
                    ctx.detector_key,
                    ctx.detector_engine,
                    ctx.image,
                )
                self.batch._emit_benchmark_event(
                    "detect_end",
                    image_path=ctx.image_path,
                    image_index=index,
                    total_images=total_images,
                    block_count=len(ctx.blk_list or []),
                    detector_key=ctx.detector_key,
                    detector_engine=ctx.detector_engine,
                )
                continue

            setattr(ctx, "no_text_detected", True)
            state = self.batch._ensure_page_state(ctx.image_path)
            state["blk_list"] = []
            state.setdefault("viewer_state", {})["rectangles"] = []
            self.window.image_ctrl.mark_processing_stage(
                ctx.image_path,
                "detect",
                "completed",
                reason="no_text_detected",
                block_count=0,
            )
            self.batch._emit_benchmark_event(
                "detect_end",
                image_path=ctx.image_path,
                image_index=index,
                total_images=total_images,
                block_count=0,
                detector_key=ctx.detector_key,
                detector_engine=ctx.detector_engine,
                skip_reason="no_text_detected",
            )
            self.batch._emit_benchmark_event(
                "page_no_text_detected",
                image_path=ctx.image_path,
                image_index=index,
                total_images=total_images,
                skip_reason="no_text_detected",
            )
        _emit_gpu_checkpoint(self.window, "detect_stage_end", self._active_container_names())

    def _ocr_all(self) -> None:
        self._mark_no_text_stage_skipped("ocr")
        return self._temporarily_skip_no_text(super()._ocr_all)

    def _inpaint_all(self) -> None:
        self._mark_no_text_stage_skipped("inpaint")
        return self._temporarily_skip_no_text(super()._inpaint_all)

    def _translate_all(self) -> None:
        self._mark_no_text_stage_skipped("translation")
        return self._temporarily_skip_no_text(super()._translate_all)

    def _archive_output_root(self) -> Path:
        if self.write_next_to_source and len(self.input_paths) == 1:
            return self.input_paths[0].parent
        return self.run_dir / "final_archive"

    def _archive_output_stem(self) -> str:
        if len(self.input_paths) == 1:
            return self.input_paths[0].stem
        return "stage_batched_archive_pipeline"

    def _finalize_archive_output(self) -> None:
        completed_paths = [
            str(ctx.final_output_path)
            for ctx in self.pages
            if not ctx.failed_stage and ctx.final_output_path and Path(str(ctx.final_output_path)).is_file()
        ]
        if not completed_paths:
            return
        staging_dirs = {str(Path(path).parent) for path in completed_paths}
        if len(staging_dirs) != 1:
            raise RuntimeError(f"Archive pages were written to multiple staging directories: {sorted(staging_dirs)}")
        staging_dir = Path(next(iter(staging_dirs)))
        archive_root = self._archive_output_root()
        archive_root.mkdir(parents=True, exist_ok=True)
        requested_path = archive_root / build_preserved_archive_file_name(
            self._archive_output_stem(),
            self.archive_format,
        )
        archive_path = reserve_unique_path(requested_path)
        started = time.perf_counter()
        self.window.emit_memlog(  # type: ignore[union-attr]
            "archive_finalize_start",
            archive_path=str(archive_path),
            archive_format=self.archive_format,
            page_count=len(completed_paths),
        )
        make_archive(
            str(staging_dir),
            output_path=str(archive_path),
            compresslevel=self.compression_level,
        )
        elapsed = time.perf_counter() - started
        self.final_archive_path = str(archive_path)
        self.final_archive_root = str(archive_root)
        self.final_archive_page_count = len(completed_paths)
        self.window.emit_memlog(  # type: ignore[union-attr]
            "archive_finalize_end",
            archive_path=str(archive_path),
            elapsed_sec=round(elapsed, 3),
            page_count=len(completed_paths),
        )
        for ctx in self.pages:
            if not ctx.failed_stage:
                self.window.image_ctrl.update_processing_summary(  # type: ignore[union-attr]
                    ctx.image_path,
                    {
                        "translated_archive_path": str(archive_path),
                        "translated_image_path": str(archive_path),
                        "translated_page_image_path": str(ctx.final_output_path),
                        "export_root": str(archive_root),
                    },
                )
        write_json(
            self.run_dir / "final_archive.json",
            {
                "archive_path": str(archive_path),
                "archive_root": str(archive_root),
                "page_count": len(completed_paths),
                "staging_dir": str(staging_dir),
                "staging_kept": self.keep_staging,
            },
        )
        if not self.keep_staging:
            shutil.rmtree(staging_dir, ignore_errors=True)

    def _write_render_fit_summary(self) -> None:
        assert self.window is not None
        page_states: list[dict[str, Any]] = []
        for ctx in self.pages:
            state = self.window.image_ctrl.ensure_page_state(ctx.image_path)
            page_states.append(
                {
                    "image_path": ctx.image_path,
                    "viewer_state": state.get("viewer_state", {}),
                }
            )
        summary = collect_render_fit_summary(page_states)
        output_path = self.run_dir / "render_fit_summary.json"
        write_json(output_path, summary)
        self.render_fit_summary_path = str(output_path)
        self.tiny_font_item_count = int(summary.get("tiny_item_count", 0) or 0)
        min_font = summary.get("min_font_size")
        self.min_render_font_size = float(min_font) if min_font is not None else None
        self.window.emit_memlog(  # type: ignore[union-attr]
            "render_fit_summary",
            path=str(output_path),
            tiny_font_item_count=self.tiny_font_item_count,
            min_render_font_size=self.min_render_font_size,
        )

    def _render_all(self) -> None:
        super()._render_all()
        self._write_render_fit_summary()
        self._finalize_archive_output()


def patch_preset_for_run(
    preset: dict[str, Any],
    *,
    ocr_mode: str,
    disable_line_protect: bool = False,
    ctd_mask_dilate_size: int | None = None,
) -> dict[str, Any]:
    patched = copy.deepcopy(preset)
    app_cfg = patched.setdefault("app", {})
    app_cfg["ocr"] = resolve_ocr_mode_value(ocr_mode)
    app_cfg["translator"] = "Custom Local Server(Gemma)"
    app_cfg["use_gpu"] = True
    if disable_line_protect:
        mask_cfg = patched.setdefault("mask_refiner_settings", {})
        mask_cfg["keep_existing_lines"] = False
    if ctd_mask_dilate_size is not None:
        mask_cfg = patched.setdefault("mask_refiner_settings", {})
        mask_cfg["ctd_mask_dilate_size"] = int(ctd_mask_dilate_size)
    return patched


def update_summary_with_archive(run_dir: Path, summary: dict[str, Any], runner: ArchiveStageBatchedRunner) -> dict[str, Any]:
    summary.update(
        {
            "final_archive_path": runner.final_archive_path,
            "final_archive_root": runner.final_archive_root,
            "final_archive_page_count": runner.final_archive_page_count,
            "archive_format": runner.archive_format,
            "archive_image_format": runner.archive_image_format,
            "archive_compression_level": runner.compression_level,
            "write_next_to_source": runner.write_next_to_source,
            "keep_staging": runner.keep_staging,
            "render_fit_summary_path": runner.render_fit_summary_path,
            "tiny_font_item_count": runner.tiny_font_item_count,
            "min_render_font_size": runner.min_render_font_size,
        }
    )
    write_json(run_dir / "summary.json", summary)
    (run_dir / "summary.md").write_text(render_summary_markdown(summary), encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a full-book Stage-Batched archive benchmark pipeline.")
    parser.add_argument("--input", required=True, nargs="+", help="Archive or image input path(s)")
    parser.add_argument("--preset", default="", help="Preset name or file path. Defaults to the matching Stage-Batched preset.")
    parser.add_argument(
        "--ocr-mode",
        default="optimal",
        choices=("fastest", "optimal", "optimal-plus", "paddleocr-vl", "hunyuanocr", "mangalmm"),
        help="OCR routing mode. For English, optimal and optimal-plus both resolve to PaddleOCR VL.",
    )
    parser.add_argument("--source-lang", default="English")
    parser.add_argument("--target-lang", default="Korean")
    parser.add_argument("--output-root", default="", help="Root directory for benchmark logs. Defaults to ignored banchmark_result_log.")
    parser.add_argument("--output-dir", default="", help="Exact run directory. Overrides --output-root.")
    parser.add_argument("--archive-format", default=OUTPUT_ARCHIVE_FORMAT_CBZ, choices=(OUTPUT_ARCHIVE_FORMAT_CBZ, OUTPUT_ARCHIVE_FORMAT_ZIP))
    parser.add_argument("--archive-image-format", default=OUTPUT_IMAGE_FORMAT_PNG, choices=("png", "jpg", "webp"))
    parser.add_argument("--compression-level", type=int, default=6)
    parser.add_argument("--write-next-to-source", action="store_true", help="Write final translated archive next to the source archive.")
    parser.add_argument("--discard-staging", action="store_true", help="Delete rendered page staging after final archive creation.")
    parser.add_argument(
        "--disable-line-protect",
        action="store_true",
        help="Disable CTD line-protection for colored text bubbles where original text is being preserved.",
    )
    parser.add_argument(
        "--ctd-mask-dilate-size",
        type=int,
        default=None,
        help="Override CTD mask dilation size for inpaint quality experiments.",
    )
    args = parser.parse_args()

    input_paths = [Path(path).expanduser().resolve() for path in args.input]
    missing = [str(path) for path in input_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Input path(s) not found: {missing}")

    preset_name = args.preset or default_preset_for_ocr_mode(args.ocr_mode)
    preset, preset_path = load_preset(preset_name)
    preset = patch_preset_for_run(
        preset,
        ocr_mode=args.ocr_mode,
        disable_line_protect=bool(args.disable_line_protect),
        ctd_mask_dilate_size=args.ctd_mask_dilate_size,
    )
    run_dir = Path(args.output_dir).resolve() if args.output_dir else create_run_dir(
        f"stage_batched_archive_{resolve_ocr_mode_value(args.ocr_mode)}",
        root=Path(args.output_root).resolve() if args.output_root else None,
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    write_json(
        run_dir / "benchmark_request.json",
        {
            "preset_name": preset.get("name", preset_name),
            "preset_path": str(preset_path),
            "mode": "stage-batched-archive-full-book",
            "ocr_mode_requested": args.ocr_mode,
            "ocr_mode_value": resolve_ocr_mode_value(args.ocr_mode),
            "source_lang": args.source_lang,
            "target_lang": args.target_lang,
            "input_paths": [str(path) for path in input_paths],
            "archive_format": args.archive_format,
            "archive_image_format": args.archive_image_format,
            "compression_level": args.compression_level,
            "write_next_to_source": bool(args.write_next_to_source),
            "disable_line_protect": bool(args.disable_line_protect),
            "ctd_mask_dilate_size": args.ctd_mask_dilate_size,
        },
    )
    write_json(run_dir / "preset_resolved.json", preset)

    app = QApplication.instance() or QApplication([])
    runner = ArchiveStageBatchedRunner(
        app=app,
        preset=preset,
        run_dir=run_dir,
        source_lang=args.source_lang,
        target_lang=args.target_lang,
        image_paths=input_paths,
        resident_ocr_mode=resolve_ocr_mode_value(args.ocr_mode),
        archive_format=args.archive_format,
        archive_image_format=args.archive_image_format,
        compression_level=args.compression_level,
        input_paths=input_paths,
        write_next_to_source=bool(args.write_next_to_source),
        keep_staging=not bool(args.discard_staging),
    )

    try:
        summary = runner.run()
        summary = update_summary_with_archive(run_dir, summary, runner)
        _log(
            "full-book archive 실행 완료: pages={pages} failed={failed} archive={archive}".format(
                pages=summary.get("page_done_count"),
                failed=summary.get("page_failed_count"),
                archive=summary.get("final_archive_path", ""),
            )
        )
        return 0
    except Exception as exc:
        _log(f"full-book archive 실행 실패: {exc}")
        raise


if __name__ == "__main__":
    raise SystemExit(main())

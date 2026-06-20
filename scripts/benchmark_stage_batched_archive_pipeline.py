#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import os
import re
import shutil
import subprocess
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
    GEMMA_CONTAINER_NAMES,
    create_run_dir,
    load_preset,
    remove_containers,
    render_summary_markdown,
    repo_relative_str,
    run_command,
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
RUNTIME_REUSE_MODES = ("cold", "signature", "warm")
SPEED_PROFILES: dict[str, dict[str, Any]] = {
    "baseline-safe": {
        "gemma_ctx_size": 4096,
        "compression_level": 6,
        "runtime_reuse_mode": "signature",
    },
    "ctx3072-fast": {
        "gemma_ctx_size": 3072,
        "compression_level": 6,
        "runtime_reuse_mode": "signature",
    },
    "ctx2560-aggressive": {
        "gemma_ctx_size": 2560,
        "compression_level": 6,
        "runtime_reuse_mode": "signature",
    },
    "ctx2560-fast-archive": {
        "gemma_ctx_size": 2560,
        "compression_level": 0,
        "runtime_reuse_mode": "signature",
    },
    "ctx2048-gpu23-fast": {
        "gemma_ctx_size": 2048,
        "gemma_gpu_layers": 23,
        "compression_level": 0,
        "runtime_reuse_mode": "signature",
    },
    "ctx1792-gpu23-extreme": {
        "gemma_ctx_size": 1792,
        "gemma_gpu_layers": 23,
        "compression_level": 0,
        "runtime_reuse_mode": "signature",
    },
    "ctx1536-gpu23-shadow": {
        "gemma_ctx_size": 1536,
        "gemma_gpu_layers": 23,
        "compression_level": 0,
        "runtime_reuse_mode": "signature",
        "allow_failure": True,
    },
    "ctx1280-gpu23-shadow": {
        "gemma_ctx_size": 1280,
        "gemma_gpu_layers": 23,
        "compression_level": 0,
        "runtime_reuse_mode": "signature",
        "allow_failure": True,
    },
    "ctx1024-gpu23-shadow": {
        "gemma_ctx_size": 1024,
        "gemma_gpu_layers": 23,
        "compression_level": 0,
        "runtime_reuse_mode": "signature",
        "allow_failure": True,
    },
    "ctx768-gpu23-shadow": {
        "gemma_ctx_size": 768,
        "gemma_gpu_layers": 23,
        "compression_level": 0,
        "runtime_reuse_mode": "signature",
        "allow_failure": True,
    },
    "ctx3072-fast-archive": {
        "gemma_ctx_size": 3072,
        "compression_level": 0,
        "runtime_reuse_mode": "signature",
    },
    "warm-reuse": {
        "gemma_ctx_size": 3072,
        "compression_level": 0,
        "runtime_reuse_mode": "warm",
    },
    "ctx3072-gpu24-extreme": {
        "gemma_ctx_size": 3072,
        "gemma_gpu_layers": 24,
        "compression_level": 0,
        "runtime_reuse_mode": "signature",
        "allow_failure": True,
    },
    "ctx2560-gpu24-extreme": {
        "gemma_ctx_size": 2560,
        "gemma_gpu_layers": 24,
        "compression_level": 0,
        "runtime_reuse_mode": "signature",
        "allow_failure": True,
    },
    "ctx2560-gpu25-danger": {
        "gemma_ctx_size": 2560,
        "gemma_gpu_layers": 25,
        "compression_level": 0,
        "runtime_reuse_mode": "cold",
        "allow_failure": True,
    },
    "ctx2048-gpu24-extreme": {
        "gemma_ctx_size": 2048,
        "gemma_gpu_layers": 24,
        "compression_level": 0,
        "runtime_reuse_mode": "signature",
        "allow_failure": True,
    },
    "ctx1792-gpu24-shadow": {
        "gemma_ctx_size": 1792,
        "gemma_gpu_layers": 24,
        "compression_level": 0,
        "runtime_reuse_mode": "signature",
        "allow_failure": True,
    },
    "ctx2048-gpu25-danger": {
        "gemma_ctx_size": 2048,
        "gemma_gpu_layers": 25,
        "compression_level": 0,
        "runtime_reuse_mode": "cold",
        "allow_failure": True,
    },
    "ctx2048-gpu26-danger": {
        "gemma_ctx_size": 2048,
        "gemma_gpu_layers": 26,
        "compression_level": 0,
        "runtime_reuse_mode": "cold",
        "allow_failure": True,
    },
    "ctx2048-threads14": {
        "gemma_ctx_size": 2048,
        "gemma_threads": 14,
        "gemma_gpu_layers": 23,
        "compression_level": 0,
        "runtime_reuse_mode": "signature",
    },
    "ctx2048-batch1024": {
        "gemma_ctx_size": 2048,
        "gemma_gpu_layers": 23,
        "gemma_batch_size": 1024,
        "gemma_ubatch_size": 512,
        "compression_level": 0,
        "runtime_reuse_mode": "signature",
        "allow_failure": True,
    },
    "ctx2048-batch2048": {
        "gemma_ctx_size": 2048,
        "gemma_gpu_layers": 23,
        "gemma_batch_size": 2048,
        "gemma_ubatch_size": 512,
        "compression_level": 0,
        "runtime_reuse_mode": "signature",
        "allow_failure": True,
    },
    "ctx2048-flash-attn": {
        "gemma_ctx_size": 2048,
        "gemma_gpu_layers": 23,
        "gemma_flash_attn": True,
        "compression_level": 0,
        "runtime_reuse_mode": "signature",
        "allow_failure": True,
    },
    "ctx2048-q8-kv": {
        "gemma_ctx_size": 2048,
        "gemma_gpu_layers": 23,
        "gemma_cache_type_k": "q8_0",
        "gemma_cache_type_v": "q8_0",
        "compression_level": 0,
        "runtime_reuse_mode": "signature",
        "allow_failure": True,
    },
    "ctx2048-no-warmup": {
        "gemma_ctx_size": 2048,
        "gemma_gpu_layers": 23,
        "gemma_no_warmup": True,
        "compression_level": 0,
        "runtime_reuse_mode": "signature",
        "allow_failure": True,
    },
    "ctx2048-np2": {
        "gemma_ctx_size": 2048,
        "gemma_gpu_layers": 23,
        "gemma_n_parallel": 2,
        "compression_level": 0,
        "runtime_reuse_mode": "signature",
        "allow_failure": True,
    },
    "ctx3072-threads12": {
        "gemma_ctx_size": 3072,
        "gemma_threads": 12,
        "compression_level": 0,
        "runtime_reuse_mode": "signature",
    },
    "danger-shadow": {
        "gemma_ctx_size": 2048,
        "compression_level": 0,
        "runtime_reuse_mode": "cold",
        "allow_failure": True,
    },
}
SUPPORTED_ARCHIVE_EXTENSIONS = {".cbz", ".zip", ".cbr", ".cbt", ".cb7", ".rar", ".7z", ".tar", ".pdf", ".epub"}
OCR_TRANSIENT_RETRY_ATTEMPTS = 6
OCR_TRANSIENT_RETRY_DELAY_SEC = 8.0
_ILLEGAL_FILE_CHARS_RE = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_WHITESPACE_RE = re.compile(r"\s+")
_GEMMA_COMMAND_OPTION_ALIASES: dict[str, tuple[str, ...]] = {
    "context_size": ("-c", "--ctx-size", "--context-size"),
    "threads": ("-t", "--threads"),
    "n_gpu_layers": ("--n-gpu-layers", "--gpu-layers"),
    "n_parallel": ("-np", "--parallel", "--n-parallel"),
    "predict": ("-n", "--n-predict", "--predict"),
    "batch_size": ("-b", "--batch-size"),
    "ubatch_size": ("-ub", "--ubatch-size"),
    "cache_type_k": ("--cache-type-k",),
    "cache_type_v": ("--cache-type-v",),
}
_GEMMA_COMMAND_FLAG_ALIASES: dict[str, tuple[str, ...]] = {
    "flash_attn": ("--flash-attn",),
    "no_warmup": ("--no-warmup",),
}


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
        return OCR_MODE_BEST_LOCAL
    if normalized in {"paddle", "paddleocr", "paddleocr-vl"}:
        return OCR_MODE_PADDLE_VL
    if normalized in {"hunyuan", "hunyuanocr"}:
        return OCR_MODE_HUNYUAN
    if normalized in {"mangalmm", "manga-lmm"}:
        return OCR_MODE_MANGALMM
    raise ValueError(f"Unsupported OCR mode: {mode}")


def default_preset_for_ocr_mode(mode: str) -> str:
    normalized = str(mode or "").strip().lower().replace("_", "-")
    return DEFAULT_OPTIMAL_PLUS_PRESET if normalized in {"optimal-plus", "optimal+", "best-local-plus"} else DEFAULT_FAST_PRESET


def resolve_speed_profile(profile_name: str | None) -> dict[str, Any]:
    normalized = str(profile_name or "").strip()
    if not normalized:
        return {}
    try:
        return copy.deepcopy(SPEED_PROFILES[normalized])
    except KeyError as exc:
        raise ValueError(f"Unsupported speed profile: {profile_name}") from exc


def _int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def build_gemma_runtime_overrides(
    *,
    context_size: int | None = None,
    threads: int | None = None,
    n_gpu_layers: int | None = None,
    n_parallel: int | None = None,
    predict: int | None = None,
    batch_size: int | None = None,
    ubatch_size: int | None = None,
    cache_type_k: str | None = None,
    cache_type_v: str | None = None,
    flash_attn: bool | None = None,
    no_warmup: bool | None = None,
) -> dict[str, Any]:
    values = {
        "context_size": _int_or_none(context_size),
        "threads": _int_or_none(threads),
        "n_gpu_layers": _int_or_none(n_gpu_layers),
        "n_parallel": _int_or_none(n_parallel),
        "predict": _int_or_none(predict),
        "batch_size": _int_or_none(batch_size),
        "ubatch_size": _int_or_none(ubatch_size),
    }
    out: dict[str, Any] = {key: value for key, value in values.items() if value is not None}
    if cache_type_k:
        out["cache_type_k"] = str(cache_type_k).strip()
    if cache_type_v:
        out["cache_type_v"] = str(cache_type_v).strip()
    if flash_attn is not None:
        out["flash_attn"] = bool(flash_attn)
    if no_warmup is not None:
        out["no_warmup"] = bool(no_warmup)
    return out


def _command_option_value(command: list[str], aliases: tuple[str, ...]) -> str | None:
    for index, token in enumerate(command):
        token_text = str(token)
        for alias in aliases:
            if token_text == alias and index + 1 < len(command):
                return str(command[index + 1])
            prefix = f"{alias}="
            if token_text.startswith(prefix):
                return token_text[len(prefix) :]
    return None


def gemma_command_matches_overrides(command: list[str], overrides: dict[str, Any]) -> bool:
    for key, expected in overrides.items():
        if expected is None:
            continue
        if key in _GEMMA_COMMAND_OPTION_ALIASES:
            actual = _command_option_value(command, _GEMMA_COMMAND_OPTION_ALIASES[key])
            if actual is None:
                return False
            if key in {"cache_type_k", "cache_type_v"}:
                if str(actual).strip() != str(expected).strip():
                    return False
            elif str(actual).strip() != str(int(expected)):
                return False
        elif key in _GEMMA_COMMAND_FLAG_ALIASES:
            aliases = _GEMMA_COMMAND_FLAG_ALIASES[key]
            has_flag = any(str(item) in aliases for item in command)
            if bool(expected) != has_flag:
                return False
    return True


def inspect_container_command(container_name: str) -> list[str] | None:
    try:
        completed = run_command(
            ["docker", "inspect", container_name, "--format", "{{json .Config.Cmd}}"],
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    payload = (completed.stdout or "").strip()
    if not payload:
        return None
    try:
        value = json.loads(payload)
    except json.JSONDecodeError:
        return None
    if not isinstance(value, list):
        return None
    return [str(item) for item in value]


def stop_gemma_for_stage_isolation() -> bool:
    running_containers = [
        name
        for name in GEMMA_CONTAINER_NAMES
        if inspect_container_command(name) is not None
    ]
    if not running_containers:
        return False
    _log(
        "Stage-Batched isolation: stopping pre-existing Gemma containers before OCR/inpaint stages: "
        f"{running_containers}"
    )
    remove_containers(running_containers)
    return True


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


def apply_attach_running_gemma_endpoint(window) -> None:
    widget = getattr(window.settings_page.ui, "credential_widgets", {}).get("Custom Local Server(Gemma)_api_url")
    if widget is not None:
        widget.setText("http://localhost:18080/v1")


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
        runtime_reuse_mode: str = "signature",
        gemma_runtime_overrides: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.archive_format = archive_format
        self.archive_image_format = archive_image_format
        self.compression_level = int(compression_level)
        self.input_paths = [path.resolve() for path in input_paths]
        self.write_next_to_source = bool(write_next_to_source)
        self.keep_staging = bool(keep_staging)
        self.runtime_reuse_mode = str(runtime_reuse_mode or "signature").strip().lower()
        if self.runtime_reuse_mode not in RUNTIME_REUSE_MODES:
            raise ValueError(f"Unsupported runtime reuse mode: {runtime_reuse_mode}")
        self.gemma_runtime_overrides = dict(gemma_runtime_overrides or {})
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
        apply_attach_running_gemma_endpoint(self.window)

    def run(self) -> dict[str, Any]:
        stop_gemma_for_stage_isolation()
        return super().run()

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

    def _prepare_translate_stage(self) -> list[dict[str, Any]]:
        self._enforce_gemma_reuse_policy()
        return super()._prepare_translate_stage()

    def _enforce_gemma_reuse_policy(self) -> None:
        if self.runtime_reuse_mode == "cold":
            _log("Gemma runtime reuse mode=cold: recreating Gemma container before translate stage")
            remove_containers(GEMMA_CONTAINER_NAMES)
            return

        if not self.gemma_runtime_overrides:
            _log(
                f"Gemma runtime reuse mode={self.runtime_reuse_mode}: no explicit runtime overrides; "
                "health-first reuse remains enabled"
            )
            return

        stale_containers: list[str] = []
        for name in GEMMA_CONTAINER_NAMES:
            command = inspect_container_command(name)
            if command is None:
                continue
            if not gemma_command_matches_overrides(command, self.gemma_runtime_overrides):
                stale_containers.append(name)

        if stale_containers:
            _log(
                "Gemma runtime signature mismatch: recreating containers={containers} overrides={overrides}".format(
                    containers=stale_containers,
                    overrides=self.gemma_runtime_overrides,
                )
            )
            remove_containers(stale_containers)
        else:
            _log(
                f"Gemma runtime reuse mode={self.runtime_reuse_mode}: existing command signature is compatible"
            )

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
    gemma_runtime_overrides: dict[str, Any] | None = None,
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
    if gemma_runtime_overrides:
        gemma_cfg = patched.setdefault("gemma", {})
        for key, value in gemma_runtime_overrides.items():
            gemma_cfg[key] = value
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
            "runtime_reuse_mode": runner.runtime_reuse_mode,
            "gemma_runtime_overrides": runner.gemma_runtime_overrides,
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
    parser.add_argument("--compression-level", type=int, default=None)
    parser.add_argument("--gemma-ctx-size", type=int, default=None, help="Override Gemma llama.cpp context size for this benchmark run.")
    parser.add_argument("--gemma-threads", type=int, default=None, help="Override Gemma llama.cpp thread count for this benchmark run.")
    parser.add_argument("--gemma-gpu-layers", type=int, default=None, help="Override Gemma llama.cpp GPU layer count for this benchmark run.")
    parser.add_argument("--gemma-n-parallel", type=int, default=None, help="Override Gemma llama.cpp parallel slot count for this benchmark run.")
    parser.add_argument("--gemma-predict", type=int, default=None, help="Override Gemma llama.cpp server default predict limit for this benchmark run.")
    parser.add_argument("--gemma-batch-size", type=int, default=None, help="Override Gemma llama.cpp batch size for this benchmark run.")
    parser.add_argument("--gemma-ubatch-size", type=int, default=None, help="Override Gemma llama.cpp ubatch size for this benchmark run.")
    parser.add_argument("--gemma-cache-type-k", default="", help="Override Gemma llama.cpp KV cache key type, e.g. f16 or q8_0.")
    parser.add_argument("--gemma-cache-type-v", default="", help="Override Gemma llama.cpp KV cache value type, e.g. f16 or q8_0.")
    parser.add_argument("--gemma-flash-attn", action="store_true", help="Enable llama.cpp flash attention for this benchmark run if supported.")
    parser.add_argument("--gemma-no-warmup", action="store_true", help="Disable llama.cpp warmup for this benchmark run if supported.")
    parser.add_argument("--runtime-reuse-mode", default="", choices=RUNTIME_REUSE_MODES)
    parser.add_argument("--speed-profile", default="", choices=tuple(SPEED_PROFILES.keys()))
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

    speed_profile = resolve_speed_profile(args.speed_profile)
    compression_level = int(
        args.compression_level
        if args.compression_level is not None
        else speed_profile.get("compression_level", 6)
    )
    runtime_reuse_mode = str(args.runtime_reuse_mode or speed_profile.get("runtime_reuse_mode", "signature"))
    gemma_runtime_overrides = build_gemma_runtime_overrides(
        context_size=args.gemma_ctx_size if args.gemma_ctx_size is not None else speed_profile.get("gemma_ctx_size"),
        threads=args.gemma_threads if args.gemma_threads is not None else speed_profile.get("gemma_threads"),
        n_gpu_layers=args.gemma_gpu_layers if args.gemma_gpu_layers is not None else speed_profile.get("gemma_gpu_layers"),
        n_parallel=args.gemma_n_parallel if args.gemma_n_parallel is not None else speed_profile.get("gemma_n_parallel"),
        predict=args.gemma_predict if args.gemma_predict is not None else speed_profile.get("gemma_predict"),
        batch_size=args.gemma_batch_size if args.gemma_batch_size is not None else speed_profile.get("gemma_batch_size"),
        ubatch_size=args.gemma_ubatch_size if args.gemma_ubatch_size is not None else speed_profile.get("gemma_ubatch_size"),
        cache_type_k=args.gemma_cache_type_k or speed_profile.get("gemma_cache_type_k"),
        cache_type_v=args.gemma_cache_type_v or speed_profile.get("gemma_cache_type_v"),
        flash_attn=True if args.gemma_flash_attn else speed_profile.get("gemma_flash_attn"),
        no_warmup=True if args.gemma_no_warmup else speed_profile.get("gemma_no_warmup"),
    )

    preset_name = args.preset or default_preset_for_ocr_mode(args.ocr_mode)
    preset, preset_path = load_preset(preset_name)
    preset = patch_preset_for_run(
        preset,
        ocr_mode=args.ocr_mode,
        disable_line_protect=bool(args.disable_line_protect),
        ctd_mask_dilate_size=args.ctd_mask_dilate_size,
        gemma_runtime_overrides=gemma_runtime_overrides,
    )
    run_dir = Path(args.output_dir).resolve() if args.output_dir else create_run_dir(
        f"stage_batched_archive_{str(args.ocr_mode).replace('+', 'plus')}_{args.speed_profile or 'custom'}",
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
            "compression_level": compression_level,
            "speed_profile": args.speed_profile,
            "runtime_reuse_mode": runtime_reuse_mode,
            "gemma_runtime_overrides": gemma_runtime_overrides,
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
        compression_level=compression_level,
        input_paths=input_paths,
        write_next_to_source=bool(args.write_next_to_source),
        keep_staging=not bool(args.discard_staging),
        runtime_reuse_mode=runtime_reuse_mode,
        gemma_runtime_overrides=gemma_runtime_overrides,
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

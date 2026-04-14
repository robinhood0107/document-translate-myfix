#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import os
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import imkit as imk
import requests

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("CT_DISABLE_UPDATE_CHECK", "1")
os.environ.setdefault("CT_ENABLE_MEMLOG", "1")
os.environ.setdefault("CT_ENABLE_GPU_BENCH", "1")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(ROOT))

from PySide6.QtCore import QCoreApplication
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QApplication

from app.path_materialization import ensure_path_materialized
from app.ui.canvas.text.text_item_properties import TextItemProperties
from app.ui.canvas.text_item import OutlineInfo, OutlineType
from app.ui.messages import Messages
from benchmark_common import (
    GEMMA_CONTAINER_NAMES,
    GEMMA_HEALTH_URLS,
    HUNYUAN_OCR_CONTAINER_NAMES,
    MANGALMM_OCR_CONTAINER_NAMES,
    MANGALMM_OCR_HEALTH_URLS,
    PADDLEOCR_VL_CONTAINER_NAMES,
    PADDLEOCR_VL_HEALTH_URLS,
    collect_managed_llama_cpp_runtimes,
    collect_runtime_snapshot,
    ensure_compose_groups_health_first,
    load_preset,
    remove_containers,
    render_summary_markdown,
    repo_relative_str,
    resolve_docker_compose_command,
    resolve_runtime_container_names,
    resolve_runtime_health_urls,
    resolve_corpus,
    run_command,
    summarize_metrics,
    write_json,
    write_snapshot_json,
    _stage_gemma_runtime,
    _stage_mangalmm_ocr_runtime,
    _stage_ocr_runtime,
)
from benchmark_pipeline import (
    _apply_gemma_env,
    _configure_window,
    _load_images,
    _log,
    _restore_env,
    _restore_settings,
    _settings_snapshot,
    _stage_selected_images,
    _write_page_snapshots,
)
from modules.rendering.render import (
    describe_render_text_markup,
    describe_render_text_sanitization,
    get_best_render_area,
    is_vertical_block,
    pyside_word_wrap,
)
from modules.translation.processor import Translator
from modules.utils.correction_dictionary import (
    apply_ocr_result_dictionary,
    apply_translation_result_dictionary,
)
from modules.utils.device import resolve_device
from modules.utils.gpu_metrics import collect_runtime_snapshot as collect_gpu_runtime_snapshot
from modules.utils.image_utils import generate_mask
from modules.utils.language_utils import get_language_code, is_no_space_lang
from modules.utils.ocr_quality import summarize_ocr_quality
from modules.utils.pipeline_config import get_config, get_inpainter_runtime
from modules.utils.render_style_policy import (
    VERTICAL_ALIGNMENT_TOP,
    build_rect_tuple,
    resolve_render_text_color,
)
from modules.utils.textblock import sort_blk_list
from modules.utils.translator_utils import (
    format_translations,
    get_raw_text,
    get_raw_translation,
)


@dataclass
class PageContext:
    image_path: str
    image_name: str
    source_lang: str
    target_lang: str
    image: Any | None = None
    blk_list: list[Any] = field(default_factory=list)
    precomputed_mask_details: dict[str, Any] | None = None
    detector_key: str = ""
    detector_engine: str = ""
    detector_device: str = ""
    failed_stage: str = ""
    failed_reason: str = ""
    page_ocr_metrics: dict[str, int] = field(default_factory=dict)
    page_translation_metrics: dict[str, int] = field(default_factory=dict)
    raw_mask: Any | None = None
    mask: Any | None = None
    mask_details: dict[str, Any] = field(default_factory=dict)
    patches: list[dict[str, Any]] = field(default_factory=list)
    inpaint_input_img: Any | None = None
    cleanup_stats: dict[str, Any] = field(default_factory=lambda: {"applied": False, "component_count": 0, "block_count": 0})
    final_output_path: str = ""


def _set_current_image(window, image_path: str) -> None:
    try:
        window.curr_img_idx = window.image_files.index(image_path)
    except ValueError:
        pass


def _load_image_array(window, image_path: str):
    image = window.image_ctrl.load_image(image_path)
    if image is None:
        ensure_path_materialized(image_path)
        image = imk.read_image(image_path)
    return image


def _runtime_report_template(runtime_services: str) -> dict[str, Any]:
    return {
        "mode": "stage_batched_health_first",
        "runtime_services": runtime_services,
        "container_names": [],
        "health_urls": [],
        "singleton_ocr_runtime": {
            "selected_ocr_runtime_kind": "stage_batched_custom",
            "removed_container_names": [],
        },
        "groups": [],
    }


def _dedupe_strs(items: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        text = str(item or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        ordered.append(text)
    return ordered


def _runtime_progress(window, **payload) -> None:
    window.emit_memlog("runtime_progress", **payload)


def _emit_gpu_checkpoint(window, point: str, container_names: list[str]) -> None:
    snapshot = collect_gpu_runtime_snapshot(container_names)
    window.emit_memlog(
        "stage_resource_snapshot",
        checkpoint=point,
        gpu=snapshot.get("gpu", {}),
        container_names=container_names,
    )


def _emit_group_runtime_events(
    window,
    *,
    phase: str,
    service: str,
    label: str,
    group_reports: list[dict[str, Any]],
    total_images: int,
    image_name: str,
) -> None:
    for report in group_reports:
        action = str(report.get("action", "") or "")
        compose_sec = float(report.get("compose_up_elapsed_sec", 0.0) or 0.0)
        health_sec = float(report.get("health_wait_elapsed_sec", 0.0) or 0.0)
        if action in {"reused"}:
            _runtime_progress(
                window,
                phase=phase,
                service=service,
                status="completed",
                step_key="health_probe",
                message=f"기존 {label} 런타임을 재사용합니다.",
                total_images=total_images,
                page_total=total_images,
                image_name=image_name,
                elapsed_sec=0.0,
            )
            continue
        if action in {"waited"}:
            _runtime_progress(
                window,
                phase=phase,
                service=service,
                status="waiting_health",
                step_key="health_wait",
                message=f"{label} health 기다리는 중...",
                detail=f"health_urls={report.get('health_urls', [])}",
                total_images=total_images,
                page_total=total_images,
                image_name=image_name,
                elapsed_sec=0.0,
            )
            _runtime_progress(
                window,
                phase=phase,
                service=service,
                status="completed",
                step_key="health_wait",
                message=f"{label} health 확인이 완료되었습니다.",
                total_images=total_images,
                page_total=total_images,
                image_name=image_name,
                elapsed_sec=health_sec,
            )
            continue

        if compose_sec > 0:
            _runtime_progress(
                window,
                phase=phase,
                service=service,
                status="starting",
                step_key="compose_up",
                message=f"{label} 컨테이너를 시작하는 중...",
                detail="docker compose up -d",
                total_images=total_images,
                page_total=total_images,
                image_name=image_name,
                elapsed_sec=0.0,
            )
            _runtime_progress(
                window,
                phase=phase,
                service=service,
                status="completed",
                step_key="compose_up",
                message=f"{label} 컨테이너 시작 명령을 보냈습니다.",
                total_images=total_images,
                page_total=total_images,
                image_name=image_name,
                elapsed_sec=compose_sec,
            )

        if health_sec > 0:
            _runtime_progress(
                window,
                phase=phase,
                service=service,
                status="waiting_health",
                step_key="health_wait",
                message=f"{label} health 기다리는 중...",
                detail=f"health_urls={report.get('health_urls', [])}",
                total_images=total_images,
                page_total=total_images,
                image_name=image_name,
                elapsed_sec=0.0,
            )
            _runtime_progress(
                window,
                phase=phase,
                service=service,
                status="completed",
                step_key="health_wait",
                message=f"{label} health 확인이 완료되었습니다.",
                total_images=total_images,
                page_total=total_images,
                image_name=image_name,
                elapsed_sec=health_sec,
            )


def _compose_down(compose_path: str | Path, cwd: str | Path | None) -> None:
    run_command(
        [*resolve_docker_compose_command(), "-f", str(compose_path), "down"],
        cwd=cwd,
        check=False,
    )


def _shutdown_runtime_groups(groups: list[dict[str, Any]]) -> None:
    for group in groups:
        compose_path = group.get("compose_path")
        cwd = group.get("cwd")
        if compose_path:
            _compose_down(compose_path, cwd)


def _mangalmm_runtime_preset(preset: dict[str, Any]) -> dict[str, Any]:
    payload = copy.deepcopy(preset)
    app_cfg = payload.setdefault("app", {})
    if isinstance(app_cfg, dict):
        app_cfg["ocr"] = "MangaLMM"
    ocr_runtime = payload.setdefault("ocr_runtime", {})
    if isinstance(ocr_runtime, dict):
        ocr_runtime["kind"] = "mangalmm"
    return payload


def _prepare_single_ocr_stage_runtime(
    preset: dict[str, Any],
    run_dir: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    runtime_root = run_dir / "runtime" / "ocr_stage"
    remove_containers(GEMMA_CONTAINER_NAMES + HUNYUAN_OCR_CONTAINER_NAMES + MANGALMM_OCR_CONTAINER_NAMES)
    staged = _stage_ocr_runtime(preset, runtime_root / "paddleocr_vl")
    groups = [
        {
            "name": "ocr_paddleocr_vl",
            "container_names": list(PADDLEOCR_VL_CONTAINER_NAMES),
            "health_urls": list(PADDLEOCR_VL_HEALTH_URLS),
            "compose_path": staged["compose_path"],
            "cwd": runtime_root / "paddleocr_vl",
        }
    ]
    reports = ensure_compose_groups_health_first(groups, log_fn=_log)
    policy = _runtime_report_template("ocr-stage-single")
    policy["container_names"] = list(PADDLEOCR_VL_CONTAINER_NAMES)
    policy["health_urls"] = list(PADDLEOCR_VL_HEALTH_URLS)
    policy["groups"] = reports
    policy["singleton_ocr_runtime"]["removed_container_names"] = list(
        GEMMA_CONTAINER_NAMES + HUNYUAN_OCR_CONTAINER_NAMES + MANGALMM_OCR_CONTAINER_NAMES
    )
    runtime_meta = {
        "ocr": {},
        "gemma": {},
    }
    return groups, policy, runtime_meta


def _prepare_dual_ocr_stage_runtime(
    preset: dict[str, Any],
    run_dir: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    runtime_root = run_dir / "runtime" / "ocr_stage"
    remove_containers(GEMMA_CONTAINER_NAMES + HUNYUAN_OCR_CONTAINER_NAMES)
    paddle_staged = _stage_ocr_runtime(preset, runtime_root / "paddleocr_vl")
    mangal_preset = _mangalmm_runtime_preset(preset)
    mangal_staged = _stage_mangalmm_ocr_runtime(mangal_preset, runtime_root / "mangalmm")
    groups = [
        {
            "name": "ocr_paddleocr_vl",
            "container_names": list(PADDLEOCR_VL_CONTAINER_NAMES),
            "health_urls": list(PADDLEOCR_VL_HEALTH_URLS),
            "compose_path": paddle_staged["compose_path"],
            "cwd": runtime_root / "paddleocr_vl",
        },
        {
            "name": "ocr_mangalmm",
            "container_names": list(MANGALMM_OCR_CONTAINER_NAMES),
            "health_urls": list(MANGALMM_OCR_HEALTH_URLS),
            "compose_path": mangal_staged["compose_path"],
            "cwd": runtime_root / "mangalmm",
        },
    ]
    reports = ensure_compose_groups_health_first(groups, log_fn=_log)
    policy = _runtime_report_template("ocr-stage-dual-resident")
    policy["container_names"] = _dedupe_strs(list(PADDLEOCR_VL_CONTAINER_NAMES) + list(MANGALMM_OCR_CONTAINER_NAMES))
    policy["health_urls"] = _dedupe_strs(list(PADDLEOCR_VL_HEALTH_URLS) + list(MANGALMM_OCR_HEALTH_URLS))
    policy["groups"] = reports
    policy["singleton_ocr_runtime"]["removed_container_names"] = list(GEMMA_CONTAINER_NAMES + HUNYUAN_OCR_CONTAINER_NAMES)
    runtime_meta = {
        "ocr": {},
        "gemma": {},
    }
    return groups, policy, runtime_meta


def _prepare_gemma_stage_runtime(
    preset: dict[str, Any],
    run_dir: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    runtime_root = run_dir / "runtime" / "translate_stage"
    remove_containers(PADDLEOCR_VL_CONTAINER_NAMES + HUNYUAN_OCR_CONTAINER_NAMES + MANGALMM_OCR_CONTAINER_NAMES)
    staged = _stage_gemma_runtime(preset, runtime_root / "gemma")
    groups = [
        {
            "name": "gemma",
            "container_names": list(GEMMA_CONTAINER_NAMES),
            "health_urls": list(GEMMA_HEALTH_URLS),
            "compose_path": staged["compose_path"],
            "cwd": runtime_root / "gemma",
        }
    ]
    reports = ensure_compose_groups_health_first(groups, log_fn=_log)
    policy = _runtime_report_template("translate-stage-gemma")
    policy["container_names"] = list(GEMMA_CONTAINER_NAMES)
    policy["health_urls"] = list(GEMMA_HEALTH_URLS)
    policy["groups"] = reports
    policy["singleton_ocr_runtime"]["removed_container_names"] = list(
        PADDLEOCR_VL_CONTAINER_NAMES + HUNYUAN_OCR_CONTAINER_NAMES + MANGALMM_OCR_CONTAINER_NAMES
    )
    runtime_meta = {
        "ocr": {},
        "gemma": collect_managed_llama_cpp_runtimes(preset, runtime_services="full"),
    }
    return groups, policy, runtime_meta


def _merge_runtime_policy(
    base: dict[str, Any],
    extra: dict[str, Any],
) -> dict[str, Any]:
    base["container_names"] = _dedupe_strs(list(base.get("container_names", [])) + list(extra.get("container_names", [])))
    base["health_urls"] = _dedupe_strs(list(base.get("health_urls", [])) + list(extra.get("health_urls", [])))
    base["groups"] = list(base.get("groups", [])) + list(extra.get("groups", []))
    singleton = base.setdefault("singleton_ocr_runtime", {"removed_container_names": []})
    singleton["removed_container_names"] = _dedupe_strs(
        list(singleton.get("removed_container_names", []))
        + list(extra.get("singleton_ocr_runtime", {}).get("removed_container_names", []))
    )
    return base


def _page_failed(batch, ctx: PageContext, *, index: int, total_images: int, stage: str, reason: str, extra: dict[str, Any] | None = None) -> None:
    ctx.failed_stage = stage
    ctx.failed_reason = reason
    batch.main_page.image_ctrl.update_processing_summary(
        ctx.image_path,
        {
            "last_failure_reason": reason,
        },
    )
    batch.main_page.image_ctrl.mark_processing_stage(
        ctx.image_path,
        stage,
        "failed",
        reason=reason,
    )
    payload = {
        "image_path": ctx.image_path,
        "image_index": index,
        "total_images": total_images,
        "failed_stage": stage,
        "reason": reason,
    }
    if extra:
        payload.update(extra)
    batch._emit_benchmark_event("page_failed", **payload)


class StageBatchedRunner:
    def __init__(
        self,
        *,
        app: QApplication,
        preset: dict[str, Any],
        run_dir: Path,
        source_lang: str,
        target_lang: str,
        image_paths: list[Path],
        resident_ocr_mode: str,
    ) -> None:
        self.app = app
        self.preset = preset
        self.run_dir = run_dir
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.image_paths = image_paths
        self.resident_ocr_mode = resident_ocr_mode
        self.window = None
        self.batch = None
        self.loaded_paths: list[str] = []
        self.pages: list[PageContext] = []
        self.runtime_policy = _runtime_report_template(
            f"stage-batched-{resident_ocr_mode}"
        )
        self.llama_cpp_runtime: dict[str, Any] = {}

    def _active_container_names(self) -> list[str]:
        return _dedupe_strs(list(self.runtime_policy.get("container_names", [])))

    def _load_window(self) -> None:
        from controller import ComicTranslate

        settings_backup = _settings_snapshot()
        gemma_env_snapshot = _apply_gemma_env(self.preset.get("gemma", {}))
        os.environ["CT_BENCH_OUTPUT_DIR"] = str(self.run_dir)
        self.window = ComicTranslate()
        try:
            _configure_window(self.window, self.preset, self.source_lang, self.target_lang)
            self.loaded_paths = _load_images(self.window, self.image_paths, self.source_lang, self.target_lang)
            self.batch = self.window.pipeline.batch_processor
            self.window._current_batch_run_type = "batch"
            for image_path in self.loaded_paths:
                state = self.window.image_ctrl.ensure_page_state(image_path)
                self.pages.append(
                    PageContext(
                        image_path=image_path,
                        image_name=Path(image_path).name,
                        source_lang=str(state.get("source_lang", self.source_lang)),
                        target_lang=str(state.get("target_lang", self.target_lang)),
                    )
                )
            self._settings_backup = settings_backup
            self._gemma_env_snapshot = gemma_env_snapshot
        except Exception:
            _restore_settings(settings_backup)
            _restore_env(gemma_env_snapshot)
            raise

    def _close_window(self) -> None:
        if self.window is None:
            return
        try:
            self.window.pipeline.release_model_caches()
        except Exception:
            pass
        try:
            self.window._skip_close_prompt = True
            self.window.close()
            self.app.processEvents()
        finally:
            _restore_settings(self._settings_backup)
            _restore_env(self._gemma_env_snapshot)

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
                self.batch._emit_benchmark_event("batch_run_cancelled", image_path=ctx.image_path, image_index=index, total_images=total_images)
                return
            ctx.image = _load_image_array(self.window, ctx.image_path)
            if self.window.pipeline.block_detection.block_detector_cache is None:
                self.window.pipeline.block_detection.block_detector_cache = self.window.pipeline.block_detection.block_detector_cache or __import__(
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
            else:
                state = self.batch._ensure_page_state(ctx.image_path)
                state["blk_list"] = []
                state.setdefault("viewer_state", {})["rectangles"] = []
                self.batch._emit_benchmark_event(
                    "detect_end",
                    image_path=ctx.image_path,
                    image_index=index,
                    total_images=total_images,
                    block_count=0,
                    detector_key=ctx.detector_key,
                    detector_engine=ctx.detector_engine,
                )
                self._write_detect_failure_debug(ctx)
                _page_failed(
                    self.batch,
                    ctx,
                    index=index,
                    total_images=total_images,
                    stage="detect",
                    reason="No text blocks detected.",
                    extra=ctx.page_ocr_metrics,
                )
        _emit_gpu_checkpoint(self.window, "detect_stage_end", self._active_container_names())

    def _write_detect_failure_debug(self, ctx: PageContext) -> None:
        assert self.batch is not None and self.window is not None
        settings_page = self.window.settings_page
        export_settings = self.batch._effective_export_settings(settings_page)
        directory, archive_bname = self._resolve_output_paths(ctx.image_path)
        export_root = self.window.image_ctrl.ensure_page_state(ctx.image_path).get("processing_summary", {}).get("export_root", "")
        if not export_root:
            export_root = str(self.run_dir)
        self.batch._write_inpaint_debug_exports(
            export_root=export_root,
            archive_bname=archive_bname,
            image_path=ctx.image_path,
            image=ctx.image,
            blk_list=[],
            export_settings=export_settings,
            raw_mask=None,
            final_mask=None,
            detector_key=ctx.detector_key,
            detector_engine=ctx.detector_engine,
            detector_device=ctx.detector_device,
            inpainter_key=settings_page.get_tool_selection("inpainter"),
            hd_strategy=settings_page.get_hd_strategy_settings().get("strategy", "Resize"),
            cleanup_stats={"applied": False, "component_count": 0, "block_count": 0},
            mask_details={
                "mask_refiner": settings_page.get_mask_refiner_settings().get("mask_refiner", "legacy_bbox"),
                "mask_inpaint_mode": settings_page.get_mask_refiner_settings().get("mask_inpaint_mode", ""),
            },
            inpainter_backend=get_inpainter_runtime(settings_page)["backend"],
        )

    def _resolve_output_paths(self, image_path: str) -> tuple[str, str]:
        directory = str(self.run_dir)
        archive_bname = Path(image_path).stem
        return directory, archive_bname

    def _prepare_ocr_stage(self) -> list[dict[str, Any]]:
        assert self.window is not None
        image_name = self.pages[0].image_name if self.pages else ""
        if self.resident_ocr_mode == "dual":
            groups, policy, runtime_meta = _prepare_dual_ocr_stage_runtime(self.preset, self.run_dir)
            _emit_group_runtime_events(
                self.window,
                phase="ocr_startup",
                service="paddleocr_vl",
                label="PaddleOCR VL",
                group_reports=[report for report in policy["groups"] if report.get("name") == "ocr_paddleocr_vl"],
                total_images=len(self.pages),
                image_name=image_name,
            )
            _emit_group_runtime_events(
                self.window,
                phase="ocr_startup",
                service="mangalmm",
                label="MangaLMM",
                group_reports=[report for report in policy["groups"] if report.get("name") == "ocr_mangalmm"],
                total_images=len(self.pages),
                image_name=image_name,
            )
        else:
            groups, policy, runtime_meta = _prepare_single_ocr_stage_runtime(self.preset, self.run_dir)
            _emit_group_runtime_events(
                self.window,
                phase="ocr_startup",
                service="paddleocr_vl",
                label="PaddleOCR VL",
                group_reports=policy["groups"],
                total_images=len(self.pages),
                image_name=image_name,
            )
        self.runtime_policy = _merge_runtime_policy(self.runtime_policy, policy)
        self.llama_cpp_runtime.update(runtime_meta.get("ocr", {}))
        _emit_gpu_checkpoint(self.window, "ocr_stage_runtime_ready", self._active_container_names())
        return groups

    def _ocr_all(self) -> None:
        assert self.batch is not None and self.window is not None
        total_images = len(self.pages)
        settings_page = self.window.settings_page
        ocr_model = settings_page.get_tool_selection("ocr")
        device = resolve_device(settings_page.is_gpu_enabled())
        for index, ctx in enumerate(self.pages):
            if ctx.failed_stage:
                continue
            _set_current_image(self.window, ctx.image_path)
            self.batch.emit_progress(index, total_images, 2, 10, False)
            self.batch._emit_benchmark_event(
                "ocr_start",
                image_path=ctx.image_path,
                image_index=index,
                total_images=total_images,
                block_count=len(ctx.blk_list or []),
            )
            cache_key = self.batch.cache_manager._get_ocr_cache_key(ctx.image, ctx.source_lang, ocr_model, device)
            cache_status = "miss"
            attempt_count = 0
            ocr_page_profile: dict[str, Any] = {}
            page_metrics = self.batch._ocr_quality_metrics(None)
            try:
                self.window.pipeline.ocr_handler.ocr.initialize(self.window, ctx.source_lang)
                if self.batch.cache_manager._can_serve_all_blocks_from_ocr_cache(cache_key, ctx.blk_list):
                    cache_status = "hit"
                    self.batch.cache_manager._apply_cached_ocr_to_blocks(cache_key, ctx.blk_list)
                    apply_ocr_result_dictionary(
                        ctx.blk_list,
                        settings_page.get_ocr_result_dictionary_rules(),
                    )
                    attempt_count = 1
                else:
                    self.window.pipeline.ocr_handler.ocr.process(ctx.image, ctx.blk_list)
                    ocr_page_profile = dict(getattr(self.window.pipeline.ocr_handler.ocr, "last_page_profile", {}) or {})
                    apply_ocr_result_dictionary(
                        ctx.blk_list,
                        settings_page.get_ocr_result_dictionary_rules(),
                    )
                    self.batch.cache_manager._cache_ocr_results(cache_key, ctx.blk_list)
                    cache_status = "refreshed"
                    attempt_count = 1

                quality = summarize_ocr_quality(ctx.blk_list)
                self.batch._log_ocr_quality(ctx.image_path, quality, attempt_count)
                if quality.get("low_quality", False):
                    attempt_count += 1
                    for blk in ctx.blk_list:
                        blk.text = ""
                    self.window.pipeline.ocr_handler.ocr.process(ctx.image, ctx.blk_list)
                    ocr_page_profile = dict(getattr(self.window.pipeline.ocr_handler.ocr, "last_page_profile", {}) or {})
                    apply_ocr_result_dictionary(
                        ctx.blk_list,
                        settings_page.get_ocr_result_dictionary_rules(),
                    )
                    self.batch.cache_manager._cache_ocr_results(cache_key, ctx.blk_list)
                    quality = summarize_ocr_quality(ctx.blk_list)
                    self.batch._log_ocr_quality(ctx.image_path, quality, attempt_count)
                    cache_status = "refreshed"

                page_metrics = self.batch._ocr_quality_metrics(quality)
                ctx.page_ocr_metrics = page_metrics
                if quality.get("low_quality", False):
                    err_msg = quality.get("reason") or self.window.tr("OCR quality too low after retry.")
                    self.window.image_ctrl.update_processing_summary(
                        ctx.image_path,
                        {
                            "ocr_quality_counts": {
                                "non_empty": quality.get("non_empty", 0),
                                "empty": quality.get("empty", 0),
                                "single_char_like": quality.get("single_char_like", 0),
                            },
                            "last_failure_reason": err_msg,
                        },
                    )
                    raise RuntimeError(err_msg)

                self.batch._persist_ocr_state(
                    ctx.image_path,
                    ctx.blk_list,
                    ocr_model,
                    self.window.pipeline.ocr_handler.ocr.last_engine_name or "",
                    device,
                    quality,
                    cache_status,
                    attempt_count,
                )
                self.batch._emit_benchmark_event(
                    "ocr_end",
                    image_path=ctx.image_path,
                    image_index=index,
                    total_images=total_images,
                    block_count=len(ctx.blk_list or []),
                    ocr_model=ocr_model,
                    ocr_engine=self.window.pipeline.ocr_handler.ocr.last_engine_name or "",
                    cache_status=cache_status,
                    attempt_count=attempt_count,
                    ocr_page_profile=ocr_page_profile,
                    **page_metrics,
                )
            except Exception as exc:
                err_msg = self._runtime_error_message(exc, context="ocr")
                _page_failed(
                    self.batch,
                    ctx,
                    index=index,
                    total_images=total_images,
                    stage="ocr",
                    reason=err_msg,
                    extra={**page_metrics},
                )
        _emit_gpu_checkpoint(self.window, "ocr_stage_end", self._active_container_names())

    def _prepare_translate_stage(self) -> list[dict[str, Any]]:
        assert self.window is not None
        image_name = self.pages[0].image_name if self.pages else ""
        groups, policy, runtime_meta = _prepare_gemma_stage_runtime(self.preset, self.run_dir)
        _emit_group_runtime_events(
            self.window,
            phase="gemma_startup",
            service="gemma",
            label="Gemma",
            group_reports=policy["groups"],
            total_images=len(self.pages),
            image_name=image_name,
        )
        self.runtime_policy = _merge_runtime_policy(self.runtime_policy, policy)
        self.llama_cpp_runtime.update(runtime_meta.get("gemma", {}))
        _emit_gpu_checkpoint(self.window, "translate_stage_runtime_ready", self._active_container_names())
        return groups

    def _translate_all(self) -> None:
        assert self.batch is not None and self.window is not None
        total_images = len(self.pages)
        settings_page = self.window.settings_page
        extra_context = settings_page.get_llm_settings()["extra_context"]
        translator_key = settings_page.get_tool_selection("translator")
        for index, ctx in enumerate(self.pages):
            if ctx.failed_stage:
                continue
            _set_current_image(self.window, ctx.image_path)
            self.batch.emit_progress(index, total_images, 7, 10, False)
            self.batch._emit_benchmark_event(
                "translate_start",
                image_path=ctx.image_path,
                image_index=index,
                total_images=total_images,
                block_count=len(ctx.blk_list or []),
                translator_key=translator_key,
            )
            translation_cache_key = self.batch.cache_manager._get_translation_cache_key(
                ctx.image, ctx.source_lang, ctx.target_lang, translator_key, extra_context
            )
            page_metrics = self.batch._translation_benchmark_metrics(None)
            try:
                translator = Translator(self.window, ctx.source_lang, ctx.target_lang)
                translation_cache_status = "miss"
                if self.batch.cache_manager._can_serve_all_blocks_from_translation_cache(translation_cache_key, ctx.blk_list):
                    self.batch.cache_manager._apply_cached_translations_to_blocks(translation_cache_key, ctx.blk_list)
                    apply_translation_result_dictionary(
                        ctx.blk_list,
                        settings_page.get_translation_result_dictionary_rules(),
                    )
                    translation_cache_status = "hit"
                else:
                    translator.translate(ctx.blk_list, ctx.image, extra_context)
                    apply_translation_result_dictionary(
                        ctx.blk_list,
                        settings_page.get_translation_result_dictionary_rules(),
                    )
                    self.batch.cache_manager._cache_translation_results(translation_cache_key, ctx.blk_list)
                    translation_cache_status = "refreshed"

                page_metrics = self.batch._translation_benchmark_metrics(translator)
                ctx.page_translation_metrics = page_metrics
                self.batch._persist_translation_state(
                    ctx.image_path,
                    ctx.blk_list,
                    translator_key,
                    translator.engine.__class__.__name__,
                    translation_cache_status,
                )
                self.batch._emit_benchmark_event(
                    "translate_end",
                    image_path=ctx.image_path,
                    image_index=index,
                    total_images=total_images,
                    block_count=len(ctx.blk_list or []),
                    translator_key=translator_key,
                    translator_engine=translator.engine.__class__.__name__,
                    cache_status=translation_cache_status,
                    **page_metrics,
                )
                self._validate_translation_json(ctx, index, total_images)
            except Exception as exc:
                err_msg = self._runtime_error_message(exc, context="translation")
                if not ctx.failed_stage:
                    _page_failed(
                        self.batch,
                        ctx,
                        index=index,
                        total_images=total_images,
                        stage="translation",
                        reason=err_msg,
                        extra={**page_metrics},
                    )
        _emit_gpu_checkpoint(self.window, "translate_stage_end", self._active_container_names())

    def _validate_translation_json(self, ctx: PageContext, index: int, total_images: int) -> None:
        assert self.batch is not None
        try:
            raw_text_obj = json.loads(get_raw_text(ctx.blk_list))
            translated_text_obj = json.loads(get_raw_translation(ctx.blk_list))
            if (not raw_text_obj) or (not translated_text_obj):
                raise RuntimeError("Translator returned empty JSON.")
        except json.JSONDecodeError as exc:
            raise RuntimeError(str(exc)) from exc
        except RuntimeError as exc:
            raise RuntimeError(str(exc)) from exc

    def _inpaint_all(self) -> None:
        assert self.batch is not None and self.window is not None
        total_images = len(self.pages)
        settings_page = self.window.settings_page
        export_settings = self.batch._effective_export_settings(settings_page)
        hd_strategy_settings = settings_page.get_hd_strategy_settings()
        hd_strategy = settings_page.ui.value_mappings.get(
            hd_strategy_settings.get("strategy", ""),
            hd_strategy_settings.get("strategy", ""),
        )
        for index, ctx in enumerate(self.pages):
            if ctx.failed_stage:
                continue
            _set_current_image(self.window, ctx.image_path)
            self.batch.emit_progress(index, total_images, 3, 10, False)
            self.batch._emit_benchmark_event(
                "inpaint_start",
                image_path=ctx.image_path,
                image_index=index,
                total_images=total_images,
                block_count=len(ctx.blk_list or []),
            )
            try:
                runtime = get_inpainter_runtime(settings_page)
                config = get_config(settings_page)
                ctx.mask_details = generate_mask(
                    ctx.image,
                    ctx.blk_list,
                    settings=settings_page.get_mask_refiner_settings(),
                    return_details=True,
                    precomputed_mask_details=ctx.precomputed_mask_details,
                )
                ctx.mask = ctx.mask_details["final_mask"]
                ctx.raw_mask = ctx.mask_details["raw_mask"]
                ctx.inpaint_input_img = self.window.pipeline.inpainting.inpaint_with_blocks(
                    ctx.image,
                    ctx.mask,
                    ctx.blk_list,
                    config=config,
                )
                ctx.inpaint_input_img = imk.convert_scale_abs(ctx.inpaint_input_img)
                ctx.patches = self.window.pipeline.inpainting.get_inpainted_patches(ctx.mask, ctx.inpaint_input_img)
                self.window.patches_processed.emit(ctx.patches, ctx.image_path)
                self.window.image_ctrl.update_processing_summary(
                    ctx.image_path,
                    {
                        "inpainter": settings_page.get_tool_selection("inpainter"),
                        "hd_strategy": hd_strategy,
                        "cleanup_applied": False,
                        "cleanup_component_count": 0,
                        "cleanup_block_count": 0,
                    },
                )
                self.batch._write_inpaint_debug_exports(
                    export_root=str(self.run_dir),
                    archive_bname=Path(ctx.image_path).stem,
                    image_path=ctx.image_path,
                    image=ctx.image,
                    blk_list=ctx.blk_list,
                    export_settings=export_settings,
                    raw_mask=ctx.raw_mask,
                    final_mask=ctx.mask,
                    detector_key=ctx.detector_key,
                    detector_engine=ctx.detector_engine,
                    detector_device=ctx.detector_device,
                    inpainter_key=runtime["key"],
                    hd_strategy=hd_strategy,
                    cleanup_stats=ctx.cleanup_stats,
                    mask_details=ctx.mask_details,
                    inpainter_backend=runtime["backend"],
                )
                self.window.image_ctrl.mark_processing_stage(
                    ctx.image_path,
                    "inpaint",
                    "completed",
                    patch_count=len(ctx.patches or []),
                )
                self.batch._emit_benchmark_event(
                    "inpaint_end",
                    image_path=ctx.image_path,
                    image_index=index,
                    total_images=total_images,
                    block_count=len(ctx.blk_list or []),
                    patch_count=len(ctx.patches or []),
                )
            except Exception as exc:
                _page_failed(
                    self.batch,
                    ctx,
                    index=index,
                    total_images=total_images,
                    stage="inpaint",
                    reason=str(exc),
                    extra={**ctx.page_ocr_metrics, **ctx.page_translation_metrics},
                )
        _emit_gpu_checkpoint(self.window, "inpaint_stage_end", self._active_container_names())

    def _render_all(self) -> None:
        assert self.batch is not None and self.window is not None
        total_images = len(self.pages)
        settings_page = self.window.settings_page
        export_settings = self.batch._effective_export_settings(settings_page)
        target_lang_en = self.window.lang_mapping.get(self.target_lang, self.target_lang)
        trg_lng_cd = get_language_code(target_lang_en)
        render_settings = self.window.render_settings()
        directory = str(self.run_dir)
        export_token = Path(self.run_dir).name
        for index, ctx in enumerate(self.pages):
            if ctx.failed_stage:
                continue
            _set_current_image(self.window, ctx.image_path)
            self.batch.emit_progress(index, total_images, 9, 10, False)
            self.batch._write_json_exports(
                directory,
                export_token,
                Path(ctx.image_path).stem,
                ctx.image_path,
                ctx.image,
                ctx.blk_list,
                self.batch._ensure_page_state(ctx.image_path),
                ctx.source_lang,
                export_settings,
            )
            self.batch._emit_benchmark_event(
                "render_start",
                image_path=ctx.image_path,
                image_index=index,
                total_images=total_images,
                block_count=len(ctx.blk_list or []),
            )
            try:
                format_translations(ctx.blk_list, trg_lng_cd, upper_case=render_settings.upper_case)
                get_best_render_area(ctx.blk_list, ctx.image, ctx.inpaint_input_img)
                self._render_page_text_items(
                    ctx,
                    index=index,
                    total_images=total_images,
                    render_settings=render_settings,
                    trg_lng_cd=trg_lng_cd,
                )
                page_state = self.batch._ensure_page_state(ctx.image_path)
                final_output_path, final_output_root = self.batch._write_final_render_export(
                    directory,
                    export_token,
                    ctx.image_path,
                    ctx.image,
                    ctx.patches,
                    page_state.get("viewer_state", {}),
                    export_settings,
                    page_index=index,
                    total_pages=total_images,
                )
                ctx.final_output_path = final_output_path
                self.window.image_ctrl.update_processing_summary(
                    ctx.image_path,
                    {
                        "translated_image_path": final_output_path,
                        "export_root": final_output_root,
                    },
                )
                self.batch._emit_benchmark_event(
                    "render_end",
                    image_path=ctx.image_path,
                    image_index=index,
                    total_images=total_images,
                    block_count=len(ctx.blk_list or []),
                    translated_image_path=final_output_path,
                )
                self.batch._emit_benchmark_event(
                    "page_done",
                    image_path=ctx.image_path,
                    image_index=index,
                    total_images=total_images,
                    block_count=len(ctx.blk_list or []),
                    patch_count=len(ctx.patches or []),
                )
                self.batch._log_page_done(
                    index,
                    total_images,
                    ctx.image_path,
                    preview_path=final_output_path,
                )
            except Exception as exc:
                _page_failed(
                    self.batch,
                    ctx,
                    index=index,
                    total_images=total_images,
                    stage="render",
                    reason=str(exc),
                    extra={**ctx.page_ocr_metrics, **ctx.page_translation_metrics},
                )
        _emit_gpu_checkpoint(self.window, "render_stage_end", self._active_container_names())

    def _render_page_text_items(
        self,
        ctx: PageContext,
        *,
        index: int,
        total_images: int,
        render_settings,
        trg_lng_cd: str,
    ) -> None:
        assert self.batch is not None and self.window is not None
        font = render_settings.font_family
        setting_font_color = QColor(render_settings.color)
        text_items_state: list[dict[str, Any]] = []
        page_state = self.batch._ensure_page_state(ctx.image_path)
        file_on_display = None
        if 0 <= self.window.curr_img_idx < len(self.window.image_files):
            file_on_display = self.window.image_files[self.window.curr_img_idx]

        for blk in ctx.blk_list:
            x1, y1, block_width, block_height = blk.xywh
            translation_raw = blk.translation
            if not translation_raw or len(translation_raw) == 1:
                continue

            render_normalization = describe_render_text_sanitization(
                translation_raw,
                font,
                block_index=getattr(blk, "_debug_block_index", None),
                image_path=ctx.image_path,
            )
            translation = render_normalization.text
            if not translation or len(translation) == 1:
                continue

            vertical = is_vertical_block(blk, trg_lng_cd)
            translation, font_size, rendered_width, rendered_height = pyside_word_wrap(
                translation,
                font,
                block_width,
                block_height,
                float(render_settings.line_spacing),
                float(render_settings.outline_width),
                render_settings.bold,
                render_settings.italic,
                render_settings.underline,
                self.window.button_to_alignment[render_settings.alignment_id],
                render_settings.direction,
                render_settings.max_font_size,
                render_settings.min_font_size,
                vertical,
                return_metrics=True,
            )
            if is_no_space_lang(trg_lng_cd):
                translation = translation.replace(" ", "")

            render_markup = describe_render_text_markup(translation)
            font_color = resolve_render_text_color(
                blk.font_color,
                setting_font_color,
                render_settings.force_font_color,
                render_settings.smart_global_apply_all,
            )
            source_rect = build_rect_tuple(x1, y1, block_width, block_height)
            outline_color = QColor(render_settings.outline_color) if render_settings.outline else None
            text_props = TextItemProperties(
                text=render_markup.html_text if render_markup.html_applied else translation,
                font_family=font,
                font_size=font_size,
                text_color=font_color,
                alignment=self.window.button_to_alignment[render_settings.alignment_id],
                line_spacing=float(render_settings.line_spacing),
                outline_color=outline_color,
                outline_width=float(render_settings.outline_width),
                bold=render_settings.bold,
                italic=render_settings.italic,
                underline=render_settings.underline,
                position=(x1, y1),
                rotation=blk.angle,
                scale=1.0,
                transform_origin=blk.tr_origin_point,
                width=rendered_width,
                height=rendered_height,
                direction=render_settings.direction,
                vertical=vertical,
                vertical_alignment=self.window.button_to_vertical_alignment.get(
                    render_settings.vertical_alignment_id,
                    VERTICAL_ALIGNMENT_TOP,
                ),
                source_rect=source_rect,
                block_anchor=source_rect,
                selection_outlines=[
                    OutlineInfo(
                        0,
                        len(translation),
                        outline_color,
                        float(render_settings.outline_width),
                        OutlineType.Full_Document,
                    )
                ] if render_settings.outline else [],
            )
            text_item_state = text_props.to_dict()
            text_item_state["translation_raw"] = str(translation_raw or "")
            text_item_state["render_text"] = str(translation or "")
            text_item_state["render_html_applied"] = bool(render_markup.html_applied)
            text_items_state.append(text_item_state)
            if ctx.image_path == file_on_display:
                self.window.blk_rendered.emit(translation, font_size, blk, ctx.image_path)

        page_state.setdefault("viewer_state", {}).update({"text_items_state": text_items_state, "push_to_stack": True})
        page_state["blk_list"] = ctx.blk_list
        self.window.image_ctrl.mark_processing_stage(
            ctx.image_path,
            "render",
            "completed",
            text_item_count=len(text_items_state),
        )
        self.window.image_ctrl.mark_processing_stage(
            ctx.image_path,
            "pipeline",
            "completed",
        )
        self.window.render_state_ready.emit(ctx.image_path)

    def _runtime_error_message(self, exc: Exception, *, context: str) -> str:
        if isinstance(exc, requests.exceptions.ConnectionError):
            return QCoreApplication.translate("Messages", "Unable to connect to the server.\nPlease check your internet connection.")
        if isinstance(exc, requests.exceptions.HTTPError):
            status_code = exc.response.status_code if exc.response is not None else 500
            if status_code >= 500:
                return Messages.get_server_error_text(status_code, context=context)
            try:
                err_json = exc.response.json()
                if "detail" in err_json and isinstance(err_json["detail"], dict):
                    return err_json["detail"].get("error_description", str(exc))
                return err_json.get("error_description", str(exc))
            except Exception:
                return str(exc)
        return str(exc)

    def run(self) -> dict[str, Any]:
        self._load_window()
        assert self.batch is not None and self.window is not None
        total_images = len(self.pages)
        self.batch._run_started_at = time.monotonic()
        self.batch._page_started_at = None
        self.batch._progress_image_path = None
        self.batch._recent_page_durations.clear()
        started = time.perf_counter()
        self.window.emit_memlog(
            "benchmark_run_start",
            benchmark_mode=f"stage_batched_{self.resident_ocr_mode}",
            total_images=total_images,
        )
        self.batch._emit_benchmark_event("batch_run_start", total_images=total_images)
        write_snapshot_json(
            self.run_dir / "runtime_snapshot.json",
            collect_runtime_snapshot(resolve_runtime_container_names(self.preset, "full")),
        )
        _emit_gpu_checkpoint(self.window, "run_start_pre_runtime", [])

        ocr_groups: list[dict[str, Any]] = []
        gemma_groups: list[dict[str, Any]] = []
        try:
            self._detect_all()
            ocr_groups = self._prepare_ocr_stage()
            self._ocr_all()
            _shutdown_runtime_groups(ocr_groups)
            _emit_gpu_checkpoint(self.window, "ocr_stage_shutdown", resolve_runtime_container_names(self.preset, "ocr-only"))

            gemma_groups = self._prepare_translate_stage()
            self._translate_all()
            _shutdown_runtime_groups(gemma_groups)
            _emit_gpu_checkpoint(self.window, "translate_stage_shutdown", GEMMA_CONTAINER_NAMES)

            self._inpaint_all()
            self._render_all()

            self.batch._emit_benchmark_event("batch_run_done", total_images=total_images)
            elapsed = time.perf_counter() - started
            self.window.emit_memlog(
                "benchmark_run_finished",
                benchmark_mode=f"stage_batched_{self.resident_ocr_mode}",
                elapsed_sec=round(elapsed, 3),
            )
            page_snapshots_path = _write_page_snapshots(self.window, self.run_dir, self.loaded_paths)
            _log(f"페이지 스냅샷 저장 완료: {page_snapshots_path}")
            write_json(self.run_dir / "runtime" / "managed_runtime_policy.json", self.runtime_policy)
            write_json(self.run_dir / "llama_cpp_runtime.json", self.llama_cpp_runtime)

            summary = summarize_metrics(self.run_dir / "metrics.jsonl")
            summary.update(
                {
                    "mode": f"stage-batched-{self.resident_ocr_mode}",
                    "image_count": total_images,
                    "image_paths": [repo_relative_str(path) for path in self.loaded_paths],
                    "workflow_mode": "stage_batched_pipeline",
                    "resident_ocr_mode": self.resident_ocr_mode,
                }
            )
            write_json(self.run_dir / "summary.json", summary)
            (self.run_dir / "summary.md").write_text(render_summary_markdown(summary), encoding="utf-8")
            return summary
        finally:
            try:
                _shutdown_runtime_groups(ocr_groups)
            except Exception:
                pass
            try:
                _shutdown_runtime_groups(gemma_groups)
            except Exception:
                pass
            self._close_window()


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the workflow split stage-batched benchmark pipeline.")
    parser.add_argument("--preset", required=True, help="Preset name or preset file path")
    parser.add_argument("--sample-dir", required=True, help="Sample directory to benchmark")
    parser.add_argument("--sample-count", type=int, required=True, help="Number of images to load from the sample directory")
    parser.add_argument("--source-lang", default="Japanese")
    parser.add_argument("--target-lang", default="Korean")
    parser.add_argument("--output-dir", required=True, help="Exact output directory for the benchmark run")
    parser.add_argument(
        "--resident-ocr-mode",
        choices=("single", "dual"),
        required=True,
        help="Whether the OCR stage keeps a single OCR runtime or both PaddleOCR VL and MangaLMM resident.",
    )
    args = parser.parse_args()

    preset, preset_path = load_preset(args.preset)
    corpus = resolve_corpus(args.sample_dir, sample_count=args.sample_count)
    selected_paths = corpus["representative"]
    run_dir = Path(args.output_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    staged_paths = _stage_selected_images(run_dir, selected_paths)

    write_json(
        run_dir / "benchmark_request.json",
        {
            "preset_name": preset.get("name", args.preset),
            "preset_path": str(preset_path),
            "mode": f"stage-batched-{args.resident_ocr_mode}",
            "runtime_mode": "managed",
            "runtime_services": "stage-batched",
            "source_lang": args.source_lang,
            "target_lang": args.target_lang,
            "selected_paths": [str(path) for path in selected_paths],
            "staged_paths": [str(path) for path in staged_paths],
        },
    )
    write_json(run_dir / "preset_resolved.json", preset)
    app = QApplication.instance() or QApplication([])
    runner = StageBatchedRunner(
        app=app,
        preset=preset,
        run_dir=run_dir,
        source_lang=args.source_lang,
        target_lang=args.target_lang,
        image_paths=staged_paths,
        resident_ocr_mode=args.resident_ocr_mode,
    )
    try:
        summary = runner.run()
        _log(
            "stage-batched 실행 완료: mode={mode} page_done={done} page_failed={failed}".format(
                mode=summary.get("mode", ""),
                done=summary.get("page_done_count"),
                failed=summary.get("page_failed_count"),
            )
        )
        return 0
    except Exception as exc:
        _log(f"stage-batched 실행 실패: {exc}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

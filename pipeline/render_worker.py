"""페이지 렌더의 순수 계산 워커 (Phase 3a).

인페인팅 sweep 뒤에 남는 렌더(약 300ms/page, 대부분 QGraphicsScene rasterize +
PNG 인코딩)를 전용 단일 스레드에 숨기기 위해 분리한 모듈이다.

**렌더 워커는 순수 함수다.** 이 모듈은 `StageBatchedProcessor`/`main_page`를
전혀 참조하지 않는다: 입력은 값(이미지 배열, 블록 목록, 렌더 설정, 미리
예약된 출력 경로)이고, 출력도 값(기록 완료 사실 + 메트릭)이다. 어떤 공유
상태도 쓰지 않고, 어떤 시그널도 발신하지 않고, 어떤 풀에도 재submit하지
않고, arbiter 락을 잡지 않는다. `main_page`에서 파생된 값(정렬, 표시 중인
페이지 여부, viewer_state 스냅샷, 최종 출력 경로)은 모두 파이프라인 스레드가
제출 전에 미리 해석해 값으로 넘긴다.
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import imkit as imk
import numpy as np
from PySide6.QtGui import QColor

from app.ui.canvas.save_renderer import ImageSaveRenderer
from app.ui.canvas.text.text_item_properties import TextItemProperties
from app.ui.canvas.text_item import OutlineInfo, OutlineType
from modules.ocr.common.result_contract import (
    PROCESSING_ACTION_TRANSLATE_INPAINT,
    finalize_ocr_processing_contract,
)
from modules.rendering.render import (
    apply_strict_render_viewer_state_guard,
    build_duplicate_bubble_render_key,
    build_render_rects_for_block,
    build_text_item_layout_geometry,
    describe_auto_render_review_status_gate,
    describe_render_text_markup,
    describe_render_text_sanitization,
    describe_text_free_large_mask_gate,
    describe_text_free_render_mask_gate,
    describe_text_free_render_translation_gate,
    describe_text_free_underfill_gate,
    get_best_render_area,
    get_render_fit_clearance_for_block,
    is_vertical_block,
    pyside_word_wrap,
    refit_detected_bubble_text_if_underfilled,
    register_duplicate_bubble_render_key,
    resolve_text_free_manga_layout,
    select_blocks_for_original_restore_after_render,
    should_skip_short_render_translation,
)
from modules.utils.automatic_output import write_image_with_format
from modules.utils.exceptions import OperationCancelledError
from modules.utils.image_utils import restore_original_for_block_masks
from modules.utils.language_utils import is_no_space_lang
from modules.utils.ocr_debug import (
    is_block_ocr_empty,
    is_bubble_panel_text_candidate,
    is_embedded_ui_panel_layout_review_candidate,
)
from modules.utils.render_style_policy import resolve_render_text_color
from modules.utils.translator_utils import format_translations


@dataclass
class RenderJobInput:
    image_path: str
    image: np.ndarray
    inpaint_input_img: np.ndarray | None
    mask: np.ndarray | None
    patches: list[dict[str, Any]]
    blk_list: list[Any]
    translation_blocks: list[Any]
    no_text_detected: bool
    trg_lng_cd: str
    render_settings: Any
    strict_render_symbols: bool
    alignment: Any
    vertical_alignment: Any
    viewer_state: dict[str, Any]
    output_path: str
    output_format: str
    is_cancelled: Callable[[], bool] = field(default=lambda: False)
    # 파이프라인 스레드가 이 작업을 제출한 시각(`time.monotonic`). 워커가 집어드는
    # 시각과의 차이가 큐 대기 시간이다.
    submitted_monotonic: float = 0.0


@dataclass
class RenderJobResult:
    viewer_state: dict[str, Any]
    blk_rendered_events: list[tuple[str, int, Any]]
    restore_applied: bool
    restore_stats: dict[str, Any]
    inpaint_input_img: np.ndarray | None
    mask: np.ndarray | None
    patches: list[dict[str, Any]]
    final_output_path: str
    # 워커 스레드가 실제로 일한 시간(초). 파이프라인 스레드는 제출만 하고 떠나므로
    # 이 값 없이는 렌더가 얼마나 비싼지 알 방법이 없다.
    worker_seconds: float = 0.0
    # 워커가 이 작업을 집어들기까지 큐에서 기다린 시간(초). 이 값이 커지면
    # 렌더가 인페인팅을 따라가지 못하고 병목이 됐다는 직접 증거다.
    queue_wait_seconds: float = 0.0


def _check_cancelled(is_cancelled: Callable[[], bool]) -> None:
    try:
        cancelled = bool(is_cancelled())
    except Exception:
        cancelled = False
    if cancelled:
        raise OperationCancelledError("Automatic translation was cancelled.")


def _slice_regular_mode_patches(
    mask: np.ndarray,
    inpainted_image: np.ndarray,
) -> list[dict[str, Any]]:
    """`pipeline/inpainting.py:InpaintingHandler.get_inpainted_patches`의 일반
    모드(웹툰 아님) 분기만 복제한다. 원본은 `self.main_page.webtoon_mode`를
    읽으므로 워커에서 직접 호출할 수 없다 — stage_batched 배치는 웹툰 모드를
    쓰지 않으므로(웹툰은 `pipeline/webtoon_batch/`가 별도 처리) 이 슬라이싱은
    원본과 동일한 결과를 낸다."""
    contours, _ = imk.find_contours(mask)
    patches: list[dict[str, Any]] = []
    for contour in contours:
        x, y, w, h = imk.bounding_rect(contour)
        patch = inpainted_image[y : y + h, x : x + w]
        patches.append({"bbox": [x, y, w, h], "image": patch.copy()})
    return patches


def _compute_render_text_items(
    blk_list: list[Any],
    *,
    image_path: str,
    render_settings: Any,
    trg_lng_cd: str,
    strict_render_symbols: bool,
    alignment: Any,
    vertical_alignment: Any,
) -> tuple[list[dict[str, Any]], list[tuple[str, int, Any]]]:
    """페이지 한 장의 블록들로부터 렌더 아이템 상태를 계산한다.

    구 `StageBatchedProcessor._render_page_text_items`의 순수 계산 부분을
    그대로 옮긴 것. `main_page.blk_rendered` 시그널 발신과 `page_state` 기록은
    호출자(파이프라인 스레드) 책임으로 남기고, 대신 값으로 반환한다.
    """
    font = render_settings.font_family
    setting_font_color = QColor(render_settings.color)
    text_items_state: list[dict[str, Any]] = []
    blk_rendered_events: list[tuple[str, int, Any]] = []
    seen_bubble_render_keys: set[tuple[tuple[int, int, int, int], str]] = set()

    for blk in blk_list:
        if is_block_ocr_empty(blk):
            continue
        finalize_ocr_processing_contract(blk)
        if getattr(blk, "processing_action", "") != PROCESSING_ACTION_TRANSLATE_INPAINT:
            blk._render_skip_reason = (
                "processing_action_" + str(getattr(blk, "processing_action", "") or "review")
            )
            continue
        x1, y1, block_width, block_height = blk.xywh
        translation_raw = blk.translation
        if should_skip_short_render_translation(blk, translation_raw):
            continue
        render_normalization = describe_render_text_sanitization(
            translation_raw,
            font,
            block_index=getattr(blk, "_debug_block_index", None),
            image_path=image_path,
            strict_symbols=strict_render_symbols,
        )
        translation = render_normalization.text
        blk._render_translation_raw = str(translation_raw or "")
        blk._render_text = str(translation or "")
        blk._render_normalization_applied = bool(render_normalization.normalization_applied)
        blk._render_normalization_reasons = list(render_normalization.reasons)
        blk._render_normalization_replacements = list(render_normalization.replacements)
        if should_skip_short_render_translation(blk, translation):
            continue
        gate_decision = describe_text_free_render_translation_gate(
            blk,
            translation,
            target_lang_code=trg_lng_cd,
        )
        if not gate_decision.render:
            blk._text_fit_status = gate_decision.status
            blk._render_skip_reason = gate_decision.status
            blk._render_normalization_reasons = sorted(
                set(getattr(blk, "_render_normalization_reasons", []) or []).union(gate_decision.reasons)
            )
            continue
        mask_gate_decision = describe_text_free_render_mask_gate(
            blk,
            target_lang_code=trg_lng_cd,
        )
        if not mask_gate_decision.render:
            blk._text_fit_status = mask_gate_decision.status
            blk._render_skip_reason = mask_gate_decision.status
            blk.mask_decision = "review"
            blk.mask_reject_reason = mask_gate_decision.status
            blk._render_normalization_reasons = sorted(
                set(getattr(blk, "_render_normalization_reasons", []) or []).union(mask_gate_decision.reasons)
            )
            continue
        duplicate_key = build_duplicate_bubble_render_key(blk)
        duplicate_gate = register_duplicate_bubble_render_key(
            blk,
            duplicate_key,
            seen_bubble_render_keys,
        )
        if not duplicate_gate.render:
            blk._text_fit_status = duplicate_gate.status
            blk._render_skip_reason = duplicate_gate.status
            blk._render_normalization_reasons = sorted(
                set(getattr(blk, "_render_normalization_reasons", []) or []).union(duplicate_gate.reasons)
            )
            continue
        if duplicate_gate.reasons:
            blk._render_normalization_reasons = sorted(
                set(getattr(blk, "_render_normalization_reasons", []) or []).union(duplicate_gate.reasons)
            )
        vertical = is_vertical_block(blk, trg_lng_cd)
        text_to_wrap = translation
        source_rect, block_anchor = build_render_rects_for_block(blk)
        block_width = int(source_rect[2])
        block_height = int(source_rect[3])
        block_alignment = alignment
        block_vertical_alignment = vertical_alignment
        layout_policy = resolve_text_free_manga_layout(
            blk,
            source_rect,
            target_lang_code=trg_lng_cd,
        )
        wrap_width = block_width
        if layout_policy.enabled:
            block_alignment = layout_policy.alignment
            block_vertical_alignment = layout_policy.vertical_alignment
            wrap_width = min(block_width, int(layout_policy.wrap_width))
        fit_clearance = get_render_fit_clearance_for_block(
            blk,
            render_settings.outline_width,
            auto_max_font_profile=getattr(render_settings, "auto_max_font_profile", "current"),
        )
        translation, font_size, rendered_width, rendered_height = pyside_word_wrap(
            text_to_wrap,
            font,
            wrap_width,
            block_height,
            float(render_settings.line_spacing),
            float(render_settings.outline_width),
            render_settings.bold,
            render_settings.italic,
            render_settings.underline,
            block_alignment,
            render_settings.direction,
            render_settings.max_font_size,
            render_settings.min_font_size,
            vertical,
            fit_clearance=fit_clearance,
            return_metrics=True,
        )
        translation, font_size, rendered_width, rendered_height = refit_detected_bubble_text_if_underfilled(
            blk,
            text_to_wrap,
            font,
            wrap_width,
            block_height,
            float(render_settings.line_spacing),
            float(render_settings.outline_width),
            render_settings.bold,
            render_settings.italic,
            render_settings.underline,
            block_alignment,
            render_settings.direction,
            render_settings.max_font_size,
            render_settings.min_font_size,
            vertical,
            fit_clearance,
            translation,
            font_size,
            rendered_width,
            rendered_height,
            auto_max_font_size=getattr(render_settings, "auto_max_font_size", True),
            auto_max_font_profile=getattr(render_settings, "auto_max_font_profile", "current"),
        )
        blk._text_fit_status = (
            "needs_review" if rendered_width > wrap_width or rendered_height > block_height else "fit"
        )
        if layout_policy.enabled and blk._text_fit_status != "fit":
            blk._text_fit_status = "needs_review_text_free_layout"
        underfill_gate = describe_text_free_underfill_gate(
            blk,
            source_rect=source_rect,
            rendered_width=rendered_width,
            rendered_height=rendered_height,
            target_lang_code=trg_lng_cd,
        )
        if blk._text_fit_status == "fit" and not underfill_gate.render:
            blk._text_fit_status = underfill_gate.status
            blk._render_normalization_reasons = sorted(
                set(getattr(blk, "_render_normalization_reasons", []) or []).union(underfill_gate.reasons)
            )
        large_mask_gate = describe_text_free_large_mask_gate(
            blk,
            source_rect=source_rect,
            target_lang_code=trg_lng_cd,
        )
        if blk._text_fit_status == "fit" and not large_mask_gate.render:
            blk._text_fit_status = large_mask_gate.status
            blk.mask_decision = "review"
            blk.mask_reject_reason = large_mask_gate.status
            blk._render_normalization_reasons = sorted(
                set(getattr(blk, "_render_normalization_reasons", []) or []).union(large_mask_gate.reasons)
            )
        if is_embedded_ui_panel_layout_review_candidate(blk) and not is_bubble_panel_text_candidate(blk):
            blk._text_fit_status = "needs_review_embedded_ui_panel_layout"
            blk._render_normalization_reasons = sorted(
                set(getattr(blk, "_render_normalization_reasons", []) or []).union(
                    {"needs_review_embedded_ui_panel_layout"}
                )
            )
        review_status_gate = describe_auto_render_review_status_gate(
            getattr(blk, "_text_fit_status", "fit")
        )
        if not review_status_gate.render:
            blk._render_skip_reason = review_status_gate.status
            blk._render_normalization_reasons = sorted(
                set(getattr(blk, "_render_normalization_reasons", []) or []).union(review_status_gate.reasons)
            )
            continue
        blk._text_fit_metrics = {
            "rendered_width": float(rendered_width),
            "rendered_height": float(rendered_height),
            "box_width": float(wrap_width),
            "box_height": float(block_height),
            "item_width": float(block_width),
            "font_size": float(font_size),
        }
        if is_no_space_lang(trg_lng_cd):
            translation = translation.replace(" ", "")
        font_color = resolve_render_text_color(
            blk.font_color,
            setting_font_color,
            render_settings.force_font_color,
            render_settings.smart_global_apply_all,
        )
        render_markup = describe_render_text_markup(
            translation,
            font_family=font,
            font_size=font_size,
            text_color=font_color,
            alignment=block_alignment,
            line_spacing=float(render_settings.line_spacing),
            bold=render_settings.bold,
            italic=render_settings.italic,
            underline=render_settings.underline,
            direction=render_settings.direction,
            strict_symbols=strict_render_symbols,
        )
        blk._render_text = str(translation or "")
        blk._render_html = str(render_markup.html_text if render_markup.html_applied else translation)
        blk._render_html_applied = bool(render_markup.html_applied)
        blk._render_fallback_font_family = str(render_markup.fallback_font_family or "")
        blk._render_normalization_applied = bool(
            render_normalization.normalization_applied or render_markup.html_applied
        )
        blk._render_normalization_reasons = sorted(
            set(render_normalization.reasons).union(render_markup.reasons).union(layout_policy.reasons)
        )
        blk._render_normalization_replacements = list(render_normalization.replacements) + list(
            render_markup.replacements
        )
        blk._render_centered_layout = bool(layout_policy.enabled)
        blk._render_layout_reasons = list(layout_policy.reasons)
        position, item_width, item_height = build_text_item_layout_geometry(
            source_rect,
            rendered_height,
            block_vertical_alignment,
        )
        outline_color = QColor(render_settings.outline_color) if render_settings.outline else None
        text_props = TextItemProperties(
            text=blk._render_html,
            font_family=font,
            font_size=font_size,
            text_color=font_color,
            alignment=block_alignment,
            line_spacing=float(render_settings.line_spacing),
            outline_color=outline_color,
            outline_width=float(render_settings.outline_width),
            bold=render_settings.bold,
            italic=render_settings.italic,
            underline=render_settings.underline,
            position=position,
            rotation=blk.angle,
            scale=1.0,
            transform_origin=blk.tr_origin_point,
            width=item_width,
            height=item_height,
            direction=render_settings.direction,
            vertical=vertical,
            vertical_alignment=block_vertical_alignment,
            source_rect=source_rect,
            block_anchor=block_anchor,
            selection_outlines=[
                OutlineInfo(
                    0,
                    len(translation),
                    outline_color,
                    float(render_settings.outline_width),
                    OutlineType.Full_Document,
                )
            ]
            if render_settings.outline
            else [],
        )
        text_item_state = text_props.to_dict()
        text_item_state["translation_raw"] = str(translation_raw or "")
        text_item_state["render_text"] = str(translation or "")
        text_item_state["render_html_applied"] = bool(render_markup.html_applied)
        text_item_state["render_fallback_font_family"] = str(render_markup.fallback_font_family or "")
        text_item_state["render_area_source"] = str(
            getattr(blk, "_render_area_source", "text_bbox") or "text_bbox"
        )
        text_item_state["render_source_xyxy"] = list(getattr(blk, "_render_area_xyxy", []) or [])
        text_item_state["render_anchor_xyxy"] = list(getattr(blk, "_render_original_xyxy", []) or [])
        text_item_state["render_bubble_xyxy"] = list(getattr(blk, "_render_bubble_xyxy", []) or [])
        text_item_state["render_normalization_applied"] = bool(blk._render_normalization_applied)
        text_item_state["render_normalization_reasons"] = list(blk._render_normalization_reasons)
        text_item_state["render_centered_layout"] = bool(layout_policy.enabled)
        text_item_state["render_layout_reasons"] = list(layout_policy.reasons)
        text_item_state["text_fit_status"] = str(getattr(blk, "_text_fit_status", "fit") or "fit")
        text_item_state["text_fit_metrics"] = dict(getattr(blk, "_text_fit_metrics", {}) or {})
        text_items_state.append(text_item_state)
        blk_rendered_events.append((translation, font_size, blk))

    return text_items_state, blk_rendered_events


def run_render_job(job: RenderJobInput) -> RenderJobResult:
    """페이지 한 장을 렌더해 최종 이미지를 `job.output_path`에 쓴다.

    입력은 전부 값, 출력은 전부 값이다. `main_page`를 참조하지 않는다.
    """
    started_monotonic = time.monotonic()
    queue_wait_seconds = (
        max(0.0, started_monotonic - float(job.submitted_monotonic))
        if job.submitted_monotonic
        else 0.0
    )
    _check_cancelled(job.is_cancelled)

    canonical_translations = [
        (getattr(block, "translation", ""), copy.deepcopy(getattr(block, "rich_text", "")))
        for block in job.blk_list
    ]
    if not job.no_text_detected:
        try:
            format_translations(
                job.translation_blocks,
                job.trg_lng_cd,
                upper_case=job.render_settings.upper_case,
            )
            _check_cancelled(job.is_cancelled)
            get_best_render_area(
                job.translation_blocks,
                job.image,
                job.inpaint_input_img,
                auto_max_font_profile=getattr(job.render_settings, "auto_max_font_profile", "current"),
            )
            text_items_state, blk_rendered_events = _compute_render_text_items(
                job.blk_list,
                image_path=job.image_path,
                render_settings=job.render_settings,
                trg_lng_cd=job.trg_lng_cd,
                strict_render_symbols=job.strict_render_symbols,
                alignment=job.alignment,
                vertical_alignment=job.vertical_alignment,
            )
        finally:
            for block, (translation, rich_text) in zip(job.blk_list, canonical_translations):
                block.translation = translation
                block.rich_text = rich_text
    else:
        text_items_state, blk_rendered_events = _compute_render_text_items(
            job.blk_list,
            image_path=job.image_path,
            render_settings=job.render_settings,
            trg_lng_cd=job.trg_lng_cd,
            strict_render_symbols=job.strict_render_symbols,
            alignment=job.alignment,
            vertical_alignment=job.vertical_alignment,
        )

    restore_applied = False
    restore_stats: dict[str, Any] = {}
    inpaint_input_img = job.inpaint_input_img
    mask = job.mask
    patches = job.patches
    restore_blocks = select_blocks_for_original_restore_after_render(job.blk_list)
    if restore_blocks and job.inpaint_input_img is not None and job.mask is not None:
        inpaint_input_img, mask, restore_stats = restore_original_for_block_masks(
            job.image,
            job.inpaint_input_img,
            job.mask,
            restore_blocks,
        )
        if restore_stats.get("applied"):
            restore_applied = True
            patches = _slice_regular_mode_patches(mask, inpaint_input_img)

    _check_cancelled(job.is_cancelled)

    viewer_state = dict(job.viewer_state)
    viewer_state["text_items_state"] = text_items_state
    viewer_state["push_to_stack"] = True
    if job.strict_render_symbols:
        apply_strict_render_viewer_state_guard(viewer_state, image_path=job.image_path)

    _check_cancelled(job.is_cancelled)
    renderer = ImageSaveRenderer(job.image)
    renderer.apply_patches(patches or [])
    renderer.add_state_to_image(viewer_state)
    final_rgb = renderer.render_to_image()
    final_output_path = write_image_with_format(job.output_path, final_rgb, job.output_format)

    return RenderJobResult(
        viewer_state=viewer_state,
        blk_rendered_events=blk_rendered_events,
        restore_applied=restore_applied,
        restore_stats=restore_stats,
        inpaint_input_img=inpaint_input_img,
        mask=mask,
        patches=patches,
        final_output_path=final_output_path,
        worker_seconds=max(0.0, time.monotonic() - started_monotonic),
        queue_wait_seconds=queue_wait_seconds,
    )

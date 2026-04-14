from __future__ import annotations

import base64
import json
import logging
import math
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from itertools import count
from pathlib import Path
from threading import Lock

import cv2
import numpy as np
import requests

from modules.utils.exceptions import (
    LocalServiceConnectionError,
    LocalServiceResponseError,
)
from modules.utils.language_utils import is_no_space_lang
from modules.utils.ocr_debug import (
    OCR_STATUS_EMPTY_AFTER_RETRY,
    OCR_STATUS_EMPTY_INITIAL,
    OCR_STATUS_OK,
    OCR_STATUS_OK_AFTER_RETRY,
    ensure_three_channel,
    set_block_ocr_crop_diagnostics,
    set_block_ocr_diagnostics,
)
from modules.utils.ocr_quality import summarize_ocr_quality
from modules.utils.textblock import TextBlock, sort_textblock_rectangles

from .base import OCREngine


logger = logging.getLogger(__name__)

SERVICE_NAME = "MangaLMM"
SETTINGS_PAGE_NAME = "MangaLMM Settings"
DEFAULT_MANGALMM_SERVER_URL = "http://127.0.0.1:28081/v1"
DEFAULT_MANGALMM_MAX_COMPLETION_TOKENS = 256
DEFAULT_MANGALMM_PARALLEL_WORKERS = 1
DEFAULT_MANGALMM_REQUEST_TIMEOUT_SEC = 60
DEFAULT_MANGALMM_SAFE_RESIZE = True
DEFAULT_MANGALMM_MAX_PIXELS = 2_116_800
DEFAULT_MANGALMM_MAX_LONG_SIDE = 1728
DEFAULT_MANGALMM_TEMPERATURE = 0.1
DEFAULT_MANGALMM_TOP_K = 32
DEFAULT_MANGALMM_PAGE_TILE_SIZE = 1280
DEFAULT_MANGALMM_TILE_OVERLAP = 192
DEFAULT_MANGALMM_RESCUE_CONTEXT_RATIO = 0.25
DEFAULT_MANGALMM_RESCUE_MIN_SIZE = 512
DEFAULT_MANGALMM_RESCUE_GAP = 96
DEFAULT_MANGALMM_TEXT_EXPANSION_RATIO_X = DEFAULT_MANGALMM_RESCUE_CONTEXT_RATIO
DEFAULT_MANGALMM_TEXT_EXPANSION_RATIO_Y = DEFAULT_MANGALMM_RESCUE_CONTEXT_RATIO
DEFAULT_MANGALMM_DEBUG_EXPORT_LIMIT = 96


@dataclass(slots=True)
class RequestUnit:
    bbox_xyxy: tuple[int, int, int, int]
    unit_kind: str


@dataclass(slots=True)
class OCRRegion:
    bbox_xyxy: list[int]
    text: str
    unit_bbox_xyxy: list[int]
    unit_kind: str
    unit_resize_scale: float
    edge_distance: float
    normalized_text: str


class MangaLMMOCREngine(OCREngine):
    MAX_COMPLETION_TOKENS_RANGE = (64, 2048)
    PARALLEL_WORKERS_RANGE = (1, 8)
    REQUEST_ENDPOINT_SUFFIX = "/chat/completions"
    PROMPT = (
        'Please perform OCR on this image and output only a JSON array of recognized text regions, '
        'where each item has "bbox_2d" and "text_content". Do not translate. Do not explain. '
        "Do not add markdown or code fences."
    )

    def __init__(self) -> None:
        self.server_url = DEFAULT_MANGALMM_SERVER_URL
        self.max_completion_tokens = DEFAULT_MANGALMM_MAX_COMPLETION_TOKENS
        self.parallel_workers = DEFAULT_MANGALMM_PARALLEL_WORKERS
        self.request_timeout_sec = DEFAULT_MANGALMM_REQUEST_TIMEOUT_SEC
        self.raw_response_logging = False
        self.safe_resize = DEFAULT_MANGALMM_SAFE_RESIZE
        self.max_pixels = DEFAULT_MANGALMM_MAX_PIXELS
        self.max_long_side = DEFAULT_MANGALMM_MAX_LONG_SIDE
        self.temperature = DEFAULT_MANGALMM_TEMPERATURE
        self.top_k = DEFAULT_MANGALMM_TOP_K
        self.page_tile_size = DEFAULT_MANGALMM_PAGE_TILE_SIZE
        self.tile_overlap = DEFAULT_MANGALMM_TILE_OVERLAP
        self.rescue_context_ratio = DEFAULT_MANGALMM_RESCUE_CONTEXT_RATIO
        self.rescue_min_size = DEFAULT_MANGALMM_RESCUE_MIN_SIZE
        self.rescue_gap = DEFAULT_MANGALMM_RESCUE_GAP
        self.text_expansion_ratio_x = DEFAULT_MANGALMM_TEXT_EXPANSION_RATIO_X
        self.text_expansion_ratio_y = DEFAULT_MANGALMM_TEXT_EXPANSION_RATIO_Y
        self.debug_root: Path | None = None
        self.debug_export_limit = DEFAULT_MANGALMM_DEBUG_EXPORT_LIMIT
        self._debug_export_counter = count(1)
        self._debug_export_lock = Lock()

    def initialize(self, settings, **kwargs) -> None:
        config = settings.get_mangalmm_ocr_settings()
        self.server_url = config.get("server_url", DEFAULT_MANGALMM_SERVER_URL) or DEFAULT_MANGALMM_SERVER_URL
        self.max_completion_tokens = self._clamp_int(
            config.get("max_completion_tokens", DEFAULT_MANGALMM_MAX_COMPLETION_TOKENS),
            DEFAULT_MANGALMM_MAX_COMPLETION_TOKENS,
            self.MAX_COMPLETION_TOKENS_RANGE,
        )
        self.parallel_workers = self._clamp_int(
            config.get("parallel_workers", DEFAULT_MANGALMM_PARALLEL_WORKERS),
            DEFAULT_MANGALMM_PARALLEL_WORKERS,
            self.PARALLEL_WORKERS_RANGE,
        )
        self.request_timeout_sec = max(
            1,
            self._clamp_int(
                config.get("request_timeout_sec", DEFAULT_MANGALMM_REQUEST_TIMEOUT_SEC),
                DEFAULT_MANGALMM_REQUEST_TIMEOUT_SEC,
                (1, 3600),
            ),
        )
        self.raw_response_logging = bool(config.get("raw_response_logging", False))
        self.safe_resize = bool(config.get("safe_resize", DEFAULT_MANGALMM_SAFE_RESIZE))
        self.max_pixels = max(
            100_000,
            self._clamp_int(
                config.get("max_pixels", DEFAULT_MANGALMM_MAX_PIXELS),
                DEFAULT_MANGALMM_MAX_PIXELS,
                (100_000, 8_000_000),
            ),
        )
        self.max_long_side = max(
            512,
            self._clamp_int(
                config.get("max_long_side", DEFAULT_MANGALMM_MAX_LONG_SIDE),
                DEFAULT_MANGALMM_MAX_LONG_SIDE,
                (512, 4096),
            ),
        )
        self.temperature = self._read_env_float(
            "CT_MANGALMM_TEMPERATURE",
            config.get("temperature", DEFAULT_MANGALMM_TEMPERATURE),
            DEFAULT_MANGALMM_TEMPERATURE,
            minimum=0.0,
            maximum=2.0,
        )
        self.top_k = self._read_env_int(
            "CT_MANGALMM_TOP_K",
            config.get("top_k", DEFAULT_MANGALMM_TOP_K),
            DEFAULT_MANGALMM_TOP_K,
            bounds=(1, 256),
        )
        default_x_ratio = config.get("text_expansion_ratio_x", DEFAULT_MANGALMM_TEXT_EXPANSION_RATIO_X)
        default_y_ratio = config.get("text_expansion_ratio_y", DEFAULT_MANGALMM_TEXT_EXPANSION_RATIO_Y)
        shared_ratio = os.getenv("CT_MANGALMM_TEXT_EXPANSION_RATIO", "").strip()
        if shared_ratio:
            default_x_ratio = shared_ratio
            default_y_ratio = shared_ratio
        self.text_expansion_ratio_x = self._read_env_float(
            "CT_MANGALMM_TEXT_EXPANSION_RATIO_X",
            default_x_ratio,
            DEFAULT_MANGALMM_TEXT_EXPANSION_RATIO_X,
            minimum=0.0,
            maximum=1.0,
        )
        self.text_expansion_ratio_y = self._read_env_float(
            "CT_MANGALMM_TEXT_EXPANSION_RATIO_Y",
            default_y_ratio,
            DEFAULT_MANGALMM_TEXT_EXPANSION_RATIO_Y,
            minimum=0.0,
            maximum=1.0,
        )
        debug_root_raw = os.getenv("CT_MANGALMM_DEBUG_ROOT", "").strip()
        self.debug_root = Path(debug_root_raw) if debug_root_raw else None
        if self.debug_root is not None:
            self.debug_root.mkdir(parents=True, exist_ok=True)
        self.debug_export_limit = self._read_env_int(
            "CT_MANGALMM_DEBUG_EXPORT_LIMIT",
            DEFAULT_MANGALMM_DEBUG_EXPORT_LIMIT,
            DEFAULT_MANGALMM_DEBUG_EXPORT_LIMIT,
            bounds=(1, 4096),
        )

    def process_image(self, img: np.ndarray, blk_list: list[TextBlock]) -> list[TextBlock]:
        blocks = list(blk_list or [])
        if not blocks:
            return blk_list

        image = ensure_three_channel(img)
        started_at = time.perf_counter()
        page_units = self._build_request_units(image.shape)
        page_regions = self._collect_regions_for_units(image, page_units)

        assignments = self._assign_regions_to_blocks(page_regions, blocks)
        quality = self._apply_assignments_to_blocks(
            blocks,
            assignments,
            attempt_count=1,
            success_status=OCR_STATUS_OK,
            empty_status=OCR_STATUS_EMPTY_INITIAL,
        )

        rescue_regions: list[OCRRegion] = []
        rescue_unit_count = 0
        if quality.get("low_quality", False):
            rescue_units = self._build_rescue_units(blocks, image.shape)
            rescue_unit_count = len(rescue_units)
            if rescue_units:
                rescue_regions = self._collect_regions_for_units(image, rescue_units)
                if rescue_regions:
                    merged_regions = self._dedupe_regions([*page_regions, *rescue_regions])
                    assignments = self._assign_regions_to_blocks(merged_regions, blocks)
                    quality = self._apply_assignments_to_blocks(
                        blocks,
                        assignments,
                        attempt_count=2,
                        success_status=OCR_STATUS_OK_AFTER_RETRY,
                        empty_status=OCR_STATUS_EMPTY_AFTER_RETRY,
                    )
                    page_regions = merged_regions

        logger.info(
            "mangalmm_page_tile_ocr complete: blocks=%d page_units=%d rescue_units=%d regions=%d quality=%s elapsed_ms=%.1f",
            len(blocks),
            len(page_units),
            rescue_unit_count,
            len(page_regions),
            quality.get("reason", "") or "ok",
            (time.perf_counter() - started_at) * 1000.0,
        )
        return blk_list

    def _build_request_units(self, image_shape: tuple[int, ...]) -> list[RequestUnit]:
        image_h, image_w = image_shape[:2]
        if image_h <= 0 or image_w <= 0:
            return []

        area = image_w * image_h
        if area <= self.max_pixels:
            return [RequestUnit((0, 0, image_w, image_h), "page_full")]

        units: list[RequestUnit] = []
        x_starts = self._build_axis_starts(image_w, self.page_tile_size, self.tile_overlap)
        y_starts = self._build_axis_starts(image_h, self.page_tile_size, self.tile_overlap)
        for y1 in y_starts:
            for x1 in x_starts:
                x2 = min(image_w, x1 + self.page_tile_size)
                y2 = min(image_h, y1 + self.page_tile_size)
                units.append(RequestUnit((int(x1), int(y1), int(x2), int(y2)), "page_tile"))
        return units

    def _build_rescue_units(self, blk_list: list[TextBlock], image_shape: tuple[int, ...]) -> list[RequestUnit]:
        support_boxes: list[tuple[int, int, int, int]] = []
        for blk in blk_list:
            if str(getattr(blk, "text", "") or "").strip():
                continue
            support_box = self._support_box(blk)
            if support_box is None:
                continue
            support_boxes.append(
                self._expand_box_with_min_size(
                    support_box,
                    image_shape,
                    x_ratio=self.text_expansion_ratio_x,
                    y_ratio=self.text_expansion_ratio_y,
                    min_size=self.rescue_min_size,
                )
            )
        merged = self._merge_boxes(support_boxes, image_shape, gap=self.rescue_gap)
        return [RequestUnit(tuple(box), "rescue_macro") for box in merged]

    def _collect_regions_for_units(self, image: np.ndarray, units: list[RequestUnit]) -> list[OCRRegion]:
        if not units:
            return []

        worker_count = min(self.parallel_workers, len(units))
        logger.info(
            "mangalmm_page_tile_ocr start: units=%d workers=%d max_completion_tokens=%d "
            "temperature=%.3f top_k=%d rescue_ratio=(%.3f, %.3f) endpoint=%s",
            len(units),
            worker_count,
            self.max_completion_tokens,
            self.temperature,
            self.top_k,
            self.text_expansion_ratio_x,
            self.text_expansion_ratio_y,
            self._chat_completions_url(),
        )

        if worker_count <= 1:
            collected = [self._request_regions_for_unit(image, unit) for unit in units]
        else:
            collected = []
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                future_map = {
                    executor.submit(self._request_regions_for_unit, image, unit): unit
                    for unit in units
                }
                try:
                    for future in as_completed(future_map):
                        collected.append(future.result())
                except Exception:
                    for future in future_map:
                        future.cancel()
                    raise

        flattened = [region for unit_regions in collected for region in unit_regions]
        return self._dedupe_regions(flattened)

    def _request_regions_for_unit(self, image: np.ndarray, unit: RequestUnit) -> list[OCRRegion]:
        crop = self._crop_image(image, unit.bbox_xyxy)
        if crop is None:
            self._export_debug_artifact(
                blk=None,
                failure_reason="Invalid OCR unit crop bounds.",
                response_kind="invalid_crop",
                raw_text="",
                crop_bbox=unit.bbox_xyxy,
                crop_source=unit.unit_kind,
                resize_scale=1.0,
                crop_image=None,
                request_image=None,
                analysis={},
                unit_bbox=unit.bbox_xyxy,
                unit_kind=unit.unit_kind,
            )
            return []

        request_image, scale = self._resize_for_request(crop)
        raw_text = self._request_response_text(request_image)
        analysis = self._analyze_region_payload(raw_text)
        regions = analysis["regions"]
        if not regions:
            response_kind = str(analysis.get("response_kind", "") or "unknown")
            self._export_debug_artifact(
                blk=None,
                failure_reason="MangaLMM returned no valid OCR regions.",
                response_kind=response_kind,
                raw_text=raw_text,
                crop_bbox=unit.bbox_xyxy,
                crop_source=unit.unit_kind,
                resize_scale=scale,
                crop_image=crop,
                request_image=request_image,
                analysis=analysis,
                unit_bbox=unit.bbox_xyxy,
                unit_kind=unit.unit_kind,
            )
            return []

        mapped = self._map_regions_to_page_coords(
            regions,
            unit.bbox_xyxy,
            crop.shape[:2],
            scale,
            unit.unit_kind,
        )
        if not mapped:
            response_kind = str(analysis.get("response_kind", "") or "unknown")
            self._export_debug_artifact(
                blk=None,
                failure_reason="MangaLMM returned OCR regions outside the unit bounds.",
                response_kind=response_kind,
                raw_text=raw_text,
                crop_bbox=unit.bbox_xyxy,
                crop_source=unit.unit_kind,
                resize_scale=scale,
                crop_image=crop,
                request_image=request_image,
                analysis=analysis,
                unit_bbox=unit.bbox_xyxy,
                unit_kind=unit.unit_kind,
            )
            return []
        return mapped

    def _assign_regions_to_blocks(
        self,
        regions: list[OCRRegion],
        blk_list: list[TextBlock],
    ) -> dict[int, list[dict[str, object]]]:
        assignments: dict[int, list[dict[str, object]]] = {index: [] for index in range(len(blk_list))}
        for region in regions:
            best_match = self._select_best_block(region, blk_list)
            if best_match is None:
                continue
            blk_index, match_metrics = best_match
            assignments[blk_index].append(
                {
                    "region": region,
                    "metrics": match_metrics,
                }
            )
        return assignments

    def _apply_assignments_to_blocks(
        self,
        blk_list: list[TextBlock],
        assignments: dict[int, list[dict[str, object]]],
        *,
        attempt_count: int,
        success_status: str,
        empty_status: str,
    ) -> dict:
        for index, blk in enumerate(blk_list):
            items = assignments.get(index, [])
            if not items:
                self._mark_empty(
                    blk,
                    "MangaLMM page/tile OCR did not match any OCR region to this block.",
                    attempt_count=attempt_count,
                    status=empty_status,
                )
                continue

            sorted_entries = sort_textblock_rectangles(
                [
                    (tuple(item["region"].bbox_xyxy), item["region"].text)
                    for item in items
                    if str(item["region"].text or "").strip()
                ],
                blk.source_lang_direction,
            )
            texts = [text for _bbox, text in sorted_entries if str(text or "").strip()]
            if is_no_space_lang(getattr(blk, "source_lang", "")):
                final_text = "".join(texts)
            else:
                final_text = " ".join(texts)

            best_item = max(
                items,
                key=lambda item: (
                    float(item["metrics"]["ownership_cover"]),
                    int(bool(item["metrics"]["center_in_ownership"])),
                    float(item["metrics"]["precision_cover"]),
                    float(item["metrics"]["ownership_iou"]),
                    -float(item["metrics"]["center_distance_norm"]),
                ),
            )
            source_region: OCRRegion = best_item["region"]
            blk.texts = texts
            blk.text = final_text
            blk.ocr_regions = [
                {
                    "bbox_xyxy": list(item["region"].bbox_xyxy),
                    "text": item["region"].text,
                }
                for item in items
            ]
            blk.ocr_crop_bbox = list(source_region.unit_bbox_xyxy)
            blk.ocr_resize_scale = float(source_region.unit_resize_scale)
            set_block_ocr_crop_diagnostics(
                blk,
                effective_crop_xyxy=source_region.unit_bbox_xyxy,
                crop_source=source_region.unit_kind,
            )
            set_block_ocr_diagnostics(
                blk,
                text=final_text,
                confidence=0.0,
                status=success_status,
                empty_reason="",
                attempt_count=attempt_count,
                raw_text=final_text,
                sanitized_text=final_text,
            )

        return summarize_ocr_quality(blk_list)

    def _select_best_block(
        self,
        region: OCRRegion,
        blk_list: list[TextBlock],
    ) -> tuple[int, dict[str, float | bool]] | None:
        region_box = tuple(region.bbox_xyxy)
        candidates: list[tuple[tuple[float, int, float, float, float, int], int, dict[str, float | bool]]] = []
        for index, blk in enumerate(blk_list):
            metrics = self._compute_match_metrics(region_box, blk)
            if metrics is None:
                continue
            sort_key = (
                -float(metrics["ownership_cover"]),
                -int(bool(metrics["center_in_ownership"])),
                -float(metrics["precision_cover"]),
                -float(metrics["ownership_iou"]),
                float(metrics["center_distance_norm"]),
                int(metrics["precision_area"]),
            )
            candidates.append((sort_key, index, metrics))

        if not candidates:
            return None

        candidates.sort(key=lambda item: item[0])
        _sort_key, index, metrics = candidates[0]
        if not self._accept_match(blk_list[index], metrics):
            return None
        return index, metrics

    def _compute_match_metrics(
        self,
        region_box: tuple[int, int, int, int],
        blk: TextBlock,
    ) -> dict[str, float | bool] | None:
        support_box = self._support_box(blk)
        ownership_box = self._ownership_box(blk)
        precision_box = self._precision_box(blk)
        if support_box is None or ownership_box is None or precision_box is None:
            return None

        region_area = self._bbox_area(region_box)
        if region_area <= 0:
            return None

        support_intersection = self._intersection_area(region_box, support_box)
        center = self._bbox_center(region_box)
        center_in_support = self._point_in_box(center, support_box)
        if not center_in_support and (support_intersection / float(region_area)) < 0.10:
            return None

        ownership_cover = self._intersection_area(region_box, ownership_box) / float(region_area)
        precision_cover = self._intersection_area(region_box, precision_box) / float(region_area)
        ownership_iou = self._bbox_iou(region_box, ownership_box)
        center_in_ownership = self._point_in_box(center, ownership_box)
        center_in_precision = self._point_in_box(center, precision_box)
        precision_center = self._bbox_center(precision_box)
        precision_diag = max(1.0, self._bbox_diagonal(precision_box))
        center_distance_norm = self._distance(center, precision_center) / precision_diag

        return {
            "ownership_cover": ownership_cover,
            "precision_cover": precision_cover,
            "ownership_iou": ownership_iou,
            "center_in_ownership": center_in_ownership,
            "center_in_precision": center_in_precision,
            "center_distance_norm": center_distance_norm,
            "precision_area": self._bbox_area(precision_box),
        }

    def _accept_match(self, blk: TextBlock, metrics: dict[str, float | bool]) -> bool:
        bubble_box = getattr(blk, "bubble_xyxy", None)
        ownership_cover = float(metrics["ownership_cover"])
        precision_cover = float(metrics["precision_cover"])
        ownership_iou = float(metrics["ownership_iou"])
        center_in_ownership = bool(metrics["center_in_ownership"])
        center_in_precision = bool(metrics["center_in_precision"])
        if bubble_box is not None:
            return ownership_cover >= 0.35 or (center_in_ownership and precision_cover >= 0.20)
        return precision_cover >= 0.35 or (center_in_precision and ownership_iou >= 0.10)

    def _dedupe_regions(self, regions: list[OCRRegion]) -> list[OCRRegion]:
        deduped: list[OCRRegion] = []
        for region in regions:
            replaced = False
            for index, existing in enumerate(deduped):
                if region.normalized_text != existing.normalized_text:
                    continue
                if self._bbox_iou(tuple(region.bbox_xyxy), tuple(existing.bbox_xyxy)) < 0.5:
                    continue
                if self._prefer_region(region, existing):
                    deduped[index] = region
                replaced = True
                break
            if not replaced:
                deduped.append(region)
        return deduped

    @staticmethod
    def _prefer_region(candidate: OCRRegion, existing: OCRRegion) -> bool:
        candidate_key = (
            float(candidate.edge_distance),
            int((candidate.bbox_xyxy[2] - candidate.bbox_xyxy[0]) * (candidate.bbox_xyxy[3] - candidate.bbox_xyxy[1])),
            len(candidate.text),
        )
        existing_key = (
            float(existing.edge_distance),
            int((existing.bbox_xyxy[2] - existing.bbox_xyxy[0]) * (existing.bbox_xyxy[3] - existing.bbox_xyxy[1])),
            len(existing.text),
        )
        return candidate_key > existing_key

    def _resize_for_request(self, crop: np.ndarray) -> tuple[np.ndarray, float]:
        if not self.safe_resize:
            return crop, 1.0

        height, width = crop.shape[:2]
        if height <= 0 or width <= 0:
            return crop, 1.0

        area = width * height
        long_side = max(width, height)
        if area <= self.max_pixels and long_side <= self.max_long_side:
            return crop, 1.0

        scale = min(
            1.0,
            math.sqrt(self.max_pixels / float(area)) if area > 0 else 1.0,
            self.max_long_side / float(long_side) if long_side > 0 else 1.0,
        )
        if scale >= 1.0:
            return crop, 1.0

        resized_width = max(1, int(round(width * scale)))
        resized_height = max(1, int(round(height * scale)))
        resized = cv2.resize(crop, (resized_width, resized_height), interpolation=cv2.INTER_AREA)
        return resized, float(scale)

    def _map_regions_to_page_coords(
        self,
        regions: list[dict[str, object]],
        unit_bbox: tuple[int, int, int, int],
        unit_shape: tuple[int, int],
        scale: float,
        unit_kind: str,
    ) -> list[OCRRegion]:
        unit_x1, unit_y1, unit_x2, unit_y2 = [int(v) for v in unit_bbox]
        unit_h, unit_w = unit_shape
        inverse_scale = 1.0 / scale if scale > 0 else 1.0
        mapped: list[OCRRegion] = []

        for region in regions:
            bbox = region.get("bbox_2d")
            text = str(region.get("text_content", "") or "").strip()
            if not isinstance(bbox, list) or len(bbox) != 4 or not text:
                continue
            x1_f, y1_f, x2_f, y2_f = [float(value) for value in bbox]
            local_x1 = max(0.0, min(x1_f, x2_f) * inverse_scale)
            local_y1 = max(0.0, min(y1_f, y2_f) * inverse_scale)
            local_x2 = min(float(unit_w), max(x1_f, x2_f) * inverse_scale)
            local_y2 = min(float(unit_h), max(y1_f, y2_f) * inverse_scale)
            if local_x2 <= local_x1 or local_y2 <= local_y1:
                continue

            page_x1 = max(unit_x1, min(int(round(unit_x1 + local_x1)), unit_x2))
            page_y1 = max(unit_y1, min(int(round(unit_y1 + local_y1)), unit_y2))
            page_x2 = max(unit_x1, min(int(round(unit_x1 + local_x2)), unit_x2))
            page_y2 = max(unit_y1, min(int(round(unit_y1 + local_y2)), unit_y2))
            if page_x2 <= page_x1 or page_y2 <= page_y1:
                continue

            edge_distance = self._edge_distance_to_box((page_x1, page_y1, page_x2, page_y2), unit_bbox)
            mapped.append(
                OCRRegion(
                    bbox_xyxy=[page_x1, page_y1, page_x2, page_y2],
                    text=text,
                    unit_bbox_xyxy=[unit_x1, unit_y1, unit_x2, unit_y2],
                    unit_kind=unit_kind,
                    unit_resize_scale=float(scale),
                    edge_distance=edge_distance,
                    normalized_text=self._normalize_text_key(text),
                )
            )

        return mapped

    def _request_response_text(self, image: np.ndarray) -> str:
        data_url = self._image_data_url(image)
        payload = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": self.PROMPT},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                }
            ],
            "temperature": self.temperature,
            "top_k": self.top_k,
            "max_completion_tokens": self.max_completion_tokens,
        }
        data = self._send_request(payload)
        response_text = self._extract_text_from_response(data)
        if self.raw_response_logging:
            logger.info("mangalmm_page_tile_ocr raw response text: %s", response_text)
        return response_text

    def _send_request(self, payload: dict) -> dict:
        try:
            response = requests.post(
                self._chat_completions_url(),
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=self.request_timeout_sec,
            )
        except requests.exceptions.RequestException as exc:
            raise LocalServiceConnectionError(
                f"Unable to reach the local {SERVICE_NAME} service.",
                service_name=SERVICE_NAME,
                settings_page_name=SETTINGS_PAGE_NAME,
            ) from exc

        if response.status_code != 200:
            raise LocalServiceResponseError(
                f"{SERVICE_NAME} service returned HTTP {response.status_code}.",
                service_name=SERVICE_NAME,
                settings_page_name=SETTINGS_PAGE_NAME,
            )

        try:
            data = response.json()
        except ValueError as exc:
            raise LocalServiceResponseError(
                f"{SERVICE_NAME} service returned invalid JSON.",
                service_name=SERVICE_NAME,
                settings_page_name=SETTINGS_PAGE_NAME,
            ) from exc

        if self.raw_response_logging:
            logger.info("mangalmm_page_tile_ocr raw response json: %s", data)
        return data

    def _extract_text_from_response(self, data: dict) -> str:
        error = data.get("error")
        if isinstance(error, dict):
            error_message = str(error.get("message", "") or f"Unknown {SERVICE_NAME} error.")
            raise LocalServiceResponseError(
                error_message,
                service_name=SERVICE_NAME,
                settings_page_name=SETTINGS_PAGE_NAME,
            )

        choices = data.get("choices")
        if not isinstance(choices, list) or not choices:
            raise LocalServiceResponseError(
                f"{SERVICE_NAME} response did not include choices.",
                service_name=SERVICE_NAME,
                settings_page_name=SETTINGS_PAGE_NAME,
            )

        message = choices[0].get("message")
        if not isinstance(message, dict):
            raise LocalServiceResponseError(
                f"{SERVICE_NAME} response did not include a message payload.",
                service_name=SERVICE_NAME,
                settings_page_name=SETTINGS_PAGE_NAME,
            )

        content = message.get("content", "")
        if isinstance(content, str):
            return content.strip()

        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if not isinstance(item, dict):
                    continue
                if item.get("type") == "text":
                    parts.append(str(item.get("text", "") or ""))
            return "\n".join(part for part in parts if part).strip()

        return str(content or "").strip()

    def _parse_region_payload(self, text: str) -> list[dict[str, object]]:
        return self._analyze_region_payload(text)["regions"]

    def _analyze_region_payload(self, text: str) -> dict[str, object]:
        raw = str(text or "").strip()
        if not raw:
            return {
                "regions": [],
                "response_kind": "empty",
                "payload_type": "empty",
                "raw_length": 0,
            }

        candidates: list[tuple[str, str]] = [("raw", raw)]
        unfenced = self._strip_code_fences(raw)
        if unfenced and unfenced != raw:
            candidates.append(("unfenced", unfenced))

        for source, candidate in candidates:
            decoded = self._extract_first_json_array(candidate)
            normalized = self._normalize_region_list(decoded)
            if normalized:
                response_kind = self._classify_array_response_kind(raw, source, candidate)
                return {
                    "regions": normalized,
                    "response_kind": response_kind,
                    "payload_type": "json_array",
                    "raw_length": len(raw),
                }
            if isinstance(decoded, list):
                response_kind = self._classify_array_response_kind(raw, source, candidate)
                return {
                    "regions": [],
                    "response_kind": f"{response_kind}_without_valid_regions",
                    "payload_type": "json_array",
                    "raw_length": len(raw),
                }

        for source, candidate in candidates:
            decoded = self._extract_first_json_payload(candidate)
            normalized, payload_type = self._extract_regions_from_payload(decoded)
            if normalized:
                response_kind = self._classify_payload_response_kind(source, payload_type)
                return {
                    "regions": normalized,
                    "response_kind": response_kind,
                    "payload_type": payload_type,
                    "raw_length": len(raw),
                }
            if decoded is not None:
                response_kind = self._classify_payload_response_kind(source, payload_type)
                return {
                    "regions": [],
                    "response_kind": f"{response_kind}_without_regions",
                    "payload_type": payload_type,
                    "raw_length": len(raw),
                }
        return {
            "regions": [],
            "response_kind": "plain_text_or_non_json",
            "payload_type": "text",
            "raw_length": len(raw),
        }

    def _extract_first_json_array(self, text: str) -> object:
        decoder = json.JSONDecoder()
        start = text.find("[")
        while start != -1:
            try:
                payload, _end = decoder.raw_decode(text[start:])
            except json.JSONDecodeError:
                start = text.find("[", start + 1)
                continue
            if isinstance(payload, list):
                return payload
            start = text.find("[", start + 1)
        return None

    def _extract_first_json_payload(self, text: str) -> object:
        decoder = json.JSONDecoder()
        starts = sorted(index for index in (text.find("{"), text.find("[")) if index != -1)
        checked: set[int] = set()
        for start in starts:
            cursor = start
            while cursor != -1 and cursor not in checked:
                checked.add(cursor)
                try:
                    payload, _end = decoder.raw_decode(text[cursor:])
                except json.JSONDecodeError:
                    next_obj = text.find("{", cursor + 1)
                    next_arr = text.find("[", cursor + 1)
                    next_candidates = [index for index in (next_obj, next_arr) if index != -1]
                    cursor = min(next_candidates) if next_candidates else -1
                    continue
                return payload
        return None

    def _extract_regions_from_payload(self, payload: object) -> tuple[list[dict[str, object]], str]:
        normalized = self._normalize_region_list(payload)
        if normalized:
            return normalized, "json_array"
        if isinstance(payload, dict):
            if {"bbox_2d", "text_content"}.issubset(payload.keys()):
                return self._normalize_region_list([payload]), "json_single_region_object"
            for key in ("regions", "items", "data", "result", "results", "ocr", "text_regions"):
                nested = payload.get(key)
                normalized = self._normalize_region_list(nested)
                if normalized:
                    return normalized, f"json_object_wrapper:{key}"
        return [], "json_object" if isinstance(payload, dict) else "unknown_payload"

    @staticmethod
    def _classify_array_response_kind(raw: str, source: str, candidate: str) -> str:
        raw_stripped = raw.lstrip()
        candidate_stripped = candidate.lstrip()
        if candidate_stripped.startswith("{"):
            return "json_object_wrapper" if source == "raw" else "salvaged_json_object_wrapper"
        if source == "unfenced" and raw_stripped.startswith("```"):
            return "fenced_json_array"
        if source == "raw" and candidate_stripped.startswith("["):
            return "json_array"
        return "wrapped_json_array"

    @staticmethod
    def _classify_payload_response_kind(source: str, payload_type: str) -> str:
        if payload_type.startswith("json_object_wrapper"):
            return "json_object_wrapper" if source == "raw" else "salvaged_json_object_wrapper"
        if payload_type == "json_single_region_object":
            return "json_single_region_object" if source == "raw" else "salvaged_json_single_region_object"
        if payload_type == "json_object":
            return "json_object" if source == "raw" else "salvaged_json_object"
        return payload_type if source == "raw" else f"salvaged_{payload_type}"

    def _normalize_region_list(self, payload: object) -> list[dict[str, object]]:
        if not isinstance(payload, list):
            return []

        normalized: list[dict[str, object]] = []
        for item in payload:
            if not isinstance(item, dict):
                continue
            bbox = item.get("bbox_2d")
            text = str(item.get("text_content", "") or "").strip()
            if not isinstance(bbox, list) or len(bbox) != 4 or not text:
                continue
            try:
                coords = [float(value) for value in bbox]
            except (TypeError, ValueError):
                continue
            normalized.append({"bbox_2d": coords, "text_content": text})
        return normalized

    @staticmethod
    def _strip_code_fences(text: str) -> str:
        stripped = str(text or "").strip()
        if not stripped.startswith("```"):
            return stripped
        stripped = re.sub(r"^\s*```(?:json)?\s*", "", stripped, count=1, flags=re.IGNORECASE)
        stripped = re.sub(r"\s*```\s*$", "", stripped, count=1)
        return stripped.strip()

    def _crop_image(self, image: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray | None:
        x1, y1, x2, y2 = [int(v) for v in bbox]
        crop = image[y1:y2, x1:x2]
        if crop is None or crop.size == 0:
            return None
        return ensure_three_channel(crop)

    def _image_data_url(self, image: np.ndarray) -> str:
        image_bytes = self._encode_image(image)
        image_b64 = base64.b64encode(image_bytes).decode("ascii")
        return f"data:image/jpeg;base64,{image_b64}"

    def _encode_image(self, image: np.ndarray) -> bytes:
        success, encoded = cv2.imencode(".jpg", image)
        if not success:
            raise LocalServiceResponseError(
                f"Failed to encode OCR image for {SERVICE_NAME}.",
                service_name=SERVICE_NAME,
                settings_page_name=SETTINGS_PAGE_NAME,
            )
        return encoded.tobytes()

    def _chat_completions_url(self) -> str:
        base = (self.server_url or DEFAULT_MANGALMM_SERVER_URL).rstrip("/")
        if base.endswith(self.REQUEST_ENDPOINT_SUFFIX):
            return base
        return f"{base}{self.REQUEST_ENDPOINT_SUFFIX}"

    def _mark_empty(
        self,
        blk: TextBlock,
        reason: str,
        *,
        attempt_count: int,
        status: str,
        raw_text: str = "",
    ) -> None:
        blk.texts = []
        blk.text = ""
        blk.ocr_regions = []
        blk.ocr_crop_bbox = None
        blk.ocr_resize_scale = 1.0
        set_block_ocr_diagnostics(
            blk,
            text="",
            confidence=0.0,
            status=status,
            empty_reason=reason,
            attempt_count=attempt_count,
            raw_text=raw_text,
            sanitized_text="",
        )

    def _ownership_box(self, blk: TextBlock) -> tuple[int, int, int, int] | None:
        bubble = getattr(blk, "bubble_xyxy", None)
        if bubble is not None:
            return self._normalize_box(bubble)
        return self._normalize_box(getattr(blk, "xyxy", None))

    def _precision_box(self, blk: TextBlock) -> tuple[int, int, int, int] | None:
        return self._normalize_box(getattr(blk, "xyxy", None))

    def _support_box(self, blk: TextBlock) -> tuple[int, int, int, int] | None:
        ownership = self._ownership_box(blk)
        precision = self._precision_box(blk)
        if ownership is None:
            return precision
        if precision is None:
            return ownership
        return (
            min(ownership[0], precision[0]),
            min(ownership[1], precision[1]),
            max(ownership[2], precision[2]),
            max(ownership[3], precision[3]),
        )

    def _build_axis_starts(self, length: int, tile_size: int, overlap: int) -> list[int]:
        if length <= tile_size:
            return [0]
        stride = max(1, tile_size - overlap)
        starts = list(range(0, max(1, length - tile_size + 1), stride))
        last_start = max(0, length - tile_size)
        if starts[-1] != last_start:
            starts.append(last_start)
        return sorted(set(int(v) for v in starts))

    def _expand_box_with_min_size(
        self,
        box: tuple[int, int, int, int],
        image_shape: tuple[int, ...],
        *,
        x_ratio: float,
        y_ratio: float,
        min_size: int,
    ) -> tuple[int, int, int, int]:
        x1, y1, x2, y2 = [float(v) for v in box]
        width = max(1.0, x2 - x1)
        height = max(1.0, y2 - y1)
        x_pad = width * x_ratio
        y_pad = height * y_ratio
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        half_w = max((width / 2.0) + x_pad, min_size / 2.0)
        half_h = max((height / 2.0) + y_pad, min_size / 2.0)
        return self._clip_box((cx - half_w, cy - half_h, cx + half_w, cy + half_h), image_shape)

    def _merge_boxes(
        self,
        boxes: list[tuple[int, int, int, int]],
        image_shape: tuple[int, ...],
        *,
        gap: int,
    ) -> list[tuple[int, int, int, int]]:
        merged: list[list[int]] = []
        for box in sorted(boxes, key=lambda current: (current[1], current[0], current[3], current[2])):
            current = list(box)
            for existing in merged:
                if self._boxes_touch_or_overlap(tuple(existing), tuple(current), gap=gap):
                    existing[0] = min(existing[0], current[0])
                    existing[1] = min(existing[1], current[1])
                    existing[2] = max(existing[2], current[2])
                    existing[3] = max(existing[3], current[3])
                    break
            else:
                merged.append(current)
        return [self._clip_box(tuple(box), image_shape) for box in merged]

    @staticmethod
    def _boxes_touch_or_overlap(
        a: tuple[int, int, int, int],
        b: tuple[int, int, int, int],
        *,
        gap: int,
    ) -> bool:
        return not (
            a[2] + gap < b[0]
            or b[2] + gap < a[0]
            or a[3] + gap < b[1]
            or b[3] + gap < a[1]
        )

    @staticmethod
    def _normalize_box(box) -> tuple[int, int, int, int] | None:
        if box is None:
            return None
        if len(box) != 4:
            return None
        x1, y1, x2, y2 = [int(round(float(v))) for v in box]
        if x2 <= x1 or y2 <= y1:
            return None
        return x1, y1, x2, y2

    @staticmethod
    def _clip_box(box, image_shape: tuple[int, ...]) -> tuple[int, int, int, int]:
        image_h, image_w = image_shape[:2]
        x1, y1, x2, y2 = [float(v) for v in box]
        clipped = (
            max(0, min(int(math.floor(x1)), image_w)),
            max(0, min(int(math.floor(y1)), image_h)),
            max(0, min(int(math.ceil(x2)), image_w)),
            max(0, min(int(math.ceil(y2)), image_h)),
        )
        if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
            return (0, 0, 0, 0)
        return clipped

    @staticmethod
    def _bbox_area(box: tuple[int, int, int, int]) -> int:
        width = max(0, int(box[2]) - int(box[0]))
        height = max(0, int(box[3]) - int(box[1]))
        return width * height

    @staticmethod
    def _intersection_area(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> int:
        x1 = max(int(a[0]), int(b[0]))
        y1 = max(int(a[1]), int(b[1]))
        x2 = min(int(a[2]), int(b[2]))
        y2 = min(int(a[3]), int(b[3]))
        if x2 <= x1 or y2 <= y1:
            return 0
        return (x2 - x1) * (y2 - y1)

    def _bbox_iou(self, a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
        inter = self._intersection_area(a, b)
        if inter <= 0:
            return 0.0
        union = self._bbox_area(a) + self._bbox_area(b) - inter
        return inter / float(max(union, 1))

    @staticmethod
    def _bbox_center(box: tuple[int, int, int, int]) -> tuple[float, float]:
        return ((float(box[0]) + float(box[2])) / 2.0, (float(box[1]) + float(box[3])) / 2.0)

    @staticmethod
    def _point_in_box(point: tuple[float, float], box: tuple[int, int, int, int]) -> bool:
        return box[0] <= point[0] <= box[2] and box[1] <= point[1] <= box[3]

    @staticmethod
    def _bbox_diagonal(box: tuple[int, int, int, int]) -> float:
        return math.hypot(float(box[2]) - float(box[0]), float(box[3]) - float(box[1]))

    @staticmethod
    def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
        return math.hypot(a[0] - b[0], a[1] - b[1])

    @staticmethod
    def _edge_distance_to_box(inner: tuple[int, int, int, int], outer: tuple[int, int, int, int]) -> float:
        left = max(0.0, float(inner[0]) - float(outer[0]))
        top = max(0.0, float(inner[1]) - float(outer[1]))
        right = max(0.0, float(outer[2]) - float(inner[2]))
        bottom = max(0.0, float(outer[3]) - float(inner[3]))
        return min(left, top, right, bottom)

    @staticmethod
    def _normalize_text_key(text: str) -> str:
        return re.sub(r"\s+", "", str(text or "")).strip()

    @staticmethod
    def _clamp_int(value, default: int, bounds: tuple[int, int]) -> int:
        low, high = bounds
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            parsed = default
        return max(low, min(parsed, high))

    @staticmethod
    def _read_env_int(
        env_name: str,
        config_value,
        default: int,
        *,
        bounds: tuple[int, int],
    ) -> int:
        raw = os.getenv(env_name, "").strip()
        candidate = raw if raw else config_value
        try:
            parsed = int(candidate)
        except (TypeError, ValueError):
            parsed = default
        low, high = bounds
        return max(low, min(parsed, high))

    @staticmethod
    def _read_env_float(
        env_name: str,
        config_value,
        default: float,
        *,
        minimum: float,
        maximum: float,
    ) -> float:
        raw = os.getenv(env_name, "").strip()
        candidate = raw if raw else config_value
        try:
            parsed = float(candidate)
        except (TypeError, ValueError):
            parsed = default
        return max(minimum, min(parsed, maximum))

    def _export_debug_artifact(
        self,
        *,
        blk: TextBlock | None,
        failure_reason: str,
        response_kind: str,
        raw_text: str,
        crop_bbox,
        crop_source: str,
        resize_scale: float,
        crop_image: np.ndarray | None,
        request_image: np.ndarray | None,
        analysis: dict[str, object],
        unit_bbox=None,
        unit_kind: str = "",
    ) -> None:
        if self.debug_root is None:
            return
        with self._debug_export_lock:
            export_index = next(self._debug_export_counter)
        if export_index > self.debug_export_limit:
            return

        artifact_dir = self.debug_root / f"{export_index:04d}_{response_kind}"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        metadata = {
            "failure_reason": failure_reason,
            "response_kind": response_kind,
            "raw_text": str(raw_text or ""),
            "crop_bbox": self._coords_or_none(crop_bbox),
            "crop_source": crop_source,
            "resize_scale": float(resize_scale),
            "crop_shape": self._image_shape_or_none(crop_image),
            "request_shape": self._image_shape_or_none(request_image),
            "block_xyxy": self._coords_or_none(getattr(blk, "xyxy", None)) if blk is not None else None,
            "bubble_xyxy": self._coords_or_none(getattr(blk, "bubble_xyxy", None)) if blk is not None else None,
            "source_lang": str(getattr(blk, "source_lang", "") or "") if blk is not None else "",
            "direction": str(getattr(blk, "direction", "") or "") if blk is not None else "",
            "unit_bbox": self._coords_or_none(unit_bbox),
            "unit_kind": unit_kind,
            "analysis": analysis,
        }
        (artifact_dir / "meta.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        if crop_image is not None and crop_image.size > 0:
            cv2.imwrite(str(artifact_dir / "crop.jpg"), crop_image)
        if request_image is not None and request_image.size > 0:
            cv2.imwrite(str(artifact_dir / "request.jpg"), request_image)

    @staticmethod
    def _coords_or_none(value) -> list[int] | None:
        if value is None:
            return None
        try:
            coords = [int(float(item)) for item in value]
        except Exception:
            return None
        return coords if len(coords) == 4 else None

    @staticmethod
    def _image_shape_or_none(image: np.ndarray | None) -> list[int] | None:
        if image is None:
            return None
        try:
            height, width = image.shape[:2]
        except Exception:
            return None
        return [int(height), int(width)]

from __future__ import annotations

from dataclasses import replace
from dataclasses import dataclass
from pathlib import Path
import time
from typing import Iterable

import cv2
import numpy as np

from modules.masking.ctd_refiner import (
    CTDRefiner,
    CTDRefinerSettings,
    _refine_mask,
)
from modules.utils.textblock import TextBlock

from .contracts import CandidateMaskResult, binary_mask
from .contracts import DetectorBox
from .reference_probe import load_ballons_ctd_runtime_reference


@dataclass(frozen=True, slots=True)
class BallonsCTDNativeDetection:
    raw_mask: np.ndarray
    refined_mask: np.ndarray
    blocks: tuple[object, ...]


class _PinnedCTDRefiner(CTDRefiner):
    def __init__(
        self,
        settings: CTDRefinerSettings,
        model_path: str | Path,
    ) -> None:
        super().__init__(settings)
        self._pinned_model_path = str(Path(model_path).resolve())

    def _choose_model_path(self) -> str:
        return self._pinned_model_path


class BallonsCTDFullPageReference:
    """Full-page CTD pixel-claim adapter built from the vendored Ballons port.

    The raw network mask and the 3 px Ballons dilation are exact pixel-claim
    candidates. Native CTD block grouping is intentionally not guessed here.
    A refined mask is emitted only when explicit detector blocks are supplied;
    that variant is recorded as a hybrid rather than native CTD parity.
    """

    def __init__(
        self,
        settings: CTDRefinerSettings | None = None,
        *,
        dilate_size: int = 3,
        model_path: str | Path | None = None,
    ) -> None:
        base = settings or CTDRefinerSettings()
        self.settings = replace(base, mask_dilate_size=0)
        self.dilate_size = max(0, int(dilate_size))
        self.refiner = (
            _PinnedCTDRefiner(self.settings, model_path)
            if model_path is not None
            else CTDRefiner(self.settings)
        )

    def infer(
        self,
        image_rgb: np.ndarray,
        *,
        ownership_blocks: Iterable[TextBlock] | None = None,
    ) -> CandidateMaskResult:
        start = time.perf_counter()
        raw = binary_mask(self.refiner._infer_raw_mask(image_rgb))
        block_list = list(ownership_blocks or [])
        if block_list:
            refined = binary_mask(_refine_mask(image_rgb, raw, block_list), raw.shape)
            refined_kind = "current_ownership_hybrid"
        else:
            refined = raw.copy()
            refined_kind = "unavailable_without_native_ctd_blocks"

        if self.dilate_size > 0:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (2 * self.dilate_size + 1, 2 * self.dilate_size + 1),
                (self.dilate_size, self.dilate_size),
            )
            dilated = binary_mask(cv2.dilate(raw, kernel), raw.shape)
        else:
            dilated = raw.copy()

        return CandidateMaskResult(
            candidate_id="ballons_ctd_fullpage",
            raw_mask=raw,
            refined_mask=refined,
            dilated_mask=dilated,
            runtime={
                "seconds": time.perf_counter() - start,
                "backend": self.refiner.backend,
                "device": self.settings.device,
                "detect_size": int(self.settings.detect_size),
                "refined_kind": refined_kind,
                "reference": "BallonsTranslator CTD pixel mask",
            },
        )

    def infer_with_ownership_rois(
        self,
        image_rgb: np.ndarray,
        ownership_mask: np.ndarray,
    ) -> CandidateMaskResult:
        """Fuse full-page and ownership-ROI CTD claims without bbox filling.

        Each connected authoritative ownership region is only detector context.
        The returned claim is clipped to the exact sparse ownership mask, so an
        OCR/text rectangle never becomes an edit mask by itself.
        """

        start = time.perf_counter()
        ownership = binary_mask(ownership_mask, image_rgb.shape[:2])
        if not np.any(ownership):
            empty = np.zeros(image_rgb.shape[:2], dtype=np.uint8)
            return CandidateMaskResult(
                candidate_id="ballons_ctd_fullpage_plus_ownership_roi",
                raw_mask=empty,
                refined_mask=empty,
                dilated_mask=empty,
                runtime={
                    "seconds": time.perf_counter() - start,
                    "backend": "not_run",
                    "device": self.settings.device,
                    "detect_size": int(self.settings.detect_size),
                    "roi_inference_call_count": 0,
                    "full_page_inference_call_count": 0,
                    "reference": (
                        "CTD full-page plus authoritative ownership ROIs"
                    ),
                },
            )
        full = binary_mask(self.refiner._infer_raw_mask(image_rgb))
        roi_claim = np.zeros_like(full)
        count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(
            (ownership > 0).astype(np.uint8),
            connectivity=8,
        )
        roi_calls = 0
        for index in range(1, count):
            x = int(stats[index, cv2.CC_STAT_LEFT])
            y = int(stats[index, cv2.CC_STAT_TOP])
            width = int(stats[index, cv2.CC_STAT_WIDTH])
            height = int(stats[index, cv2.CC_STAT_HEIGHT])
            if width <= 5 or height <= 5:
                continue
            padding = max(8, min(64, round(max(width, height) * 0.08)))
            x1, y1 = max(0, x - padding), max(0, y - padding)
            x2 = min(image_rgb.shape[1], x + width + padding)
            y2 = min(image_rgb.shape[0], y + height + padding)
            crop = np.ascontiguousarray(image_rgb[y1:y2, x1:x2])
            if crop.size == 0:
                continue
            local = binary_mask(self.refiner._infer_raw_mask(crop))
            owned_local = ownership[y1:y2, x1:x2] > 0
            roi_claim[y1:y2, x1:x2][owned_local & (local > 0)] = 255
            roi_calls += 1

        raw = np.where(
            ((full > 0) | (roi_claim > 0)) & (ownership > 0),
            255,
            0,
        ).astype(np.uint8)
        if self.dilate_size > 0:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (2 * self.dilate_size + 1, 2 * self.dilate_size + 1),
                (self.dilate_size, self.dilate_size),
            )
            dilated = cv2.dilate(raw, kernel)
            dilated[ownership == 0] = 0
        else:
            dilated = raw.copy()
        return CandidateMaskResult(
            candidate_id="ballons_ctd_fullpage_plus_ownership_roi",
            raw_mask=raw,
            refined_mask=raw,
            dilated_mask=dilated,
            runtime={
                "seconds": time.perf_counter() - start,
                "backend": self.refiner.backend,
                "device": self.settings.device,
                "detect_size": int(self.settings.detect_size),
                "roi_inference_call_count": roi_calls,
                "full_page_inference_call_count": 1,
                "reference": "CTD full-page plus authoritative ownership ROIs",
            },
        )


class BallonsCTDOriginalReference:
    """Execute Ballons' original Python CTD runtime for golden and Stage 1."""

    def __init__(
        self,
        *,
        ballons_root: str,
        model_path: str,
        device: str = "cpu",
        detect_size: int = 1280,
        dilate_size: int = 3,
    ) -> None:
        module = load_ballons_ctd_runtime_reference(Path(ballons_root))
        self.module = module
        self.detector = module.TextDetector(
            model_path=model_path,
            detect_size=int(detect_size),
            device=device,
            half=False,
            det_rearrange_max_batches=4,
        )
        self.device = device
        self.detect_size = int(detect_size)
        self.dilate_size = max(0, int(dilate_size))

    def detect_native(self, image_rgb: np.ndarray) -> BallonsCTDNativeDetection:
        raw, refined, blocks = self.detector(
            image_rgb,
            refine_mode=self.module.REFINEMASK_INPAINT,
            keep_undetected_mask=False,
        )
        raw = binary_mask(raw)
        refined = binary_mask(refined, raw.shape)
        return BallonsCTDNativeDetection(raw, refined, tuple(blocks))

    def infer(self, image_rgb: np.ndarray) -> CandidateMaskResult:
        start = time.perf_counter()
        detection = self.detect_native(image_rgb)
        raw = detection.raw_mask
        refined = detection.refined_mask
        if self.dilate_size > 0:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (2 * self.dilate_size + 1, 2 * self.dilate_size + 1),
                (self.dilate_size, self.dilate_size),
            )
            dilated = binary_mask(cv2.dilate(raw, kernel), raw.shape)
        else:
            dilated = raw.copy()
        records: list[DetectorBox] = []
        for block in detection.blocks:
            xyxy = getattr(block, "xyxy", None)
            if xyxy is None or len(xyxy) < 4:
                continue
            records.append(
                DetectorBox(
                    tuple(map(int, xyxy[:4])),
                    "text",
                    1.0,
                    "ballons_ctd_original",
                )
            )
        return CandidateMaskResult(
            candidate_id="ballons_ctd_original",
            raw_mask=raw,
            refined_mask=refined,
            dilated_mask=dilated,
            boxes=tuple(records),
            runtime={
                "seconds": time.perf_counter() - start,
                "backend": self.detector.backend,
                "device": self.device,
                "detect_size": self.detect_size,
                "reference": "BallonsTranslator original CTD Python runtime",
            },
        )

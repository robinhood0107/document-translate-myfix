from __future__ import annotations

from dataclasses import replace
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
    ) -> None:
        base = settings or CTDRefinerSettings()
        self.settings = replace(base, mask_dilate_size=0)
        self.dilate_size = max(0, int(dilate_size))
        self.refiner = CTDRefiner(self.settings)

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

    def infer(self, image_rgb: np.ndarray) -> CandidateMaskResult:
        start = time.perf_counter()
        raw, refined, blocks = self.detector(
            image_rgb,
            refine_mode=self.module.REFINEMASK_INPAINT,
            keep_undetected_mask=False,
        )
        raw = binary_mask(raw)
        refined = binary_mask(refined, raw.shape)
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
        for block in blocks:
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

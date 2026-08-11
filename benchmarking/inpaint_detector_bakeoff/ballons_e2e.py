from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time

import cv2
import numpy as np

from modules.inpainting.source_lama_blockwise import SourceLaMaLarge
from modules.source_parity_vendor.utils.imgproc_utils import enlarge_window
from modules.source_parity_vendor.utils.textblock_mask import extract_ballon_mask

from .ballons_ctd import BallonsCTDOriginalReference
from .contracts import binary_mask


@dataclass(frozen=True, slots=True)
class BallonsEndToEndResult:
    image_bgr: np.ndarray
    detector_mask: np.ndarray
    raw_mask: np.ndarray
    runtime: dict[str, object]


class BallonsEndToEndReference:
    """Run Ballons' native CTD and successful-path blockwise inpaint routing.

    BallonsTranslator and this project use the same LaMa Large architecture and
    checkpoint. Native detector blocks come from the pinned Ballons Python
    reference. The wrapper below retains Ballons' sequential crop and whole-
    bubble flat-fill semantics, while the pixel-exact local LaMa core supplies
    CUDA diagnostics. OOM behavior remains a separately recorded local runtime
    contract and is not presented as Ballons parity.
    """

    def __init__(
        self,
        *,
        ballons_root: str | Path,
        detector_model_path: str | Path,
        device: str = "cuda",
        precision: str = "bf16",
        detect_size: int = 1280,
        inpaint_size: int = 1536,
    ) -> None:
        self.detector = BallonsCTDOriginalReference(
            ballons_root=str(ballons_root),
            model_path=str(detector_model_path),
            device=device,
            detect_size=detect_size,
            dilate_size=3,
        )
        self.inpainter = SourceLaMaLarge(
            device=device,
            precision=precision,
            inpaint_size=inpaint_size,
        )
        self.device = str(device)
        self.precision = str(precision)
        self.detect_size = int(detect_size)
        self.inpaint_size = int(inpaint_size)

    def _inpaint_with_ballons_routing(
        self,
        image_rgb: np.ndarray,
        mask: np.ndarray,
        blocks: tuple[object, ...],
    ) -> np.ndarray:
        """Mirror Ballons ``InpainterBase.inpaint`` default block behavior."""

        image_height, image_width = image_rgb.shape[:2]
        inpainted = np.asarray(image_rgb).copy()
        work_mask = np.asarray(mask).copy()
        for block in blocks:
            xyxy = [int(value) for value in getattr(block, "xyxy")]
            expanded = enlarge_window(xyxy, image_width, image_height, ratio=1.7)
            image_crop = inpainted[
                expanded[1] : expanded[3], expanded[0] : expanded[2]
            ]
            mask_crop = work_mask[
                expanded[1] : expanded[3], expanded[0] : expanded[2]
            ]
            need_inpaint = True
            balloon_mask, non_text_mask = extract_ballon_mask(image_crop, mask_crop)
            if balloon_mask is not None:
                non_text_pixels = image_crop[np.where(non_text_mask > 0)]
                average_background = np.median(non_text_pixels, axis=0)
                rgb_std = np.std(non_text_pixels - average_background, axis=0)
                threshold = 7 if np.std(rgb_std) > 1 else 10
                if np.max(rgb_std) < threshold:
                    need_inpaint = False
                    image_crop[np.where(balloon_mask > 0)] = average_background
            if need_inpaint:
                inpainted[
                    expanded[1] : expanded[3], expanded[0] : expanded[2]
                ] = self.inpainter.memory_safe_inpaint(image_crop, mask_crop)
            work_mask[xyxy[1] : xyxy[3], xyxy[0] : xyxy[2]] = 0
        return np.ascontiguousarray(inpainted)

    def infer(self, image_bgr: np.ndarray) -> BallonsEndToEndResult:
        image = np.ascontiguousarray(np.asarray(image_bgr))
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        detect_started = time.perf_counter()
        detection = self.detector.detect_native(image_rgb)
        detect_seconds = time.perf_counter() - detect_started

        diagnostics_before = len(self.inpainter.run_diagnostics)
        inpaint_started = time.perf_counter()
        self.inpainter.ensure_loaded()
        candidate_rgb = self._inpaint_with_ballons_routing(
            image_rgb,
            detection.refined_mask,
            detection.blocks,
        )
        inpaint_seconds = time.perf_counter() - inpaint_started
        diagnostics = self.inpainter.run_diagnostics[diagnostics_before:]
        return BallonsEndToEndResult(
            image_bgr=np.ascontiguousarray(
                cv2.cvtColor(candidate_rgb, cv2.COLOR_RGB2BGR)
            ),
            detector_mask=binary_mask(detection.refined_mask, image.shape[:2]),
            raw_mask=binary_mask(detection.raw_mask, image.shape[:2]),
            runtime={
                "detector_seconds": detect_seconds,
                "inpaint_seconds": inpaint_seconds,
                "total_seconds": detect_seconds + inpaint_seconds,
                "block_count": len(detection.blocks),
                "inference_call_count": len(diagnostics),
                "cpu_fallback_count": sum(
                    int(bool(row.get("cpu_fallback_used", False)))
                    for row in diagnostics
                ),
                "device": self.device,
                "precision": self.precision,
                "detect_size": self.detect_size,
                "inpaint_size": self.inpaint_size,
                "diagnostics": diagnostics,
            },
        )

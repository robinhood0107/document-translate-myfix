from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
from PySide6.QtCore import QCoreApplication


LEARNED_INPAINTER_KEYS = frozenset(
    {
        "AOT",
        "MI-GAN",
        "lama_large_512px",
        "lama_mpe",
    }
)

INPAINT_RUNTIME_CONTRACT_VERSION = "cuda-learned-inpaint-v1"
INPAINT_RETRY_POLICY_VERSION = "single-tighter-roi-v1"


class InpaintingRuntimeContractError(RuntimeError):
    """Raised when learned inpainting would violate the product contract."""


class InpaintingCudaOOMError(InpaintingRuntimeContractError):
    """Raised when the single bounded CUDA retry cannot complete."""

    def __init__(
        self,
        message: str,
        *,
        diagnostics: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.diagnostics = dict(diagnostics or {})


def learned_inpaint_cuda_required_message() -> str:
    return QCoreApplication.translate(
        "InpaintingRuntime",
        "Learned inpainting requires CUDA; CPU fallback is disabled.",
    )


def inpaint_cuda_oom_message() -> str:
    return QCoreApplication.translate(
        "InpaintingRuntime",
        "CUDA inpainting ran out of memory after the single bounded ROI retry.",
    )


def inpaint_outside_mask_message() -> str:
    return QCoreApplication.translate(
        "InpaintingRuntime",
        "Inpaint output changed pixels outside the final edit mask.",
    )


def inpaint_release_unconfirmed_message() -> str:
    return QCoreApplication.translate(
        "InpaintingRuntime",
        "The previous inpainter VRAM release was not confirmed.",
    )


def inpaint_cuda_fft_failed_message() -> str:
    return QCoreApplication.translate(
        "InpaintingRuntime",
        "CUDA FFT execution failed; CPU fallback is disabled.",
    )


@dataclass(frozen=True)
class BoundedRetryROI:
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def area(self) -> int:
        return max(0, self.x2 - self.x1) * max(0, self.y2 - self.y1)

    def as_list(self) -> list[int]:
        return [self.x1, self.y1, self.x2, self.y2]


def is_cuda_device(value: Any) -> bool:
    return str(value or "").strip().lower().startswith("cuda")


def is_cuda_oom_error(exc: BaseException) -> bool:
    try:
        import torch

        if isinstance(exc, torch.cuda.OutOfMemoryError):
            return True
    except (ImportError, AttributeError):
        pass
    message = str(exc or "").lower()
    return "cuda" in message and "out of memory" in message


def validate_learned_inpaint_runtime(
    *,
    inpainter_key: str,
    device: str,
    precision: str,
) -> dict[str, Any]:
    key = str(inpainter_key or "")
    normalized_device = str(device or "").strip().lower()
    normalized_precision = str(precision or "fp32").strip().lower()
    if key in LEARNED_INPAINTER_KEYS and not is_cuda_device(normalized_device):
        raise InpaintingRuntimeContractError(
            learned_inpaint_cuda_required_message()
        )
    return {
        "contract_version": INPAINT_RUNTIME_CONTRACT_VERSION,
        "inpainter_key": key,
        "actual_device": normalized_device,
        "actual_precision": normalized_precision,
        "cpu_fallback_used": False,
        "fp32_promotion_eligible": (
            key in LEARNED_INPAINTER_KEYS
            and is_cuda_device(normalized_device)
            and normalized_precision == "fp32"
        ),
    }


def inspect_learned_inpainter_runtime(
    inpainter: Any,
    *,
    inpainter_key: str,
    requested_device: str,
    requested_precision: str,
) -> dict[str, Any]:
    report = validate_learned_inpaint_runtime(
        inpainter_key=inpainter_key,
        device=str(
            getattr(
                inpainter,
                "runtime_device",
                getattr(inpainter, "device", requested_device),
            )
            or requested_device
        ),
        precision=str(
            getattr(inpainter, "precision", requested_precision)
            or requested_precision
        ),
    )
    model = getattr(inpainter, "model", None)
    parameter_device = ""
    parameter_dtype = ""
    model_candidates: list[Any] = []
    if model is not None:
        model_candidates.append(model)
        for attribute in ("generator", "mpe", "model"):
            nested = getattr(model, attribute, None)
            if nested is not None and nested is not model:
                model_candidates.append(nested)
    for candidate in model_candidates:
        try:
            parameter = next(iter(candidate.parameters()))
        except (AttributeError, StopIteration, TypeError):
            parameter = None
        if parameter is None:
            try:
                parameter = next(iter(candidate.buffers()))
            except (AttributeError, StopIteration, TypeError):
                parameter = None
        if parameter is not None:
            parameter_device = str(getattr(parameter, "device", "") or "")
            parameter_dtype = str(getattr(parameter, "dtype", "") or "")
            break
    session = getattr(inpainter, "session", None)
    session_providers: list[str] = []
    if session is not None:
        try:
            session_providers = [
                str(provider)
                for provider in list(session.get_providers() or [])
            ]
        except (AttributeError, TypeError):
            session_providers = []
    report["model_parameter_device"] = parameter_device
    report["model_parameter_dtype"] = parameter_dtype
    report["session_providers"] = session_providers
    gpu_session_verified = any(
        provider in {
            "CUDAExecutionProvider",
            "TensorrtExecutionProvider",
        }
        for provider in session_providers
    )
    report["device_verified_from_model"] = bool(
        parameter_device or gpu_session_verified
    )
    if parameter_device and not is_cuda_device(parameter_device):
        raise InpaintingRuntimeContractError(
            learned_inpaint_cuda_required_message()
        )
    if (
        inpainter_key in LEARNED_INPAINTER_KEYS
        and session_providers
        and not gpu_session_verified
    ):
        raise InpaintingRuntimeContractError(
            learned_inpaint_cuda_required_message()
        )
    if (
        inpainter_key in LEARNED_INPAINTER_KEYS
        and not report["device_verified_from_model"]
    ):
        raise InpaintingRuntimeContractError(
            learned_inpaint_cuda_required_message()
        )
    return report


def mask_bbox(mask: np.ndarray | None) -> list[int] | None:
    if mask is None:
        return None
    binary = np.where(np.asarray(mask) > 0, 255, 0).astype(np.uint8)
    if binary.ndim == 3:
        binary = binary[:, :, 0]
    points = cv2.findNonZero(binary)
    if points is None:
        return None
    x, y, width, height = cv2.boundingRect(points)
    if width <= 0 or height <= 0:
        return None
    return [int(x), int(y), int(x + width), int(y + height)]


def bounded_retry_roi(
    mask: np.ndarray | None,
    image_shape: tuple[int, ...],
    *,
    minimum_context_px: int = 24,
    context_ratio: float = 0.35,
) -> BoundedRetryROI | None:
    bbox = mask_bbox(mask)
    if bbox is None:
        return None
    image_height, image_width = image_shape[:2]
    x1, y1, x2, y2 = bbox
    width = max(1, x2 - x1)
    height = max(1, y2 - y1)
    context = max(
        int(minimum_context_px),
        int(round(max(width, height) * float(context_ratio))),
    )
    retry = BoundedRetryROI(
        x1=max(0, x1 - context),
        y1=max(0, y1 - context),
        x2=min(int(image_width), x2 + context),
        y2=min(int(image_height), y2 + context),
    )
    full_area = max(1, int(image_height) * int(image_width))
    if retry.area <= 0 or retry.area >= full_area:
        return None
    return retry


def runtime_mask_diagnostics(
    mask: np.ndarray | None,
    image_shape: tuple[int, ...],
) -> dict[str, Any]:
    normalized = np.zeros(image_shape[:2], dtype=np.uint8)
    if mask is not None:
        source = np.asarray(mask)
        if source.ndim == 3:
            source = source[:, :, 0]
        height = min(normalized.shape[0], source.shape[0])
        width = min(normalized.shape[1], source.shape[1])
        normalized[:height, :width] = np.where(
            source[:height, :width] > 0,
            255,
            0,
        ).astype(np.uint8)
    return {
        "mask_bbox": mask_bbox(normalized),
        "mask_pixel_count": int(np.count_nonzero(normalized)),
    }

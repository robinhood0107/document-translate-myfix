from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
from typing import Any

import cv2
import numpy as np

from modules.masking.ctd_refiner import (
    _det_rearrange_forward,
    _postprocess_mask,
    _preprocess_img_rgb,
)
from modules.utils.download import ModelDownloader, ModelID
from modules.utils.onnx import make_session


@dataclass(frozen=True, slots=True)
class CTDPositiveClaimResult:
    raw_mask: np.ndarray
    providers: tuple[str, ...]
    detect_size: int
    model_sha256: str
    model_opset: int


_SESSION_CACHE: dict[tuple[str, tuple[str, ...], int], Any] = {}
_SESSION_CACHE_LOCK = Lock()


def _provider_names(providers) -> tuple[str, ...]:
    names: list[str] = []
    for provider in providers:
        names.append(str(provider[0] if isinstance(provider, tuple) else provider))
    return tuple(names)


def _required_providers(device: str) -> list[str]:
    if str(device or "").lower() == "cpu":
        return ["CPUExecutionProvider"]
    return ["CUDAExecutionProvider", "CPUExecutionProvider"]


def _cached_session(model_path: str, providers: list[str], detect_size: int):
    key = (str(model_path), tuple(providers), int(detect_size))
    with _SESSION_CACHE_LOCK:
        session = _SESSION_CACHE.get(key)
        if session is None:
            session = make_session(model_path, providers=providers)
            _SESSION_CACHE[key] = session
    actual = tuple(session.get_providers())
    if not actual or actual[0] != providers[0]:
        raise RuntimeError(
            "positive_claim_primary_provider_not_honored:"
            f"requested={providers[0]}:actual={list(actual)}"
        )
    return session


class CTDPositiveClaimProvider:
    """Fixed-1280 CTD raw-mask provider used only as positive evidence."""

    MODEL_OPSET = 12

    def __init__(
        self,
        *,
        device: str = "cuda",
        detect_size: int = 1280,
        max_batch_size: int = 4,
    ) -> None:
        self.detect_size = int(detect_size)
        self.max_batch_size = max(1, int(max_batch_size))
        self.requested_providers = _required_providers(device)
        model_spec = ModelDownloader.registry[ModelID.CTD_POSITIVE_CLAIM_ONNX]
        self.model_sha256 = str(model_spec.sha256[0] or "")
        self.model_path = ModelDownloader.primary_path(
            ModelID.CTD_POSITIVE_CLAIM_ONNX
        )
        self.session = _cached_session(
            self.model_path,
            self.requested_providers,
            self.detect_size,
        )
        self.providers = tuple(self.session.get_providers())
        model_input = self.session.get_inputs()[0]
        self.input_name = str(model_input.name)
        if list(model_input.shape[-2:]) != [self.detect_size, self.detect_size]:
            raise ValueError("positive_claim_model_input_size_mismatch")
        output_names = {str(item.name) for item in self.session.get_outputs()}
        if not {"det", "seg"}.issubset(output_names):
            raise ValueError("positive_claim_model_outputs_invalid")

    @staticmethod
    def _rgb_batch_to_nchw(batch: np.ndarray) -> np.ndarray:
        array = np.asarray(batch, dtype=np.float32) / 255.0
        if array.ndim != 4 or array.shape[-1] != 3:
            raise ValueError("positive_claim_batch_invalid")
        return np.ascontiguousarray(np.transpose(array, (0, 3, 1, 2)))

    def _run_one(self, nchw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        outputs = self.session.run(
            ["det", "seg"],
            {self.input_name: np.ascontiguousarray(nchw, dtype=np.float32)},
        )
        return np.asarray(outputs[0]), np.asarray(outputs[1])

    def _batch_forward(self, batch: np.ndarray, _device: str):
        nchw = self._rgb_batch_to_nchw(batch)
        lines: list[np.ndarray] = []
        masks: list[np.ndarray] = []
        for index in range(nchw.shape[0]):
            line, mask = self._run_one(nchw[index:index + 1])
            lines.append(line)
            masks.append(mask)
        return np.concatenate(lines, axis=0), np.concatenate(masks, axis=0)

    def _letterboxed_input(self, image_rgb: np.ndarray):
        image_bgr, _ratio, dw, dh = _preprocess_img_rgb(
            image_rgb,
            detect_size=(self.detect_size, self.detect_size),
            device="cpu",
            half=False,
            to_tensor=False,
        )
        image_rgb_padded = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        return self._rgb_batch_to_nchw(image_rgb_padded[None, ...]), dw, dh

    def infer(self, image_rgb: np.ndarray) -> CTDPositiveClaimResult:
        lines_map, mask = _det_rearrange_forward(
            image_rgb,
            self._batch_forward,
            tgt_size=self.detect_size,
            max_batch_size=self.max_batch_size,
            device="cpu",
        )
        if lines_map is None:
            batch, dw, dh = self._letterboxed_input(image_rgb)
            _lines, mask = self._run_one(batch)
            mask = np.asarray(mask).squeeze()
            mask = mask[..., : mask.shape[0] - dh, : mask.shape[1] - dw]
        raw = _postprocess_mask(mask)
        raw = cv2.resize(
            raw,
            (image_rgb.shape[1], image_rgb.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )
        raw = np.where(raw > 0, 255, 0).astype(np.uint8)
        return CTDPositiveClaimResult(
            raw_mask=np.ascontiguousarray(raw),
            providers=self.providers,
            detect_size=self.detect_size,
            model_sha256=self.model_sha256,
            model_opset=self.MODEL_OPSET,
        )

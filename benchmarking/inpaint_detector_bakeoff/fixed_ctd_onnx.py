from __future__ import annotations

from pathlib import Path
import time

import cv2
import numpy as np
import onnxruntime as ort

from modules.masking.ctd_refiner import (
    _det_rearrange_forward,
    _postprocess_mask,
    _preprocess_img_rgb,
)

from .contracts import CandidateMaskResult, binary_mask


def _require_primary_provider(requested: str, actual: tuple[str, ...]) -> None:
    if not actual or actual[0] != requested:
        raise RuntimeError(
            "CTD ONNX primary provider was not honored: "
            f"requested={requested!r}, actual={list(actual)!r}"
        )


def _rgb_batch_to_nchw(batch: np.ndarray) -> np.ndarray:
    array = np.asarray(batch, dtype=np.float32) / 255.0
    if array.ndim != 4 or array.shape[-1] != 3:
        raise ValueError("CTD ONNX batch must be NHWC RGB")
    return np.ascontiguousarray(np.transpose(array, (0, 3, 1, 2)))


def _letterboxed_rgb_to_nchw(image_rgb: np.ndarray, detect_size: int):
    image_bgr, ratio, dw, dh = _preprocess_img_rgb(
        image_rgb,
        detect_size=(int(detect_size), int(detect_size)),
        device="cpu",
        half=False,
        to_tensor=False,
    )
    image_rgb_padded = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    batch = _rgb_batch_to_nchw(image_rgb_padded[None, ...])
    return batch, ratio, dw, dh


class FixedSizeCTDONNXReference:
    """Run a fixed-size CTD export with the Python reference pre/postprocess."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        providers: list[str],
        detect_size: int = 1280,
        max_batch_size: int = 4,
    ) -> None:
        self.model_path = str(Path(model_path).resolve())
        self.detect_size = int(detect_size)
        self.max_batch_size = max(1, int(max_batch_size))
        self.session = ort.InferenceSession(self.model_path, providers=providers)
        self.providers = tuple(self.session.get_providers())
        _require_primary_provider(providers[0], self.providers)
        self.input_name = self.session.get_inputs()[0].name
        input_shape = self.session.get_inputs()[0].shape
        if list(input_shape[-2:]) != [self.detect_size, self.detect_size]:
            raise ValueError(
                "CTD ONNX input size does not match --detect-size: "
                f"{input_shape!r} vs {self.detect_size}"
            )
        output_names = {item.name for item in self.session.get_outputs()}
        if not {"seg", "det"}.issubset(output_names):
            raise ValueError(f"unexpected CTD ONNX outputs: {sorted(output_names)!r}")

    def _run_one(self, nchw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        outputs = self.session.run(
            ["det", "seg"],
            {self.input_name: np.ascontiguousarray(nchw, dtype=np.float32)},
        )
        return np.asarray(outputs[0]), np.asarray(outputs[1])

    def _batch_forward(self, batch: np.ndarray, _device: str):
        nchw = _rgb_batch_to_nchw(batch)
        lines: list[np.ndarray] = []
        masks: list[np.ndarray] = []
        for index in range(nchw.shape[0]):
            line, mask = self._run_one(nchw[index : index + 1])
            lines.append(line)
            masks.append(mask)
        return np.concatenate(lines, axis=0), np.concatenate(masks, axis=0)

    def raw_mask(self, image_rgb: np.ndarray) -> np.ndarray:
        lines_map, mask = _det_rearrange_forward(
            image_rgb,
            self._batch_forward,
            tgt_size=self.detect_size,
            max_batch_size=self.max_batch_size,
            device="cpu",
        )
        if lines_map is None:
            batch, _ratio, dw, dh = _letterboxed_rgb_to_nchw(
                image_rgb,
                self.detect_size,
            )
            _lines, mask = self._run_one(batch)
            mask = np.asarray(mask).squeeze()
            mask = mask[..., : mask.shape[0] - dh, : mask.shape[1] - dw]
        raw = _postprocess_mask(mask)
        raw = cv2.resize(
            raw,
            (image_rgb.shape[1], image_rgb.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )
        return binary_mask(raw)

    def infer(self, image_rgb: np.ndarray) -> CandidateMaskResult:
        started = time.perf_counter()
        raw = self.raw_mask(image_rgb)
        return CandidateMaskResult(
            candidate_id="ctd_fixed_onnx",
            raw_mask=raw,
            refined_mask=raw.copy(),
            dilated_mask=raw.copy(),
            runtime={
                "seconds": time.perf_counter() - started,
                "backend": "onnxruntime",
                "providers": list(self.providers),
                "detect_size": self.detect_size,
            },
        )

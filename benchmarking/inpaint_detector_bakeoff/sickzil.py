from __future__ import annotations

import os
import time

import cv2
import numpy as np

from .contracts import CandidateMaskResult, binary_mask


def prepare_sickzil_runtime() -> None:
    """Keep the long-running frozen-graph reference stable on TensorFlow 2.x."""
    os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")


def modulo_padded(image: np.ndarray, modulo: int = 16) -> np.ndarray:
    height, width = image.shape[:2]
    height_padding = (modulo - (height % modulo)) % modulo
    width_padding = (modulo - (width % modulo)) % modulo
    padding = [(0, height_padding), (0, width_padding)]
    if image.ndim == 3:
        padding.append((0, 0))
    return np.pad(image, padding, mode="reflect")


def preprocess_sickzil(image_bgr: np.ndarray) -> np.ndarray:
    if image_bgr.ndim == 2:
        image_bgr = cv2.cvtColor(image_bgr, cv2.COLOR_GRAY2RGB)
    elif image_bgr.ndim != 3 or image_bgr.shape[2] not in {3, 4}:
        raise ValueError("SickZil expects a grayscale, BGR, or BGRA image")
    elif image_bgr.shape[2] == 4:
        image_bgr = cv2.cvtColor(image_bgr, cv2.COLOR_BGRA2BGR)
    return np.ascontiguousarray((image_bgr / 255).astype(np.float32))


def class_map_to_mask(segmentation: np.ndarray) -> np.ndarray:
    if segmentation.ndim != 3 or segmentation.shape[2] != 2:
        raise ValueError(f"unexpected SickZil output shape: {segmentation.shape}")
    return binary_mask(np.where(np.argmax(segmentation, axis=2) == 1, 255, 0))


class SickZilSegmentationReference:
    """Exact SickZil SegNet 0.1.0 frozen-graph segmentation reference."""

    def __init__(self, model_path: str, *, segment_pixel_limit: int = 4_000_000) -> None:
        # A long sequence of differently sized pages can exhaust TensorFlow
        # 2.21's oneDNN primitive cache.  The frozen graph is pixel-identical
        # with oneDNN disabled, so select the stable reference runtime before
        # importing TensorFlow.
        prepare_sickzil_runtime()
        import tensorflow as tf

        tf.compat.v1.disable_eager_execution()
        graph_definition = tf.compat.v1.GraphDef()
        with tf.io.gfile.GFile(model_path, "rb") as stream:
            graph_definition.ParseFromString(stream.read())
        self.graph = tf.Graph()
        with self.graph.as_default():
            tf.import_graph_def(graph_definition, name="snet")
        self.session = tf.compat.v1.Session(graph=self.graph)
        self.input_tensor = self.graph.get_tensor_by_name("snet/input_1:0")
        self.output_tensor = self.graph.get_tensor_by_name("snet/conv2d_19/truediv:0")
        self.segment_pixel_limit = int(segment_pixel_limit)

    def _segment_single(self, image: np.ndarray) -> np.ndarray:
        height, width = image.shape[:2]
        padded = modulo_padded(image, 16)
        result = self.session.run(
            self.output_tensor,
            feed_dict={self.input_tensor: np.expand_dims(padded, axis=0)},
        )
        return np.squeeze(result[:, :height, :width, :], axis=0)

    def _segment(self, image: np.ndarray) -> np.ndarray:
        height, width = image.shape[:2]
        if height * width < self.segment_pixel_limit:
            return self._segment_single(image)
        if height > width:
            return np.concatenate(
                (self._segment(image[: height // 2]), self._segment(image[height // 2 :])),
                axis=0,
            )
        return np.concatenate(
            (self._segment(image[:, : width // 2]), self._segment(image[:, width // 2 :])),
            axis=1,
        )

    def infer(self, image_bgr: np.ndarray) -> CandidateMaskResult:
        start = time.perf_counter()
        model_input = preprocess_sickzil(image_bgr)
        segmentation = self._segment(model_input)
        mask = class_map_to_mask(segmentation)
        return CandidateMaskResult(
            candidate_id="sickzil_segnet_0.1.0",
            raw_mask=mask,
            refined_mask=mask,
            dilated_mask=mask,
            stage_tensors={"segmentation": segmentation},
            runtime={
                "seconds": time.perf_counter() - start,
                "provider": "tensorflow_cpu_frozen_graph",
                "reference": "SickZil-Machine SegNet 0.1.0",
                "onednn_enabled": os.environ.get("TF_ENABLE_ONEDNN_OPTS") != "0",
            },
        )

    def close(self) -> None:
        self.session.close()

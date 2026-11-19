from __future__ import annotations

import argparse
import contextlib
import importlib
import json
import math
import os
import pickle
import sys
import traceback
import types
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from modules.ocr.mit48_assets import (
    MIT48_AR_CHECKPOINT,
    MIT48_AR_DICTIONARY,
    MIT48_CTC_CHECKPOINT,
    MIT48_CTC_DICTIONARY,
    MIT48_CTC_EXTRACT_DIRECTORY,
    MIT48_RUNTIME_IDENTITY,
    MIT48_UPSTREAM_COMMIT,
)


TEXT_HEIGHT = 48
MAX_BATCH_SIZE = 16
MAX_TRANSFORMED_LINE_DIMENSION = 8192
MAX_TRANSFORMED_LINE_PIXELS = TEXT_HEIGHT * MAX_TRANSFORMED_LINE_DIMENSION


def _read_exact(stream, count: int) -> bytes:
    chunks: list[bytes] = []
    remaining = count
    while remaining:
        chunk = stream.read(remaining)
        if not chunk:
            raise EOFError("The OCR worker request ended early.")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _send(stream, payload: dict[str, Any]) -> None:
    stream.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8") + b"\n")
    stream.flush()


def _install_upstream_imports(ocr_source_dir: Path) -> None:
    root_package = types.ModuleType("manga_translator")
    root_package.__path__ = [str(ocr_source_dir.parents[1])]
    ocr_package = types.ModuleType("manga_translator.ocr")
    ocr_package.__path__ = [str(ocr_source_dir)]
    config_module = types.ModuleType("manga_translator.config")
    common_module = types.ModuleType("manga_translator.ocr.common")
    utils_module = types.ModuleType("manga_translator.utils")
    utils_module.__path__ = []
    generic_module = types.ModuleType("manga_translator.utils.generic")
    bubble_module = types.ModuleType("manga_translator.utils.bubble")

    class OcrConfig:
        pass

    class OfflineOCR:
        pass

    class UpstreamTextBlock:
        pass

    class Quadrilateral:
        pass

    config_module.OcrConfig = OcrConfig
    common_module.OfflineOCR = OfflineOCR
    utils_module.TextBlock = UpstreamTextBlock
    utils_module.Quadrilateral = Quadrilateral
    utils_module.chunks = None
    utils_module.AvgMeter = None
    generic_module.AvgMeter = None
    bubble_module.is_ignore = None

    sys.modules.update(
        {
            "manga_translator": root_package,
            "manga_translator.ocr": ocr_package,
            "manga_translator.config": config_module,
            "manga_translator.ocr.common": common_module,
            "manga_translator.utils": utils_module,
            "manga_translator.utils.generic": generic_module,
            "manga_translator.utils.bubble": bubble_module,
        }
    )


def _available_device(torch, device: str) -> bool:
    if device == "cpu":
        return True
    if device.startswith("cuda"):
        return bool(torch.cuda.is_available())
    if device == "mps":
        return bool(
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        )
    if device == "xpu":
        return bool(hasattr(torch, "xpu") and torch.xpu.is_available())
    return False


def _load_model(variant: str, device: str, model_root: Path):
    import torch

    if not _available_device(torch, device):
        raise RuntimeError(f"The selected MIT 48px device is unavailable: {device}")

    source_dir = (
        model_root
        / "runtime"
        / MIT48_UPSTREAM_COMMIT
        / "manga_translator"
        / "ocr"
    )
    _install_upstream_imports(source_dir)
    if variant == "ar":
        model_module = importlib.import_module("manga_translator.ocr.model_48px")
        checkpoint = model_root / "ar" / MIT48_AR_CHECKPOINT
        dictionary_path = model_root / "ar" / MIT48_AR_DICTIONARY
        with dictionary_path.open("r", encoding="utf-8") as stream:
            dictionary = stream.read().splitlines()
        model = model_module.OCR(dictionary, 768)
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
    elif variant == "ctc":
        model_module = importlib.import_module("manga_translator.ocr.model_48px_ctc")
        ctc_dir = model_root / "ctc" / MIT48_CTC_EXTRACT_DIRECTORY
        checkpoint = ctc_dir / MIT48_CTC_CHECKPOINT
        dictionary_path = ctc_dir / MIT48_CTC_DICTIONARY
        with dictionary_path.open("r", encoding="utf-8") as stream:
            dictionary = stream.read().splitlines()
        model = model_module.OCR(dictionary, 768)
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        state = state["model"] if "model" in state else state
        expected_missing = {
            "encoders.layers.0.pe.pe",
            "encoders.layers.1.pe.pe",
            "encoders.layers.2.pe.pe",
        }
        for key in expected_missing:
            if key not in state:
                raise RuntimeError(f"MIT 48px CTC checkpoint is missing {key}.")
            del state[key]
        incompatible = model.load_state_dict(state, strict=False)
        if (
            set(incompatible.missing_keys) != expected_missing
            or incompatible.unexpected_keys
        ):
            raise RuntimeError(
                "MIT 48px CTC checkpoint parameters do not match the pinned model."
            )
    else:
        raise ValueError(f"Unsupported MIT 48px variant: {variant}")

    model.eval()
    model.to(device)
    return model, torch


def _line_polygons(block: dict[str, Any]) -> list[list[list[float]]]:
    lines = block.get("lines")
    if lines is None or (isinstance(lines, (list, tuple)) and not lines):
        segment = block.get("segment_points")
        if segment is not None:
            lines = [segment]
        else:
            raise ValueError("The text block has no native line geometry.")
    if isinstance(lines, np.ndarray):
        if lines.shape == (4, 2):
            lines = [lines.tolist()]
        else:
            lines = lines.tolist()
    elif (
        isinstance(lines, list)
        and len(lines) == 4
        and all(isinstance(point, (list, tuple)) and len(point) == 2 for point in lines)
    ):
        lines = [lines]
    if not isinstance(lines, list) or not lines:
        raise ValueError("The text block has no usable line geometry.")
    polygons: list[list[list[float]]] = []
    for points in lines:
        polygon = np.asarray(points, dtype=np.float32)
        if (
            polygon.shape != (4, 2)
            or not np.isfinite(polygon).all()
        ):
            raise ValueError("MIT 48px requires four-point native line geometry.")
        polygons.append(polygon.tolist())
    return polygons


def _projection_intervals(profile: np.ndarray, cross_size: int) -> list[tuple[int, int]]:
    peak = int(profile.max(initial=0))
    if peak <= 0:
        raise ValueError("The source crop has no detectable text strokes.")
    gap_limit = max(1, int(round(cross_size * 0.025)))
    threshold_levels = (0.025, 0.075)
    candidates: list[list[tuple[int, int]]] = []
    for fraction in threshold_levels:
        active = profile >= max(1, int(math.ceil(peak * fraction)))
        indices = np.flatnonzero(active)
        if not len(indices):
            raise ValueError("The source crop has no stable text projection.")
        groups: list[tuple[int, int]] = []
        start = previous = int(indices[0])
        for value in indices[1:]:
            current = int(value)
            if current - previous - 1 > gap_limit:
                groups.append((start, previous + 1))
                start = current
            previous = current
        groups.append((start, previous + 1))
        candidates.append(groups)
    if len(candidates[0]) != len(candidates[1]):
        raise ValueError("The source crop line projection is ambiguous.")
    for low, high in zip(*candidates):
        low_center = (low[0] + low[1]) / 2.0
        high_center = (high[0] + high[1]) / 2.0
        if abs(low_center - high_center) > max(2.0, cross_size * 0.06):
            raise ValueError("The source crop line projection is unstable.")
        if min(low[1] - low[0], high[1] - high[0]) < max(2, cross_size * 0.12):
            raise ValueError("The source crop contains unresolved small text regions.")
    intervals = candidates[1]
    if len(intervals) > 32:
        raise ValueError("The source crop has too many projected text lines.")
    return intervals


def _foreground_crop(
    image: np.ndarray,
    box: list[float],
) -> tuple[np.ndarray, int, int]:
    image_height, image_width = image.shape[:2]
    x1, y1, x2, y2 = (float(value) for value in box)
    left = int(np.clip(np.floor(x1), 0, image_width))
    top = int(np.clip(np.floor(y1), 0, image_height))
    right = int(np.clip(np.ceil(x2), 0, image_width))
    bottom = int(np.clip(np.ceil(y2), 0, image_height))
    if left >= right or top >= bottom:
        raise ValueError("The source text block crop is empty.")
    crop = image[top:bottom, left:right]
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    _threshold, bright = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY | cv2.THRESH_OTSU,
    )
    dark = cv2.bitwise_not(bright)
    foreground = bright if cv2.countNonZero(bright) < cv2.countNonZero(dark) else dark
    foreground_ratio = cv2.countNonZero(foreground) / float(foreground.size)
    if not 0.0005 <= foreground_ratio <= 0.55:
        raise ValueError("The source crop has no unambiguous stroke polarity.")
    return foreground, left, top


def _polygons_from_foreground(
    foreground: np.ndarray,
    left: int,
    top: int,
    intervals: list[tuple[int, int]],
    *,
    vertical: bool,
) -> list[list[list[float]]]:
    cross_size = foreground.shape[1] if vertical else foreground.shape[0]

    polygons: list[list[list[float]]] = []
    margin = max(1, int(round(cross_size * 0.02)))
    for start, end in intervals:
        if vertical:
            line_mask = foreground[:, start:end]
            occupied = np.flatnonzero(np.count_nonzero(line_mask, axis=1))
            if not len(occupied):
                raise ValueError("The source crop contains an empty projected line.")
            line_top = max(0, int(occupied[0]) - margin)
            line_bottom = min(
                foreground.shape[0],
                int(occupied[-1]) + margin + 1,
            )
            line_left = max(0, start - margin)
            line_right = min(foreground.shape[1], end + margin)
            polygon = [
                [left + line_left, top + line_top],
                [left + line_right - 1, top + line_top],
                [left + line_right - 1, top + line_bottom - 1],
                [left + line_left, top + line_bottom - 1],
            ]
        else:
            line_mask = foreground[start:end, :]
            occupied = np.flatnonzero(np.count_nonzero(line_mask, axis=0))
            if not len(occupied):
                raise ValueError("The source crop contains an empty projected line.")
            line_left = max(0, int(occupied[0]) - margin)
            line_right = min(
                foreground.shape[1],
                int(occupied[-1]) + margin + 1,
            )
            line_top = max(0, start - margin)
            line_bottom = min(foreground.shape[0], end + margin)
            polygon = [
                [left + line_left, top + line_top],
                [left + line_right - 1, top + line_top],
                [left + line_right - 1, top + line_bottom - 1],
                [left + line_left, top + line_bottom - 1],
            ]
        polygons.append([[float(x), float(y)] for x, y in polygon])
    if vertical:
        polygons.sort(key=lambda polygon: -sum(point[0] for point in polygon) / 4.0)
    else:
        polygons.sort(key=lambda polygon: sum(point[1] for point in polygon) / 4.0)
    return polygons


def _infer_vertical_from_polygons(
    polygons: list[list[list[float]]],
) -> bool:
    ratios: list[float] = []
    for polygon in polygons:
        points = np.asarray(polygon, dtype=np.float32)
        midpoints = (points[[1, 2, 3, 0]] + points) / 2.0
        norm_v = float(np.linalg.norm(midpoints[2] - midpoints[0]))
        norm_h = float(np.linalg.norm(midpoints[1] - midpoints[3]))
        if norm_v <= 0.0 or norm_h <= 0.0:
            raise ValueError("The native line direction is degenerate.")
        ratios.append(norm_v / norm_h)
    if not ratios:
        raise ValueError("The text block has no native line geometry.")
    vertical_ratio = float(np.median(ratios))
    if vertical_ratio >= 1.15:
        return True
    if vertical_ratio <= 1.0 / 1.15:
        return False
    raise ValueError("The native line direction is ambiguous.")


def _projected_line_geometry(
    image: np.ndarray,
    box: list[float],
    *,
    vertical: bool | None,
    source_lang: str,
) -> tuple[list[list[list[float]]], bool]:
    foreground, left, top = _foreground_crop(image, box)
    if vertical is None:
        row_intervals = _projection_intervals(
            np.count_nonzero(foreground, axis=1),
            foreground.shape[0],
        )
        normalized_language = str(source_lang or "").lower().split("-")[0]
        if normalized_language not in {"ja", "zh"}:
            vertical = False
        else:
            column_intervals = _projection_intervals(
                np.count_nonzero(foreground, axis=0),
                foreground.shape[1],
            )
            row_lines = len(row_intervals)
            column_lines = len(column_intervals)
            if column_lines > 1 and row_lines == 1:
                if foreground.shape[0] >= foreground.shape[1] * 1.15:
                    vertical = True
                elif foreground.shape[1] >= foreground.shape[0] * 1.15:
                    vertical = False
                else:
                    raise ValueError("The source crop text direction is ambiguous.")
            elif (
                column_lines > 1
                and row_lines > 1
                and foreground.shape[0] >= foreground.shape[1] * 1.15
                and row_lines >= 3
            ):
                vertical = True
            elif row_lines > 1 and column_lines == 1:
                if foreground.shape[1] >= foreground.shape[0] * 1.5:
                    vertical = False
                elif foreground.shape[0] >= foreground.shape[1] * 1.5 and row_lines >= 4:
                    vertical = True
                else:
                    raise ValueError("The source crop direction is ambiguous.")
            elif row_lines == 1 and column_lines == 1:
                if foreground.shape[1] >= foreground.shape[0] * 1.5:
                    vertical = False
                elif foreground.shape[0] >= foreground.shape[1] * 1.5:
                    vertical = True
                else:
                    raise ValueError("The source crop text direction is ambiguous.")
            else:
                raise ValueError("The source crop line layout is ambiguous.")

    if vertical:
        intervals = _projection_intervals(
            np.count_nonzero(foreground, axis=0),
            foreground.shape[1],
        )
    else:
        intervals = _projection_intervals(
            np.count_nonzero(foreground, axis=1),
            foreground.shape[0],
        )
    return (
        _polygons_from_foreground(
            foreground,
            left,
            top,
            intervals,
            vertical=vertical,
        ),
        vertical,
    )


def _resolve_line_polygons(
    image: np.ndarray,
    block: dict[str, Any],
    *,
    vertical: bool | None,
    source_lang: str,
) -> tuple[list[list[list[float]]], str, bool]:
    lines = block.get("lines")
    if lines is not None and not (
        isinstance(lines, (list, tuple, np.ndarray)) and len(lines) == 0
    ):
        polygons = _line_polygons(block)
        return (
            polygons,
            "native_line_quad",
            vertical if vertical is not None else _infer_vertical_from_polygons(polygons),
        )
    if block.get("segment_points") is not None:
        polygons = _line_polygons(block)
        return (
            polygons,
            "native_segment_quad",
            vertical if vertical is not None else _infer_vertical_from_polygons(polygons),
        )
    polygons, inferred_vertical = _projected_line_geometry(
        image,
        block["xyxy"],
        vertical=vertical,
        source_lang=source_lang,
    )
    return polygons, "bbox_projection_v1", inferred_vertical


def _transform_line(
    image: np.ndarray,
    polygon: list[list[float]],
    *,
    vertical: bool,
    geometry_source: str = "unknown",
) -> tuple[np.ndarray, int, list[int]]:
    image_height, image_width = image.shape[:2]
    points = np.round(np.asarray(polygon, dtype=np.float32)).astype(np.int64)
    x1 = int(np.clip(points[:, 0].min(), 0, image_width))
    y1 = int(np.clip(points[:, 1].min(), 0, image_height))
    x2 = int(np.clip(points[:, 0].max() + 1, 0, image_width))
    y2 = int(np.clip(points[:, 1].max() + 1, 0, image_height))
    if x1 >= x2 or y1 >= y2:
        raise ValueError("The text line is outside the source image.")
    cropped = image[y1:y2, x1:x2, :]
    source = points.copy()
    source[:, 0] -= x1
    source[:, 1] -= y1
    midpoints = (source[[1, 2, 3, 0]] + source) / 2.0
    norm_v = float(np.linalg.norm(midpoints[2] - midpoints[0]))
    norm_h = float(np.linalg.norm(midpoints[1] - midpoints[3]))
    if norm_v <= 0.0 or norm_h <= 0.0:
        raise ValueError("The text line has a degenerate quadrilateral.")
    ratio = norm_v / norm_h
    if vertical:
        target_width = TEXT_HEIGHT
        target_height = int(round(TEXT_HEIGHT * ratio))
    else:
        target_height = TEXT_HEIGHT
        target_width = int(round(TEXT_HEIGHT / ratio))
    if target_width <= 0 or target_height <= 0:
        raise ValueError("The transformed text line has an empty size.")
    if (
        max(target_width, target_height) > MAX_TRANSFORMED_LINE_DIMENSION
        or target_width * target_height > MAX_TRANSFORMED_LINE_PIXELS
    ):
        raise ValueError(
            "The transformed text line exceeds the "
            f"{MAX_TRANSFORMED_LINE_DIMENSION}-pixel dimension bound "
            f"(source={geometry_source}, size={target_width}x{target_height})."
        )
    destination = np.array(
        [
            [0, 0],
            [target_width - 1, 0],
            [target_width - 1, target_height - 1],
            [0, target_height - 1],
        ],
        dtype=np.float32,
    )
    transform, _mask = cv2.findHomography(source.astype(np.float32), destination, cv2.RANSAC, 5.0)
    if transform is None:
        raise ValueError("The text line perspective transform failed.")
    region = cv2.warpPerspective(cropped, transform, (target_width, target_height))
    if vertical:
        region = cv2.rotate(region, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return region, int(region.shape[1]), [x1, y1, x2, y2]


def _normalize_text(token_ids, dictionary: list[str], *, autoregressive: bool) -> str:
    text: list[str] = []
    for token_id in token_ids:
        token = dictionary[int(token_id)]
        if autoregressive and token == "<S>":
            continue
        if token == "</S>":
            break
        text.append(" " if token == "<SP>" else token)
    return "".join(text)


def _infer_lines(
    model,
    torch,
    variant: str,
    crops: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    import einops

    dictionary = model.dictionary
    outputs: list[dict[str, Any]] = []
    for start in range(0, len(crops), MAX_BATCH_SIZE):
        batch = crops[start : start + MAX_BATCH_SIZE]
        widths = [int(item["width"]) for item in batch]
        max_width = 4 * (max(widths) + 7) // 4
        if variant == "ctc":
            max_width += 128
        regions = np.zeros((len(batch), TEXT_HEIGHT, max_width, 3), dtype=np.uint8)
        for index, item in enumerate(batch):
            region = item["image"]
            regions[index, :, : region.shape[1], :] = region
        images = (torch.from_numpy(regions).float() - 127.5) / 127.5
        images = einops.rearrange(images, "N H W C -> N C H W").to(next(model.parameters()).device)
        with torch.inference_mode():
            if variant == "ar":
                recognized = model.infer_beam_batch_tensor(
                    images,
                    widths,
                    beams_k=5,
                    max_seq_length=255,
                )
                if len(recognized) != len(batch):
                    raise RuntimeError("MIT 48px AR returned an incomplete line batch.")
                for item, result in zip(batch, recognized):
                    token_ids, probability = result[0], float(result[1])
                    text = _normalize_text(
                        token_ids.detach().cpu().tolist(),
                        dictionary,
                        autoregressive=True,
                    )
                    if probability < 0.2:
                        text = ""
                    outputs.append(
                        {
                            "block_id": item["block_id"],
                            "line_index": item["line_index"],
                            "text": text,
                            "confidence": probability,
                            "bbox": item["bbox"],
                            "geometry_source": item["geometry_source"],
                            "empty_reason": "low_confidence" if not text else "",
                        }
                    )
            else:
                recognized = model.decode(images, widths, 0)
                if len(recognized) != len(batch):
                    raise RuntimeError("MIT 48px CTC returned an incomplete line batch.")
                for item, characters in zip(batch, recognized):
                    text_parts: list[str] = []
                    log_probabilities: list[float] = []
                    for character in characters:
                        token_id = int(character[0])
                        token = dictionary[token_id]
                        text_parts.append(" " if token == "<SP>" else token)
                        log_probabilities.append(float(character[1]))
                    probability = (
                        math.exp(sum(log_probabilities) / len(log_probabilities))
                        if log_probabilities
                        else 0.0
                    )
                    text = "".join(text_parts)
                    if probability < 0.5:
                        text = ""
                    outputs.append(
                        {
                            "block_id": item["block_id"],
                            "line_index": item["line_index"],
                            "text": text,
                            "confidence": probability,
                            "bbox": item["bbox"],
                            "geometry_source": item["geometry_source"],
                            "empty_reason": "low_confidence" if not text else "",
                        }
                    )
    return outputs


def _infer(request: dict[str, Any], model, torch) -> dict[str, Any]:
    variant = str(request["variant"])
    device = str(request["device"])
    source_language = str(request.get("source_language") or "")
    runtime = {
        "identity": MIT48_RUNTIME_IDENTITY,
        "torch_version": str(torch.__version__),
        "device": device,
    }
    results: list[dict[str, Any]] = []
    crops: list[dict[str, Any]] = []
    for block in request["blocks"]:
        results.append(
            {
                "block_id": str(block["block_id"]),
                "text": "",
                "confidence": 0.0,
                "empty_reason": "",
                "regions": [],
            }
        )
        direction = str(block.get("direction") or "")
        if direction == "vertical":
            vertical = True
        elif direction == "horizontal":
            vertical = False
        else:
            vertical = None
        try:
            polygons, geometry_source, vertical = _resolve_line_polygons(
                request["image"],
                block,
                vertical=vertical,
                source_lang=str(block.get("source_lang") or source_language),
            )
        except (KeyError, TypeError, ValueError) as exc:
            results[-1]["empty_reason"] = "geometry_review"
            results[-1]["geometry_error"] = str(exc)
            continue
        for line_index, polygon in enumerate(polygons):
            try:
                region, width, bbox = _transform_line(
                    request["image"],
                    polygon,
                    vertical=vertical,
                    geometry_source=geometry_source,
                )
            except (TypeError, ValueError, cv2.error) as exc:
                results[-1]["empty_reason"] = "geometry_review"
                results[-1]["geometry_error"] = (
                    f"source={geometry_source}: {exc}"
                )
                results[-1]["regions"] = []
                break
            crops.append(
                {
                    "block_id": str(block["block_id"]),
                    "block_index": len(results) - 1,
                    "line_index": line_index,
                    "image": region,
                    "width": width,
                    "bbox": bbox,
                    "geometry_source": geometry_source,
                }
            )

    line_outputs = _infer_lines(model, torch, variant, crops) if crops else []
    by_block: dict[str, list[dict[str, Any]]] = {}
    for line in line_outputs:
        by_block.setdefault(str(line["block_id"]), []).append(line)
    no_space = source_language.lower().startswith(("ja", "zh", "th"))
    for result in results:
        if result["empty_reason"] == "geometry_review":
            continue
        regions = sorted(
            by_block.get(result["block_id"], []),
            key=lambda item: int(item["line_index"]),
        )
        result["regions"] = [
            {
                "line_index": int(item["line_index"]),
                "text": str(item["text"] or ""),
                "confidence": float(item["confidence"]),
                "bbox": list(item["bbox"]),
                "geometry_source": str(item["geometry_source"]),
                "empty_reason": str(item["empty_reason"] or ""),
            }
            for item in regions
        ]
        text_fragments = [item["text"] for item in result["regions"] if item["text"]]
        result["text"] = ("" if no_space else " ").join(text_fragments)
        confidences = [item["confidence"] for item in result["regions"]]
        result["confidence"] = (
            sum(confidences) / len(confidences) if confidences else 0.0
        )
        if not result["text"] and result["regions"]:
            result["empty_reason"] = "low_confidence"
    return {"type": "result", "runtime": runtime, "blocks": results}


def _wait_for_shutdown(stream) -> None:
    line = stream.readline()
    if not line:
        return
    try:
        message = json.loads(line.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return
    if message.get("type") not in {"shutdown", "cancel"}:
        raise RuntimeError("The MIT 48px worker received an invalid shutdown message.")


def _release_torch(torch, device: str) -> None:
    if not str(device).startswith("cuda"):
        return
    try:
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    except Exception:
        pass


def run_worker(variant: str, device: str, model_root: str) -> int:
    protocol_output = sys.stdout.buffer
    torch = None
    model = None
    loaded = False
    try:
        header = _read_exact(sys.stdin.buffer, 8)
        size = int.from_bytes(header, "big")
        if size <= 0 or size > 512 * 1024**2:
            raise ValueError("The MIT 48px worker request size is invalid.")
        request = pickle.loads(_read_exact(sys.stdin.buffer, size))
        if not isinstance(request, dict):
            raise ValueError("The MIT 48px worker request must be an object.")
        if request.get("variant") != variant or request.get("device") != device:
            raise ValueError("The MIT 48px worker request does not match its command line.")
        request["model_root"] = model_root
        with contextlib.redirect_stdout(sys.stderr):
            model, torch = _load_model(variant, device, Path(model_root).resolve())
            loaded = True
        runtime = {
            "identity": MIT48_RUNTIME_IDENTITY,
            "torch_version": str(torch.__version__),
            "device": device,
            "worker_pid": os.getpid(),
        }
        _send(protocol_output, {"type": "loaded", "runtime": runtime})
        with contextlib.redirect_stdout(sys.stderr):
            response = _infer(request, model, torch)
        response["runtime"] = runtime
        response["worker_pid"] = os.getpid()
        _send(protocol_output, response)
        _wait_for_shutdown(sys.stdin.buffer)
        return 0
    except BaseException as exc:
        _send(
            protocol_output,
            {
                "type": "error",
                "loaded": loaded,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc()[-12000:],
            },
        )
        try:
            _wait_for_shutdown(sys.stdin.buffer)
        except Exception:
            pass
        return 1
    finally:
        if torch is not None:
            del model
            _release_torch(torch, device)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=("ar", "ctc"), required=True)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps", "xpu"), required=True)
    parser.add_argument("--model-root", required=True)
    args = parser.parse_args()
    return run_worker(args.variant, args.device, args.model_root)


if __name__ == "__main__":
    raise SystemExit(main())

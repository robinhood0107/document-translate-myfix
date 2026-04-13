from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
from PIL import Image

from .export_paths import normalize_export_source_record


OUTPUT_FORMAT_SAME = "same_as_source"
OUTPUT_FORMAT_PNG = "png"
OUTPUT_FORMAT_JPG = "jpg"
OUTPUT_FORMAT_WEBP = "webp"
OUTPUT_FORMAT_BMP = "bmp"

OUTPUT_PRESET_FAST = "fast"
OUTPUT_PRESET_BALANCED = "balanced"
OUTPUT_PRESET_SMALL = "small"

OUTPUT_OVERRIDE_MODE_GLOBAL = "global"
OUTPUT_OVERRIDE_MODE_PROJECT = "project"

SUPPORTED_OUTPUT_FORMATS = (
    OUTPUT_FORMAT_SAME,
    OUTPUT_FORMAT_PNG,
    OUTPUT_FORMAT_JPG,
    OUTPUT_FORMAT_WEBP,
)
SUPPORTED_OUTPUT_PRESETS = (
    OUTPUT_PRESET_FAST,
    OUTPUT_PRESET_BALANCED,
    OUTPUT_PRESET_SMALL,
)
SAME_AS_SOURCE_ALLOWED_FORMATS = {
    OUTPUT_FORMAT_PNG,
    OUTPUT_FORMAT_JPG,
    OUTPUT_FORMAT_WEBP,
    OUTPUT_FORMAT_BMP,
}

DEFAULT_PNG_COMPRESSION = 6
DEFAULT_JPG_QUALITY = 90
DEFAULT_WEBP_QUALITY = 90
DEFAULT_OUTPUT_FORMAT = OUTPUT_FORMAT_SAME
DEFAULT_OUTPUT_PRESET = OUTPUT_PRESET_BALANCED

_OUTPUT_FORMAT_BY_EXTENSION = {
    ".png": OUTPUT_FORMAT_PNG,
    ".jpg": OUTPUT_FORMAT_JPG,
    ".jpeg": OUTPUT_FORMAT_JPG,
    ".webp": OUTPUT_FORMAT_WEBP,
    ".bmp": OUTPUT_FORMAT_BMP,
}
_OUTPUT_EXTENSION_BY_FORMAT = {
    OUTPUT_FORMAT_PNG: ".png",
    OUTPUT_FORMAT_JPG: ".jpg",
    OUTPUT_FORMAT_WEBP: ".webp",
    OUTPUT_FORMAT_BMP: ".bmp",
}
_ILLEGAL_FILE_CHARS_RE = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_BRACKET_RE = re.compile(r"[\(\)\[\]\{\}]")
_TRAILING_VERSION_RE = re.compile(
    r"(?i)(?:[\s._-]*(?:c\d{1,3}(?:[\s._-]*v\d{1,3})?|v\d{1,3}))+$"
)
_WHITESPACE_RE = re.compile(r"\s+")
_SEPARATOR_RE = re.compile(r"[_-]{2,}")

_PRESET_OFFSETS = {
    OUTPUT_FORMAT_PNG: {
        OUTPUT_PRESET_FAST: -5,
        OUTPUT_PRESET_BALANCED: 0,
        OUTPUT_PRESET_SMALL: 3,
    },
    OUTPUT_FORMAT_JPG: {
        OUTPUT_PRESET_FAST: 2,
        OUTPUT_PRESET_BALANCED: -2,
        OUTPUT_PRESET_SMALL: -12,
    },
    OUTPUT_FORMAT_WEBP: {
        OUTPUT_PRESET_FAST: 2,
        OUTPUT_PRESET_BALANCED: -5,
        OUTPUT_PRESET_SMALL: -15,
    },
}
_PNG_ANCHORS = {
    1: (0.95, 220.0),
    6: (0.82, 150.0),
    9: (0.72, 95.0),
}
_JPG_ANCHORS = {
    92: (0.58, 280.0),
    88: (0.40, 250.0),
    78: (0.26, 230.0),
}
_WEBP_ANCHORS = {
    92: (0.46, 190.0),
    85: (0.31, 155.0),
    75: (0.21, 115.0),
}
_BMP_RATIO = 1.0
_BMP_THROUGHPUT = 300.0


def default_global_output_settings() -> dict[str, object]:
    return {
        "automatic_output_format": DEFAULT_OUTPUT_FORMAT,
        "automatic_output_preset": DEFAULT_OUTPUT_PRESET,
        "automatic_output_png_compression_level": DEFAULT_PNG_COMPRESSION,
        "automatic_output_jpg_quality": DEFAULT_JPG_QUALITY,
        "automatic_output_webp_quality": DEFAULT_WEBP_QUALITY,
    }


def default_project_output_preferences() -> dict[str, str]:
    return {
        "output_format_override_mode": OUTPUT_OVERRIDE_MODE_GLOBAL,
        "output_format_override_value": DEFAULT_OUTPUT_FORMAT,
        "output_preset_override_mode": OUTPUT_OVERRIDE_MODE_GLOBAL,
        "output_preset_override_value": DEFAULT_OUTPUT_PRESET,
    }


def normalize_output_format(value: str | None) -> str:
    normalized = str(value or "").strip().lower()
    if normalized in SUPPORTED_OUTPUT_FORMATS:
        return normalized
    return DEFAULT_OUTPUT_FORMAT


def normalize_output_preset(value: str | None) -> str:
    normalized = str(value or "").strip().lower()
    if normalized in SUPPORTED_OUTPUT_PRESETS:
        return normalized
    return DEFAULT_OUTPUT_PRESET


def normalize_output_override_mode(value: str | None) -> str:
    normalized = str(value or "").strip().lower()
    if normalized == OUTPUT_OVERRIDE_MODE_PROJECT:
        return OUTPUT_OVERRIDE_MODE_PROJECT
    return OUTPUT_OVERRIDE_MODE_GLOBAL


def clamp_png_compression(value: object) -> int:
    return max(0, min(9, int(value or DEFAULT_PNG_COMPRESSION)))


def clamp_quality(value: object, default: int) -> int:
    return max(1, min(100, int(value or default)))


def normalize_global_output_settings(settings: Mapping[str, object] | None) -> dict[str, object]:
    raw = dict(default_global_output_settings())
    raw.update(dict(settings or {}))
    raw["automatic_output_format"] = normalize_output_format(raw.get("automatic_output_format"))
    raw["automatic_output_preset"] = normalize_output_preset(raw.get("automatic_output_preset"))
    raw["automatic_output_png_compression_level"] = clamp_png_compression(
        raw.get("automatic_output_png_compression_level", DEFAULT_PNG_COMPRESSION)
    )
    raw["automatic_output_jpg_quality"] = clamp_quality(
        raw.get("automatic_output_jpg_quality", DEFAULT_JPG_QUALITY),
        DEFAULT_JPG_QUALITY,
    )
    raw["automatic_output_webp_quality"] = clamp_quality(
        raw.get("automatic_output_webp_quality", DEFAULT_WEBP_QUALITY),
        DEFAULT_WEBP_QUALITY,
    )
    return raw


def normalize_project_output_preferences(preferences: Mapping[str, object] | None) -> dict[str, str]:
    raw = dict(default_project_output_preferences())
    raw.update(dict(preferences or {}))
    raw["output_format_override_mode"] = normalize_output_override_mode(
        raw.get("output_format_override_mode")
    )
    raw["output_format_override_value"] = normalize_output_format(
        raw.get("output_format_override_value")
    )
    raw["output_preset_override_mode"] = normalize_output_override_mode(
        raw.get("output_preset_override_mode")
    )
    raw["output_preset_override_value"] = normalize_output_preset(
        raw.get("output_preset_override_value")
    )
    return raw


def resolve_automatic_output_settings(
    global_settings: Mapping[str, object] | None,
    project_preferences: Mapping[str, object] | None = None,
) -> dict[str, object]:
    settings = normalize_global_output_settings(global_settings)
    project = normalize_project_output_preferences(project_preferences)
    resolved_format = (
        project["output_format_override_value"]
        if project["output_format_override_mode"] == OUTPUT_OVERRIDE_MODE_PROJECT
        else str(settings["automatic_output_format"])
    )
    resolved_preset = (
        project["output_preset_override_value"]
        if project["output_preset_override_mode"] == OUTPUT_OVERRIDE_MODE_PROJECT
        else str(settings["automatic_output_preset"])
    )
    settings.update(
        {
            "output_format_override_mode": project["output_format_override_mode"],
            "output_format_override_value": project["output_format_override_value"],
            "output_preset_override_mode": project["output_preset_override_mode"],
            "output_preset_override_value": project["output_preset_override_value"],
            "resolved_automatic_output_format": normalize_output_format(resolved_format),
            "resolved_automatic_output_preset": normalize_output_preset(resolved_preset),
        }
    )
    return settings


def source_format_from_path(path: str | None) -> str:
    suffix = Path(str(path or "")).suffix.lower()
    return _OUTPUT_FORMAT_BY_EXTENSION.get(suffix, "")


def resolve_effective_output_format(source_path: str | None, requested_format: str | None) -> str:
    normalized_requested = normalize_output_format(requested_format)
    if normalized_requested != OUTPUT_FORMAT_SAME:
        return normalized_requested
    source_format = source_format_from_path(source_path)
    if source_format in SAME_AS_SOURCE_ALLOWED_FORMATS:
        return source_format
    return OUTPUT_FORMAT_PNG


def resolve_output_extension(source_path: str | None, requested_format: str | None) -> str:
    effective = resolve_effective_output_format(source_path, requested_format)
    return _OUTPUT_EXTENSION_BY_FORMAT.get(effective, ".png")


def _with_preset_offset(value: int, output_format: str, preset: str) -> int:
    return int(value) + int(_PRESET_OFFSETS.get(output_format, {}).get(preset, 0))


def resolve_encode_settings(
    resolved_settings: Mapping[str, object] | None,
    source_path: str | None,
) -> dict[str, object]:
    settings = normalize_global_output_settings(resolved_settings)
    requested_format = str(
        (resolved_settings or {}).get("resolved_automatic_output_format")
        or settings["automatic_output_format"]
    )
    preset = normalize_output_preset(
        (resolved_settings or {}).get("resolved_automatic_output_preset")
        or settings["automatic_output_preset"]
    )
    effective_format = resolve_effective_output_format(source_path, requested_format)
    png_level = clamp_png_compression(
        _with_preset_offset(
            int(settings["automatic_output_png_compression_level"]),
            OUTPUT_FORMAT_PNG,
            preset,
        )
    )
    jpg_quality = clamp_quality(
        _with_preset_offset(
            int(settings["automatic_output_jpg_quality"]),
            OUTPUT_FORMAT_JPG,
            preset,
        ),
        DEFAULT_JPG_QUALITY,
    )
    webp_quality = clamp_quality(
        _with_preset_offset(
            int(settings["automatic_output_webp_quality"]),
            OUTPUT_FORMAT_WEBP,
            preset,
        ),
        DEFAULT_WEBP_QUALITY,
    )
    return {
        "requested_format": requested_format,
        "preset": preset,
        "effective_format": effective_format,
        "extension": resolve_output_extension(source_path, requested_format),
        "png_compression_level": png_level,
        "jpg_quality": jpg_quality,
        "webp_quality": webp_quality,
    }


def build_output_file_name(
    page_base_name: str,
    variant: str,
    source_path: str | None,
    resolved_settings: Mapping[str, object] | None,
) -> str:
    encode = resolve_encode_settings(resolved_settings, source_path)
    return f"{page_base_name}_{variant}{encode['extension']}"


def _ensure_uint8_rgb(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    if arr.ndim == 3 and arr.shape[2] == 4:
        arr = arr[:, :, :3]
    return arr


def write_output_image(
    output_path: str,
    image: np.ndarray,
    *,
    source_path: str | None,
    resolved_settings: Mapping[str, object] | None,
) -> str:
    encode = resolve_encode_settings(resolved_settings, source_path)
    effective_format = str(encode["effective_format"])
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    pil_image = Image.fromarray(_ensure_uint8_rgb(image), mode="RGB")
    if effective_format == OUTPUT_FORMAT_PNG:
        pil_image.save(
            output_path,
            format="PNG",
            compress_level=int(encode["png_compression_level"]),
        )
    elif effective_format == OUTPUT_FORMAT_JPG:
        pil_image.save(
            output_path,
            format="JPEG",
            quality=int(encode["jpg_quality"]),
            optimize=True,
        )
    elif effective_format == OUTPUT_FORMAT_WEBP:
        pil_image.save(
            output_path,
            format="WEBP",
            quality=int(encode["webp_quality"]),
            method=6,
        )
    elif effective_format == OUTPUT_FORMAT_BMP:
        pil_image.save(output_path, format="BMP")
    else:
        pil_image.save(output_path, format="PNG", compress_level=DEFAULT_PNG_COMPRESSION)
    return output_path


def strip_trailing_version_suffix(stem: str) -> str:
    text = str(stem or "").strip()
    if not text:
        return ""
    stripped = _TRAILING_VERSION_RE.sub("", text).strip(" ._-")
    return stripped or text


def sanitize_series_folder_name(stem: str, max_length: int = 255) -> str:
    original = str(stem or "").strip()
    candidate = strip_trailing_version_suffix(original) or original
    candidate = _ILLEGAL_FILE_CHARS_RE.sub("", candidate)
    candidate = candidate.replace("\n", " ").replace("\r", " ").strip(" .")
    candidate = _WHITESPACE_RE.sub(" ", candidate)
    candidate = _SEPARATOR_RE.sub("_", candidate)
    if len(candidate) > max_length:
        candidate = _BRACKET_RE.sub("", candidate)
        candidate = _WHITESPACE_RE.sub(" ", candidate).strip(" .")
    if len(candidate) > max_length:
        candidate = candidate[:max_length].rstrip(" .")
    return candidate or "comic_translate_output"


def _path_within(path: str | None, base_dir: str | None) -> bool:
    if not path or not base_dir:
        return False
    abs_path = os.path.abspath(path)
    abs_base = os.path.abspath(base_dir)
    prefix = abs_base if abs_base.endswith(os.sep) else f"{abs_base}{os.sep}"
    return abs_path == abs_base or abs_path.startswith(prefix)


def resolve_series_folder_name(
    anchor_path: str | None,
    *,
    source_records: Mapping[str, Mapping[str, object]] | None = None,
    project_file: str | None = None,
    temp_dir: str | None = None,
) -> str:
    abs_anchor = os.path.abspath(str(anchor_path or "")) if anchor_path else ""
    source_record = None
    if source_records and abs_anchor:
        source_record = normalize_export_source_record(source_records.get(anchor_path))
        if source_record is None:
            source_record = normalize_export_source_record(source_records.get(abs_anchor))
    if source_record is not None:
        stem = os.path.splitext(os.path.basename(source_record["source_path"]))[0].strip()
        return sanitize_series_folder_name(stem)
    if project_file and _path_within(abs_anchor, temp_dir):
        stem = os.path.splitext(os.path.basename(project_file))[0].strip()
        return sanitize_series_folder_name(stem)
    stem = os.path.splitext(os.path.basename(abs_anchor))[0].strip()
    return sanitize_series_folder_name(stem)


def build_series_output_dir(base_dir: str, series_folder_name: str) -> str:
    return os.path.join(base_dir, sanitize_series_folder_name(series_folder_name))


def interpolate_metric(value: int, anchors: Mapping[int, tuple[float, float]]) -> tuple[float, float]:
    ordered = sorted((int(k), tuple(v)) for k, v in anchors.items())
    if not ordered:
        return 1.0, 1.0
    if value <= ordered[0][0]:
        return ordered[0][1]
    if value >= ordered[-1][0]:
        return ordered[-1][1]
    for index in range(1, len(ordered)):
        left_key, left_value = ordered[index - 1]
        right_key, right_value = ordered[index]
        if left_key <= value <= right_key:
            span = max(right_key - left_key, 1)
            factor = (value - left_key) / span
            ratio = left_value[0] + (right_value[0] - left_value[0]) * factor
            throughput = left_value[1] + (right_value[1] - left_value[1]) * factor
            return ratio, throughput
    return ordered[-1][1]


def estimate_for_encode_settings(encode_settings: Mapping[str, object]) -> tuple[float, float]:
    effective_format = str(encode_settings.get("effective_format") or OUTPUT_FORMAT_PNG)
    if effective_format == OUTPUT_FORMAT_PNG:
        return interpolate_metric(int(encode_settings.get("png_compression_level", DEFAULT_PNG_COMPRESSION)), _PNG_ANCHORS)
    if effective_format == OUTPUT_FORMAT_JPG:
        return interpolate_metric(int(encode_settings.get("jpg_quality", DEFAULT_JPG_QUALITY)), _JPG_ANCHORS)
    if effective_format == OUTPUT_FORMAT_WEBP:
        return interpolate_metric(int(encode_settings.get("webp_quality", DEFAULT_WEBP_QUALITY)), _WEBP_ANCHORS)
    if effective_format == OUTPUT_FORMAT_BMP:
        return _BMP_RATIO, _BMP_THROUGHPUT
    return interpolate_metric(DEFAULT_PNG_COMPRESSION, _PNG_ANCHORS)


def estimate_output_for_pages(
    page_metrics: Iterable[Mapping[str, object]],
    resolved_settings: Mapping[str, object] | None,
) -> dict[str, object]:
    total_input_bytes = 0.0
    total_output_bytes = 0.0
    total_seconds = 0.0
    total_pages = 0
    total_megapixels = 0.0

    for metric in page_metrics or []:
        source_path = str(metric.get("source_path", "") or "")
        byte_size = max(float(metric.get("byte_size", 0.0) or 0.0), 0.0)
        megapixels = max(float(metric.get("megapixels", 0.0) or 0.0), 0.0)
        encode = resolve_encode_settings(resolved_settings, source_path)
        ratio, throughput = estimate_for_encode_settings(encode)
        total_input_bytes += byte_size
        total_output_bytes += byte_size * ratio
        total_megapixels += megapixels
        if throughput > 0:
            total_seconds += megapixels / throughput
        total_pages += 1

    compression_ratio = 0.0
    if total_input_bytes > 0:
        compression_ratio = max(0.0, min(1.0, total_output_bytes / total_input_bytes))

    return {
        "page_count": total_pages,
        "input_bytes": int(round(total_input_bytes)),
        "output_bytes": int(round(total_output_bytes)),
        "compression_ratio": compression_ratio,
        "seconds": max(total_seconds, 0.0),
        "megapixels": total_megapixels,
    }


def format_estimate_ratio_text(estimate: Mapping[str, object]) -> str:
    ratio = float(estimate.get("compression_ratio", 0.0) or 0.0)
    return f"{int(round(ratio * 100))}%"


def format_estimate_seconds_text(seconds: float | int | None) -> str:
    if seconds is None:
        return "0s"
    total_seconds = max(int(round(float(seconds))), 0)
    minutes, secs = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours > 0:
        return f"{hours}h {minutes}m"
    if minutes > 0:
        return f"{minutes}m {secs}s"
    return f"{secs}s"

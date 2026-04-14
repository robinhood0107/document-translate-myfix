from __future__ import annotations

import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
from PIL import Image

from .export_paths import normalize_export_source_record


OUTPUT_TARGET_IMAGES = "individual_images"
OUTPUT_TARGET_ARCHIVE = "single_archive"

OUTPUT_IMAGE_FORMAT_SAME = "same_as_source"
OUTPUT_IMAGE_FORMAT_PNG = "png"
OUTPUT_IMAGE_FORMAT_JPG = "jpg"
OUTPUT_IMAGE_FORMAT_WEBP = "webp"
OUTPUT_IMAGE_FORMAT_BMP = "bmp"

OUTPUT_ARCHIVE_FORMAT_ZIP = "zip"
OUTPUT_ARCHIVE_FORMAT_CBZ = "cbz"

SUPPORTED_OUTPUT_TARGETS = (
    OUTPUT_TARGET_IMAGES,
    OUTPUT_TARGET_ARCHIVE,
)
SUPPORTED_IMAGE_FORMATS = (
    OUTPUT_IMAGE_FORMAT_SAME,
    OUTPUT_IMAGE_FORMAT_PNG,
    OUTPUT_IMAGE_FORMAT_JPG,
    OUTPUT_IMAGE_FORMAT_WEBP,
)
SUPPORTED_ARCHIVE_IMAGE_FORMATS = (
    OUTPUT_IMAGE_FORMAT_PNG,
    OUTPUT_IMAGE_FORMAT_JPG,
    OUTPUT_IMAGE_FORMAT_WEBP,
)
SUPPORTED_ARCHIVE_FORMATS = (
    OUTPUT_ARCHIVE_FORMAT_ZIP,
    OUTPUT_ARCHIVE_FORMAT_CBZ,
)
SAME_AS_SOURCE_ALLOWED_FORMATS = {
    OUTPUT_IMAGE_FORMAT_PNG,
    OUTPUT_IMAGE_FORMAT_JPG,
    OUTPUT_IMAGE_FORMAT_WEBP,
    OUTPUT_IMAGE_FORMAT_BMP,
}

DEFAULT_OUTPUT_TARGET = OUTPUT_TARGET_IMAGES
DEFAULT_OUTPUT_IMAGE_FORMAT = OUTPUT_IMAGE_FORMAT_SAME
DEFAULT_OUTPUT_ARCHIVE_FORMAT = OUTPUT_ARCHIVE_FORMAT_CBZ
DEFAULT_OUTPUT_ARCHIVE_IMAGE_FORMAT = OUTPUT_IMAGE_FORMAT_PNG
DEFAULT_OUTPUT_ARCHIVE_COMPRESSION_LEVEL = 6

_OUTPUT_FORMAT_BY_EXTENSION = {
    ".png": OUTPUT_IMAGE_FORMAT_PNG,
    ".jpg": OUTPUT_IMAGE_FORMAT_JPG,
    ".jpeg": OUTPUT_IMAGE_FORMAT_JPG,
    ".webp": OUTPUT_IMAGE_FORMAT_WEBP,
    ".bmp": OUTPUT_IMAGE_FORMAT_BMP,
}
_OUTPUT_EXTENSION_BY_FORMAT = {
    OUTPUT_IMAGE_FORMAT_PNG: ".png",
    OUTPUT_IMAGE_FORMAT_JPG: ".jpg",
    OUTPUT_IMAGE_FORMAT_WEBP: ".webp",
    OUTPUT_IMAGE_FORMAT_BMP: ".bmp",
}
_ILLEGAL_FILE_CHARS_RE = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_BRACKET_RE = re.compile(r"[\(\)\[\]\{\}]")
_TRAILING_VERSION_RE = re.compile(
    r"(?i)(?:[\s._-]*(?:c\d{1,3}(?:[\s._-]*v\d{1,3})?|v\d{1,3}))+$"
)
_WHITESPACE_RE = re.compile(r"\s+")
_SEPARATOR_RE = re.compile(r"[_-]{2,}")

_ARCHIVE_ESTIMATE_ANCHORS = {
    OUTPUT_IMAGE_FORMAT_PNG: {
        0: (1.10, 225.0),
        3: (0.98, 185.0),
        6: (0.88, 145.0),
        9: (0.80, 110.0),
    },
    OUTPUT_IMAGE_FORMAT_JPG: {
        0: (0.66, 260.0),
        3: (0.63, 230.0),
        6: (0.60, 205.0),
        9: (0.57, 170.0),
    },
    OUTPUT_IMAGE_FORMAT_WEBP: {
        0: (0.54, 210.0),
        3: (0.51, 188.0),
        6: (0.49, 166.0),
        9: (0.46, 142.0),
    },
}


def default_global_output_settings() -> dict[str, object]:
    return {
        "automatic_output_target": DEFAULT_OUTPUT_TARGET,
        "automatic_output_image_format": DEFAULT_OUTPUT_IMAGE_FORMAT,
        "automatic_output_archive_format": DEFAULT_OUTPUT_ARCHIVE_FORMAT,
        "automatic_output_archive_image_format": DEFAULT_OUTPUT_ARCHIVE_IMAGE_FORMAT,
        "automatic_output_archive_compression_level": DEFAULT_OUTPUT_ARCHIVE_COMPRESSION_LEVEL,
    }


def default_project_output_preferences() -> dict[str, object]:
    return {
        "output_use_global": True,
        "output_target": DEFAULT_OUTPUT_TARGET,
        "output_image_format": DEFAULT_OUTPUT_IMAGE_FORMAT,
        "output_archive_format": DEFAULT_OUTPUT_ARCHIVE_FORMAT,
        "output_archive_image_format": DEFAULT_OUTPUT_ARCHIVE_IMAGE_FORMAT,
        "output_archive_compression_level": DEFAULT_OUTPUT_ARCHIVE_COMPRESSION_LEVEL,
    }


def normalize_output_target(value: str | None) -> str:
    normalized = str(value or "").strip().lower()
    if normalized in SUPPORTED_OUTPUT_TARGETS:
        return normalized
    return DEFAULT_OUTPUT_TARGET


def normalize_image_format(value: str | None) -> str:
    normalized = str(value or "").strip().lower()
    if normalized in SUPPORTED_IMAGE_FORMATS:
        return normalized
    return DEFAULT_OUTPUT_IMAGE_FORMAT


def normalize_archive_image_format(value: str | None) -> str:
    normalized = str(value or "").strip().lower()
    if normalized in SUPPORTED_ARCHIVE_IMAGE_FORMATS:
        return normalized
    return DEFAULT_OUTPUT_ARCHIVE_IMAGE_FORMAT


def normalize_archive_format(value: str | None) -> str:
    normalized = str(value or "").strip().lower()
    if normalized in SUPPORTED_ARCHIVE_FORMATS:
        return normalized
    return DEFAULT_OUTPUT_ARCHIVE_FORMAT


def clamp_archive_compression_level(value: object) -> int:
    return max(0, min(9, int(value or DEFAULT_OUTPUT_ARCHIVE_COMPRESSION_LEVEL)))


def normalize_global_output_settings(settings: Mapping[str, object] | None) -> dict[str, object]:
    raw = dict(default_global_output_settings())
    raw.update(dict(settings or {}))
    legacy_image_format = raw.get("automatic_output_format")
    raw["automatic_output_target"] = normalize_output_target(raw.get("automatic_output_target"))
    raw["automatic_output_image_format"] = normalize_image_format(
        raw.get("automatic_output_image_format", legacy_image_format)
    )
    raw["automatic_output_archive_format"] = normalize_archive_format(
        raw.get("automatic_output_archive_format")
    )
    raw["automatic_output_archive_image_format"] = normalize_archive_image_format(
        raw.get("automatic_output_archive_image_format")
    )
    raw["automatic_output_archive_compression_level"] = clamp_archive_compression_level(
        raw.get("automatic_output_archive_compression_level", DEFAULT_OUTPUT_ARCHIVE_COMPRESSION_LEVEL)
    )
    for legacy_key in (
        "automatic_output_format",
        "automatic_output_preset",
        "automatic_output_png_compression_level",
        "automatic_output_jpg_quality",
        "automatic_output_webp_quality",
    ):
        raw.pop(legacy_key, None)
    return raw


def normalize_project_output_preferences(preferences: Mapping[str, object] | None) -> dict[str, object]:
    raw = dict(default_project_output_preferences())
    raw.update(dict(preferences or {}))

    if "output_use_global" not in raw:
        legacy_target_mode = str(raw.get("output_target_override_mode", "") or "").strip().lower()
        legacy_image_mode = str(raw.get("output_format_override_mode", "") or "").strip().lower()
        raw["output_use_global"] = not (
            legacy_target_mode == "project" or legacy_image_mode == "project"
        )

    raw["output_use_global"] = bool(raw.get("output_use_global", True))
    raw["output_target"] = normalize_output_target(
        raw.get("output_target", raw.get("output_target_override_value"))
    )
    raw["output_image_format"] = normalize_image_format(
        raw.get("output_image_format", raw.get("output_format_override_value"))
    )
    raw["output_archive_format"] = normalize_archive_format(raw.get("output_archive_format"))
    raw["output_archive_image_format"] = normalize_archive_image_format(
        raw.get("output_archive_image_format")
    )
    raw["output_archive_compression_level"] = clamp_archive_compression_level(
        raw.get("output_archive_compression_level")
    )
    for legacy_key in (
        "output_target_override_mode",
        "output_target_override_value",
        "output_format_override_mode",
        "output_format_override_value",
        "output_preset_override_mode",
        "output_preset_override_value",
    ):
        raw.pop(legacy_key, None)
    return raw


def resolve_automatic_output_settings(
    global_settings: Mapping[str, object] | None,
    project_preferences: Mapping[str, object] | None = None,
) -> dict[str, object]:
    settings = normalize_global_output_settings(global_settings)
    project = normalize_project_output_preferences(project_preferences)
    if project["output_use_global"]:
        resolved = {
            "resolved_automatic_output_target": settings["automatic_output_target"],
            "resolved_automatic_output_image_format": settings["automatic_output_image_format"],
            "resolved_automatic_output_archive_format": settings["automatic_output_archive_format"],
            "resolved_automatic_output_archive_image_format": settings["automatic_output_archive_image_format"],
            "resolved_automatic_output_archive_compression_level": settings["automatic_output_archive_compression_level"],
        }
    else:
        resolved = {
            "resolved_automatic_output_target": project["output_target"],
            "resolved_automatic_output_image_format": project["output_image_format"],
            "resolved_automatic_output_archive_format": project["output_archive_format"],
            "resolved_automatic_output_archive_image_format": project["output_archive_image_format"],
            "resolved_automatic_output_archive_compression_level": project["output_archive_compression_level"],
        }
    settings.update(project)
    settings.update(resolved)
    return settings


def is_individual_images_mode(resolved_settings: Mapping[str, object] | None) -> bool:
    target = str(
        (resolved_settings or {}).get("resolved_automatic_output_target", DEFAULT_OUTPUT_TARGET)
    )
    return normalize_output_target(target) == OUTPUT_TARGET_IMAGES


def is_single_archive_mode(resolved_settings: Mapping[str, object] | None) -> bool:
    return not is_individual_images_mode(resolved_settings)


def source_format_from_path(path: str | None) -> str:
    suffix = Path(str(path or "")).suffix.lower()
    return _OUTPUT_FORMAT_BY_EXTENSION.get(suffix, "")


def resolve_individual_output_format(source_path: str | None, requested_format: str | None) -> str:
    normalized_requested = normalize_image_format(requested_format)
    if normalized_requested != OUTPUT_IMAGE_FORMAT_SAME:
        return normalized_requested
    source_format = source_format_from_path(source_path)
    if source_format in SAME_AS_SOURCE_ALLOWED_FORMATS:
        return source_format
    return OUTPUT_IMAGE_FORMAT_PNG


def resolve_output_extension_for_format(output_format: str) -> str:
    return _OUTPUT_EXTENSION_BY_FORMAT.get(output_format, ".png")


def resolve_individual_output_extension(source_path: str | None, requested_format: str | None) -> str:
    effective = resolve_individual_output_format(source_path, requested_format)
    return resolve_output_extension_for_format(effective)


def build_output_file_name(
    page_base_name: str,
    variant: str,
    source_path: str | None,
    resolved_settings: Mapping[str, object] | None,
) -> str:
    requested_format = str(
        (resolved_settings or {}).get("resolved_automatic_output_image_format", DEFAULT_OUTPUT_IMAGE_FORMAT)
    )
    ext = resolve_individual_output_extension(source_path, requested_format)
    return f"{page_base_name}_{variant}{ext}"


def build_archive_page_file_name(
    page_index: int,
    total_pages: int,
    page_base_name: str,
    archive_image_format: str,
) -> str:
    digits = max(len(str(max(total_pages, 1))), 3)
    ext = resolve_output_extension_for_format(normalize_archive_image_format(archive_image_format))
    sanitized = sanitize_series_folder_name(page_base_name, max_length=120)
    return f"{page_index + 1:0{digits}d}_{sanitized}{ext}"


def build_archive_file_name(bundle_name: str, archive_format: str) -> str:
    ext = f".{normalize_archive_format(archive_format)}"
    stem = sanitize_series_folder_name(bundle_name, max_length=180)
    return f"{stem}_translated{ext}"


def build_archive_staging_dir(series_dir: str, export_token: str) -> str:
    return os.path.join(series_dir, f".archive_staging_{sanitize_series_folder_name(export_token, 64)}")


def _ensure_uint8_rgb(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    if arr.ndim == 3 and arr.shape[2] == 4:
        arr = arr[:, :, :3]
    return arr


def write_image_with_format(output_path: str, image: np.ndarray, output_format: str) -> str:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    pil_image = Image.fromarray(_ensure_uint8_rgb(image), mode="RGB")
    fmt = str(output_format or "").strip().lower()
    if fmt == OUTPUT_IMAGE_FORMAT_PNG:
        pil_image.save(output_path, format="PNG", compress_level=0, optimize=False)
    elif fmt == OUTPUT_IMAGE_FORMAT_JPG:
        pil_image.save(
            output_path,
            format="JPEG",
            quality=100,
            subsampling=0,
            optimize=False,
        )
    elif fmt == OUTPUT_IMAGE_FORMAT_WEBP:
        pil_image.save(
            output_path,
            format="WEBP",
            quality=100,
            method=6,
        )
    elif fmt == OUTPUT_IMAGE_FORMAT_BMP:
        pil_image.save(output_path, format="BMP")
    else:
        pil_image.save(output_path, format="PNG", compress_level=0, optimize=False)
    return output_path


def write_output_image(
    output_path: str,
    image: np.ndarray,
    *,
    source_path: str | None,
    resolved_settings: Mapping[str, object] | None,
) -> str:
    requested = str(
        (resolved_settings or {}).get("resolved_automatic_output_image_format", DEFAULT_OUTPUT_IMAGE_FORMAT)
    )
    effective = resolve_individual_output_format(source_path, requested)
    return write_image_with_format(output_path, image, effective)


def write_archive_image(
    output_path: str,
    image: np.ndarray,
    *,
    resolved_settings: Mapping[str, object] | None,
) -> str:
    output_format = str(
        (resolved_settings or {}).get(
            "resolved_automatic_output_archive_image_format",
            DEFAULT_OUTPUT_ARCHIVE_IMAGE_FORMAT,
        )
    )
    return write_image_with_format(output_path, image, output_format)


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


def estimate_archive_for_pages(
    page_metrics: Iterable[Mapping[str, object]],
    archive_image_format: str,
    compression_level: int,
) -> dict[str, object]:
    total_input_bytes = 0.0
    total_output_bytes = 0.0
    total_seconds = 0.0
    total_pages = 0
    total_megapixels = 0.0
    normalized_format = normalize_archive_image_format(archive_image_format)
    ratio, throughput = interpolate_metric(
        clamp_archive_compression_level(compression_level),
        _ARCHIVE_ESTIMATE_ANCHORS[normalized_format],
    )

    for metric in page_metrics or []:
        byte_size = max(float(metric.get("byte_size", 0.0) or 0.0), 0.0)
        megapixels = max(float(metric.get("megapixels", 0.0) or 0.0), 0.0)
        total_input_bytes += byte_size
        total_output_bytes += byte_size * ratio
        total_megapixels += megapixels
        if throughput > 0:
            total_seconds += megapixels / throughput
        total_pages += 1

    compression_ratio = 0.0
    if total_input_bytes > 0:
        compression_ratio = max(0.0, min(10.0, total_output_bytes / total_input_bytes))

    return {
        "page_count": total_pages,
        "input_bytes": int(round(total_input_bytes)),
        "output_bytes": int(round(total_output_bytes)),
        "compression_ratio": compression_ratio,
        "seconds": max(total_seconds, 0.0),
        "megapixels": total_megapixels,
        "archive_image_format": normalized_format,
        "compression_level": clamp_archive_compression_level(compression_level),
    }


def estimate_archive_options_for_pages(
    page_metrics: Iterable[Mapping[str, object]],
    compression_level: int,
) -> dict[str, dict[str, object]]:
    return {
        image_format: estimate_archive_for_pages(page_metrics, image_format, compression_level)
        for image_format in SUPPORTED_ARCHIVE_IMAGE_FORMATS
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


def format_estimate_size_text(num_bytes: int | float | None) -> str:
    value = max(float(num_bytes or 0.0), 0.0)
    units = ["B", "KB", "MB", "GB", "TB"]
    index = 0
    while value >= 1024.0 and index < len(units) - 1:
        value /= 1024.0
        index += 1
    if index == 0:
        return f"{int(value)} {units[index]}"
    return f"{value:.1f} {units[index]}"


def preserve_preview_file(
    preview_path: str | None,
    *,
    temp_root: str | None = None,
) -> str:
    path = str(preview_path or "").strip()
    if not path or not os.path.exists(path):
        return ""
    target_dir = tempfile.mkdtemp(prefix="comic_translate_preview_", dir=temp_root or None)
    target_path = os.path.join(target_dir, os.path.basename(path))
    shutil.copy2(path, target_path)
    return target_path

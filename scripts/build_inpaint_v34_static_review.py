#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Sequence

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.validation_artifact_harness import (  # noqa: E402
    select_managed_output_directory,
)


FAMILY = "inpaint-v34-static-review"
CATEGORY = "40-inpaint-mask-render"
LEDGER_SCHEMA_VERSION = "inpaint-v34-static-review-ledger-v1"
LEDGER_SEAL_SCHEMA_VERSION = "inpaint-v34-static-review-ledger-seal-v1"
OUTPUT_SCHEMA_VERSION = "inpaint-v34-static-review-output-v1"
OUTPUT_SEAL_SCHEMA_VERSION = "inpaint-v34-static-review-output-seal-v1"

CAUSES = frozenset(
    {
        "미검출",
        "마스크 부족",
        "semantic 거절",
        "LaMa 재생성",
        "회귀 없음",
    }
)
FINALIST_LABELS = {
    "context_additive": "context-additive",
    "narrow_replacement": "narrow replacement",
    "conditional_segmenter": "conditional segmenter",
}
MASK_COLORS_RGB = {
    "core": (0, 210, 0),
    "effect": (255, 215, 0),
    "protect": (230, 35, 35),
    "preserve": (30, 90, 235),
}
MASK_ROLES = tuple(MASK_COLORS_RGB)
OUTPUT_IMAGE_SUFFIXES = frozenset({".png", ".webp"})
SEALED_SOURCE_INVENTORY_FILE_SHA256 = "8672a8a2a82be9b2805698372dd967cc46737386495151f0fe76cac823e409bc"
SEALED_SOURCE_INVENTORY_SHA256 = "63a58d0b7d09c8431af2202b980ef10e847a392b0fbc07fc87d2430859989667"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def _sealed_review_expectations(
    source_inventory_path: Path,
    *,
    expected_page_count: int,
    expected_selected_count: int,
) -> tuple[dict[str, tuple[int, str]], set[str], set[tuple[str, tuple[int, int, int, int]]]]:
    source_inventory_path = source_inventory_path.resolve()
    seal_path = source_inventory_path.with_suffix(
        source_inventory_path.suffix + ".seal.json"
    )
    payload = _read_json(source_inventory_path)
    seal = _read_json(seal_path)
    if (
        _sha256(source_inventory_path) != SEALED_SOURCE_INVENTORY_FILE_SHA256
        or seal.get("manifest_sha256") != SEALED_SOURCE_INVENTORY_FILE_SHA256
        or payload.get("source_inventory_sha256") != SEALED_SOURCE_INVENTORY_SHA256
        or seal.get("source_inventory_sha256") != SEALED_SOURCE_INVENTORY_SHA256
        or payload.get("candidate_seen") is not False
        or seal.get("candidate_seen") is not False
        or payload.get("page_count") != expected_page_count
        or seal.get("page_count") != expected_page_count
    ):
        raise ValueError("static review canonical source inventory seal differs")
    raw_pages = payload.get("pages")
    annotations = payload.get("translucent_carrier_annotations")
    if not isinstance(raw_pages, list) or not isinstance(annotations, list):
        raise ValueError("static review canonical source inventory is incomplete")
    expected_pages: dict[str, tuple[int, str]] = {}
    selected_ids: set[str] = set()
    for raw in raw_pages:
        if not isinstance(raw, Mapping):
            raise ValueError("static review canonical page is invalid")
        page_id = _safe_id(raw.get("page_id"), label="canonical page id")
        number = _positive_int(raw.get("inventory_number"), label="canonical page number")
        filename = str(raw.get("source_filename") or "").strip()
        if page_id in expected_pages or not filename:
            raise ValueError("static review canonical page identity is invalid")
        expected_pages[page_id] = (number, filename)
        if number > expected_page_count - expected_selected_count:
            selected_ids.add(page_id)
    if len(expected_pages) != expected_page_count:
        raise ValueError("static review canonical page count differs")
    if {number for number, _filename in expected_pages.values()} != set(
        range(1, expected_page_count + 1)
    ):
        raise ValueError("static review canonical page numbering differs")
    highres: set[tuple[str, tuple[int, int, int, int]]] = set()
    for raw in annotations:
        if not isinstance(raw, Mapping) or raw.get("source_only") is not True:
            raise ValueError("static review canonical carrier is invalid")
        page_id = _safe_id(raw.get("page_id"), label="canonical carrier page id")
        bbox = raw.get("bbox_xyxy")
        if (
            page_id not in selected_ids
            or not isinstance(bbox, list)
            or len(bbox) != 4
            or any(isinstance(value, bool) or not isinstance(value, int) for value in bbox)
        ):
            raise ValueError("static review canonical carrier identity is invalid")
        highres.add((page_id, tuple(bbox)))
    if len(highres) != len(annotations):
        raise ValueError("static review canonical carrier inventory is duplicated")
    return expected_pages, selected_ids, highres


def _atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_name(f".{path.name}.partial")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)


def _read_image(path: Path, flags: int) -> np.ndarray:
    result = cv2.imdecode(np.fromfile(path, dtype=np.uint8), flags)
    if result is None or result.size == 0:
        raise FileNotFoundError(path)
    return np.ascontiguousarray(result)


def _safe_id(value: object, *, label: str) -> str:
    normalized = str(value or "").strip()
    if (
        not normalized
        or normalized in {".", ".."}
        or "/" in normalized
        or "\\" in normalized
    ):
        raise ValueError(f"{label} must be a path-safe identifier")
    return normalized


def _sha_value(value: object, *, label: str) -> str:
    normalized = str(value or "").lower()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise ValueError(f"{label} must be a SHA-256 value")
    return normalized


def _positive_int(value: object, *, label: str, allow_zero: bool = False) -> int:
    minimum = 0 if allow_zero else 1
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def _resolve_artifact_path(ledger_path: Path, value: object) -> Path:
    raw = str(value or "").strip()
    if not raw:
        raise ValueError("review artifact path is missing")
    path = Path(raw)
    if not path.is_absolute():
        path = ledger_path.parent / path
    return path.resolve()


@dataclass(frozen=True, slots=True)
class ArtifactRef:
    path: Path
    file_sha256: str
    pixel_sha256: str
    shape: tuple[int, ...]
    dtype: str
    kind: str


@dataclass(frozen=True, slots=True)
class FinalistRef:
    candidate_id: str
    label: str
    image: ArtifactRef


@dataclass(frozen=True, slots=True)
class CropSpec:
    number: int
    bbox: tuple[int, int, int, int]
    cause: str
    required_highres: bool


@dataclass(frozen=True, slots=True)
class ReviewPage:
    number: int
    page_id: str
    filename: str
    selected_review: bool
    causes: tuple[str, ...]
    source: ArtifactRef
    pr6: ArtifactRef
    finalists: tuple[FinalistRef, ...]
    masks: Mapping[str, ArtifactRef]
    crops: tuple[CropSpec, ...]


@dataclass(frozen=True, slots=True)
class ValidatedReviewLedger:
    ledger_path: Path
    ledger_file_sha256: str
    ledger_payload_sha256: str
    seal_file_sha256: str
    page_inventory_sha256: str
    pages: tuple[ReviewPage, ...]
    input_artifact_inventory: tuple[dict[str, object], ...]


def _artifact_ref(
    ledger_path: Path,
    value: object,
    *,
    kind: str,
    expected_shape: tuple[int, ...] | None = None,
) -> ArtifactRef:
    if not isinstance(value, Mapping):
        raise ValueError(f"review {kind} artifact record is invalid")
    path = _resolve_artifact_path(ledger_path, value.get("path"))
    shape_value = value.get("shape")
    expected_dimensions = 3 if kind == "image" else 2
    if (
        not isinstance(shape_value, list)
        or len(shape_value) != expected_dimensions
        or any(
            isinstance(dimension, bool)
            or not isinstance(dimension, int)
            or dimension <= 0
            for dimension in shape_value
        )
    ):
        raise ValueError(f"review {kind} artifact shape is invalid")
    shape = tuple(int(dimension) for dimension in shape_value)
    if kind == "image" and shape[2] != 3:
        raise ValueError("review images must decode to three channels")
    if expected_shape is not None and shape != expected_shape:
        raise ValueError(f"review {kind} artifact shape differs from its page")
    dtype = str(value.get("dtype") or "")
    if dtype != "uint8":
        raise ValueError("review artifacts must use uint8 pixels")
    return ArtifactRef(
        path=path,
        file_sha256=_sha_value(value.get("file_sha256"), label="artifact file"),
        pixel_sha256=_sha_value(
            value.get("pixel_sha256"), label="artifact pixel"
        ),
        shape=shape,
        dtype=dtype,
        kind=kind,
    )


def read_bound_artifact(reference: ArtifactRef) -> np.ndarray:
    if not reference.path.is_file() or _sha256(reference.path) != (
        reference.file_sha256
    ):
        raise ValueError("review artifact file SHA differs")
    flags = cv2.IMREAD_COLOR if reference.kind == "image" else cv2.IMREAD_GRAYSCALE
    value = _read_image(reference.path, flags)
    if tuple(value.shape) != reference.shape or str(value.dtype) != reference.dtype:
        raise ValueError("review artifact decoded shape/dtype differs")
    if reference.kind == "mask" and np.any((value != 0) & (value != 255)):
        raise ValueError("review mask artifact is not strict binary")
    if _array_sha256(value) != reference.pixel_sha256:
        raise ValueError("review artifact pixel SHA differs")
    return value


def _input_artifact_row(
    page_id: str,
    role: str,
    reference: ArtifactRef,
) -> dict[str, object]:
    return {
        "page_id": page_id,
        "role": role,
        "file_sha256": reference.file_sha256,
        "pixel_sha256": reference.pixel_sha256,
        "shape": list(reference.shape),
        "dtype": reference.dtype,
        "kind": reference.kind,
    }


def validate_review_ledger(
    ledger_path: Path,
    *,
    source_inventory_path: Path,
    expected_page_count: int = 130,
    expected_selected_count: int = 12,
    expected_highres_count: int = 9,
) -> ValidatedReviewLedger:
    ledger_path = ledger_path.resolve()
    seal_path = ledger_path.with_suffix(ledger_path.suffix + ".seal.json")
    if not ledger_path.is_file() or not seal_path.is_file():
        raise ValueError("static review ledger or seal is missing")
    ledger_file_sha = _sha256(ledger_path)
    seal_file_sha = _sha256(seal_path)
    seal = _read_json(seal_path)
    if (
        seal.get("schema_version") != LEDGER_SEAL_SCHEMA_VERSION
        or seal.get("ledger_file_sha256") != ledger_file_sha
        or seal.get("review_ledger_frozen") is not True
        or seal.get("candidate_generation_complete") is not True
    ):
        raise ValueError("static review ledger seal differs")
    payload = _read_json(ledger_path)
    if payload.get("schema_version") != LEDGER_SCHEMA_VERSION:
        raise ValueError("static review ledger schema differs")
    unsigned = {
        key: value for key, value in payload.items() if key != "ledger_sha256"
    }
    payload_sha = _canonical_sha256(unsigned)
    if (
        payload.get("ledger_sha256") != payload_sha
        or seal.get("ledger_payload_sha256") != payload_sha
        or payload.get("review_ledger_frozen") is not True
        or payload.get("candidate_generation_complete") is not True
    ):
        raise ValueError("static review ledger payload SHA/status differs")
    raw_pages = payload.get("pages")
    if not isinstance(raw_pages, list):
        raise ValueError("static review ledger pages are invalid")
    expected_page_count = _positive_int(
        expected_page_count, label="expected page count"
    )
    expected_selected_count = _positive_int(
        expected_selected_count, label="expected selected count", allow_zero=True
    )
    expected_highres_count = _positive_int(
        expected_highres_count, label="expected highres count", allow_zero=True
    )
    canonical_pages, canonical_selected, canonical_highres = _sealed_review_expectations(
        source_inventory_path,
        expected_page_count=expected_page_count,
        expected_selected_count=expected_selected_count,
    )
    if (
        len(canonical_selected) != expected_selected_count
        or len(canonical_highres) != expected_highres_count
    ):
        raise ValueError("static review canonical selected/highres count differs")
    if payload.get("page_count") != expected_page_count or len(raw_pages) != (
        expected_page_count
    ):
        raise ValueError("static review ledger page count differs")
    if payload.get("page_inventory_sha256") != _canonical_sha256(raw_pages):
        raise ValueError("static review ledger page inventory SHA differs")
    if seal.get("page_inventory_sha256") != payload.get(
        "page_inventory_sha256"
    ):
        raise ValueError("static review ledger seal page inventory differs")

    pages: list[ReviewPage] = []
    page_ids: set[str] = set()
    filenames: set[str] = set()
    input_artifacts: list[dict[str, object]] = []
    selected_ids: list[str] = []
    highres_count = 0
    highres_regions: set[tuple[str, tuple[int, int, int, int]]] = set()
    for raw in raw_pages:
        if not isinstance(raw, Mapping):
            raise ValueError("static review page record is invalid")
        number = _positive_int(raw.get("number"), label="review page number")
        page_id = _safe_id(raw.get("page_id"), label="review page id")
        filename = str(raw.get("filename") or "").strip()
        if (
            not filename
            or filename in filenames
            or "/" in filename
            or "\\" in filename
        ):
            raise ValueError("review filename must be a unique basename")
        if page_id in page_ids:
            raise ValueError("static review page id is duplicated")
        if canonical_pages.get(page_id) != (number, filename):
            raise ValueError("static review page differs from canonical source inventory")
        page_ids.add(page_id)
        filenames.add(filename)
        selected = raw.get("selected_review") is True
        raw_causes = raw.get("causes")
        if (
            not isinstance(raw_causes, list)
            or not raw_causes
            or any(str(cause) not in CAUSES for cause in raw_causes)
            or len({str(cause) for cause in raw_causes}) != len(raw_causes)
            or ("회귀 없음" in raw_causes and len(raw_causes) != 1)
        ):
            raise ValueError("static review page causes are invalid")
        causes = tuple(str(cause) for cause in raw_causes)

        source = _artifact_ref(ledger_path, raw.get("source"), kind="image")
        source_value = read_bound_artifact(source)
        shape = tuple(source_value.shape)
        pr6 = _artifact_ref(
            ledger_path, raw.get("pr6"), kind="image", expected_shape=shape
        )
        read_bound_artifact(pr6)
        input_artifacts.extend(
            (
                _input_artifact_row(page_id, "source", source),
                _input_artifact_row(page_id, "pr6", pr6),
            )
        )

        raw_finalists = raw.get("finalists")
        if not isinstance(raw_finalists, list) or len(raw_finalists) > 2:
            raise ValueError("static review finalists must contain at most two rows")
        finalists: list[FinalistRef] = []
        finalist_ids: set[str] = set()
        for finalist in raw_finalists:
            if not isinstance(finalist, Mapping):
                raise ValueError("static review finalist record is invalid")
            candidate_id = str(finalist.get("candidate_id") or "")
            if candidate_id not in FINALIST_LABELS or candidate_id in finalist_ids:
                raise ValueError("static review finalist identity is invalid")
            finalist_ids.add(candidate_id)
            image = _artifact_ref(
                ledger_path,
                finalist.get("image"),
                kind="image",
                expected_shape=shape,
            )
            read_bound_artifact(image)
            finalists.append(
                FinalistRef(
                    candidate_id=candidate_id,
                    label=FINALIST_LABELS[candidate_id],
                    image=image,
                )
            )
            input_artifacts.append(
                _input_artifact_row(page_id, f"finalist:{candidate_id}", image)
            )
        if selected and not finalists:
            raise ValueError("selected static review page lacks a finalist")

        raw_masks = raw.get("masks")
        if not isinstance(raw_masks, Mapping) or set(raw_masks) != set(MASK_ROLES):
            raise ValueError("static review mask roles differ")
        masks: dict[str, ArtifactRef] = {}
        mask_shape = shape[:2]
        for role in MASK_ROLES:
            reference = _artifact_ref(
                ledger_path,
                raw_masks[role],
                kind="mask",
                expected_shape=mask_shape,
            )
            read_bound_artifact(reference)
            masks[role] = reference
            input_artifacts.append(
                _input_artifact_row(page_id, f"mask:{role}", reference)
            )

        raw_crops = raw.get("crops")
        if not isinstance(raw_crops, list):
            raise ValueError("static review crops are invalid")
        crops: list[CropSpec] = []
        crop_numbers: set[int] = set()
        height, width = mask_shape
        for raw_crop in raw_crops:
            if not isinstance(raw_crop, Mapping):
                raise ValueError("static review crop record is invalid")
            crop_number = _positive_int(
                raw_crop.get("number"), label="review crop number"
            )
            raw_bbox = raw_crop.get("bbox")
            if (
                crop_number in crop_numbers
                or not isinstance(raw_bbox, list)
                or len(raw_bbox) != 4
                or any(
                    isinstance(coordinate, bool)
                    or not isinstance(coordinate, int)
                    for coordinate in raw_bbox
                )
            ):
                raise ValueError("static review crop identity/bbox is invalid")
            x1, y1, x2, y2 = (int(value) for value in raw_bbox)
            if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
                raise ValueError("static review crop bbox is out of bounds")
            cause = str(raw_crop.get("cause") or "")
            if cause not in CAUSES:
                raise ValueError("static review crop cause is invalid")
            required_highres = raw_crop.get("required_highres") is True
            highres_count += int(required_highres)
            if required_highres:
                highres_regions.add((page_id, (x1, y1, x2, y2)))
            crop_numbers.add(crop_number)
            crops.append(
                CropSpec(
                    number=crop_number,
                    bbox=(x1, y1, x2, y2),
                    cause=cause,
                    required_highres=required_highres,
                )
            )
        if crop_numbers and crop_numbers != set(range(1, len(crop_numbers) + 1)):
            raise ValueError("static review crop numbers must be contiguous")
        if selected:
            selected_ids.append(page_id)
        pages.append(
            ReviewPage(
                number=number,
                page_id=page_id,
                filename=filename,
                selected_review=selected,
                causes=causes,
                source=source,
                pr6=pr6,
                finalists=tuple(finalists),
                masks=masks,
                crops=tuple(crops),
            )
        )

    if {page.number for page in pages} != set(range(1, expected_page_count + 1)):
        raise ValueError("static review page numbers must be contiguous")
    pages.sort(key=lambda page: page.number)
    if len(selected_ids) != expected_selected_count:
        raise ValueError("static review selected-page count differs")
    if highres_count != expected_highres_count:
        raise ValueError("static review required-highres count differs")
    if set(page_ids) != set(canonical_pages) or set(selected_ids) != canonical_selected:
        raise ValueError("static review selected pages differ from canonical inventory")
    if highres_regions != canonical_highres:
        raise ValueError("static review highres crops differ from canonical inventory")
    if payload.get("selected_page_ids") != sorted(selected_ids):
        raise ValueError("static review selected-page inventory differs")
    if payload.get("required_highres_count") != highres_count:
        raise ValueError("static review highres inventory differs")
    input_artifacts.sort(key=lambda row: (str(row["page_id"]), str(row["role"])))
    if payload.get("input_artifact_inventory_sha256") != _canonical_sha256(
        input_artifacts
    ):
        raise ValueError("static review input artifact inventory SHA differs")
    if seal.get("input_artifact_inventory_sha256") != payload.get(
        "input_artifact_inventory_sha256"
    ):
        raise ValueError("static review seal input artifact inventory differs")
    return ValidatedReviewLedger(
        ledger_path=ledger_path,
        ledger_file_sha256=ledger_file_sha,
        ledger_payload_sha256=payload_sha,
        seal_file_sha256=seal_file_sha,
        page_inventory_sha256=str(payload["page_inventory_sha256"]),
        pages=tuple(pages),
        input_artifact_inventory=tuple(input_artifacts),
    )


def _font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
    names = (
        "malgunbd.ttf" if bold else "malgun.ttf",
        "NotoSansCJK-Bold.ttc" if bold else "NotoSansCJK-Regular.ttc",
        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
    )
    system_paths = (
        Path("C:/Windows/Fonts"),
        Path("/usr/share/fonts/opentype/noto"),
        Path("/usr/share/fonts/truetype/dejavu"),
    )
    for name in names:
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            pass
        for root in system_paths:
            candidate = root / name
            if candidate.is_file():
                try:
                    return ImageFont.truetype(str(candidate), size=size)
                except OSError:
                    pass
    return ImageFont.load_default()


def _rgb(value: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(value, cv2.COLOR_BGR2RGB))


def _fit_image(
    image: Image.Image,
    *,
    max_width: int,
    max_height: int,
    upscale: bool = False,
) -> tuple[Image.Image, float, float]:
    width, height = image.size
    scale = min(max_width / width, max_height / height)
    if not upscale:
        scale = min(scale, 1.0)
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    resized = image.resize(size, Image.Resampling.LANCZOS)
    return resized, size[0] / width, size[1] / height


def _fit_text(draw: ImageDraw.ImageDraw, value: str, width: int, font) -> str:
    if draw.textbbox((0, 0), value, font=font)[2] <= width:
        return value
    suffix = "…"
    result = value
    while result and draw.textbbox((0, 0), result + suffix, font=font)[2] > width:
        result = result[:-1]
    return result + suffix


def _mark_crops(
    draw: ImageDraw.ImageDraw,
    crops: Sequence[CropSpec],
    *,
    offset_x: int,
    offset_y: int,
    scale_x: float,
    scale_y: float,
) -> None:
    marker_font = _font(20, bold=True)
    for crop in crops:
        x1, y1, x2, y2 = crop.bbox
        box = (
            offset_x + round(x1 * scale_x),
            offset_y + round(y1 * scale_y),
            offset_x + round(x2 * scale_x),
            offset_y + round(y2 * scale_y),
        )
        draw.rectangle(box, outline=(255, 45, 45), width=3)
        badge = (box[0], box[1], box[0] + 28, box[1] + 28)
        draw.ellipse(badge, fill=(220, 20, 20), outline="white", width=2)
        text = str(crop.number)
        bounds = draw.textbbox((0, 0), text, font=marker_font)
        draw.text(
            (
                badge[0] + (28 - (bounds[2] - bounds[0])) / 2,
                badge[1] + (28 - (bounds[3] - bounds[1])) / 2 - 2,
            ),
            text,
            font=marker_font,
            fill="white",
        )


def _write_pil_png(path: Path, image: Image.Image) -> dict[str, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"static review output must be fresh: {path}")
    temporary = path.with_name(f".{path.name}.partial")
    image.save(temporary, format="PNG", optimize=False)
    temporary.replace(path)
    decoded = _read_image(path, cv2.IMREAD_COLOR)
    return {
        "file_sha256": _sha256(path),
        "pixel_sha256": _array_sha256(decoded),
        "shape": list(decoded.shape),
        "dtype": str(decoded.dtype),
        "size_bytes": path.stat().st_size,
    }


def _comparison_board(page: ReviewPage) -> Image.Image:
    columns = [
        ("source", read_bound_artifact(page.source)),
        ("PR6", read_bound_artifact(page.pr6)),
        *(
            (finalist.label, read_bound_artifact(finalist.image))
            for finalist in page.finalists
        ),
    ]
    if len(columns) > 4:
        raise AssertionError("static review comparison exceeded four columns")
    column_width = 620
    content_height = 980
    header_height = 86
    footer_height = 42
    gap = 10
    board = Image.new(
        "RGB",
        (
            len(columns) * column_width + (len(columns) - 1) * gap,
            header_height + content_height + footer_height,
        ),
        "white",
    )
    draw = ImageDraw.Draw(board)
    title_font = _font(24, bold=True)
    label_font = _font(20, bold=True)
    cause_font = _font(18)
    title = f"{page.number:03d}  {page.filename}"
    draw.text((12, 8), _fit_text(draw, title, board.width - 24, title_font), fill="black", font=title_font)
    for index, (label, value) in enumerate(columns):
        x = index * (column_width + gap)
        draw.rectangle(
            (x, header_height - 34, x + column_width - 1, header_height - 1),
            fill=(240, 243, 247),
        )
        draw.text((x + 10, header_height - 31), label, fill="black", font=label_font)
        fitted, scale_x, scale_y = _fit_image(
            _rgb(value), max_width=column_width, max_height=content_height
        )
        paste_x = x + (column_width - fitted.width) // 2
        paste_y = header_height + (content_height - fitted.height) // 2
        board.paste(fitted, (paste_x, paste_y))
        _mark_crops(
            draw,
            page.crops,
            offset_x=paste_x,
            offset_y=paste_y,
            scale_x=scale_x,
            scale_y=scale_y,
        )
        draw.rectangle(
            (x, header_height, x + column_width - 1, header_height + content_height - 1),
            outline=(185, 190, 200),
            width=1,
        )
    causes = " / ".join(f"{crop.number}: {crop.cause}" for crop in page.crops)
    if not causes:
        causes = " / ".join(page.causes)
    draw.text(
        (12, header_height + content_height + 8),
        _fit_text(draw, causes, board.width - 24, cause_font),
        fill=(75, 75, 75),
        font=cause_font,
    )
    return board


def _combined_mask_overlay(page: ReviewPage) -> np.ndarray:
    source = cv2.cvtColor(read_bound_artifact(page.source), cv2.COLOR_BGR2RGB)
    result = source.astype(np.float32)
    # Safety masks are applied last so red protect and blue preserve remain
    # recognizable where masks overlap.
    for role in ("core", "effect", "preserve", "protect"):
        selected = read_bound_artifact(page.masks[role]) > 0
        if not np.any(selected):
            continue
        color = np.asarray(MASK_COLORS_RGB[role], dtype=np.float32)
        result[selected] = result[selected] * 0.38 + color * 0.62
    return np.clip(np.round(result), 0, 255).astype(np.uint8)


def _mask_board(page: ReviewPage) -> Image.Image:
    source = _rgb(read_bound_artifact(page.source))
    overlay = Image.fromarray(_combined_mask_overlay(page))
    column_width = 700
    content_height = 1050
    header_height = 112
    gap = 12
    board = Image.new(
        "RGB", (column_width * 2 + gap, header_height + content_height), "white"
    )
    draw = ImageDraw.Draw(board)
    title_font = _font(23, bold=True)
    legend_font = _font(18, bold=True)
    draw.text(
        (12, 8),
        _fit_text(
            draw,
            f"{page.number:03d}  {page.filename}  mask evidence",
            board.width - 24,
            title_font,
        ),
        fill="black",
        font=title_font,
    )
    legend_x = 12
    for role in MASK_ROLES:
        color = MASK_COLORS_RGB[role]
        draw.rectangle((legend_x, 49, legend_x + 22, 71), fill=color)
        draw.text((legend_x + 28, 49), role, fill="black", font=legend_font)
        legend_x += 155
    for index, (label, image) in enumerate((("source", source), ("overlay", overlay))):
        x = index * (column_width + gap)
        draw.text((x + 10, 82), label, fill="black", font=legend_font)
        fitted, scale_x, scale_y = _fit_image(
            image, max_width=column_width, max_height=content_height
        )
        paste_x = x + (column_width - fitted.width) // 2
        paste_y = header_height + (content_height - fitted.height) // 2
        board.paste(fitted, (paste_x, paste_y))
        _mark_crops(
            draw,
            page.crops,
            offset_x=paste_x,
            offset_y=paste_y,
            scale_x=scale_x,
            scale_y=scale_y,
        )
    return board


def _highres_crop_board(page: ReviewPage, crop: CropSpec) -> Image.Image:
    images = [
        ("source", read_bound_artifact(page.source)),
        ("PR6", read_bound_artifact(page.pr6)),
        *(
            (finalist.label, read_bound_artifact(finalist.image))
            for finalist in page.finalists
        ),
    ]
    if len(images) > 4:
        raise AssertionError("high-resolution crop exceeded four columns")
    x1, y1, x2, y2 = crop.bbox
    crop_width = x2 - x1
    crop_height = y2 - y1
    column_width = max(crop_width, 190)
    header_height = 72
    footer_height = 38
    gap = 10
    board = Image.new(
        "RGB",
        (
            len(images) * column_width + (len(images) - 1) * gap,
            header_height + crop_height + footer_height,
        ),
        "white",
    )
    draw = ImageDraw.Draw(board)
    label_font = _font(18, bold=True)
    cause_font = _font(17)
    for index, (label, value) in enumerate(images):
        x = index * (column_width + gap)
        panel = _rgb(value[y1:y2, x1:x2])
        # No resize: every source/candidate crop pixel remains at 1:1 scale.
        paste_x = x + (column_width - crop_width) // 2
        board.paste(panel, (paste_x, header_height))
        draw.text((x + 8, 42), label, fill="black", font=label_font)
        draw.rectangle(
            (paste_x, header_height, paste_x + crop_width - 1, header_height + crop_height - 1),
            outline=(170, 175, 185),
            width=1,
        )
    title = f"{page.number:03d} / crop {crop.number} / native 1:1"
    draw.text((8, 8), title, fill="black", font=label_font)
    draw.text(
        (8, header_height + crop_height + 8),
        crop.cause,
        fill=(70, 70, 70),
        font=cause_font,
    )
    return board


def _index_board(pages: Sequence[ReviewPage]) -> Image.Image:
    columns = 5
    cell_width = 300
    cell_height = 235
    header_height = 78
    rows = (len(pages) + columns - 1) // columns
    board = Image.new(
        "RGB", (columns * cell_width, header_height + rows * cell_height), "white"
    )
    draw = ImageDraw.Draw(board)
    title_font = _font(28, bold=True)
    meta_font = _font(16, bold=True)
    cause_font = _font(15)
    draw.text(
        (14, 15),
        f"Inpaint v3.4 static review index — {len(pages)} pages",
        fill="black",
        font=title_font,
    )
    for offset, page in enumerate(pages):
        row, column = divmod(offset, columns)
        x = column * cell_width
        y = header_height + row * cell_height
        selected_color = (30, 115, 210) if page.selected_review else (195, 200, 208)
        draw.rectangle(
            (x + 3, y + 3, x + cell_width - 4, y + cell_height - 4),
            outline=selected_color,
            width=3 if page.selected_review else 1,
        )
        source = _rgb(read_bound_artifact(page.source))
        thumbnail, _sx, _sy = _fit_image(
            source, max_width=282, max_height=158, upscale=False
        )
        board.paste(
            thumbnail,
            (x + (cell_width - thumbnail.width) // 2, y + 8),
        )
        label = f"{page.number:03d}  {page.filename}"
        draw.text(
            (x + 9, y + 171),
            _fit_text(draw, label, cell_width - 18, meta_font),
            fill="black",
            font=meta_font,
        )
        cause = " / ".join(page.causes)
        draw.text(
            (x + 9, y + 199),
            _fit_text(draw, cause, cell_width - 18, cause_font),
            fill=(85, 85, 85),
            font=cause_font,
        )
    return board


def _output_record(
    *,
    output_root: Path,
    path: Path,
    role: str,
    metadata: Mapping[str, object],
    page: ReviewPage | None = None,
    crop: CropSpec | None = None,
) -> dict[str, object]:
    if path.suffix.lower() not in OUTPUT_IMAGE_SUFFIXES:
        raise ValueError("static review output must be PNG or WebP")
    row: dict[str, object] = {
        "role": role,
        "relative_path": path.relative_to(output_root).as_posix(),
        **metadata,
    }
    if page is not None:
        row["page_id"] = page.page_id
        row["page_number"] = page.number
    if crop is not None:
        row["crop_number"] = crop.number
        row["bbox"] = list(crop.bbox)
        row["cause"] = crop.cause
        row["native_scale"] = "1:1"
    return row


def validate_output_manifest(
    payload: Mapping[str, object], *, output_root: Path
) -> None:
    unsigned = {
        key: value for key, value in payload.items() if key != "output_sha256"
    }
    if payload.get("output_sha256") != _canonical_sha256(unsigned):
        raise ValueError("static review output manifest SHA differs")
    if payload.get("schema_version") != OUTPUT_SCHEMA_VERSION:
        raise ValueError("static review output schema differs")
    records = payload.get("artifacts")
    if not isinstance(records, list) or not records:
        raise ValueError("static review output artifact inventory is empty")
    identities: set[str] = set()
    for row in records:
        if not isinstance(row, Mapping):
            raise ValueError("static review output artifact row is invalid")
        relative = Path(str(row.get("relative_path") or ""))
        key = relative.as_posix()
        if (
            not relative.parts
            or relative.is_absolute()
            or ".." in relative.parts
            or relative.suffix.lower() not in OUTPUT_IMAGE_SUFFIXES
            or key in identities
        ):
            raise ValueError("static review output artifact path is invalid")
        identities.add(key)
        path = (output_root / relative).resolve()
        try:
            path.relative_to(output_root.resolve())
        except ValueError as error:
            raise ValueError("static review output artifact escapes its run") from error
        if _sha256(path) != row.get("file_sha256"):
            raise ValueError("static review output artifact file SHA differs")
        value = _read_image(path, cv2.IMREAD_COLOR)
        if (
            list(value.shape) != row.get("shape")
            or str(value.dtype) != row.get("dtype")
            or _array_sha256(value) != row.get("pixel_sha256")
            or path.stat().st_size != row.get("size_bytes")
        ):
            raise ValueError("static review output artifact pixels differ")
    roles = [str(row.get("role") or "") for row in records if isinstance(row, Mapping)]
    if roles.count("page_index") != 1:
        raise ValueError("static review output requires exactly one page index")


def _assert_inputs_unchanged(ledger: ValidatedReviewLedger) -> None:
    if (
        _sha256(ledger.ledger_path) != ledger.ledger_file_sha256
        or _sha256(ledger.ledger_path.with_suffix(ledger.ledger_path.suffix + ".seal.json"))
        != ledger.seal_file_sha256
    ):
        raise RuntimeError("static review ledger changed during rendering")
    seen: set[Path] = set()
    references: Iterable[ArtifactRef] = (
        reference
        for page in ledger.pages
        for reference in (
            page.source,
            page.pr6,
            *(finalist.image for finalist in page.finalists),
            *(page.masks[role] for role in MASK_ROLES),
        )
    )
    for reference in references:
        if reference.path in seen:
            continue
        seen.add(reference.path)
        if _sha256(reference.path) != reference.file_sha256:
            raise RuntimeError("static review input artifact changed during rendering")


def build_static_review(
    *,
    ledger_path: Path,
    source_inventory_path: Path,
    output_root: Path,
    expected_page_count: int = 130,
    expected_selected_count: int = 12,
    expected_highres_count: int = 9,
) -> dict[str, object]:
    ledger = validate_review_ledger(
        ledger_path,
        source_inventory_path=source_inventory_path,
        expected_page_count=expected_page_count,
        expected_selected_count=expected_selected_count,
        expected_highres_count=expected_highres_count,
    )
    output_root.mkdir(parents=True, exist_ok=True)
    if any(output_root.iterdir()):
        raise FileExistsError("static review output directory must be fresh")
    artifacts: list[dict[str, object]] = []

    index_path = output_root / "index" / "000-all-pages-index.png"
    artifacts.append(
        _output_record(
            output_root=output_root,
            path=index_path,
            role="page_index",
            metadata=_write_pil_png(index_path, _index_board(ledger.pages)),
        )
    )
    for page in ledger.pages:
        if not page.selected_review:
            continue
        prefix = f"{page.number:03d}-{page.page_id}"
        comparison_path = output_root / "selected" / f"{prefix}-comparison.png"
        artifacts.append(
            _output_record(
                output_root=output_root,
                path=comparison_path,
                role="page_comparison",
                metadata=_write_pil_png(comparison_path, _comparison_board(page)),
                page=page,
            )
        )
        mask_path = output_root / "selected" / f"{prefix}-masks.png"
        artifacts.append(
            _output_record(
                output_root=output_root,
                path=mask_path,
                role="mask_board",
                metadata=_write_pil_png(mask_path, _mask_board(page)),
                page=page,
            )
        )
        for crop in page.crops:
            if not crop.required_highres:
                continue
            highres_path = (
                output_root
                / "highres"
                / f"{prefix}-crop-{crop.number:02d}.png"
            )
            artifacts.append(
                _output_record(
                    output_root=output_root,
                    path=highres_path,
                    role="highres_crop",
                    metadata=_write_pil_png(
                        highres_path, _highres_crop_board(page, crop)
                    ),
                    page=page,
                    crop=crop,
                )
            )
    _assert_inputs_unchanged(ledger)
    artifacts.sort(key=lambda row: str(row["relative_path"]))
    selected_count = sum(page.selected_review for page in ledger.pages)
    highres_count = sum(
        crop.required_highres for page in ledger.pages for crop in page.crops
    )
    manifest_unsigned: dict[str, object] = {
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "source_ledger_file_sha256": ledger.ledger_file_sha256,
        "source_ledger_payload_sha256": ledger.ledger_payload_sha256,
        "source_ledger_seal_file_sha256": ledger.seal_file_sha256,
        "source_page_inventory_sha256": ledger.page_inventory_sha256,
        "input_artifact_inventory_sha256": _canonical_sha256(
            list(ledger.input_artifact_inventory)
        ),
        "page_count": len(ledger.pages),
        "selected_page_count": selected_count,
        "required_highres_count": highres_count,
        "html_generated": False,
        "javascript_generated": False,
        "mask_colors_rgb": {
            role: list(color) for role, color in MASK_COLORS_RGB.items()
        },
        "artifacts": artifacts,
    }
    manifest = dict(manifest_unsigned)
    manifest["output_sha256"] = _canonical_sha256(manifest_unsigned)
    validate_output_manifest(manifest, output_root=output_root)
    manifest_path = output_root / "static-review-output-manifest.json"
    _atomic_json(manifest_path, manifest)
    seal = {
        "schema_version": OUTPUT_SEAL_SCHEMA_VERSION,
        "manifest_file_sha256": _sha256(manifest_path),
        "output_sha256": manifest["output_sha256"],
        "artifact_inventory_sha256": _canonical_sha256(artifacts),
        "html_generated": False,
        "javascript_generated": False,
    }
    seal_path = manifest_path.with_suffix(manifest_path.suffix + ".seal.json")
    _atomic_json(seal_path, seal)
    _assert_inputs_unchanged(ledger)
    return {
        **manifest,
        "manifest_relative_path": manifest_path.relative_to(output_root).as_posix(),
        "manifest_file_sha256": _sha256(manifest_path),
        "seal_relative_path": seal_path.relative_to(output_root).as_posix(),
        "seal_file_sha256": _sha256(seal_path),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build sealed static PNG review boards for inpaint v3.4."
    )
    parser.add_argument("--review-ledger", type=Path, required=True)
    parser.add_argument("--source-inventory", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_root, managed = select_managed_output_directory(
        family=FAMILY,
        category=CATEGORY,
        explicit_output_directory=args.output_dir,
    )
    try:
        result = build_static_review(
            ledger_path=args.review_ledger.resolve(),
            source_inventory_path=args.source_inventory.resolve(),
            output_root=output_root,
        )
        if managed is not None:
            managed.complete(
                metadata={
                    "page_count": result["page_count"],
                    "selected_page_count": result["selected_page_count"],
                    "required_highres_count": result["required_highres_count"],
                    "output_sha256": result["output_sha256"],
                }
            )
            mismatches = managed.verify()
            if mismatches:
                raise RuntimeError(
                    "managed artifact verification failed: " + "; ".join(mismatches)
                )
            print(managed.run_root)
        else:
            print(output_root / "static-review-output-manifest.json")
        return 0
    except BaseException as error:
        if managed is not None:
            managed.fail(error)
        raise


if __name__ == "__main__":
    raise SystemExit(main())

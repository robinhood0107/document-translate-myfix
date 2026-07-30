#!/usr/bin/env python3
"""Frozen inpaint replay and blind-review protocol for difficult manga regions.

This tool intentionally keeps source images, annotations, raw masks, candidate
outputs, timings, and the blind key outside Git.  It has five independent
operations:

* ``capture`` freezes source/page-snapshot inputs and one CTD inference.
* ``run`` replays either the mask or model screen without rerunning OCR.
* ``attach-renders`` copies externally produced full-pipeline renders into a
  new immutable result contract.
* ``blind`` creates a candidate-name/timing-free visual review bundle.
* ``validate-review`` / ``unblind`` enforce complete review before disclosure.

Real page names, coordinates, and source strings belong only in the external
case manifest.  Nothing in this module is tailored to a particular page.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import os
import secrets
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


PROTOCOL_VERSION = 1
CAPTURE_FILENAME = "frozen_inpaint_contract.json"
RESULT_FILENAME = "inpaint_quality_results.json"
STATE_FILENAME = "review_state.json"
REVIEW_FILENAME = "blind_review.csv"
REVIEW_HTML_FILENAME = "blind_review.html"
PRIVATE_DIRNAME = "private"
PRIVATE_KEY_FILENAME = "blind_key.json"
PRIVATE_PAYLOAD_FILENAME = "blind_payload.json"
UNBLIND_FILENAME = "unblind_summary.json"
RENDER_MANIFEST_KIND = "inpaint-quality-render-attachments"

VALID_ROLES = frozenset(
    {
        "dialogue_bubble",
        "dialogue_free",
        "narration",
        "ui_or_sign",
        "sfx",
        "decorative",
        "ambiguous",
    }
)
VALID_ACTIONS = frozenset({"translate_inpaint", "preserve", "review"})
PROMOTION_REVIEW_FIELDS = (
    "residue",
    "structure",
    "outside_preservation",
    "hallucination",
    "render",
)
REVIEW_VALUES = frozenset({"pass", "fail", "na"})


class ProtocolError(ValueError):
    """Raised when a frozen/result/review contract is incomplete or altered."""


class ReviewIncompleteError(ProtocolError):
    """Raised when an unblind is attempted before every review field is filled."""


class CandidateRunError(RuntimeError):
    """Raised when a candidate inference fails with structured diagnostics."""

    def __init__(self, message: str, *, diagnostics: Mapping[str, Any]) -> None:
        super().__init__(message)
        self.diagnostics = dict(diagnostics)


@dataclass(frozen=True)
class CandidateProfile:
    slug: str
    label: str
    phase: str
    inpainter_key: str
    precision: str
    inpaint_size: int
    mask_mode: str
    dilation: int | None
    structure_protect: bool
    baseline: bool = False
    promotable: bool = True
    feasibility_only: bool = False


BASELINE = CandidateProfile(
    slug="baseline-lama-large-bf16-1536-product-mask",
    label="LaMa Large BF16 1536 current-mask baseline",
    phase="baseline",
    inpainter_key="lama_large_512px",
    precision="bf16",
    inpaint_size=1536,
    mask_mode="product",
    dilation=None,
    structure_protect=False,
    baseline=True,
    promotable=False,
)

MASK_SCREEN_PROFILES: tuple[CandidateProfile, ...] = tuple(
    CandidateProfile(
        slug=f"mask-dilate-{dilation}-lama-large-fp32-2048",
        label=f"glyph mask dilation {dilation} + LaMa Large FP32 2048",
        phase="mask",
        inpainter_key="lama_large_512px",
        precision="fp32",
        inpaint_size=2048,
        mask_mode="glyph",
        dilation=dilation,
        structure_protect=True,
    )
    for dilation in (1, 2, 4)
)

MODEL_SCREEN_TEMPLATES: tuple[CandidateProfile, ...] = (
    CandidateProfile(
        slug="model-lama-large-fp32-1536",
        label="LaMa Large FP32 1536",
        phase="model",
        inpainter_key="lama_large_512px",
        precision="fp32",
        inpaint_size=1536,
        mask_mode="glyph",
        dilation=None,
        structure_protect=True,
    ),
    CandidateProfile(
        slug="model-lama-large-fp32-2048",
        label="LaMa Large FP32 2048",
        phase="model",
        inpainter_key="lama_large_512px",
        precision="fp32",
        inpaint_size=2048,
        mask_mode="glyph",
        dilation=None,
        structure_protect=True,
    ),
    CandidateProfile(
        slug="model-lama-mpe-fp32-2048",
        label="LaMa MPE FP32 2048",
        phase="model",
        inpainter_key="lama_mpe",
        precision="fp32",
        inpaint_size=2048,
        mask_mode="glyph",
        dilation=None,
        structure_protect=True,
    ),
    CandidateProfile(
        slug="model-aot-fp32-2048",
        label="AOT FP32 2048",
        phase="model",
        inpainter_key="AOT",
        precision="fp32",
        inpaint_size=2048,
        mask_mode="glyph",
        dilation=None,
        structure_protect=True,
    ),
    CandidateProfile(
        slug="model-zits-fp32-feasibility",
        label="ZITS/ZITS++ GPU FP32 feasibility",
        phase="model",
        inpainter_key="ZITS",
        precision="fp32",
        inpaint_size=2048,
        mask_mode="glyph",
        dilation=None,
        structure_protect=True,
        promotable=False,
        feasibility_only=True,
    ),
)


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict):
        raise ProtocolError(f"Expected a JSON object: {path}")
    return payload


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _ensure_new_output_dir(path: Path) -> Path:
    output = path.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Output directory already exists: {output}")
    try:
        relative = output.relative_to(ROOT.resolve())
    except ValueError:
        pass
    else:
        if not relative.parts or relative.parts[0] == ".git":
            raise ProtocolError("Benchmark output cannot be inside .git")
        check = subprocess.run(
            ["git", "check-ignore", "--quiet", "--", relative.as_posix()],
            cwd=ROOT,
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if check.returncode != 0:
            raise ProtocolError(
                "Benchmark output inside the repository must be ignored by Git"
            )
    output.mkdir(parents=True)
    return output


def _read_image(path: Path):
    import cv2
    import numpy as np

    data = np.fromfile(str(path), dtype=np.uint8)
    image_bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ProtocolError(f"Unable to decode image: {path}")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def _write_image(path: Path, image: Any) -> None:
    import cv2
    import numpy as np

    array = np.asarray(image)
    if array.ndim == 3 and array.shape[2] >= 3:
        encoded_source = cv2.cvtColor(array[:, :, :3], cv2.COLOR_RGB2BGR)
    else:
        encoded_source = array
    suffix = path.suffix.lower()
    extension = suffix if suffix in {".png", ".jpg", ".jpeg", ".webp"} else ".png"
    ok, encoded = cv2.imencode(extension, encoded_source)
    if not ok:
        raise RuntimeError(f"Unable to encode image: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded.tofile(str(path))


def _copy_lossless_image(source: Path, target: Path) -> None:
    image = _read_image(source)
    _write_image(target.with_suffix(".png"), image)


def _normalize_box(
    value: Any,
    shape: Sequence[int],
    *,
    label: str,
) -> list[int]:
    try:
        x1, y1, x2, y2 = [int(round(float(item))) for item in list(value)[:4]]
    except (TypeError, ValueError):
        raise ProtocolError(f"{label} must contain four numeric coordinates")
    height, width = int(shape[0]), int(shape[1])
    x1 = max(0, min(width, x1))
    x2 = max(0, min(width, x2))
    y1 = max(0, min(height, y1))
    y2 = max(0, min(height, y2))
    if x2 <= x1 or y2 <= y1:
        raise ProtocolError(f"{label} is empty after image clipping")
    return [x1, y1, x2, y2]


def _snapshot_page(
    payload: Mapping[str, Any],
    *,
    page_name: str,
) -> Mapping[str, Any]:
    pages = payload.get("pages")
    if not isinstance(pages, list) or not pages:
        raise ProtocolError("page_snapshots.json must contain a non-empty pages list")
    matches = [
        page
        for page in pages
        if isinstance(page, Mapping)
        and str(page.get("image_name") or "") == page_name
    ]
    if len(matches) == 1:
        return matches[0]
    if len(pages) == 1 and isinstance(pages[0], Mapping):
        return pages[0]
    raise ProtocolError(
        f"Snapshot page selection is ambiguous or missing: {page_name}"
    )


def _default_role(text_class: str) -> str:
    if text_class == "text_bubble":
        return "dialogue_bubble"
    if text_class == "text_free":
        return "ambiguous"
    return "ambiguous"


def _default_action(block: Mapping[str, Any]) -> str:
    if (
        str(block.get("text_class") or "") == "text_bubble"
        and str(block.get("translation") or "").strip()
    ):
        return "translate_inpaint"
    return "review"


def _case_annotations(
    case: Mapping[str, Any],
    block_count: int,
) -> dict[int, dict[str, str]]:
    annotations: dict[int, dict[str, str]] = {}
    raw = case.get("annotations") or []
    if not isinstance(raw, list):
        raise ProtocolError("case annotations must be a list")
    for item in raw:
        if not isinstance(item, Mapping):
            raise ProtocolError("each case annotation must be an object")
        index = int(item.get("block_index", -1))
        if index < 0 or index >= block_count:
            raise ProtocolError(f"annotation block_index is out of range: {index}")
        if index in annotations:
            raise ProtocolError(f"duplicate annotation for block_index={index}")
        role = str(item.get("semantic_role") or "")
        action = str(item.get("processing_action") or "")
        if role not in VALID_ROLES:
            raise ProtocolError(f"invalid semantic_role for block {index}: {role}")
        if action not in VALID_ACTIONS:
            raise ProtocolError(
                f"invalid processing_action for block {index}: {action}"
            )
        annotations[index] = {
            "semantic_role": role,
            "processing_action": action,
        }
    return annotations


def _snapshot_block_records(
    *,
    case_id: str,
    page: Mapping[str, Any],
    case: Mapping[str, Any],
    image_shape: Sequence[int],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from modules.ocr.result_contract import canonicalize_exact_duplicate_blocks
    from modules.utils.textblock import TextBlock

    raw_blocks = page.get("blocks")
    if not isinstance(raw_blocks, list):
        raise ProtocolError(f"snapshot blocks missing for case {case_id}")
    annotations = _case_annotations(case, len(raw_blocks))
    blocks: list[TextBlock] = []
    original_records: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_blocks):
        if not isinstance(raw, Mapping):
            raise ProtocolError(f"case {case_id} block {index} is not an object")
        xyxy = _normalize_box(
            raw.get("xyxy"),
            image_shape,
            label=f"case {case_id} block {index} xyxy",
        )
        bubble_raw = raw.get("bubble_xyxy")
        bubble_xyxy = (
            _normalize_box(
                bubble_raw,
                image_shape,
                label=f"case {case_id} block {index} bubble_xyxy",
            )
            if bubble_raw is not None
            else None
        )
        text_class = str(raw.get("text_class") or "")
        annotation = annotations.get(
            index,
            {
                "semantic_role": _default_role(text_class),
                "processing_action": _default_action(raw),
            },
        )
        block_id = canonical_sha256(
            {
                "case_id": case_id,
                "index": index,
                "xyxy": xyxy,
                "bubble_xyxy": bubble_xyxy,
                "text": str(raw.get("text") or ""),
            }
        )[:32]
        block = TextBlock(
            block_id=block_id,
            text_bbox=xyxy,
            bubble_bbox=bubble_xyxy,
            text_class=text_class,
            angle=float(raw.get("angle", 0.0) or 0.0),
            text=str(raw.get("text") or ""),
            translation=str(raw.get("translation") or ""),
        )
        block.semantic_role = annotation["semantic_role"]
        block.processing_action = annotation["processing_action"]
        blocks.append(block)
        original_records.append(
            {
                "source_index": index,
                "block_id": block_id,
                "xyxy": xyxy,
                "bubble_xyxy": bubble_xyxy,
                "text_class": text_class,
                "text": block.text,
                "translation": block.translation,
                **annotation,
            }
        )

    canonical, summary = canonicalize_exact_duplicate_blocks(blocks)
    original_by_id = {
        str(record["block_id"]): record for record in original_records
    }
    canonical_records: list[dict[str, Any]] = []
    for block in canonical:
        canonical_annotation = (
            str(getattr(block, "semantic_role", "ambiguous") or "ambiguous"),
            str(getattr(block, "processing_action", "review") or "review"),
        )
        for alias_id in list(
            getattr(block, "duplicate_alias_block_ids", []) or []
        ):
            alias_record = original_by_id.get(str(alias_id))
            if alias_record is None:
                raise ProtocolError(
                    "canonical duplicate alias is missing from source records"
                )
            alias_annotation = (
                str(alias_record["semantic_role"]),
                str(alias_record["processing_action"]),
            )
            if alias_annotation != canonical_annotation:
                raise ProtocolError(
                    "exact duplicate annotations disagree for case "
                    f"{case_id}"
                )
        canonical_records.append(
            {
                "block_id": str(block.block_id),
                "canonical_block_id": str(
                    getattr(block, "canonical_block_id", block.block_id)
                ),
                "duplicate_alias_block_ids": list(
                    getattr(block, "duplicate_alias_block_ids", []) or []
                ),
                "duplicate_alias_count": int(
                    getattr(block, "duplicate_alias_count", 0) or 0
                ),
                "xyxy": [int(value) for value in block.xyxy],
                "bubble_xyxy": (
                    [int(value) for value in block.bubble_xyxy]
                    if block.bubble_xyxy is not None
                    else None
                ),
                "text_class": str(block.text_class or ""),
                "text": str(block.text or ""),
                "translation": str(block.translation or ""),
                "semantic_role": str(
                    getattr(block, "semantic_role", "ambiguous") or "ambiguous"
                ),
                "processing_action": str(
                    getattr(block, "processing_action", "review") or "review"
                ),
            }
        )
    return canonical_records, {
        "input_block_count": len(original_records),
        "canonical_block_count": len(canonical_records),
        "duplicate_alias_count": sum(
            int(item.get("duplicate_alias_count", 0) or 0)
            for item in canonical_records
        ),
        "canonicalization": summary,
        "original_records_sha256": canonical_sha256(original_records),
    }


def _records_to_blocks(records: Iterable[Mapping[str, Any]]) -> list[Any]:
    from modules.utils.textblock import TextBlock

    blocks = []
    for record in records:
        action = str(record.get("processing_action") or "")
        role = str(record.get("semantic_role") or "")
        if action not in VALID_ACTIONS or role not in VALID_ROLES:
            raise ProtocolError("frozen block has an invalid role/action")
        block = TextBlock(
            block_id=str(record.get("block_id") or ""),
            text_bbox=list(record.get("xyxy") or []),
            bubble_bbox=(
                list(record.get("bubble_xyxy") or [])
                if record.get("bubble_xyxy") is not None
                else None
            ),
            text_class=str(record.get("text_class") or ""),
            text=str(record.get("text") or ""),
            translation=str(record.get("translation") or ""),
        )
        block.canonical_block_id = str(
            record.get("canonical_block_id") or block.block_id
        )
        block.duplicate_alias_block_ids = list(
            record.get("duplicate_alias_block_ids") or []
        )
        block.duplicate_alias_count = int(
            record.get("duplicate_alias_count", 0) or 0
        )
        block.semantic_role = role
        block.processing_action = action
        blocks.append(block)
    return blocks


def _allowed_window_mask(image_shape: Sequence[int], blocks: Sequence[Any]):
    import numpy as np

    from modules.utils.mask_roi import resolve_block_ctd_roi

    mask = np.zeros(tuple(image_shape[:2]), dtype=np.uint8)
    for block in blocks:
        if str(getattr(block, "processing_action", "") or "") != "translate_inpaint":
            continue
        roi = resolve_block_ctd_roi(block, tuple(image_shape))
        if roi is None:
            continue
        x1, y1, x2, y2 = roi
        mask[y1:y2, x1:x2] = 255
    return mask


def _structure_protect_mask(image: Any, blocks: Sequence[Any]):
    """Experimental Hough-line protection, scoped to translate ROIs only."""

    import cv2
    import numpy as np

    from modules.utils.mask_roi import resolve_block_ctd_roi

    output = np.zeros(image.shape[:2], dtype=np.uint8)
    for block in blocks:
        if str(getattr(block, "processing_action", "") or "") != "translate_inpaint":
            continue
        roi = resolve_block_ctd_roi(block, image.shape)
        if roi is None:
            continue
        x1, y1, x2, y2 = roi
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 40, 120)
        minimum = max(28, int(round(min(gray.shape[:2]) * 0.18)))
        lines = cv2.HoughLinesP(
            edges,
            1,
            np.pi / 180.0,
            threshold=max(18, minimum // 2),
            minLineLength=minimum,
            maxLineGap=6,
        )
        if lines is None:
            continue
        local = np.zeros(gray.shape, dtype=np.uint8)
        for x_start, y_start, x_end, y_end in lines[:, 0, :]:
            length = float(
                np.hypot(int(x_end) - int(x_start), int(y_end) - int(y_start))
            )
            if length < minimum:
                continue
            cv2.line(
                local,
                (int(x_start), int(y_start)),
                (int(x_end), int(y_end)),
                255,
                thickness=4,
                lineType=cv2.LINE_AA,
            )
        output[y1:y2, x1:x2] = cv2.bitwise_or(
            output[y1:y2, x1:x2],
            local,
        )
    return (output > 0).astype("uint8") * 255


def _capture_masks(image: Any, blocks: Sequence[Any]) -> dict[str, Any]:
    import numpy as np

    from modules.masking import (
        CTDRefiner,
        CTDRefinerSettings,
        ProtectMaskSettings,
        build_protect_mask,
    )
    from modules.utils.image_utils import (
        _dilate_ctd_final_mask_by_block_policy,
    )

    active = [
        block
        for block in blocks
        if str(getattr(block, "processing_action", "") or "") == "translate_inpaint"
    ]
    if not active:
        raise ProtocolError("frozen case has no translate_inpaint blocks")
    refiner = CTDRefiner(
        CTDRefinerSettings(
            detect_size=1280,
            det_rearrange_max_batches=4,
            device="cuda",
            font_size_multiplier=1.0,
            font_size_max=-1,
            font_size_min=-1,
            mask_dilate_size=2,
        )
    )
    refined = refiner.refine(image, active)
    bubble_protect = build_protect_mask(
        image,
        active,
        ProtectMaskSettings(keep_existing_lines=True),
    )
    structure_protect = _structure_protect_mask(image, active)
    allowed = _allowed_window_mask(image.shape, active)
    glyph_base = np.where(
        (np.asarray(refined.final_mask_pre_expand) > 0)
        & (np.asarray(bubble_protect) <= 0)
        & (allowed > 0),
        255,
        0,
    ).astype(np.uint8)
    ctd_or_mask = np.where(
        (np.asarray(refined.raw_mask) > 0)
        | (np.asarray(refined.refined_mask) > 0)
        | (np.asarray(refined.final_mask) > 0),
        255,
        0,
    ).astype(np.uint8)
    product_base = np.where(
        (ctd_or_mask > 0) & (np.asarray(bubble_protect) <= 0),
        255,
        0,
    ).astype(np.uint8)
    if not np.any(product_base) and np.any(ctd_or_mask):
        product_base = ctd_or_mask.copy()
    product_mask, _text_free_dilate_count = (
        _dilate_ctd_final_mask_by_block_policy(
            product_base,
            image.shape,
            active,
            final_dilate_size=8,
            text_free_dilate_size=1,
        )
    )
    product_mask = np.where(
        (np.asarray(product_mask) > 0) & (allowed > 0),
        255,
        0,
    ).astype(np.uint8)
    return {
        "raw_mask": np.asarray(refined.raw_mask),
        "refined_mask": np.asarray(refined.refined_mask),
        "glyph_base_mask": glyph_base,
        "bubble_protect_mask": np.asarray(bubble_protect),
        "structure_protect_mask": np.asarray(structure_protect),
        "allowed_window_mask": allowed,
        "product_final_mask": product_mask,
        "refiner_backend": str(refined.backend or ""),
        "refiner_device": str(refined.device or ""),
        "refiner_fallback_used": bool(refined.fallback_used),
        "product_mask_pixel_count": int(np.count_nonzero(product_mask)),
        "glyph_base_mask_pixel_count": int(np.count_nonzero(glyph_base)),
    }


def capture_cases(manifest_path: Path, output_dir: Path) -> dict[str, Any]:
    manifest_path = manifest_path.expanduser().resolve()
    source_manifest = read_json(manifest_path)
    if int(source_manifest.get("protocol_version", 0) or 0) != PROTOCOL_VERSION:
        raise ProtocolError("case manifest protocol_version is unsupported")
    raw_cases = source_manifest.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise ProtocolError("case manifest must contain at least one case")
    output = _ensure_new_output_dir(output_dir)
    frozen_cases: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    for order, raw_case in enumerate(raw_cases):
        if not isinstance(raw_case, Mapping):
            raise ProtocolError("every case must be an object")
        case_id = str(raw_case.get("case_id") or "").strip()
        if not case_id or case_id in seen_ids:
            raise ProtocolError(f"case_id is empty or duplicated: {case_id!r}")
        seen_ids.add(case_id)
        source_path = Path(str(raw_case.get("source_image") or "")).expanduser().resolve()
        snapshot_path = Path(
            str(raw_case.get("page_snapshot") or "")
        ).expanduser().resolve()
        if not source_path.is_file():
            raise FileNotFoundError(f"source image is missing: {source_path}")
        if not snapshot_path.is_file():
            raise FileNotFoundError(f"page snapshot is missing: {snapshot_path}")
        expected_source_sha = str(raw_case.get("source_sha256") or "").lower()
        actual_source_sha = sha256_file(source_path)
        if expected_source_sha and expected_source_sha != actual_source_sha:
            raise ProtocolError(f"source SHA-256 differs for case {case_id}")
        image = _read_image(source_path)
        page_name = str(
            raw_case.get("page_name") or source_path.name
        )
        snapshot_payload = read_json(snapshot_path)
        page = _snapshot_page(snapshot_payload, page_name=page_name)
        block_records, canonicalization = _snapshot_block_records(
            case_id=case_id,
            page=page,
            case=raw_case,
            image_shape=image.shape,
        )
        blocks = _records_to_blocks(block_records)
        masks = _capture_masks(image, blocks)
        case_dir = output / "cases" / f"{order:03d}"
        source_target = case_dir / "source.png"
        _write_image(source_target, image)
        artifacts: dict[str, dict[str, Any]] = {}
        for artifact_name in (
            "raw_mask",
            "refined_mask",
            "glyph_base_mask",
            "bubble_protect_mask",
            "structure_protect_mask",
            "allowed_window_mask",
            "product_final_mask",
        ):
            artifact_path = case_dir / f"{artifact_name}.png"
            _write_image(artifact_path, masks[artifact_name])
            artifacts[artifact_name] = {
                "path": artifact_path.relative_to(output).as_posix(),
                "sha256": sha256_file(artifact_path),
            }
        roi_raw = raw_case.get("review_roi")
        roi = (
            _normalize_box(
                roi_raw,
                image.shape,
                label=f"case {case_id} review_roi",
            )
            if roi_raw is not None
            else [0, 0, int(image.shape[1]), int(image.shape[0])]
        )
        reference_render = raw_case.get("reference_render")
        reference_record: dict[str, Any] | None = None
        if reference_render:
            reference_path = Path(str(reference_render)).expanduser().resolve()
            if not reference_path.is_file():
                raise FileNotFoundError(
                    f"reference_render is missing for case {case_id}"
                )
            reference_target = case_dir / "reference_render.png"
            _copy_lossless_image(reference_path, reference_target)
            reference_record = {
                "path": reference_target.relative_to(output).as_posix(),
                "sha256": sha256_file(reference_target),
            }
        source_record = {
            "path": source_target.relative_to(output).as_posix(),
            "sha256": sha256_file(source_target),
            "original_file_sha256": actual_source_sha,
            "shape": [int(value) for value in image.shape],
        }
        case_contract = {
            "order": order,
            "case_id": case_id,
            "source": source_record,
            "source_snapshot_sha256": sha256_file(snapshot_path),
            "page_name_sha256": hashlib.sha256(
                page_name.encode("utf-8")
            ).hexdigest(),
            "review_roi": roi,
            "blocks": block_records,
            "block_contract_sha256": canonical_sha256(block_records),
            "canonicalization": canonicalization,
            "artifacts": artifacts,
            "reference_render": reference_record,
            "mask_capture": {
                key: value
                for key, value in masks.items()
                if not hasattr(value, "shape")
            },
        }
        case_contract["case_contract_sha256"] = canonical_sha256(case_contract)
        frozen_cases.append(case_contract)

    contract_without_digest = {
        "protocol_version": PROTOCOL_VERSION,
        "kind": "frozen-inpaint-cases",
        "source_manifest_sha256": sha256_file(manifest_path),
        "case_count": len(frozen_cases),
        "case_order": [case["case_id"] for case in frozen_cases],
        "cases": frozen_cases,
    }
    contract = {
        **contract_without_digest,
        "contract_sha256": canonical_sha256(contract_without_digest),
    }
    atomic_write_json(output / CAPTURE_FILENAME, contract)
    return contract


def _artifact_path(
    root: Path,
    record: Mapping[str, Any],
    *,
    label: str,
) -> Path:
    relative = Path(str(record.get("path") or ""))
    if relative.is_absolute() or ".." in relative.parts:
        raise ProtocolError(f"{label} path is not a safe relative path")
    path = (root / relative).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError:
        raise ProtocolError(f"{label} escapes the frozen/result root")
    if not path.is_file():
        raise FileNotFoundError(f"{label} is missing: {path}")
    expected = str(record.get("sha256") or "").lower()
    actual = sha256_file(path)
    if expected != actual:
        raise ProtocolError(f"{label} SHA-256 differs")
    return path


def validate_frozen_contract(root: Path) -> dict[str, Any]:
    frozen_root = root.expanduser().resolve()
    contract = read_json(frozen_root / CAPTURE_FILENAME)
    if int(contract.get("protocol_version", 0) or 0) != PROTOCOL_VERSION:
        raise ProtocolError("frozen contract protocol_version is unsupported")
    expected_contract = str(contract.get("contract_sha256") or "")
    without_digest = dict(contract)
    without_digest.pop("contract_sha256", None)
    if canonical_sha256(without_digest) != expected_contract:
        raise ProtocolError("frozen contract digest differs")
    cases = contract.get("cases")
    if not isinstance(cases, list) or len(cases) != int(
        contract.get("case_count", -1)
    ):
        raise ProtocolError("frozen case count differs")
    order = [str(case.get("case_id") or "") for case in cases if isinstance(case, Mapping)]
    if order != list(contract.get("case_order") or []):
        raise ProtocolError("frozen case order differs")
    for case in cases:
        if not isinstance(case, Mapping):
            raise ProtocolError("frozen case is not an object")
        expected_case = str(case.get("case_contract_sha256") or "")
        case_without_digest = dict(case)
        case_without_digest.pop("case_contract_sha256", None)
        if canonical_sha256(case_without_digest) != expected_case:
            raise ProtocolError(f"case contract digest differs: {case.get('case_id')}")
        blocks = case.get("blocks")
        if not isinstance(blocks, list):
            raise ProtocolError("frozen case blocks are missing")
        if canonical_sha256(blocks) != str(case.get("block_contract_sha256") or ""):
            raise ProtocolError("frozen block contract digest differs")
        _artifact_path(frozen_root, case["source"], label="frozen source")
        artifacts = case.get("artifacts")
        if not isinstance(artifacts, Mapping):
            raise ProtocolError("frozen mask artifacts are missing")
        for name in (
            "raw_mask",
            "refined_mask",
            "glyph_base_mask",
            "bubble_protect_mask",
            "structure_protect_mask",
            "allowed_window_mask",
            "product_final_mask",
        ):
            record = artifacts.get(name)
            if not isinstance(record, Mapping):
                raise ProtocolError(f"frozen artifact is missing: {name}")
            _artifact_path(frozen_root, record, label=f"frozen {name}")
        reference = case.get("reference_render")
        if reference is not None:
            if not isinstance(reference, Mapping):
                raise ProtocolError("reference render record is invalid")
            _artifact_path(frozen_root, reference, label="reference render")
    return contract


def _profiles_for_phase(
    phase: str,
    *,
    selected_dilation: int | None,
    include_feasibility: bool,
) -> list[CandidateProfile]:
    profiles = [BASELINE]
    if phase == "mask":
        profiles.extend(MASK_SCREEN_PROFILES)
    elif phase == "model":
        if selected_dilation not in {1, 2, 4}:
            raise ProtocolError(
                "model phase requires --selected-dilation 1, 2, or 4"
            )
        for template in MODEL_SCREEN_TEMPLATES:
            if template.feasibility_only and not include_feasibility:
                continue
            profiles.append(replace(template, dilation=selected_dilation))
    else:
        raise ProtocolError(f"unsupported screen phase: {phase}")
    for profile in profiles:
        if profile.promotable and profile.precision != "fp32":
            raise ProtocolError(
                f"promotable candidate is not FP32: {profile.slug}"
            )
    return profiles


def _dilate_mask(mask: Any, dilation: int):
    import cv2
    import numpy as np

    source = (np.asarray(mask) > 0).astype(np.uint8) * 255
    radius = max(0, int(dilation))
    if radius <= 0 or not np.any(source):
        return source
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (radius * 2 + 1, radius * 2 + 1),
        (radius, radius),
    )
    return (cv2.dilate(source, kernel, iterations=1) > 0).astype(np.uint8) * 255


def _load_mask(path: Path):
    import cv2
    import numpy as np

    data = np.fromfile(str(path), dtype=np.uint8)
    mask = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ProtocolError(f"Unable to decode mask: {path}")
    return (mask > 0).astype(np.uint8) * 255


def build_candidate_mask(
    *,
    profile: CandidateProfile,
    product_mask: Any,
    glyph_base_mask: Any,
    bubble_protect_mask: Any,
    structure_protect_mask: Any,
    allowed_window_mask: Any,
):
    import numpy as np

    if profile.mask_mode == "product":
        return (np.asarray(product_mask) > 0).astype(np.uint8) * 255
    if profile.mask_mode != "glyph" or profile.dilation is None:
        raise ProtocolError(f"candidate has no usable mask contract: {profile.slug}")
    candidate = _dilate_mask(glyph_base_mask, profile.dilation)
    protect = np.asarray(bubble_protect_mask) > 0
    if profile.structure_protect:
        protect = protect | (np.asarray(structure_protect_mask) > 0)
    return np.where(
        (candidate > 0)
        & (~protect)
        & (np.asarray(allowed_window_mask) > 0),
        255,
        0,
    ).astype(np.uint8)


def _mask_bbox(mask: Any) -> list[int] | None:
    import cv2
    import numpy as np

    points = cv2.findNonZero((np.asarray(mask) > 0).astype(np.uint8))
    if points is None:
        return None
    x, y, width, height = cv2.boundingRect(points)
    if width <= 0 or height <= 0:
        return None
    return [int(x), int(y), int(x + width), int(y + height)]


def _context_roi(
    mask: Any,
    image_shape: Sequence[int],
    *,
    requested_roi: Sequence[int],
) -> list[int]:
    bbox = _mask_bbox(mask)
    if bbox is None:
        return _normalize_box(
            requested_roi,
            image_shape,
            label="review ROI",
        )
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1
    margin = max(96, int(round(max(width, height) * 0.55)))
    auto = [
        x1 - margin,
        y1 - margin,
        x2 + margin,
        y2 + margin,
    ]
    _normalize_box(
        requested_roi,
        image_shape,
        label="review ROI",
    )
    return _normalize_box(auto, image_shape, label="automatic model ROI")


def _instantiate_inpainter(profile: CandidateProfile):
    from modules.inpainting.runtime_contract import (
        inspect_learned_inpainter_runtime,
        validate_learned_inpaint_runtime,
    )
    from modules.utils.pipeline_config import inpaint_map

    if profile.feasibility_only:
        return None, {
            "status": "not_implemented",
            "reason": "ZITS adapter is intentionally lab-only and not present",
            "fp32_promotion_eligible": False,
        }
    cls = inpaint_map.get(profile.inpainter_key)
    if cls is None:
        raise ProtocolError(f"unknown inpainter key: {profile.inpainter_key}")
    validate_learned_inpaint_runtime(
        inpainter_key=profile.inpainter_key,
        device="cuda",
        precision=profile.precision,
    )
    inpainter = cls(
        "cuda",
        backend="torch",
        runtime_device="cuda",
        inpaint_size=profile.inpaint_size,
        precision=profile.precision,
    )
    runtime = inspect_learned_inpainter_runtime(
        inpainter,
        inpainter_key=profile.inpainter_key,
        requested_device="cuda",
        requested_precision=profile.precision,
    )
    if profile.promotable and not runtime.get("fp32_promotion_eligible"):
        raise ProtocolError(
            f"candidate is not FP32 promotion eligible: {profile.slug}"
        )
    return inpainter, runtime


def _load_profile_runtime(
    profile: CandidateProfile,
) -> tuple[Any | None, dict[str, Any], str]:
    try:
        inpainter, runtime = _instantiate_inpainter(profile)
    except Exception as exc:
        _release_inpainter(None)
        return (
            None,
            {
                "status": "load_failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "fp32_promotion_eligible": False,
            },
            f"{type(exc).__name__}: {exc}",
        )
    return inpainter, dict(runtime), ""


def _release_inpainter(inpainter: Any) -> None:
    if inpainter is not None:
        for attribute in ("model", "session"):
            if hasattr(inpainter, attribute):
                setattr(inpainter, attribute, None)
        del inpainter
    try:
        import gc
        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except (ImportError, RuntimeError):
        pass


def _run_direct_roi(
    *,
    inpainter: Any,
    image: Any,
    mask: Any,
    roi: Sequence[int],
) -> tuple[Any, float, dict[str, Any]]:
    import numpy as np

    from modules.inpainting.runtime_contract import (
        bounded_retry_roi,
        is_cuda_oom_error,
    )
    from modules.inpainting.schema import Config
    from modules.utils.inpaint_composite import composite_with_edit_mask

    initial_roi = [int(value) for value in roi]
    diagnostics: dict[str, Any] = {
        "status": "running",
        "initial_roi": initial_roi,
        "oom_retry_count": 0,
        "oom_retry_roi": None,
    }

    def run_once(active_roi: Sequence[int]) -> tuple[Any, Any, list[int]]:
        x1, y1, x2, y2 = [int(value) for value in active_roi]
        crop = np.ascontiguousarray(image[y1:y2, x1:x2])
        mask_crop = np.ascontiguousarray(mask[y1:y2, x1:x2])
        if not np.any(mask_crop):
            return crop, crop.copy(), [x1, y1, x2, y2]
        candidate_crop = inpainter(
            crop,
            mask_crop,
            Config(hd_strategy="Original"),
        )
        cleaned_crop = composite_with_edit_mask(
            crop,
            candidate_crop,
            mask_crop,
        )
        return crop, cleaned_crop, [x1, y1, x2, y2]

    started = time.perf_counter()
    try:
        _source_crop, cleaned_crop, used_roi = run_once(initial_roi)
    except Exception as exc:
        if not is_cuda_oom_error(exc):
            diagnostics["status"] = "failed"
            diagnostics["first_error"] = type(exc).__name__
            raise CandidateRunError(
                f"{type(exc).__name__}: {exc}",
                diagnostics=diagnostics,
            ) from exc
        retry = bounded_retry_roi(mask, tuple(np.asarray(image).shape))
        diagnostics["oom_retry_count"] = 1
        diagnostics["first_error"] = type(exc).__name__
        if retry is None:
            diagnostics["status"] = "failed_no_smaller_roi"
            raise CandidateRunError(
                "CUDA OOM and no smaller bounded ROI is available",
                diagnostics=diagnostics,
            ) from exc
        retry_roi = retry.as_list()
        initial_area = max(
            0,
            (initial_roi[2] - initial_roi[0])
            * (initial_roi[3] - initial_roi[1]),
        )
        retry_area = max(
            0,
            (retry_roi[2] - retry_roi[0])
            * (retry_roi[3] - retry_roi[1]),
        )
        if retry_area <= 0 or retry_area >= initial_area:
            diagnostics["status"] = "failed_no_smaller_roi"
            raise CandidateRunError(
                "CUDA OOM retry ROI is not smaller than the initial ROI",
                diagnostics=diagnostics,
            ) from exc
        diagnostics["oom_retry_roi"] = retry_roi
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except (ImportError, RuntimeError):
            pass
        try:
            _source_crop, cleaned_crop, used_roi = run_once(retry_roi)
        except Exception as retry_exc:
            diagnostics["status"] = (
                "failed_after_roi_retry"
                if is_cuda_oom_error(retry_exc)
                else "failed_during_roi_retry"
            )
            diagnostics["retry_error"] = type(retry_exc).__name__
            raise CandidateRunError(
                f"{type(retry_exc).__name__}: {retry_exc}",
                diagnostics=diagnostics,
            ) from retry_exc
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except (ImportError, RuntimeError):
        pass
    elapsed = time.perf_counter() - started
    cleaned = np.asarray(image).copy()
    x1, y1, x2, y2 = used_roi
    cleaned[y1:y2, x1:x2] = cleaned_crop
    diagnostics["status"] = (
        "completed_after_roi_retry"
        if diagnostics["oom_retry_count"]
        else "completed"
    )
    return (
        composite_with_edit_mask(image, cleaned, mask),
        elapsed,
        diagnostics,
    )


def _changed_pixel_stats(source: Any, cleaned: Any, mask: Any) -> dict[str, int]:
    import numpy as np

    source_arr = np.asarray(source)
    cleaned_arr = np.asarray(cleaned)
    changed = np.any(source_arr != cleaned_arr, axis=2)
    edit = np.asarray(mask) > 0
    return {
        "changed_pixel_count": int(np.count_nonzero(changed)),
        "changed_inside_mask_pixel_count": int(np.count_nonzero(changed & edit)),
        "changed_outside_mask_pixel_count": int(np.count_nonzero(changed & ~edit)),
    }


def _mask_overlay(image: Any, mask: Any):
    import numpy as np

    source = np.asarray(image).astype(np.float32)
    overlay = source.copy()
    active = np.asarray(mask) > 0
    overlay[active, 0] = 255
    overlay[active, 1] *= 0.35
    overlay[active, 2] *= 0.35
    return np.clip(source * 0.45 + overlay * 0.55, 0, 255).astype(np.uint8)


def _diff_image(source: Any, cleaned: Any):
    import cv2
    import numpy as np

    delta = np.max(
        np.abs(
            np.asarray(cleaned).astype(np.int16)
            - np.asarray(source).astype(np.int16)
        ),
        axis=2,
    ).astype(np.uint8)
    heat = cv2.applyColorMap(delta, cv2.COLORMAP_TURBO)
    return cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)


def _crop(image: Any, roi: Sequence[int]):
    import numpy as np

    x1, y1, x2, y2 = [int(value) for value in roi]
    return np.ascontiguousarray(np.asarray(image)[y1:y2, x1:x2])


def _artifact_record(path: Path, root: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": sha256_file(path),
    }


def run_screen(
    frozen_root: Path,
    output_dir: Path,
    *,
    phase: str,
    selected_dilation: int | None,
    include_feasibility: bool,
) -> dict[str, Any]:
    frozen_root = frozen_root.expanduser().resolve()
    frozen = validate_frozen_contract(frozen_root)
    output = _ensure_new_output_dir(output_dir)
    profiles = _profiles_for_phase(
        phase,
        selected_dilation=selected_dilation,
        include_feasibility=include_feasibility,
    )
    results: list[dict[str, Any]] = []

    for profile_order, profile in enumerate(profiles):
        profile_root = output / "candidates" / f"{profile_order:03d}"
        profile_started = time.perf_counter()
        model_started = time.perf_counter()
        inpainter, runtime, model_load_failure = _load_profile_runtime(profile)
        model_load_seconds = time.perf_counter() - model_started
        case_results: list[dict[str, Any]] = []
        try:
            for case in frozen["cases"]:
                case_id = str(case["case_id"])
                source_path = _artifact_path(
                    frozen_root,
                    case["source"],
                    label=f"{case_id} source",
                )
                image = _read_image(source_path)
                artifacts = case["artifacts"]
                product_mask = _load_mask(
                    _artifact_path(
                        frozen_root,
                        artifacts["product_final_mask"],
                        label=f"{case_id} product mask",
                    )
                )
                glyph_base = _load_mask(
                    _artifact_path(
                        frozen_root,
                        artifacts["glyph_base_mask"],
                        label=f"{case_id} glyph base mask",
                    )
                )
                bubble_protect = _load_mask(
                    _artifact_path(
                        frozen_root,
                        artifacts["bubble_protect_mask"],
                        label=f"{case_id} bubble protect mask",
                    )
                )
                structure_protect = _load_mask(
                    _artifact_path(
                        frozen_root,
                        artifacts["structure_protect_mask"],
                        label=f"{case_id} structure protect mask",
                    )
                )
                allowed = _load_mask(
                    _artifact_path(
                        frozen_root,
                        artifacts["allowed_window_mask"],
                        label=f"{case_id} allowed window mask",
                    )
                )
                mask = build_candidate_mask(
                    profile=profile,
                    product_mask=product_mask,
                    glyph_base_mask=glyph_base,
                    bubble_protect_mask=bubble_protect,
                    structure_protect_mask=structure_protect,
                    allowed_window_mask=allowed,
                )
                roi = _context_roi(
                    mask,
                    image.shape,
                    requested_roi=case["review_roi"],
                )
                case_dir = profile_root / f"{int(case['order']):03d}"
                case_dir.mkdir(parents=True, exist_ok=True)
                status = "completed"
                failure = ""
                run_diagnostics: dict[str, Any] = {
                    "status": "not_started",
                    "oom_retry_count": 0,
                    "oom_retry_roi": None,
                }
                if not bool((mask > 0).any()):
                    cleaned = image.copy()
                    inference_seconds = 0.0
                    status = "failed_empty_mask"
                    failure = "candidate edit mask is empty"
                    run_diagnostics["status"] = status
                elif model_load_failure:
                    cleaned = image.copy()
                    inference_seconds = 0.0
                    status = "model_load_failed"
                    failure = model_load_failure
                    run_diagnostics["status"] = "model_load_failed"
                elif profile.feasibility_only:
                    cleaned = image.copy()
                    inference_seconds = 0.0
                    status = "feasibility_not_implemented"
                    failure = str(runtime.get("reason") or "")
                    run_diagnostics["status"] = status
                else:
                    try:
                        (
                            cleaned,
                            inference_seconds,
                            run_diagnostics,
                        ) = _run_direct_roi(
                            inpainter=inpainter,
                            image=image,
                            mask=mask,
                            roi=roi,
                        )
                    except CandidateRunError as exc:
                        cleaned = image.copy()
                        inference_seconds = 0.0
                        status = "failed"
                        failure = str(exc)
                        run_diagnostics = dict(exc.diagnostics)
                    except Exception as exc:
                        cleaned = image.copy()
                        inference_seconds = 0.0
                        status = "failed"
                        failure = f"{type(exc).__name__}: {exc}"
                        run_diagnostics = {
                            "status": "failed",
                            "error_type": type(exc).__name__,
                            "oom_retry_count": 0,
                            "oom_retry_roi": None,
                        }
                changed = _changed_pixel_stats(image, cleaned, mask)
                if (
                    status == "completed"
                    and changed["changed_outside_mask_pixel_count"] != 0
                ):
                    status = "failed_outside_mask_change"
                output_images = {
                    "original": _crop(image, case["review_roi"]),
                    "raw_mask": _crop(glyph_base, case["review_roi"]),
                    "final_mask": _crop(mask, case["review_roi"]),
                    "mask_overlay": _crop(
                        _mask_overlay(image, mask),
                        case["review_roi"],
                    ),
                    "cleaned": _crop(cleaned, case["review_roi"]),
                    "diff": _crop(
                        _diff_image(image, cleaned),
                        case["review_roi"],
                    ),
                }
                artifact_records: dict[str, dict[str, Any]] = {}
                for name, artifact in output_images.items():
                    path = case_dir / f"{name}.png"
                    _write_image(path, artifact)
                    artifact_records[name] = _artifact_record(path, output)
                reference_render = case.get("reference_render")
                if isinstance(reference_render, Mapping):
                    reference_path = _artifact_path(
                        frozen_root,
                        reference_render,
                        label=f"{case_id} reference render",
                    )
                    target = case_dir / "reference_render.png"
                    _copy_lossless_image(reference_path, target)
                    target = target.with_suffix(".png")
                    artifact_records["reference_render"] = _artifact_record(
                        target,
                        output,
                    )
                case_result = {
                    "case_order": int(case["order"]),
                    "case_id": case_id,
                    "case_contract_sha256": str(
                        case["case_contract_sha256"]
                    ),
                    "status": status,
                    "failure": failure,
                    "review_roi": list(case["review_roi"]),
                    "model_roi": roi,
                    "mask_pixel_count": int((mask > 0).sum()),
                    "inference_seconds": round(float(inference_seconds), 6),
                    "run_diagnostics": run_diagnostics,
                    "changed_pixels": changed,
                    "artifacts": artifact_records,
                }
                case_result["result_sha256"] = canonical_sha256(case_result)
                case_results.append(case_result)
        finally:
            _release_inpainter(inpainter)
        profile_result = {
            "profile_order": profile_order,
            "profile": asdict(profile),
            "runtime": runtime,
            "model_load_seconds": round(float(model_load_seconds), 6),
            "total_seconds": round(
                float(time.perf_counter() - profile_started),
                6,
            ),
            "case_results": case_results,
            "case_order": [item["case_id"] for item in case_results],
            "hard_gate_passed": bool(
                case_results
                and all(item["status"] == "completed" for item in case_results)
                and all(
                    item["changed_pixels"][
                        "changed_outside_mask_pixel_count"
                    ]
                    == 0
                    for item in case_results
                )
                and (not profile.feasibility_only)
            ),
        }
        profile_result["profile_result_sha256"] = canonical_sha256(
            profile_result
        )
        results.append(profile_result)

    payload_without_digest = {
        "protocol_version": PROTOCOL_VERSION,
        "kind": "inpaint-quality-screen-results",
        "phase": phase,
        "selected_dilation": selected_dilation,
        "frozen_contract_sha256": str(frozen["contract_sha256"]),
        "case_order": list(frozen["case_order"]),
        "candidate_order": [
            str(result["profile"]["slug"]) for result in results
        ],
        "results": results,
    }
    payload = {
        **payload_without_digest,
        "result_contract_sha256": canonical_sha256(payload_without_digest),
    }
    atomic_write_json(output / RESULT_FILENAME, payload)
    return payload


def _refresh_result_digests(payload: dict[str, Any]) -> dict[str, Any]:
    for result in payload["results"]:
        for case in result["case_results"]:
            case.pop("result_sha256", None)
            case["result_sha256"] = canonical_sha256(case)
        result.pop("profile_result_sha256", None)
        result["profile_result_sha256"] = canonical_sha256(result)
    payload.pop("result_contract_sha256", None)
    payload["result_contract_sha256"] = canonical_sha256(payload)
    return payload


def attach_renders(
    results_root: Path,
    manifest_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    source_root = results_root.expanduser().resolve()
    source_results = validate_results(source_root)
    manifest_path = manifest_path.expanduser().resolve()
    manifest = read_json(manifest_path)
    if int(manifest.get("protocol_version", 0) or 0) != PROTOCOL_VERSION:
        raise ProtocolError("render manifest protocol_version is unsupported")
    if str(manifest.get("kind") or "") != RENDER_MANIFEST_KIND:
        raise ProtocolError("render manifest kind differs")
    if str(manifest.get("result_contract_sha256") or "") != str(
        source_results["result_contract_sha256"]
    ):
        raise ProtocolError("render manifest result contract differs")
    attachments = manifest.get("renders")
    if not isinstance(attachments, list):
        raise ProtocolError("render manifest renders must be a list")
    expected_pairs = [
        (str(result["profile"]["slug"]), str(case["case_id"]))
        for result in source_results["results"]
        for case in result["case_results"]
    ]
    attachment_by_pair: dict[tuple[str, str], tuple[Path, str]] = {}
    for item in attachments:
        if not isinstance(item, Mapping):
            raise ProtocolError("each render attachment must be an object")
        pair = (
            str(item.get("candidate_slug") or ""),
            str(item.get("case_id") or ""),
        )
        if pair in attachment_by_pair:
            raise ProtocolError(f"duplicate render attachment: {pair}")
        source = Path(str(item.get("render_image") or "")).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(f"render attachment is missing: {source}")
        expected_sha = str(item.get("render_sha256") or "").lower()
        actual_sha = sha256_file(source)
        if not expected_sha or expected_sha != actual_sha:
            raise ProtocolError(f"render attachment SHA-256 differs: {pair}")
        attachment_by_pair[pair] = (source, actual_sha)
    if list(attachment_by_pair) != expected_pairs:
        raise ProtocolError(
            "render attachment order or candidate/case coverage differs"
        )

    resolved_output = output_dir.expanduser().resolve()
    try:
        resolved_output.relative_to(source_root)
    except ValueError:
        pass
    else:
        raise ProtocolError("render output cannot be nested inside source results")
    output = _ensure_new_output_dir(resolved_output)
    updated = json.loads(canonical_json(source_results))
    for result in source_results["results"]:
        for case in result["case_results"]:
            for artifact_name, record in case["artifacts"].items():
                if not isinstance(record, Mapping):
                    raise ProtocolError(
                        f"result artifact is invalid: {artifact_name}"
                    )
                source = _artifact_path(
                    source_root,
                    record,
                    label=f"result {artifact_name}",
                )
                relative = Path(str(record["path"]))
                target = output / relative
                _copy_artifact(source, target)
    for result in updated["results"]:
        candidate_slug = str(result["profile"]["slug"])
        for case in result["case_results"]:
            case_id = str(case["case_id"])
            source, _source_sha = attachment_by_pair[
                (candidate_slug, case_id)
            ]
            cleaned_record = case["artifacts"]["cleaned"]
            case_dir = (output / Path(str(cleaned_record["path"]))).parent
            target = case_dir / "render.png"
            _copy_lossless_image(source, target)
            case["artifacts"]["render"] = _artifact_record(target, output)
    updated["render_attachment_manifest_sha256"] = sha256_file(manifest_path)
    _refresh_result_digests(updated)
    atomic_write_json(output / RESULT_FILENAME, updated)
    validate_results(output)
    return updated


def validate_results(root: Path) -> dict[str, Any]:
    result_root = root.expanduser().resolve()
    payload = read_json(result_root / RESULT_FILENAME)
    if int(payload.get("protocol_version", 0) or 0) != PROTOCOL_VERSION:
        raise ProtocolError("result protocol_version is unsupported")
    expected = str(payload.get("result_contract_sha256") or "")
    without_digest = dict(payload)
    without_digest.pop("result_contract_sha256", None)
    if canonical_sha256(without_digest) != expected:
        raise ProtocolError("result contract digest differs")
    results = payload.get("results")
    if not isinstance(results, list) or not results:
        raise ProtocolError("result candidate list is empty")
    candidate_order = [
        str(item.get("profile", {}).get("slug") or "")
        for item in results
        if isinstance(item, Mapping)
    ]
    if candidate_order != list(payload.get("candidate_order") or []):
        raise ProtocolError("candidate order differs")
    case_order = list(payload.get("case_order") or [])
    for result in results:
        if not isinstance(result, Mapping):
            raise ProtocolError("candidate result is not an object")
        expected_profile = str(result.get("profile_result_sha256") or "")
        result_without_digest = dict(result)
        result_without_digest.pop("profile_result_sha256", None)
        if canonical_sha256(result_without_digest) != expected_profile:
            raise ProtocolError("candidate result digest differs")
        profile = result.get("profile")
        if not isinstance(profile, Mapping):
            raise ProtocolError("candidate profile is missing")
        if bool(profile.get("promotable")) and str(
            profile.get("precision") or ""
        ) != "fp32":
            raise ProtocolError("promotable result is not FP32")
        runtime = result.get("runtime")
        if not isinstance(runtime, Mapping):
            raise ProtocolError("candidate runtime contract is missing")
        if bool(profile.get("promotable")) and bool(
            result.get("hard_gate_passed")
        ):
            if not str(runtime.get("actual_device") or "").lower().startswith(
                "cuda"
            ):
                raise ProtocolError(
                    "hard-gate promotable candidate did not run on CUDA"
                )
            if str(runtime.get("actual_precision") or "").lower() != "fp32":
                raise ProtocolError(
                    "hard-gate promotable candidate did not run in FP32"
                )
            if not bool(runtime.get("fp32_promotion_eligible")):
                raise ProtocolError(
                    "hard-gate candidate is not FP32 promotion eligible"
                )
        cases = result.get("case_results")
        if not isinstance(cases, list):
            raise ProtocolError("candidate case results are missing")
        if [str(case.get("case_id") or "") for case in cases] != case_order:
            raise ProtocolError("candidate case order differs")
        for case in cases:
            expected_case = str(case.get("result_sha256") or "")
            case_without_digest = dict(case)
            case_without_digest.pop("result_sha256", None)
            if canonical_sha256(case_without_digest) != expected_case:
                raise ProtocolError("case result digest differs")
            artifacts = case.get("artifacts")
            if not isinstance(artifacts, Mapping):
                raise ProtocolError("case result artifacts are missing")
            for name in (
                "original",
                "raw_mask",
                "final_mask",
                "mask_overlay",
                "cleaned",
                "diff",
            ):
                record = artifacts.get(name)
                if not isinstance(record, Mapping):
                    raise ProtocolError(f"result artifact is missing: {name}")
                _artifact_path(
                    result_root,
                    record,
                    label=f"result {name}",
                )
            for name in ("reference_render", "render"):
                record = artifacts.get(name)
                if record is not None:
                    if not isinstance(record, Mapping):
                        raise ProtocolError(
                            f"optional result artifact is invalid: {name}"
                        )
                    _artifact_path(
                        result_root,
                        record,
                        label=f"result {name}",
                    )
            changed = case.get("changed_pixels")
            if not isinstance(changed, Mapping):
                raise ProtocolError("changed-pixel contract is missing")
            diagnostics = case.get("run_diagnostics")
            if not isinstance(diagnostics, Mapping):
                raise ProtocolError("candidate run diagnostics are missing")
            if int(changed.get("changed_outside_mask_pixel_count", -1)) != 0:
                if str(case.get("status") or "") == "completed":
                    raise ProtocolError(
                        "completed candidate changed pixels outside its mask"
                    )
    return payload


def _neutral_labels(count: int) -> list[str]:
    if count <= 0 or count > 26:
        raise ProtocolError("blind review supports 1 to 26 candidates")
    return [chr(ord("A") + index) for index in range(count)]


def _select_mapping(candidates: Sequence[str]) -> dict[str, str]:
    shuffled = list(candidates)
    secrets.SystemRandom().shuffle(shuffled)
    return dict(zip(_neutral_labels(len(shuffled)), shuffled))


def _copy_artifact(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def _candidate_by_slug(
    payload: Mapping[str, Any],
) -> dict[str, Mapping[str, Any]]:
    return {
        str(result["profile"]["slug"]): result
        for result in payload["results"]
    }


def build_blind_review(
    results_root: Path,
    output_dir: Path,
    *,
    candidate_slugs: Sequence[str] | None = None,
    require_render: bool = False,
    mapping: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    results_root = results_root.expanduser().resolve()
    results = validate_results(results_root)
    by_slug = _candidate_by_slug(results)
    selected = list(candidate_slugs or results["candidate_order"])
    if len(selected) < 2:
        raise ProtocolError("blind review requires at least two candidates")
    if len(selected) != len(set(selected)) or any(
        slug not in by_slug for slug in selected
    ):
        raise ProtocolError("blind candidate selection is invalid")
    failed_hard_gates = [
        slug
        for slug in selected
        if not bool(by_slug[slug].get("hard_gate_passed"))
    ]
    if failed_hard_gates:
        raise ProtocolError(
            "blind candidates must pass automated hard gates: "
            + ", ".join(failed_hard_gates)
        )
    label_to_candidate = dict(mapping or _select_mapping(selected))
    neutral_labels = _neutral_labels(len(selected))
    if list(label_to_candidate) != neutral_labels:
        raise ProtocolError("blind labels must be contiguous A..N")
    if set(label_to_candidate.values()) != set(selected):
        raise ProtocolError("blind mapping candidates differ from selection")
    output = _ensure_new_output_dir(output_dir)
    private = output / PRIVATE_DIRNAME
    private.mkdir()

    public_cases: list[dict[str, Any]] = []
    for case_index, case_id in enumerate(results["case_order"]):
        public_candidates: list[dict[str, Any]] = []
        for label, slug in label_to_candidate.items():
            candidate = by_slug[slug]
            case = candidate["case_results"][case_index]
            if str(case.get("case_id") or "") != case_id:
                raise ProtocolError("blind case alignment differs")
            source_artifacts = case["artifacts"]
            if require_render and "render" not in source_artifacts:
                raise ProtocolError(
                    f"final blind review requires render artifact for {case_id}"
                )
            public_artifacts: dict[str, dict[str, str]] = {}
            for artifact_name in (
                "original",
                "raw_mask",
                "final_mask",
                "mask_overlay",
                "cleaned",
                "diff",
                "render",
            ):
                record = source_artifacts.get(artifact_name)
                if not isinstance(record, Mapping):
                    continue
                source = _artifact_path(
                    results_root,
                    record,
                    label=f"blind source {artifact_name}",
                )
                target = (
                    output
                    / "assets"
                    / f"case-{case_index + 1:03d}"
                    / label
                    / f"{artifact_name}.png"
                )
                _copy_artifact(source, target)
                public_artifacts[artifact_name] = {
                    "path": target.relative_to(output).as_posix(),
                    "sha256": sha256_file(target),
                }
            public_candidates.append(
                {
                    "label": label,
                    "status": str(case.get("status") or ""),
                    "outside_mask_changed_pixel_count": int(
                        case["changed_pixels"][
                            "changed_outside_mask_pixel_count"
                        ]
                    ),
                    "artifacts": public_artifacts,
                }
            )
        public_cases.append(
            {
                "case_number": case_index + 1,
                "case_key": hashlib.sha256(
                    str(case_id).encode("utf-8")
                ).hexdigest()[:16],
                "candidates": public_candidates,
            }
        )

    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "kind": "blind-inpaint-quality-review",
        "case_count": len(public_cases),
        "candidate_count": len(selected),
        "labels": neutral_labels,
        "candidate_names_hidden": True,
        "timings_hidden": True,
        "require_render": bool(require_render),
        "result_contract_sha256": str(results["result_contract_sha256"]),
        "cases": public_cases,
    }
    payload_digest = canonical_sha256(payload)
    atomic_write_json(private / PRIVATE_PAYLOAD_FILENAME, payload)
    private_key = {
        "protocol_version": PROTOCOL_VERSION,
        "label_to_candidate": label_to_candidate,
        "candidate_contracts": {
            slug: {
                "promotable": bool(by_slug[slug]["profile"]["promotable"]),
                "baseline": bool(by_slug[slug]["profile"]["baseline"]),
                "feasibility_only": bool(
                    by_slug[slug]["profile"]["feasibility_only"]
                ),
                "hard_gate_passed": bool(
                    by_slug[slug]["hard_gate_passed"]
                ),
                "precision": str(
                    by_slug[slug]["profile"]["precision"] or ""
                ),
            }
            for slug in selected
        },
        "result_contract_sha256": str(results["result_contract_sha256"]),
        "require_render": bool(require_render),
        "disclosure_status": "private_until_complete_review",
    }
    atomic_write_json(private / PRIVATE_KEY_FILENAME, private_key)
    _write_review_csv(output / REVIEW_FILENAME, payload)
    (output / REVIEW_HTML_FILENAME).write_text(
        _render_review_html(payload, payload_digest=payload_digest),
        encoding="utf-8",
    )
    state = {
        "protocol_version": PROTOCOL_VERSION,
        "status": "awaiting_complete_blind_review",
        "case_count": len(public_cases),
        "candidate_count": len(selected),
        "review_row_count": len(public_cases) * len(selected),
        "candidate_names_hidden": True,
        "timings_hidden": True,
        "payload_sha256": payload_digest,
        "private_key_sha256": canonical_sha256(private_key),
        "results_root_sha256": str(results["result_contract_sha256"]),
    }
    atomic_write_json(output / STATE_FILENAME, state)
    return state


def _review_columns() -> list[str]:
    return [
        "case_number",
        "case_key",
        "candidate",
        *PROMOTION_REVIEW_FIELDS,
        "rank",
        "notes",
    ]


def _write_review_csv(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=_review_columns())
        writer.writeheader()
        for case in payload["cases"]:
            for candidate in case["candidates"]:
                writer.writerow(
                    {
                        "case_number": case["case_number"],
                        "case_key": case["case_key"],
                        "candidate": candidate["label"],
                        **{field: "" for field in PROMOTION_REVIEW_FIELDS},
                        "rank": "",
                        "notes": "",
                    }
                )


def _render_review_html(
    payload: Mapping[str, Any],
    *,
    payload_digest: str,
) -> str:
    cards: list[str] = []
    for case in payload["cases"]:
        candidate_cards: list[str] = []
        for candidate in case["candidates"]:
            images = []
            for artifact_name, label in (
                ("original", "원본"),
                ("raw_mask", "glyph base"),
                ("final_mask", "최종 mask"),
                ("mask_overlay", "mask overlay"),
                ("cleaned", "cleaned"),
                ("diff", "diff"),
                ("render", "render"),
            ):
                record = candidate["artifacts"].get(artifact_name)
                if isinstance(record, Mapping):
                    relative = str(record.get("path") or "")
                    images.append(
                        '<figure><figcaption>'
                        + html.escape(label)
                        + '</figcaption><img loading="lazy" src="'
                        + html.escape(relative, quote=True)
                        + '" alt="'
                        + html.escape(label, quote=True)
                        + '"></figure>'
                    )
            fields = "".join(
                '<label>'
                + html.escape(field.replace("_", " "))
                + '<select data-field="'
                + html.escape(field, quote=True)
                + '"><option value="">미검수</option>'
                + '<option value="pass">통과</option>'
                + '<option value="fail">실패</option>'
                + (
                    '<option value="na">해당 없음</option>'
                    if field == "render" and not bool(payload["require_render"])
                    else ""
                )
                + "</select></label>"
                for field in PROMOTION_REVIEW_FIELDS
            )
            candidate_cards.append(
                '<article class="candidate" data-case="'
                + str(case["case_number"])
                + '" data-key="'
                + html.escape(case["case_key"], quote=True)
                + '" data-candidate="'
                + html.escape(candidate["label"], quote=True)
                + '"><h3>후보 '
                + html.escape(candidate["label"])
                + "</h3><div class=\"images\">"
                + "".join(images)
                + '</div><div class="review">'
                + fields
                + '<label>순위<input data-field="rank" type="number" min="1"></label>'
                + '<label>메모<textarea data-field="notes"></textarea></label>'
                + "</div></article>"
            )
        cards.append(
            "<section><h2>케이스 "
            + str(case["case_number"])
            + "</h2>"
            + "".join(candidate_cards)
            + "</section>"
        )
    browser_payload = json.dumps(
        {
            "columns": _review_columns(),
            "caseCount": payload["case_count"],
            "candidateCount": payload["candidate_count"],
        },
        ensure_ascii=False,
    ).replace("<", "\\u003c")
    return f"""<!doctype html>
<html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>인페인트 블라인드 품질 검수</title>
<style>
body{{font-family:Segoe UI,Noto Sans KR,sans-serif;margin:0;background:#101318;color:#eee}}
header{{position:sticky;top:0;z-index:5;background:#18202a;padding:14px 20px}}
h1{{font-size:21px;margin:0 0 8px}} header p{{margin:4px 0}}
button{{padding:9px 14px;background:#2c8a57;color:white;border:0;border-radius:6px}}
section{{margin:18px;padding:14px;background:#1a2028;border-radius:10px}}
.candidate{{border:1px solid #45505d;border-radius:8px;padding:12px;margin:12px 0}}
.images{{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:8px}}
figure{{margin:0;background:#0b0e12;padding:6px}} figcaption{{margin-bottom:5px}}
img{{width:100%;height:auto;image-rendering:auto}}
.review{{display:flex;flex-wrap:wrap;gap:10px;margin-top:10px}}
label{{display:flex;flex-direction:column;gap:3px}} select,input,textarea{{min-width:130px}}
.incomplete{{outline:3px solid #d29322}}
</style></head><body>
<header><h1>GPU FP32 인페인트 블라인드 검수</h1>
<p>후보명과 속도는 숨겨져 있습니다. 원문 잔상 → 구조 연속성 → 외부 보존 → 새 왜곡 → render 순서로 판정하세요.</p>
<p id="progress">0 / {payload['case_count'] * payload['candidate_count']} 완료</p>
<button id="export">완료 CSV 내보내기</button></header>
<main>{''.join(cards)}</main>
<script>
"use strict";
const meta={browser_payload};
const storageKey="ct-inpaint-quality-{payload_digest}-"+location.pathname;
const cards=[...document.querySelectorAll(".candidate")];
function state(card){{
 const row={{case_number:card.dataset.case,case_key:card.dataset.key,candidate:card.dataset.candidate}};
 for(const input of card.querySelectorAll("[data-field]")) row[input.dataset.field]=input.value;
 return row;
}}
function complete(row){{
 return {json.dumps(list(PROMOTION_REVIEW_FIELDS))}.every(k=>
   k==="render" ? ["pass","fail","na"].includes(row[k]) : ["pass","fail"].includes(row[k])) &&
   String(row.rank||"").trim()!=="";
}}
function save(){{
 const rows=cards.map(state); localStorage.setItem(storageKey,JSON.stringify(rows));
 let done=0; rows.forEach((row,i)=>{{const ok=complete(row);cards[i].classList.toggle("incomplete",!ok);if(ok)done++;}});
 document.getElementById("progress").textContent=`${{done}} / ${{rows.length}} 완료`;
}}
function restore(){{
 let rows=[];try{{rows=JSON.parse(localStorage.getItem(storageKey)||"[]")}}catch(_e){{}}
 const byKey=new Map(rows.map(row=>[row.case_key+"|"+row.candidate,row]));
 cards.forEach(card=>{{const row=byKey.get(card.dataset.key+"|"+card.dataset.candidate);if(!row)return;
  card.querySelectorAll("[data-field]").forEach(input=>input.value=row[input.dataset.field]||"");}});
 save();
}}
function csv(value){{let text=String(value??"");if(/^\\s*[=+\\-@]/.test(text))text="'"+text;return '"'+text.replaceAll('"','""')+'"';}}
function exportCsv(){{
 save();const rows=cards.map(state);const columns=meta.columns;
 const lines=[columns.map(csv).join(",")];
 rows.forEach(row=>lines.push(columns.map(column=>csv(row[column]||"")).join(",")));
 const blob=new Blob(["\\ufeff"+lines.join("\\r\\n")+"\\r\\n"],{{type:"text/csv;charset=utf-8"}});
 const link=document.createElement("a");link.href=URL.createObjectURL(blob);link.download="blind_review_completed.csv";link.click();URL.revokeObjectURL(link.href);
}}
cards.forEach(card=>{{card.addEventListener("change",save);card.addEventListener("input",save);}});
document.getElementById("export").addEventListener("click",exportCsv);restore();
</script></body></html>"""


def _load_review_rows(review_path: Path) -> list[dict[str, str]]:
    with review_path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ReviewIncompleteError("review CSV is empty")
    if list(rows[0]) != _review_columns():
        raise ProtocolError("review CSV columns differ")
    return rows


def validate_review(
    review_root: Path,
    review_path: Path,
) -> dict[str, Any]:
    root = review_root.expanduser().resolve()
    state = read_json(root / STATE_FILENAME)
    payload = read_json(root / PRIVATE_DIRNAME / PRIVATE_PAYLOAD_FILENAME)
    if canonical_sha256(payload) != str(state.get("payload_sha256") or ""):
        raise ProtocolError("blind payload digest differs")
    for case in payload["cases"]:
        for candidate in case["candidates"]:
            artifacts = candidate.get("artifacts")
            if not isinstance(artifacts, Mapping):
                raise ProtocolError("blind candidate artifacts are missing")
            for artifact_name, record in artifacts.items():
                if not isinstance(record, Mapping):
                    raise ProtocolError("blind artifact record is invalid")
                _artifact_path(
                    root,
                    record,
                    label=f"blind {artifact_name}",
                )
    rows = _load_review_rows(review_path.expanduser().resolve())
    expected_pairs = [
        (
            str(case["case_number"]),
            str(case["case_key"]),
            str(candidate["label"]),
        )
        for case in payload["cases"]
        for candidate in case["candidates"]
    ]
    actual_pairs = [
        (
            str(row.get("case_number") or ""),
            str(row.get("case_key") or ""),
            str(row.get("candidate") or ""),
        )
        for row in rows
    ]
    if actual_pairs != expected_pairs:
        raise ProtocolError("review row order or identity differs")
    failures: list[str] = []
    for index, row in enumerate(rows, start=1):
        for field in PROMOTION_REVIEW_FIELDS:
            value = str(row.get(field) or "").strip().lower()
            allowed = (
                {"pass", "fail"}
                if field != "render" or bool(payload.get("require_render"))
                else REVIEW_VALUES
            )
            if value not in allowed:
                failures.append(f"row {index} {field}")
        try:
            rank = int(str(row.get("rank") or ""))
        except ValueError:
            rank = 0
        if rank < 1 or rank > int(payload["candidate_count"]):
            failures.append(f"row {index} rank")
    if failures:
        raise ReviewIncompleteError(
            "blind review is incomplete: " + ", ".join(failures[:12])
        )
    expected_ranks = list(range(1, int(payload["candidate_count"]) + 1))
    by_case: dict[str, list[int]] = {}
    for row in rows:
        by_case.setdefault(str(row["case_key"]), []).append(int(row["rank"]))
    for case_key, ranks in by_case.items():
        if sorted(ranks) != expected_ranks:
            raise ReviewIncompleteError(
                "candidate ranks must be unique and complete for "
                f"case {case_key}"
            )
    result = {
        "protocol_version": PROTOCOL_VERSION,
        "status": "complete",
        "row_count": len(rows),
        "review_sha256": sha256_file(review_path),
        "payload_sha256": str(state["payload_sha256"]),
    }
    return result


def unblind_review(
    review_root: Path,
    review_path: Path,
    *,
    confirmation: str,
) -> dict[str, Any]:
    root = review_root.expanduser().resolve()
    state = read_json(root / STATE_FILENAME)
    expected_confirmation = (
        f"{int(state['review_row_count'])}-ROWS-REVIEWED"
    )
    if confirmation != expected_confirmation:
        raise ProtocolError(
            f"confirmation must be exactly {expected_confirmation}"
        )
    validated = validate_review(root, review_path)
    key = read_json(root / PRIVATE_DIRNAME / PRIVATE_KEY_FILENAME)
    if canonical_sha256(key) != str(state.get("private_key_sha256") or ""):
        raise ProtocolError("blind private key digest differs")
    rows = _load_review_rows(review_path.expanduser().resolve())
    label_to_candidate = key.get("label_to_candidate")
    if not isinstance(label_to_candidate, Mapping):
        raise ProtocolError("blind key mapping is missing")
    candidate_contracts = key.get("candidate_contracts")
    if not isinstance(candidate_contracts, Mapping):
        raise ProtocolError("blind candidate contracts are missing")
    candidate_failures: dict[str, list[dict[str, Any]]] = {
        str(candidate): [] for candidate in label_to_candidate.values()
    }
    candidate_ranks: dict[str, list[int]] = {
        str(candidate): [] for candidate in label_to_candidate.values()
    }
    for row in rows:
        label = str(row["candidate"])
        candidate = str(label_to_candidate[label])
        failed_fields = [
            field
            for field in PROMOTION_REVIEW_FIELDS
            if str(row[field]).lower() == "fail"
        ]
        if failed_fields:
            candidate_failures[candidate].append(
                {
                    "case_number": int(row["case_number"]),
                    "failed_fields": failed_fields,
                    "notes": str(row.get("notes") or ""),
                }
            )
        candidate_ranks[candidate].append(int(row["rank"]))
    screen_eligible_candidates = [
        candidate
        for candidate, failures in candidate_failures.items()
        if (
            not failures
            and isinstance(candidate_contracts.get(candidate), Mapping)
            and bool(candidate_contracts[candidate].get("promotable"))
            and bool(candidate_contracts[candidate].get("hard_gate_passed"))
            and str(candidate_contracts[candidate].get("precision") or "")
            == "fp32"
            and not bool(candidate_contracts[candidate].get("baseline"))
            and not bool(
                candidate_contracts[candidate].get("feasibility_only")
            )
        )
    ]
    summary = {
        "protocol_version": PROTOCOL_VERSION,
        "status": "unblinded",
        "review": validated,
        "label_to_candidate": dict(label_to_candidate),
        "candidate_failures": candidate_failures,
        "candidate_mean_rank": {
            candidate: (
                sum(ranks) / len(ranks) if ranks else None
            )
            for candidate, ranks in candidate_ranks.items()
        },
        "screen_eligible_candidates": screen_eligible_candidates,
        "promotion_eligible_candidates": (
            screen_eligible_candidates
            if bool(key.get("require_render"))
            else []
        ),
    }
    atomic_write_json(root / UNBLIND_FILENAME, summary)
    updated_state = {
        **state,
        "status": "unblinded",
        "review_sha256": validated["review_sha256"],
    }
    atomic_write_json(root / STATE_FILENAME, updated_state)
    return summary


def _parse_candidate_list(value: str) -> list[str] | None:
    values = [item.strip() for item in str(value or "").split(",") if item.strip()]
    return values or None


def _print_profiles() -> None:
    payload = {
        "baseline": asdict(BASELINE),
        "mask_screen": [asdict(profile) for profile in MASK_SCREEN_PROFILES],
        "model_screen": [
            asdict(profile) for profile in MODEL_SCREEN_TEMPLATES
        ],
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Frozen CUDA inpaint quality-gate protocol."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    capture = subparsers.add_parser("capture")
    capture.add_argument("--manifest", required=True)
    capture.add_argument("--output", required=True)

    run = subparsers.add_parser("run")
    run.add_argument("--frozen", required=True)
    run.add_argument("--output", required=True)
    run.add_argument("--phase", required=True, choices=("mask", "model"))
    run.add_argument("--selected-dilation", type=int)
    run.add_argument("--include-feasibility", action="store_true")

    attach = subparsers.add_parser("attach-renders")
    attach.add_argument("--results", required=True)
    attach.add_argument("--manifest", required=True)
    attach.add_argument("--output", required=True)

    blind = subparsers.add_parser("blind")
    blind.add_argument("--results", required=True)
    blind.add_argument("--output", required=True)
    blind.add_argument(
        "--candidates",
        default="",
        help="Comma-separated candidate slugs; defaults to all.",
    )
    blind.add_argument("--require-render", action="store_true")

    validate = subparsers.add_parser("validate-review")
    validate.add_argument("--review-root", required=True)
    validate.add_argument("--review-csv", required=True)

    unblind = subparsers.add_parser("unblind")
    unblind.add_argument("--review-root", required=True)
    unblind.add_argument("--review-csv", required=True)
    unblind.add_argument("--confirmation", required=True)

    subparsers.add_parser("profiles")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "capture":
        contract = capture_cases(Path(args.manifest), Path(args.output))
        print(json.dumps(
            {
                "output": str(Path(args.output).expanduser().resolve()),
                "case_count": contract["case_count"],
                "contract_sha256": contract["contract_sha256"],
            },
            ensure_ascii=False,
            indent=2,
        ))
        return 0
    if args.command == "attach-renders":
        result = attach_renders(
            Path(args.results),
            Path(args.manifest),
            Path(args.output),
        )
        print(json.dumps(
            {
                "output": str(Path(args.output).expanduser().resolve()),
                "candidate_count": len(result["candidate_order"]),
                "case_count": len(result["case_order"]),
                "result_contract_sha256": result[
                    "result_contract_sha256"
                ],
            },
            ensure_ascii=False,
            indent=2,
        ))
        return 0
    if args.command == "run":
        result = run_screen(
            Path(args.frozen),
            Path(args.output),
            phase=args.phase,
            selected_dilation=args.selected_dilation,
            include_feasibility=bool(args.include_feasibility),
        )
        print(json.dumps(
            {
                "output": str(Path(args.output).expanduser().resolve()),
                "phase": result["phase"],
                "candidates": result["candidate_order"],
                "result_contract_sha256": result[
                    "result_contract_sha256"
                ],
            },
            ensure_ascii=False,
            indent=2,
        ))
        return 0
    if args.command == "blind":
        state = build_blind_review(
            Path(args.results),
            Path(args.output),
            candidate_slugs=_parse_candidate_list(args.candidates),
            require_render=bool(args.require_render),
        )
        print(json.dumps(state, ensure_ascii=False, indent=2))
        return 0
    if args.command == "validate-review":
        result = validate_review(
            Path(args.review_root),
            Path(args.review_csv),
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    if args.command == "unblind":
        result = unblind_review(
            Path(args.review_root),
            Path(args.review_csv),
            confirmation=args.confirmation,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    if args.command == "profiles":
        _print_profiles()
        return 0
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())

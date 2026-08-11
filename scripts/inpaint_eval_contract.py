#!/usr/bin/env python3
"""Private, fail-closed evaluation contracts for inpaint debug exports."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Iterable

import cv2
import numpy as np
from PIL import Image, ImageDraw, PngImagePlugin


MANIFEST_SCHEMA_VERSION = 2
SUPPORTED_MANIFEST_SCHEMA_VERSIONS = frozenset({1, 2})
MANIFEST_SEAL_FIELD = "manifest_sha256"
MANIFEST_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,47}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
EXPECTED_EDIT_VALUES = frozenset({"required", "none", "optional"})
FINAL_EXPECTED_EDIT_VALUES = frozenset({"required", "none"})
EXPECTED_EDIT_DECISION_BASIS = "source-only-review"
SOURCE_ONLY_EVIDENCE_BASIS = "source-only-inpaint-evidence-v1"
SOURCE_ONLY_EVIDENCE_BASIS_V2 = "source-only-inpaint-evidence-v2"
SOURCE_ONLY_ADJUDICATED_EVIDENCE_BASIS_V2 = (
    "source-only-inpaint-evidence-v2-target-adjudicated"
)
SPLIT_ROLE_VALUES = frozenset(
    {
        "tuning",
        "internal-holdout",
        "final-holdout-primary",
        "final-holdout-reserve",
        "locked-regression",
        "cross-language-diagnostic",
        "cross-language-final-holdout",
    }
)
BLIND_REVIEW_VERSION = "inpaint-blind-review-v2"
WINDOWS_RESERVED_STEMS = frozenset(
    {
        "aux",
        "con",
        "nul",
        "prn",
        *(f"com{index}" for index in range(1, 10)),
        *(f"lpt{index}" for index in range(1, 10)),
    }
)
MANIFEST_KEYS = frozenset(
    {
        "schema_version",
        "corpus_id",
        "expected_count",
        "pages",
        MANIFEST_SEAL_FIELD,
        "split_role",
        "source_lock_git_sha",
        "parent_manifest_sha256",
        "expected_edit_basis",
        "expected_edit_decisions_sha256",
        "evidence_parent_manifest_sha256",
        "evidence_basis",
        "evidence_review_sha256",
    }
)
PAGE_KEYS = frozenset(
    {
        "page_id",
        "path",
        "sha256",
        "size_bytes",
        "width",
        "height",
        "expected_edit",
        "baseline",
        "baseline_mask",
        "target_glyph_mask",
        "target_text_mask",
        "protected_structure_mask",
        "ambiguous_structure_mask",
    }
)
REFERENCE_KEYS = frozenset({"path", "sha256"})


class InpaintEvalManifestError(ValueError):
    """A path-free validation failure safe to retain in private summaries."""

    def __init__(
        self,
        code: str,
        *,
        corpus_id: str = "",
        page_id: str = "",
    ) -> None:
        self.code = str(code or "manifest_invalid")
        self.corpus_id = str(corpus_id or "")
        self.page_id = str(page_id or "")
        suffix = ":".join(value for value in (self.corpus_id, self.page_id) if value)
        super().__init__(f"{self.code}:{suffix}" if suffix else self.code)

    def as_record(self) -> dict[str, str]:
        return {
            "error_code": self.code,
            "corpus_id": self.corpus_id,
            "page_id": self.page_id,
        }


@dataclass(frozen=True)
class EvalImageReference:
    path: Path
    sha256: str


@dataclass(frozen=True)
class EvalPageSpec:
    corpus_id: str
    page_id: str
    source: EvalImageReference
    width: int
    height: int
    size_bytes: int
    expected_edit: str
    baseline: EvalImageReference | None = None
    baseline_mask: EvalImageReference | None = None
    target_text_mask: EvalImageReference | None = None
    protected_structure_mask: EvalImageReference | None = None
    ambiguous_structure_mask: EvalImageReference | None = None

    @property
    def target_glyph_mask(self) -> EvalImageReference | None:
        """Compatibility alias for schema-v1 private manifests only."""

        return self.target_text_mask


@dataclass(frozen=True)
class EvalManifest:
    schema_version: int
    corpus_id: str
    split_role: str
    source_lock_git_sha: str
    expected_count: int
    manifest_sha256: str
    pages: tuple[EvalPageSpec, ...]
    parent_manifest_sha256: str | None = None
    expected_edit_basis: str | None = None
    expected_edit_decisions_sha256: str | None = None
    evidence_parent_manifest_sha256: str | None = None
    evidence_basis: str | None = None
    evidence_review_sha256: str | None = None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pixel_sha256(array: np.ndarray) -> str:
    value = np.ascontiguousarray(np.asarray(array))
    header = json.dumps(
        {"shape": list(value.shape), "dtype": str(value.dtype)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    digest = hashlib.sha256(header)
    digest.update(b"\0")
    digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def canonical_manifest_sha256(payload: dict[str, Any]) -> str:
    sealed_payload = {
        key: value
        for key, value in payload.items()
        if key != MANIFEST_SEAL_FIELD
    }
    encoded = json.dumps(
        sealed_payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def seal_manifest_payload(payload: dict[str, Any]) -> dict[str, Any]:
    sealed = dict(payload)
    sealed[MANIFEST_SEAL_FIELD] = canonical_manifest_sha256(sealed)
    return sealed


def _validate_id(value: Any, *, code: str) -> str:
    candidate = str(value or "")
    reserved_stem = candidate.split(".", 1)[0].lower()
    if (
        not MANIFEST_ID_RE.fullmatch(candidate)
        or candidate.endswith(".")
        or ".." in candidate
        or reserved_stem in WINDOWS_RESERVED_STEMS
    ):
        raise InpaintEvalManifestError(code)
    return candidate


def _resolve_private_path(raw_path: Any, manifest_path: Path) -> Path:
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise InpaintEvalManifestError("manifest_path_missing")
    try:
        candidate = Path(raw_path).expanduser()
        if not candidate.is_absolute():
            candidate = manifest_path.parent / candidate
        return candidate.resolve()
    except OSError as exc:
        raise InpaintEvalManifestError("manifest_path_unresolvable") from exc


def _read_image_dimensions(
    path: Path,
    *,
    corpus_id: str,
    page_id: str,
) -> tuple[int, int]:
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            width, height = image.size
            image.load()
    except Exception as exc:
        raise InpaintEvalManifestError(
            "manifest_image_invalid",
            corpus_id=corpus_id,
            page_id=page_id,
        ) from exc
    if width <= 0 or height <= 0:
        raise InpaintEvalManifestError(
            "manifest_image_invalid",
            corpus_id=corpus_id,
            page_id=page_id,
        )
    return int(width), int(height)


def _validate_reference(
    payload: Any,
    *,
    manifest_path: Path,
    corpus_id: str,
    page_id: str,
    expected_dimensions: tuple[int, int] | None,
    expected_size_bytes: int | None,
    required: bool,
) -> EvalImageReference | None:
    if payload is None and not required:
        return None
    if not isinstance(payload, dict):
        raise InpaintEvalManifestError(
            "manifest_reference_invalid",
            corpus_id=corpus_id,
            page_id=page_id,
        )
    if set(payload) - REFERENCE_KEYS:
        raise InpaintEvalManifestError(
            "manifest_reference_unknown_key",
            corpus_id=corpus_id,
            page_id=page_id,
        )
    path = _resolve_private_path(payload.get("path"), manifest_path)
    expected_hash = str(payload.get("sha256") or "").lower()
    if not SHA256_RE.fullmatch(expected_hash):
        raise InpaintEvalManifestError(
            "manifest_hash_invalid",
            corpus_id=corpus_id,
            page_id=page_id,
        )
    try:
        if not path.is_file():
            raise InpaintEvalManifestError(
                "manifest_file_missing",
                corpus_id=corpus_id,
                page_id=page_id,
            )
        if expected_size_bytes is not None and path.stat().st_size != expected_size_bytes:
            raise InpaintEvalManifestError(
                "manifest_size_mismatch",
                corpus_id=corpus_id,
                page_id=page_id,
            )
        if sha256_file(path) != expected_hash:
            raise InpaintEvalManifestError(
                "manifest_hash_mismatch",
                corpus_id=corpus_id,
                page_id=page_id,
            )
    except InpaintEvalManifestError:
        raise
    except OSError as exc:
        raise InpaintEvalManifestError(
            "manifest_file_unreadable",
            corpus_id=corpus_id,
            page_id=page_id,
        ) from exc
    dimensions = _read_image_dimensions(
        path,
        corpus_id=corpus_id,
        page_id=page_id,
    )
    if expected_dimensions is not None and dimensions != expected_dimensions:
        raise InpaintEvalManifestError(
            "manifest_dimension_mismatch",
            corpus_id=corpus_id,
            page_id=page_id,
        )
    return EvalImageReference(path=path, sha256=expected_hash)


def load_eval_manifest(path: Path) -> EvalManifest:
    manifest_path = Path(path).expanduser().resolve()
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise InpaintEvalManifestError("manifest_unreadable") from exc
    if not isinstance(payload, dict):
        raise InpaintEvalManifestError("manifest_root_invalid")
    if set(payload) - MANIFEST_KEYS:
        raise InpaintEvalManifestError("manifest_unknown_key")
    schema_version = payload.get("schema_version")
    if schema_version not in SUPPORTED_MANIFEST_SCHEMA_VERSIONS:
        raise InpaintEvalManifestError("manifest_schema_unsupported")
    corpus_id = _validate_id(payload.get("corpus_id"), code="manifest_corpus_id_invalid")
    split_role = _validate_id(
        payload.get("split_role"),
        code="manifest_split_role_invalid",
    )
    if split_role not in SPLIT_ROLE_VALUES:
        raise InpaintEvalManifestError(
            "manifest_split_role_invalid",
            corpus_id=corpus_id,
        )
    source_lock_git_sha = str(payload.get("source_lock_git_sha") or "").lower()
    if not GIT_SHA_RE.fullmatch(source_lock_git_sha):
        raise InpaintEvalManifestError(
            "manifest_source_lock_invalid",
            corpus_id=corpus_id,
        )
    parent_manifest_sha256 = str(
        payload.get("parent_manifest_sha256") or ""
    ).lower()
    expected_edit_basis = str(payload.get("expected_edit_basis") or "")
    expected_edit_decisions_sha256 = str(
        payload.get("expected_edit_decisions_sha256") or ""
    ).lower()
    evidence_parent_manifest_sha256 = str(
        payload.get("evidence_parent_manifest_sha256") or ""
    ).lower()
    evidence_basis = str(payload.get("evidence_basis") or "")
    evidence_review_sha256 = str(
        payload.get("evidence_review_sha256") or ""
    ).lower()
    finalization_fields_present = tuple(
        bool(value)
        for value in (
            parent_manifest_sha256,
            expected_edit_basis,
            expected_edit_decisions_sha256,
        )
    )
    if any(finalization_fields_present) and not all(finalization_fields_present):
        raise InpaintEvalManifestError(
            "manifest_finalization_incomplete",
            corpus_id=corpus_id,
        )
    if all(finalization_fields_present):
        if not SHA256_RE.fullmatch(parent_manifest_sha256):
            raise InpaintEvalManifestError(
                "manifest_parent_seal_invalid",
                corpus_id=corpus_id,
            )
        if expected_edit_basis != EXPECTED_EDIT_DECISION_BASIS:
            raise InpaintEvalManifestError(
                "manifest_expected_edit_basis_invalid",
                corpus_id=corpus_id,
            )
        if not SHA256_RE.fullmatch(expected_edit_decisions_sha256):
            raise InpaintEvalManifestError(
                "manifest_expected_edit_decisions_seal_invalid",
                corpus_id=corpus_id,
            )
    evidence_fields_present = tuple(
        bool(value)
        for value in (
            evidence_parent_manifest_sha256,
            evidence_basis,
            evidence_review_sha256,
        )
    )
    if any(evidence_fields_present) and not all(evidence_fields_present):
        raise InpaintEvalManifestError(
            "manifest_evidence_finalization_incomplete",
            corpus_id=corpus_id,
        )
    if all(evidence_fields_present):
        if not SHA256_RE.fullmatch(evidence_parent_manifest_sha256):
            raise InpaintEvalManifestError(
                "manifest_evidence_parent_seal_invalid",
                corpus_id=corpus_id,
            )
        if evidence_basis not in {
            SOURCE_ONLY_EVIDENCE_BASIS,
            SOURCE_ONLY_EVIDENCE_BASIS_V2,
            SOURCE_ONLY_ADJUDICATED_EVIDENCE_BASIS_V2,
        }:
            raise InpaintEvalManifestError(
                "manifest_evidence_basis_invalid",
                corpus_id=corpus_id,
            )
        if not SHA256_RE.fullmatch(evidence_review_sha256):
            raise InpaintEvalManifestError(
                "manifest_evidence_review_seal_invalid",
                corpus_id=corpus_id,
            )
    expected_count = payload.get("expected_count")
    if not isinstance(expected_count, int) or isinstance(expected_count, bool) or expected_count <= 0:
        raise InpaintEvalManifestError(
            "manifest_expected_count_invalid",
            corpus_id=corpus_id,
        )
    pages_payload = payload.get("pages")
    if not isinstance(pages_payload, list):
        raise InpaintEvalManifestError(
            "manifest_pages_invalid",
            corpus_id=corpus_id,
        )
    if len(pages_payload) != expected_count:
        raise InpaintEvalManifestError(
            "manifest_count_mismatch",
            corpus_id=corpus_id,
        )
    expected_seal = str(payload.get(MANIFEST_SEAL_FIELD) or "").lower()
    if not SHA256_RE.fullmatch(expected_seal):
        raise InpaintEvalManifestError(
            "manifest_seal_invalid",
            corpus_id=corpus_id,
        )
    if canonical_manifest_sha256(payload) != expected_seal:
        raise InpaintEvalManifestError(
            "manifest_seal_mismatch",
            corpus_id=corpus_id,
        )

    pages: list[EvalPageSpec] = []
    seen_page_ids: set[str] = set()
    for raw_page in pages_payload:
        if not isinstance(raw_page, dict):
            raise InpaintEvalManifestError(
                "manifest_page_invalid",
                corpus_id=corpus_id,
            )
        if set(raw_page) - PAGE_KEYS:
            raise InpaintEvalManifestError(
                "manifest_page_unknown_key",
                corpus_id=corpus_id,
            )
        if schema_version == 1 and (
            "target_text_mask" in raw_page
            or "ambiguous_structure_mask" in raw_page
        ):
            raise InpaintEvalManifestError(
                "manifest_page_schema_key_invalid",
                corpus_id=corpus_id,
            )
        if schema_version == 2 and "target_glyph_mask" in raw_page:
            raise InpaintEvalManifestError(
                "manifest_page_schema_key_invalid",
                corpus_id=corpus_id,
            )
        try:
            page_id = _validate_id(
                raw_page.get("page_id"),
                code="manifest_page_id_invalid",
            )
        except InpaintEvalManifestError as exc:
            raise InpaintEvalManifestError(
                exc.code,
                corpus_id=corpus_id,
            ) from exc
        if page_id in seen_page_ids:
            raise InpaintEvalManifestError(
                "manifest_duplicate_page_id",
                corpus_id=corpus_id,
                page_id=page_id,
            )
        seen_page_ids.add(page_id)
        width = raw_page.get("width")
        height = raw_page.get("height")
        size_bytes = raw_page.get("size_bytes")
        if (
            not isinstance(width, int)
            or isinstance(width, bool)
            or width <= 0
            or not isinstance(height, int)
            or isinstance(height, bool)
            or height <= 0
            or not isinstance(size_bytes, int)
            or isinstance(size_bytes, bool)
            or size_bytes <= 0
        ):
            raise InpaintEvalManifestError(
                "manifest_dimensions_invalid",
                corpus_id=corpus_id,
                page_id=page_id,
            )
        expected_edit = str(raw_page.get("expected_edit") or "")
        if expected_edit not in EXPECTED_EDIT_VALUES:
            raise InpaintEvalManifestError(
                "manifest_expected_edit_invalid",
                corpus_id=corpus_id,
                page_id=page_id,
            )
        dimensions = (int(width), int(height))
        source = _validate_reference(
            {
                "path": raw_page.get("path"),
                "sha256": raw_page.get("sha256"),
            },
            manifest_path=manifest_path,
            corpus_id=corpus_id,
            page_id=page_id,
            expected_dimensions=dimensions,
            expected_size_bytes=int(size_bytes),
            required=True,
        )
        if source is None:
            raise InpaintEvalManifestError(
                "manifest_source_missing",
                corpus_id=corpus_id,
                page_id=page_id,
            )
        target_text_reference = _validate_reference(
            raw_page.get(
                "target_text_mask"
                if schema_version == 2
                else "target_glyph_mask"
            ),
            manifest_path=manifest_path,
            corpus_id=corpus_id,
            page_id=page_id,
            expected_dimensions=dimensions,
            expected_size_bytes=None,
            required=schema_version == 2,
        )
        protected_structure_reference = _validate_reference(
            raw_page.get("protected_structure_mask"),
            manifest_path=manifest_path,
            corpus_id=corpus_id,
            page_id=page_id,
            expected_dimensions=dimensions,
            expected_size_bytes=None,
            required=schema_version == 2,
        )
        ambiguous_structure_reference = _validate_reference(
            raw_page.get("ambiguous_structure_mask"),
            manifest_path=manifest_path,
            corpus_id=corpus_id,
            page_id=page_id,
            expected_dimensions=dimensions,
            expected_size_bytes=None,
            required=schema_version == 2,
        )
        if schema_version == 2:
            annotation_masks = (
                load_binary_mask(target_text_reference, dimensions[::-1]),
                load_binary_mask(protected_structure_reference, dimensions[::-1]),
                load_binary_mask(ambiguous_structure_reference, dimensions[::-1]),
            )
            if any(
                np.any((annotation_masks[left] > 0) & (annotation_masks[right] > 0))
                for left, right in ((0, 1), (0, 2), (1, 2))
            ):
                raise InpaintEvalManifestError(
                    "manifest_annotation_masks_overlap",
                    corpus_id=corpus_id,
                    page_id=page_id,
                )
        pages.append(
            EvalPageSpec(
                corpus_id=corpus_id,
                page_id=page_id,
                source=source,
                width=dimensions[0],
                height=dimensions[1],
                size_bytes=int(size_bytes),
                expected_edit=expected_edit,
                baseline=_validate_reference(
                    raw_page.get("baseline"),
                    manifest_path=manifest_path,
                    corpus_id=corpus_id,
                    page_id=page_id,
                    expected_dimensions=dimensions,
                    expected_size_bytes=None,
                    required=False,
                ),
                baseline_mask=_validate_reference(
                    raw_page.get("baseline_mask"),
                    manifest_path=manifest_path,
                    corpus_id=corpus_id,
                    page_id=page_id,
                    expected_dimensions=dimensions,
                    expected_size_bytes=None,
                    required=False,
                ),
                target_text_mask=target_text_reference,
                protected_structure_mask=protected_structure_reference,
                ambiguous_structure_mask=ambiguous_structure_reference,
            )
        )
    if all(finalization_fields_present) and any(
        page.expected_edit == "optional" for page in pages
    ):
        raise InpaintEvalManifestError(
            "manifest_finalization_optional_remaining",
            corpus_id=corpus_id,
        )
    return EvalManifest(
        schema_version=int(schema_version),
        corpus_id=corpus_id,
        split_role=split_role,
        source_lock_git_sha=source_lock_git_sha,
        expected_count=expected_count,
        manifest_sha256=expected_seal,
        pages=tuple(pages),
        parent_manifest_sha256=(
            parent_manifest_sha256 if all(finalization_fields_present) else None
        ),
        expected_edit_basis=(
            expected_edit_basis if all(finalization_fields_present) else None
        ),
        expected_edit_decisions_sha256=(
            expected_edit_decisions_sha256
            if all(finalization_fields_present)
            else None
        ),
        evidence_parent_manifest_sha256=(
            evidence_parent_manifest_sha256
            if all(evidence_fields_present)
            else None
        ),
        evidence_basis=evidence_basis if all(evidence_fields_present) else None,
        evidence_review_sha256=(
            evidence_review_sha256 if all(evidence_fields_present) else None
        ),
    )


def finalize_optional_eval_manifest(
    parent_manifest_path: Path,
    decisions_path: Path,
    output_path: Path,
) -> EvalManifest:
    """Seal source-only required/none decisions into a derived manifest."""
    parent_path = Path(parent_manifest_path).expanduser().resolve()
    decision_file = Path(decisions_path).expanduser().resolve()
    destination = Path(output_path).expanduser().resolve()
    if destination == parent_path:
        raise InpaintEvalManifestError("manifest_finalization_overwrite_parent")
    if destination.exists():
        raise InpaintEvalManifestError("manifest_finalization_output_exists")
    if destination.parent != parent_path.parent:
        raise InpaintEvalManifestError("manifest_finalization_output_directory_invalid")

    try:
        parent_bytes = parent_path.read_bytes()
        parent_payload = json.loads(parent_bytes.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise InpaintEvalManifestError("manifest_unreadable") from exc
    parent = load_eval_manifest(parent_path)
    if (
        not isinstance(parent_payload, dict)
        or str(parent_payload.get(MANIFEST_SEAL_FIELD) or "").lower()
        != parent.manifest_sha256
        or canonical_manifest_sha256(parent_payload)
        != parent.manifest_sha256
    ):
        raise InpaintEvalManifestError(
            "manifest_finalization_parent_changed",
            corpus_id=parent.corpus_id,
        )
    if parent.parent_manifest_sha256 is not None:
        raise InpaintEvalManifestError(
            "manifest_already_finalized",
            corpus_id=parent.corpus_id,
        )
    optional_page_ids = {
        page.page_id
        for page in parent.pages
        if page.expected_edit == "optional"
    }
    if not optional_page_ids:
        raise InpaintEvalManifestError(
            "manifest_finalization_no_optional_pages",
            corpus_id=parent.corpus_id,
        )

    try:
        decision_bytes = decision_file.read_bytes()
        decision_payload = json.loads(decision_bytes.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise InpaintEvalManifestError(
            "manifest_finalization_decisions_unreadable",
            corpus_id=parent.corpus_id,
        ) from exc
    if not isinstance(decision_payload, dict) or set(decision_payload) != {
        "schema_version",
        "parent_manifest_sha256",
        "decision_basis",
        "pages",
    }:
        raise InpaintEvalManifestError(
            "manifest_finalization_decisions_schema_invalid",
            corpus_id=parent.corpus_id,
        )
    if decision_payload.get("schema_version") != 1:
        raise InpaintEvalManifestError(
            "manifest_finalization_decisions_schema_invalid",
            corpus_id=parent.corpus_id,
        )
    if str(decision_payload.get("parent_manifest_sha256") or "").lower() != (
        parent.manifest_sha256
    ):
        raise InpaintEvalManifestError(
            "manifest_finalization_parent_mismatch",
            corpus_id=parent.corpus_id,
        )
    if decision_payload.get("decision_basis") != EXPECTED_EDIT_DECISION_BASIS:
        raise InpaintEvalManifestError(
            "manifest_finalization_basis_invalid",
            corpus_id=parent.corpus_id,
        )
    decision_rows = decision_payload.get("pages")
    if not isinstance(decision_rows, list):
        raise InpaintEvalManifestError(
            "manifest_finalization_decisions_invalid",
            corpus_id=parent.corpus_id,
        )
    decisions: dict[str, str] = {}
    for row in decision_rows:
        if not isinstance(row, dict) or set(row) != {"page_id", "expected_edit"}:
            raise InpaintEvalManifestError(
                "manifest_finalization_decision_invalid",
                corpus_id=parent.corpus_id,
            )
        try:
            page_id = _validate_id(
                row.get("page_id"),
                code="manifest_finalization_page_id_invalid",
            )
        except InpaintEvalManifestError as exc:
            raise InpaintEvalManifestError(
                exc.code,
                corpus_id=parent.corpus_id,
            ) from exc
        expected_edit = str(row.get("expected_edit") or "")
        if page_id in decisions:
            raise InpaintEvalManifestError(
                "manifest_finalization_duplicate_page",
                corpus_id=parent.corpus_id,
                page_id=page_id,
            )
        if expected_edit not in FINAL_EXPECTED_EDIT_VALUES:
            raise InpaintEvalManifestError(
                "manifest_finalization_expected_edit_invalid",
                corpus_id=parent.corpus_id,
                page_id=page_id,
            )
        decisions[page_id] = expected_edit
    if set(decisions) != optional_page_ids:
        raise InpaintEvalManifestError(
            "manifest_finalization_page_set_mismatch",
            corpus_id=parent.corpus_id,
        )

    derived_payload = {
        key: value
        for key, value in parent_payload.items()
        if key != MANIFEST_SEAL_FIELD
    }
    for page_payload in derived_payload["pages"]:
        page_id = str(page_payload.get("page_id") or "")
        if page_id in decisions:
            page_payload["expected_edit"] = decisions[page_id]
    derived_payload.update(
        {
            "parent_manifest_sha256": parent.manifest_sha256,
            "expected_edit_basis": EXPECTED_EDIT_DECISION_BASIS,
            "expected_edit_decisions_sha256": hashlib.sha256(
                decision_bytes
            ).hexdigest(),
        }
    )
    sealed_payload = seal_manifest_payload(derived_payload)
    serialized_payload = (
        json.dumps(sealed_payload, ensure_ascii=False, indent=2) + "\n"
    )
    temporary_path: Path | None = None
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            text=True,
        )
        temporary_path = Path(temporary_name)
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(serialized_payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary_path, destination)
    except FileExistsError as exc:
        raise InpaintEvalManifestError(
            "manifest_finalization_output_exists",
            corpus_id=parent.corpus_id,
        ) from exc
    except OSError as exc:
        raise InpaintEvalManifestError(
            "manifest_finalization_output_unwritable",
            corpus_id=parent.corpus_id,
        ) from exc
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass
    return load_eval_manifest(destination)


def seal_source_only_evidence_manifest(
    parent_manifest_path: Path,
    review_path: Path,
    output_path: Path,
) -> EvalManifest:
    """Seal reviewed baselines and source-only structure masks into a new manifest."""
    parent_path = Path(parent_manifest_path).expanduser().resolve()
    review_file = Path(review_path).expanduser().resolve()
    destination = Path(output_path).expanduser().resolve()
    if destination == parent_path:
        raise InpaintEvalManifestError("manifest_evidence_overwrite_parent")
    if destination.exists():
        raise InpaintEvalManifestError("manifest_evidence_output_exists")
    if destination.parent != parent_path.parent:
        raise InpaintEvalManifestError("manifest_evidence_output_directory_invalid")

    try:
        parent_bytes = parent_path.read_bytes()
        parent_payload = json.loads(parent_bytes.decode("utf-8"))
        review_bytes = review_file.read_bytes()
        review_payload = json.loads(review_bytes.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise InpaintEvalManifestError("manifest_evidence_input_unreadable") from exc
    parent = load_eval_manifest(parent_path)
    if (
        not isinstance(parent_payload, dict)
        or str(parent_payload.get(MANIFEST_SEAL_FIELD) or "").lower()
        != parent.manifest_sha256
        or canonical_manifest_sha256(parent_payload) != parent.manifest_sha256
    ):
        raise InpaintEvalManifestError(
            "manifest_evidence_parent_changed",
            corpus_id=parent.corpus_id,
        )
    if parent.evidence_parent_manifest_sha256 is not None:
        raise InpaintEvalManifestError(
            "manifest_evidence_already_finalized",
            corpus_id=parent.corpus_id,
        )
    if not isinstance(review_payload, dict) or set(review_payload) != {
        "schema_version",
        "parent_manifest_sha256",
        "decision_basis",
        "pages",
    }:
        raise InpaintEvalManifestError(
            "manifest_evidence_review_schema_invalid",
            corpus_id=parent.corpus_id,
        )
    review_schema_version = review_payload.get("schema_version")
    review_basis = review_payload.get("decision_basis")
    if (
        review_schema_version not in SUPPORTED_MANIFEST_SCHEMA_VERSIONS
        or review_basis
        != (
            SOURCE_ONLY_EVIDENCE_BASIS_V2
            if review_schema_version == 2
            else SOURCE_ONLY_EVIDENCE_BASIS
        )
        or str(review_payload.get("parent_manifest_sha256") or "").lower()
        != parent.manifest_sha256
    ):
        raise InpaintEvalManifestError(
            "manifest_evidence_review_contract_invalid",
            corpus_id=parent.corpus_id,
        )
    review_rows = review_payload.get("pages")
    if not isinstance(review_rows, list):
        raise InpaintEvalManifestError(
            "manifest_evidence_review_pages_invalid",
            corpus_id=parent.corpus_id,
        )

    parent_pages = {page.page_id: page for page in parent.pages}
    reviewed: dict[str, dict[str, dict[str, str]]] = {}
    required_row_keys = {
        "page_id",
        "baseline",
        "baseline_mask",
        "protected_structure_mask",
    }
    if review_schema_version == 2:
        required_row_keys |= {
            "target_text_mask",
            "ambiguous_structure_mask",
        }
        allowed_row_keys = required_row_keys
    else:
        allowed_row_keys = required_row_keys | {"target_glyph_mask"}
    for row in review_rows:
        if (
            not isinstance(row, dict)
            or not required_row_keys.issubset(row)
            or set(row) - allowed_row_keys
        ):
            raise InpaintEvalManifestError(
                "manifest_evidence_review_page_invalid",
                corpus_id=parent.corpus_id,
            )
        page_id = _validate_id(
            row.get("page_id"),
            code="manifest_evidence_review_page_id_invalid",
        )
        if page_id in reviewed or page_id not in parent_pages:
            raise InpaintEvalManifestError(
                "manifest_evidence_review_page_set_mismatch",
                corpus_id=parent.corpus_id,
                page_id=page_id,
            )
        page = parent_pages[page_id]
        references: dict[str, dict[str, str]] = {}
        annotation_fields = (
            (
                "target_text_mask",
                "protected_structure_mask",
                "ambiguous_structure_mask",
            )
            if review_schema_version == 2
            else ("protected_structure_mask", "target_glyph_mask")
        )
        for field_name in ("baseline", "baseline_mask", *annotation_fields):
            if field_name not in row:
                continue
            reference = _validate_reference(
                row.get(field_name),
                manifest_path=review_file,
                corpus_id=parent.corpus_id,
                page_id=page_id,
                expected_dimensions=(page.width, page.height),
                expected_size_bytes=None,
                required=(
                    review_schema_version == 2
                    or field_name != "target_glyph_mask"
                ),
            )
            if reference is not None:
                references[field_name] = {
                    "path": str(reference.path),
                    "sha256": reference.sha256,
                }
        reviewed[page_id] = references
    if set(reviewed) != set(parent_pages):
        raise InpaintEvalManifestError(
            "manifest_evidence_review_page_set_mismatch",
            corpus_id=parent.corpus_id,
        )

    derived_payload = {
        key: value
        for key, value in parent_payload.items()
        if key != MANIFEST_SEAL_FIELD
    }
    if review_schema_version == 2:
        derived_payload["schema_version"] = 2
    for page_payload in derived_payload["pages"]:
        page_id = str(page_payload.get("page_id") or "")
        if review_schema_version == 2:
            page_payload.pop("target_glyph_mask", None)
        page_payload.update(reviewed[page_id])
    derived_payload.update(
        {
            "evidence_parent_manifest_sha256": parent.manifest_sha256,
            "evidence_basis": review_basis,
            "evidence_review_sha256": hashlib.sha256(review_bytes).hexdigest(),
        }
    )
    sealed_payload = seal_manifest_payload(derived_payload)
    serialized_payload = (
        json.dumps(sealed_payload, ensure_ascii=False, indent=2) + "\n"
    )
    temporary_path: Path | None = None
    try:
        if parent_path.read_bytes() != parent_bytes:
            raise InpaintEvalManifestError(
                "manifest_evidence_parent_changed",
                corpus_id=parent.corpus_id,
            )
        descriptor, temporary_name = tempfile.mkstemp(
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            text=True,
        )
        temporary_path = Path(temporary_name)
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(serialized_payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary_path, destination)
    except InpaintEvalManifestError:
        raise
    except FileExistsError as exc:
        raise InpaintEvalManifestError(
            "manifest_evidence_output_exists",
            corpus_id=parent.corpus_id,
        ) from exc
    except OSError as exc:
        raise InpaintEvalManifestError(
            "manifest_evidence_output_unwritable",
            corpus_id=parent.corpus_id,
        ) from exc
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass
    return load_eval_manifest(destination)


def load_eval_manifests(paths: Iterable[Path]) -> tuple[EvalManifest, ...]:
    manifests = tuple(load_eval_manifest(path) for path in paths)
    seen_corpora: set[str] = set()
    seen_pages: set[str] = set()
    seen_source_hashes: set[str] = set()
    source_locks: set[str] = set()
    for manifest in manifests:
        if manifest.corpus_id in seen_corpora:
            raise InpaintEvalManifestError(
                "manifest_duplicate_corpus_id",
                corpus_id=manifest.corpus_id,
            )
        seen_corpora.add(manifest.corpus_id)
        source_locks.add(manifest.source_lock_git_sha)
        for page in manifest.pages:
            if page.page_id in seen_pages:
                raise InpaintEvalManifestError(
                    "manifest_duplicate_page_id_global",
                    corpus_id=page.corpus_id,
                    page_id=page.page_id,
                )
            seen_pages.add(page.page_id)
            if page.source.sha256 in seen_source_hashes:
                raise InpaintEvalManifestError(
                    "manifest_duplicate_source_hash",
                    corpus_id=page.corpus_id,
                    page_id=page.page_id,
                )
            seen_source_hashes.add(page.source.sha256)
    if len(source_locks) > 1:
        raise InpaintEvalManifestError("manifest_source_lock_mismatch")
    return manifests


def load_rgb_reference_array(
    reference: EvalImageReference,
    shape: tuple[int, ...],
) -> np.ndarray:
    try:
        data = reference.path.read_bytes()
    except OSError as exc:
        raise ValueError("evaluation_reference_unreadable") from exc
    if hashlib.sha256(data).hexdigest() != reference.sha256:
        raise ValueError("evaluation_reference_hash_mismatch")
    try:
        with Image.open(BytesIO(data)) as image:
            array = np.asarray(image.convert("RGB")).copy()
    except Exception as exc:
        raise ValueError("evaluation_reference_image_invalid") from exc
    if array.shape[:2] != shape[:2]:
        raise ValueError("evaluation_reference_shape_mismatch")
    return array


def verify_eval_page_spec(page: EvalPageSpec) -> None:
    source = _validate_reference(
        {"path": str(page.source.path), "sha256": page.source.sha256},
        manifest_path=page.source.path,
        corpus_id=page.corpus_id,
        page_id=page.page_id,
        expected_dimensions=(page.width, page.height),
        expected_size_bytes=page.size_bytes,
        required=True,
    )
    if source is None:
        raise InpaintEvalManifestError(
            "manifest_source_missing",
            corpus_id=page.corpus_id,
            page_id=page.page_id,
        )
    for reference in (
        page.baseline,
        page.baseline_mask,
        page.target_text_mask,
        page.protected_structure_mask,
        page.ambiguous_structure_mask,
    ):
        if reference is None:
            continue
        _validate_reference(
            {"path": str(reference.path), "sha256": reference.sha256},
            manifest_path=reference.path,
            corpus_id=page.corpus_id,
            page_id=page.page_id,
            expected_dimensions=(page.width, page.height),
            expected_size_bytes=None,
            required=True,
        )


def load_eval_source_array(page: EvalPageSpec) -> np.ndarray:
    try:
        data = page.source.path.read_bytes()
    except OSError as exc:
        raise InpaintEvalManifestError(
            "manifest_file_unreadable",
            corpus_id=page.corpus_id,
            page_id=page.page_id,
        ) from exc
    if len(data) != page.size_bytes:
        raise InpaintEvalManifestError(
            "manifest_size_mismatch",
            corpus_id=page.corpus_id,
            page_id=page.page_id,
        )
    if hashlib.sha256(data).hexdigest() != page.source.sha256:
        raise InpaintEvalManifestError(
            "manifest_hash_mismatch",
            corpus_id=page.corpus_id,
            page_id=page.page_id,
        )
    try:
        with Image.open(BytesIO(data)) as image:
            if image.size != (page.width, page.height):
                raise InpaintEvalManifestError(
                    "manifest_dimension_mismatch",
                    corpus_id=page.corpus_id,
                    page_id=page.page_id,
                )
            return np.asarray(image.convert("RGB")).copy()
    except InpaintEvalManifestError:
        raise
    except Exception as exc:
        raise InpaintEvalManifestError(
            "manifest_image_invalid",
            corpus_id=page.corpus_id,
            page_id=page.page_id,
        ) from exc


def _normalize_binary_mask(mask: np.ndarray | None, shape: tuple[int, ...]) -> np.ndarray:
    if mask is None:
        return np.zeros(shape[:2], dtype=np.uint8)
    array = np.asarray(mask)
    if array.ndim == 3:
        array = array[:, :, 0]
    if array.shape[:2] != shape[:2]:
        raise ValueError("evaluation_mask_shape_mismatch")
    return np.where(array > 0, 255, 0).astype(np.uint8)


def load_binary_mask(reference: EvalImageReference | None, shape: tuple[int, ...]) -> np.ndarray:
    if reference is None:
        return np.zeros(shape[:2], dtype=np.uint8)
    try:
        data = reference.path.read_bytes()
    except OSError as exc:
        raise ValueError("evaluation_reference_unreadable") from exc
    if hashlib.sha256(data).hexdigest() != reference.sha256:
        raise ValueError("evaluation_reference_hash_mismatch")
    try:
        with Image.open(BytesIO(data)) as image:
            return _normalize_binary_mask(np.asarray(image.convert("L")), shape)
    except ValueError as exc:
        if str(exc) == "evaluation_mask_shape_mismatch":
            raise
        raise ValueError("evaluation_reference_image_invalid") from exc
    except Exception as exc:
        raise ValueError("evaluation_reference_image_invalid") from exc


def _to_gray(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim == 2:
        return array.astype(np.uint8, copy=False)
    return cv2.cvtColor(array[:, :, :3], cv2.COLOR_RGB2GRAY)


def build_quality_metrics(
    source_image: np.ndarray,
    candidate_image: np.ndarray,
    final_mask: np.ndarray | None,
    *,
    residue_target_mask: np.ndarray | None = None,
    residue_target_is_annotation: bool = False,
    protected_structure_mask: np.ndarray | None = None,
    pre_composite_candidate_image: np.ndarray | None = None,
) -> dict[str, object]:
    source = np.asarray(source_image)
    candidate = np.asarray(candidate_image)
    if source.shape != candidate.shape:
        raise ValueError("evaluation_image_shape_mismatch")
    if source.ndim != 3 or source.shape[2] < 3:
        raise ValueError("evaluation_image_channels_invalid")
    pre_composite = (
        np.asarray(pre_composite_candidate_image)
        if pre_composite_candidate_image is not None
        else candidate
    )
    if pre_composite.shape != source.shape:
        raise ValueError("evaluation_pre_composite_shape_mismatch")
    mask = _normalize_binary_mask(final_mask, source.shape)
    target = _normalize_binary_mask(residue_target_mask, source.shape)
    target_available = bool(np.any(target))
    protected = _normalize_binary_mask(protected_structure_mask, source.shape)

    changed_exact = np.any(candidate[:, :, :3] != source[:, :, :3], axis=2)
    pre_composite_changed_exact = np.any(
        pre_composite[:, :, :3] != source[:, :, :3],
        axis=2,
    )
    outside = mask <= 0
    outside_pixel_count = int(np.count_nonzero(outside))
    outside_changed = int(np.count_nonzero(changed_exact & outside))
    pre_composite_outside_changed = int(
        np.count_nonzero(pre_composite_changed_exact & outside)
    )

    delta = np.abs(candidate.astype(np.int16) - source.astype(np.int16))
    if delta.ndim == 3:
        delta_per_pixel = np.mean(delta[:, :, :3], axis=2)
    else:
        delta_per_pixel = delta.astype(np.float32)
    inside_delta = delta_per_pixel[mask > 0]
    color_delta_mean = float(np.mean(inside_delta)) if inside_delta.size else None
    color_delta_p95 = float(np.percentile(inside_delta, 95)) if inside_delta.size else None

    if target_available:
        source_gray = _to_gray(source)
        candidate_gray = _to_gray(candidate)
        source_background = cv2.GaussianBlur(source_gray, (15, 15), 0)
        candidate_background = cv2.GaussianBlur(candidate_gray, (15, 15), 0)
        source_contrast = np.abs(
            source_gray.astype(np.int16) - source_background.astype(np.int16)
        )
        candidate_contrast = np.abs(
            candidate_gray.astype(np.int16) - candidate_background.astype(np.int16)
        )
        residue_source = (target > 0) & (source_contrast >= 8)
        residue_threshold = np.maximum(
            8.0,
            source_contrast.astype(np.float32) * 0.35,
        )
        residue_pixels = residue_source & (
            candidate_contrast.astype(np.float32) >= residue_threshold
        )
        residue_source_count = int(np.count_nonzero(residue_source))
        residue_pixel_count = int(np.count_nonzero(residue_pixels))
    else:
        residue_source = np.zeros(source.shape[:2], dtype=bool)
        source_contrast = np.zeros(source.shape[:2], dtype=np.int16)
        candidate_contrast = np.zeros(source.shape[:2], dtype=np.int16)
        residue_source_count = 0
        residue_pixel_count = 0
    if residue_source_count > 0:
        contrast_ratio = np.minimum(
            candidate_contrast[residue_source].astype(np.float32)
            / np.maximum(source_contrast[residue_source].astype(np.float32), 1.0),
            1.0,
        )
        residue_score = float(np.mean(contrast_ratio))
    else:
        residue_score = None

    target_pixel_count = int(np.count_nonzero(target))
    target_covered_count = int(np.count_nonzero((target > 0) & (mask > 0)))
    target_component_coverages: list[float] = []
    if target_pixel_count > 0 and residue_target_is_annotation:
        component_count, component_labels, component_stats, _centroids = (
            cv2.connectedComponentsWithStats(
                (target > 0).astype(np.uint8),
                connectivity=8,
            )
        )
        for component_index in range(1, component_count):
            component_pixel_count = int(
                component_stats[component_index, cv2.CC_STAT_AREA]
            )
            if component_pixel_count <= 0:
                continue
            x = int(component_stats[component_index, cv2.CC_STAT_LEFT])
            y = int(component_stats[component_index, cv2.CC_STAT_TOP])
            width = int(component_stats[component_index, cv2.CC_STAT_WIDTH])
            height = int(component_stats[component_index, cv2.CC_STAT_HEIGHT])
            component = (
                component_labels[y : y + height, x : x + width]
                == component_index
            )
            local_mask = mask[y : y + height, x : x + width] > 0
            target_component_coverages.append(
                float(np.count_nonzero(component & local_mask))
                / float(component_pixel_count)
            )
    protected_pixel_count = int(np.count_nonzero(protected))
    protected_changed_count = int(np.count_nonzero((protected > 0) & changed_exact))
    pre_composite_protected_changed_count = int(
        np.count_nonzero((protected > 0) & pre_composite_changed_exact)
    )

    return {
        "outside_pixel_count": outside_pixel_count,
        "outside_changed_pixel_count_exact": outside_changed,
        "outside_change_ratio": (
            float(outside_changed / outside_pixel_count)
            if outside_pixel_count > 0
            else None
        ),
        "pre_composite_outside_changed_pixel_count_exact": pre_composite_outside_changed,
        "pre_composite_outside_change_ratio": (
            float(pre_composite_outside_changed / outside_pixel_count)
            if outside_pixel_count > 0
            else None
        ),
        "color_delta_mean": color_delta_mean,
        "color_delta_p95": color_delta_p95,
        "color_delta_score": (
            float(color_delta_mean / 255.0)
            if color_delta_mean is not None
            else None
        ),
        "residue_target_pixel_count": target_pixel_count,
        "residue_target_covered_pixel_count": target_covered_count,
        "residue_target_is_annotation": bool(residue_target_is_annotation),
        "residue_target_source": (
            "private_annotation"
            if residue_target_is_annotation
            else ("derived_mask" if target_available else "unavailable")
        ),
        "residue_target_coverage": (
            float(target_covered_count / target_pixel_count)
            if target_pixel_count > 0 and residue_target_is_annotation
            else None
        ),
        "residue_target_component_coverages": target_component_coverages,
        "residue_target_minimum_component_coverage": (
            min(target_component_coverages)
            if target_component_coverages
            else None
        ),
        "residue_source_contrast_pixel_count": residue_source_count,
        "residue_pixel_count": residue_pixel_count,
        "residue_ratio": (
            float(residue_pixel_count / residue_source_count)
            if residue_source_count > 0
            else None
        ),
        "residue_score": residue_score,
        "residue_score_sum": (
            float(residue_score * residue_source_count)
            if residue_score is not None
            else None
        ),
        "protected_structure_pixel_count": protected_pixel_count,
        "protected_structure_changed_pixel_count_exact": protected_changed_count,
        "outline_damage_ratio": (
            float(protected_changed_count / protected_pixel_count)
            if protected_pixel_count > 0
            else None
        ),
        "pre_composite_protected_structure_changed_pixel_count_exact": (
            pre_composite_protected_changed_count
        ),
        "pre_composite_outline_damage_ratio": (
            float(pre_composite_protected_changed_count / protected_pixel_count)
            if protected_pixel_count > 0
            else None
        ),
    }


def _load_rgb(path: Path, *, expected_sha256: str | None = None) -> Image.Image:
    data = Path(path).read_bytes()
    if expected_sha256 is not None and hashlib.sha256(data).hexdigest() != expected_sha256:
        raise ValueError("evaluation_reference_hash_mismatch")
    with Image.open(BytesIO(data)) as image:
        return image.convert("RGB")


def _fit_panel(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    fitted = image.copy()
    fitted.thumbnail(size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, "white")
    canvas.paste(
        fitted,
        ((size[0] - fitted.width) // 2, (size[1] - fitted.height) // 2),
    )
    return canvas


def _labelled_panel(image: Image.Image, label: str, size: tuple[int, int]) -> Image.Image:
    label_height = 28
    panel = Image.new("RGB", (size[0], size[1] + label_height), "white")
    panel.paste(_fit_panel(image, size), (0, 0))
    ImageDraw.Draw(panel).text((6, size[1] + 7), label, fill="black")
    return panel


def _mask_delta_image(
    source: Image.Image,
    candidate: Image.Image,
    final_mask: Image.Image,
) -> Image.Image:
    source_arr = np.asarray(source.convert("RGB"))
    candidate_arr = np.asarray(candidate.convert("RGB"))
    mask_arr = np.asarray(final_mask.convert("L")) > 0
    changed = np.any(source_arr != candidate_arr, axis=2)
    visualization = np.zeros_like(source_arr)
    visualization[mask_arr] = (235, 235, 235)
    visualization[changed & mask_arr] = (220, 50, 47)
    visualization[changed & ~mask_arr] = (40, 90, 220)
    return Image.fromarray(visualization, mode="RGB")


def _build_blind_panel(
    baseline: Image.Image,
    candidate: Image.Image,
    *,
    candidate_is_a: bool,
    panel_size: tuple[int, int],
) -> Image.Image:
    blind_images = (candidate, baseline) if candidate_is_a else (baseline, candidate)
    blind = Image.new("RGB", (panel_size[0] * 2, panel_size[1] + 28), "white")
    blind.paste(_labelled_panel(blind_images[0], "A", panel_size), (0, 0))
    blind.paste(
        _labelled_panel(blind_images[1], "B", panel_size),
        (panel_size[0], 0),
    )
    return blind


def derive_blind_review_seed(
    manifests: Iterable[EvalManifest],
    candidate_sha256s: Iterable[str] = (),
) -> str:
    """Derive a deterministic seed from sealed inputs and candidate artifacts."""
    seals = sorted(
        f"{manifest.corpus_id}:{manifest.manifest_sha256}"
        for manifest in manifests
    )
    if not seals:
        raise ValueError("blind_review_seed_unavailable")
    candidate_seals = sorted(str(value or "").lower() for value in candidate_sha256s)
    if any(not SHA256_RE.fullmatch(value) for value in candidate_seals):
        raise ValueError("blind_review_candidate_hash_invalid")
    payload = (
        f"{BLIND_REVIEW_VERSION}:seed:"
        + ":".join(seals)
        + ":candidates:"
        + ":".join(candidate_seals)
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _seeded_blind_digest(seed: str, purpose: str, identity: str) -> bytes:
    normalized_seed = str(seed or "").lower()
    if not SHA256_RE.fullmatch(normalized_seed):
        raise ValueError("blind_review_seed_invalid")
    return hashlib.sha256(
        f"{BLIND_REVIEW_VERSION}:{normalized_seed}:{purpose}:{identity}".encode(
            "utf-8"
        )
    ).digest()


def _save_anonymous_review_panel(
    panel: Image.Image,
    path: Path,
    *,
    review_id: str,
) -> None:
    metadata = PngImagePlugin.PngInfo()
    metadata.add_text("review_id", review_id)
    panel.save(
        path,
        format="PNG",
        optimize=False,
        pnginfo=metadata,
    )


def write_comparison_and_blind_panels(
    *,
    root_output: Path,
    corpus_id: str,
    page_id: str,
    source_path: Path,
    baseline_path: Path | None,
    baseline_sha256: str | None = None,
    candidate_path: Path,
    final_mask_path: Path,
) -> dict[str, Any]:
    source = _load_rgb(source_path)
    baseline_available = baseline_path is not None
    baseline = (
        _load_rgb(baseline_path, expected_sha256=baseline_sha256)
        if baseline_path is not None
        else Image.new("RGB", source.size, (224, 224, 224))
    )
    candidate = _load_rgb(candidate_path)
    with Image.open(final_mask_path) as opened_mask:
        final_mask = opened_mask.convert("L")
    delta = _mask_delta_image(source, candidate, final_mask)
    panel_size = (420, 560)

    comparison = Image.new("RGB", (panel_size[0] * 4, panel_size[1] + 28), "white")
    for index, (image, label) in enumerate(
        (
            (source, "ORIGINAL"),
            (baseline, "BASELINE" if baseline_available else "BASELINE_UNAVAILABLE"),
            (candidate, "CANDIDATE"),
            (delta, "MASK_DELTA"),
        )
    ):
        comparison.paste(_labelled_panel(image, label, panel_size), (index * panel_size[0], 0))
    comparison_dir = root_output / "comparison_panels" / corpus_id
    comparison_dir.mkdir(parents=True, exist_ok=True)
    comparison_path = comparison_dir / f"{page_id}_comparison.png"
    comparison.save(comparison_path, format="PNG", optimize=False)

    if not baseline_available:
        return {
            "corpus_id": corpus_id,
            "page_id": page_id,
            "comparison_panel": comparison_path,
            "blind_eligible": False,
        }

    return {
        "corpus_id": corpus_id,
        "page_id": page_id,
        "comparison_panel": comparison_path,
        "blind_eligible": True,
        "baseline_path": baseline_path,
        "baseline_sha256": baseline_sha256 or sha256_file(baseline_path),
        "candidate_path": candidate_path,
        "candidate_sha256": sha256_file(candidate_path),
        "panel_size": panel_size,
    }


def select_blind_review_duplicates(
    records: Iterable[dict[str, Any]],
    count: int,
    *,
    assignment_seed: str,
) -> list[dict[str, Any]]:
    candidates = list(records)
    if count < 0 or count > len(candidates):
        raise ValueError("blind_duplicate_count_out_of_range")
    ranked = sorted(
        candidates,
        key=lambda record: _seeded_blind_digest(
            assignment_seed,
            "duplicate",
            f"{record['corpus_id']}:{record['page_id']}",
        ),
    )
    return ranked[:count]


def write_blind_review_jsonl(
    root_output: Path,
    panel_records: Iterable[dict[str, Any]],
    *,
    duplicate_count: int = 0,
    assignment_seed: str,
) -> tuple[Path, Path]:
    root = Path(root_output)
    review_dir = root / "review"
    key_dir = root / "blind_keys"
    records = sorted(
        list(panel_records),
        key=lambda record: (str(record["corpus_id"]), str(record["page_id"])),
    )
    prepared_images = {
        id(record): (
            _load_rgb(
                Path(record["baseline_path"]),
                expected_sha256=record.get("baseline_sha256"),
            ),
            _load_rgb(
                Path(record["candidate_path"]),
                expected_sha256=record.get("candidate_sha256"),
            ),
        )
        for record in records
    }
    if review_dir.exists() or key_dir.exists():
        raise FileExistsError("blind_review_output_exists")
    review_dir.mkdir(parents=True, exist_ok=False)
    key_dir.mkdir(parents=True, exist_ok=False)
    duplicates = select_blind_review_duplicates(
        records,
        int(duplicate_count),
        assignment_seed=assignment_seed,
    )
    entries: list[dict[str, Any]] = [
        {
            "identity": f"original:{record['corpus_id']}:{record['page_id']}",
            "record": record,
            "duplicate_of": None,
        }
        for record in records
    ]
    for index, record in enumerate(duplicates, start=1):
        entries.append(
            {
                "identity": (
                    f"duplicate:{index}:{record['corpus_id']}:{record['page_id']}"
                ),
                "record": record,
                "duplicate_of": f"{record['corpus_id']}:{record['page_id']}",
            }
        )
    entries.sort(
        key=lambda entry: _seeded_blind_digest(
            assignment_seed,
            "order",
            entry["identity"],
        )
    )
    review_rows: list[dict[str, Any]] = []
    key_rows: list[dict[str, Any]] = []
    review_panel_dir = review_dir / "panels"
    review_panel_dir.mkdir(parents=True, exist_ok=True)
    for sequence, entry in enumerate(entries, start=1):
        record = entry["record"]
        review_id = f"review-{sequence:04d}"
        review_panel = review_panel_dir / f"{review_id}.png"
        is_duplicate = entry["duplicate_of"] is not None
        baseline, candidate = prepared_images[id(record)]
        original_candidate_is_a = bool(
            _seeded_blind_digest(
                assignment_seed,
                "assignment",
                f"{record['corpus_id']}:{record['page_id']}",
            )[0]
            & 1
        )
        candidate_is_a = (
            not original_candidate_is_a
            if is_duplicate
            else original_candidate_is_a
        )
        blind_panel = _build_blind_panel(
            baseline,
            candidate,
            candidate_is_a=candidate_is_a,
            panel_size=tuple(record["panel_size"]),
        )
        _save_anonymous_review_panel(
            blind_panel,
            review_panel,
            review_id=review_id,
        )
        candidate_label = "A" if candidate_is_a else "B"
        review_rows.append(
            {
                "review_id": review_id,
                "panel": review_panel.relative_to(review_dir).as_posix(),
                "preferred": "",
                "major_defect": "",
                "minor_defect": "",
                "undetected_text": "",
                "notes": "",
            }
        )
        key_rows.append(
            {
                "review_id": review_id,
                "corpus_id": record["corpus_id"],
                "page_id": record["page_id"],
                "candidate_label": candidate_label,
                "baseline_label": "B" if candidate_label == "A" else "A",
                "duplicate_of": entry["duplicate_of"],
            }
        )

    review_path = review_dir / "blind-review.jsonl"
    review_contract_path = review_dir / "review-contract.json"
    key_path = key_dir / "blind-key.jsonl"
    review_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in review_rows),
        encoding="utf-8",
    )
    key_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in key_rows),
        encoding="utf-8",
    )
    review_contract_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "reviewer_package": "review",
                "deterministic_assignment": True,
                "share_only": [
                    "blind-review.jsonl",
                    "panels",
                    "review-contract.json",
                ],
                "administrator_only_siblings": [
                    "blind_keys",
                    "comparison_panels",
                    "metrics",
                ],
                "warning": (
                    "Blinding is valid only when the review directory is "
                    "distributed without sibling artifacts."
                ),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return review_path, key_path

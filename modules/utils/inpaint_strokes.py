from __future__ import annotations

from typing import Iterable


STROKE_ROLE_GENERATED = "generated"
STROKE_ROLE_ADD = "add"
STROKE_ROLE_EXCLUDE = "exclude"
STROKE_ROLE_RESTORE_PREVIEW = "restore_preview"

PATCH_KIND_INPAINT = "inpaint"
PATCH_KIND_RESTORE = "restore"

STORABLE_STROKE_ROLES = {
    STROKE_ROLE_GENERATED,
    STROKE_ROLE_ADD,
    STROKE_ROLE_EXCLUDE,
}
MANUAL_STROKE_ROLES = {
    STROKE_ROLE_ADD,
    STROKE_ROLE_EXCLUDE,
}
PATCH_KINDS = {
    PATCH_KIND_INPAINT,
    PATCH_KIND_RESTORE,
}


def normalize_stroke_role(value: str | None, *, brush: str | None = None) -> str:
    role = str(value or "").strip().lower()
    if role in STORABLE_STROKE_ROLES or role == STROKE_ROLE_RESTORE_PREVIEW:
        return role

    brush_hex = str(brush or "").strip().lower()
    if brush_hex == "#80ff0000":
        return STROKE_ROLE_GENERATED
    return STROKE_ROLE_ADD


def normalize_patch_kind(value: str | None) -> str:
    kind = str(value or "").strip().lower()
    if kind in PATCH_KINDS:
        return kind
    return PATCH_KIND_INPAINT


def is_storable_stroke_role(value: str | None, *, brush: str | None = None) -> bool:
    return normalize_stroke_role(value, brush=brush) in STORABLE_STROKE_ROLES


def is_manual_stroke_role(value: str | None, *, brush: str | None = None) -> bool:
    return normalize_stroke_role(value, brush=brush) in MANUAL_STROKE_ROLES


def filter_strokes_by_role(
    strokes: Iterable[dict] | None,
    roles: set[str],
) -> list[dict]:
    return [
        stroke
        for stroke in (strokes or [])
        if normalize_stroke_role(stroke.get("role"), brush=stroke.get("brush")) in roles
    ]


def retain_non_manual_strokes(strokes: Iterable[dict] | None) -> list[dict]:
    return [
        stroke
        for stroke in (strokes or [])
        if normalize_stroke_role(stroke.get("role"), brush=stroke.get("brush"))
        not in MANUAL_STROKE_ROLES
    ]

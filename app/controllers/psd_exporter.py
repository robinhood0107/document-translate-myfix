from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from typing import Any

import imkit as imk
import numpy as np
from PySide6 import QtCore, QtGui

from app.controllers.psd_support import import_photoshopapi
from app.path_materialization import ensure_path_materialized
from app.ui.canvas.text.text_item_properties import TextItemProperties
from app.ui.canvas.text_item import OutlineType
from modules.rendering.rich_text import repair_text_item_html, should_use_rich_text

logger = logging.getLogger(__name__)


class _PsApiProxy:
    def __getattr__(self, name: str) -> Any:
        return getattr(import_photoshopapi(), name)


psapi = _PsApiProxy()


@dataclass
class PsdPageData:
    file_path: str
    rgb_image: np.ndarray
    viewer_state: dict[str, Any]
    patches: list[dict[str, Any]]


def export_psd_pages(
    output_folder: str,
    pages: list[PsdPageData],
    bundle_name: str,
    single_file_path: str | None = None,
) -> str:
    import_photoshopapi()
    if not pages:
        raise ValueError("No images available to export.")

    os.makedirs(output_folder, exist_ok=True)

    if len(pages) == 1:
        page = pages[0]
        out_path = single_file_path or os.path.join(output_folder, f"{_safe_stem(page.file_path)}.psd")
        _write_page_psd(page, out_path)
        return out_path

    for page in pages:
        out_path = os.path.join(output_folder, f"{_safe_stem(page.file_path)}.psd")
        _write_page_psd(page, out_path)
    return output_folder


def _write_page_psd(page: PsdPageData, out_path: str) -> None:
    image = _ensure_rgb_uint8(page.rgb_image)
    height, width, _ = image.shape

    doc = psapi.LayeredFile_8bit(psapi.enum.ColorMode.rgb, width, height)
    try:
        doc.dpi = 300.0
    except Exception:
        pass

    text_group = psapi.GroupLayer_8bit("Editable Text")
    doc.add_layer(text_group)
    for idx, text_state in enumerate(page.viewer_state.get("text_items_state", []) or [], start=1):
        text_layer = _build_text_layer(text_state, idx)
        if text_layer is not None:
            text_group.add_layer(doc, text_layer)

    patch_group = psapi.GroupLayer_8bit("Inpaint Patches")
    doc.add_layer(patch_group)
    for idx, patch in enumerate(page.patches or [], start=1):
        patch_layer = _build_patch_layer(patch, idx)
        if patch_layer is not None:
            patch_group.add_layer(doc, patch_layer)

    base_layer = psapi.ImageLayer_8bit(
        _to_psapi_image_data(image),
        "Raw Image",
        width=width,
        height=height,
        pos_x=width / 2,
        pos_y=height / 2,
    )
    doc.add_layer(base_layer)

    invalidate = getattr(doc, "invalidate_text_cache", None)
    if callable(invalidate):
        try:
            invalidate()
        except Exception:
            pass

    doc.write(out_path, force_overwrite=True)


def _build_patch_layer(patch: dict[str, Any], index: int) -> Any | None:
    bbox = patch.get("bbox")
    if not bbox or len(bbox) != 4:
        return None
    x, y, w, h = [int(round(v)) for v in bbox]
    if w <= 0 or h <= 0:
        return None

    patch_img = None
    if patch.get("png_path"):
        png_path = patch["png_path"]
        ensure_path_materialized(png_path)
        if os.path.isfile(png_path):
            patch_img = imk.read_image(png_path)
    if patch_img is None:
        patch_img = patch.get("image")
    if patch_img is None:
        return None

    patch_img = _ensure_rgb_uint8(patch_img)
    ph, pw, _ = patch_img.shape
    return psapi.ImageLayer_8bit(
        _to_psapi_image_data(patch_img),
        f"Patch {index}",
        width=pw,
        height=ph,
        pos_x=x + pw / 2,
        pos_y=y + ph / 2,
    )


def _build_text_layer(state: dict[str, Any], index: int) -> Any | None:
    props = TextItemProperties.from_dict(state)
    plain_text, doc_margin = _extract_plain_text(props.text)
    if not plain_text:
        return None

    pos_x, pos_y = props.position
    box_width = float(props.width) if props.width else 0.0
    box_height = float(props.height) if props.height else 0.0
    pos_x = float(pos_x) + doc_margin
    pos_y = float(pos_y) + doc_margin
    if box_width > 0:
        box_width = max(0.0, box_width - 2.0 * doc_margin)
    if box_height > 0:
        box_height = max(0.0, box_height - 2.0 * doc_margin)

    font_name = str(props.font_family or "ArialMT")
    layer = psapi.TextLayer_8bit(
        layer_name=f"Text {index}",
        text=plain_text,
        font=font_name,
        font_size=float(props.font_size),
        fill_color=_to_argb_floats(props.text_color),
        position_x=pos_x,
        position_y=pos_y,
        box_width=box_width,
        box_height=box_height,
    )

    _set_text_antialias(layer)
    _apply_default_text_style(layer, props)
    _apply_paragraph_style(layer, props)
    _apply_text_direction(layer, props)
    _apply_text_rotation(layer, props)
    return layer


def _extract_plain_text(html_value: str) -> tuple[str, float]:
    if not html_value:
        return "", 4.0

    document = QtGui.QTextDocument()
    if should_use_rich_text(html_value):
        document.setHtml(repair_text_item_html(html_value, {}))
    else:
        document.setPlainText(html_value)
    plain_text = (
        document.toPlainText()
        .replace("\u2028", "\n")
        .replace("\u2029", "\n")
        .replace("\r\n", "\n")
        .replace("\r", "\n")
    )
    return plain_text, float(document.documentMargin())


def _apply_default_text_style(layer: Any, props: TextItemProperties) -> None:
    style_all = getattr(layer, "style_all", None)
    if not callable(style_all):
        return
    try:
        editor = style_all()
    except Exception:
        return

    _set_if_exists(editor, "set_font", str(props.font_family or "ArialMT"))
    _set_if_exists(editor, "set_font_size", float(props.font_size))
    _set_if_exists(editor, "set_fill_color", _to_argb_floats(props.text_color))
    _set_if_exists(editor, "set_bold", bool(props.bold))
    _set_if_exists(editor, "set_italic", bool(props.italic))
    _set_if_exists(editor, "set_underline", bool(props.underline))

    font = QtGui.QFont(props.font_family or "Arial")
    font.setPointSizeF(max(1.0, float(props.font_size)))
    font.setBold(bool(props.bold))
    font.setItalic(bool(props.italic))
    line_spacing = max(1.0, float(props.line_spacing or 1.0))
    _set_if_exists(editor, "set_auto_leading", False)
    _set_if_exists(editor, "set_leading", float(QtGui.QFontMetricsF(font).lineSpacing() * line_spacing))

    outline = _full_outline_style(props)
    has_outline = outline is not None
    _set_if_exists(editor, "set_stroke_flag", has_outline)
    if has_outline:
        _set_if_exists(editor, "set_stroke_color", _to_argb_floats(outline["color"]))
        _set_if_exists(editor, "set_outline_width", float(outline["width"]))


def _full_outline_style(props: TextItemProperties) -> dict[str, Any] | None:
    outlines = getattr(props, "selection_outlines", None) or []
    for outline in outlines:
        outline_type = outline.get("type") if isinstance(outline, dict) else getattr(outline, "type", None)
        if outline_type == OutlineType.Full_Document or (
            isinstance(outline_type, str) and outline_type.lower() == "full_document"
        ):
            color = outline.get("color") if isinstance(outline, dict) else getattr(outline, "color", None)
            width = outline.get("width") if isinstance(outline, dict) else getattr(outline, "width", None)
            qcolor = color if isinstance(color, QtGui.QColor) else QtGui.QColor(color)
            if qcolor.isValid() and width:
                return {"color": qcolor, "width": float(width)}

    if props.outline and props.outline_color is not None:
        return {"color": props.outline_color, "width": float(props.outline_width or 1.0)}
    return None


def _apply_paragraph_style(layer: Any, props: TextItemProperties) -> None:
    justification = _map_justification(props.alignment)
    if justification is None:
        return

    paragraph_all = getattr(layer, "paragraph_all", None)
    if callable(paragraph_all):
        try:
            editor = paragraph_all()
            _set_if_exists(editor, "set_justification", justification)
            return
        except Exception:
            pass

    _set_if_exists(layer, "set_paragraph_normal_justification", justification)


def _apply_text_direction(layer: Any, props: TextItemProperties) -> None:
    writing_direction_enum = getattr(psapi.enum, "WritingDirection", None)
    if writing_direction_enum is not None:
        orientation = getattr(writing_direction_enum, "Vertical" if props.vertical else "Horizontal", None)
        if orientation is not None:
            applied = _set_if_exists(layer, "set_orientation", orientation)
            if not applied and hasattr(orientation, "value"):
                _set_if_exists(layer, "set_orientation", int(getattr(orientation, "value")))

    if props.vertical:
        return

    character_direction_enum = getattr(psapi.enum, "CharacterDirection", None)
    if character_direction_enum is None:
        return

    is_rtl = (
        props.direction == QtCore.Qt.LayoutDirection.RightToLeft
        or (
            isinstance(props.direction, int)
            and props.direction == QtCore.Qt.LayoutDirection.RightToLeft.value
        )
    )
    direction = getattr(character_direction_enum, "RightToLeft" if is_rtl else "LeftToRight", None)
    if direction is None:
        return

    _set_if_exists(layer, "set_style_normal_character_direction", direction)
    run_count = getattr(layer, "style_run_count", None)
    set_run_direction = getattr(layer, "set_style_run_character_direction", None)
    if callable(set_run_direction) and isinstance(run_count, int):
        for idx in range(run_count):
            try:
                set_run_direction(idx, direction)
            except Exception:
                pass


def _apply_text_rotation(layer: Any, props: TextItemProperties) -> None:
    try:
        rotation = float(props.rotation or 0.0)
    except Exception:
        rotation = 0.0
    if abs(rotation) < 1e-6:
        return
    if _set_if_exists(layer, "set_rotation_angle", rotation):
        return
    if _set_if_exists(layer, "set_rotation", rotation):
        return
    try:
        setattr(layer, "rotation_angle", rotation)
    except Exception:
        logger.debug("Unable to set PSD text rotation.", exc_info=True)


def _map_justification(alignment: Any) -> Any | None:
    try:
        alignment_value = int(alignment)
    except Exception:
        alignment_value = int(QtCore.Qt.AlignmentFlag.AlignLeft)

    if alignment_value & int(QtCore.Qt.AlignmentFlag.AlignHCenter):
        candidates = ("Center", "center", "CENTER")
    elif alignment_value & int(QtCore.Qt.AlignmentFlag.AlignRight):
        candidates = ("Right", "right", "RIGHT")
    else:
        candidates = ("Left", "left", "LEFT")

    enum_obj = getattr(getattr(psapi, "enum", None), "Justification", None)
    if enum_obj is None:
        return None

    for name in candidates:
        if hasattr(enum_obj, name):
            return getattr(enum_obj, name)
    return None


def _set_text_antialias(layer: Any) -> None:
    anti_alias_enum = getattr(getattr(psapi, "enum", None), "AntiAliasMethod", None)
    if anti_alias_enum is None:
        return
    for name in ("Sharp", "Strong", "Crisp", "Smooth"):
        anti_alias_value = getattr(anti_alias_enum, name, None)
        if anti_alias_value is not None:
            _set_if_exists(layer, "set_anti_alias", anti_alias_value)
            return


def _set_if_exists(obj: Any, method_name: str, *args: Any) -> bool:
    method = getattr(obj, method_name, None)
    if callable(method):
        try:
            method(*args)
            return True
        except Exception:
            return False
    return False


def _to_psapi_image_data(rgb_image: np.ndarray) -> np.ndarray:
    image = np.asarray(rgb_image)
    if image.ndim != 3:
        raise ValueError("Expected an image array with shape (H, W, C)")
    if image.shape[2] == 3:
        alpha = np.full((image.shape[0], image.shape[1], 1), 255, dtype=np.uint8)
        image = np.concatenate([image, alpha], axis=2)
    return np.ascontiguousarray(np.transpose(image, (2, 0, 1)))


def _ensure_rgb_uint8(image: np.ndarray) -> np.ndarray:
    img = np.asarray(image)
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    if img.ndim != 3:
        raise ValueError("Expected an RGB image array with shape (H, W, C)")
    if img.shape[2] == 4:
        img = img[:, :, :3]
    if img.shape[2] != 3:
        raise ValueError("Expected an RGB image array with 3 channels")
    return img


def _to_argb_floats(color: Any | None) -> list[float]:
    if color is None:
        return [1.0, 0.0, 0.0, 0.0]
    qcolor = color if isinstance(color, QtGui.QColor) else QtGui.QColor(color)
    if not qcolor.isValid():
        qcolor = QtGui.QColor(0, 0, 0)
    return [qcolor.alphaF(), qcolor.redF(), qcolor.greenF(), qcolor.blueF()]


def _safe_stem(path: str) -> str:
    stem = os.path.splitext(os.path.basename(path))[0]
    return _safe_name(stem)


def _safe_name(name: str) -> str:
    cleaned = re.sub(r'[\\/:*?"<>|]', "_", name or "")
    cleaned = cleaned.strip().strip(".")
    return cleaned or "untitled"

from __future__ import annotations

import html
import re
from dataclasses import dataclass
from typing import Any

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QFont, QTextBlockFormat, QTextCursor, QTextDocument, QTextOption
from PySide6.QtWidgets import QApplication

from modules.utils.text_normalization import RENDER_NORMALIZABLE_GLYPHS


QT_RICH_TEXT_META = '<meta name="qrichtext" content="1" />'


@dataclass(frozen=True)
class StyledRenderHtmlResult:
    text: str
    html_text: str
    fallback_font_family: str
    replacements: list[dict]


def looks_like_trusted_qt_html(text: str | None) -> bool:
    if not text:
        return False
    sample = str(text).lstrip()[:2000].lower()
    return (
        "<html" in sample
        and "<body" in sample
    ) or 'meta name="qrichtext"' in sample or "<!doctype html" in sample


def looks_like_render_html_fragment(text: str | None) -> bool:
    if not text:
        return False
    value = str(text)
    sample = value[:2000].lower()
    if "<span" not in sample or "font-family" not in sample:
        return False
    return any(ch in value for ch in RENDER_NORMALIZABLE_GLYPHS)


def should_use_rich_text(text: str | None) -> bool:
    return looks_like_trusted_qt_html(text) or looks_like_render_html_fragment(text)


def plain_text_from_trusted_html(text: str) -> str:
    doc = QTextDocument()
    doc.setHtml(text or "")
    return _normalize_qt_plain_text(doc.toPlainText())


def build_styled_render_html(
    text: str,
    *,
    font_family: str = "",
    font_size: float = 20,
    text_color: QColor | str | None = None,
    alignment: Qt.AlignmentFlag | Qt.Alignment = Qt.AlignmentFlag.AlignCenter,
    line_spacing: float = 1.0,
    bold: bool = False,
    italic: bool = False,
    underline: bool = False,
    direction: Qt.LayoutDirection = Qt.LayoutDirection.LeftToRight,
    fallback_font_family: str = "",
) -> StyledRenderHtmlResult:
    raw_text = str(text or "")
    body = _render_text_to_html(raw_text, fallback_font_family)
    html_text = _wrap_body_content(
        body,
        font_family=font_family,
        font_size=font_size,
        text_color=text_color,
        alignment=alignment,
        line_spacing=line_spacing,
        bold=bold,
        italic=italic,
        underline=underline,
        direction=direction,
    )
    replacements = [
        {
            "index": int(index),
            "char": ch,
            "replacement": ch,
            "reason": "symbol-fallback-font",
        }
        for index, ch in enumerate(raw_text)
        if ch in RENDER_NORMALIZABLE_GLYPHS and fallback_font_family
    ]
    return StyledRenderHtmlResult(
        text=raw_text,
        html_text=html_text,
        fallback_font_family=fallback_font_family if replacements else "",
        replacements=replacements,
    )


def repair_render_html_style(
    text: str,
    *,
    font_family: str = "",
    font_size: float = 20,
    text_color: QColor | str | None = None,
    alignment: Qt.AlignmentFlag | Qt.Alignment = Qt.AlignmentFlag.AlignCenter,
    line_spacing: float = 1.0,
    bold: bool = False,
    italic: bool = False,
    underline: bool = False,
    direction: Qt.LayoutDirection = Qt.LayoutDirection.LeftToRight,
) -> str:
    if not text:
        return ""
    value = str(text)
    if looks_like_render_html_fragment(value) and not looks_like_trusted_qt_html(value):
        return _wrap_body_content(
            value,
            font_family=font_family,
            font_size=font_size,
            text_color=text_color,
            alignment=alignment,
            line_spacing=line_spacing,
            bold=bold,
            italic=italic,
            underline=underline,
            direction=direction,
        )
    if not looks_like_trusted_qt_html(value):
        return value

    body_inner = _extract_body_inner(value)
    if body_inner is None:
        body_inner = value
    return _wrap_body_content(
        body_inner,
        font_family=font_family,
        font_size=font_size,
        text_color=text_color,
        alignment=alignment,
        line_spacing=line_spacing,
        bold=bold,
        italic=italic,
        underline=underline,
        direction=direction,
    )


def repair_text_item_html(text: str, props: Any) -> str:
    return repair_render_html_style(
        text,
        font_family=_prop(props, "font_family", ""),
        font_size=_prop(props, "font_size", 20),
        text_color=_prop(props, "text_color", None),
        alignment=_prop(props, "alignment", Qt.AlignmentFlag.AlignCenter),
        line_spacing=_prop(props, "line_spacing", 1.0),
        bold=bool(_prop(props, "bold", False)),
        italic=bool(_prop(props, "italic", False)),
        underline=bool(_prop(props, "underline", False)),
        direction=_prop(props, "direction", Qt.LayoutDirection.LeftToRight),
    )


def apply_document_base_style(
    doc: QTextDocument,
    *,
    font_family: str = "",
    font_size: float = 20,
    text_color: QColor | str | None = None,
    alignment: Qt.AlignmentFlag | Qt.Alignment = Qt.AlignmentFlag.AlignCenter,
    line_spacing: float = 1.0,
    bold: bool = False,
    italic: bool = False,
    underline: bool = False,
    direction: Qt.LayoutDirection = Qt.LayoutDirection.LeftToRight,
) -> None:
    family = _effective_font_family(font_family)
    font = QFont(family)
    font.setPointSizeF(max(1.0, float(font_size or 20)))
    font.setBold(bool(bold))
    font.setItalic(bool(italic))
    font.setUnderline(bool(underline))
    doc.setDefaultFont(font)

    text_option = doc.defaultTextOption()
    text_option.setTextDirection(direction)
    doc.setDefaultTextOption(text_option)

    cursor = QTextCursor(doc)
    cursor.select(QTextCursor.SelectionType.Document)
    block_format = QTextBlockFormat()
    block_format.setAlignment(alignment)
    block_format.setLineHeight(
        max(1.0, float(line_spacing or 1.0)) * 100,
        QTextBlockFormat.LineHeightTypes.ProportionalHeight.value,
    )
    cursor.mergeBlockFormat(block_format)
    if text_color is not None:
        color = _coerce_color(text_color)
        if color.isValid():
            doc.setDefaultStyleSheet(f"body {{ color: {color.name()}; }}")


def _normalize_qt_plain_text(value: str) -> str:
    return (
        str(value or "")
        .replace("\u2028", "\n")
        .replace("\u2029", "\n")
        .replace("\r\n", "\n")
        .replace("\r", "\n")
    )


def _render_text_to_html(text: str, fallback_font_family: str) -> str:
    parts: list[str] = []
    fallback_family = html.escape(str(fallback_font_family or ""), quote=True)
    for ch in str(text or ""):
        if ch == "\n":
            parts.append("<br />")
            continue
        escaped = html.escape(ch, quote=False)
        if fallback_family and ch in RENDER_NORMALIZABLE_GLYPHS:
            parts.append(f'<span style="font-family:\'{fallback_family}\';">{escaped}</span>')
        else:
            parts.append(escaped)
    return "".join(parts)


def _wrap_body_content(
    body_content: str,
    *,
    font_family: str,
    font_size: float,
    text_color: QColor | str | None,
    alignment: Qt.AlignmentFlag | Qt.Alignment,
    line_spacing: float,
    bold: bool,
    italic: bool,
    underline: bool,
    direction: Qt.LayoutDirection,
) -> str:
    body_content = str(body_content or "").strip()
    body_style = _body_style(
        font_family=font_family,
        font_size=font_size,
        text_color=text_color,
        bold=bold,
        italic=italic,
        underline=underline,
    )
    paragraph_style = _paragraph_style(
        alignment=alignment,
        line_spacing=line_spacing,
        direction=direction,
    )
    if re.search(r"<p\b", body_content, flags=re.IGNORECASE):
        paragraph_content = _restyle_paragraphs(body_content, paragraph_style)
    else:
        paragraph_content = f"<p style=\"{paragraph_style}\">{body_content}</p>"
    return (
        '<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.0//EN" '
        '"http://www.w3.org/TR/REC-html40/strict.dtd">\n'
        f"<html><head>{QT_RICH_TEXT_META}</head>"
        f"<body style=\"{body_style}\">"
        f"{paragraph_content}"
        "</body></html>"
    )


def _body_style(
    *,
    font_family: str,
    font_size: float,
    text_color: QColor | str | None,
    bold: bool,
    italic: bool,
    underline: bool,
) -> str:
    family = html.escape(_effective_font_family(font_family), quote=True)
    color = _coerce_color(text_color).name() if text_color is not None else "#000000"
    weight = 700 if bold else 400
    style = "italic" if italic else "normal"
    decoration = "underline" if underline else "none"
    size = max(1.0, float(font_size or 20))
    return (
        f" font-family:'{family}'; font-size:{size:g}pt; "
        f"font-weight:{weight}; font-style:{style}; "
        f"text-decoration:{decoration}; color:{color};"
    )


def _paragraph_style(
    *,
    alignment: Qt.AlignmentFlag | Qt.Alignment,
    line_spacing: float,
    direction: Qt.LayoutDirection,
) -> str:
    line_height = max(1.0, float(line_spacing or 1.0)) * 100
    return (
        " margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; "
        "-qt-block-indent:0; text-indent:0px; "
        f"text-align:{_alignment_to_css(alignment)}; "
        f"line-height:{line_height:g}%; "
        f"direction:{_direction_to_css(direction)};"
    )


def _effective_font_family(font_family: str) -> str:
    value = str(font_family or "").strip()
    if value:
        return value
    app = QApplication.instance()
    if app is not None:
        return QApplication.font().family()
    return "Arial"


def _coerce_color(color: QColor | str | None) -> QColor:
    if isinstance(color, QColor):
        return QColor(color)
    if color:
        return QColor(str(color))
    return QColor("#000000")


def _alignment_to_css(alignment: Qt.AlignmentFlag | Qt.Alignment) -> str:
    value = _alignment_value(alignment)
    if value & _alignment_value(Qt.AlignmentFlag.AlignRight):
        return "right"
    if value & _alignment_value(Qt.AlignmentFlag.AlignLeft):
        return "left"
    if value & _alignment_value(Qt.AlignmentFlag.AlignJustify):
        return "justify"
    return "center"


def _alignment_value(alignment: Qt.AlignmentFlag | Qt.Alignment) -> int:
    try:
        return int(alignment)
    except TypeError:
        return int(getattr(alignment, "value", 0))


def _direction_to_css(direction: Qt.LayoutDirection) -> str:
    if isinstance(direction, int):
        return "rtl" if direction == int(Qt.LayoutDirection.RightToLeft.value) else "ltr"
    return "rtl" if direction == Qt.LayoutDirection.RightToLeft else "ltr"


def _prop(props: Any, key: str, default: Any) -> Any:
    if isinstance(props, dict):
        return props.get(key, default)
    return getattr(props, key, default)


def _extract_body_inner(value: str) -> str | None:
    match = re.search(r"<body\b[^>]*>(.*)</body>", value, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return None
    return match.group(1)


def _restyle_paragraphs(value: str, paragraph_style: str) -> str:
    return re.sub(
        r"<p\b[^>]*>",
        lambda _match: f'<p style="{paragraph_style}">',
        value,
        flags=re.IGNORECASE,
    )

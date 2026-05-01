import logging
import numpy as np
import html
import re
from typing import Tuple, List
import unicodedata
from functools import lru_cache

from PIL import Image, ImageFont, ImageDraw
from PySide6.QtGui import QFont, QFontMetrics, QTextDocument,\
      QTextCursor, QTextBlockFormat, QTextOption, QFontDatabase
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from .hyphen_textwrap import wrap as hyphen_wrap
from modules.utils.textblock import TextBlock
from modules.utils.textblock import adjust_blks_size
from modules.detection.utils.geometry import shrink_bbox
from app.ui.canvas.text.vertical_layout import VerticalTextDocumentLayout
from modules.utils.language_utils import get_language_code
from modules.utils.text_normalization import (
    OCR_DECORATIVE_NOISE_GLYPHS,
    RENDER_NORMALIZABLE_GLYPHS,
    canonicalize_ellipsis_runs,
)
from modules.utils.render_style_policy import (
    VERTICAL_ALIGNMENT_TOP,
    compute_vertical_aligned_y,
)
from modules.rendering.rich_text import build_styled_render_html

from dataclasses import dataclass


logger = logging.getLogger(__name__)

@dataclass
class TextRenderingSettings:
    alignment_id: int
    vertical_alignment_id: int
    font_family: str
    min_font_size: int
    max_font_size: int
    color: str
    force_font_color: bool
    smart_global_apply_all: bool
    upper_case: bool
    outline: bool
    outline_color: str
    outline_width: str
    bold: bool
    italic: bool
    underline: bool
    line_spacing: str
    direction: Qt.LayoutDirection


@dataclass
class RenderSanitizationResult:
    raw_text: str
    text: str
    normalization_applied: bool
    reasons: list[str]
    replacements: list[dict]


@dataclass
class RenderMarkupResult:
    text: str
    html_text: str
    html_applied: bool
    reasons: list[str]
    fallback_font_family: str
    replacements: list[dict]


RENDER_SYMBOL_FALLBACK_FONT_CANDIDATES = (
    "Malgun Gothic",
    "Yu Gothic UI",
    "Meiryo",
    "MS Gothic",
    "Segoe UI Symbol",
    "Segoe UI Emoji",
)

HORIZONTAL_BUBBLE_SHRINK_PERCENT = 0.18
VERTICAL_BUBBLE_SHRINK_PERCENT = 0.30
MIN_BUBBLE_TEXT_CONTAINMENT = 0.60
MIN_BUBBLE_RENDER_AREA_GAIN = 0.90
DETECTED_BUBBLE_FIT_CLEARANCE_PX = 8.0
DETECTED_BUBBLE_OUTLINE_CLEARANCE_MULTIPLIER = 2.0
DETECTED_BUBBLE_MIN_FIT_DIMENSION_PX = 16.0

_CJK_RE = re.compile(r"[\uac00-\ud7a3\u3040-\u30ff\u4e00-\u9fff]")
_BREAK_BEFORE_FORBIDDEN = set(".,!?;:)]}，。！？、；：）」』】》〉…")
_BREAK_AFTER_FORBIDDEN = set("([{（「『【《〈")

def array_to_pil(rgb_image: np.ndarray):
    # Image is already in RGB format, just convert to PIL
    pil_image = Image.fromarray(rgb_image)
    return pil_image

def pil_to_array(pil_image: Image):
    # Convert the PIL image to a numpy array (already in RGB)
    numpy_image = np.array(pil_image)
    return numpy_image

def is_vertical_language_code(lang_code: str | None) -> bool:
    """Return True if the language code should use vertical layout.

    Currently treats Japanese and simplified/traditional Chinese as
    vertical-capable languages.
    """
    if not lang_code:
        return False
    code = lang_code.lower()
    return code in {"zh-cn", "zh-tw", "ja"}

def is_vertical_block(blk, lang_code: str | None) -> bool:
    """Return True if this block should be rendered vertically.

    A block is considered vertical when its direction flag is "vertical"
    and the target language code is one of the vertical-capable ones.
    """
    return getattr(blk, "direction", "") == "vertical" and is_vertical_language_code(lang_code)


def _render_font_supports(metrics: QFontMetrics, ch: str) -> bool:
    try:
        return metrics.inFontUcs4(ord(ch))
    except AttributeError:
        return metrics.inFont(ch)


def _canonicalize_render_symbol_variants(text: str) -> str:
    if not text:
        return ""
    return (
        text.replace("❤︎", "♥")
        .replace("❤️", "♥")
        .replace("❤", "♥")
        .replace("♡", "♥")
    )


@lru_cache(maxsize=1)
def resolve_render_symbol_fallback_font_family() -> str:
    database = QFontDatabase()
    families = {family.casefold(): family for family in database.families()}
    required_chars = tuple(sorted(RENDER_NORMALIZABLE_GLYPHS))
    for candidate in RENDER_SYMBOL_FALLBACK_FONT_CANDIDATES:
        actual = families.get(candidate.casefold())
        if not actual:
            continue
        metrics = QFontMetrics(QFont(actual, 12))
        if all(_render_font_supports(metrics, ch) for ch in required_chars):
            return actual
    return ""


def describe_render_text_sanitization(
    text: str,
    font_family: str,
    *,
    block_index: int | None = None,
    image_path: str = "",
) -> RenderSanitizationResult:
    if not text:
        return RenderSanitizationResult("", "", False, [], [])

    raw_text = str(text or "")
    sanitized = canonicalize_ellipsis_runs(_canonicalize_render_symbol_variants(raw_text))
    effective_family = font_family.strip() if isinstance(font_family, str) and font_family.strip() else QApplication.font().family()
    metrics = QFontMetrics(QFont(effective_family, 12))
    symbol_fallback_family = resolve_render_symbol_fallback_font_family()
    cleaned_parts: list[str] = []
    replacements: list[dict] = []
    reasons: list[str] = []

    for index, ch in enumerate(sanitized):
        replacement = ch
        reason = ""
        if ch in OCR_DECORATIVE_NOISE_GLYPHS:
            replacement = ""
            reason = "decorative-noise"
        elif (
            ch in {"「", "」", "『", "』"}
            and not symbol_fallback_family
            and not _render_font_supports(metrics, ch)
        ):
            replacement = "\""
            reason = "quote-to-ascii"
        elif ch == "♥" and not symbol_fallback_family and not _render_font_supports(metrics, ch):
            replacement = ""
            reason = "heart-dropped"
        elif ch not in {"\n", "\r", "\t"} and not _render_font_supports(metrics, ch):
            category = unicodedata.category(ch)
            if ch in {"…", "⋯"}:
                replacement = "..."
                reason = "unsupported-ellipsis"
            elif category.startswith("S"):
                replacement = ""
                reason = "unsupported-symbol"

        if replacement != ch:
            logger.warning(
                "render glyph sanitized: image=%s block=%s codepoint=U+%04X char=%r replacement=%r reason=%s font=%s",
                image_path or "",
                block_index if block_index is not None else -1,
                ord(ch),
                ch,
                replacement,
                reason or "render-normalization",
                effective_family,
            )
            reasons.append(reason or "render-normalization")
            replacements.append(
                {
                    "index": int(index),
                    "char": ch,
                    "replacement": replacement,
                    "reason": reason or "render-normalization",
                }
            )
        cleaned_parts.append(replacement)

    normalized = "".join(cleaned_parts)
    return RenderSanitizationResult(
        raw_text=raw_text,
        text=normalized,
        normalization_applied=bool(replacements),
        reasons=sorted(set(reasons)),
        replacements=replacements,
    )


def describe_render_text_markup(
    text: str,
    *,
    font_family: str = "",
    font_size: float | None = None,
    text_color=None,
    alignment: Qt.AlignmentFlag = Qt.AlignmentFlag.AlignCenter,
    line_spacing: float = 1.0,
    bold: bool = False,
    italic: bool = False,
    underline: bool = False,
    direction: Qt.LayoutDirection = Qt.LayoutDirection.LeftToRight,
) -> RenderMarkupResult:
    if not text:
        return RenderMarkupResult("", "", False, [], "", [])

    raw_text = str(text or "")
    fallback_font_family = resolve_render_symbol_fallback_font_family()
    use_full_html = font_size is not None
    if use_full_html:
        styled = build_styled_render_html(
            raw_text,
            font_family=font_family,
            font_size=float(font_size or 20),
            text_color=text_color,
            alignment=alignment,
            line_spacing=float(line_spacing or 1.0),
            bold=bold,
            italic=italic,
            underline=underline,
            direction=direction,
            fallback_font_family=fallback_font_family,
        )
        reasons = ["styled-render-html"]
        if styled.replacements:
            reasons.append("symbol-fallback-font")
        return RenderMarkupResult(
            text=raw_text,
            html_text=styled.html_text,
            html_applied=True,
            reasons=reasons,
            fallback_font_family=styled.fallback_font_family,
            replacements=styled.replacements,
        )

    if not fallback_font_family:
        return RenderMarkupResult(raw_text, raw_text, False, [], "", [])

    html_parts: list[str] = []
    replacements: list[dict] = []
    for index, ch in enumerate(raw_text):
        if ch == "\n":
            html_parts.append("<br/>")
            continue
        escaped = html.escape(ch)
        if ch in RENDER_NORMALIZABLE_GLYPHS:
            html_parts.append(
                f'<span style="font-family:\'{html.escape(fallback_font_family, quote=True)}\';">{escaped}</span>'
            )
            replacements.append(
                {
                    "index": int(index),
                    "char": ch,
                    "replacement": ch,
                    "reason": "symbol-fallback-font",
                }
            )
        else:
            html_parts.append(escaped)

    html_text = "".join(html_parts)
    return RenderMarkupResult(
        text=raw_text,
        html_text=html_text,
        html_applied=bool(replacements),
        reasons=["symbol-fallback-font"] if replacements else [],
        fallback_font_family=fallback_font_family if replacements else "",
        replacements=replacements,
    )


def sanitize_render_text(
    text: str,
    font_family: str,
    *,
    block_index: int | None = None,
    image_path: str = "",
) -> str:
    return describe_render_text_sanitization(
        text,
        font_family,
        block_index=block_index,
        image_path=image_path,
    ).text

def pil_word_wrap(image: Image, tbbox_top_left: Tuple, font_pth: str, text: str, 
                  roi_width, roi_height, align: str, spacing, init_font_size: int, min_font_size: int = 10):
    """Break long text to multiple lines, and reduce point size
    until all text fits within a bounding box."""
    mutable_message = text
    font_size = init_font_size
    font = ImageFont.truetype(font_pth, font_size)

    def eval_metrics(txt, font):
        """Quick helper function to calculate width/height of text."""
        (left, top, right, bottom) = ImageDraw.Draw(image).multiline_textbbox(xy=tbbox_top_left, text=txt, font=font, align=align, spacing=spacing)
        return (right-left, bottom-top)

    while font_size > min_font_size:
        font = font.font_variant(size=font_size)
        width, height = eval_metrics(mutable_message, font)
        if height > roi_height:
            font_size -= 0.75  # Reduce pointsize
            mutable_message = text  # Restore original text
        elif width > roi_width:
            columns = len(mutable_message)
            while columns > 0:
                columns -= 1
                if columns == 0:
                    break
                mutable_message = '\n'.join(hyphen_wrap(text, columns, break_on_hyphens=False, break_long_words=False, hyphenate_broken_words=True)) 
                wrapped_width, _ = eval_metrics(mutable_message, font)
                if wrapped_width <= roi_width:
                    break
            if columns < 1:
                font_size -= 0.75  # Reduce pointsize
                mutable_message = text  # Restore original text
        else:
            break

    if font_size <= min_font_size:
        font_size = min_font_size
        mutable_message = text
        font = font.font_variant(size=font_size)

        # Wrap text to fit within as much as possible
        # Minimize cost function: (width - roi_width)^2 + (height - roi_height)^2
        # This is a brute force approach, but it works well enough
        min_cost = 1e9
        min_text = text
        for columns in range(1, len(text)):
            wrapped_text = '\n'.join(hyphen_wrap(text, columns, break_on_hyphens=False, break_long_words=False, hyphenate_broken_words=True))
            wrapped_width, wrapped_height = eval_metrics(wrapped_text, font)
            cost = (wrapped_width - roi_width)**2 + (wrapped_height - roi_height)**2
            if cost < min_cost:
                min_cost = cost
                min_text = wrapped_text

        mutable_message = min_text

    return mutable_message, font_size

def draw_text(image: np.ndarray, blk_list: List[TextBlock], font_pth: str, colour: str = "#000", init_font_size: int = 40, min_font_size=10, outline: bool = True):
    image = array_to_pil(image)
    draw = ImageDraw.Draw(image)

    font = ImageFont.truetype(font_pth, size=init_font_size)

    for block_index, blk in enumerate(blk_list):
        x1, y1, width, height = blk.xywh
        tbbox_top_left = (x1, y1)

        translation = sanitize_render_text(
            blk.translation,
            "",
            block_index=block_index,
        )
        if not translation or len(translation) == 1:
            continue

        if blk.min_font_size > 0:
            min_font_size = blk.min_font_size
        if blk.max_font_size > 0:
            init_font_size = blk.max_font_size
        if blk.font_color:
            colour = blk.font_color

        translation, font_size = pil_word_wrap(image, tbbox_top_left, font_pth, translation, width, height,
                                               align=blk.alignment, spacing=blk.line_spacing, init_font_size=init_font_size, min_font_size=min_font_size)
        font = font.font_variant(size=font_size)

        # Font Detection Workaround. Draws white color offset around text
        if outline:
            offsets = [(dx, dy) for dx in (-2, -1, 0, 1, 2) for dy in (-2, -1, 0, 1, 2) if dx != 0 or dy != 0]
            for dx, dy in offsets:
                draw.multiline_text((tbbox_top_left[0] + dx, tbbox_top_left[1] + dy), translation, font=font, fill="#FFF", align=blk.alignment, spacing=1)
        draw.multiline_text(tbbox_top_left, translation, colour, font, align=blk.alignment, spacing=1)
        
    image = pil_to_array(image)  # Already in RGB format
    return image

def get_best_render_area(blk_list: List[TextBlock], img, inpainted_img=None):
    """Select safe text render areas without losing the original OCR anchor."""
    for blk in blk_list:
        _reset_render_area_metadata(blk)
        text_draw_bounds = _detected_bubble_render_bounds(blk, img)
        if text_draw_bounds is None:
            continue
        bdx1, bdy1, bdx2, bdy2 = text_draw_bounds
        blk.xyxy[:] = [bdx1, bdy1, bdx2, bdy2]
        blk._render_area_source = "detected_bubble"
        blk._render_area_xyxy = [int(bdx1), int(bdy1), int(bdx2), int(bdy2)]

    if img is not None and blk_list and blk_list[0].source_lang not in ['ko', 'zh']:
        adjust_blks_size(blk_list, img, -5, -5)

    return blk_list


def build_render_rects_for_block(blk: TextBlock) -> tuple[tuple[float, float, float, float], tuple[float, float, float, float]]:
    """Return layout source_rect and original OCR block_anchor for a render block."""
    source_rect = _xyxy_to_rect_tuple(getattr(blk, "xyxy", None))
    anchor_xyxy = _current_anchor_xyxy(blk)
    block_anchor = _xyxy_to_rect_tuple(anchor_xyxy)
    return source_rect, block_anchor


def build_text_item_layout_geometry(
    source_rect: tuple[float, float, float, float],
    rendered_height: float | None = None,
    vertical_alignment: str | None = VERTICAL_ALIGNMENT_TOP,
) -> tuple[tuple[float, float], float, float | None]:
    """Return text item geometry that keeps paragraph alignment relative to source_rect."""
    source_x, source_y, source_width, source_height = [float(v) for v in source_rect]
    position_y = source_y
    if rendered_height is not None:
        position_y = compute_vertical_aligned_y(
            source_y,
            source_height,
            rendered_height,
            vertical_alignment,
        )
    return (source_x, position_y), source_width, rendered_height


def get_render_fit_clearance_for_block(
    blk: TextBlock,
    outline_width: float | int | str = 0.0,
) -> float:
    """Return extra inner fit clearance for text rendered inside detected bubbles."""
    if getattr(blk, "_render_area_source", "") != "detected_bubble":
        return 0.0
    try:
        outline = max(0.0, float(outline_width))
    except (TypeError, ValueError):
        outline = 0.0
    return max(
        DETECTED_BUBBLE_FIT_CLEARANCE_PX,
        (outline * DETECTED_BUBBLE_OUTLINE_CLEARANCE_MULTIPLIER)
        + DETECTED_BUBBLE_FIT_CLEARANCE_PX,
    )


def _reset_render_area_metadata(blk: TextBlock) -> None:
    original = _current_anchor_xyxy(blk)
    bubble = _normalize_xyxy(getattr(blk, "bubble_xyxy", None))
    blk._render_original_xyxy = list(original) if original is not None else None
    blk._render_bubble_xyxy = list(bubble) if bubble is not None else None
    blk._render_area_source = "text_bbox"
    blk._render_area_xyxy = list(original) if original is not None else None


def _detected_bubble_render_bounds(blk: TextBlock, img) -> tuple[int, int, int, int] | None:
    if getattr(blk, "text_class", "") != "text_bubble":
        return None
    text_xyxy = _current_anchor_xyxy(blk)
    bubble_xyxy = _normalize_xyxy(getattr(blk, "bubble_xyxy", None))
    if text_xyxy is None or bubble_xyxy is None:
        return None
    if not _bbox_has_area(text_xyxy) or not _bbox_has_area(bubble_xyxy):
        return None
    if not _text_bbox_belongs_to_bubble(text_xyxy, bubble_xyxy):
        return None

    shrink_percent = (
        VERTICAL_BUBBLE_SHRINK_PERCENT
        if getattr(blk, "source_lang_direction", "") == "vertical"
        else HORIZONTAL_BUBBLE_SHRINK_PERCENT
    )
    candidate = _clamp_xyxy_to_image(shrink_bbox(bubble_xyxy, shrink_percent), img)
    if candidate is None or not _bbox_has_area(candidate):
        return None
    if not _bbox_contains_point(candidate, _bbox_center(text_xyxy)):
        return None
    if _bbox_area(candidate) < (_bbox_area(text_xyxy) * MIN_BUBBLE_RENDER_AREA_GAIN):
        return None
    return candidate


def _current_anchor_xyxy(blk: TextBlock) -> tuple[int, int, int, int] | None:
    current = _normalize_xyxy(getattr(blk, "xyxy", None))
    previous_area = _normalize_xyxy(getattr(blk, "_render_area_xyxy", None))
    previous_original = _normalize_xyxy(getattr(blk, "_render_original_xyxy", None))
    if previous_original is not None and previous_area is not None and current == previous_area:
        return previous_original
    return current


def _text_bbox_belongs_to_bubble(
    text_xyxy: tuple[int, int, int, int],
    bubble_xyxy: tuple[int, int, int, int],
) -> bool:
    if _bbox_contains_point(bubble_xyxy, _bbox_center(text_xyxy)):
        return True
    return _intersection_area(text_xyxy, bubble_xyxy) / max(1.0, _bbox_area(text_xyxy)) >= MIN_BUBBLE_TEXT_CONTAINMENT


def _normalize_xyxy(value) -> tuple[int, int, int, int] | None:
    if value is None:
        return None
    try:
        x1, y1, x2, y2 = [int(round(float(v))) for v in list(value)[:4]]
    except (TypeError, ValueError):
        return None
    return min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)


def _xyxy_to_rect_tuple(value) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = _normalize_xyxy(value) or (0, 0, 1, 1)
    return float(x1), float(y1), float(max(1, x2 - x1)), float(max(1, y2 - y1))


def _bbox_has_area(box: tuple[int, int, int, int]) -> bool:
    return box[2] > box[0] and box[3] > box[1]


def _bbox_area(box: tuple[int, int, int, int]) -> float:
    return float(max(0, box[2] - box[0]) * max(0, box[3] - box[1]))


def _bbox_center(box: tuple[int, int, int, int]) -> tuple[float, float]:
    return (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0


def _bbox_contains_point(box: tuple[int, int, int, int], point: tuple[float, float]) -> bool:
    px, py = point
    return box[0] <= px <= box[2] and box[1] <= py <= box[3]


def _intersection_area(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    return float(max(0, x2 - x1) * max(0, y2 - y1))


def _clamp_xyxy_to_image(box: tuple[int, int, int, int], img) -> tuple[int, int, int, int] | None:
    if img is None or not hasattr(img, "shape"):
        return box
    try:
        height, width = int(img.shape[0]), int(img.shape[1])
    except (TypeError, ValueError, IndexError):
        return box
    if width <= 0 or height <= 0:
        return None
    x1 = max(0, min(width - 1, box[0]))
    y1 = max(0, min(height - 1, box[1]))
    x2 = max(0, min(width, box[2]))
    y2 = max(0, min(height, box[3]))
    return x1, y1, x2, y2


def _greedy_wrap_lines(text: str, font_size: float, eval_metrics, roi_width: float, roi_height: float, vertical: bool) -> list[str]:
    units = _wrap_units(text)
    if not units:
        return [str(text or "")]
    lines: list[str] = []
    cursor = 0
    while cursor < len(units):
        best = cursor + 1
        for end in range(cursor + 1, len(units) + 1):
            candidate = _join_units(units, cursor, end)
            w, h = eval_metrics(candidate, font_size, vertical)
            side, side_roi = (h, roi_height) if vertical else (w, roi_width)
            if side <= side_roi or end == cursor + 1:
                best = end
            else:
                break
        lines.append(_join_units(units, cursor, best))
        cursor = best
    return lines


def _balanced_wrap_lines(text: str, font_size: float, eval_metrics, roi_width: float, roi_height: float) -> list[str]:
    paragraphs = str(text or "").split("\n")
    output: list[str] = []
    for paragraph in paragraphs:
        units = _wrap_units(paragraph)
        if not units:
            output.append("")
            continue
        wrapped = _balanced_wrap_paragraph(units, font_size, eval_metrics, roi_width, roi_height)
        output.extend(wrapped or [_join_units(units, 0, len(units))])
    return output


def _balanced_wrap_paragraph(
    units: list[tuple[str, str]],
    font_size: float,
    eval_metrics,
    roi_width: float,
    roi_height: float,
) -> list[str]:
    count = len(units)
    if count == 1:
        return [_join_units(units, 0, 1)]

    line_widths: dict[tuple[int, int], float] = {}
    for start in range(count):
        for end in range(start + 1, count + 1):
            if not _is_legal_line_break(units, start, end):
                continue
            candidate = _join_units(units, start, end)
            width, _height = eval_metrics(candidate, font_size, False, include_outline=False)
            if width <= roi_width or end == start + 1:
                line_widths[(start, end)] = width
            elif end > start + 1:
                break

    max_lines = min(count, 12)
    best_lines: list[str] | None = None
    best_score = float("inf")
    for line_count in range(1, max_lines + 1):
        if line_count > count:
            break
        candidate = _wrap_paragraph_exact_lines(units, line_widths, roi_width, line_count)
        if candidate is None:
            continue
        lines, score = candidate
        width, height = eval_metrics("\n".join(lines), font_size, False)
        if width > roi_width or height > roi_height:
            continue
        score += (height / max(1.0, roi_height)) * 0.05
        if score < best_score:
            best_lines = lines
            best_score = score
    return best_lines or []


def _wrap_paragraph_exact_lines(
    units: list[tuple[str, str]],
    line_widths: dict[tuple[int, int], float],
    roi_width: float,
    line_count: int,
) -> tuple[list[str], float] | None:
    count = len(units)
    dp: list[dict[int, tuple[float, int]]] = [dict() for _ in range(line_count + 1)]
    dp[0][0] = (0.0, -1)
    for line_idx in range(1, line_count + 1):
        for end in range(line_idx, count + 1):
            best: tuple[float, int] | None = None
            for start in range(line_idx - 1, end):
                prev = dp[line_idx - 1].get(start)
                width = line_widths.get((start, end))
                if prev is None or width is None:
                    continue
                is_last = line_idx == line_count and end == count
                cost = prev[0] + _line_wrap_cost(width, roi_width, is_last, line_count)
                if best is None or cost < best[0]:
                    best = (cost, start)
            if best is not None:
                dp[line_idx][end] = best
    if count not in dp[line_count]:
        return None

    lines: list[str] = []
    end = count
    score = dp[line_count][count][0]
    for line_idx in range(line_count, 0, -1):
        start = dp[line_idx][end][1]
        lines.append(_join_units(units, start, end))
        end = start
    lines.reverse()
    return lines, score


def _line_wrap_cost(width: float, roi_width: float, is_last: bool, line_count: int) -> float:
    ratio = width / max(1.0, roi_width)
    slack = max(0.0, 1.0 - ratio)
    cost = slack * slack
    if is_last and line_count > 1 and ratio < 0.38:
        cost += (0.38 - ratio) * 1.2
    cost += line_count * 0.01
    return cost


def _wrap_units(paragraph: str) -> list[tuple[str, str]]:
    units: list[tuple[str, str]] = []
    value = str(paragraph or "")
    for match in re.finditer(r"\S+", value):
        word = match.group(0)
        prefix = " " if units and match.start() > 0 and value[match.start() - 1].isspace() else ""
        if _should_split_inside_word(word):
            for index, char in enumerate(word):
                units.append((prefix if index == 0 else "", char))
        else:
            units.append((prefix, word))
    return units


def _should_split_inside_word(word: str) -> bool:
    value = str(word or "")
    return len(value) > 12 and bool(_CJK_RE.search(value))


def _join_units(units: list[tuple[str, str]], start: int, end: int) -> str:
    if start >= end:
        return ""
    parts = [units[start][1]]
    for index in range(start + 1, end):
        prefix, text = units[index]
        parts.append(f"{prefix}{text}")
    return "".join(parts)


def _is_legal_line_break(units: list[tuple[str, str]], start: int, end: int) -> bool:
    if start >= end:
        return False
    current = _join_units(units, start, end).strip()
    if not current:
        return False
    if current[-1] in _BREAK_AFTER_FORBIDDEN:
        return False
    if end < len(units):
        next_text = units[end][1].strip()
        if next_text and next_text[0] in _BREAK_BEFORE_FORBIDDEN:
            return False
    return True


def pyside_word_wrap(
    text: str, 
    font_input: str, 
    roi_width: int, 
    roi_height: int,
    line_spacing: float, 
    outline_width: float, 
    bold: bool, 
    italic: bool, 
    underline: bool, 
    alignment: Qt.AlignmentFlag,
    direction: Qt.LayoutDirection, 
    init_font_size: int, 
    min_font_size: int = 10, 
    vertical: bool = False,
    fit_clearance: float = 0.0,
    return_metrics: bool = False
) -> tuple:
    
    """Break long text to multiple lines, and find the largest point size
        so that all wrapped text fits within the box."""
    
    def prepare_font(font_size):
        effective_family = font_input.strip() if isinstance(font_input, str) and font_input.strip() else QApplication.font().family()
        font = QFont(effective_family, font_size)
        font.setBold(bold)
        font.setItalic(italic)
        font.setUnderline(underline)

        return font

    fallback_font_family = resolve_render_symbol_fallback_font_family()

    def eval_metrics(
        txt: str,
        font_sz: float,
        vertical: bool = False,
        include_outline: bool = True
    ) -> Tuple[float, float]:
        """Quick helper function to calculate width/height of text using QTextDocument."""
        
        doc = QTextDocument()
        doc.setDefaultFont(prepare_font(font_sz))
        if not vertical and fallback_font_family and any(ch in RENDER_NORMALIZABLE_GLYPHS for ch in txt):
            styled = build_styled_render_html(
                txt,
                font_family=font_input,
                font_size=font_sz,
                alignment=alignment,
                line_spacing=line_spacing,
                bold=bold,
                italic=italic,
                underline=underline,
                direction=direction,
                fallback_font_family=fallback_font_family,
            )
            doc.setHtml(styled.html_text)
        else:
            doc.setPlainText(txt)

        # Set text direction
        text_option = QTextOption()
        text_option.setTextDirection(direction)
        doc.setDefaultTextOption(text_option)

        if vertical:
            layout = VerticalTextDocumentLayout(
                document=doc,
                line_spacing=line_spacing
            )

            doc.setDocumentLayout(layout)
            layout.update_layout()
        else:
            # Apply line spacing
            cursor = QTextCursor(doc)
            cursor.select(QTextCursor.SelectionType.Document)
            block_format = QTextBlockFormat()
            spacing = line_spacing * 100
            block_format.setLineHeight(spacing, QTextBlockFormat.LineHeightTypes.ProportionalHeight.value)
            block_format.setAlignment(alignment)
            cursor.mergeBlockFormat(block_format)
        
        # Get the size of the document
        size = doc.size()
        width, height = size.width(), size.height()
        
        # Add outline width to the size
        if include_outline and outline_width > 0:
            width += 2 * outline_width
            height += 2 * outline_width
        
        return width, height

    try:
        clearance = max(0.0, float(fit_clearance))
    except (TypeError, ValueError):
        clearance = 0.0
    fit_roi_width = max(
        DETECTED_BUBBLE_MIN_FIT_DIMENSION_PX,
        float(roi_width) - (clearance * 2.0),
    )
    fit_roi_height = max(
        DETECTED_BUBBLE_MIN_FIT_DIMENSION_PX,
        float(roi_height) - (clearance * 2.0),
    )

    def wrap_and_size(font_size):
        if vertical:
            lines = _greedy_wrap_lines(text, font_size, eval_metrics, fit_roi_width, fit_roi_height, vertical)
        else:
            lines = _balanced_wrap_lines(text, font_size, eval_metrics, fit_roi_width, fit_roi_height)
            if not lines:
                lines = _greedy_wrap_lines(text, font_size, eval_metrics, fit_roi_width, fit_roi_height, vertical)
        wrapped = "\n".join(lines)
        w, h = eval_metrics(wrapped, font_size, vertical)
        return wrapped, w, h
    
    # Initialize
    best_text, best_size = text, init_font_size
    found_fit = False

    readable_min_font_size = min(int(init_font_size), max(int(min_font_size), 12))
    lo, hi = readable_min_font_size, init_font_size
    while lo <= hi:
        mid = (lo + hi) // 2
        wrapped, w, h = wrap_and_size(mid)
        if w <= fit_roi_width and h <= fit_roi_height:
            found_fit = True
            best_text, best_size = wrapped, mid
            lo = mid + 1
        else:
            hi = mid - 1

    # If nothing fits, keep the configured readable floor instead of shrinking
    # text into an unreadable 1pt fallback. The caller can flag the block for
    # review from the returned metrics if the document exceeds the box.
    if not found_fit:
        best_text, _w, _h = wrap_and_size(readable_min_font_size)
        best_size = readable_min_font_size

    if return_metrics:
        # Match persisted state to the text item's actual geometry.
        rendered_w, rendered_h = eval_metrics(best_text, best_size, vertical, include_outline=False)
        return best_text, best_size, rendered_w, rendered_h

    return best_text, best_size

    # mutable_message = text
    # font_size = init_font_size
    # # font_size = max(roi_width, roi_height)

    # while font_size > min_font_size:
    #     width, height = eval_metrics(mutable_message, font_size)
    #     if height > roi_height:
    #         font_size -= 1  # Reduce pointsize
    #         mutable_message = text  # Restore original text
    #     elif width > roi_width:
    #         columns = len(mutable_message)
    #         while columns > 0:
    #             columns -= 1
    #             if columns == 0:
    #                 break
    #             mutable_message = '\n'.join(hyphen_wrap(text, columns, break_on_hyphens=False, break_long_words=False, hyphenate_broken_words=True)) 
    #             wrapped_width, _ = eval_metrics(mutable_message, font_size)
    #             if wrapped_width <= roi_width:
    #                 break
    #         if columns < 1:
    #             font_size -= 1  # Reduce pointsize
    #             mutable_message = text  # Restore original text
    #     else:
    #         break

    # if font_size <= min_font_size:
    #     font_size = min_font_size
    #     mutable_message = text

    #     # Wrap text to fit within as much as possible
    #     # Minimize cost function: (width - roi_width)^2 + (height - roi_height)^2
    #     min_cost = 1e9
    #     min_text = text
    #     for columns in range(1, len(text)):
    #         wrapped_text = '\n'.join(hyphen_wrap(text, columns, break_on_hyphens=False, break_long_words=False, hyphenate_broken_words=True))
    #         wrapped_width, wrapped_height = eval_metrics(wrapped_text, font_size)
    #         cost = (wrapped_width - roi_width)**2 + (wrapped_height - roi_height)**2
    #         if cost < min_cost:
    #             min_cost = cost
    #             min_text = wrapped_text

    #     mutable_message = min_text

    # return mutable_message, font_size

def manual_wrap(
    main_page, 
    blk_list: List[TextBlock], 
    image_path: str,
    font_family: str, 
    line_spacing: float, 
    outline_width: float, 
    bold: bool, 
    italic: bool, 
    underline: bool, 
    alignment: Qt.AlignmentFlag, 
    direction: Qt.LayoutDirection, 
    init_font_size: int = 40, 
    min_font_size: int = 10
):
    
    target_lang = main_page.lang_mapping.get(main_page.t_combo.currentText(), None)
    trg_lng_cd = get_language_code(target_lang)
    get_best_render_area(blk_list, getattr(main_page, "image", None))

    for block_index, blk in enumerate(blk_list):
        x1, y1, width, height = blk.xywh

        translation = sanitize_render_text(
            blk.translation,
            font_family,
            block_index=block_index,
            image_path=image_path,
        )
        if not translation or len(translation) == 1:
            continue

        vertical = is_vertical_block(blk, trg_lng_cd)

        translation, font_size = pyside_word_wrap(
            translation, 
            font_family, 
            width, 
            height,
            line_spacing, 
            outline_width, 
            bold, 
            italic, 
            underline,
            alignment, 
            direction, 
            init_font_size, 
            min_font_size,
            vertical,
            fit_clearance=get_render_fit_clearance_for_block(blk, outline_width),
        )
        render_markup = describe_render_text_markup(
            translation,
            font_family=font_family,
            font_size=font_size,
            alignment=alignment,
            line_spacing=line_spacing,
            bold=bold,
            italic=italic,
            underline=underline,
            direction=direction,
        )
        blk._render_text = str(translation or "")
        blk._render_html = str(
            render_markup.html_text if render_markup.html_applied else translation or ""
        )
        blk._render_html_applied = bool(render_markup.html_applied)
        blk._render_fallback_font_family = str(render_markup.fallback_font_family or "")
        main_page.blk_rendered.emit(translation, font_size, blk, image_path)



        

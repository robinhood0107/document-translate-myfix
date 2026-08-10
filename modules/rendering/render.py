import logging
import numpy as np
import html
import re
import os
from typing import Tuple, List
import unicodedata
from functools import lru_cache

from PIL import Image, ImageFont, ImageDraw
from PySide6.QtGui import QFont, QFontMetrics, QTextDocument,\
      QTextCursor, QTextBlockFormat, QTextOption, QFontDatabase, QRawFont
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from .hyphen_textwrap import wrap as hyphen_wrap
from modules.utils.block_geometry import (
    copy_block_xyxy,
    render_geometry_is_coherent,
    resolve_render_recompute_anchor_xyxy,
)
from modules.utils.textblock import TextBlock
from modules.utils.textblock import adjust_blks_size
from modules.detection.utils.geometry import shrink_bbox
from app.ui.canvas.text.vertical_layout import VerticalTextDocumentLayout
from modules.utils.language_utils import get_language_code
from modules.utils.text_normalization import (
    OCR_DECORATIVE_NOISE_GLYPHS,
    RENDER_NORMALIZABLE_GLYPHS,
    canonicalize_ellipsis_runs,
    strip_unsafe_text_control_chars,
)
from modules.utils.repetition_guard import guard_severe_repetition
from modules.utils.render_style_policy import (
    VERTICAL_ALIGNMENT_CENTER,
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
    auto_max_font_size: bool = True
    auto_max_font_profile: str = "current"


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


@dataclass(frozen=True)
class TextFreeMangaLayoutPolicy:
    enabled: bool
    alignment: Qt.AlignmentFlag
    vertical_alignment: str
    wrap_width: int
    item_width: float
    reasons: list[str]


@dataclass(frozen=True)
class RenderBlockGateDecision:
    render: bool
    status: str
    reasons: tuple[str, ...] = ()


STRICT_RENDER_DROP_GLYPHS = frozenset(
    {
        "♥",
        "♡",
        "❤",
        "❤︎",
        "❤️",
        "・",
        "･",
        "♪",
        "♫",
        "♬",
        "★",
        "☆",
        "※",
        "→",
        "←",
        "↑",
        "↓",
        "↗",
        "↘",
        "↙",
        "↖",
    }
)

TEXT_FREE_MARKER_ONLY_GLYPHS = frozenset(
    {
        "-",
        "‐",
        "‑",
        "‒",
        "–",
        "—",
        "―",
        "−",
        "ー",
        "ｰ",
        "─",
        "━",
        "_",
        "＿",
        "~",
        "〜",
        "～",
        "…",
        "⋯",
        "⋮",
        "・",
        "･",
    }
)

AUTO_RENDER_SKIP_REVIEW_STATUSES = frozenset(
    {
        "needs_review_embedded_ui_panel_layout",
        "needs_review_text_free_translation",
        "needs_review_text_free_mask",
        "needs_review_text_free_underfilled",
        "skipped_duplicate_bubble_text",
        "skipped_text_free_marker_only",
    }
)

TEXT_FREE_UNDERFILL_MIN_SOURCE_AREA = 24000.0
TEXT_FREE_UNDERFILL_MIN_RENDERED_AREA_RATIO = 0.085


def should_use_strict_render_symbols(target_lang_code: str | None) -> bool:
    return str(target_lang_code or "").strip().lower() in {"ko", "kor", "korean"}


def _is_text_free_marker_only_text(text: object) -> bool:
    chars: list[str] = []
    for ch in str(text or ""):
        category = unicodedata.category(ch)
        if ch.isspace() or category in {"Cc", "Cf", "Mn", "Me"}:
            continue
        chars.append(ch)
    if not chars:
        return False
    for ch in chars:
        category = unicodedata.category(ch)
        if ch in TEXT_FREE_MARKER_ONLY_GLYPHS:
            continue
        if category[0] in {"P", "S"}:
            continue
        return False
    return True


def describe_auto_render_review_status_gate(status: object) -> RenderBlockGateDecision:
    normalized = str(status or "").strip()
    if normalized in AUTO_RENDER_SKIP_REVIEW_STATUSES:
        return RenderBlockGateDecision(
            False,
            normalized,
            ("render_skipped_review_gate", normalized),
        )
    return RenderBlockGateDecision(True, "ok")


def register_duplicate_bubble_render_key(
    blk: TextBlock,
    duplicate_key: tuple[tuple[int, int, int, int], str] | None,
    seen_bubble_render_keys: set[tuple[tuple[int, int, int, int], str]],
) -> RenderBlockGateDecision:
    if duplicate_key is None:
        return RenderBlockGateDecision(True, "ok")
    if duplicate_key in seen_bubble_render_keys:
        return RenderBlockGateDecision(
            False,
            "skipped_duplicate_bubble_text",
            ("skipped_duplicate_bubble_text",),
        )
    seen_bubble_render_keys.add(duplicate_key)
    if str(getattr(blk, "bubble_panel_merge_decision", "") or "") == "duplicate_member":
        blk.bubble_panel_merge_decision = "render_primary"
        return RenderBlockGateDecision(
            True,
            "ok",
            ("bubble_panel_group_render_primary",),
        )
    return RenderBlockGateDecision(True, "ok")


def should_skip_short_render_translation(blk: TextBlock, translation: object) -> bool:
    text = str(translation or "")
    if not text.strip():
        return True
    if len(text) != 1:
        return False
    return str(getattr(blk, "text_class", "") or "").strip().lower() != "text_bubble"


def _meaningful_render_char_count(text: object) -> int:
    count = 0
    for ch in str(text or ""):
        category = unicodedata.category(ch)
        if category.startswith("L") or category.startswith("N"):
            count += 1
    return count


def _normalized_render_identity_text(text: object) -> str:
    chars: list[str] = []
    for ch in str(text or "").casefold():
        category = unicodedata.category(ch)
        if category.startswith("L") or category.startswith("N"):
            chars.append(ch)
    return "".join(chars)


def _block_source_text(blk: TextBlock) -> str:
    getter = getattr(blk, "get_text", None)
    if callable(getter):
        try:
            return str(getter() or "")
        except Exception:
            pass
    return str(getattr(blk, "text", "") or "")


def describe_text_free_render_translation_gate(
    blk: TextBlock,
    translation: object,
    *,
    target_lang_code: str | None,
) -> RenderBlockGateDecision:
    """Reject text_free render results that are far larger than source evidence."""
    if str(target_lang_code or "").strip().lower() not in {"ko", "kor", "korean"}:
        return RenderBlockGateDecision(True, "ok")
    if str(getattr(blk, "text_class", "") or "").strip().lower() != "text_free":
        return RenderBlockGateDecision(True, "ok")

    source_text = _block_source_text(blk)
    if _is_text_free_marker_only_text(source_text):
        return RenderBlockGateDecision(
            False,
            "skipped_text_free_marker_only",
            ("text_free_marker_only",),
        )

    source_len = _meaningful_render_char_count(source_text)
    target_len = _meaningful_render_char_count(translation)
    if source_len <= 0 or target_len <= 0:
        return RenderBlockGateDecision(True, "ok")

    overexpanded = source_len <= 8 and target_len >= max(24, source_len * 6)
    if not overexpanded:
        return RenderBlockGateDecision(True, "ok")

    reasons = ["text_free_translation_overexpanded"]
    source_xyxy = _normalize_xyxy(getattr(blk, "xyxy", None))
    if source_xyxy is not None:
        width = max(1, source_xyxy[2] - source_xyxy[0])
        height = max(1, source_xyxy[3] - source_xyxy[1])
        if height >= width * 2.0 or width >= height * 2.0:
            reasons.append("text_free_marker_like_bbox")
    return RenderBlockGateDecision(
        False,
        "needs_review_text_free_translation",
        tuple(reasons),
    )


def describe_text_free_render_mask_gate(
    blk: TextBlock,
    *,
    target_lang_code: str | None,
) -> RenderBlockGateDecision:
    if str(target_lang_code or "").strip().lower() not in {"ko", "kor", "korean"}:
        return RenderBlockGateDecision(True, "ok")
    if str(getattr(blk, "text_class", "") or "").strip().lower() != "text_free":
        return RenderBlockGateDecision(True, "ok")
    if not hasattr(blk, "block_final_mask_pixel_count"):
        return RenderBlockGateDecision(True, "ok")

    mask_pixels = int(getattr(blk, "block_final_mask_pixel_count", 0) or 0)
    if mask_pixels <= 0:
        return RenderBlockGateDecision(
            False,
            "needs_review_text_free_mask",
            ("render_without_erase_mask",),
        )

    return RenderBlockGateDecision(True, "ok")


def describe_text_free_underfill_gate(
    blk: TextBlock,
    *,
    source_rect: tuple[float, float, float, float],
    rendered_width: float,
    rendered_height: float,
    target_lang_code: str | None,
) -> RenderBlockGateDecision:
    if str(target_lang_code or "").strip().lower() not in {"ko", "kor", "korean"}:
        return RenderBlockGateDecision(True, "ok")
    if str(getattr(blk, "text_class", "") or "").strip().lower() != "text_free":
        return RenderBlockGateDecision(True, "ok")
    source_width = max(1.0, float(source_rect[2]))
    source_height = max(1.0, float(source_rect[3]))
    source_area = source_width * source_height
    if source_area < TEXT_FREE_UNDERFILL_MIN_SOURCE_AREA:
        return RenderBlockGateDecision(True, "ok")
    rendered_area = max(1.0, float(rendered_width) * float(rendered_height))
    ratio = rendered_area / source_area
    if ratio >= TEXT_FREE_UNDERFILL_MIN_RENDERED_AREA_RATIO:
        return RenderBlockGateDecision(True, "ok")
    return RenderBlockGateDecision(
        False,
        "needs_review_text_free_underfilled",
        ("text_free_underfilled",),
    )


def describe_text_free_large_mask_gate(
    blk: TextBlock,
    *,
    source_rect: tuple[float, float, float, float],
    target_lang_code: str | None,
) -> RenderBlockGateDecision:
    return RenderBlockGateDecision(True, "ok")


def block_needs_original_restore_after_render(blk: TextBlock) -> bool:
    mask_pixels = int(getattr(blk, "block_final_mask_pixel_count", getattr(blk, "_final_mask_pixel_count", 0)) or 0)
    if mask_pixels <= 0:
        return False
    if bool(getattr(blk, "_render_restore_applied", False)):
        return False
    status = str(getattr(blk, "_render_skip_reason", "") or getattr(blk, "_text_fit_status", "") or "")
    if status in AUTO_RENDER_SKIP_REVIEW_STATUSES:
        return True
    render_text = str(getattr(blk, "_render_text", "") or "")
    translation_raw = str(getattr(blk, "_render_translation_raw", getattr(blk, "translation", "")) or "")
    if not translation_raw.strip():
        return True
    if not render_text.strip() or _meaningful_render_char_count(render_text) <= 0:
        return True
    return False


def select_blocks_for_original_restore_after_render(blocks) -> list[TextBlock]:
    block_list = list(blocks or [])
    rendered_bubble_panel_groups: set[str] = set()
    for block in block_list:
        group_id = str(getattr(block, "bubble_panel_group_id", "") or "")
        if not group_id or not bool(getattr(block, "bubble_panel_text_candidate", False)):
            continue
        status = str(getattr(block, "_render_skip_reason", "") or getattr(block, "_text_fit_status", "") or "")
        render_text = str(getattr(block, "_render_text", "") or "")
        if status in AUTO_RENDER_SKIP_REVIEW_STATUSES:
            continue
        if render_text.strip() and _meaningful_render_char_count(render_text) > 0:
            rendered_bubble_panel_groups.add(group_id)

    restore_blocks: list[TextBlock] = []
    for block in block_list:
        if not block_needs_original_restore_after_render(block):
            continue
        group_id = str(getattr(block, "bubble_panel_group_id", "") or "")
        if (
            group_id
            and bool(getattr(block, "bubble_panel_text_candidate", False))
            and group_id in rendered_bubble_panel_groups
        ):
            block._render_restore_suppressed_reason = "bubble_panel_group_rendered"
            continue
        restore_blocks.append(block)
    return restore_blocks


def build_duplicate_bubble_render_key(blk: TextBlock) -> tuple[tuple[int, int, int, int], str] | None:
    """Return a stable page-local key for duplicate OCR blocks inside one bubble."""
    if str(getattr(blk, "text_class", "") or "").strip().lower() != "text_bubble":
        return None
    bubble_xyxy = _normalize_xyxy(getattr(blk, "bubble_xyxy", None))
    if bubble_xyxy is None or not _bbox_has_area(bubble_xyxy):
        return None
    source_key = _normalized_render_identity_text(_block_source_text(blk))
    if len(source_key) < 4:
        return None
    quantized = tuple(int(round(float(v) / 32.0)) for v in bubble_xyxy)
    return quantized, source_key


def _is_strict_render_forbidden_symbol(ch: str) -> bool:
    if not ch or ch in {"\n", "\r", "\t"}:
        return False
    if ch in STRICT_RENDER_DROP_GLYPHS:
        return True
    return unicodedata.category(ch).startswith("S")


@dataclass(frozen=True)
class AutoMaxFontProfile:
    horizontal_shrink_percent: float
    vertical_shrink_percent: float
    fit_clearance_px: float
    height_ratio: float
    width_ratio: float
    font_cap: int


RENDER_SYMBOL_FALLBACK_FONT_CANDIDATES = (
    "Malgun Gothic",
    "Yu Gothic UI",
    "Meiryo",
    "MS Gothic",
    "Segoe UI Symbol",
    "Segoe UI Emoji",
)

RENDER_FALLBACK_SYSTEM_FONT_FILES = (
    "malgun.ttf",
    "malgunbd.ttf",
    "meiryo.ttc",
    "YuGothR.ttc",
    "YuGothM.ttc",
    "msgothic.ttc",
    "seguisym.ttf",
    "seguiemj.ttf",
)

AUTO_MAX_FONT_PROFILE_CURRENT = "current"
AUTO_MAX_FONT_PROFILE_STRONG = "strong"
DEFAULT_AUTO_MAX_FONT_PROFILE = AUTO_MAX_FONT_PROFILE_CURRENT

AUTO_MAX_FONT_PROFILES = {
    AUTO_MAX_FONT_PROFILE_CURRENT: AutoMaxFontProfile(
        horizontal_shrink_percent=0.18,
        vertical_shrink_percent=0.30,
        fit_clearance_px=8.0,
        height_ratio=0.45,
        width_ratio=0.32,
        font_cap=160,
    ),
    AUTO_MAX_FONT_PROFILE_STRONG: AutoMaxFontProfile(
        horizontal_shrink_percent=0.14,
        vertical_shrink_percent=0.26,
        fit_clearance_px=7.0,
        height_ratio=0.58,
        width_ratio=0.42,
        font_cap=190,
    ),
}

HORIZONTAL_BUBBLE_SHRINK_PERCENT = AUTO_MAX_FONT_PROFILES[AUTO_MAX_FONT_PROFILE_CURRENT].horizontal_shrink_percent
VERTICAL_BUBBLE_SHRINK_PERCENT = AUTO_MAX_FONT_PROFILES[AUTO_MAX_FONT_PROFILE_CURRENT].vertical_shrink_percent
MIN_BUBBLE_TEXT_CONTAINMENT = 0.60
MIN_BUBBLE_RENDER_AREA_GAIN = 0.90
DETECTED_BUBBLE_FIT_CLEARANCE_PX = AUTO_MAX_FONT_PROFILES[AUTO_MAX_FONT_PROFILE_CURRENT].fit_clearance_px
DETECTED_BUBBLE_OUTLINE_CLEARANCE_MULTIPLIER = 2.0
DETECTED_BUBBLE_MIN_FIT_DIMENSION_PX = 16.0
DETECTED_BUBBLE_DYNAMIC_FONT_HEIGHT_RATIO = AUTO_MAX_FONT_PROFILES[AUTO_MAX_FONT_PROFILE_CURRENT].height_ratio
DETECTED_BUBBLE_DYNAMIC_FONT_WIDTH_RATIO = AUTO_MAX_FONT_PROFILES[AUTO_MAX_FONT_PROFILE_CURRENT].width_ratio
DETECTED_BUBBLE_DYNAMIC_FONT_CAP = AUTO_MAX_FONT_PROFILES[AUTO_MAX_FONT_PROFILE_CURRENT].font_cap
DETECTED_BUBBLE_MAX_RENDER_AREA_OVERLAP_RATIO = 0.12

_CJK_RE = re.compile(r"[\uac00-\ud7a3\u3040-\u30ff\u4e00-\u9fff]")
_BREAK_BEFORE_FORBIDDEN = set(".,!?;:)]}，。！？、；：）」』】》〉…")
_BREAK_AFTER_FORBIDDEN = set("([{（「『【《〈")


def normalize_auto_max_font_profile(value: object) -> str:
    profile = str(value or "").strip().casefold()
    if profile in AUTO_MAX_FONT_PROFILES:
        return profile
    return DEFAULT_AUTO_MAX_FONT_PROFILE


def get_auto_max_font_profile(value: object = DEFAULT_AUTO_MAX_FONT_PROFILE) -> AutoMaxFontProfile:
    return AUTO_MAX_FONT_PROFILES[normalize_auto_max_font_profile(value)]

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


@lru_cache(maxsize=4096)
def _render_font_has_real_glyph(font_family: str, ch: str) -> bool:
    if not ch or ch in {"\n", "\r", "\t"}:
        return True
    family = str(font_family or "").strip()
    if not family:
        return True
    try:
        raw_font = QRawFont.fromFont(QFont(family, 32))
        glyph_indexes = raw_font.glyphIndexesForString(ch)
    except Exception:
        return True
    if not glyph_indexes:
        return False
    return all(int(index or 0) > 0 for index in glyph_indexes)


@lru_cache(maxsize=1)
def _register_render_fallback_system_fonts() -> tuple[str, ...]:
    font_roots = []
    for root in (os.environ.get("WINDIR"), os.environ.get("SystemRoot"), r"C:\Windows"):
        if root:
            font_roots.append(os.path.join(root, "Fonts"))
    registered: list[str] = []
    seen_paths: set[str] = set()
    for font_root in font_roots:
        for filename in RENDER_FALLBACK_SYSTEM_FONT_FILES:
            path = os.path.normpath(os.path.join(font_root, filename))
            lower_path = path.casefold()
            if lower_path in seen_paths:
                continue
            seen_paths.add(lower_path)
            if not os.path.isfile(path):
                continue
            try:
                font_id = QFontDatabase.addApplicationFont(path)
            except Exception:
                continue
            if font_id == -1:
                continue
            try:
                registered.extend(QFontDatabase.applicationFontFamilies(font_id))
            except Exception:
                continue
    return tuple(dict.fromkeys(registered))


def _render_fallback_family_map() -> dict[str, str]:
    families = {family.casefold(): family for family in QFontDatabase.families()}
    if not any(candidate.casefold() in families for candidate in RENDER_SYMBOL_FALLBACK_FONT_CANDIDATES):
        _register_render_fallback_system_fonts()
        families = {family.casefold(): family for family in QFontDatabase.families()}
    return families


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
    families = _render_fallback_family_map()
    required_chars = tuple(sorted(RENDER_NORMALIZABLE_GLYPHS))
    for candidate in RENDER_SYMBOL_FALLBACK_FONT_CANDIDATES:
        actual = families.get(candidate.casefold())
        if not actual:
            continue
        metrics = QFontMetrics(QFont(actual, 12))
        if all(_render_font_supports(metrics, ch) for ch in required_chars):
            return actual
    return ""


@lru_cache(maxsize=128)
def resolve_render_glyph_fallback_font_family(required_chars: tuple[str, ...]) -> str:
    chars = tuple(
        ch
        for ch in required_chars
        if ch and ch not in {"\n", "\r", "\t"}
    )
    if not chars:
        return ""

    families = _render_fallback_family_map()
    for candidate in RENDER_SYMBOL_FALLBACK_FONT_CANDIDATES:
        actual = families.get(candidate.casefold())
        if not actual:
            continue
        metrics = QFontMetrics(QFont(actual, 12))
        if all(
            _render_font_supports(metrics, ch)
            and _render_font_has_real_glyph(actual, ch)
            for ch in chars
        ):
            return actual
    return ""


def describe_render_text_sanitization(
    text: str,
    font_family: str,
    *,
    block_index: int | None = None,
    image_path: str = "",
    strict_symbols: bool = False,
) -> RenderSanitizationResult:
    if not text:
        return RenderSanitizationResult("", "", False, [], [])

    raw_text = str(text or "")
    cleaned_parts: list[str] = []
    replacements: list[dict] = []
    reasons: list[str] = []
    sanitized = canonicalize_ellipsis_runs(_canonicalize_render_symbol_variants(raw_text))
    unsafe_sanitized = strip_unsafe_text_control_chars(sanitized)
    unsafe_controls_removed = unsafe_sanitized != sanitized
    if unsafe_controls_removed:
        reasons.append("unsafe-control")
        replacements.append(
            {
                "index": 0,
                "char": sanitized,
                "replacement": unsafe_sanitized,
                "reason": "unsafe-control",
            }
        )
        sanitized = unsafe_sanitized
    repetition_guard = guard_severe_repetition(sanitized)
    if repetition_guard.changed:
        analysis = repetition_guard.analysis
        logger.warning(
            "render repetition guard applied: image=%s block=%s comparable_length=%d longest_run_char=%r longest_run_length=%d",
            image_path or "",
            block_index if block_index is not None else -1,
            analysis.comparable_length,
            analysis.longest_run_char,
            analysis.longest_run_length,
        )
        reasons.append(repetition_guard.reason)
        replacements.append(
            {
                "index": 0,
                "char": sanitized,
                "replacement": repetition_guard.text,
                "reason": repetition_guard.reason,
            }
        )
        sanitized = repetition_guard.text
    effective_family = font_family.strip() if isinstance(font_family, str) and font_family.strip() else QApplication.font().family()
    metrics = QFontMetrics(QFont(effective_family, 12))
    symbol_fallback_family = resolve_render_symbol_fallback_font_family()

    for index, ch in enumerate(sanitized):
        replacement = ch
        reason = ""
        if strict_symbols and _is_strict_render_forbidden_symbol(ch):
            replacement = ""
            reason = "render_sanitized_symbols"
        elif ch in OCR_DECORATIVE_NOISE_GLYPHS:
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

    normalized = re.sub(r"[ \t]{2,}", " ", "".join(cleaned_parts)).strip()
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
    strict_symbols: bool = False,
) -> RenderMarkupResult:
    if not text:
        return RenderMarkupResult("", "", False, [], "", [])

    raw_text = str(text or "")
    sanitization_reasons: list[str] = []
    sanitization_replacements: list[dict] = []
    if strict_symbols:
        sanitized = describe_render_text_sanitization(
            raw_text,
            font_family,
            strict_symbols=True,
        )
        raw_text = sanitized.text
        sanitization_reasons = list(sanitized.reasons)
        sanitization_replacements = list(sanitized.replacements)
        if not raw_text:
            return RenderMarkupResult(
                "",
                "",
                False,
                sanitization_reasons,
                "",
                sanitization_replacements,
            )
    fallback_font_family = resolve_render_symbol_fallback_font_family()
    use_full_html = font_size is not None
    if use_full_html:
        fallback_chars: set[str] = set()
        effective_font_family = str(font_family or "").strip() or QApplication.font().family()
        if fallback_font_family and effective_font_family:
            base_metrics = QFontMetrics(QFont(effective_font_family, max(1, int(round(float(font_size or 20))))))
        elif effective_font_family:
            base_metrics = QFontMetrics(QFont(effective_font_family, max(1, int(round(float(font_size or 20))))))
        else:
            base_metrics = None
        if base_metrics is not None:
            missing_chars: set[str] = set()
            for ch in raw_text:
                if ch in {"\n", "\r", "\t"}:
                    continue
                if ch in RENDER_NORMALIZABLE_GLYPHS:
                    continue
                if unicodedata.category(ch).startswith("S"):
                    continue
                base_has_glyph = _render_font_supports(base_metrics, ch) and _render_font_has_real_glyph(effective_font_family, ch)
                if not base_has_glyph:
                    missing_chars.add(ch)
            if missing_chars:
                glyph_fallback_family = fallback_font_family
                if not glyph_fallback_family:
                    glyph_fallback_family = resolve_render_glyph_fallback_font_family(tuple(sorted(missing_chars)))
                elif not all(
                    _render_font_has_real_glyph(glyph_fallback_family, ch)
                    for ch in missing_chars
                ):
                    glyph_fallback_family = resolve_render_glyph_fallback_font_family(tuple(sorted(missing_chars))) or glyph_fallback_family
                if glyph_fallback_family:
                    fallback_font_family = glyph_fallback_family
                    fallback_metrics = QFontMetrics(QFont(glyph_fallback_family, max(1, int(round(float(font_size or 20))))))
                    for ch in missing_chars:
                        fallback_has_glyph = (
                            _render_font_supports(fallback_metrics, ch)
                            and _render_font_has_real_glyph(glyph_fallback_family, ch)
                        )
                        if fallback_has_glyph:
                            fallback_chars.add(ch)
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
            fallback_chars=fallback_chars,
        )
        reasons = list(sanitization_reasons) + ["styled-render-html"]
        if any(item.get("reason") == "symbol-fallback-font" for item in styled.replacements):
            reasons.append("symbol-fallback-font")
        if any(item.get("reason") == "glyph-fallback-font" for item in styled.replacements):
            reasons.append("glyph-fallback-font")
        return RenderMarkupResult(
            text=raw_text,
            html_text=styled.html_text,
            html_applied=True,
            reasons=reasons,
            fallback_font_family=styled.fallback_font_family,
            replacements=sanitization_replacements + styled.replacements,
        )

    if not fallback_font_family:
        return RenderMarkupResult(
            raw_text,
            raw_text,
            False,
            sanitization_reasons,
            "",
            sanitization_replacements,
        )

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
        reasons=sanitization_reasons + (["symbol-fallback-font"] if replacements else []),
        fallback_font_family=fallback_font_family if replacements else "",
        replacements=sanitization_replacements + replacements,
    )


def sanitize_render_text(
    text: str,
    font_family: str,
    *,
    block_index: int | None = None,
    image_path: str = "",
    strict_symbols: bool = False,
) -> str:
    return describe_render_text_sanitization(
        text,
        font_family,
        block_index=block_index,
        image_path=image_path,
        strict_symbols=strict_symbols,
    ).text


def apply_strict_render_state_guard(
    text_item_state: dict,
    *,
    block_index: int | None = None,
    image_path: str = "",
) -> bool:
    """Final fail-safe for automatic exports that must not render symbols."""
    if not isinstance(text_item_state, dict):
        return False
    raw_text = str(
        text_item_state.get("render_text")
        or text_item_state.get("text")
        or ""
    )
    if not raw_text:
        return False
    result = describe_render_text_sanitization(
        raw_text,
        str(text_item_state.get("font_family", "") or ""),
        block_index=block_index,
        image_path=image_path,
        strict_symbols=True,
    )
    if result.text == raw_text and not result.normalization_applied:
        return False

    text_item_state["text"] = result.text
    text_item_state["render_text"] = result.text
    text_item_state["render_html_applied"] = False
    text_item_state["render_fallback_font_family"] = ""
    text_item_state["render_forbidden_symbol_guard"] = True
    reasons = set(text_item_state.get("render_normalization_reasons", []) or [])
    reasons.update(result.reasons)
    reasons.add("render_forbidden_symbol_guard")
    text_item_state["render_normalization_reasons"] = sorted(reasons)
    replacements = list(text_item_state.get("render_normalization_replacements", []) or [])
    replacements.extend(result.replacements)
    text_item_state["render_normalization_replacements"] = replacements
    return True


def apply_strict_render_viewer_state_guard(
    viewer_state: dict,
    *,
    image_path: str = "",
) -> int:
    if not isinstance(viewer_state, dict):
        return 0
    changed = 0
    for index, text_item_state in enumerate(viewer_state.get("text_items_state", []) or []):
        if apply_strict_render_state_guard(
            text_item_state,
            block_index=index,
            image_path=image_path,
        ):
            changed += 1
    if changed:
        viewer_state["render_forbidden_symbol_guard_count"] = changed
    return changed


def resolve_text_free_manga_layout(
    blk: TextBlock,
    source_rect: tuple[float, float, float, float],
    *,
    target_lang_code: str | None,
) -> TextFreeMangaLayoutPolicy:
    source_width = float(source_rect[2])
    source_height = float(source_rect[3])
    default_width = max(1, int(round(source_width)))
    disabled = TextFreeMangaLayoutPolicy(
        enabled=False,
        alignment=Qt.AlignmentFlag.AlignCenter,
        vertical_alignment=VERTICAL_ALIGNMENT_TOP,
        wrap_width=default_width,
        item_width=source_width,
        reasons=[],
    )
    if str(target_lang_code or "").strip().lower() != "ko":
        return disabled
    if str(getattr(blk, "text_class", "") or "").strip().lower() != "text_free":
        return disabled

    source_lang = str(getattr(blk, "source_lang", "") or "").strip().lower()
    direction = str(getattr(blk, "direction", "") or "").strip().lower()
    looks_japanese = source_lang in {"ja", "jpn", "japanese"}
    source_vertical = direction == "vertical" or str(getattr(blk, "source_lang_direction", "")).startswith("ver")
    tall_or_free_panel = source_height >= source_width * 1.15
    if not (looks_japanese or source_vertical or tall_or_free_panel):
        return disabled

    if source_width >= source_height:
        wrap_width = min(source_width, max(72.0, source_height * 0.9))
    else:
        wrap_width = min(source_width, max(48.0, source_width * 0.9))
    return TextFreeMangaLayoutPolicy(
        enabled=True,
        alignment=Qt.AlignmentFlag.AlignCenter,
        vertical_alignment=VERTICAL_ALIGNMENT_CENTER,
        wrap_width=max(1, int(round(wrap_width))),
        item_width=source_width,
        reasons=["render_centered_layout"],
    )

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
        if should_skip_short_render_translation(blk, translation):
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

def get_best_render_area(
    blk_list: List[TextBlock],
    img,
    inpainted_img=None,
    *,
    auto_max_font_profile: object = DEFAULT_AUTO_MAX_FONT_PROFILE,
):
    """Select safe text render areas without losing the original OCR anchor."""
    font_profile = get_auto_max_font_profile(auto_max_font_profile)
    image_shape = tuple(getattr(img, "shape", ()) or ())
    candidates: list[tuple[int, tuple[int, int, int, int]]] = []
    for block_index, blk in enumerate(blk_list):
        if isinstance(getattr(blk, "xyxy", None), tuple):
            blk.xyxy = list(blk.xyxy)
        preserve_original = (
            len(image_shape) >= 2
            and render_geometry_is_coherent(blk, image_shape)
        )
        original_anchor = (
            resolve_render_recompute_anchor_xyxy(blk, image_shape)
            if preserve_original
            else copy_block_xyxy(getattr(blk, "xyxy", None))
        )
        _reset_render_area_metadata(blk, original_anchor)
        if preserve_original and original_anchor is not None:
            _assign_block_xyxy(blk, original_anchor)
        text_draw_bounds = _detected_bubble_render_bounds(
            blk,
            img,
            font_profile,
            text_xyxy=original_anchor,
        )
        if text_draw_bounds is None:
            continue
        candidates.append((block_index, text_draw_bounds))

    conflict_candidate_indexes = _find_overlapping_detected_bubble_candidate_indexes(
        candidates,
        blk_list,
    )
    conflict_candidate_indexes.update(
        _find_detected_bubble_candidates_covering_other_text_indexes(candidates, blk_list)
    )
    for candidate_index, (block_index, text_draw_bounds) in enumerate(candidates):
        if candidate_index in conflict_candidate_indexes:
            continue
        blk = blk_list[block_index]
        bdx1, bdy1, bdx2, bdy2 = text_draw_bounds
        _assign_block_xyxy(blk, (bdx1, bdy1, bdx2, bdy2))
        blk._render_area_source = "detected_bubble"
        blk._render_area_xyxy = [int(bdx1), int(bdy1), int(bdx2), int(bdy2)]

    if img is not None and blk_list and blk_list[0].source_lang not in ['ko', 'zh']:
        adjust_blks_size(blk_list, img, -5, -5)

    return blk_list


def _blocks_are_duplicate_bubble_text(
    first_index: int,
    second_index: int,
    blk_list: List[TextBlock],
) -> bool:
    try:
        first_block = blk_list[first_index]
        second_block = blk_list[second_index]
    except (IndexError, TypeError):
        return False
    if not (
        bool(getattr(first_block, "bubble_panel_text_candidate", False))
        and bool(getattr(second_block, "bubble_panel_text_candidate", False))
    ):
        return False
    first_key = build_duplicate_bubble_render_key(first_block)
    second_key = build_duplicate_bubble_render_key(second_block)
    return first_key is not None and first_key == second_key


def _find_overlapping_detected_bubble_candidate_indexes(
    candidates: list[tuple[int, tuple[int, int, int, int]]],
    blk_list: List[TextBlock],
) -> set[int]:
    conflicts: set[int] = set()
    for i, (first_block_index, first) in enumerate(candidates):
        first_area = _bbox_area(first)
        if first_area <= 0.0:
            continue
        for j in range(i + 1, len(candidates)):
            second_block_index, second = candidates[j]
            if _blocks_are_duplicate_bubble_text(first_block_index, second_block_index, blk_list):
                continue
            second_area = _bbox_area(second)
            if second_area <= 0.0:
                continue
            overlap = _intersection_area(first, second)
            if overlap <= 0.0:
                continue
            overlap_ratio = overlap / max(1.0, min(first_area, second_area))
            if overlap_ratio > DETECTED_BUBBLE_MAX_RENDER_AREA_OVERLAP_RATIO:
                conflicts.update({i, j})
    return conflicts


def _find_detected_bubble_candidates_covering_other_text_indexes(
    candidates: list[tuple[int, tuple[int, int, int, int]]],
    blk_list: List[TextBlock],
) -> set[int]:
    conflicts: set[int] = set()
    original_boxes = [
        _normalize_xyxy(getattr(blk, "_render_original_xyxy", None)) or _current_anchor_xyxy(blk)
        for blk in blk_list
    ]
    for candidate_index, (block_index, candidate_box) in enumerate(candidates):
        candidate_area = _bbox_area(candidate_box)
        if candidate_area <= 0.0:
            continue
        for other_index, other_box in enumerate(original_boxes):
            if other_index == block_index or other_box is None:
                continue
            if _blocks_are_duplicate_bubble_text(block_index, other_index, blk_list):
                continue
            other_area = _bbox_area(other_box)
            if other_area <= 0.0:
                continue
            overlap = _intersection_area(candidate_box, other_box)
            if overlap <= 0.0:
                continue
            overlap_ratio = overlap / max(1.0, min(candidate_area, other_area))
            if overlap_ratio > DETECTED_BUBBLE_MAX_RENDER_AREA_OVERLAP_RATIO:
                conflicts.add(candidate_index)
                break
    return conflicts


def build_render_rects_for_block(blk: TextBlock) -> tuple[tuple[float, float, float, float], tuple[float, float, float, float]]:
    """Return layout source_rect and original OCR block_anchor for a render block."""
    render_area = None
    if getattr(blk, "_render_area_source", "") == "detected_bubble":
        render_area = _normalize_xyxy(getattr(blk, "_render_area_xyxy", None))
    if getattr(blk, "bubble_panel_text_candidate", False):
        render_area = _normalize_xyxy(getattr(blk, "bubble_panel_render_xyxy", None)) or render_area
    source_rect = _xyxy_to_rect_tuple(render_area or getattr(blk, "xyxy", None))
    anchor_xyxy = _normalize_xyxy(getattr(blk, "_render_original_xyxy", None)) or _current_anchor_xyxy(blk)
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
    *,
    auto_max_font_profile: object = DEFAULT_AUTO_MAX_FONT_PROFILE,
) -> float:
    """Return extra inner fit clearance for text rendered inside detected bubbles."""
    if getattr(blk, "_render_area_source", "") != "detected_bubble":
        return 0.0
    font_profile = get_auto_max_font_profile(auto_max_font_profile)
    try:
        outline = max(0.0, float(outline_width))
    except (TypeError, ValueError):
        outline = 0.0
    return max(
        font_profile.fit_clearance_px,
        (outline * DETECTED_BUBBLE_OUTLINE_CLEARANCE_MULTIPLIER)
        + font_profile.fit_clearance_px,
    )


def get_dynamic_bubble_font_cap(
    blk: TextBlock,
    configured_max_font_size: int | float,
    rendered_width: float,
    rendered_height: float,
    vertical: bool,
    final_font_size: int | float | None = None,
    *,
    auto_max_font_profile: object = DEFAULT_AUTO_MAX_FONT_PROFILE,
) -> int:
    """Return a larger cap only for underfilled detected text bubbles."""
    try:
        base_max = int(round(float(configured_max_font_size)))
    except (TypeError, ValueError):
        return 0
    if base_max <= 0:
        return base_max
    if vertical or getattr(blk, "direction", "") == "vertical":
        return base_max
    if getattr(blk, "text_class", "") != "text_bubble":
        return base_max
    if getattr(blk, "_render_area_source", "") != "detected_bubble":
        return base_max
    if final_font_size is not None:
        try:
            if float(final_font_size) < float(base_max):
                return base_max
        except (TypeError, ValueError):
            return base_max

    source_xyxy = _normalize_xyxy(getattr(blk, "_render_area_xyxy", None))
    if source_xyxy is None or not _bbox_has_area(source_xyxy):
        return base_max
    source_width = float(source_xyxy[2] - source_xyxy[0])
    source_height = float(source_xyxy[3] - source_xyxy[1])
    if source_width <= 0.0 or source_height <= 0.0:
        return base_max

    font_profile = get_auto_max_font_profile(auto_max_font_profile)
    height_cap = int(source_height * font_profile.height_ratio)
    width_cap = int(source_width * font_profile.width_ratio)
    dynamic_cap = max(base_max, height_cap, width_cap)
    return int(min(font_profile.font_cap, dynamic_cap))


def _reset_render_area_metadata(
    blk: TextBlock,
    original: tuple[int | float, int | float, int | float, int | float] | None,
) -> None:
    bubble = _normalize_xyxy(getattr(blk, "bubble_xyxy", None))
    blk._render_original_xyxy = list(original) if original is not None else None
    blk._render_bubble_xyxy = list(bubble) if bubble is not None else None
    blk._render_area_source = "text_bbox"
    blk._render_area_xyxy = list(original) if original is not None else None


def _assign_block_xyxy(
    blk: TextBlock,
    value: tuple[int, int, int, int],
) -> None:
    try:
        blk.xyxy[:] = value
    except (AttributeError, TypeError):
        blk.xyxy = list(value)


def _detected_bubble_render_bounds(
    blk: TextBlock,
    img,
    font_profile: AutoMaxFontProfile | None = None,
    *,
    text_xyxy: tuple[int, int, int, int] | None = None,
) -> tuple[int, int, int, int] | None:
    if getattr(blk, "text_class", "") != "text_bubble":
        return None
    if text_xyxy is None:
        text_xyxy = _current_anchor_xyxy(blk)
    bubble_xyxy = _normalize_xyxy(getattr(blk, "bubble_xyxy", None))
    if text_xyxy is None or bubble_xyxy is None:
        return None
    if not _bbox_has_area(text_xyxy) or not _bbox_has_area(bubble_xyxy):
        return None
    if not _text_bbox_belongs_to_bubble(text_xyxy, bubble_xyxy):
        return None

    font_profile = font_profile or get_auto_max_font_profile()
    shrink_percent = (
        font_profile.vertical_shrink_percent
        if getattr(blk, "source_lang_direction", "") == "vertical"
        else font_profile.horizontal_shrink_percent
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


def refit_detected_bubble_text_if_underfilled(
    blk: TextBlock,
    text: str,
    font_input: str,
    roi_width: int | float,
    roi_height: int | float,
    line_spacing: float,
    outline_width: float,
    bold: bool,
    italic: bool,
    underline: bool,
    alignment: Qt.AlignmentFlag,
    direction: Qt.LayoutDirection,
    configured_max_font_size: int | float,
    min_font_size: int | float,
    vertical: bool,
    fit_clearance: float,
    current_wrapped_text: str,
    current_font_size: int | float,
    current_rendered_width: float,
    current_rendered_height: float,
    *,
    auto_max_font_size: bool = True,
    auto_max_font_profile: object = DEFAULT_AUTO_MAX_FONT_PROFILE,
) -> tuple[str, int | float, float, float]:
    if not auto_max_font_size:
        return (
            current_wrapped_text,
            current_font_size,
            current_rendered_width,
            current_rendered_height,
        )

    dynamic_cap = get_dynamic_bubble_font_cap(
        blk,
        configured_max_font_size,
        current_rendered_width,
        current_rendered_height,
        vertical,
        final_font_size=current_font_size,
        auto_max_font_profile=auto_max_font_profile,
    )
    try:
        if dynamic_cap <= int(round(float(configured_max_font_size))):
            return (
                current_wrapped_text,
                current_font_size,
                current_rendered_width,
                current_rendered_height,
            )
    except (TypeError, ValueError):
        return (
            current_wrapped_text,
            current_font_size,
            current_rendered_width,
            current_rendered_height,
        )

    candidate_text, candidate_size, candidate_width, candidate_height = pyside_word_wrap(
        text,
        font_input,
        int(roi_width),
        int(roi_height),
        line_spacing,
        outline_width,
        bold,
        italic,
        underline,
        alignment,
        direction,
        dynamic_cap,
        int(min_font_size),
        vertical,
        fit_clearance=fit_clearance,
        return_metrics=True,
    )
    if (
        candidate_width <= float(roi_width)
        and candidate_height <= float(roi_height)
        and float(candidate_size) >= float(current_font_size)
    ):
        return candidate_text, candidate_size, candidate_width, candidate_height
    return (
        current_wrapped_text,
        current_font_size,
        current_rendered_width,
        current_rendered_height,
    )

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
    try:
        render_settings = main_page.render_settings()
    except Exception:
        render_settings = None
    auto_max_font_profile = getattr(render_settings, "auto_max_font_profile", "current")
    get_best_render_area(
        blk_list,
        getattr(main_page, "image", None),
        auto_max_font_profile=auto_max_font_profile,
    )

    for block_index, blk in enumerate(blk_list):
        x1, y1, width, height = blk.xywh

        translation = sanitize_render_text(
            blk.translation,
            font_family,
            block_index=block_index,
            image_path=image_path,
        )
        if should_skip_short_render_translation(blk, translation):
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
            fit_clearance=get_render_fit_clearance_for_block(
                blk,
                outline_width,
                auto_max_font_profile=auto_max_font_profile,
            ),
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



        

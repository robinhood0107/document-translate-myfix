from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
import math
from pathlib import Path
import random
import struct
from typing import Iterable, Iterator

import cv2
import numpy as np


BACKGROUND_KINDS = ("flat", "gradient", "halftone", "hatching", "paper")
TEXT_STYLES = ("dark", "bright", "outline", "shadow", "embossed")
FONT_TEXT_PHRASES = ("文字", "テスト", "小", "한글", "효과")
_FONT_TEXT_CODEPOINTS = frozenset(
    ord(character)
    for phrase in FONT_TEXT_PHRASES
    for character in phrase
)


@dataclass(frozen=True, slots=True)
class SyntheticTrainingSample:
    image: np.ndarray
    target: np.ndarray
    background: np.ndarray
    sample_id: str
    background_kind: str
    text_style: str
    has_text: bool


def _read_u16(data: bytes, offset: int) -> int:
    if offset < 0 or offset + 2 > len(data):
        raise ValueError("truncated font table")
    return int(struct.unpack_from(">H", data, offset)[0])


def _read_u32(data: bytes, offset: int) -> int:
    if offset < 0 or offset + 4 > len(data):
        raise ValueError("truncated font table")
    return int(struct.unpack_from(">I", data, offset)[0])


def _font_face_offset(data: bytes) -> int:
    if data[:4] != b"ttcf":
        return 0
    if _read_u32(data, 8) < 1:
        raise ValueError("font collection has no faces")
    return _read_u32(data, 12)


def _format4_contains(data: bytes, offset: int, codepoint: int) -> bool:
    length = _read_u16(data, offset + 2)
    limit = offset + length
    if limit > len(data):
        raise ValueError("truncated cmap format 4 table")
    segment_count = _read_u16(data, offset + 6) // 2
    if segment_count < 1:
        return False
    end_codes = offset + 14
    start_codes = end_codes + 2 * segment_count + 2
    deltas = start_codes + 2 * segment_count
    range_offsets = deltas + 2 * segment_count
    if range_offsets + 2 * segment_count > limit:
        raise ValueError("invalid cmap format 4 layout")
    for index in range(segment_count):
        end_code = _read_u16(data, end_codes + 2 * index)
        if codepoint > end_code:
            continue
        start_code = _read_u16(data, start_codes + 2 * index)
        if codepoint < start_code:
            return False
        delta = _read_u16(data, deltas + 2 * index)
        range_word = range_offsets + 2 * index
        range_offset = _read_u16(data, range_word)
        if range_offset == 0:
            return ((codepoint + delta) & 0xFFFF) != 0
        glyph_offset = range_word + range_offset + 2 * (codepoint - start_code)
        if glyph_offset + 2 > limit:
            raise ValueError("invalid cmap format 4 glyph offset")
        glyph = _read_u16(data, glyph_offset)
        if glyph == 0:
            return False
        return ((glyph + delta) & 0xFFFF) != 0
    return False


def _format12_or_13_contains(data: bytes, offset: int, codepoint: int) -> bool:
    cmap_format = _read_u16(data, offset)
    length = _read_u32(data, offset + 4)
    limit = offset + length
    if limit > len(data):
        raise ValueError("truncated cmap format 12/13 table")
    group_count = _read_u32(data, offset + 12)
    groups = offset + 16
    if groups + 12 * group_count > limit:
        raise ValueError("invalid cmap format 12/13 layout")
    for index in range(group_count):
        group = groups + 12 * index
        start_codepoint = _read_u32(data, group)
        end_codepoint = _read_u32(data, group + 4)
        if codepoint < start_codepoint:
            return False
        if codepoint <= end_codepoint:
            glyph = _read_u32(data, group + 8)
            if cmap_format == 12:
                glyph += codepoint - start_codepoint
            return glyph != 0
    return False


@lru_cache(maxsize=64)
def _font_supported_codepoints_cached(
    path_text: str,
    file_size: int,
    modified_ns: int,
) -> frozenset[int]:
    del file_size, modified_ns
    data = Path(path_text).read_bytes()
    face_offset = _font_face_offset(data)
    table_count = _read_u16(data, face_offset + 4)
    cmap_offset: int | None = None
    cmap_length: int | None = None
    for index in range(table_count):
        record = face_offset + 12 + 16 * index
        if record + 16 > len(data):
            raise ValueError("truncated font table directory")
        if data[record : record + 4] == b"cmap":
            cmap_offset = _read_u32(data, record + 8)
            cmap_length = _read_u32(data, record + 12)
            break
    if cmap_offset is None or cmap_length is None:
        raise ValueError("font has no cmap table")
    cmap_limit = cmap_offset + cmap_length
    if cmap_limit > len(data):
        raise ValueError("truncated cmap table")
    encoding_count = _read_u16(data, cmap_offset + 2)
    unicode_subtables: list[int] = []
    for index in range(encoding_count):
        record = cmap_offset + 4 + 8 * index
        if record + 8 > cmap_limit:
            raise ValueError("truncated cmap encoding records")
        platform_id = _read_u16(data, record)
        encoding_id = _read_u16(data, record + 2)
        if platform_id != 0 and not (platform_id == 3 and encoding_id in {1, 10}):
            continue
        subtable = cmap_offset + _read_u32(data, record + 4)
        if subtable + 2 > cmap_limit:
            raise ValueError("invalid cmap subtable offset")
        unicode_subtables.append(subtable)
    supported: set[int] = set()
    for codepoint in _FONT_TEXT_CODEPOINTS:
        for subtable in unicode_subtables:
            cmap_format = _read_u16(data, subtable)
            if cmap_format == 4 and codepoint <= 0xFFFF:
                present = _format4_contains(data, subtable, codepoint)
            elif cmap_format in {12, 13}:
                present = _format12_or_13_contains(data, subtable, codepoint)
            else:
                present = False
            if present:
                supported.add(codepoint)
                break
    return frozenset(supported)


def font_supported_codepoints(font_path: str | Path) -> frozenset[int]:
    path = Path(font_path).resolve()
    stat = path.stat()
    return _font_supported_codepoints_cached(
        str(path),
        int(stat.st_size),
        int(stat.st_mtime_ns),
    )


def supported_font_phrase_pairs(
    font_paths: Iterable[str | Path],
) -> tuple[tuple[Path, str], ...]:
    normalized = tuple(Path(value).resolve() for value in font_paths)
    pairs: list[tuple[Path, str]] = []
    for font_path in normalized:
        if not font_path.is_file():
            raise FileNotFoundError(font_path)
        supported = font_supported_codepoints(font_path)
        for phrase in FONT_TEXT_PHRASES:
            if all(ord(character) in supported for character in phrase):
                pairs.append((font_path, phrase))
    if normalized and not pairs:
        raise ValueError("CJK font assets support none of the synthetic phrases")
    return tuple(pairs)


def _background(
    rng: np.random.Generator,
    shape: tuple[int, int],
    kind: str,
) -> np.ndarray:
    height, width = shape
    base = int(rng.integers(175, 246))
    image = np.full((height, width, 3), base, np.uint8)
    if kind == "gradient":
        x = np.linspace(0.0, 1.0, width, dtype=np.float32)
        y = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
        field = np.clip(base + (x - 0.5) * 30 + (y - 0.5) * 24, 0, 255)
        image[:] = field[:, :, None].astype(np.uint8)
    elif kind == "halftone":
        ink = max(20, base - int(rng.integers(45, 110)))
        pitch = int(rng.integers(6, 13))
        radius = int(rng.integers(1, max(2, pitch // 3)))
        phase_x, phase_y = int(rng.integers(0, pitch)), int(rng.integers(0, pitch))
        for y in range(phase_y, height, pitch):
            for x in range(phase_x, width, pitch):
                cv2.circle(image, (x, y), radius, (ink, ink, ink), -1)
    elif kind == "hatching":
        ink = max(15, base - int(rng.integers(35, 100)))
        pitch = int(rng.integers(7, 15))
        angle = math.radians(float(rng.uniform(-70.0, 70.0)))
        direction = np.array((math.cos(angle), math.sin(angle)), np.float64)
        normal = np.array((-direction[1], direction[0]), np.float64)
        center = np.array((width / 2.0, height / 2.0), np.float64)
        extent = float(math.hypot(width, height) * 1.5)
        for offset in range(-int(extent), int(extent) + 1, pitch):
            start = center + normal * offset - direction * extent
            end = center + normal * offset + direction * extent
            cv2.line(
                image,
                tuple(np.rint(start).astype(int)),
                tuple(np.rint(end).astype(int)),
                (ink, ink, ink),
                int(rng.integers(1, 3)),
            )
    elif kind == "paper":
        noise = rng.normal(0.0, 5.0, size=(height, width, 1)).astype(np.float32)
        image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return image


def _text_mask(
    rng: np.random.Generator,
    shape: tuple[int, int],
    font_phrase_pairs: tuple[tuple[Path, str], ...] = (),
) -> np.ndarray:
    height, width = shape
    mask = np.zeros(shape, np.uint8)
    line_count = int(rng.integers(1, 4))
    if font_phrase_pairs and rng.random() < 0.65:
        return _font_text_mask(rng, shape, font_phrase_pairs)
    vertical = bool(rng.integers(0, 2))
    if vertical:
        x0 = int(rng.integers(width // 4, max(width // 4 + 1, width * 3 // 4)))
        y0 = int(rng.integers(4, max(5, height // 4)))
        step = int(rng.integers(10, 22))
        for line in range(line_count):
            for index in range(int(rng.integers(2, 6))):
                x = x0 + line * step
                y = y0 + index * step
                cv2.rectangle(
                    mask,
                    (x, y),
                    (min(width - 1, x + int(rng.integers(5, 11))), min(height - 1, y + int(rng.integers(8, 17)))),
                    255,
                    int(rng.choice((-1, 2))),
                )
                if rng.random() < 0.7:
                    cv2.line(mask, (x - 2, y + 6), (x + 11, y + 6), 255, 2)
    else:
        text = random.Random(int(rng.integers(0, 2**31 - 1))).choice(
            ("TEXT", "SFX", "!?", "III", "OOO")
        )
        scale = float(rng.uniform(0.45, 1.15))
        thickness = int(rng.integers(1, 4))
        size, _baseline = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness
        )
        x = int(rng.integers(-max(1, size[0] // 4), max(1, width - size[0] + 1)))
        y = int(rng.integers(max(1, size[1]), max(size[1] + 1, height)))
        cv2.putText(
            mask,
            text,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            255,
            thickness,
            cv2.LINE_AA,
        )
    return np.where(mask > 0, 255, 0).astype(np.uint8)


def _font_text_mask(
    rng: np.random.Generator,
    shape: tuple[int, int],
    font_phrase_pairs: tuple[tuple[Path, str], ...],
) -> np.ndarray:
    from PIL import Image, ImageDraw, ImageFont

    height, width = shape
    pair_index = int(rng.integers(0, len(font_phrase_pairs)))
    font_path, phrase = font_phrase_pairs[pair_index]
    font_size = int(rng.integers(max(9, min(shape) // 24), max(16, min(shape) // 7)))
    font = ImageFont.truetype(str(font_path), font_size, index=0)
    canvas = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(canvas)
    vertical = bool(rng.integers(0, 2))
    if vertical:
        x = int(rng.integers(-font_size // 3, max(1, width - font_size)))
        y = int(rng.integers(-font_size // 2, max(1, height // 3)))
        for character in phrase:
            draw.text((x, y), character, fill=255, font=font, stroke_width=0)
            y += max(7, int(round(font_size * 0.9)))
    else:
        bounds = draw.textbbox((0, 0), phrase, font=font, stroke_width=0)
        text_width = max(1, int(bounds[2] - bounds[0]))
        text_height = max(1, int(bounds[3] - bounds[1]))
        x = int(rng.integers(-max(1, text_width // 4), max(1, width - text_width + 1)))
        y = int(rng.integers(-max(1, text_height // 4), max(1, height - text_height + 1)))
        draw.text((x, y), phrase, fill=255, font=font, stroke_width=0)
    return np.where(np.asarray(canvas) > 0, 255, 0).astype(np.uint8, copy=True)


def _paint_text(
    rng: np.random.Generator,
    background: np.ndarray,
    target: np.ndarray,
    style: str,
) -> np.ndarray:
    image = background.copy()
    base = int(np.median(background))
    if style == "dark":
        value = max(0, base - int(rng.integers(45, 150)))
        image[target > 0] = value
    elif style == "bright":
        value = min(255, base + int(rng.integers(10, 55)))
        image[target > 0] = value
    elif style == "outline":
        halo = cv2.dilate(target, np.ones((5, 5), np.uint8))
        image[(halo > 0) & (target == 0)] = max(0, base - 110)
        image[target > 0] = min(255, base + 35)
        target[halo > 0] = 255
    elif style == "shadow":
        shifted = _shift_mask(
            cv2.dilate(target, np.ones((5, 5), np.uint8)), 3, 3
        )
        image[(shifted > 0) & (target == 0)] = max(0, base - 70)
        image[target > 0] = min(255, base + 25)
        target[shifted > 0] = 255
    elif style == "embossed":
        light = _shift_mask(target, -1, -1)
        dark = _shift_mask(target, 1, 1)
        image[light > 0] = min(255, base + int(rng.integers(8, 28)))
        image[dark > 0] = max(0, base - int(rng.integers(8, 28)))
        target[(light > 0) | (dark > 0)] = 255
    return image


def _shift_mask(mask: np.ndarray, delta_x: int, delta_y: int) -> np.ndarray:
    return cv2.warpAffine(
        mask,
        np.array(((1.0, 0.0, float(delta_x)), (0.0, 1.0, float(delta_y)))),
        (mask.shape[1], mask.shape[0]),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def synthetic_training_sample(
    seed: int,
    *,
    shape: tuple[int, int] = (128, 128),
    font_paths: Iterable[str | Path] = (),
) -> SyntheticTrainingSample:
    rng = np.random.default_rng(int(seed))
    background_kind = BACKGROUND_KINDS[int(rng.integers(0, len(BACKGROUND_KINDS)))]
    text_style = TEXT_STYLES[int(rng.integers(0, len(TEXT_STYLES)))]
    background = _background(rng, shape, background_kind)
    has_text = int(seed) % 5 != 0
    normalized_fonts = tuple(Path(value).resolve() for value in font_paths)
    font_phrase_pairs = supported_font_phrase_pairs(normalized_fonts)
    target = (
        _text_mask(rng, shape, font_phrase_pairs)
        if has_text
        else np.zeros(shape, np.uint8)
    )
    image = _paint_text(rng, background, target, text_style)
    identity_state = hashlib.sha256()
    identity_state.update(b"inpaint-synthetic-training-sample-v4\0")
    identity_state.update(str((int(shape[0]), int(shape[1]))).encode("ascii"))
    identity_state.update(b"\0")
    identity_state.update(str(int(seed)).encode("ascii"))
    for value in (background, image, target):
        payload = value.tobytes(order="C")
        identity_state.update(len(payload).to_bytes(8, byteorder="big"))
        identity_state.update(payload)
    identity = identity_state.hexdigest()
    return SyntheticTrainingSample(
        image=image,
        target=target,
        background=background,
        sample_id=identity,
        background_kind=background_kind,
        text_style=text_style,
        has_text=has_text,
    )


def synthetic_training_digest(
    seeds: Iterable[int],
    *,
    shape: tuple[int, int] = (128, 128),
    font_paths: Iterable[str | Path] = (),
) -> str:
    ordered_seeds = tuple(int(seed) for seed in seeds)
    normalized_fonts = tuple(font_paths)
    state = hashlib.sha256()
    state.update(b"inpaint-synthetic-training-dataset-v4\0")
    state.update(len(ordered_seeds).to_bytes(8, byteorder="big"))
    for index, seed in enumerate(ordered_seeds):
        sample = synthetic_training_sample(
            seed,
            shape=shape,
            font_paths=normalized_fonts,
        )
        state.update(index.to_bytes(8, byteorder="big"))
        state.update(str(seed).encode("ascii"))
        state.update(b"\0")
        state.update(bytes.fromhex(sample.sample_id))
    return state.hexdigest()


def synthetic_training_stream(
    seeds: Iterable[int],
    *,
    shape: tuple[int, int] = (128, 128),
    font_paths: Iterable[str | Path] = (),
) -> Iterator[SyntheticTrainingSample]:
    normalized_fonts = tuple(font_paths)
    for seed in seeds:
        yield synthetic_training_sample(seed, shape=shape, font_paths=normalized_fonts)

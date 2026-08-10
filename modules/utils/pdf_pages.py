from __future__ import annotations

import hashlib
import math
import os
import shutil
import tempfile
import threading
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from io import BytesIO
from typing import Callable, Iterable

from PIL import Image

from .exceptions import OperationCancelledError


PDF_PAGE_PLAN_SCHEMA = 1
PDF_CACHE_LIMIT = 4
PDF_FILE_SIZE_LIMIT = 8 * 1024**3
PDF_PAGE_COUNT_LIMIT = 20_000
PDF_OBJECT_COUNT_LIMIT = 100_000
PDF_FORM_DEPTH_LIMIT = 15
PDF_RENDER_PIXEL_LIMIT = 100_000_000
PDF_RENDER_LONG_SIDE_LIMIT = 25_000
PDF_DECODE_BUFFER_LIMIT = 512 * 1024**2
PDF_RENDER_BASE_DPI = 600.0

_RENDER_SEMAPHORE = threading.Semaphore(1)
_CACHE_LOCK = threading.RLock()
_CACHE_CONDITION = threading.Condition(_CACHE_LOCK)
_PDF_CACHE: "OrderedDict[tuple[int, str], _PdfCacheEntry]" = OrderedDict()
_CACHE_GENERATION = 0
_ACTIVE_SCANS = 0
_CACHE_CLOSING = False

_ERROR_CODES = {
    "PDF_PASSWORD_REQUIRED",
    "PDF_SOURCE_CHANGED",
    "PDF_PAGE_PLAN_FAILED",
    "PDF_PAGE_MATERIALIZATION_FAILED",
    "PDF_DISK_SPACE_INSUFFICIENT",
    "PDF_RESOURCE_LIMIT",
}
_DETAIL_CODES = {
    "password_required",
    "source_stat_changed",
    "source_hash_changed",
    "file_size_limit",
    "page_count_limit",
    "page_count_mismatch",
    "invalid_user_unit",
    "object_count_limit",
    "form_depth_limit",
    "page_scan_failed",
    "plan_schema_mismatch",
    "plan_source_mismatch",
    "page_index_invalid",
    "native_validation_failed",
    "lossless_validation_failed",
    "render_failed",
    "output_validation_failed",
    "disk_space_insufficient",
    "publish_failed",
}


class PdfImportError(Exception):
    def __init__(
        self,
        code: str,
        *,
        page_index: int | None = None,
        retryable: bool = False,
        detail_code: str = "page_scan_failed",
    ) -> None:
        safe_code = code if code in _ERROR_CODES else "PDF_PAGE_PLAN_FAILED"
        safe_detail = detail_code if detail_code in _DETAIL_CODES else "page_scan_failed"
        self.code = safe_code
        self.page_index = page_index if isinstance(page_index, int) and page_index >= 0 else None
        self.retryable = bool(retryable)
        self.detail_code = safe_detail
        page = f" page={self.page_index + 1}" if self.page_index is not None else ""
        super().__init__(f"{self.code}{page} detail={self.detail_code}")


@dataclass(frozen=True)
class PdfSourceIdentity:
    size: int
    mtime_ns: int
    ctime_ns: int
    device_id: int
    file_id: int
    sha256: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "PdfSourceIdentity":
        return cls(
            size=int(value.get("size", -1)),
            mtime_ns=int(value.get("mtime_ns", -1)),
            ctime_ns=int(value.get("ctime_ns", -1)),
            device_id=int(value.get("device_id", -1)),
            file_id=int(value.get("file_id", -1)),
            sha256=str(value.get("sha256", "")),
        )


@dataclass(frozen=True)
class PdfPagePlan:
    page_index: int
    strategy: str
    extension: str
    width: int
    height: int
    encoded_sha256: str | None
    canonical_pixel_sha256: str | None
    sample_sha256: str | None
    source_mode: str | None
    encoded_size: int | None
    requested_dpi: float
    applied_dpi: float
    requested_size: tuple[int, int]
    render_size: tuple[int, int]
    cap_applied: bool
    reason: str
    xobject_name: str | None
    source_identity: PdfSourceIdentity
    schema_version: int = PDF_PAGE_PLAN_SCHEMA

    def to_dict(self) -> dict[str, object]:
        value = asdict(self)
        value["source_identity"] = self.source_identity.to_dict()
        value["requested_size"] = list(self.requested_size)
        value["render_size"] = list(self.render_size)
        return value

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "PdfPagePlan":
        requested_size = value.get("requested_size", (0, 0))
        render_size = value.get("render_size", (0, 0))
        source_identity = value.get("source_identity", {})
        if not isinstance(source_identity, dict):
            source_identity = {}
        return cls(
            page_index=int(value.get("page_index", -1)),
            strategy=str(value.get("strategy", "render")),
            extension=str(value.get("extension", ".png")),
            width=int(value.get("width", 0)),
            height=int(value.get("height", 0)),
            encoded_sha256=_optional_text(value.get("encoded_sha256")),
            canonical_pixel_sha256=_optional_text(value.get("canonical_pixel_sha256")),
            sample_sha256=_optional_text(value.get("sample_sha256")),
            source_mode=_optional_text(value.get("source_mode")),
            encoded_size=_optional_int(value.get("encoded_size")),
            requested_dpi=float(value.get("requested_dpi", PDF_RENDER_BASE_DPI)),
            applied_dpi=float(value.get("applied_dpi", PDF_RENDER_BASE_DPI)),
            requested_size=_size_tuple(requested_size),
            render_size=_size_tuple(render_size),
            cap_applied=bool(value.get("cap_applied", False)),
            reason=str(value.get("reason", "render_required")),
            xobject_name=_optional_text(value.get("xobject_name")),
            source_identity=PdfSourceIdentity.from_dict(source_identity),
            schema_version=int(value.get("schema_version", -1)),
        )

    def warning_payload(self) -> dict[str, object] | None:
        if not self.cap_applied:
            return None
        return {
            "code": "PDF_RESOURCE_LIMIT",
            "page_number": self.page_index + 1,
            "requested_width": self.requested_size[0],
            "requested_height": self.requested_size[1],
            "applied_width": self.render_size[0],
            "applied_height": self.render_size[1],
        }


@dataclass
class _PdfCacheEntry:
    path: str
    identity: PdfSourceIdentity
    pike_pdf: object
    pdfium_pdf: object
    plans: tuple[PdfPagePlan, ...]
    lock: threading.RLock


def _optional_text(value: object) -> str | None:
    text = str(value or "")
    return text or None


def _optional_int(value: object) -> int | None:
    return None if value is None else int(value)


def _size_tuple(value: object) -> tuple[int, int]:
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return int(value[0]), int(value[1])
    return 0, 0


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _pixel_sha(image: Image.Image) -> str:
    rgb = image.convert("RGB")
    return _sha256_bytes(rgb.tobytes())


def _sample_sha(image: Image.Image) -> str:
    image.load()
    return _sha256_bytes(image.tobytes())


def _stat_signature(stat: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        int(stat.st_size),
        int(stat.st_mtime_ns),
        int(stat.st_ctime_ns),
        int(getattr(stat, "st_dev", 0)),
        int(getattr(stat, "st_ino", 0)),
    )


def _source_identity(file_path: str) -> PdfSourceIdentity:
    try:
        before = os.stat(file_path)
    except OSError as exc:
        raise PdfImportError(
            "PDF_SOURCE_CHANGED", retryable=True, detail_code="source_stat_changed"
        ) from exc
    if int(before.st_size) > PDF_FILE_SIZE_LIMIT:
        raise PdfImportError("PDF_RESOURCE_LIMIT", detail_code="file_size_limit")

    digest = hashlib.sha256()
    try:
        with open(file_path, "rb") as source:
            while True:
                chunk = source.read(8 * 1024**2)
                if not chunk:
                    break
                digest.update(chunk)
        after = os.stat(file_path)
    except OSError as exc:
        raise PdfImportError(
            "PDF_SOURCE_CHANGED", retryable=True, detail_code="source_stat_changed"
        ) from exc
    if _stat_signature(before) != _stat_signature(after):
        raise PdfImportError(
            "PDF_SOURCE_CHANGED", retryable=True, detail_code="source_stat_changed"
        )
    size, mtime_ns, ctime_ns, device_id, file_id = _stat_signature(after)
    return PdfSourceIdentity(
        size=size,
        mtime_ns=mtime_ns,
        ctime_ns=ctime_ns,
        device_id=device_id,
        file_id=file_id,
        sha256=digest.hexdigest(),
    )


def _check_cancel(should_cancel: Callable[[], bool] | None) -> None:
    if callable(should_cancel) and should_cancel():
        raise OperationCancelledError("PDF import was cancelled.")


def _same_identity(left: PdfSourceIdentity, right: PdfSourceIdentity) -> bool:
    return left == right


def _close_entry(entry: _PdfCacheEntry) -> None:
    with entry.lock:
        for handle in (entry.pdfium_pdf, entry.pike_pdf):
            try:
                handle.close()
            except Exception:
                pass


def close_pdf_cache(file_path: str | None = None) -> None:
    global _CACHE_CLOSING, _CACHE_GENERATION
    entries: list[_PdfCacheEntry] = []
    with _CACHE_CONDITION:
        while _CACHE_CLOSING:
            _CACHE_CONDITION.wait()
        _CACHE_CLOSING = True
        _CACHE_GENERATION += 1
        if file_path is None:
            entries = list(_PDF_CACHE.values())
            _PDF_CACHE.clear()
        else:
            entry = _PDF_CACHE.pop(
                (PDF_PAGE_PLAN_SCHEMA, os.path.abspath(file_path)), None
            )
            if entry is not None:
                entries.append(entry)
        while _ACTIVE_SCANS:
            _CACHE_CONDITION.wait()
    try:
        for entry in entries:
            _close_entry(entry)
    finally:
        with _CACHE_CONDITION:
            _CACHE_CLOSING = False
            _CACHE_CONDITION.notify_all()


def _open_documents(file_path: str, identity: PdfSourceIdentity) -> _PdfCacheEntry:
    import pikepdf
    import pypdfium2 as pdfium

    pike_pdf = None
    pdfium_pdf = None
    try:
        pike_pdf = pikepdf.open(file_path)
    except pikepdf.PasswordError as exc:
        raise PdfImportError(
            "PDF_PASSWORD_REQUIRED", retryable=False, detail_code="password_required"
        ) from exc
    except Exception as exc:
        raise PdfImportError("PDF_PAGE_PLAN_FAILED", detail_code="page_scan_failed") from exc
    try:
        pdfium_pdf = pdfium.PdfDocument(file_path)
        try:
            pdfium_pdf.init_forms()
        except Exception as exc:
            raise PdfImportError(
                "PDF_PAGE_PLAN_FAILED", detail_code="page_scan_failed"
            ) from exc
        pike_count = len(pike_pdf.pages)
        pdfium_count = len(pdfium_pdf)
        if pike_count != pdfium_count:
            raise PdfImportError(
                "PDF_PAGE_PLAN_FAILED", detail_code="page_count_mismatch"
            )
        if pike_count <= 0:
            raise PdfImportError(
                "PDF_PAGE_PLAN_FAILED", detail_code="page_scan_failed"
            )
        if pike_count > PDF_PAGE_COUNT_LIMIT:
            raise PdfImportError("PDF_RESOURCE_LIMIT", detail_code="page_count_limit")
        return _PdfCacheEntry(
            path=file_path,
            identity=identity,
            pike_pdf=pike_pdf,
            pdfium_pdf=pdfium_pdf,
            plans=(),
            lock=threading.RLock(),
        )
    except Exception as exc:
        if pdfium_pdf is not None:
            try:
                pdfium_pdf.close()
            except Exception:
                pass
        try:
            pike_pdf.close()
        except Exception:
            pass
        if isinstance(exc, PdfImportError):
            raise
        raise PdfImportError(
            "PDF_PAGE_PLAN_FAILED", detail_code="page_scan_failed"
        ) from exc


def _inherited(page_obj: object, key: str, default: object = None) -> object:
    current = page_obj
    visited: set[tuple[int, int]] = set()
    while current is not None:
        try:
            value = current.get(key)
        except Exception:
            value = None
        if value is not None:
            return value
        try:
            objgen = tuple(current.objgen)
            if objgen in visited:
                break
            visited.add(objgen)
        except Exception:
            pass
        try:
            current = current.get("/Parent")
        except Exception:
            current = None
    return default


def _number(value: object, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _box(value: object) -> tuple[float, float, float, float] | None:
    try:
        if value is None or len(value) != 4:
            return None
        left, bottom, right, top = (float(item) for item in value)
        if right <= left or top <= bottom:
            return None
        return left, bottom, right, top
    except Exception:
        return None


def _boxes_match(left: tuple[float, ...], right: tuple[float, ...]) -> bool:
    long_side = max(left[2] - left[0], left[3] - left[1], 1.0)
    tolerance = max(0.01, long_side * 0.0001)
    return all(abs(a - b) <= tolerance for a, b in zip(left, right))


def _intersection_area(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> float:
    width = max(0.0, min(left[2], right[2]) - max(left[0], right[0]))
    height = max(0.0, min(left[3], right[3]) - max(left[1], right[1]))
    return width * height


def _page_objects(pdfium_page: object) -> tuple[list[object], list[object]]:
    import pypdfium2.raw as pdfium_c

    objects: list[object] = []
    images: list[object] = []
    for page_object in pdfium_page.get_objects(max_depth=PDF_FORM_DEPTH_LIMIT):
        objects.append(page_object)
        if len(objects) > PDF_OBJECT_COUNT_LIMIT:
            raise PdfImportError("PDF_RESOURCE_LIMIT", detail_code="object_count_limit")
        if page_object.type == pdfium_c.FPDF_PAGEOBJ_IMAGE:
            images.append(page_object)
        if (
            page_object.type == pdfium_c.FPDF_PAGEOBJ_FORM
            and int(getattr(page_object, "level", 0)) >= PDF_FORM_DEPTH_LIMIT - 1
            and pdfium_c.FPDFFormObj_CountObjects(page_object) > 0
        ):
            raise PdfImportError("PDF_RESOURCE_LIMIT", detail_code="form_depth_limit")
    return objects, images


def _render_dimensions(
    page_size: tuple[float, float], user_unit: float, requested_dpi: float
) -> tuple[tuple[int, int], tuple[int, int], float, bool]:
    scale = requested_dpi * user_unit / 72.0
    raw_width = float(page_size[0]) * scale
    raw_height = float(page_size[1]) * scale
    requested_width = max(1, math.ceil(raw_width))
    requested_height = max(1, math.ceil(raw_height))
    decoded_bytes = requested_width * requested_height * 3
    ratios = [1.0]
    if requested_width * requested_height > PDF_RENDER_PIXEL_LIMIT:
        ratios.append(
            math.sqrt(PDF_RENDER_PIXEL_LIMIT / (requested_width * requested_height))
        )
    if requested_width > PDF_RENDER_LONG_SIDE_LIMIT:
        ratios.append(PDF_RENDER_LONG_SIDE_LIMIT / requested_width)
    if requested_height > PDF_RENDER_LONG_SIDE_LIMIT:
        ratios.append(PDF_RENDER_LONG_SIDE_LIMIT / requested_height)
    if decoded_bytes > PDF_DECODE_BUFFER_LIMIT:
        ratios.append(math.sqrt(PDF_DECODE_BUFFER_LIMIT / decoded_bytes))
    ratio = min(ratios)
    applied_width = max(1, math.ceil(raw_width * ratio))
    applied_height = max(1, math.ceil(raw_height * ratio))
    while (
        applied_width * applied_height > PDF_RENDER_PIXEL_LIMIT
        or max(applied_width, applied_height) > PDF_RENDER_LONG_SIDE_LIMIT
        or applied_width * applied_height * 3 > PDF_DECODE_BUFFER_LIMIT
    ):
        ratio *= 0.999999
        applied_width = max(1, math.ceil(raw_width * ratio))
        applied_height = max(1, math.ceil(raw_height * ratio))
    return (
        (requested_width, requested_height),
        (applied_width, applied_height),
        requested_dpi * ratio,
        ratio < 1.0,
    )


def _requested_dpi(
    images: Iterable[object],
    visible_box: tuple[float, float, float, float],
    user_unit: float,
) -> float:
    page_area = (visible_box[2] - visible_box[0]) * (visible_box[3] - visible_box[1])
    candidates = [PDF_RENDER_BASE_DPI]
    for image in images:
        try:
            bounds = tuple(float(value) for value in image.get_bounds())
            if page_area <= 0 or _intersection_area(bounds, visible_box) / page_area < 0.05:
                continue
            px_width, px_height = image.get_px_size()
            a, b, c, d, _e, _f = image.get_matrix().get()
            width_units = math.hypot(float(a), float(b))
            height_units = math.hypot(float(c), float(d))
            if width_units <= 0 or height_units <= 0:
                continue
            dpi_x = int(px_width) * 72.0 / (user_unit * width_units)
            dpi_y = int(px_height) * 72.0 / (user_unit * height_units)
            if math.isfinite(dpi_x) and math.isfinite(dpi_y):
                candidates.append(max(dpi_x, dpi_y))
        except Exception:
            continue
    return max(candidates)


_STATE_ONLY_OPERATORS = {
    "w", "J", "j", "M", "d", "i", "G", "g", "RG", "rg", "K", "k",
    "CS", "cs", "SC", "SCN", "sc", "scn",
}


def _name(value: object) -> str:
    return str(value or "")


def _is_identity_extgstate(resources: object, gs_name: str) -> bool:
    try:
        ext_states = resources.get("/ExtGState")
        state = ext_states.get(gs_name) if ext_states is not None else None
        if state is None:
            return False
        allowed = {"/Type", "/BM", "/CA", "/ca", "/SMask"}
        if any(str(key) not in allowed for key in state.keys()):
            return False
        state_type = state.get("/Type")
        if state_type is not None and _name(state_type) != "/ExtGState":
            return False
        blend_mode = state.get("/BM")
        if blend_mode is not None and _name(blend_mode) != "/Normal":
            return False
        for key in ("/CA", "/ca"):
            alpha = state.get(key)
            if alpha is not None and abs(float(alpha) - 1.0) > 1e-12:
                return False
        soft_mask = state.get("/SMask")
        if soft_mask is not None and _name(soft_mask) != "/None":
            return False
        return True
    except Exception:
        return False


def _content_is_single_image(page: object, selected_name: str) -> bool:
    import pikepdf

    resources = _inherited(page.obj, "/Resources")
    if resources is None:
        return False
    depth = 0
    paints: list[str] = []
    try:
        instructions = pikepdf.parse_content_stream(page)
    except Exception:
        return False
    for instruction in instructions:
        operator = str(instruction.operator)
        operands = instruction.operands
        if operator == "q":
            depth += 1
        elif operator == "Q":
            depth -= 1
            if depth < 0:
                return False
        elif operator == "cm" or operator in _STATE_ONLY_OPERATORS:
            continue
        elif operator == "gs":
            if len(operands) != 1 or not _is_identity_extgstate(resources, str(operands[0])):
                return False
        elif operator == "Do":
            if len(operands) != 1:
                return False
            paints.append(str(operands[0]))
        else:
            return False
    return depth == 0 and paints == [selected_name]


def _filters(stream: object) -> list[str]:
    value = stream.get("/Filter")
    if value is None:
        return []
    try:
        if value._type_name == "array":
            return [str(item) for item in value]
    except Exception:
        pass
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    return [str(value)]


def _standard_colorspace(stream: object) -> str | None:
    value = stream.get("/ColorSpace")
    name = _name(value)
    return name if name in {"/DeviceGray", "/DeviceRGB", "/DeviceCMYK"} else None


def _default_decode(stream: object, components: int) -> bool:
    value = stream.get("/Decode")
    if value is None:
        return True
    try:
        values = [float(item) for item in value]
    except Exception:
        return False
    expected = [item for _ in range(components) for item in (0.0, 1.0)]
    return values == expected


def _image_dictionary_is_safe(stream: object, colorspace: str, bits: int) -> bool:
    try:
        if not _image_dictionary_keys_are_safe(stream):
            return False
        for key in ("/Mask", "/SMask", "/ImageMask"):
            if stream.get(key) is not None:
                return False
        components = {"/DeviceGray": 1, "/DeviceRGB": 3, "/DeviceCMYK": 4}[colorspace]
        if not _default_decode(stream, components):
            return False
        return bits in {1, 8}
    except Exception:
        return False


def _jpx_without_declared_colorspace_is_safe(stream: object, bits: int) -> bool:
    try:
        if not _image_dictionary_keys_are_safe(stream):
            return False
        if stream.get("/ColorSpace") is not None:
            return False
        if any(stream.get(key) is not None for key in ("/Mask", "/SMask", "/ImageMask")):
            return False
        return stream.get("/Decode") is None and bits == 8
    except Exception:
        return False


_SAFE_IMAGE_DICTIONARY_KEYS = {
    "/Type",
    "/Subtype",
    "/Width",
    "/Height",
    "/ColorSpace",
    "/BitsPerComponent",
    "/Filter",
    "/DecodeParms",
    "/Length",
    "/DL",
    "/Decode",
    "/Mask",
    "/SMask",
    "/ImageMask",
    "/Metadata",
    "/StructParent",
    "/Name",
}


def _image_dictionary_keys_are_safe(stream: object) -> bool:
    try:
        if any(str(key) not in _SAFE_IMAGE_DICTIONARY_KEYS for key in stream.keys()):
            return False
        stream_type = stream.get("/Type")
        if stream_type is not None and _name(stream_type) != "/XObject":
            return False
        return _name(stream.get("/Subtype")) == "/Image"
    except Exception:
        return False


def _standalone_image(raw: bytes) -> Image.Image:
    image = Image.open(BytesIO(raw))
    image.load()
    return image


def _pdfium_image_rgb(pdfium_image: object) -> Image.Image:
    bitmap = pdfium_image.get_bitmap(render=False, scale_to_original=True)
    try:
        image = bitmap.to_pil()
        image.load()
        return image.convert("RGB")
    finally:
        try:
            bitmap.close()
        except Exception:
            pass


def _native_candidate(
    stream: object,
    pdfium_image: object,
) -> tuple[str, int, int, str, str, int, str] | None:
    filters = _filters(stream)
    if len(filters) != 1 or filters[0] not in {"/DCTDecode", "/JPXDecode"}:
        return None
    if stream.get("/DecodeParms") is not None:
        return None
    try:
        raw = bytes(stream.read_raw_bytes())
        standalone = _standalone_image(raw)
        width, height = standalone.size
        if (width, height) != tuple(int(value) for value in pdfium_image.get_px_size()):
            return None
        colorspace = _standard_colorspace(stream)
        if filters[0] == "/DCTDecode":
            if not raw.startswith(b"\xff\xd8\xff") or colorspace is None:
                return None
            bits = int(stream.get("/BitsPerComponent", 8))
            expected_mode = {
                "/DeviceGray": "L", "/DeviceRGB": "RGB", "/DeviceCMYK": "CMYK"
            }[colorspace]
            if bits != 8 or standalone.mode != expected_mode:
                return None
            extension = ".jpg"
        else:
            jp2_magic = b"\x00\x00\x00\x0cjP  \r\n\x87\n"
            if raw.startswith(jp2_magic):
                extension = ".jp2"
            elif raw.startswith(b"\xffO\xffQ"):
                extension = ".j2k"
            else:
                return None
            if standalone.mode not in {"L", "RGB", "CMYK"}:
                return None
            bits = int(stream.get("/BitsPerComponent", 8))
            if bits != 8:
                return None
            if colorspace is not None:
                expected_mode = {
                    "/DeviceGray": "L", "/DeviceRGB": "RGB", "/DeviceCMYK": "CMYK"
                }[colorspace]
                if standalone.mode != expected_mode:
                    return None
        standalone_hash = _pixel_sha(standalone)
        pdfium_rgb = _pdfium_image_rgb(pdfium_image)
        if pdfium_rgb.size != standalone.size or _pixel_sha(pdfium_rgb) != standalone_hash:
            return None
        return (
            extension,
            width,
            height,
            _sha256_bytes(raw),
            standalone_hash,
            len(raw),
            standalone.mode,
        )
    except Exception:
        return None


def _lossless_candidate(stream: object) -> tuple[str, int, int, str, str, str] | None:
    import pikepdf

    colorspace = _standard_colorspace(stream)
    bits = int(stream.get("/BitsPerComponent", 0) or 0)
    if colorspace is None or not _image_dictionary_is_safe(stream, colorspace, bits):
        return None
    if not (
        (colorspace == "/DeviceGray" and bits in {1, 8})
        or (colorspace in {"/DeviceRGB", "/DeviceCMYK"} and bits == 8)
    ):
        return None
    try:
        image = pikepdf.PdfImage(stream).as_pil_image()
        image.load()
        allowed_modes = {"1", "L"} if colorspace == "/DeviceGray" else {
            "/DeviceRGB": {"RGB"}, "/DeviceCMYK": {"CMYK"}
        }[colorspace]
        if image.mode not in allowed_modes:
            return None
        extension = ".tiff" if image.mode == "CMYK" else ".png"
        return (
            extension,
            image.width,
            image.height,
            _pixel_sha(image),
            _sample_sha(image),
            image.mode,
        )
    except Exception:
        return None


def _direct_geometry_is_safe(
    pike_pdf: object,
    page: object,
    pdfium_image: object,
    visible_box: tuple[float, float, float, float],
    media_box: tuple[float, float, float, float],
    crop_box: tuple[float, float, float, float],
    user_unit: float,
    rotation: int,
) -> bool:
    if rotation != 0 or abs(user_unit - 1.0) > 1e-12 or not _boxes_match(media_box, crop_box):
        return False
    try:
        if (
            page.obj.get("/Annots") is not None
            or page.obj.get("/Group") is not None
            or pike_pdf.Root.get("/AcroForm") is not None
        ):
            return False
        bounds = tuple(float(value) for value in pdfium_image.get_bounds())
        if not _boxes_match(bounds, visible_box):
            return False
        a, b, c, d, _e, _f = (float(value) for value in pdfium_image.get_matrix().get())
        tolerance = 1e-7
        return a > 0 and d > 0 and abs(b) <= tolerance and abs(c) <= tolerance
    except Exception:
        return False


def _scan_page(entry: _PdfCacheEntry, page_index: int) -> PdfPagePlan:
    page = entry.pike_pdf.pages[page_index]
    pdfium_page = entry.pdfium_pdf[page_index]
    try:
        media_box = _box(_inherited(page.obj, "/MediaBox"))
        crop_box = _box(_inherited(page.obj, "/CropBox", media_box))
        if media_box is None or crop_box is None:
            raise PdfImportError(
                "PDF_PAGE_PLAN_FAILED", page_index=page_index, detail_code="page_scan_failed"
            )
        user_unit = _number(_inherited(page.obj, "/UserUnit", 1), 1.0)
        if not math.isfinite(user_unit) or user_unit <= 0 or user_unit > 75_000:
            raise PdfImportError(
                "PDF_RESOURCE_LIMIT", page_index=page_index, detail_code="invalid_user_unit"
            )
        rotation = int(_number(_inherited(page.obj, "/Rotate", 0), 0)) % 360
        objects, pdfium_images = _page_objects(pdfium_page)
        page_size = tuple(float(value) for value in pdfium_page.get_size())
        requested_dpi = _requested_dpi(pdfium_images, crop_box, user_unit)
        requested_size, render_size, applied_dpi, cap_applied = _render_dimensions(
            page_size, user_unit, requested_dpi
        )

        strategy = "render"
        extension = ".png"
        width, height = render_size
        encoded_sha = None
        pixel_sha = None
        sample_sha = None
        source_mode = None
        encoded_size = None
        xobject_name = None
        reason = "render_required"

        pike_images = list(page.images.items())
        if (
            not cap_applied
            and len(objects) == 1
            and len(pdfium_images) == 1
            and len(pike_images) == 1
        ):
            name, stream = pike_images[0]
            selected_name = str(name)
            pdfium_image = pdfium_images[0]
            try:
                pike_size = (int(stream.get("/Width", 0)), int(stream.get("/Height", 0)))
                pdfium_size = tuple(int(value) for value in pdfium_image.get_px_size())
            except Exception:
                pike_size = (0, 0)
                pdfium_size = (-1, -1)
            if (
                pike_size == pdfium_size
                and pike_size[0] > 0
                and _content_is_single_image(page, selected_name)
                and _direct_geometry_is_safe(
                    entry.pike_pdf,
                    page,
                    pdfium_image,
                    crop_box,
                    media_box,
                    crop_box,
                    user_unit,
                    rotation,
                )
            ):
                colorspace = _standard_colorspace(stream)
                bits = int(stream.get("/BitsPerComponent", 8) or 8)
                dictionary_safe = (
                    colorspace is not None
                    and _image_dictionary_is_safe(stream, colorspace, bits)
                )
                if (
                    not dictionary_safe
                    and _filters(stream) == ["/JPXDecode"]
                    and _jpx_without_declared_colorspace_is_safe(stream, bits)
                ):
                    dictionary_safe = True
                if dictionary_safe:
                    filters = _filters(stream)
                    if filters in (["/DCTDecode"], ["/JPXDecode"]):
                        native = _native_candidate(stream, pdfium_image)
                        if native is not None:
                            (
                                extension, width, height, encoded_sha, pixel_sha,
                                encoded_size, source_mode,
                            ) = native
                            strategy = "native"
                            reason = "safe_native_image"
                            xobject_name = selected_name
                    else:
                        lossless = _lossless_candidate(stream)
                        if lossless is not None:
                            (
                                extension, width, height, pixel_sha, sample_sha, source_mode
                            ) = lossless
                            strategy = "lossless"
                            reason = "safe_lossless_image"
                            xobject_name = selected_name

        if cap_applied:
            strategy = "render"
            extension = ".png"
            width, height = render_size
            encoded_sha = None
            pixel_sha = None
            sample_sha = None
            source_mode = None
            encoded_size = None
            xobject_name = None
            reason = "resource_cap"

        return PdfPagePlan(
            page_index=page_index,
            strategy=strategy,
            extension=extension,
            width=width,
            height=height,
            encoded_sha256=encoded_sha,
            canonical_pixel_sha256=pixel_sha,
            sample_sha256=sample_sha,
            source_mode=source_mode,
            encoded_size=encoded_size,
            requested_dpi=requested_dpi,
            applied_dpi=applied_dpi,
            requested_size=requested_size,
            render_size=render_size,
            cap_applied=cap_applied,
            reason=reason,
            xobject_name=xobject_name,
            source_identity=entry.identity,
        )
    except PdfImportError:
        raise
    except Exception as exc:
        raise PdfImportError(
            "PDF_PAGE_PLAN_FAILED", page_index=page_index, detail_code="page_scan_failed"
        ) from exc
    finally:
        try:
            pdfium_page.close()
        except Exception:
            pass


def scan_pdf(file_path: str) -> tuple[PdfSourceIdentity, list[PdfPagePlan]]:
    global _ACTIVE_SCANS
    abs_path = os.path.abspath(file_path)
    cache_key = (PDF_PAGE_PLAN_SCHEMA, abs_path)
    with _CACHE_CONDITION:
        while _CACHE_CLOSING:
            _CACHE_CONDITION.wait()
        scan_generation = _CACHE_GENERATION
        _ACTIVE_SCANS += 1

    stale_entry = None
    entry = None
    try:
        identity = _source_identity(abs_path)
        with _CACHE_CONDITION:
            cached = _PDF_CACHE.get(cache_key)
            if cached is not None and _same_identity(cached.identity, identity):
                _PDF_CACHE.move_to_end(cache_key)
                return cached.identity, list(cached.plans)
            if cached is not None:
                stale_entry = _PDF_CACHE.pop(cache_key)
        if stale_entry is not None:
            _close_entry(stale_entry)
        entry = _open_documents(abs_path, identity)
        with entry.lock:
            entry.plans = tuple(
                _scan_page(entry, page_index)
                for page_index in range(len(entry.pike_pdf.pages))
            )
        plans = list(entry.plans)

        evicted: list[_PdfCacheEntry] = []
        with _CACHE_CONDITION:
            cache_entry = scan_generation == _CACHE_GENERATION
            if cache_entry:
                replaced = _PDF_CACHE.pop(cache_key, None)
                if replaced is not None:
                    evicted.append(replaced)
                _PDF_CACHE[cache_key] = entry
                while len(_PDF_CACHE) > PDF_CACHE_LIMIT:
                    _path, old_entry = _PDF_CACHE.popitem(last=False)
                    evicted.append(old_entry)
        for old_entry in evicted:
            _close_entry(old_entry)
        if not cache_entry:
            _close_entry(entry)
        return identity, plans
    except Exception:
        if entry is not None:
            _close_entry(entry)
        raise
    finally:
        with _CACHE_CONDITION:
            _ACTIVE_SCANS -= 1
            _CACHE_CONDITION.notify_all()


@contextmanager
def _entry_for_plan(file_path: str, plan: PdfPagePlan):
    if plan.schema_version != PDF_PAGE_PLAN_SCHEMA:
        raise PdfImportError(
            "PDF_SOURCE_CHANGED", page_index=plan.page_index,
            retryable=True, detail_code="plan_schema_mismatch"
        )
    identity, _plans = scan_pdf(file_path)
    if not _same_identity(identity, plan.source_identity):
        raise PdfImportError(
            "PDF_SOURCE_CHANGED", page_index=plan.page_index,
            retryable=True, detail_code="plan_source_mismatch"
        )
    cache_key = (PDF_PAGE_PLAN_SCHEMA, os.path.abspath(file_path))
    with _CACHE_LOCK:
        entry = _PDF_CACHE.get(cache_key)
        if entry is None:
            raise PdfImportError(
                "PDF_SOURCE_CHANGED", page_index=plan.page_index,
                retryable=True, detail_code="plan_source_mismatch"
            )
        if (
            plan.page_index < 0
            or plan.page_index >= len(entry.plans)
            or entry.plans[plan.page_index] != plan
        ):
            raise PdfImportError(
                "PDF_SOURCE_CHANGED", page_index=plan.page_index,
                retryable=True, detail_code="plan_source_mismatch"
            )
        _PDF_CACHE.move_to_end(cache_key)
        entry.lock.acquire()
    try:
        yield entry
    finally:
        entry.lock.release()


def _plan_stream(entry: _PdfCacheEntry, plan: PdfPagePlan) -> object:
    if plan.page_index < 0 or plan.page_index >= len(entry.pike_pdf.pages):
        raise PdfImportError(
            "PDF_PAGE_MATERIALIZATION_FAILED", page_index=plan.page_index,
            detail_code="page_index_invalid"
        )
    page = entry.pike_pdf.pages[plan.page_index]
    for name, stream in page.images.items():
        if str(name) == plan.xobject_name:
            return stream
    raise PdfImportError(
        "PDF_PAGE_MATERIALIZATION_FAILED", page_index=plan.page_index,
        detail_code="output_validation_failed"
    )


def _write_native(
    entry: _PdfCacheEntry,
    plan: PdfPagePlan,
    output_path: str,
    should_cancel: Callable[[], bool] | None = None,
) -> None:
    try:
        _check_cancel(should_cancel)
        raw = bytes(_plan_stream(entry, plan).read_raw_bytes())
        _check_cancel(should_cancel)
        if _sha256_bytes(raw) != plan.encoded_sha256:
            raise ValueError("encoded mismatch")
        image = _standalone_image(raw)
        _check_cancel(should_cancel)
        if image.size != (plan.width, plan.height) or _pixel_sha(image) != plan.canonical_pixel_sha256:
            raise ValueError("pixel mismatch")
        _check_cancel(should_cancel)
        with open(output_path, "wb") as output:
            output.write(raw)
        _check_cancel(should_cancel)
    except (PdfImportError, OperationCancelledError):
        raise
    except Exception as exc:
        raise PdfImportError(
            "PDF_PAGE_MATERIALIZATION_FAILED", page_index=plan.page_index,
            detail_code="native_validation_failed"
        ) from exc


def _write_lossless(
    entry: _PdfCacheEntry,
    plan: PdfPagePlan,
    output_path: str,
    should_cancel: Callable[[], bool] | None = None,
) -> None:
    import pikepdf

    try:
        _check_cancel(should_cancel)
        source = pikepdf.PdfImage(_plan_stream(entry, plan)).as_pil_image()
        source.load()
        _check_cancel(should_cancel)
        if (
            source.mode != plan.source_mode
            or source.size != (plan.width, plan.height)
            or _sample_sha(source) != plan.sample_sha256
            or _pixel_sha(source) != plan.canonical_pixel_sha256
        ):
            raise ValueError("source mismatch")
        _check_cancel(should_cancel)
        if plan.extension == ".tiff":
            source.save(output_path, format="TIFF", compression="tiff_adobe_deflate")
        else:
            source.save(output_path, format="PNG", compress_level=6)
        _check_cancel(should_cancel)
        with Image.open(output_path) as saved:
            saved.load()
            _check_cancel(should_cancel)
            if (
                saved.mode != plan.source_mode
                or saved.size != source.size
                or _sample_sha(saved) != plan.sample_sha256
                or _pixel_sha(saved) != plan.canonical_pixel_sha256
            ):
                raise ValueError("roundtrip mismatch")
    except (PdfImportError, OperationCancelledError):
        raise
    except Exception as exc:
        raise PdfImportError(
            "PDF_PAGE_MATERIALIZATION_FAILED", page_index=plan.page_index,
            detail_code="lossless_validation_failed"
        ) from exc


def _write_render(
    entry: _PdfCacheEntry,
    plan: PdfPagePlan,
    output_path: str,
    should_cancel: Callable[[], bool] | None = None,
) -> None:
    try:
        _check_cancel(should_cancel)
        with _RENDER_SEMAPHORE:
            _check_cancel(should_cancel)
            pdfium_page = entry.pdfium_pdf[plan.page_index]
            try:
                user_unit = plan.source_identity and _number(
                    _inherited(entry.pike_pdf.pages[plan.page_index].obj, "/UserUnit", 1), 1.0
                )
                scale = plan.applied_dpi * float(user_unit) / 72.0
                bitmap = pdfium_page.render(
                    scale=scale,
                    optimize_mode="print",
                    draw_annots=True,
                    fill_color=(255, 255, 255, 255),
                    limit_image_cache=True,
                    rev_byteorder=True,
                )
                try:
                    _check_cancel(should_cancel)
                    image = bitmap.to_pil().convert("RGB")
                    image.load()
                    _check_cancel(should_cancel)
                finally:
                    try:
                        bitmap.close()
                    except Exception:
                        pass
                if image.size != plan.render_size:
                    raise ValueError("render size mismatch")
                _check_cancel(should_cancel)
                image.save(output_path, format="PNG", compress_level=6)
                _check_cancel(should_cancel)
            finally:
                try:
                    pdfium_page.close()
                except Exception:
                    pass
    except OperationCancelledError:
        raise
    except Exception as exc:
        raise PdfImportError(
            "PDF_PAGE_MATERIALIZATION_FAILED", page_index=plan.page_index,
            detail_code="render_failed"
        ) from exc


def validate_materialized_page(plan: PdfPagePlan, output_path: str) -> bool:
    try:
        if not os.path.isfile(output_path) or os.path.getsize(output_path) <= 0:
            return False
        if plan.strategy == "native":
            with open(output_path, "rb") as source:
                raw = source.read()
            if _sha256_bytes(raw) != plan.encoded_sha256:
                return False
            image = _standalone_image(raw)
            return (
                image.size == (plan.width, plan.height)
                and _pixel_sha(image) == plan.canonical_pixel_sha256
            )
        with Image.open(output_path) as image:
            image.load()
            if image.size != (plan.width, plan.height):
                return False
            if plan.strategy == "lossless":
                return (
                    image.mode == plan.source_mode
                    and _sample_sha(image) == plan.sample_sha256
                    and _pixel_sha(image) == plan.canonical_pixel_sha256
                )
            return image.mode == "RGB" and image.size == plan.render_size
    except Exception:
        return False


def _materialize_exact(
    entry: _PdfCacheEntry,
    plan: PdfPagePlan,
    output_path: str,
    should_cancel: Callable[[], bool] | None = None,
) -> None:
    _check_cancel(should_cancel)
    if plan.strategy == "native":
        _write_native(entry, plan, output_path, should_cancel)
    elif plan.strategy == "lossless":
        _write_lossless(entry, plan, output_path, should_cancel)
    elif plan.strategy == "render":
        _write_render(entry, plan, output_path, should_cancel)
    else:
        raise PdfImportError(
            "PDF_PAGE_MATERIALIZATION_FAILED", page_index=plan.page_index,
            detail_code="output_validation_failed"
        )
    _check_cancel(should_cancel)
    if not validate_materialized_page(plan, output_path):
        raise PdfImportError(
            "PDF_PAGE_MATERIALIZATION_FAILED", page_index=plan.page_index,
            detail_code="output_validation_failed"
        )


def materialize_page(file_path: str, plan: PdfPagePlan, output_path: str) -> None:
    output_path = os.path.abspath(output_path)
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    fd, temp_path = tempfile.mkstemp(prefix=".pdf_page_", suffix=plan.extension, dir=output_dir)
    os.close(fd)
    try:
        with _entry_for_plan(os.path.abspath(file_path), plan) as entry:
            _materialize_exact(entry, plan, temp_path)
        os.replace(temp_path, output_path)
    except PdfImportError:
        raise
    except Exception as exc:
        raise PdfImportError(
            "PDF_PAGE_MATERIALIZATION_FAILED", page_index=plan.page_index,
            detail_code="publish_failed"
        ) from exc
    finally:
        try:
            if os.path.exists(temp_path):
                os.remove(temp_path)
        except OSError:
            pass


def _estimated_transaction_bytes(items: list[tuple[PdfPagePlan, str]]) -> int:
    estimate = 0
    for plan, _output_path in items:
        if plan.strategy == "native" and plan.encoded_size is not None:
            estimate += max(1, plan.encoded_size)
        else:
            estimate += max(1, plan.width * plan.height * (4 if plan.source_mode == "CMYK" else 3))
    return estimate + max(1024**3, math.ceil(estimate * 0.10))


def materialize_transaction(
    file_path: str,
    items: list[tuple[PdfPagePlan, str]],
    should_cancel: Callable[[], bool] | None = None,
) -> None:
    if not items:
        return
    abs_path = os.path.abspath(file_path)
    _check_cancel(should_cancel)
    parent = os.path.dirname(os.path.abspath(items[0][1]))
    os.makedirs(parent, exist_ok=True)
    required = _estimated_transaction_bytes(items)
    try:
        free = shutil.disk_usage(parent).free
    except OSError as exc:
        raise PdfImportError(
            "PDF_DISK_SPACE_INSUFFICIENT", retryable=True,
            detail_code="disk_space_insufficient"
        ) from exc
    if free < required:
        raise PdfImportError(
            "PDF_DISK_SPACE_INSUFFICIENT", retryable=True,
            detail_code="disk_space_insufficient"
        )

    staging_dir = tempfile.mkdtemp(prefix=".pdf_staging_", dir=parent)
    staged: list[tuple[PdfPagePlan, str, str]] = []
    published: list[str] = []
    try:
        with _entry_for_plan(abs_path, items[0][0]) as entry:
            for plan, _output_path in items:
                if (
                    not _same_identity(plan.source_identity, entry.identity)
                    or plan.page_index < 0
                    or plan.page_index >= len(entry.plans)
                    or entry.plans[plan.page_index] != plan
                ):
                    raise PdfImportError(
                        "PDF_SOURCE_CHANGED", page_index=plan.page_index,
                        retryable=True, detail_code="plan_source_mismatch"
                    )
            for sequence, (plan, destination) in enumerate(items):
                _check_cancel(should_cancel)
                stage_path = os.path.join(
                    staging_dir, f"{sequence + 1:06d}{plan.extension}"
                )
                _materialize_exact(entry, plan, stage_path, should_cancel)
                staged.append((plan, stage_path, os.path.abspath(destination)))
            if not all(validate_materialized_page(plan, stage) for plan, stage, _ in staged):
                raise PdfImportError(
                    "PDF_PAGE_MATERIALIZATION_FAILED", detail_code="output_validation_failed"
                )
            for plan, stage_path, destination in staged:
                _check_cancel(should_cancel)
                if os.path.exists(destination):
                    if validate_materialized_page(plan, destination):
                        continue
                    raise PdfImportError(
                        "PDF_PAGE_MATERIALIZATION_FAILED", page_index=plan.page_index,
                        detail_code="output_validation_failed"
                    )
                os.makedirs(os.path.dirname(destination), exist_ok=True)
                os.replace(stage_path, destination)
                published.append(destination)
    except (PdfImportError, OperationCancelledError):
        for destination in published:
            try:
                os.remove(destination)
            except OSError:
                pass
        raise
    except Exception as exc:
        for destination in published:
            try:
                os.remove(destination)
            except OSError:
                pass
        raise PdfImportError(
            "PDF_PAGE_MATERIALIZATION_FAILED", detail_code="publish_failed"
        ) from exc
    finally:
        shutil.rmtree(staging_dir, ignore_errors=True)

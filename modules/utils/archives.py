import io
import math
import os
import re
import shutil
import tarfile
import tempfile
import threading
import zipfile

from .pdf_pages import (
    PdfPagePlan,
    close_pdf_cache,
    materialize_page,
    materialize_transaction,
    scan_pdf,
)
from .image_safety import inspect_image_stream_dimensions

SUPPORTED_SAVE_AS_EXTS = {'.pdf', '.cbz', '.cb7', '.zip'}
_IMAGE_EXTENSIONS = (
    '.jpg',
    '.jpeg',
    '.png',
    '.bmp',
    '.webp',
    '.jp2',
    '.j2k',
    '.jpf',
    '.jpx',
    '.j2c',
    '.tif',
    '.tiff',
)
_COMIC_CACHE_LOCK = threading.RLock()
_COMIC_CACHE: dict[str, dict] = {}


def close_comic_cache(file_path: str | None = None) -> None:
    with _COMIC_CACHE_LOCK:
        if file_path is not None:
            abs_path = os.path.abspath(file_path)
            _COMIC_CACHE.pop(abs_path, None)
        else:
            _COMIC_CACHE.clear()

def resolve_save_as_ext(input_archive_ext: str, save_as: str | None = None) -> str:
    """Resolve the output archive extension for auto-saved translated archives.

    Returns a dotted extension (e.g. '.zip') accepted by `make()`.
    `input_archive_ext` is ignored except for backward-compatible callers.
    """
    def _normalize_target(value: str | None) -> str | None:
        if not value:
            return None
        v = str(value).strip().lower()
        if not v:
            return None
        return v if v.startswith('.') else f'.{v}'

    forced = _normalize_target(save_as)
    if forced in SUPPORTED_SAVE_AS_EXTS:
        return forced

    # Default: zip
    return '.zip'

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(r'(\d+)', str(s))]

def is_image_file(filename):
    return filename.lower().endswith(_IMAGE_EXTENSIONS)


def _safe_ext(path: str, default: str = ".png") -> str:
    ext = os.path.splitext(os.path.basename(path))[1].lower()
    if ext in _IMAGE_EXTENSIONS:
        return ext
    return default


def _is_cbz_native_archive(file_lower: str) -> bool:
    return file_lower.endswith((".cbr",))


def _load_comic_archive(file_path: str):
    from cbz import ComicInfo

    file_lower = file_path.lower()
    if file_lower.endswith(".cbz"):
        return ComicInfo.from_cbz(file_path)
    if file_lower.endswith(".cbr"):
        return ComicInfo.from_cbr(file_path)
    raise ValueError("Unsupported cbz-native comic format")


def _get_cached_comic(file_path: str):
    abs_path = os.path.abspath(file_path)
    stat = os.stat(abs_path)
    size = int(stat.st_size)
    mtime_ns = int(stat.st_mtime_ns)

    with _COMIC_CACHE_LOCK:
        cached = _COMIC_CACHE.get(abs_path)
        if cached and cached.get("size") == size and cached.get("mtime_ns") == mtime_ns:
            return cached["comic"]

        comic = _load_comic_archive(abs_path)
        _COMIC_CACHE[abs_path] = {
            "comic": comic,
            "size": size,
            "mtime_ns": mtime_ns,
        }
        return comic


def _comic_entry_name(page_index: int, page_name: str, ext: str) -> str:
    safe_name = os.path.basename(page_name or "").strip()
    if not safe_name:
        safe_name = f"page{page_index + 1:04d}{ext}"
    return f"{page_index + 1:06d}_{safe_name}"


def _is_safe_archive_member_name(name: str) -> bool:
    normalized = str(name or "").replace("\\", "/")
    if (
        not normalized
        or normalized.startswith("/")
        or re.match(r"^[A-Za-z]:", normalized)
        or any(ord(character) < 32 for character in normalized)
    ):
        return False
    parts = normalized.split("/")
    return all(part not in {"", ".", ".."} for part in parts)


def _archive_member_name(entry: dict) -> str:
    candidate = str(
        entry.get("archive_member_name") or entry.get("entry_name") or ""
    )
    return candidate if _is_safe_archive_member_name(candidate) else ""


def _list_cbz_zip_entries(file_path: str) -> list[dict]:
    with zipfile.ZipFile(file_path, 'r') as archive:
        names = [name for name in archive.namelist() if is_image_file(name)]

    entries: list[dict] = []
    for page_index, name in enumerate(sorted(names, key=natural_sort_key)):
        ext = _safe_ext(name)
        entries.append({
            "kind": "archive_entry",
            "entry_name": _comic_entry_name(page_index, name, ext),
            "archive_member_name": name,
            "ext": ext,
            "page_index": page_index,
        })
    return entries


def _list_cbz_native_entries(file_path: str) -> list[dict]:
    comic = _get_cached_comic(file_path)
    entries: list[dict] = []

    for page_index, page in enumerate(comic):
        page_name = str(getattr(page, "name", "") or "")
        page_suffix = str(getattr(page, "suffix", "") or "")
        ext = _safe_ext(page_suffix or page_name)
        entries.append({
            "kind": "archive_entry",
            "entry_name": _comic_entry_name(page_index, page_name, ext),
            "ext": ext,
            "page_index": page_index,
        })

    return entries


def _materialize_cbz_native_entry(file_path: str, entry: dict, output_path: str) -> bool:
    page_index = int(entry.get("page_index", -1))
    if page_index < 0:
        return False

    comic = _get_cached_comic(file_path)
    if page_index >= len(comic):
        return False

    content = getattr(comic[page_index], "content", None)
    if not isinstance(content, (bytes, bytearray)):
        return False

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(output_path, "wb") as dst:
        dst.write(bytes(content))
    return True


def list_archive_image_entries(file_path: str) -> list[dict]:
    file_lower = file_path.lower()
    entries: list[dict] = []

    if file_lower.endswith('.cbz'):
        entries = _list_cbz_zip_entries(file_path)

    elif _is_cbz_native_archive(file_lower):
        entries = _list_cbz_native_entries(file_path)

    elif file_lower.endswith(('.zip', '.epub')):
        with zipfile.ZipFile(file_path, 'r') as archive:
            for name in archive.namelist():
                if is_image_file(name):
                    entries.append({
                        "kind": "archive_entry",
                        "entry_name": name,
                        "ext": _safe_ext(name),
                    })

    elif file_lower.endswith(('.rar',)):
        import rarfile
        with rarfile.RarFile(file_path, 'r') as archive:
            for name in archive.namelist():
                if is_image_file(name):
                    entries.append({
                        "kind": "archive_entry",
                        "entry_name": name,
                        "ext": _safe_ext(name),
                    })

    elif file_lower.endswith(('.cbt', '.tar')):
        with tarfile.open(file_path, 'r') as archive:
            for member in archive:
                if member.isfile() and is_image_file(member.name):
                    entries.append({
                        "kind": "archive_entry",
                        "entry_name": member.name,
                        "ext": _safe_ext(member.name),
                    })

    elif file_lower.endswith(('.cb7', '.7z')):
        import py7zr
        with py7zr.SevenZipFile(file_path, 'r') as archive:
            for name in archive.getnames():
                if is_image_file(name):
                    entries.append({
                        "kind": "archive_entry",
                        "entry_name": name,
                        "ext": _safe_ext(name),
                    })

    elif file_lower.endswith('.pdf'):
        _source_identity, plans = scan_pdf(file_path)
        for plan in plans:
            entries.append({
                "kind": "pdf_page",
                "page_index": plan.page_index,
                "ext": plan.extension,
                "pdf_plan": plan.to_dict(),
            })

    else:
        raise ValueError("Unsupported file format")

    if entries and entries[0]["kind"] == "pdf_page":
        return entries
    safe_entries = [entry for entry in entries if _archive_member_name(entry)]
    return sorted(
        safe_entries,
        key=lambda entry: natural_sort_key(entry.get("entry_name", "")),
    )


def materialize_archive_entry(file_path: str, entry: dict, output_path: str) -> bool:
    kind = str(entry.get("kind", ""))
    if kind == "pdf_page":
        materialize_page(file_path, _pdf_plan_from_entry(file_path, entry), output_path)
        return True
    if kind != "archive_entry":
        return False

    entry_name = _archive_member_name(entry)
    if not entry_name:
        return False

    file_lower = file_path.lower()
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    if file_lower.endswith(('.zip', '.epub', '.cbz')):
        with zipfile.ZipFile(file_path, 'r') as archive:
            with archive.open(entry_name) as src, open(output_path, "wb") as dst:
                shutil.copyfileobj(src, dst)
        return True

    if _is_cbz_native_archive(file_lower):
        return _materialize_cbz_native_entry(file_path, entry, output_path)

    if file_lower.endswith(('.rar',)):
        import rarfile
        with rarfile.RarFile(file_path, 'r') as archive:
            with archive.open(entry_name) as src, open(output_path, "wb") as dst:
                shutil.copyfileobj(src, dst)
        return True

    if file_lower.endswith(('.cbt', '.tar')):
        with tarfile.open(file_path, 'r') as archive:
            member = archive.getmember(entry_name)
            src = archive.extractfile(member)
            if src is None:
                return False
            with src, open(output_path, "wb") as dst:
                shutil.copyfileobj(src, dst)
        return True

    if file_lower.endswith(('.cb7', '.7z')):
        import py7zr
        with tempfile.TemporaryDirectory(prefix="ct_7z_extract_") as temp_dir:
            with py7zr.SevenZipFile(file_path, 'r') as archive:
                archive.extract(targets=[entry_name], path=temp_dir)
            extracted = os.path.join(temp_dir, *entry_name.replace("\\", "/").split("/"))
            if not os.path.isfile(extracted):
                return False
            shutil.copyfile(extracted, output_path)
        return True

    return False


def inspect_archive_entry_dimensions(file_path: str, entry: dict) -> tuple[int, int]:
    """Read one archive page header without creating its lazy output file."""

    kind = str(entry.get("kind", ""))
    if kind == "pdf_page":
        plan = _pdf_plan_from_entry(file_path, entry)
        return int(plan.width), int(plan.height)
    if kind != "archive_entry":
        raise ValueError("Unsupported archive image entry")

    entry_name = _archive_member_name(entry)
    if not entry_name:
        raise ValueError("Unsafe archive image entry")
    source_label = f"{os.path.basename(file_path)}::{entry_name}"
    file_lower = file_path.lower()

    if file_lower.endswith((".zip", ".epub", ".cbz")):
        with zipfile.ZipFile(file_path, "r") as archive:
            with archive.open(entry_name) as stream:
                return inspect_image_stream_dimensions(
                    stream,
                    source_label=source_label,
                )

    if _is_cbz_native_archive(file_lower):
        page_index = int(entry.get("page_index", -1))
        comic = _get_cached_comic(file_path)
        if page_index < 0 or page_index >= len(comic):
            raise ValueError("Archive page index is out of range")
        content = getattr(comic[page_index], "content", None)
        if not isinstance(content, (bytes, bytearray)):
            raise ValueError("Archive page content is unavailable")
        with io.BytesIO(bytes(content)) as stream:
            return inspect_image_stream_dimensions(
                stream,
                source_label=source_label,
            )

    if file_lower.endswith(".rar"):
        import rarfile

        with rarfile.RarFile(file_path, "r") as archive:
            with archive.open(entry_name) as stream:
                return inspect_image_stream_dimensions(
                    stream,
                    source_label=source_label,
                )

    if file_lower.endswith((".cbt", ".tar")):
        with tarfile.open(file_path, "r") as archive:
            member = archive.getmember(entry_name)
            stream = archive.extractfile(member)
            if stream is None:
                raise ValueError("Archive page content is unavailable")
            with stream:
                return inspect_image_stream_dimensions(
                    stream,
                    source_label=source_label,
                )

    if file_lower.endswith((".cb7", ".7z")):
        import py7zr

        # py7zr exposes an in-memory writer factory rather than a member
        # stream. The bounded buffer avoids creating the lazy output on disk.
        factory = py7zr.io.BytesIOFactory(limit=32 * 1024 * 1024)
        with py7zr.SevenZipFile(file_path, "r") as archive:
            archive.extract(targets=[entry_name], factory=factory)
        stream = factory.get(entry_name)
        stream.seek(0)
        with io.BytesIO(stream.read()) as buffered_stream:
            return inspect_image_stream_dimensions(
                buffered_stream,
                source_label=source_label,
            )

    raise ValueError("Unsupported archive format")


def materialize_archive_entries(file_path: str, items: list[tuple[dict, str]]) -> int:
    if not items:
        return 0

    file_lower = file_path.lower()
    completed = 0

    if file_lower.endswith('.pdf'):
        pdf_items = [
            (_pdf_plan_from_entry(file_path, entry), output_path)
            for entry, output_path in items
        ]
        materialize_transaction(file_path, pdf_items)
        return len(pdf_items)

    if file_lower.endswith(('.zip', '.epub', '.cbz')):
        with zipfile.ZipFile(file_path, 'r') as archive:
            for entry, output_path in items:
                entry_name = _archive_member_name(entry)
                if not entry_name:
                    continue
                out_dir = os.path.dirname(output_path)
                if out_dir:
                    os.makedirs(out_dir, exist_ok=True)
                try:
                    with archive.open(entry_name) as src, open(output_path, "wb") as dst:
                        shutil.copyfileobj(src, dst)
                    completed += 1
                except Exception:
                    continue
        return completed

    if _is_cbz_native_archive(file_lower):
        for entry, output_path in items:
            if _materialize_cbz_native_entry(file_path, entry, output_path):
                completed += 1
        return completed

    if file_lower.endswith(('.rar',)):
        import rarfile
        with rarfile.RarFile(file_path, 'r') as archive:
            for entry, output_path in items:
                entry_name = _archive_member_name(entry)
                if not entry_name:
                    continue
                out_dir = os.path.dirname(output_path)
                if out_dir:
                    os.makedirs(out_dir, exist_ok=True)
                try:
                    with archive.open(entry_name) as src, open(output_path, "wb") as dst:
                        shutil.copyfileobj(src, dst)
                    completed += 1
                except Exception:
                    continue
        return completed

    if file_lower.endswith(('.cbt', '.tar')):
        with tarfile.open(file_path, 'r') as archive:
            for entry, output_path in items:
                entry_name = _archive_member_name(entry)
                if not entry_name:
                    continue
                out_dir = os.path.dirname(output_path)
                if out_dir:
                    os.makedirs(out_dir, exist_ok=True)
                try:
                    member = archive.getmember(entry_name)
                    src = archive.extractfile(member)
                    if src is None:
                        continue
                    with src, open(output_path, "wb") as dst:
                        shutil.copyfileobj(src, dst)
                    completed += 1
                except Exception:
                    continue
        return completed

    if file_lower.endswith(('.cb7', '.7z')):
        import py7zr
        targets = []
        name_to_output: dict[str, str] = {}
        for entry, output_path in items:
            entry_name = _archive_member_name(entry)
            if not entry_name:
                continue
            targets.append(entry_name)
            name_to_output[entry_name] = output_path
        if not targets:
            return 0
        with tempfile.TemporaryDirectory(prefix="ct_7z_extract_") as temp_dir:
            with py7zr.SevenZipFile(file_path, 'r') as archive:
                archive.extract(targets=targets, path=temp_dir)
            for entry_name, output_path in name_to_output.items():
                extracted = os.path.join(temp_dir, *entry_name.replace("\\", "/").split("/"))
                if not os.path.isfile(extracted):
                    continue
                out_dir = os.path.dirname(output_path)
                if out_dir:
                    os.makedirs(out_dir, exist_ok=True)
                try:
                    shutil.copyfile(extracted, output_path)
                    completed += 1
                except Exception:
                    continue
        return completed

    for entry, output_path in items:
        if materialize_archive_entry(file_path, entry, output_path):
            completed += 1
    return completed


def _pdf_plan_from_entry(file_path: str, entry: dict) -> PdfPagePlan:
    value = entry.get("pdf_plan")
    if isinstance(value, dict):
        return PdfPagePlan.from_dict(value)
    page_index = int(entry.get("page_index", -1))
    _identity, plans = scan_pdf(file_path)
    if page_index < 0 or page_index >= len(plans):
        from .pdf_pages import PdfImportError

        raise PdfImportError(
            "PDF_PAGE_MATERIALIZATION_FAILED",
            page_index=page_index,
            detail_code="page_index_invalid",
        )
    return plans[page_index]

def extract_archive(file_path: str, extract_to: str):
    image_paths = []
    entries = list_archive_image_entries(file_path)
    total = len(entries)
    digits = math.floor(math.log10(total)) + 1 if total > 0 else 1

    for index, entry in enumerate(entries, start=1):
        ext = str(entry.get("ext", ".png"))
        if not ext.startswith("."):
            ext = f".{ext}"
        image_path = os.path.join(extract_to, f"{index:0{digits}d}{ext}")
        if materialize_archive_entry(file_path, entry, image_path):
            image_paths.append(image_path)

    return image_paths

def make_cbz(input_dir, output_path='', output_dir='', output_base_name='', save_as_ext='.cbz', compresslevel=None):
    if not output_path:
        output_path = os.path.join(output_dir, f"{output_base_name}_translated{save_as_ext}")
    if os.path.exists(output_path):
        raise FileExistsError(f"Output archive already exists: {output_path}")

    zip_kwargs = {"mode": "w"}
    if compresslevel is None:
        zip_kwargs["compression"] = zipfile.ZIP_STORED
    elif int(compresslevel) <= 0:
        zip_kwargs["compression"] = zipfile.ZIP_STORED
    else:
        zip_kwargs["compression"] = zipfile.ZIP_DEFLATED
        zip_kwargs["compresslevel"] = max(1, min(9, int(compresslevel)))

    with zipfile.ZipFile(output_path, **zip_kwargs) as archive:
        for root, dirs, files in os.walk(input_dir):
            dirs.sort(key=natural_sort_key)
            for file in sorted(files, key=natural_sort_key):
                if is_image_file(file):
                    file_path = os.path.join(root, file)
                    archive.write(file_path, arcname=os.path.relpath(file_path, input_dir))

def make_cb7(input_dir, output_path="", output_dir="", output_base_name=""):
    if not output_path:
        output_path = os.path.join(output_dir, f"{output_base_name}_translated.cb7")
    if os.path.exists(output_path):
        raise FileExistsError(f"Output archive already exists: {output_path}")

    import py7zr
    with py7zr.SevenZipFile(output_path, 'w') as archive:
        for root, dirs, files in os.walk(input_dir):
            for file in files:
                if is_image_file(file):
                    file_path = os.path.join(root, file)
                    archive.write(file_path, arcname=os.path.relpath(file_path, input_dir))

def make_pdf(input_dir, output_path="", output_dir="", output_base_name=""):
    import img2pdf
    
    if not output_path:
        output_path = os.path.join(output_dir, f"{output_base_name}_translated.pdf")
    if os.path.exists(output_path):
        raise FileExistsError(f"Output archive already exists: {output_path}")

    image_paths = []
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if is_image_file(file):
                image_paths.append(os.path.join(root, file))
    
    sorted_paths = sorted(image_paths, key=lambda p: natural_sort_key(os.path.basename(p)))
    
    with open(output_path, "wb") as f:
        f.write(img2pdf.convert(sorted_paths))

def make(input_dir, output_path="", save_as_ext="", output_dir="", output_base_name="", compresslevel=None):
    if not output_path and (not output_dir or not output_base_name):
        raise ValueError("Either output_path or both output_dir and output_base_name must be provided")
    
    if output_path:
        save_as_ext = os.path.splitext(output_path)[1]

    if save_as_ext in ['.cbz', '.zip']:
        make_cbz(
            input_dir,
            output_path,
            output_dir,
            output_base_name,
            save_as_ext,
            compresslevel=compresslevel,
        )
    elif save_as_ext == '.cb7':
        make_cb7(input_dir, output_path, output_dir, output_base_name)
    elif save_as_ext == '.pdf':
        make_pdf(input_dir, output_path, output_dir, output_base_name)
    else:
        raise ValueError(f"Unsupported save_as_ext: {save_as_ext}")

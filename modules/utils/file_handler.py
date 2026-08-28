import os
import shutil
import tempfile
import threading

from PySide6.QtCore import QCoreApplication

from .archives import (
    close_comic_cache,
    close_pdf_cache,
    inspect_archive_entry_dimensions,
    list_archive_image_entries,
    materialize_archive_entry,
    materialize_archive_entries,
)
from .export_paths import sanitize_export_path_component
from .image_safety import (
    ImageResourcePlan,
    build_image_resource_plan,
    inspect_image_dimensions,
)
from .pdf_pages import (
    PdfImportError,
    PdfPagePlan,
    scan_pdf,
    validate_materialized_page,
)

_LAZY_SOURCE_LOCK = threading.RLock()
_LAZY_SOURCE_BY_PATH: dict[str, dict] = {}


def _register_lazy_source(path: str, source: dict) -> None:
    with _LAZY_SOURCE_LOCK:
        _LAZY_SOURCE_BY_PATH[os.path.abspath(path)] = source


def _clear_lazy_sources_under_dir(base_dir: str) -> None:
    base = os.path.abspath(base_dir)
    with _LAZY_SOURCE_LOCK:
        stale_paths = [p for p in _LAZY_SOURCE_BY_PATH if p.startswith(base)]
        for p in stale_paths:
            _LAZY_SOURCE_BY_PATH.pop(p, None)


def get_prepared_path_source(path: str) -> dict | None:
    if not path:
        return None
    abs_path = os.path.abspath(path)
    with _LAZY_SOURCE_LOCK:
        source = _LAZY_SOURCE_BY_PATH.get(abs_path)
    if not isinstance(source, dict):
        return None
    return dict(source)


def ensure_prepared_path_materialized(path: str) -> bool:
    if not path:
        return False
    abs_path = os.path.abspath(path)
    with _LAZY_SOURCE_LOCK:
        source = _LAZY_SOURCE_BY_PATH.get(abs_path)
    if source is not None:
        archive_path = str(source.get("archive_path", ""))
        entry = source.get("entry")
        if archive_path.lower().endswith(".pdf") and isinstance(entry, dict):
            plan_value = entry.get("pdf_plan")
            if isinstance(plan_value, dict):
                plan = PdfPagePlan.from_dict(plan_value)
                if validate_materialized_page(plan, abs_path):
                    return True
                try:
                    if os.path.isfile(abs_path):
                        os.remove(abs_path)
                except OSError:
                    return False
    try:
        if os.path.isfile(abs_path) and os.path.getsize(abs_path) > 0:
            return True
    except Exception:
        pass

    if source is None:
        return os.path.isfile(abs_path)

    archive_path = str(source.get("archive_path", ""))
    entry = source.get("entry")
    if not archive_path or not isinstance(entry, dict):
        return False

    return materialize_archive_entry(archive_path, entry, abs_path)


class FileHandler:
    def __init__(self):
        self.file_paths = []
        self.archive_info = []
        self._pdf_import_warnings: list[dict[str, object]] = []
        self._pdf_warning_cursor = 0
        self._image_resource_plan = ImageResourcePlan()

    def cleanup(self) -> None:
        for archive in list(self.archive_info):
            temp_dir = str(archive.get("temp_dir") or "")
            archive_path = archive.get("archive_path")
            close_comic_cache(archive_path)
            close_pdf_cache(archive_path)
            if temp_dir:
                _clear_lazy_sources_under_dir(temp_dir)
                shutil.rmtree(temp_dir, ignore_errors=True)
        self.archive_info = []
        self.file_paths = []
        self._pdf_import_warnings = []
        self._pdf_warning_cursor = 0
        self._image_resource_plan = ImageResourcePlan()

    def prepare_files(
        self,
        file_paths: list[str],
        extend: bool = False,
        progress_callback=None,
    ):
        def report(stage: str, message: str, *, current: int = 0, total: int = 0, detail: str = ""):
            if callable(progress_callback):
                progress_callback(
                    {
                        "stage": stage,
                        "message": message,
                        "current": current,
                        "total": total,
                        "detail": detail,
                    }
                )

        all_image_paths = []
        if not extend:
            self.cleanup()

        selected_total = len(file_paths)
        for selected_index, path in enumerate(file_paths, start=1):
            report(
                "validate",
                QCoreApplication.translate("OpenWorkspace", "Checking selected files..."),
                current=selected_index,
                total=selected_total,
                detail=os.path.basename(path),
            )
            if path.lower().endswith((
                '.cbr', '.cbz', '.cbt', '.cb7',
                '.zip', '.rar', '.7z', '.tar',
                '.pdf', '.epub',
            )):
                print('Indexing archive:', path)
                report(
                    "index",
                    QCoreApplication.translate("OpenWorkspace", "Indexing archive..."),
                    detail=os.path.basename(path),
                )
                archive_dir = os.path.dirname(path)
                archive_stem = os.path.splitext(os.path.basename(path))[0]
                temp_prefix = f"tmp_{sanitize_export_path_component(archive_stem)[:80]}_"
                temp_dir = tempfile.mkdtemp(prefix=temp_prefix, dir=archive_dir)
                warning_start = len(self._pdf_import_warnings)
                try:
                    entries = list_archive_image_entries(path)
                    total = len(entries)
                    digits = len(str(total)) if total > 0 else 1
                    image_paths: list[str] = []

                    for index, entry in enumerate(entries, start=1):
                        report(
                            "index",
                            QCoreApplication.translate(
                                "OpenWorkspace", "Indexing archive pages..."
                            ),
                            current=index,
                            total=total,
                            detail=os.path.basename(path),
                        )
                        ext = str(entry.get("ext", ".png"))
                        if not ext.startswith("."):
                            ext = f".{ext}"
                        lazy_path = os.path.join(temp_dir, f"{index:0{digits}d}{ext.lower()}")
                        _register_lazy_source(
                            lazy_path,
                            {"archive_path": path, "entry": entry},
                        )
                        image_paths.append(lazy_path)
                        plan_value = entry.get("pdf_plan")
                        if isinstance(plan_value, dict):
                            warning = PdfPagePlan.from_dict(plan_value).warning_payload()
                            if warning is not None:
                                self._pdf_import_warnings.append(warning)

                    # Improve first paint latency by ensuring page 1 is ready.
                    if image_paths:
                        report(
                            "materialize",
                            QCoreApplication.translate(
                                "OpenWorkspace", "Preparing the first page..."
                            ),
                            current=1,
                            total=total,
                            detail=os.path.basename(path),
                        )
                        ensure_prepared_path_materialized(image_paths[0])
                except Exception:
                    del self._pdf_import_warnings[warning_start:]
                    self._pdf_warning_cursor = min(
                        self._pdf_warning_cursor, len(self._pdf_import_warnings)
                    )
                    close_comic_cache(path)
                    close_pdf_cache(path)
                    _clear_lazy_sources_under_dir(temp_dir)
                    shutil.rmtree(temp_dir, ignore_errors=True)
                    raise

                all_image_paths.extend(image_paths)
                self.archive_info.append({
                    'archive_path': path,
                    'extracted_images': image_paths,
                    'temp_dir': temp_dir,
                })
            else:
                all_image_paths.append(path)

        self.file_paths = self.file_paths + all_image_paths if extend else all_image_paths
        return all_image_paths

    def get_pdf_import_warnings(
        self,
        paths: list[str] | None = None,
    ) -> list[dict[str, object]]:
        if paths is None:
            return [dict(warning) for warning in self._pdf_import_warnings]

        warnings: list[dict[str, object]] = []
        for path in paths:
            with _LAZY_SOURCE_LOCK:
                source = _LAZY_SOURCE_BY_PATH.get(os.path.abspath(path))
            if not isinstance(source, dict):
                continue
            entry = source.get("entry")
            plan_value = entry.get("pdf_plan") if isinstance(entry, dict) else None
            if not isinstance(plan_value, dict):
                continue
            warning = PdfPagePlan.from_dict(plan_value).warning_payload()
            if warning is not None:
                warnings.append(warning)
        return warnings

    def consume_pdf_import_warnings(self) -> list[dict[str, object]]:
        warnings = self._pdf_import_warnings[self._pdf_warning_cursor:]
        self._pdf_warning_cursor = len(self._pdf_import_warnings)
        return [dict(warning) for warning in warnings]

    def preflight_for_processing(
        self,
        paths: list[str],
        should_cancel=None,
    ) -> int:
        canonical_plans: dict[str, list[PdfPagePlan]] = {}
        selected_pdf_paths = 0
        resources: list[tuple[str, int, int]] = []

        for path in list(paths or []):
            if should_cancel is not None and should_cancel():
                from .exceptions import OperationCancelledError

                raise OperationCancelledError("Cancelled during image resource preflight.")
            abs_path = os.path.abspath(path)
            with _LAZY_SOURCE_LOCK:
                source = _LAZY_SOURCE_BY_PATH.get(abs_path)
            if not isinstance(source, dict):
                width, height = inspect_image_dimensions(abs_path)
                resources.append((abs_path, width, height))
                continue
            archive_path = str(source.get("archive_path", ""))
            entry = source.get("entry")
            if not archive_path or not isinstance(entry, dict):
                raise FileNotFoundError(abs_path)
            if archive_path.lower().endswith(".pdf"):
                plan_value = entry.get("pdf_plan")
                if not isinstance(plan_value, dict):
                    raise PdfImportError(
                        "PDF_SOURCE_CHANGED",
                        page_index=int(entry.get("page_index", -1)),
                        retryable=True,
                        detail_code="plan_schema_mismatch",
                    )
                plan = PdfPagePlan.from_dict(plan_value)
                selected_pdf_paths += 1
                plans = canonical_plans.get(archive_path)
                if plans is None:
                    _identity, plans = scan_pdf(archive_path)
                    canonical_plans[archive_path] = plans
                if (
                    plan.page_index < 0
                    or plan.page_index >= len(plans)
                    or plans[plan.page_index] != plan
                ):
                    raise PdfImportError(
                        "PDF_SOURCE_CHANGED",
                        page_index=plan.page_index,
                        retryable=True,
                        detail_code="plan_source_mismatch",
                    )
                width, height = int(plan.width), int(plan.height)
            else:
                width, height = inspect_archive_entry_dimensions(archive_path, entry)
            resources.append((abs_path, width, height))

        self._image_resource_plan = build_image_resource_plan(resources)
        return selected_pdf_paths

    def image_resource_plan(self) -> ImageResourcePlan:
        return self._image_resource_plan

    def should_pre_materialize(self, target_paths: list[str] | None = None) -> bool:
        paths = list(target_paths or [])
        if not paths:
            return False
        all_paths = list(self.file_paths or [])
        if not all_paths:
            return False
        target_count = len(set(paths))
        total_count = max(1, len(set(all_paths)))
        ratio = target_count / total_count
        return target_count == total_count or ratio >= 0.7

    def pre_materialize(self, target_paths: list[str] | None = None) -> int:
        paths = list(target_paths or self.file_paths or [])
        if not paths:
            return 0

        grouped: dict[str, list[tuple[dict, str]]] = {}
        fallback_paths: list[str] = []

        for path in paths:
            abs_path = os.path.abspath(path)
            try:
                if os.path.isfile(abs_path) and os.path.getsize(abs_path) > 0:
                    continue
            except Exception:
                pass

            with _LAZY_SOURCE_LOCK:
                source = _LAZY_SOURCE_BY_PATH.get(abs_path)
            if source is None:
                continue

            archive_path = str(source.get("archive_path", ""))
            entry = source.get("entry")
            if archive_path and isinstance(entry, dict):
                grouped.setdefault(archive_path, []).append((entry, abs_path))
            else:
                fallback_paths.append(abs_path)

        completed = 0
        for archive_path, items in grouped.items():
            completed += materialize_archive_entries(archive_path, items)

        for abs_path in fallback_paths:
            with _LAZY_SOURCE_LOCK:
                source = _LAZY_SOURCE_BY_PATH.get(abs_path)
            if source is None:
                continue
            archive_path = str(source.get("archive_path", ""))
            entry = source.get("entry")
            if archive_path and isinstance(entry, dict):
                if materialize_archive_entry(archive_path, entry, abs_path):
                    completed += 1

        return completed

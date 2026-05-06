from __future__ import annotations

import copy
import hashlib
import logging
import os
import re
import shutil
import tempfile
import time
from collections import OrderedDict
from datetime import datetime
from typing import TYPE_CHECKING
from dataclasses import asdict, is_dataclass

import msgpack

from PySide6 import QtWidgets, QtCore, QtGui
from PySide6.QtCore import QSettings, QPointF

from app.thread_worker import GenericWorker
from app.controllers.psd_exporter import PsdPageData, export_psd_pages
from app.controllers.psd_support import ensure_photoshopapi_available
from app.ui.canvas.text_item import TextBlockItem
from app.ui.canvas.text.text_item_properties import TextItemProperties
from app.ui.canvas.save_renderer import ImageSaveRenderer
from app.ui.export_chapters_dialog import ExportChaptersDialog, ExportChapterRow
from app.projects.project_state import (
    close_state_store,
    load_state_from_proj_file,
    save_state_to_proj_file,
)
from app.projects.project_types import (
    PROJECT_FILE_EXT,
    PROJECT_KIND_SERIES,
    PROJECT_KIND_SINGLE,
    SERIES_PROJECT_FILE_EXT,
    ensure_project_extension,
    project_extension_for_kind,
    project_file_filter_for_kind,
)
from app.projects.parsers import ProjectEncoder
from modules.utils.archives import make
from modules.utils.automatic_output import (
    build_archive_file_name,
    build_archive_page_file_name,
    build_archive_staging_dir,
    build_output_file_name,
    is_single_archive_mode,
    write_archive_image,
    write_output_image,
)
from modules.utils.paths import get_user_data_dir, get_default_project_autosave_dir

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from controller import ComicTranslate
    

class ProjectController:
    def __init__(self, main: ComicTranslate):
        self.main = main
        self._autosave_timer = QtCore.QTimer(self.main)
        self._autosave_timer.setSingleShot(False)
        self._autosave_timer.timeout.connect(self._on_autosave_timeout)
        self._realtime_autosave_timer = QtCore.QTimer(self.main)
        self._realtime_autosave_timer.setSingleShot(True)
        self._realtime_autosave_timer.setInterval(800)
        self._realtime_autosave_timer.timeout.connect(self._on_realtime_autosave_timeout)
        self._autosave_signals_connected = False
        self._autosave_save_pending = False
        self._autosave_retrigger_requested = False
        self._active_save_workers: list = []  # keeps Python refs alive until workers finish

    # Recent projects (persisted via QSettings)

    MAX_RECENT = 15

    def _current_project_kind(self) -> str:
        return str(getattr(self.main, "project_kind", PROJECT_KIND_SINGLE) or PROJECT_KIND_SINGLE)

    def _project_extension(self, kind: str | None = None) -> str:
        return project_extension_for_kind(kind or self._current_project_kind())

    def _recovered_project_display_name(self, kind: str | None = None) -> str:
        effective_kind = kind or self._current_project_kind()
        if effective_kind == PROJECT_KIND_SERIES:
            return self.main.tr("RecoveredProject.seriesctpr")
        return self.main.tr("RecoveredProject.ctpr")

    def _read_autosave_enabled_setting(self) -> bool:
        settings = QSettings("ComicLabs", "ComicTranslate")
        settings.beginGroup("export")
        value = settings.value("project_autosave_enabled", False, type=bool)
        settings.endGroup()
        return bool(value)

    def _write_autosave_enabled_setting(self, enabled: bool) -> None:
        settings = QSettings("ComicLabs", "ComicTranslate")
        settings.beginGroup("export")
        settings.setValue("project_autosave_enabled", bool(enabled))
        settings.endGroup()
        settings.sync()

    def add_recent_project(self, path: str) -> None:
        """Push *path* to the front of the recent list; cap at MAX_RECENT."""
        if not path or not os.path.isfile(path):
            return
        path = os.path.normpath(os.path.abspath(path))
        entries = self.get_recent_projects()
        # Preserve pin state if already present
        existing = next((e for e in entries if os.path.normpath(e["path"]) == path), None)
        pinned = existing.get("pinned", False) if existing else False
        opened_at = time.time()
        # Remove duplicates
        entries = [e for e in entries if os.path.normpath(e["path"]) != path]
        try:
            mtime = os.path.getmtime(path)
        except OSError:
            mtime = 0.0
        entries.insert(0, {"path": path, "mtime": mtime, "pinned": pinned, "opened_at": opened_at})
        entries = entries[: self.MAX_RECENT]
        self._save_entries(entries)

    def get_recent_projects(self) -> list:
        """Return list of ``{path, mtime, pinned, opened_at}`` dicts sorted by open recency."""
        settings = QSettings("ComicLabs", "ComicTranslate")
        settings.beginGroup("recent_projects")
        paths   = settings.value("paths",   []) or []
        mtimes  = settings.value("mtimes",  []) or []
        pinneds = settings.value("pinned",  []) or []
        opened_ats = settings.value("opened_at", []) or []
        settings.endGroup()
        # Normalise types — QSettings may return a single string if only 1 entry
        if isinstance(paths, str):   paths   = [paths]
        if not isinstance(mtimes,  list): mtimes  = [mtimes]
        if not isinstance(pinneds, list): pinneds = [pinneds]
        if not isinstance(opened_ats, list): opened_ats = [opened_ats]
        result = []
        fallback_base = float(len(paths))
        for i, (path, mtime) in enumerate(zip(paths, mtimes)):
            try:
                m = float(mtime)
            except (TypeError, ValueError):
                m = 0.0
            # Refresh from filesystem when possible so ordering reflects real
            # modified time, not only the previously-saved snapshot.
            try:
                if os.path.isfile(path):
                    m = float(os.path.getmtime(path))
            except OSError:
                pass
            try:
                p = str(pinneds[i]).lower() == "true" if i < len(pinneds) else False
            except Exception:
                p = False
            try:
                opened_at = float(opened_ats[i]) if i < len(opened_ats) else 0.0
            except (TypeError, ValueError):
                opened_at = 0.0
            if opened_at <= 0:
                opened_at = fallback_base - float(i)
            result.append({"path": str(path), "mtime": m, "pinned": p, "opened_at": opened_at})
        result.sort(key=lambda e: float(e.get("opened_at", 0.0) or 0.0), reverse=True)
        return result

    def remove_recent_project(self, path: str) -> None:
        """Remove *path* from the recent list."""
        path = os.path.normpath(os.path.abspath(path))
        entries = [
            e for e in self.get_recent_projects()
            if os.path.normpath(e["path"]) != path
        ]
        self._save_entries(entries)

    def toggle_pin_project(self, path: str, pinned: bool) -> None:
        """Set the pinned flag for *path* and persist."""
        path = os.path.normpath(os.path.abspath(path))
        entries = self.get_recent_projects()
        for e in entries:
            if os.path.normpath(e["path"]) == path:
                e["pinned"] = pinned
                break
        self._save_entries(entries)

    def clear_recent_projects(self) -> None:
        """Wipe the entire recent list."""
        settings = QSettings("ComicLabs", "ComicTranslate")
        settings.beginGroup("recent_projects")
        settings.remove("")
        settings.endGroup()

    @staticmethod
    def _save_entries(entries: list) -> None:
        """Write the full entries list to QSettings."""
        settings = QSettings("ComicLabs", "ComicTranslate")
        settings.beginGroup("recent_projects")
        settings.setValue("paths",  [e["path"] for e in entries])
        settings.setValue("mtimes", [e["mtime"] for e in entries])
        settings.setValue("pinned", [e.get("pinned", False) for e in entries])
        settings.setValue("opened_at", [e.get("opened_at", 0.0) for e in entries])
        settings.endGroup()
        settings.sync()

    def initialize_autosave(self):
        # Restore persisted auto-save toggle state as a single source of truth.
        persisted_enabled = self._read_autosave_enabled_setting()
        self.main.title_bar.set_autosave_checked(persisted_enabled)

        if self._autosave_signals_connected:
            self._apply_autosave_settings()
            return

        self.main.title_bar.autosave_switch.toggled.connect(self._on_autosave_setting_changed)
        self.main.settings_page.ui.project_autosave_interval_spinbox.valueChanged.connect(
            self._on_autosave_setting_changed
        )
        self._autosave_signals_connected = True
        self._apply_autosave_settings()

    def _on_autosave_setting_changed(self, *_):
        autosave_enabled = bool(self.main.title_bar.autosave_switch.isChecked())

        # Defer auto-generating a project file until the user has actually
        # entered the workspace (not startup home) and loaded/started pages.
        self._ensure_autosave_project_file_if_needed()

        # Persist this key directly and sync so it is independent of UI widget
        # availability/order during shutdown.
        try:
            self._write_autosave_enabled_setting(autosave_enabled)
        except Exception:
            logger.debug("Failed to persist autosave toggle directly.", exc_info=True)

        self._apply_autosave_settings()

    def _apply_autosave_settings(self):
        export_settings = self.main.settings_page.get_export_settings()
        interval_min = int(export_settings.get("project_autosave_interval_min", 3) or 3)
        interval_min = max(1, min(interval_min, 120))

        self._autosave_timer.setInterval(interval_min * 60 * 1000)
        # Crash recovery snapshots remain interval-based.
        self._autosave_timer.start()

    def _is_startup_home_visible(self) -> bool:
        try:
            center_stack = self.main._center_stack
            home_screen = self.main.startup_home
            if center_stack is not None and home_screen is not None:
                return center_stack.currentWidget() is home_screen
        except Exception:
            pass
        return False

    def _ensure_autosave_project_file_if_needed(
        self,
        require_images: bool = True,
        *,
        ignore_home_visibility: bool = False,
    ) -> None:
        autosave_enabled = bool(
            self.main.settings_page.get_export_settings().get("project_autosave_enabled", False)
        )
        if not autosave_enabled or self.main.project_file:
            return
        if require_images and not self.main.image_files:
            return
        # Avoid creating a "rogue" auto-save file on plain startup before user intent.
        if self._is_startup_home_visible() and not ignore_home_visibility:
            return

        generated_project_file = self._generate_autosave_project_file_path()
        self.main.project_file = generated_project_file
        self.main.setWindowTitle(f"{os.path.basename(generated_project_file)}[*]")

    def ensure_autosave_project_file_for_new_project(self) -> None:
        """Create an auto-save project file after explicit New Project intent."""
        self._ensure_autosave_project_file_if_needed(require_images=False)

    def ensure_autosave_project_file_for_transition(self) -> None:
        """Force-create an auto-save target before guarded project transitions."""
        self._ensure_autosave_project_file_if_needed(
            require_images=False,
            ignore_home_visibility=True,
        )

    def shutdown_autosave(self, clear_recovery: bool = True):
        try:
            self._autosave_timer.stop()
        except Exception:
            pass
        try:
            self._realtime_autosave_timer.stop()
        except Exception:
            pass
        close_state_store()
        if clear_recovery:
            self.clear_recovery_checkpoint()

    def _autosave_dir(self) -> str:
        return os.path.join(get_user_data_dir(), "autosave")

    def _recovery_project_path(self) -> str:
        return os.path.join(self._autosave_dir(), f"project_recovery{self._project_extension()}")

    def _configured_project_autosave_dir(self) -> str:
        export_settings = self.main.settings_page.get_export_settings()
        configured_folder = str(export_settings.get("project_autosave_folder", "") or "").strip()
        if configured_folder:
            return configured_folder
        return get_default_project_autosave_dir()

    def _generate_autosave_project_file_path(self) -> str:
        autosave_dir = self._configured_project_autosave_dir()
        try:
            os.makedirs(autosave_dir, exist_ok=True)
        except Exception:
            logger.warning("Failed to create configured auto-save folder: %s", autosave_dir)
            autosave_dir = self._autosave_dir()
            os.makedirs(autosave_dir, exist_ok=True)

        base_name = "project"

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        extension = self._project_extension()
        candidate = os.path.join(autosave_dir, f"{base_name}_{timestamp}{extension}")
        if not os.path.exists(candidate):
            return candidate

        for seq in range(1, 1000):
            seq_candidate = os.path.join(
                autosave_dir,
                f"{base_name}_{timestamp}_{seq:03d}{extension}",
            )
            if not os.path.exists(seq_candidate):
                return seq_candidate

        # Extremely unlikely fallback; keeps behavior deterministic if all sequence
        # slots are exhausted for the same timestamp.
        fallback_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        return os.path.join(autosave_dir, f"{base_name}_{fallback_timestamp}{extension}")

    def clear_recovery_checkpoint(self):
        for extension in (PROJECT_FILE_EXT, SERIES_PROJECT_FILE_EXT):
            recovery_file = os.path.join(self._autosave_dir(), f"project_recovery{extension}")
            if os.path.exists(recovery_file):
                try:
                    os.remove(recovery_file)
                except Exception:
                    logger.debug("Failed to remove recovery project file: %s", recovery_file)

    def _on_autosave_timeout(self):
        # Interval timer is reserved for recovery checkpoints.
        self.autosave_project(prefer_project_file=False)

    def _on_realtime_autosave_timeout(self):
        # Real-time autosave writes directly to the open project file.
        self.autosave_project(prefer_project_file=True)

    def _on_batch_page_done(self, image_path: str):
        """Triggered by render_state_ready after each page is processed during batch.
        Saves the project file immediately so progress is not lost between pages.

        NOTE: this deliberately bypasses the task_runner_ctrl queue (which is
        blocked while the batch is running) and avoids touching current_worker
        (which the batch processor uses for cancel detection). A GenericWorker
        is started directly on the shared threadpool instead.
        """
        if self._current_project_kind() == PROJECT_KIND_SERIES:
            return
        autosave_enabled = bool(
            self.main.settings_page.get_export_settings().get("project_autosave_enabled", False)
        )
        if not autosave_enabled or not self.main.project_file:
            return
        if self._autosave_save_pending:
            # A save is already in flight; request a follow-up once it finishes.
            self._autosave_retrigger_requested = True
            return

        autosave_start_revision = self.main._dirty_revision
        self._autosave_save_pending = True
        target_file = self.main.project_file

        worker = GenericWorker(self.save_project, target_file)
        self._active_save_workers.append(worker)  # prevent GC until done

        def on_error(error_tuple):
            try:
                self._active_save_workers.remove(worker)
            except ValueError:
                pass
            self._autosave_save_pending = False
            exctype, value, _ = error_tuple
            logger.warning("Per-page autosave failed for %s: %s: %s", image_path, exctype.__name__, value)
            if self._autosave_retrigger_requested:
                self._autosave_retrigger_requested = False
                self._on_batch_page_done(image_path)

        def on_finished():
            try:
                self._active_save_workers.remove(worker)
            except ValueError:
                pass
            self._autosave_save_pending = False
            self.clear_recovery_checkpoint()
            self.add_recent_project(target_file)
            self._refresh_home_screen()
            if self.main._dirty_revision == autosave_start_revision:
                self.main.set_project_clean()
            if self._autosave_retrigger_requested:
                self._autosave_retrigger_requested = False
                self._on_batch_page_done(image_path)

        worker.signals.error.connect(
            lambda err: QtCore.QTimer.singleShot(0, self.main, lambda: on_error(err))
        )
        worker.signals.finished.connect(
            lambda: QtCore.QTimer.singleShot(0, self.main, on_finished)
        )
        self.main.threadpool.start(worker)

    def notify_project_dirty_revision_changed(self):
        autosave_enabled = bool(
            self.main.settings_page.get_export_settings().get("project_autosave_enabled", False)
        )
        if not autosave_enabled:
            return
        self._ensure_autosave_project_file_if_needed()
        # Debounce bursts of edits (typing, drag, rapid undo/redo).
        # If no project file exists yet, autosave_project() falls back to
        # the recovery checkpoint path.
        self._realtime_autosave_timer.start()

    def autosave_project(self, prefer_project_file: bool = True):
        if self._current_project_kind() == PROJECT_KIND_SERIES:
            self._autosave_series_project(prefer_project_file=prefer_project_file)
            return
        if self._autosave_save_pending:
            if prefer_project_file:
                self._autosave_retrigger_requested = True
            return
        if not self.main.image_files:
            return
        if getattr(self.main, "_batch_active", False):
            return

        # Flush pending text-edit command batching so autosave captures the latest edits.
        try:
            self.main.text_ctrl._commit_pending_text_command()
        except Exception:
            pass

        if not self.main.has_unsaved_changes():
            return

        self.save_current_state()

        autosave_enabled = bool(
            self.main.settings_page.get_export_settings().get("project_autosave_enabled", False)
        )
        self._ensure_autosave_project_file_if_needed()
        use_project_file = bool(prefer_project_file and autosave_enabled and self.main.project_file)
        target_file = self.main.project_file if use_project_file else self._recovery_project_path()
        if not target_file:
            return

        is_regular_project_save = bool(self.main.project_file and target_file == self.main.project_file)
        autosave_start_revision = self.main._dirty_revision
        self._autosave_save_pending = True

        def on_error(error_tuple):
            self._autosave_save_pending = False
            exctype, value, _ = error_tuple
            logger.warning("Project autosave failed: %s: %s", exctype.__name__, value)
            if self._autosave_retrigger_requested:
                self._autosave_retrigger_requested = False
                self._realtime_autosave_timer.start()

        def on_finished():
            self._autosave_save_pending = False
            if is_regular_project_save:
                self.clear_recovery_checkpoint()
                self.add_recent_project(target_file)
                self._refresh_home_screen()
                if self.main._dirty_revision == autosave_start_revision:
                    self.main.set_project_clean()
            if self._autosave_retrigger_requested or (
                is_regular_project_save and self.main.has_unsaved_changes()
            ):
                self._autosave_retrigger_requested = False
                self._realtime_autosave_timer.start()

        self.main.run_threaded(self.save_project, None, on_error, on_finished, target_file)

    def _autosave_series_project(self, prefer_project_file: bool = True):
        if self._autosave_save_pending:
            if prefer_project_file:
                self._autosave_retrigger_requested = True
            return
        if not getattr(self.main, "series_ctrl", None) or not self.main.series_ctrl.has_series_loaded():
            return
        if getattr(self.main, "_batch_active", False):
            return
        if not self.main.has_unsaved_changes():
            return

        autosave_enabled = bool(
            self.main.settings_page.get_export_settings().get("project_autosave_enabled", False)
        )
        self._ensure_autosave_project_file_if_needed()
        use_project_file = bool(prefer_project_file and autosave_enabled and self.main.project_file)
        target_file = self.main.project_file if use_project_file else self._recovery_project_path()
        if not target_file:
            return

        autosave_start_revision = self.main._dirty_revision
        self._autosave_save_pending = True
        try:
            self.main.series_ctrl.sync_active_child_to_series()
        except Exception:
            self._autosave_save_pending = False
            logger.warning("Series autosave sync failed.", exc_info=True)
            return

        def worker() -> str:
            current_series = str(self.main.series_ctrl.series_file or "")
            if not current_series:
                raise FileNotFoundError("Series project file is not available for autosave.")
            if target_file != current_series:
                shutil.copyfile(current_series, target_file)
            return target_file

        def on_error(error_tuple):
            self._autosave_save_pending = False
            exctype, value, _ = error_tuple
            logger.warning("Series autosave failed: %s: %s", exctype.__name__, value)
            if self._autosave_retrigger_requested:
                self._autosave_retrigger_requested = False
                self._realtime_autosave_timer.start()

        def on_finished():
            self._autosave_save_pending = False
            if use_project_file and self.main._dirty_revision == autosave_start_revision:
                self.main.set_project_clean()
                self.add_recent_project(target_file)
                self._refresh_home_screen()
            if self._autosave_retrigger_requested:
                self._autosave_retrigger_requested = False
                self._realtime_autosave_timer.start()

        self.main.run_threaded(worker, None, on_error, on_finished)

    def prompt_restore_recovery_if_available(self) -> bool:
        if self.main.image_files:
            return False

        candidates = [
            os.path.join(self._autosave_dir(), f"project_recovery{PROJECT_FILE_EXT}"),
            os.path.join(self._autosave_dir(), f"project_recovery{SERIES_PROJECT_FILE_EXT}"),
        ]
        existing = [path for path in candidates if os.path.exists(path)]
        if not existing:
            return False
        recovery_file = max(existing, key=lambda path: os.path.getmtime(path))
        if not os.path.exists(recovery_file):
            return False

        saved_at = datetime.fromtimestamp(os.path.getmtime(recovery_file)).strftime("%Y-%m-%d %H:%M:%S")

        msg_box = QtWidgets.QMessageBox(self.main)
        msg_box.setIcon(QtWidgets.QMessageBox.Question)
        msg_box.setWindowTitle(self.main.tr("Project Recovery"))
        msg_box.setText(self.main.tr("An autosaved project from a previous session was found."))
        msg_box.setInformativeText(
            self.main.tr("Last autosave: {saved_at}\nDo you want to restore it?").format(saved_at=saved_at)
        )
        restore_btn = msg_box.addButton(self.main.tr("Restore"), QtWidgets.QMessageBox.ButtonRole.AcceptRole)
        discard_btn = msg_box.addButton(self.main.tr("Discard"), QtWidgets.QMessageBox.ButtonRole.DestructiveRole)
        msg_box.setDefaultButton(restore_btn)
        msg_box.exec()

        if msg_box.clickedButton() == restore_btn:
            self.restore_recovery_project(recovery_file)
            return True

        if msg_box.clickedButton() == discard_btn:
            self.clear_recovery_checkpoint()

        return False

    def restore_recovery_project(self, recovery_file: str | None = None):
        recovery_file = recovery_file or self._recovery_project_path()
        if not os.path.exists(recovery_file):
            return

        if recovery_file.lower().endswith(SERIES_PROJECT_FILE_EXT):
            self.main.series_ctrl.thread_load_series_project(recovery_file, recovery_loaded=True)
            return

        self.main.image_ctrl.clear_state()
        self.main.setWindowTitle(f"{self._recovered_project_display_name(PROJECT_KIND_SINGLE)}[*]")

        load_failed = {"value": False}

        def on_error(error_tuple):
            load_failed["value"] = True
            self.main.default_error_handler(error_tuple)

        def on_finished():
            if load_failed["value"]:
                return
            self.update_ui_from_project()
            # Keep recovered data as an unsaved project so users can choose a destination.
            self.main.project_file = None
            self.main.project_kind = PROJECT_KIND_SINGLE
            self.main.setWindowTitle(f"{self._recovered_project_display_name(PROJECT_KIND_SINGLE)}[*]")
            self.main.mark_project_dirty()
            self.clear_recovery_checkpoint()

        self.main.run_threaded(
            self.load_project,
            self.load_state_to_ui,
            on_error,
            on_finished,
            recovery_file,
        )

    def save_and_make(self, output_path: str):
        self.main.loading.setVisible(True)
        self.main.run_threaded(
            self.save_and_make_worker,
            None,
            self.main.default_error_handler,
            lambda: self.main.loading.setVisible(False),
            output_path,
        )

    def export_to_psd_dialog(self):
        if not self.main.image_files:
            return
        if not ensure_photoshopapi_available(self.main):
            return

        default_dir = self._get_default_export_dir()
        if len(self.main.image_files) == 1:
            default_name = f"{os.path.splitext(os.path.basename(self.main.image_files[0]))[0]}.psd"
            selected_path, _ = QtWidgets.QFileDialog.getSaveFileName(
                self.main,
                self.main.tr("Export PSD As"),
                os.path.join(default_dir, default_name),
                self.main.tr("PSD Files (*.psd);;All Files (*)"),
            )
            if not selected_path:
                return
            if not selected_path.lower().endswith(".psd"):
                selected_path = f"{selected_path}.psd"
            self.export_to_psd(os.path.dirname(selected_path), single_file_path=selected_path)
            return

        export_rows = self._build_export_rows()
        if self._should_show_partition_dialog(export_rows):
            partition_result = self._prompt_for_partition(
                export_rows,
                os.path.join(default_dir, "untitled"),
            )
            if partition_result is None:
                return
            chapter_names_by_path, output_dir = partition_result
            export_plan = self._build_export_plan_for_directory(output_dir, "", chapter_names_by_path)
            self.export_psd_plan(export_plan)
            return

        selected_folder = self._launch_export_folder_dialog(
            self.main.tr("Export PSD"),
            suggested_name=self._get_export_bundle_name(),
            initial_dir=default_dir,
        )
        if selected_folder:
            self.export_to_psd(selected_folder)

    def export_to_psd(self, output_folder: str, single_file_path: str | None = None):
        if not ensure_photoshopapi_available(self.main):
            return
        self.main.image_ctrl.save_current_image_state()
        all_pages_current_state = self._build_all_pages_current_state()
        bundle_name = self._get_export_bundle_name()
        self.main.loading.setVisible(True)
        self.main.run_threaded(
            self._write_psd_worker,
            None,
            self.main.default_error_handler,
            lambda: self.main.loading.setVisible(False),
            output_folder,
            all_pages_current_state,
            bundle_name,
            single_file_path,
        )

    def export_psd_plan(self, export_plan: list[dict]) -> None:
        if not ensure_photoshopapi_available(self.main):
            return
        self.main.image_ctrl.save_current_image_state()
        all_pages_current_state = self._build_all_pages_current_state()
        bundle_name = self._get_export_bundle_name()
        self.main.loading.setVisible(True)
        self.main.run_threaded(
            self._write_psd_plan_worker,
            None,
            self.main.default_error_handler,
            lambda: self.main.loading.setVisible(False),
            export_plan,
            all_pages_current_state,
            bundle_name,
        )

    def _write_psd_worker(
        self,
        output_folder: str,
        all_pages_current_state: dict[str, dict],
        bundle_name: str,
        single_file_path: str | None = None,
    ):
        pages = self._gather_psd_pages(all_pages_current_state)
        export_psd_pages(
            output_folder=output_folder,
            pages=pages,
            bundle_name=bundle_name,
            single_file_path=single_file_path,
        )

    def _write_psd_plan_worker(
        self,
        export_plan: list[dict],
        all_pages_current_state: dict[str, dict],
        default_bundle_name: str,
    ) -> None:
        pages = self._gather_psd_pages(all_pages_current_state)
        for group in export_plan:
            group_pages = [
                pages[page_idx]
                for page_idx in group.get("page_indices", [])
                if 0 <= int(page_idx) < len(pages)
            ]
            if not group_pages:
                continue
            output_path = str(group.get("output_path") or "").strip()
            if not output_path:
                continue
            export_psd_pages(
                output_folder=output_path,
                pages=group_pages,
                bundle_name=str(group.get("group_name") or default_bundle_name),
            )

    @staticmethod
    def _sanitize_export_stem(value: str) -> str:
        sanitized = re.sub(r'[\\/:*?"<>|\r\n\t]+', "_", str(value or ""))
        sanitized = re.sub(r"\s+", " ", sanitized).strip().strip(".")
        return sanitized.strip("._-") or "chapter"

    def _default_export_group_name(self, file_path: str) -> str:
        state = self.main.image_states.get(file_path, {})
        group_name = str(state.get("export_group_name", "")).strip()
        if group_name:
            return group_name
        return self._get_export_bundle_name()

    def _build_export_rows(self) -> list[ExportChapterRow]:
        rows: list[ExportChapterRow] = []
        for page_index, file_path in enumerate(self.main.image_files):
            rows.append(
                ExportChapterRow(
                    page_index=page_index,
                    file_path=file_path,
                    file_name=os.path.basename(file_path),
                    group_name=self._default_export_group_name(file_path),
                )
            )
        return rows

    def _should_show_partition_dialog(self, rows: list[ExportChapterRow]) -> bool:
        if len(rows) <= 1:
            return False
        distinct_groups = {row.group_name for row in rows if row.group_name}
        if len(distinct_groups) > 1:
            return True
        basenames = [row.file_name.lower() for row in rows]
        return len(set(basenames)) != len(basenames)

    def _prompt_for_partition(
        self,
        export_rows: list[ExportChapterRow],
        output_path_hint: str,
    ) -> tuple[dict[str, str], str] | None:
        dialog = ExportChaptersDialog(
            export_rows,
            os.path.dirname(output_path_hint) or os.path.expanduser("~"),
            os.path.splitext(output_path_hint)[1],
            self._build_export_filename,
            self.main,
        )
        if dialog.exec() != QtWidgets.QDialog.Accepted:
            return None

        chapter_names_by_path = dialog.chapter_names_by_path()
        self._persist_export_group_names(chapter_names_by_path)
        return chapter_names_by_path, dialog.selected_output_dir()

    def _persist_export_group_names(self, chapter_names_by_path: dict[str, str]) -> None:
        changed = False
        for file_path, group_name in chapter_names_by_path.items():
            state = self.main.image_states.setdefault(file_path, {})
            normalized = str(group_name or "").strip()
            if state.get("export_group_name") != normalized:
                state["export_group_name"] = normalized
                changed = True
        if changed:
            self.main.mark_project_dirty()

    def _build_export_filename(self, group_name: str, extension: str, used_names: set[str]) -> str:
        ext = str(extension or "").strip()
        if ext and not ext.startswith("."):
            ext = f".{ext}"
        stem = self._sanitize_export_stem(group_name)
        candidate = f"{stem}{ext}"
        suffix = 2
        while candidate.lower() in used_names:
            candidate = f"{stem}_{suffix:02d}{ext}"
            suffix += 1
        used_names.add(candidate.lower())
        return candidate

    def _launch_export_folder_dialog(
        self,
        title: str,
        *,
        suggested_name: str | None = None,
        initial_dir: str | None = None,
    ) -> str:
        default_dir = initial_dir or self._get_default_export_dir()
        selected_path = QtWidgets.QFileDialog.getExistingDirectory(
            self.main,
            title,
            default_dir,
        )
        return str(selected_path or "").strip()

    def _build_export_plan_for_directory(
        self,
        output_dir: str,
        extension: str,
        chapter_names_by_path: dict[str, str],
    ) -> list[dict]:
        groups = self._group_page_indices(chapter_names_by_path)
        used_names: set[str] = set()
        plan: list[dict] = []
        for group_name, page_indices in groups.items():
            file_name = self._build_export_filename(group_name, extension, used_names)
            plan.append(
                {
                    "group_name": group_name,
                    "page_indices": page_indices,
                    "output_path": os.path.join(output_dir, file_name),
                }
            )
        return plan

    def _group_page_indices(self, chapter_names_by_path: dict[str, str]) -> OrderedDict[str, list[int]]:
        groups: OrderedDict[str, list[int]] = OrderedDict()
        for page_index, file_path in enumerate(self.main.image_files):
            group_name = str(chapter_names_by_path.get(file_path) or self._default_export_group_name(file_path)).strip()
            groups.setdefault(group_name, []).append(page_index)
        return groups

    def save_and_make_worker(self, output_path: str):
        self.main.image_ctrl.save_current_image_state()
        all_pages_current_state = self._build_all_pages_current_state()
        try:
            if self.main.file_handler.should_pre_materialize(self.main.image_files):
                count = self.main.file_handler.pre_materialize(self.main.image_files)
                logger.info("Export pre-materialized %d paths before save-and-make.", count)
        except Exception:
            logger.debug("Export pre-materialization failed; continuing lazily.", exc_info=True)
        temp_dir = tempfile.mkdtemp()
        try:            
            temp_main_page_context = None
            if self.main.webtoon_mode:
                temp_main_page_context = type('TempMainPage', (object,), {
                    'image_files': self.main.image_files,
                    'image_states': all_pages_current_state
                })()

            for page_idx, file_path in enumerate(self.main.image_files):
                bname = os.path.basename(file_path)
                rgb_img = self.main.load_image(file_path)
                renderer = ImageSaveRenderer(rgb_img)
                viewer_state = all_pages_current_state[file_path]['viewer_state']

                renderer.apply_patches(self.main.image_patches.get(file_path, []))
                if self.main.webtoon_mode and temp_main_page_context is not None:
                    renderer.add_state_to_image(viewer_state, page_idx, temp_main_page_context)
                else:
                    renderer.add_state_to_image(viewer_state)

                sv_pth = os.path.join(temp_dir, bname)
                renderer.save_image(sv_pth)

            # Call make function
            make(temp_dir, output_path)
        finally:
            # Clean up temp directory
            shutil.rmtree(temp_dir)

    def _render_signature_for_path(
        self,
        file_path: str,
        viewer_state: dict,
        export_settings: dict[str, object],
    ) -> str:
        state = self.main.image_states.get(file_path, {})
        payload = {
            "viewer_state": viewer_state or {},
            "blk_list": state.get("blk_list", []),
            "brush_strokes": state.get("brush_strokes", []),
            "image_patches": self.main.image_patches.get(file_path, []),
            "source_lang": state.get("source_lang", ""),
            "target_lang": state.get("target_lang", ""),
            "skip": bool(state.get("skip", False)),
            "export_settings": {
                key: export_settings.get(key)
                for key in (
                    "resolved_automatic_output_target",
                    "resolved_automatic_output_image_format",
                    "resolved_automatic_output_archive_format",
                    "resolved_automatic_output_archive_image_format",
                    "resolved_automatic_output_archive_compression_level",
                )
            },
        }
        packed = msgpack.packb(
            payload,
            default=ProjectEncoder().encode,
            use_bin_type=True,
            strict_types=False,
        )
        return hashlib.sha256(packed).hexdigest()

    def _render_page_output(
        self,
        *,
        file_path: str,
        page_index: int,
        total_pages: int,
        viewer_state: dict,
        temp_main_page_context,
        export_settings: dict[str, object],
        series_dir: str,
        staging_dir: str,
    ) -> str:
        rgb_img = self.main.load_image(file_path)
        renderer = ImageSaveRenderer(rgb_img)
        renderer.apply_patches(copy.deepcopy(self.main.image_patches.get(file_path, [])))
        if self.main.webtoon_mode and temp_main_page_context is not None:
            renderer.add_state_to_image(copy.deepcopy(viewer_state), page_index, temp_main_page_context)
        else:
            renderer.add_state_to_image(copy.deepcopy(viewer_state))
        final_rgb = renderer.render_to_image()

        page_base_name = os.path.splitext(os.path.basename(file_path))[0]
        if is_single_archive_mode(export_settings):
            os.makedirs(staging_dir, exist_ok=True)
            output_path = os.path.join(
                staging_dir,
                build_archive_page_file_name(
                    page_index,
                    total_pages,
                    page_base_name,
                    str(export_settings.get("resolved_automatic_output_archive_image_format", "png")),
                ),
            )
            write_archive_image(
                output_path,
                final_rgb,
                resolved_settings=export_settings,
            )
            return output_path

        os.makedirs(series_dir, exist_ok=True)
        output_path = os.path.join(
            series_dir,
            build_output_file_name(
                page_base_name,
                "translated",
                file_path,
                export_settings,
            ),
        )
        write_output_image(
            output_path,
            final_rgb,
            source_path=file_path,
            resolved_settings=export_settings,
        )
        return output_path

    def _archive_stage_path_for_page(
        self,
        *,
        file_path: str,
        page_index: int,
        total_pages: int,
        export_settings: dict[str, object],
        staging_dir: str,
    ) -> str:
        page_base_name = os.path.splitext(os.path.basename(file_path))[0]
        return os.path.join(
            staging_dir,
            build_archive_page_file_name(
                page_index,
                total_pages,
                page_base_name,
                str(export_settings.get("resolved_automatic_output_archive_image_format", "png")),
            ),
        )

    def _rerender_output_worker(
        self,
        requested_paths: list[str],
        scope: str,
        all_pages_current_state: dict[str, dict],
        export_settings: dict[str, object],
    ) -> dict[str, object]:
        base_dir = self._get_default_export_dir()
        anchor = self.main.image_files[0] if self.main.image_files else ""
        series_dir = self.main.get_automatic_output_series_dir(base_dir, anchor_path=anchor)
        os.makedirs(series_dir, exist_ok=True)

        total_pages = len(self.main.image_files)
        archive_mode = is_single_archive_mode(export_settings)
        staging_dir = ""
        archive_path = ""
        render_paths = [path for path in requested_paths if path in self.main.image_files]
        expanded_to_all = False

        if archive_mode:
            staging_dir = build_archive_staging_dir(series_dir, "manual_rerender")
            archive_format = str(export_settings.get("resolved_automatic_output_archive_format", "cbz") or "cbz")
            archive_path = os.path.join(
                series_dir,
                build_archive_file_name(self._get_export_bundle_name(), archive_format),
            )
            if scope != "all":
                missing_stage = []
                for page_index, file_path in enumerate(self.main.image_files):
                    stage_path = self._archive_stage_path_for_page(
                        file_path=file_path,
                        page_index=page_index,
                        total_pages=total_pages,
                        export_settings=export_settings,
                        staging_dir=staging_dir,
                    )
                    if not os.path.isfile(stage_path):
                        missing_stage.append(file_path)
                if missing_stage:
                    render_paths = list(self.main.image_files)
                    expanded_to_all = True
            if scope == "all" or expanded_to_all:
                shutil.rmtree(staging_dir, ignore_errors=True)

        temp_main_page_context = None
        if self.main.webtoon_mode:
            temp_main_page_context = type(
                "TempMainPage",
                (object,),
                {
                    "image_files": self.main.image_files,
                    "image_states": all_pages_current_state,
                },
            )()

        rendered_paths: list[str] = []
        output_by_path: dict[str, str] = {}
        rendered_at = datetime.now().isoformat(timespec="seconds")
        for file_path in render_paths:
            page_index = self.main.image_files.index(file_path)
            viewer_state = all_pages_current_state[file_path].get("viewer_state", {})
            output_path = self._render_page_output(
                file_path=file_path,
                page_index=page_index,
                total_pages=total_pages,
                viewer_state=viewer_state,
                temp_main_page_context=temp_main_page_context,
                export_settings=export_settings,
                series_dir=series_dir,
                staging_dir=staging_dir,
            )
            signature = self._render_signature_for_path(file_path, viewer_state, export_settings)
            self.main.mark_render_clean(
                file_path,
                signature=signature,
                rendered_at=rendered_at,
                output_path=output_path,
            )
            self.main.image_ctrl.update_processing_summary(
                file_path,
                {
                    "translated_image_path": output_path,
                    "export_root": series_dir,
                    "render_trigger": scope,
                },
            )
            rendered_paths.append(file_path)
            output_by_path[file_path] = output_path

        if archive_mode:
            archive_format = str(export_settings.get("resolved_automatic_output_archive_format", "cbz") or "cbz")
            compression_level = int(
                export_settings.get("resolved_automatic_output_archive_compression_level", 6) or 6
            )
            make(
                staging_dir,
                output_path=archive_path,
                compresslevel=compression_level,
            )
            for file_path in self.main.image_files:
                state = self.main.image_ctrl.ensure_page_state(file_path)
                state["last_render_output_path"] = archive_path
                summary = state.get("processing_summary", {})
                if isinstance(summary, dict):
                    summary["translated_image_path"] = archive_path
                    summary["export_root"] = series_dir
                    state["processing_summary"] = summary
            output_root = archive_path
        else:
            output_root = series_dir

        return {
            "rendered_paths": rendered_paths,
            "output_by_path": output_by_path,
            "output_root": output_root,
            "series_dir": series_dir,
            "archive_path": archive_path,
            "archive_mode": archive_mode,
            "expanded_to_all": expanded_to_all,
        }

    def _paths_for_rerender_scope(self, scope: str, explicit_paths: list[str] | None = None) -> list[str]:
        if explicit_paths is not None:
            return [path for path in explicit_paths if path in self.main.image_files]
        if scope == "current":
            current_path = self.main._current_image_path()
            return [current_path] if current_path else []
        if scope == "dirty":
            return self.main.render_dirty_paths()
        if scope == "all":
            return list(self.main.image_files)
        return []

    def rerender_output_scope(
        self,
        scope: str,
        *,
        paths: list[str] | None = None,
        quiet: bool = False,
        save_state_first: bool = True,
        post_render_callback=None,
    ) -> bool:
        if not self.main.image_files:
            return False
        if save_state_first:
            try:
                self.main.text_ctrl._commit_pending_text_command()
            except Exception:
                pass
            self.save_current_state()

        requested_paths = self._paths_for_rerender_scope(scope, paths)
        if not requested_paths:
            if not quiet:
                QtWidgets.QMessageBox.information(
                    self.main,
                    self.main.tr("Rerender Output"),
                    self.main.tr("There are no render changes to apply."),
                )
            if post_render_callback:
                post_render_callback()
            return False

        all_pages_current_state = self._build_all_pages_current_state()
        export_settings = self.main.get_resolved_export_settings()
        result_holder: dict[str, object] = {}
        failed = {"value": False}
        self.main.loading.setVisible(True)
        self.main.disable_hbutton_group()

        def on_result(result: dict[str, object]) -> None:
            result_holder.update(result or {})

        def on_error(error_tuple) -> None:
            failed["value"] = True
            self.main.default_error_handler(error_tuple)

        def on_finished() -> None:
            self.main.on_manual_finished()
            self.main.refresh_render_dirty_ui()
            if not failed["value"] and not quiet:
                output_root = str(result_holder.get("output_root", "") or "")
                message = self.main.tr("Render output was updated.")
                if result_holder.get("expanded_to_all"):
                    message += "\n" + self.main.tr(
                        "Archive staging was incomplete, so all pages were rendered before rebuilding the archive."
                    )
                if output_root:
                    message += "\n\n" + output_root
                QtWidgets.QMessageBox.information(
                    self.main,
                    self.main.tr("Rerender Output"),
                    message,
                )
            if post_render_callback and not failed["value"]:
                post_render_callback()

        self.main.run_threaded(
            self._rerender_output_worker,
            on_result,
            on_error,
            on_finished,
            requested_paths,
            scope,
            all_pages_current_state,
            export_settings,
        )
        return True

    def rerender_dirty_page_on_leave(self, file_path: str) -> bool:
        autosave_enabled = bool(
            self.main.settings_page.get_export_settings().get("project_autosave_enabled", False)
        )
        if not autosave_enabled:
            return False
        if file_path not in self.main.image_files:
            return False
        state = self.main.image_ctrl.ensure_page_state(file_path)
        if not state.get("render_dirty", False):
            return False
        return self.rerender_output_scope(
            "current",
            paths=[file_path],
            quiet=True,
            save_state_first=False,
        )

    def _after_manual_project_save(self, post_save_callback=None) -> None:
        current_path = self.main._current_image_path()

        def finish_after_optional_other_pages() -> None:
            other_dirty = self.main.render_dirty_paths(exclude_current=True)
            if other_dirty:
                answer = QtWidgets.QMessageBox.question(
                    self.main,
                    self.main.tr("Rerender Output"),
                    self.main.tr(
                        "변경된 {count}개의 이미지가 렌더링 저장됩니다. 저장하시겠습니까?"
                    ).format(count=len(other_dirty)),
                    QtWidgets.QMessageBox.StandardButton.Save
                    | QtWidgets.QMessageBox.StandardButton.No,
                    QtWidgets.QMessageBox.StandardButton.Save,
                )
                if answer == QtWidgets.QMessageBox.StandardButton.Save:
                    self.rerender_output_scope(
                        "dirty",
                        paths=other_dirty,
                        quiet=True,
                        save_state_first=False,
                        post_render_callback=post_save_callback,
                    )
                    return
            if post_save_callback:
                post_save_callback()

        if (
            current_path
            and current_path in self.main.image_files
            and self.main.image_ctrl.ensure_page_state(current_path).get("render_dirty", False)
        ):
            self.rerender_output_scope(
                "current",
                paths=[current_path],
                quiet=True,
                save_state_first=False,
                post_render_callback=finish_after_optional_other_pages,
            )
            return

        finish_after_optional_other_pages()

    def open_output_folder(self) -> None:
        if not self.main.image_files:
            return
        export_settings = self.main.get_resolved_export_settings()
        base_dir = self._get_default_export_dir()
        anchor = self.main.image_files[0]
        series_dir = self.main.get_automatic_output_series_dir(base_dir, anchor_path=anchor)
        target = series_dir
        if is_single_archive_mode(export_settings):
            archive_format = str(export_settings.get("resolved_automatic_output_archive_format", "cbz") or "cbz")
            archive_path = os.path.join(
                series_dir,
                build_archive_file_name(self._get_export_bundle_name(), archive_format),
            )
            target = archive_path if os.path.isfile(archive_path) else series_dir
        if not os.path.exists(target):
            os.makedirs(series_dir, exist_ok=True)
            target = series_dir
        QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(target))

    def _build_all_pages_current_state(self) -> dict[str, dict]:
        all_pages_current_state: dict[str, dict] = {}

        if self.main.webtoon_mode:
            loaded_pages = self.main.image_viewer.webtoon_manager.loaded_pages
            for page_idx, file_path in enumerate(self.main.image_files):
                if page_idx in loaded_pages:
                    viewer_state = self._create_text_items_state_from_scene(page_idx)
                else:
                    viewer_state = self.main.image_states.get(file_path, {}).get('viewer_state', {}).copy()
                all_pages_current_state[file_path] = {'viewer_state': viewer_state}
            return all_pages_current_state

        for file_path in self.main.image_files:
            viewer_state = self.main.image_states.get(file_path, {}).get('viewer_state', {}).copy()
            all_pages_current_state[file_path] = {'viewer_state': viewer_state}

        return all_pages_current_state

    def _gather_psd_pages(self, all_pages_current_state: dict[str, dict]) -> list[PsdPageData]:
        try:
            if self.main.file_handler.should_pre_materialize(self.main.image_files):
                count = self.main.file_handler.pre_materialize(self.main.image_files)
                logger.info("PSD export pre-materialized %d paths before writing.", count)
        except Exception:
            logger.debug("PSD export pre-materialization failed; continuing lazily.", exc_info=True)

        temp_main_page_context = None
        if self.main.webtoon_mode:
            temp_main_page_context = type(
                "TempMainPage",
                (object,),
                {
                    "image_files": self.main.image_files,
                    "image_states": all_pages_current_state,
                },
            )()

        pages: list[PsdPageData] = []
        for page_idx, file_path in enumerate(self.main.image_files):
            rgb_img = self.main.load_image(file_path)
            viewer_state = copy.deepcopy(all_pages_current_state[file_path].get("viewer_state", {}))

            if self.main.webtoon_mode and temp_main_page_context is not None:
                renderer = ImageSaveRenderer(rgb_img)
                renderer.add_spanning_text_items(viewer_state, page_idx, temp_main_page_context)

            patch_list = copy.deepcopy(self.main.image_patches.get(file_path, []))
            pages.append(
                PsdPageData(
                    file_path=file_path,
                    rgb_image=rgb_img,
                    viewer_state=viewer_state,
                    patches=patch_list,
                )
            )

        return pages

    def _get_export_bundle_name(self) -> str:
        if self.main.project_file:
            return os.path.splitext(os.path.basename(self.main.project_file))[0]
        if self.main.image_files:
            return os.path.splitext(os.path.basename(self.main.image_files[0]))[0]
        return "comic_translate_export"

    def _get_default_export_dir(self) -> str:
        portable_output_dir = str(os.environ.get("COMIC_TRANSLATE_PORTABLE_OUTPUT_DIR", "") or "").strip()
        if portable_output_dir:
            os.makedirs(portable_output_dir, exist_ok=True)
            return portable_output_dir
        if self.main.project_file:
            return os.path.dirname(self.main.project_file)
        if self.main.image_files:
            return os.path.dirname(self.main.image_files[0])
        return os.path.expanduser("~")

    def _create_text_items_state_from_scene(self, page_idx: int) -> dict:
        """
        Create text items state from current scene items for a loaded page in webtoon mode.
        An item "belongs" to a page if its origin point is within that page's vertical bounds.
        """
        
        webtoon_manager = self.main.image_viewer.webtoon_manager
        page_y_start = webtoon_manager.image_positions[page_idx]
        
        # Calculate page bottom boundary
        if page_idx < len(webtoon_manager.image_positions) - 1:
            page_y_end = webtoon_manager.image_positions[page_idx + 1]
        else:
            # For the last page, calculate its end based on its image height
            file_path = self.main.image_files[page_idx]
            rgb_img = self.main.load_image(file_path)
            page_y_end = page_y_start + rgb_img.shape[0]
        
        text_items_data = []
        
        # Find all text items that BELONG to this page
        for item in self.main.image_viewer._scene.items():
            if isinstance(item, TextBlockItem):
                text_item = item
                text_y = text_item.pos().y()
                
                # Check if the text item's origin is on this page
                if text_y >= page_y_start and text_y < page_y_end:
                    # Convert to page-local coordinates
                    scene_pos = text_item.pos()
                    page_local_x = scene_pos.x()
                    page_local_y = scene_pos.y() - page_y_start
                    
                    # Use TextItemProperties for consistent serialization
                    text_props = TextItemProperties.from_text_item(text_item)
                    # Override position to use page-local coordinates
                    text_props.position = (page_local_x, page_local_y)
                    if text_props.source_rect is not None:
                        source_x, source_y, width, height = text_props.source_rect
                        source_page_local = webtoon_manager.coordinate_converter.scene_to_page_local_position(
                            QPointF(source_x, source_y),
                            page_idx,
                        )
                        text_props.source_rect = (
                            source_page_local.x(),
                            source_page_local.y(),
                            width,
                            height,
                        )
                    if text_props.block_anchor is not None:
                        anchor_x, anchor_y, width, height = text_props.block_anchor
                        anchor_page_local = webtoon_manager.coordinate_converter.scene_to_page_local_position(
                            QPointF(anchor_x, anchor_y),
                            page_idx,
                        )
                        text_props.block_anchor = (
                            anchor_page_local.x(),
                            anchor_page_local.y(),
                            width,
                            height,
                        )
                    
                    text_items_data.append(text_props.to_dict())
        
        # Return viewer state with the collected text items
        return {
            'text_items_state': text_items_data,
            'push_to_stack': False  # Don't push to undo stack during save
        }

    def launch_save_proj_dialog(self):
        kind = self._current_project_kind()
        file_dialog = QtWidgets.QFileDialog()
        file_name, _ = file_dialog.getSaveFileName(
            self.main,
            self.main.tr("Save Series Project As") if kind == PROJECT_KIND_SERIES else self.main.tr("Save Project As"),
            f"untitled{self._project_extension(kind)}",
            project_file_filter_for_kind(kind),
        )

        return file_name

    def run_save_proj(self, file_name, post_save_callback=None):
        prev_project_file = self.main.project_file
        prev_window_title = self.main.windowTitle()
        self.main.project_file = file_name
        self.main.setWindowTitle(f"{os.path.basename(file_name)}[*]")
        self.main.loading.setVisible(True)
        self.main.disable_hbutton_group()
        save_failed = {'value': False}
        save_start_revision = self.main._dirty_revision

        def on_error(error_tuple):
            save_failed['value'] = True
            self.main.project_file = prev_project_file
            self.main.setWindowTitle(prev_window_title)
            self.main.default_error_handler(error_tuple)

        def on_finished():
            self.main.on_manual_finished()
            if not save_failed['value']:
                # Close the old project's DB connection only after the save
                # has completed, so that lazy blobs can be read from it.
                if prev_project_file and prev_project_file != file_name:
                    close_state_store(prev_project_file)
                if self.main._dirty_revision == save_start_revision:
                    self.main.set_project_clean()
                self.clear_recovery_checkpoint()
                self.add_recent_project(file_name)
                self._refresh_home_screen()
                if post_save_callback:
                    post_save_callback()

        self.main.run_threaded(self.save_project, None, on_error, on_finished, file_name)
        
    def save_current_state(self):
        if self.main.webtoon_mode:
            webtoon_manager = self.main.image_viewer.webtoon_manager
            webtoon_manager.scene_item_manager.save_all_scene_items_to_states()
            webtoon_manager.save_view_state()
        else:
            self.main.image_ctrl.save_current_image_state()

    def thread_save_project(self, post_save_callback=None) -> bool:
        if self._current_project_kind() == PROJECT_KIND_SERIES:
            return bool(self.main.series_ctrl.thread_save_series(post_save_callback=post_save_callback))
        file_name = ""
        self.save_current_state()
        if self.main.project_file:
            file_name = self.main.project_file
        else:
            file_name = self.launch_save_proj_dialog()

        if file_name:
            self.run_save_proj(
                file_name,
                lambda: self._after_manual_project_save(post_save_callback),
            )
            return True
        return False

    def thread_save_as_project(self, post_save_callback=None) -> bool:
        if self._current_project_kind() == PROJECT_KIND_SERIES:
            file_name = self.launch_save_proj_dialog()
            if file_name:
                return bool(self.main.series_ctrl.thread_save_series(file_name, post_save_callback=post_save_callback))
            return False
        file_name = self.launch_save_proj_dialog()
        if file_name:
            self.save_current_state()
            self.run_save_proj(
                file_name,
                lambda: self._after_manual_project_save(post_save_callback),
            )
            return True
        return False

    def thread_change_project_file(self, target_path: str) -> bool:
        if self._current_project_kind() == PROJECT_KIND_SERIES:
            current_path = (
                os.path.normpath(os.path.abspath(self.main.series_ctrl.series_file))
                if self.main.series_ctrl.series_file
                else None
            )
            target_path = ensure_project_extension(target_path, SERIES_PROJECT_FILE_EXT)
            if current_path and target_path == current_path:
                return self.thread_save_project()
            if os.path.exists(target_path) and target_path != current_path:
                overwrite = QtWidgets.QMessageBox.question(
                    self.main,
                    self.main.tr("Overwrite Project File"),
                    self.main.tr(
                        "A project file already exists at this location.\n\n{path}\n\nOverwrite it?"
                    ).format(path=target_path),
                    QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
                    QtWidgets.QMessageBox.StandardButton.No,
                )
                if overwrite != QtWidgets.QMessageBox.StandardButton.Yes:
                    return False

            def _post_save() -> None:
                if current_path and current_path != target_path and os.path.isfile(current_path):
                    try:
                        os.remove(current_path)
                    except OSError as exc:
                        QtWidgets.QMessageBox.warning(
                            self.main,
                            self.main.tr("Old Project File Kept"),
                            self.main.tr(
                                "The project was saved to the new location, but the old file could not be removed.\n\n{path}\n\n{error}"
                            ).format(path=current_path, error=str(exc)),
                        )
                if current_path and current_path != target_path:
                    self.remove_recent_project(current_path)
                self.add_recent_project(target_path)
                self._refresh_home_screen()
                action_text = (
                    self.main.tr("Project file saved.")
                    if not current_path
                    else self.main.tr("Project file renamed.")
                    if os.path.dirname(current_path) == os.path.dirname(target_path)
                    else self.main.tr("Project file moved.")
                )
                QtWidgets.QMessageBox.information(
                    self.main,
                    self.main.tr("Project File"),
                    action_text,
                )

            return bool(self.main.series_ctrl.thread_save_series(target_path, post_save_callback=_post_save))

        target_path = os.path.normpath(os.path.abspath(os.path.expanduser(target_path or "")))
        if not target_path:
            return False
        if not target_path.lower().endswith(".ctpr"):
            target_path = f"{target_path}.ctpr"

        current_path = (
            os.path.normpath(os.path.abspath(self.main.project_file))
            if self.main.project_file
            else None
        )

        target_dir = os.path.dirname(target_path)
        if not target_dir:
            QtWidgets.QMessageBox.warning(
                self.main,
                self.main.tr("Project File"),
                self.main.tr("Choose an existing folder for the project file."),
            )
            return False

        try:
            os.makedirs(target_dir, exist_ok=True)
        except OSError as exc:
            QtWidgets.QMessageBox.warning(
                self.main,
                self.main.tr("Project File"),
                self.main.tr(
                    "Could not create the selected project folder.\n\n{error}"
                ).format(error=str(exc)),
            )
            return False

        if current_path and target_path == current_path:
            return self.thread_save_project()

        if os.path.exists(target_path) and target_path != current_path:
            overwrite = QtWidgets.QMessageBox.question(
                self.main,
                self.main.tr("Overwrite Project File"),
                self.main.tr(
                    "A project file already exists at this location.\n\n{path}\n\nOverwrite it?"
                ).format(path=target_path),
                QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
                QtWidgets.QMessageBox.StandardButton.No,
            )
            if overwrite != QtWidgets.QMessageBox.StandardButton.Yes:
                return False

        self.save_current_state()

        def _post_save() -> None:
            if current_path and current_path != target_path:
                if os.path.isfile(current_path):
                    try:
                        os.remove(current_path)
                    except OSError as exc:
                        QtWidgets.QMessageBox.warning(
                            self.main,
                            self.main.tr("Old Project File Kept"),
                            self.main.tr(
                                "The project was saved to the new location, but the old file could not be removed.\n\n{path}\n\n{error}"
                            ).format(path=current_path, error=str(exc)),
                        )
                self.remove_recent_project(current_path)

            self.add_recent_project(target_path)
            self._refresh_home_screen()

            action_text = (
                self.main.tr("Project file saved.")
                if not current_path
                else self.main.tr("Project file renamed.")
                if os.path.dirname(current_path) == os.path.dirname(target_path)
                else self.main.tr("Project file moved.")
            )
            QtWidgets.QMessageBox.information(
                self.main,
                self.main.tr("Project File"),
                action_text,
            )

        self.run_save_proj(target_path, post_save_callback=_post_save)
        return True

    def save_project(self, file_name):
        if self._current_project_kind() == PROJECT_KIND_SERIES:
            raise RuntimeError("Series projects must be saved through SeriesController.")
        save_state_to_proj_file(self.main, file_name)

    def update_ui_from_project(self):
        if not self.main.image_files:
            self.main.curr_img_idx = -1
            self.main.central_stack.setCurrentWidget(self.main.drag_browser)
            self.main.batch_report_ctrl.refresh_action_buttons()
            return

        index = self.main.curr_img_idx
        if not (0 <= index < len(self.main.image_files)):
            index = 0
            self.main.curr_img_idx = 0
        self.main.image_ctrl.update_image_cards()

        # highlight the row that matches the current image
        self.main.page_list.blockSignals(True)
        if 0 <= index < self.main.page_list.count():
            self.main.page_list.setCurrentRow(index)
            self.main.image_ctrl.highlight_card(index)
        self.main.page_list.blockSignals(False)

        for file in self.main.image_files:
            self.main.create_undo_stack_for_path(file)

        self.main.run_threaded(
            lambda: self.main.load_image(self.main.image_files[index]),
            lambda result: self._display_image_and_set_mode(result, index),
            self.main.default_error_handler
        )

    def _display_image_and_set_mode(self, rgb_image, index: int):
        """Display the image and then set the appropriate mode."""
        # First display the image normally
        self.main.image_ctrl.display_image_from_loaded(rgb_image, index, switch_page=False)
        
        # Now that the UI is ready, activate webtoon mode
        if self.main.webtoon_mode:
            self.main.webtoon_toggle.setChecked(True)
            self.main.webtoon_ctrl.switch_to_webtoon_mode()
        self.main.set_project_clean()
        self.main.batch_report_ctrl.refresh_action_buttons()

    def _refresh_home_screen(self) -> None:
        """Repopulate the home screen recent list if it is currently visible."""
        home = self.main.startup_home
        if home is None:
            return
        home.populate(self.get_recent_projects())

    def thread_load_project(self, file_name: str, clear_recovery: bool = True):
        normalized_path = os.path.normpath(os.path.abspath(file_name))
        if not os.path.isfile(normalized_path):
            self.remove_recent_project(normalized_path)
            self._refresh_home_screen()
            self.main.setWindowTitle("Project1.ctpr[*]")
            self.main.project_file = None
            self.main.project_kind = PROJECT_KIND_SINGLE
            QtWidgets.QMessageBox.warning(
                self.main,
                self.main.tr("Project Not Found"),
                self.main.tr(
                    "The selected project file could not be found.\n"
                    "It may have been moved, renamed, or deleted.\n\n{path}"
                ).format(path=normalized_path),
            )
            return

        prev_project_file = self.main.project_file
        if prev_project_file and prev_project_file != normalized_path:
            close_state_store(prev_project_file)
        if clear_recovery:
            self.clear_recovery_checkpoint()
        self.main.image_ctrl.clear_state()
        self.main.project_kind = PROJECT_KIND_SINGLE
        self.main.setWindowTitle(f"{os.path.basename(normalized_path)}[*]")

        def _on_load_finished():
            self.add_recent_project(normalized_path)
            self._refresh_home_screen()
            self.update_ui_from_project()

        def _on_load_error(error_tuple):
            self.main.default_error_handler(error_tuple)
            exctype, value, _ = error_tuple
            self.main.project_file = None
            self.main.project_kind = PROJECT_KIND_SINGLE
            self.main.setWindowTitle("Project1.ctpr[*]")
            if exctype is FileNotFoundError or isinstance(value, FileNotFoundError):
                self.remove_recent_project(normalized_path)
                self._refresh_home_screen()

        self.main.run_threaded(
            self.load_project,
            self.load_state_to_ui,
            _on_load_error,
            _on_load_finished,
            normalized_path
        )

    def load_project(self, file_name):
        if not os.path.isfile(file_name):
            raise FileNotFoundError(file_name)
        self.main.project_file = file_name
        self.main.project_kind = PROJECT_KIND_SINGLE
        return load_state_from_proj_file(self.main, file_name)
    
    def load_state_to_ui(self, saved_ctx: str):
        self.main.settings_page.ui.extra_context.setPlainText(saved_ctx)

    def save_main_page_settings(self):
        settings = QSettings("ComicLabs", "ComicTranslate")

        self.process_group('text_rendering', self.main.render_settings(), settings)

        settings.beginGroup("main_page")
        # Save languages in English
        settings.setValue("source_language", self.main.lang_mapping[self.main.s_combo.currentText()])
        settings.setValue("target_language", self.main.lang_mapping[self.main.t_combo.currentText()])

        settings.setValue("mode", "manual" if self.main.manual_radio.isChecked() else "automatic")

        # Save brush and eraser sizes
        settings.setValue("brush_size", self.main.image_viewer.brush_size)
        settings.setValue("eraser_size", self.main.image_viewer.eraser_size)

        settings.endGroup()

        # Save window state
        settings.beginGroup("MainWindow")
        settings.setValue("geometry", self.main.saveGeometry())
        settings.setValue("state", self.main.saveState())
        settings.endGroup()

    def load_main_page_settings(self):
        settings = QSettings("ComicLabs", "ComicTranslate")
        settings.beginGroup("main_page")

        # Load languages and convert back to current language
        source_lang = settings.value("source_language", "Korean")
        target_lang = settings.value("target_language", "English")

        # Use reverse mapping to get the translated language names
        self.main.s_combo.setCurrentText(self.main.reverse_lang_mapping.get(source_lang, self.main.tr("Korean")))
        self.main.t_combo.setCurrentText(self.main.reverse_lang_mapping.get(target_lang, self.main.tr("English")))

        mode = settings.value("mode", "manual")
        if mode == "manual":
            self.main.manual_radio.setChecked(True)
            self.main.manual_mode_selected()
        else:
            self.main.automatic_radio.setChecked(True)
            self.main.batch_mode_selected()

        # Load brush and eraser sizes
        brush_size = int(settings.value("brush_size", 10))  # Default value is 10
        eraser_size = int(settings.value("eraser_size", 20))  # Default value is 20
        self.main.image_viewer.brush_size = brush_size
        self.main.image_viewer.eraser_size = eraser_size

        settings.endGroup()

        # Load window state
        settings.beginGroup("MainWindow")
        geometry = settings.value("geometry")
        state = settings.value("state")
        if geometry is not None:
            self.main.restoreGeometry(geometry)
        if state is not None:
            self.main.restoreState(state)
        settings.endGroup()

        # Load text rendering settings
        settings.beginGroup('text_rendering')
        alignment = settings.value('alignment_id', 1, type=int) # Default value is 1 which is Center
        self.main.alignment_tool_group.set_dayu_checked(alignment)
        vertical_alignment = settings.value('vertical_alignment_id', 0, type=int)
        self.main.vertical_alignment_tool_group.set_dayu_checked(vertical_alignment)

        saved_font_family = settings.value('font_family', '')
        if saved_font_family:
            self.main.set_font(saved_font_family)
        else:
            self.main.font_dropdown.setCurrentText('')
        min_font_size = settings.value('min_font_size', 5)  # Default value is 5
        max_font_size = settings.value('max_font_size', 40) # Default value is 40
        self.main.settings_page.ui.min_font_spinbox.setValue(int(min_font_size))
        self.main.settings_page.ui.max_font_spinbox.setValue(int(max_font_size))

        color = settings.value('color', '#000000')
        self.main.block_font_color_button.setStyleSheet(f"background-color: {color}; border: none; border-radius: 5px;")
        self.main.block_font_color_button.setProperty('selected_color', color)
        force_font_color = settings.value('force_font_color', False, type=bool)
        smart_global_apply_all = settings.value('smart_global_apply_all', False, type=bool)
        self.main.force_font_color_checkbox.setChecked(
            bool(force_font_color or smart_global_apply_all)
        )
        self.main.smart_global_apply_all_checkbox.setChecked(False)
        self.main.settings_page.ui.uppercase_checkbox.setChecked(settings.value('upper_case', False, type=bool))
        self.main.outline_checkbox.setChecked(settings.value('outline', True, type=bool))
        self.main.outline_mode_group.set_dayu_checked(
            1 if self.main.outline_checkbox.isChecked() else 0
        )
        self.main.outline_font_color_button.setEnabled(self.main.outline_checkbox.isChecked())
        self.main.outline_width_dropdown.setEnabled(self.main.outline_checkbox.isChecked())

        self.main.line_spacing_dropdown.setCurrentText(settings.value('line_spacing', '1.0'))
        self.main.outline_width_dropdown.setCurrentText(settings.value('outline_width', '1.0'))
        outline_color = settings.value('outline_color', '#FFFFFF')
        self.main.outline_font_color_button.setStyleSheet(f"background-color: {outline_color}; border: none; border-radius: 5px;")
        self.main.outline_font_color_button.setProperty('selected_color', outline_color)

        self.main.bold_button.setChecked(settings.value('bold', False, type=bool))
        self.main.italic_button.setChecked(settings.value('italic', False, type=bool))
        self.main.underline_button.setChecked(settings.value('underline', False, type=bool))
        settings.endGroup()

    def process_group(self, group_key, group_value, settings_obj: QSettings):
        """Helper function to process a group and its nested values."""
        if is_dataclass(group_value):
            group_value = asdict(group_value)
        if isinstance(group_value, dict):
            settings_obj.beginGroup(group_key)
            for sub_key, sub_value in group_value.items():
                self.process_group(sub_key, sub_value, settings_obj)
            settings_obj.endGroup()
        else:
            # Convert value to English using mappings if available
            mapped_value = self.main.settings_page.ui.value_mappings.get(group_value, group_value)
            settings_obj.setValue(group_key, mapped_value)

from __future__ import annotations

import os
import shutil
import tempfile
from typing import TYPE_CHECKING, Callable

from PySide6 import QtCore, QtWidgets

from app.projects.project_state import (
    close_state_store,
    load_state_from_proj_file,
    save_state_to_proj_file,
)
from app.projects.project_types import (
    PROJECT_KIND_SERIES,
    PROJECT_KIND_SINGLE,
    SERIES_PROJECT_FILE_EXT,
    ensure_project_extension,
)
from app.projects.series_state_v1 import (
    add_series_paths,
    build_series_item_from_path,
    create_series_project,
    load_series_project,
    materialize_series_child_project,
    normalize_series_global_settings,
    normalize_series_settings,
    remove_series_item,
    save_series_manifest,
    scan_series_source_files,
    update_series_child_from_file,
    update_series_global_settings,
    update_series_item_status,
    update_series_items_order,
    update_series_navigation_history,
    update_series_queue_runtime,
    update_series_settings,
)
from app.ui.series_import_dialog import SeriesImportDialog
from app.ui.settings.series_page import SeriesPage
from app.ui.messages import Messages

if TYPE_CHECKING:
    from controller import ComicTranslate


class SeriesController(QtCore.QObject):
    def __init__(self, main: "ComicTranslate"):
        super().__init__(main)
        self.main = main
        self.series_file: str | None = None
        self.series_manifest: dict[str, object] = {}
        self.series_items: list[dict[str, object]] = []
        self.active_child_item_id: str | None = None
        self.active_child_project_path: str | None = None
        self.active_child_temp_dir: str | None = None
        self.history_back: list[dict[str, object]] = []
        self.history_forward: list[dict[str, object]] = []
        self._queue_active = False
        self._queue_pending_ids: list[str] = []
        self._queue_completed_ids: list[str] = []
        self._queue_retry_remaining: dict[str, int] = {}

    def has_series_loaded(self) -> bool:
        return bool(self.series_file)

    def is_child_project_active(self) -> bool:
        return bool(self.series_file and self.active_child_item_id and self.active_child_project_path)

    def is_series_board_active(self) -> bool:
        return bool(self.series_file and not self.active_child_item_id)

    def is_queue_running(self) -> bool:
        return bool(self._queue_active)

    def reset_series_context(self) -> None:
        self._clear_active_child_materialization()
        self.series_file = None
        self.series_manifest = {}
        self.series_items = []
        self.history_back = []
        self.history_forward = []
        self._queue_active = False
        self._queue_pending_ids = []
        self._queue_completed_ids = []
        self._queue_retry_remaining = {}

    def _current_view_state(self) -> dict[str, object]:
        if self.is_child_project_active():
            return {
                "kind": "child",
                "item_id": self.active_child_item_id,
            }
        return {"kind": "board"}

    def _push_history(self) -> None:
        state = self._current_view_state()
        if self.history_back and self.history_back[-1] == state:
            return
        self.history_back.append(state)
        self.history_forward.clear()
        self._persist_navigation_history()

    def _persist_navigation_history(self) -> None:
        if not self.series_file:
            return
        self.series_manifest = update_series_navigation_history(
            self.series_file,
            back=list(self.history_back),
            forward=list(self.history_forward),
        )
        self._refresh_workspace_navigation()

    def _refresh_workspace_navigation(self) -> None:
        if hasattr(self.main, "series_workspace") and self.main.series_workspace is not None:
            self.main.series_workspace.set_navigation_state(
                can_back=bool(self.history_back),
                can_forward=bool(self.history_forward),
            )

    def _current_series_display_name(self) -> str:
        return os.path.basename(self.series_file or "")

    def _set_series_window_title(self, child_name: str | None = None) -> None:
        series_name = self._current_series_display_name() or f"Series{SERIES_PROJECT_FILE_EXT}"
        if child_name:
            self.main.setWindowTitle(
                self.main.tr("Child Project - {child} · {series}[*]").format(
                    child=child_name,
                    series=series_name,
                )
            )
        else:
            self.main.setWindowTitle(
                self.main.tr("Series Project - {series}[*]").format(series=series_name)
            )

    def _series_global_settings_from_main(self) -> dict[str, object]:
        source_label = self.main.s_combo.currentText()
        target_label = self.main.t_combo.currentText()
        translator_display = self.main.settings_page.ui.translator_combo.currentText()
        translator_value = self.main.settings_page.ui.value_mappings.get(
            translator_display,
            translator_display,
        )
        return normalize_series_global_settings(
            {
                "source_language": self.main.lang_mapping.get(source_label, source_label),
                "target_language": self.main.lang_mapping.get(target_label, target_label),
                "ocr": self.main.settings_page.get_tool_selection("ocr"),
                "translator": translator_value,
                "workflow_mode": self.main.settings_page.get_workflow_mode(),
                "use_gpu": self.main.settings_page.is_gpu_enabled(),
            }
        )

    def _series_workspace_options(self) -> dict[str, list[tuple[str, str]]]:
        language_options = [
            (canonical, display)
            for display, canonical in self.main.lang_mapping.items()
        ]
        translator_options = []
        translator_combo = self.main.settings_page.ui.translator_combo
        for index in range(translator_combo.count()):
            label = translator_combo.itemText(index)
            translator_options.append(
                (
                    str(self.main.settings_page.ui.value_mappings.get(label, label)),
                    label,
                )
            )
        ocr_options = []
        ocr_combo = self.main.settings_page.ui.ocr_combo
        for index in range(ocr_combo.count()):
            ocr_options.append(
                (
                    str(ocr_combo.itemData(index) or ""),
                    ocr_combo.itemText(index),
                )
            )
        workflow_options = []
        workflow_combo = self.main.settings_page.ui.workflow_mode_combo
        for index in range(workflow_combo.count()):
            workflow_options.append(
                (
                    str(workflow_combo.itemData(index) or ""),
                    workflow_combo.itemText(index),
                )
            )
        return {
            "languages": language_options,
            "translators": translator_options,
            "ocr_modes": ocr_options,
            "workflow_modes": workflow_options,
        }

    def _apply_workspace_state(self) -> None:
        if not self.series_file:
            return
        queue_runtime = self.series_manifest.get("series_queue_runtime") or {}
        self.main.series_workspace.configure_options(**self._series_workspace_options())
        self.main.series_workspace.set_global_settings(
            normalize_series_global_settings(self.series_manifest.get("global_settings"))
        )
        self.main.series_workspace.set_series_state(
            series_file=self._current_series_display_name(),
            items=list(self.series_items),
            queue_running=self._queue_active,
            active_item_id=str(queue_runtime.get("active_item_id") or ""),
        )
        self._refresh_workspace_navigation()

    def _queue_change_locked(self) -> bool:
        return bool(self._queue_active and self.series_file)

    def _show_queue_locked_message(self) -> None:
        Messages.show_info(
            self.main,
            self.main.tr(
                "Queue changes are locked while automatic translation is running."
            ),
            duration=5,
            closable=True,
            source="series",
        )

    def _clear_active_child_materialization(self) -> None:
        self.active_child_item_id = None
        self.active_child_project_path = None
        if self.active_child_temp_dir and os.path.isdir(self.active_child_temp_dir):
            shutil.rmtree(self.active_child_temp_dir, ignore_errors=True)
        self.active_child_temp_dir = None

    def _load_series_worker(self, file_name: str) -> dict[str, object]:
        return load_series_project(file_name)

    def thread_load_series_project(self, file_name: str) -> None:
        normalized_path = os.path.normpath(os.path.abspath(file_name or ""))
        if not os.path.isfile(normalized_path):
            self.main.project_ctrl.remove_recent_project(normalized_path)
            self.main.project_ctrl._refresh_home_screen()
            QtWidgets.QMessageBox.warning(
                self.main,
                self.main.tr("Project Not Found"),
                self.main.tr(
                    "The selected series project file could not be found.\n"
                    "It may have been moved, renamed, or deleted.\n\n{path}"
                ).format(path=normalized_path),
            )
            return

        self.main.loading.setVisible(True)
        previous_project = getattr(self.main, "project_file", None)
        if isinstance(previous_project, str) and previous_project and previous_project != normalized_path:
            close_state_store(previous_project)

        def on_result(state: dict[str, object]) -> None:
            self.main.image_ctrl.clear_state()
            self._clear_active_child_materialization()
            self.series_file = normalized_path
            self.series_manifest = dict(state.get("manifest") or {})
            self.series_items = list(state.get("items") or [])
            nav = self.series_manifest.get("series_navigation_history") or {}
            self.history_back = list(nav.get("back") or [])
            self.history_forward = list(nav.get("forward") or [])
            self.main.project_file = normalized_path
            self.main.project_kind = PROJECT_KIND_SERIES
            self._apply_workspace_state()
            self.main.show_series_page()
            self.main.set_project_clean()
            self._set_series_window_title()

        def on_error(error_tuple) -> None:
            self.main.loading.setVisible(False)
            self.main.default_error_handler(error_tuple)

        def on_finished() -> None:
            self.main.loading.setVisible(False)
            self.main.project_ctrl.add_recent_project(normalized_path)
            self.main.project_ctrl._refresh_home_screen()

        self.main.run_threaded(
            self._load_series_worker,
            on_result,
            on_error,
            on_finished,
            normalized_path,
        )

    def _build_series_project_worker(
        self,
        file_name: str,
        root_dir: str,
        selected_paths: list[str],
        global_settings: dict[str, object],
        series_settings: dict[str, object],
    ) -> dict[str, object]:
        items = []
        embedded_projects = []
        source_lang = str(global_settings.get("source_language") or "Japanese")
        target_lang = str(global_settings.get("target_language") or "English")
        for index, path in enumerate(selected_paths, start=1):
            item, project = build_series_item_from_path(
                path,
                root_dir=root_dir,
                queue_index=index,
                source_lang=source_lang,
                target_lang=target_lang,
            )
            items.append(item)
            embedded_projects.append(project)
        create_series_project(
            file_name,
            root_dir=root_dir,
            items=items,
            embedded_projects=embedded_projects,
            global_settings=global_settings,
            series_settings=series_settings,
        )
        return load_series_project(file_name)

    def prompt_new_series_project(self) -> None:
        root_dir = QtWidgets.QFileDialog.getExistingDirectory(
            self.main,
            self.main.tr("Select Series Root Folder"),
            os.path.expanduser("~"),
        )
        if not root_dir:
            return

        self.main.loading.setVisible(True)

        def on_result(paths: list[str]) -> None:
            self.main.loading.setVisible(False)
            if not paths:
                QtWidgets.QMessageBox.information(
                    self.main,
                    self.main.tr("Create Series Project"),
                    self.main.tr("No supported files were found in the selected folder."),
                )
                return

            dialog = SeriesImportDialog(root_dir, paths, self.main)
            if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
                return
            selected_paths = dialog.selected_paths()
            if not selected_paths:
                return

            default_name = os.path.basename(os.path.normpath(root_dir)) or "series"
            file_name, _ = QtWidgets.QFileDialog.getSaveFileName(
                self.main,
                self.main.tr("Save Series Project As"),
                os.path.join(root_dir, f"{default_name}{SERIES_PROJECT_FILE_EXT}"),
                self.main.tr("Series Project Files (*.seriesctpr);;All Files (*)"),
            )
            if not file_name:
                return
            target_path = ensure_project_extension(file_name, SERIES_PROJECT_FILE_EXT)
            global_settings = self._series_global_settings_from_main()
            series_settings = self.main.settings_page.get_series_settings()
            self.main.loading.setVisible(True)

            self.main.run_threaded(
                self._build_series_project_worker,
                lambda state: self._apply_new_series_result(target_path, state),
                self.main.default_error_handler,
                lambda: self.main.loading.setVisible(False),
                target_path,
                root_dir,
                selected_paths,
                global_settings,
                series_settings,
            )

        def on_error(error_tuple) -> None:
            self.main.loading.setVisible(False)
            self.main.default_error_handler(error_tuple)

        self.main.run_threaded(
            scan_series_source_files,
            on_result,
            on_error,
            None,
            root_dir,
        )

    def _apply_new_series_result(self, file_name: str, state: dict[str, object]) -> None:
        self.main.image_ctrl.clear_state()
        self._clear_active_child_materialization()
        self.series_file = os.path.normpath(os.path.abspath(file_name))
        self.series_manifest = dict(state.get("manifest") or {})
        self.series_items = list(state.get("items") or [])
        self.history_back = []
        self.history_forward = []
        self.main.project_file = self.series_file
        self.main.project_kind = PROJECT_KIND_SERIES
        self._apply_workspace_state()
        self.main.show_series_page()
        self.main.project_ctrl.add_recent_project(self.series_file)
        self.main.project_ctrl._refresh_home_screen()
        self.main.set_project_clean()
        self._set_series_window_title()

    def _find_item(self, item_id: str) -> dict[str, object] | None:
        for item in self.series_items:
            if str(item.get("series_item_id")) == str(item_id):
                return item
        return None

    def _active_child_display_name(self) -> str | None:
        if not self.active_child_item_id:
            return None
        item = self._find_item(str(self.active_child_item_id))
        if item is None:
            return None
        return str(item.get("display_name") or "").strip() or None

    def _apply_series_globals_to_main(self) -> None:
        global_settings = normalize_series_global_settings(self.series_manifest.get("global_settings"))
        source_lang = global_settings.get("source_language")
        target_lang = global_settings.get("target_language")
        if source_lang:
            self.main.s_combo.setCurrentText(
                self.main.reverse_lang_mapping.get(str(source_lang), str(source_lang))
            )
        if target_lang:
            self.main.t_combo.setCurrentText(
                self.main.reverse_lang_mapping.get(str(target_lang), str(target_lang))
            )
        if global_settings.get("ocr"):
            self.main.settings_page._set_ocr_mode(str(global_settings["ocr"]))
        if global_settings.get("workflow_mode"):
            self.main.settings_page._set_workflow_mode(str(global_settings["workflow_mode"]))
        if global_settings.get("translator"):
            translator_value = str(global_settings["translator"])
            translator_label = self.main.settings_page.ui.reverse_mappings.get(
                translator_value,
                translator_value,
            )
            index = self.main.settings_page.ui.translator_combo.findText(translator_label)
            if index >= 0:
                self.main.settings_page.ui.translator_combo.setCurrentIndex(index)
        self.main.settings_page.ui.use_gpu_checkbox.setChecked(bool(global_settings.get("use_gpu", True)))

    def _open_child_worker(self, child_project_path: str) -> str:
        return load_state_from_proj_file(self.main, child_project_path)

    def request_open_item(self, item_id: str) -> None:
        if self._queue_change_locked():
            self._show_queue_locked_message()
            return
        self.main._run_guarded_project_transition(
            lambda: self._open_item(item_id, push_history=True)
        )

    def _open_item(
        self,
        item_id: str,
        *,
        push_history: bool,
        after_loaded: Callable[[], None] | None = None,
    ) -> None:
        if not self.series_file:
            return
        item = self._find_item(item_id)
        if item is None:
            return
        if push_history:
            self._push_history()

        work_dir = tempfile.mkdtemp(prefix="series_child_", dir=self.main.temp_dir)
        child_project_path = materialize_series_child_project(
            self.series_file,
            item,
            temp_dir=work_dir,
        )
        old_temp_dir = self.active_child_temp_dir
        self.main.image_ctrl.clear_state()
        self.main.loading.setVisible(True)

        def on_result(saved_ctx: str) -> None:
            self.main.project_ctrl.load_state_to_ui(saved_ctx)

        def on_error(error_tuple) -> None:
            self.main.loading.setVisible(False)
            self.main.default_error_handler(error_tuple)
            shutil.rmtree(work_dir, ignore_errors=True)

        def on_finished() -> None:
            self.main.loading.setVisible(False)
            if old_temp_dir and old_temp_dir != work_dir:
                shutil.rmtree(old_temp_dir, ignore_errors=True)
            self.active_child_item_id = str(item_id)
            self.active_child_project_path = child_project_path
            self.active_child_temp_dir = work_dir
            self.main.project_file = self.series_file
            self.main.project_kind = PROJECT_KIND_SERIES
            self.main.show_main_page()
            self.main.project_ctrl.update_ui_from_project()
            self._set_series_window_title(str(item.get("display_name") or ""))
            if after_loaded is not None:
                after_loaded()

        self.main.run_threaded(
            self._open_child_worker,
            on_result,
            on_error,
            on_finished,
            child_project_path,
        )

    def request_show_board(self) -> None:
        if self._queue_change_locked():
            self._show_queue_locked_message()
            return
        self.main._run_guarded_project_transition(
            lambda: self._show_board(push_history=True)
        )

    def _show_board(self, *, push_history: bool) -> None:
        if not self.series_file:
            return
        if push_history:
            self._push_history()
        self.main.image_ctrl.clear_state()
        self._clear_active_child_materialization()
        self.main.project_file = self.series_file
        self.main.project_kind = PROJECT_KIND_SERIES
        self._apply_workspace_state()
        self.main.show_series_page()
        self.main.set_project_clean()
        self._set_series_window_title()

    def request_back(self) -> None:
        if self._queue_change_locked():
            self._show_queue_locked_message()
            return
        if not self.history_back:
            return
        self.main._run_guarded_project_transition(self._navigate_back)

    def request_forward(self) -> None:
        if self._queue_change_locked():
            self._show_queue_locked_message()
            return
        if not self.history_forward:
            return
        self.main._run_guarded_project_transition(self._navigate_forward)

    def _navigate_back(self) -> None:
        if not self.history_back:
            return
        current = self._current_view_state()
        target = self.history_back.pop()
        self.history_forward.append(current)
        self._persist_navigation_history()
        self._restore_view_state(target)

    def _navigate_forward(self) -> None:
        if not self.history_forward:
            return
        current = self._current_view_state()
        target = self.history_forward.pop()
        self.history_back.append(current)
        self._persist_navigation_history()
        self._restore_view_state(target)

    def request_tree_jump(self) -> None:
        if self._queue_change_locked():
            self._show_queue_locked_message()
            return
        if not self.series_file:
            return
        target = self.main.series_workspace.prompt_tree_jump(self.series_items)
        if not target:
            return
        self.main._run_guarded_project_transition(
            lambda: self._restore_view_state(
                {"kind": "board"} if target == "__board__" else {"kind": "child", "item_id": target},
                push_history=True,
            )
        )

    def _restore_view_state(self, state: dict[str, object], push_history: bool = False) -> None:
        kind = str(state.get("kind") or "board")
        if push_history:
            self._push_history()
        if kind == "child" and state.get("item_id"):
            self._open_item(str(state["item_id"]), push_history=False)
            return
        self._show_board(push_history=False)

    def request_remove_item(self, item_id: str) -> None:
        if self._queue_change_locked():
            self._show_queue_locked_message()
            return
        if not self.series_file:
            return
        item = self._find_item(item_id)
        if item is None:
            return
        answer = QtWidgets.QMessageBox.question(
            self.main,
            self.main.tr("Remove From Series"),
            self.main.tr(
                "Remove '{name}' from this series project?"
            ).format(name=str(item.get("display_name") or "")),
            QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
            QtWidgets.QMessageBox.StandardButton.No,
        )
        if answer != QtWidgets.QMessageBox.StandardButton.Yes:
            return
        self.series_items = remove_series_item(self.series_file, item_id)
        self.series_manifest = load_series_project(self.series_file)["manifest"]
        self._apply_workspace_state()

    def request_reorder(self, ordered_ids: list[str]) -> None:
        if self._queue_change_locked():
            self._show_queue_locked_message()
            return
        if not self.series_file:
            return
        self.series_items = update_series_items_order(self.series_file, ordered_ids)
        self.series_manifest = load_series_project(self.series_file)["manifest"]
        self._apply_workspace_state()

    def request_queue_index_change(self, item_id: str, requested_index: int) -> None:
        if self._queue_change_locked():
            self._show_queue_locked_message()
            return
        ordered_ids = self.main.series_workspace.queue_table.ordered_item_ids()
        if item_id not in ordered_ids:
            return
        ordered_ids.remove(item_id)
        insert_at = max(0, min(len(ordered_ids), requested_index - 1))
        ordered_ids.insert(insert_at, item_id)
        self.request_reorder(ordered_ids)

    def request_add_files(self) -> None:
        if self._queue_change_locked():
            self._show_queue_locked_message()
            return
        if not self.series_file:
            return
        exts = " ".join(
            f"*{ext}"
            for ext in sorted(
                [
                    ".png",
                    ".jpg",
                    ".jpeg",
                    ".webp",
                    ".bmp",
                    ".psd",
                    ".pdf",
                    ".epub",
                    ".zip",
                    ".rar",
                    ".7z",
                    ".tar",
                    ".cbz",
                    ".cbr",
                    ".cb7",
                    ".cbt",
                    ".ctpr",
                ]
            )
        )
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self.main,
            self.main.tr("Add Files To Series"),
            os.path.expanduser("~"),
            self.main.tr(f"Supported Files ({exts});;All Files (*)"),
        )
        if not paths:
            return
        self._append_paths_to_series(paths)

    def request_add_folder(self) -> None:
        if self._queue_change_locked():
            self._show_queue_locked_message()
            return
        if not self.series_file:
            return
        root_dir = QtWidgets.QFileDialog.getExistingDirectory(
            self.main,
            self.main.tr("Add Folder To Series"),
            str(self.series_manifest.get("root_dir") or os.path.expanduser("~")),
        )
        if not root_dir:
            return
        self.main.loading.setVisible(True)

        def on_result(paths: list[str]) -> None:
            self.main.loading.setVisible(False)
            if not paths:
                return
            dialog = SeriesImportDialog(root_dir, paths, self.main)
            if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
                return
            self._append_paths_to_series(dialog.selected_paths())

        self.main.run_threaded(
            scan_series_source_files,
            on_result,
            self.main.default_error_handler,
            lambda: self.main.loading.setVisible(False),
            root_dir,
        )

    def _append_paths_to_series(self, paths: list[str]) -> None:
        if not self.series_file:
            return
        root_dir = str(self.series_manifest.get("root_dir") or os.path.dirname(paths[0]))
        global_settings = normalize_series_global_settings(self.series_manifest.get("global_settings"))
        self.main.loading.setVisible(True)

        def on_result(items: list[dict[str, object]]) -> None:
            loaded = load_series_project(self.series_file)
            self.series_manifest = dict(loaded["manifest"])
            self.series_items = list(loaded["items"])
            self._apply_workspace_state()

        self.main.run_threaded(
            add_series_paths,
            on_result,
            self.main.default_error_handler,
            lambda: self.main.loading.setVisible(False),
            self.series_file,
            root_dir=root_dir,
            paths=list(paths),
            source_lang=str(global_settings.get("source_language") or "Japanese"),
            target_lang=str(global_settings.get("target_language") or "English"),
        )

    def request_global_settings_change(self, values: dict[str, object]) -> None:
        if self._queue_change_locked():
            self._show_queue_locked_message()
            return
        if not self.series_file:
            return
        normalized = normalize_series_global_settings(values)
        self.series_manifest = update_series_global_settings(self.series_file, normalized)
        loaded = load_series_project(self.series_file)
        self.series_manifest = dict(loaded["manifest"])
        self.series_items = list(loaded["items"])
        self._apply_series_globals_to_main()
        self._apply_workspace_state()

    def edit_series_settings_dialog(self) -> None:
        if self._queue_change_locked():
            self._show_queue_locked_message()
            return
        if not self.series_file:
            return
        dialog = QtWidgets.QDialog(self.main)
        dialog.setWindowTitle(self.main.tr("Series Settings"))
        dialog.resize(520, 320)
        layout = QtWidgets.QVBoxLayout(dialog)
        page = SeriesPage(dialog)
        page.set_settings(normalize_series_settings(self.series_manifest.get("series_settings")))
        layout.addWidget(page)
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return
        self.series_manifest = update_series_settings(self.series_file, page.get_settings())
        loaded = load_series_project(self.series_file)
        self.series_manifest = dict(loaded["manifest"])
        self.series_items = list(loaded["items"])
        self._apply_workspace_state()

    def sync_active_child_to_series(self) -> None:
        if not self.is_child_project_active() or not self.series_file:
            return
        self.main.project_ctrl.save_current_state()
        previous_project_file = self.main.project_file
        previous_project_kind = getattr(self.main, "project_kind", PROJECT_KIND_SERIES)
        self.main.project_file = self.active_child_project_path
        self.main.project_kind = PROJECT_KIND_SINGLE
        try:
            save_state_to_proj_file(self.main, self.active_child_project_path)
        finally:
            self.main.project_file = previous_project_file
            self.main.project_kind = previous_project_kind
        update_series_child_from_file(
            self.series_file,
            series_item_id=str(self.active_child_item_id),
            child_project_path=self.active_child_project_path,
        )
        loaded = load_series_project(self.series_file)
        self.series_manifest = dict(loaded["manifest"])
        self.series_items = list(loaded["items"])
        self._apply_workspace_state()

    def thread_save_series(self, target_path: str | None = None, post_save_callback: Callable[[], None] | None = None) -> bool:
        if not self.series_file:
            return False
        target = ensure_project_extension(
            target_path or self.series_file,
            SERIES_PROJECT_FILE_EXT,
        )
        if self.is_child_project_active():
            try:
                self.sync_active_child_to_series()
            except Exception as exc:
                QtWidgets.QMessageBox.warning(
                    self.main,
                    self.main.tr("Save Series Project"),
                    self.main.tr("Failed to synchronize the active child project before saving.\n\n{error}").format(
                        error=str(exc)
                    ),
                )
                return False
        self.main.loading.setVisible(True)

        def worker() -> str:
            target_dir = os.path.dirname(os.path.abspath(target))
            if target_dir:
                os.makedirs(target_dir, exist_ok=True)
            if target != self.series_file:
                shutil.copyfile(self.series_file, target)
            return target

        def on_result(saved_path: str) -> None:
            current = self.series_file
            self.series_file = saved_path
            self.main.project_file = saved_path
            self.main.project_kind = PROJECT_KIND_SERIES
            if current and current != saved_path:
                self.main.project_ctrl.remove_recent_project(current)
            self.main.project_ctrl.add_recent_project(saved_path)
            self.main.project_ctrl._refresh_home_screen()
            self._set_series_window_title(
                self._active_child_display_name() if self.is_child_project_active() else None
            )
            self._apply_workspace_state()
            self.main.set_project_clean()

        def on_finished() -> None:
            self.main.loading.setVisible(False)
            if post_save_callback is not None:
                post_save_callback()

        self.main.run_threaded(worker, on_result, self.main.default_error_handler, on_finished)
        return True

    def on_batch_process_finished(self, *, was_cancelled: bool, failed: bool) -> None:
        if not self.is_child_project_active():
            return

        try:
            self.sync_active_child_to_series()
        except Exception:
            # Keep the main batch flow stable even if the series sync fails.
            return

        if not self._queue_active:
            return

        current_item_id = str(self.active_child_item_id or "")
        series_settings = normalize_series_settings(self.series_manifest.get("series_settings"))

        if was_cancelled:
            self._queue_active = False
            self.series_manifest = update_series_queue_runtime(
                self.series_file,
                active_item_id=None,
                failed_item_id=current_item_id or None,
                completed_item_ids=self._queue_completed_ids,
            )
            return

        if failed:
            retry_budget = self._queue_retry_remaining.get(
                current_item_id,
                int(series_settings.get("retry_count", 0) or 0),
            )
            if (
                str(series_settings.get("queue_failure_policy")) == "retry"
                and retry_budget > 0
            ):
                self._queue_retry_remaining[current_item_id] = retry_budget - 1
                QtCore.QTimer.singleShot(0, self.main, self._start_batch_for_active_child)
                return

            self.series_items = update_series_item_status(
                self.series_file,
                series_item_id=current_item_id,
                status="failed",
            )
            self.series_manifest = update_series_queue_runtime(
                self.series_file,
                active_item_id=None,
                failed_item_id=current_item_id,
                completed_item_ids=self._queue_completed_ids,
            )
            loaded = load_series_project(self.series_file)
            self.series_manifest = dict(loaded["manifest"])
            self.series_items = list(loaded["items"])

            if str(series_settings.get("queue_failure_policy")) == "skip":
                QtCore.QTimer.singleShot(0, self.main, self._run_next_queue_item)
                return

            self._queue_active = False
            if not bool(series_settings.get("auto_open_failed_child", True)):
                QtCore.QTimer.singleShot(0, self.main, lambda: self._show_board(push_history=False))
            return

        self._queue_completed_ids.append(current_item_id)
        self.series_items = update_series_item_status(
            self.series_file,
            series_item_id=current_item_id,
            status="done",
        )
        self.series_manifest = update_series_queue_runtime(
            self.series_file,
            active_item_id=None,
            failed_item_id=None,
            completed_item_ids=self._queue_completed_ids,
        )
        loaded = load_series_project(self.series_file)
        self.series_manifest = dict(loaded["manifest"])
        self.series_items = list(loaded["items"])
        QtCore.QTimer.singleShot(0, self.main, self._run_next_queue_item)

    def start_queue_translation(self) -> None:
        if not self.series_file or self._queue_active:
            return
        series_settings = normalize_series_settings(self.series_manifest.get("series_settings"))
        items = sorted(self.series_items, key=lambda item: int(item.get("queue_index", 0)))
        pending_ids = [str(item["series_item_id"]) for item in items]
        if series_settings.get("resume_from_first_incomplete"):
            pending_ids = [
                item_id
                for item_id in pending_ids
                if str(self._find_item(item_id).get("status") or "pending") not in {"done"}
            ]
        if not pending_ids:
            Messages.show_info(
                self.main,
                self.main.tr("There are no queue items left to run."),
                duration=5,
                closable=True,
                source="series",
            )
            return

        self._queue_active = True
        self._queue_pending_ids = pending_ids
        self._queue_completed_ids = []
        self._queue_retry_remaining = {}
        self._apply_workspace_state()
        self._run_next_queue_item()

    def _run_next_queue_item(self) -> None:
        if not self._queue_active:
            return
        if not self._queue_pending_ids:
            self._queue_active = False
            self.series_manifest = update_series_queue_runtime(
                self.series_file,
                active_item_id=None,
                failed_item_id=None,
                completed_item_ids=self._queue_completed_ids,
            )
            loaded = load_series_project(self.series_file)
            self.series_manifest = dict(loaded["manifest"])
            self.series_items = list(loaded["items"])
            self._apply_workspace_state()
            if normalize_series_settings(self.series_manifest.get("series_settings")).get(
                "return_to_series_after_completion",
                True,
            ):
                self._show_board(push_history=False)
            return

        next_item_id = self._queue_pending_ids.pop(0)
        self.series_items = update_series_item_status(
            self.series_file,
            series_item_id=next_item_id,
            status="running",
        )
        self.series_manifest = update_series_queue_runtime(
            self.series_file,
            active_item_id=next_item_id,
            failed_item_id=None,
            completed_item_ids=self._queue_completed_ids,
        )
        loaded = load_series_project(self.series_file)
        self.series_manifest = dict(loaded["manifest"])
        self.series_items = list(loaded["items"])
        self._open_item(
            next_item_id,
            push_history=False,
            after_loaded=self._start_batch_for_active_child,
        )

    def _start_batch_for_active_child(self) -> None:
        if not self.is_child_project_active():
            return
        self._apply_series_globals_to_main()
        source_label = self.main.reverse_lang_mapping.get(
            str(normalize_series_global_settings(self.series_manifest.get("global_settings")).get("source_language") or ""),
            self.main.s_combo.currentText(),
        )
        target_label = self.main.reverse_lang_mapping.get(
            str(normalize_series_global_settings(self.series_manifest.get("global_settings")).get("target_language") or ""),
            self.main.t_combo.currentText(),
        )
        self.main.image_ctrl.apply_languages_to_paths(self.main.image_files, source_label, target_label)
        self.main._start_batch_process_for_paths(list(self.main.image_files), run_type="series_queue")

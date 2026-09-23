from __future__ import annotations

import logging
import os
import shutil
import tempfile
import traceback
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

from PySide6 import QtCore, QtWidgets

from app.projects.project_state import (
    close_state_store,
    load_state_from_proj_file,
    save_state_to_proj_file,
)
from app.projects.project_types import (
    PROJECT_KIND_SERIES,
    SERIES_PROJECT_FILE_EXT,
    ensure_project_extension,
)
from app.projects.series_state_v1 import (
    SERIES_FIELD_UNSET,
    add_series_paths,
    build_series_item_from_path,
    build_series_run_summary,
    create_series_project,
    filter_series_candidate_paths,
    load_series_project,
    materialize_series_child_project,
    merge_series_global_settings,
    normalize_series_global_settings,
    normalize_series_queue_runtime,
    normalize_series_recovery_state,
    normalize_series_settings,
    pending_series_item_ids,
    remove_series_item,
    save_series_manifest,
    scan_series_source_files,
    update_series_child_from_file,
    update_series_global_settings,
    update_series_item_manual_status,
    update_series_item_status,
    update_series_items_order,
    update_series_navigation_history,
    update_series_queue_runtime,
    update_series_settings,
)
from app.ui.series_import_dialog import SeriesImportDialog
from app.ui.series_workspace import SeriesSettingsDialog
from app.ui.messages import Messages
from modules.utils.file_handler import FileHandler

if TYPE_CHECKING:
    from controller import ComicTranslate


logger = logging.getLogger(__name__)

# 시리즈 상태 모듈과 **같은** sentinel 을 써야 한다. 여기서 별도 `object()` 를
# 만들면 생략한 인자가 저쪽에서 "빈 값이 주어졌다"로 해석돼 큐 런타임 필드가
# 통째로 지워진다.
_UNSET = SERIES_FIELD_UNSET

_SERIES_TRANSACTION_ATTRIBUTES = (
    "series_file",
    "series_manifest",
    "series_items",
    "active_child_item_id",
    "active_child_project_path",
    "active_child_temp_dir",
    "history_back",
    "history_forward",
    "_queue_active",
    "_pause_requested",
    "_queue_pending_ids",
    "_queue_completed_ids",
    "_queue_failed_ids",
    "_queue_skipped_ids",
    "_queue_retry_remaining",
    "_child_unsynced_dirty",
    "_recovery_loaded",
    "_main_globals_snapshot",
)


def _snapshot_series_controller(controller: Any) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for name in _SERIES_TRANSACTION_ATTRIBUTES:
        value = getattr(controller, name)
        if isinstance(value, dict):
            value = dict(value)
        elif isinstance(value, list):
            value = list(value)
        values[name] = value
    return values


def _restore_series_controller(controller: Any, values: dict[str, Any]) -> None:
    for name, value in values.items():
        setattr(controller, name, value)


@dataclass
class _PreparedSeriesManifest:
    controller: Any
    state: dict[str, object]
    _committed: bool = False
    _old_child_temp_dir: str = ""
    _success_callbacks: list[Callable[[], None]] = field(default_factory=list)

    def capture_for_commit(self, _main: Any):
        previous = _snapshot_series_controller(self.controller)
        self._old_child_temp_dir = str(previous.get("active_child_temp_dir") or "")
        return lambda: _restore_series_controller(self.controller, previous)

    def cleanup(self) -> None:
        return

    def transfer_ownership(self, main: Any) -> None:
        self._committed = True
        main._workspace_owned_temp_roots = []

    def defer_success(self, callback: Callable[[], None]) -> None:
        self._success_callbacks.append(callback)

    def commit_success(self) -> None:
        if self._old_child_temp_dir:
            shutil.rmtree(self._old_child_temp_dir, ignore_errors=True)
        for callback in self._success_callbacks:
            try:
                callback()
            except Exception:
                logger.warning("Series success cleanup failed.", exc_info=True)
        self._success_callbacks.clear()


@dataclass
class _PreparedSeriesChild:
    controller: Any
    child_project_path: str
    snapshot: Any
    work_dir: str
    _committed: bool = False
    _old_child_temp_dir: str = ""

    def capture_for_commit(self, _main: Any):
        previous = _snapshot_series_controller(self.controller)
        self._old_child_temp_dir = str(previous.get("active_child_temp_dir") or "")
        return lambda: _restore_series_controller(self.controller, previous)

    def cleanup(self) -> None:
        if self._committed:
            return
        self.snapshot.cleanup()
        shutil.rmtree(self.work_dir, ignore_errors=True)

    def transfer_ownership(self, main: Any) -> None:
        self._committed = True
        self.snapshot.transfer_ownership(main)
        main._workspace_owned_temp_roots = [
            os.path.abspath(self.work_dir),
            os.path.abspath(str(self.snapshot.state.temp_dir)),
        ]

    def commit_success(self) -> None:
        if self._old_child_temp_dir and os.path.abspath(self._old_child_temp_dir) != os.path.abspath(self.work_dir):
            shutil.rmtree(self._old_child_temp_dir, ignore_errors=True)


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
        self._pause_requested = False
        self._queue_pending_ids: list[str] = []
        self._queue_completed_ids: list[str] = []
        self._queue_failed_ids: list[str] = []
        self._queue_skipped_ids: list[str] = []
        self._queue_retry_remaining: dict[str, int] = {}
        self._child_unsynced_dirty = False
        self._recovery_loaded = False
        # 시리즈 전역 설정은 설정 페이지 위젯에 그대로 써 넣는다. 시리즈를 닫을
        # 때 되돌리지 않으면 사용자의 앱 기본 설정이 그 시리즈 값으로 바뀐 채
        # 남는다. 진입 시 한 번 찍어두고 컨텍스트 해제 때 복원한다.
        self._main_globals_snapshot: dict[str, object] | None = None

    def has_series_loaded(self) -> bool:
        return bool(self.series_file)

    def is_child_project_active(self) -> bool:
        return bool(self.series_file and self.active_child_item_id and self.active_child_project_path)

    def is_series_board_active(self) -> bool:
        return bool(self.series_file and not self.active_child_item_id)

    def is_queue_running(self) -> bool:
        return bool(self._queue_active)

    def is_queue_paused(self) -> bool:
        queue_runtime = normalize_series_queue_runtime(
            self.series_manifest.get("series_queue_runtime")
            if isinstance(self.series_manifest.get("series_queue_runtime"), dict)
            else None
        )
        return bool(queue_runtime.get("queue_state") == "paused")

    def active_queue_runtime(self) -> dict[str, object]:
        return normalize_series_queue_runtime(
            self.series_manifest.get("series_queue_runtime")
            if isinstance(self.series_manifest.get("series_queue_runtime"), dict)
            else None
        )

    def reset_series_context(self) -> None:
        self._clear_active_child_materialization()
        # 시리즈 전역 설정으로 덮어썼던 앱 설정을 원래대로 돌린다.
        self._restore_main_globals_snapshot()
        self.series_file = None
        self.series_manifest = {}
        self.series_items = []
        self.history_back = []
        self.history_forward = []
        self._queue_active = False
        self._pause_requested = False
        self._queue_pending_ids = []
        self._queue_completed_ids = []
        self._queue_failed_ids = []
        self._queue_skipped_ids = []
        self._queue_retry_remaining = {}
        self._child_unsynced_dirty = False
        self._recovery_loaded = False

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
        self._refresh_breadcrumb()

    def _current_series_display_name(self) -> str:
        return os.path.basename(self.series_file or "")

    def breadcrumb_state(self) -> dict[str, object] | None:
        """자식 컨텍스트 표시줄에 넘길 상태. 자식이 없으면 `None`."""
        if not self.is_child_project_active():
            return None
        return {
            "series_name": self._current_series_display_name(),
            "child_name": self._active_child_display_name() or "",
            "unsynced": bool(self._child_unsynced_dirty),
            "can_back": bool(self.history_back),
            "locked_reason": (
                self.main.tr(
                    "Queue changes are locked while automatic translation is running.\n"
                    "The current running item stays fixed, and you can change the queue "
                    "after the run finishes."
                )
                if self._queue_change_locked()
                else ""
            ),
        }

    def _refresh_breadcrumb(self) -> None:
        refresh = getattr(self.main, "refresh_series_breadcrumb", None)
        if callable(refresh):
            refresh()

    def _set_series_window_title(self, child_name: str | None = None) -> None:
        series_name = self._current_series_display_name() or f"Series{SERIES_PROJECT_FILE_EXT}"
        status_suffixes: list[str] = []
        if self._recovery_loaded:
            status_suffixes.append(self.main.tr("Recovered Snapshot"))
        if child_name and self._child_unsynced_dirty:
            status_suffixes.append(self.main.tr("Child Changes Not Synced"))
        suffix = ""
        if status_suffixes:
            suffix = " · " + " / ".join(status_suffixes)
        if child_name:
            self.main.setWindowTitle(
                self.main.tr("Child Project - {child} · {series}[*]").format(
                    child=child_name,
                    series=series_name,
                )
                + suffix
            )
        else:
            self.main.setWindowTitle(
                self.main.tr("Series Project - {series}[*]").format(series=series_name)
                + suffix
            )
        # 창 제목이 갱신되는 지점은 곧 시리즈 컨텍스트가 바뀌는 지점이다.
        # 커스텀 타이틀바는 폭이 좁으면 제목을 숨기므로 표시줄이 실질적인
        # 컨텍스트 표시 수단이다.
        self._refresh_breadcrumb()

    def notify_active_child_dirty(self) -> None:
        if not self.is_child_project_active():
            return
        self._child_unsynced_dirty = True
        self._set_series_window_title(self._active_child_display_name())
        if self.is_series_board_active():
            self._apply_workspace_state()

    def _clear_child_unsynced_dirty(self) -> None:
        self._child_unsynced_dirty = False
        self._set_series_window_title(self._active_child_display_name())
        if self.is_series_board_active():
            self._apply_workspace_state()

    def _clear_recovery_loaded(self) -> None:
        self._recovery_loaded = False
        self._set_series_window_title(self._active_child_display_name())
        if self.is_series_board_active():
            self._apply_workspace_state()

    def _sync_paused_pending_runtime(self) -> None:
        if not self.series_file:
            return
        queue_runtime = self.active_queue_runtime()
        if queue_runtime.get("queue_state") != "paused":
            return
        pending_ids = pending_series_item_ids(self.series_items)
        failed_item_id = str(queue_runtime.get("failed_item_id") or "").strip() or None
        if failed_item_id and self._find_item(failed_item_id) is None:
            failed_item_id = None
        self.series_manifest = update_series_queue_runtime(
            self.series_file,
            queue_state="paused",
            pause_requested=False,
            pending_item_ids=pending_ids,
            active_item_id=None,
            failed_item_ids=queue_runtime.get("failed_item_ids") or [],
            skipped_item_ids=queue_runtime.get("skipped_item_ids") or [],
            failed_item_id=failed_item_id,
            completed_item_ids=queue_runtime.get("completed_item_ids") or [],
            retry_remaining_by_item=queue_runtime.get("retry_remaining_by_item") or {},
            last_run_started_at=queue_runtime.get("last_run_started_at"),
            last_run_finished_at=queue_runtime.get("last_run_finished_at"),
            last_run_summary=queue_runtime.get("last_run_summary") or {},
        )
        loaded = load_series_project(self.series_file)
        self.series_manifest = dict(loaded["manifest"])
        self.series_items = list(loaded["items"])

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
                "export_settings": self.main.settings_page.get_export_settings(),
                "render_settings": self._series_render_settings_from_main(),
            }
        )

    def _series_render_settings_from_main(self) -> dict[str, object]:
        render_settings = self.main.render_settings()
        return {
            "alignment_id": int(render_settings.alignment_id),
            "vertical_alignment_id": int(render_settings.vertical_alignment_id),
            "font_family": str(render_settings.font_family or ""),
            "min_font_size": int(render_settings.min_font_size),
            "max_font_size": int(render_settings.max_font_size),
            "auto_max_font_size": bool(render_settings.auto_max_font_size),
            "auto_max_font_profile": str(render_settings.auto_max_font_profile or "current"),
            "color": str(render_settings.color or ""),
            "force_font_color": bool(render_settings.force_font_color),
            "upper_case": bool(render_settings.upper_case),
            "outline": bool(render_settings.outline),
            "outline_color": str(render_settings.outline_color or ""),
            "outline_width": str(render_settings.outline_width or "1.0"),
            "bold": bool(render_settings.bold),
            "italic": bool(render_settings.italic),
            "underline": bool(render_settings.underline),
            "line_spacing": str(render_settings.line_spacing or "1.0"),
        }

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

    def _series_font_options_from_main(self) -> list[str]:
        combo = self.main.font_dropdown
        fonts = [combo.itemText(index) for index in range(combo.count())]
        current = combo.currentText()
        if current and current not in fonts:
            fonts.insert(0, current)
        return [font for font in fonts if font]

    def _combo_options_from_main(self, combo: QtWidgets.QComboBox) -> list[tuple[str, str]]:
        return [
            (str(combo.itemData(index) or ""), combo.itemText(index))
            for index in range(combo.count())
        ]

    def _series_output_options_from_main(self) -> dict[str, list[tuple[str, str]]]:
        ui = self.main.settings_page.ui
        return {
            "automatic_output_target": self._combo_options_from_main(ui.automatic_output_target_combo),
            "automatic_output_image_format": self._combo_options_from_main(ui.automatic_output_image_format_combo),
            "automatic_output_archive_format": self._combo_options_from_main(ui.automatic_output_archive_format_combo),
            "automatic_output_archive_image_format": self._combo_options_from_main(ui.automatic_output_archive_image_format_combo),
        }

    def _apply_workspace_state(self) -> None:
        if not self.series_file:
            return
        queue_runtime = self.active_queue_runtime()
        self.main.series_workspace.configure_options(**self._series_workspace_options())
        self.main.series_workspace.set_global_settings(
            normalize_series_global_settings(self.series_manifest.get("global_settings"))
        )
        self.main.series_workspace.set_series_state(
            series_file=self._current_series_display_name(),
            items=list(self.series_items),
            queue_running=self._queue_active,
            active_item_id=str(queue_runtime.get("active_item_id") or ""),
            queue_runtime=queue_runtime,
            child_unsynced_dirty=self._child_unsynced_dirty,
            recovery_loaded=self._recovery_loaded,
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

    def _clear_active_child_materialization(self, *, preserve_workdir: bool = False) -> None:
        """자식 materialization 을 놓는다.

        `preserve_workdir` 는 시리즈로 반영하지 못한 변경이 남았을 때만 쓴다.
        작업 디렉터리를 지우면 그 변경이 복구 불가능하게 사라지므로, 반영에
        실패한 경우에는 디스크에 남겨 두고 사용자에게 경로를 알린다.
        """
        self.active_child_item_id = None
        self.active_child_project_path = None
        self._child_unsynced_dirty = False
        if not preserve_workdir and self.active_child_temp_dir and os.path.isdir(self.active_child_temp_dir):
            shutil.rmtree(self.active_child_temp_dir, ignore_errors=True)
        self.active_child_temp_dir = None

    def _sync_active_child_before_teardown(self) -> bool:
        """자식 작업본을 버리기 전에 미반영 변경을 시리즈로 밀어 넣는다.

        큐 종료·실패·일시정지 경로의 `_show_board(push_history=False)` 는
        `_run_guarded_project_transition` 가드를 타지 않는다. 그 경로가
        작업 디렉터리를 지우고 `set_project_clean()` 까지 부르기 때문에,
        여기서 반영하지 못한 변경은 경고 없이 사라졌다.
        """
        if not self.is_child_project_active() or not self._child_unsynced_dirty:
            return True
        try:
            self.sync_active_child_to_series()
            return True
        except Exception:
            logger.warning(
                "Failed to sync the active child project back into the series project.",
                exc_info=True,
            )
            return False

    def _warn_child_sync_failed(self, work_dir: str | None) -> None:
        message = self.main.tr(
            "Could not write this chapter's changes back to the series project."
        )
        if work_dir:
            message += "\n" + self.main.tr(
                "The working copy was kept so nothing is lost: {path}"
            ).format(path=work_dir)
        Messages.show_warning(
            self.main,
            message,
            duration=None,
            closable=True,
            source="series",
        )

    def _load_series_worker(self, file_name: str, recovery_loaded: bool = False) -> dict[str, object]:
        if recovery_loaded:
            state = load_series_project(file_name)
            next_manifest, next_items, changed = normalize_series_recovery_state(
                dict(state.get("manifest") or {}),
                list(state.get("items") or []),
            )
            if changed:
                save_series_manifest(file_name, manifest=next_manifest, items=next_items)
                return load_series_project(file_name)
            return {
                "manifest": next_manifest,
                "items": next_items,
            }
        return load_series_project(file_name)

    def thread_load_series_project(self, file_name: str, *, recovery_loaded: bool = False) -> None:
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

        previous_project = getattr(self.main, "project_file", None)
        coordinator = self.main.open_workspace_ctrl

        def prepare(report_progress):
            report_progress(
                {
                    "stage": "index",
                    "message": self.main.tr("Loading series project..."),
                    "detail": os.path.basename(normalized_path),
                }
            )
            return _PreparedSeriesManifest(
                self,
                self._load_series_worker(normalized_path, recovery_loaded),
            )

        def commit(prepared: _PreparedSeriesManifest) -> None:
            state = prepared.state
            if isinstance(previous_project, str) and previous_project and previous_project != normalized_path:
                prepared.defer_success(lambda: close_state_store(previous_project))
            self.main.file_handler = FileHandler()
            self.main.image_ctrl.clear_state()
            self.active_child_item_id = None
            self.active_child_project_path = None
            self.active_child_temp_dir = None
            self._child_unsynced_dirty = False
            self.series_file = normalized_path
            self.series_manifest = dict(state.get("manifest") or {})
            self.series_items = list(state.get("items") or [])
            self._pause_requested = False
            self._queue_active = False
            self._queue_pending_ids = list(
                self.active_queue_runtime().get("pending_item_ids") or []
            )
            self._queue_completed_ids = list(
                self.active_queue_runtime().get("completed_item_ids") or []
            )
            self._queue_failed_ids = list(
                self.active_queue_runtime().get("failed_item_ids") or []
            )
            self._queue_skipped_ids = list(
                self.active_queue_runtime().get("skipped_item_ids") or []
            )
            self._queue_retry_remaining = dict(
                self.active_queue_runtime().get("retry_remaining_by_item") or {}
            )
            self._recovery_loaded = bool(recovery_loaded)
            nav = self.series_manifest.get("series_navigation_history") or {}
            self.history_back = list(nav.get("back") or [])
            self.history_forward = list(nav.get("forward") or [])
            self.main.project_file = normalized_path
            self.main.project_kind = PROJECT_KIND_SERIES
            self._apply_workspace_state()
            self.main.show_series_page()
            self.main.set_project_clean()
            self._set_series_window_title()
            if recovery_loaded:
                self.main.mark_project_dirty()
                Messages.show_info(
                    self.main,
                    self.main.tr(
                        "The previous automatic translation run was interrupted and restored as paused."
                    ),
                    duration=6,
                    closable=True,
                    source="series",
                )

            self.main.project_ctrl.add_recent_project(normalized_path)
            self.main.project_ctrl._refresh_home_screen()

        coordinator.run(
            message=self.main.tr("Loading series project..."),
            prepare=prepare,
            commit=commit,
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

    def _show_duplicate_paths_message(
        self,
        *,
        skipped_existing: list[str],
        skipped_duplicates: list[str],
    ) -> None:
        if not skipped_existing and not skipped_duplicates:
            return
        message_parts = []
        if skipped_existing:
            message_parts.append(
                self.main.tr("Already in this series: {count}").format(
                    count=len(skipped_existing)
                )
            )
        if skipped_duplicates:
            message_parts.append(
                self.main.tr("Duplicate selections removed: {count}").format(
                    count=len(skipped_duplicates)
                )
            )
        Messages.show_info(
            self.main,
            "\n".join(message_parts),
            duration=6,
            closable=True,
            source="series",
        )

    def _filter_appendable_paths(self, paths: list[str]) -> list[str]:
        existing_paths = [
            str(item.get("source_origin_path") or "")
            for item in self.series_items
            if str(item.get("source_origin_path") or "").strip()
        ]
        filtered = filter_series_candidate_paths(existing_paths, paths)
        self._show_duplicate_paths_message(
            skipped_existing=filtered["skipped_existing"],
            skipped_duplicates=filtered["skipped_duplicates"],
        )
        return list(filtered["accepted"])

    def prompt_new_series_project(self) -> None:
        root_dir = QtWidgets.QFileDialog.getExistingDirectory(
            self.main,
            self.main.tr("Select Series Root Folder"),
            os.path.expanduser("~"),
        )
        if not root_dir:
            return

        self.main.loading.setVisible(True)
        scan_dialog = Messages.show_busy(
            self.main,
            self.main.tr("Scanning series folder..."),
            title=self.main.tr("Create Series Project"),
        )

        def on_result(paths: list[str]) -> None:
            Messages.close_busy(scan_dialog, force=True)
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
            filtered = filter_series_candidate_paths([], selected_paths)
            self._show_duplicate_paths_message(
                skipped_existing=filtered["skipped_existing"],
                skipped_duplicates=filtered["skipped_duplicates"],
            )
            selected_paths = list(filtered.get("accepted") or [])
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
            create_dialog = Messages.show_busy(
                self.main,
                self.main.tr("Creating series project..."),
                title=self.main.tr("Create Series Project"),
            )

            def on_create_error(error_tuple) -> None:
                Messages.close_busy(create_dialog, force=True)
                self.main.default_error_handler(error_tuple)

            def on_create_finished() -> None:
                Messages.close_busy(create_dialog)
                self.main.loading.setVisible(False)

            self.main.run_threaded(
                self._build_series_project_worker,
                lambda state: self._apply_new_series_result(target_path, state),
                on_create_error,
                on_create_finished,
                target_path,
                root_dir,
                selected_paths,
                global_settings,
                series_settings,
            )

        def on_error(error_tuple) -> None:
            Messages.close_busy(scan_dialog, force=True)
            self.main.loading.setVisible(False)
            self.main.default_error_handler(error_tuple)

        self.main.run_threaded(
            scan_series_source_files,
            on_result,
            on_error,
            lambda: Messages.close_busy(scan_dialog, force=True),
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
        self._queue_active = False
        self._pause_requested = False
        self._queue_pending_ids = []
        self._queue_completed_ids = []
        self._queue_failed_ids = []
        self._queue_skipped_ids = []
        self._queue_retry_remaining = {}
        self._recovery_loaded = False
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

    def _snapshot_main_global_settings(self) -> dict[str, object]:
        """현재 설정 페이지 상태를 시리즈 global_settings 와 같은 스키마로 뜬다.

        복원 경로가 `_apply_global_settings_to_main` 을 그대로 재사용할 수
        있도록 스키마를 맞춘다.
        """
        settings_page = self.main.settings_page
        return {
            "source_language": self.main.lang_mapping.get(self.main.s_combo.currentText(), ""),
            "target_language": self.main.lang_mapping.get(self.main.t_combo.currentText(), ""),
            "ocr": settings_page.get_tool_selection("ocr"),
            "translator": settings_page.get_tool_selection("translator"),
            "workflow_mode": settings_page.get_workflow_mode(),
            "use_gpu": bool(settings_page.ui.use_gpu_checkbox.isChecked()),
            "export_settings": dict(settings_page.get_export_settings()),
            "render_settings": self._series_render_settings_from_main(),
        }

    def _capture_main_globals_snapshot(self) -> None:
        """시리즈 값이 위젯을 덮기 전에 원래 값을 한 번만 기록한다."""
        if self._main_globals_snapshot is not None:
            return
        try:
            self._main_globals_snapshot = self._snapshot_main_global_settings()
        except Exception:
            # 스냅샷 실패가 시리즈 열기를 막을 이유는 없다. 복원을 포기할 뿐이다.
            self._main_globals_snapshot = None
            logger.warning("Failed to snapshot main global settings.", exc_info=True)

    def _restore_main_globals_snapshot(self) -> None:
        snapshot = self._main_globals_snapshot
        self._main_globals_snapshot = None
        if not snapshot:
            return
        try:
            self._apply_global_settings_to_main(snapshot)
        except Exception:
            logger.warning("Failed to restore main global settings.", exc_info=True)

    def _apply_series_globals_to_main(self) -> None:
        self._capture_main_globals_snapshot()
        self._apply_global_settings_to_main(
            normalize_series_global_settings(self.series_manifest.get("global_settings"))
        )

    def _apply_global_settings_to_main(self, global_settings: dict[str, object]) -> None:
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
        export_settings = global_settings.get("export_settings")
        if isinstance(export_settings, dict):
            self._apply_series_export_settings_to_main(export_settings)
        render_settings = global_settings.get("render_settings")
        if isinstance(render_settings, dict):
            self._apply_series_render_settings_to_main(render_settings)

    def _set_combo_data(self, combo: QtWidgets.QComboBox, value: object) -> None:
        index = combo.findData(value)
        if index >= 0:
            combo.setCurrentIndex(index)

    def _apply_series_export_settings_to_main(self, settings: dict[str, object]) -> None:
        ui = self.main.settings_page.ui
        checkbox_map = {
            "export_raw_text": ui.raw_text_checkbox,
            "export_translated_text": ui.translated_text_checkbox,
            "export_inpainted_image": ui.inpainted_image_checkbox,
            "export_ocr_debug": ui.ocr_debug_checkbox,
            "export_detector_overlay": ui.detector_overlay_checkbox,
            "export_raw_mask": ui.raw_mask_checkbox,
            "export_mask_overlay": ui.mask_overlay_checkbox,
            "export_cleanup_mask_delta": ui.cleanup_mask_delta_checkbox,
            "export_debug_metadata": ui.debug_metadata_checkbox,
        }
        for key, checkbox in checkbox_map.items():
            if key in settings:
                checkbox.setChecked(bool(settings.get(key, False)))
        self._set_combo_data(ui.automatic_output_target_combo, settings.get("automatic_output_target"))
        self._set_combo_data(ui.automatic_output_image_format_combo, settings.get("automatic_output_image_format"))
        self._set_combo_data(ui.automatic_output_archive_format_combo, settings.get("automatic_output_archive_format"))
        self._set_combo_data(
            ui.automatic_output_archive_image_format_combo,
            settings.get("automatic_output_archive_image_format"),
        )
        if "automatic_output_archive_compression_level" in settings:
            ui.automatic_output_archive_level_spinbox.setValue(
                max(0, min(9, int(settings.get("automatic_output_archive_compression_level") or 6)))
            )

    def _apply_series_render_settings_to_main(self, settings: dict[str, object]) -> None:
        if "alignment_id" in settings:
            self.main.alignment_tool_group.set_dayu_checked(int(settings.get("alignment_id") or 0))
        if "vertical_alignment_id" in settings:
            self.main.vertical_alignment_tool_group.set_dayu_checked(int(settings.get("vertical_alignment_id") or 0))
        if settings.get("font_family"):
            self.main.set_font(str(settings.get("font_family") or ""))
        ui = self.main.settings_page.ui
        if "min_font_size" in settings:
            ui.min_font_spinbox.setValue(int(settings.get("min_font_size") or ui.min_font_spinbox.value()))
        if "max_font_size" in settings:
            ui.max_font_spinbox.setValue(int(settings.get("max_font_size") or ui.max_font_spinbox.value()))
        if "auto_max_font_size" in settings:
            ui.auto_max_font_checkbox.setChecked(bool(settings.get("auto_max_font_size", True)))
        if "auto_max_font_profile" in settings:
            profile_combo = getattr(ui, "auto_max_font_profile_combo", None)
            if profile_combo is not None:
                index = profile_combo.findData(str(settings.get("auto_max_font_profile") or "current"))
                profile_combo.setCurrentIndex(index if index >= 0 else 0)
                profile_combo.setEnabled(ui.auto_max_font_checkbox.isChecked())
        if settings.get("color"):
            color = str(settings.get("color") or "")
            self.main.block_font_color_button.setStyleSheet(
                f"background-color: {color}; border: none; border-radius: 5px;"
            )
            self.main.block_font_color_button.setProperty("selected_color", color)
        if "force_font_color" in settings:
            self.main.force_font_color_checkbox.setChecked(bool(settings.get("force_font_color", False)))
        if "upper_case" in settings:
            ui.uppercase_checkbox.setChecked(bool(settings.get("upper_case", False)))
        if "outline" in settings:
            outline_value = bool(settings.get("outline", False))
            self.main.outline_checkbox.setChecked(outline_value)
            self.main.outline_mode_group.set_dayu_checked(1 if outline_value else 0)
        if settings.get("outline_color"):
            outline_color = str(settings.get("outline_color") or "#ffffff")
            self.main.outline_font_color_button.setStyleSheet(
                f"background-color: {outline_color}; border: none; border-radius: 5px;"
            )
            self.main.outline_font_color_button.setProperty("selected_color", outline_color)
        if settings.get("outline_width"):
            self.main.outline_width_dropdown.setCurrentText(str(settings.get("outline_width") or "1.0"))
        if "bold" in settings:
            self.main.bold_button.setChecked(bool(settings.get("bold", False)))
        if "italic" in settings:
            self.main.italic_button.setChecked(bool(settings.get("italic", False)))
        if "underline" in settings:
            self.main.underline_button.setChecked(bool(settings.get("underline", False)))
        if settings.get("line_spacing"):
            self.main.line_spacing_dropdown.setCurrentText(str(settings.get("line_spacing") or "1.0"))
        outline_enabled = bool(settings.get("outline", self.main.outline_checkbox.isChecked()))
        self.main.outline_font_color_button.setEnabled(outline_enabled)
        self.main.outline_width_dropdown.setEnabled(outline_enabled)

    def _open_child_worker(
        self,
        report_progress,
        item: dict,
        work_dir: str,
    ):
        report_progress(
            {
                "stage": "materialize",
                "message": self.main.tr("Preparing the series item..."),
                "detail": str(item.get("display_name") or ""),
            }
        )
        child_project_path = materialize_series_child_project(
            self.series_file,
            item,
            temp_dir=work_dir,
        )
        snapshot = self.main.open_workspace_ctrl.prepare_project(
            report_progress,
            self.main,
            child_project_path,
        )
        return _PreparedSeriesChild(
            self,
            child_project_path,
            snapshot,
            work_dir,
        )

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
        work_dir = tempfile.mkdtemp(prefix="series_child_", dir=self.main.temp_dir)
        coordinator = self.main.open_workspace_ctrl

        def prepare(report_progress):
            try:
                return self._open_child_worker(report_progress, item, work_dir)
            except Exception:
                shutil.rmtree(work_dir, ignore_errors=True)
                raise

        def commit(prepared: _PreparedSeriesChild) -> None:
            if push_history:
                self._push_history()
            child_project_path = prepared.child_project_path
            snapshot = prepared.snapshot
            self.main.file_handler = FileHandler()
            self.main.image_ctrl.clear_state()
            snapshot.apply(self.main)
            self.main.project_ctrl.load_state_to_ui(snapshot.saved_context)
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

        coordinator.run(
            message=self.main.tr("Opening series item..."),
            prepare=prepare,
            commit=commit,
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
        # 작업본을 버리기 전에 미반영 변경을 시리즈로 밀어 넣는다. 실패하면
        # 작업 디렉터리를 남겨 두고 경고한다 — 아래 `set_project_clean()` 이
        # dirty 표시까지 지우기 때문에, 여기서 놓치면 조용히 사라진다.
        stale_work_dir = self.active_child_temp_dir
        sync_ok = self._sync_active_child_before_teardown()
        self.main.image_ctrl.clear_state()
        self._clear_active_child_materialization(preserve_workdir=not sync_ok)
        if not sync_ok:
            self._warn_child_sync_failed(stale_work_dir)
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
        self._sync_paused_pending_runtime()
        self._apply_workspace_state()

    def request_reorder(self, ordered_ids: list[str]) -> None:
        if self._queue_change_locked():
            self._show_queue_locked_message()
            return
        if not self.series_file:
            return
        self.series_items = update_series_items_order(self.series_file, ordered_ids)
        self.series_manifest = load_series_project(self.series_file)["manifest"]
        self._sync_paused_pending_runtime()
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

    def request_item_status_change(self, item_id: str, target_status: str) -> None:
        if self._queue_change_locked():
            self._show_queue_locked_message()
            return
        if not self.series_file:
            return
        target = str(target_status or "").strip().lower()
        if target not in {"pending", "done"}:
            Messages.show_info(
                self.main,
                self.main.tr("Series item status can only be changed to Pending or Done."),
                duration=5,
                closable=True,
                source="series",
            )
            return
        busy_dialog = Messages.show_busy(
            self.main,
            self.main.tr("Updating series item status..."),
            title=self.main.tr("Series Project"),
            minimum_visible_ms=300,
        )
        try:
            state = update_series_item_manual_status(
                self.series_file,
                series_item_id=item_id,
                status=target,
            )
        except KeyError:
            return
        finally:
            Messages.close_busy(busy_dialog)
        self.series_manifest = dict(state["manifest"])
        self.series_items = list(state["items"])
        queue_runtime = self.active_queue_runtime()
        self._pause_requested = bool(queue_runtime.get("pause_requested", False))
        self._queue_pending_ids = list(queue_runtime.get("pending_item_ids") or [])
        self._queue_completed_ids = list(queue_runtime.get("completed_item_ids") or [])
        self._queue_failed_ids = list(queue_runtime.get("failed_item_ids") or [])
        self._queue_skipped_ids = list(queue_runtime.get("skipped_item_ids") or [])
        self._queue_retry_remaining = dict(queue_runtime.get("retry_remaining_by_item") or {})
        if hasattr(self.main, "pipeline_status_panel"):
            queue_state = str(queue_runtime.get("queue_state") or "idle").strip().lower()
            self.main.pipeline_status_panel.set_series_queue_pause_visible(
                queue_state == "paused",
                pause_requested=False,
            )
        self._apply_workspace_state()

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
                    ".jp2",
                    ".j2k",
                    ".jpf",
                    ".jpx",
                    ".j2c",
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
        scan_dialog = Messages.show_busy(
            self.main,
            self.main.tr("Scanning series folder..."),
            title=self.main.tr("Series Project"),
            minimum_visible_ms=300,
        )

        def on_result(paths: list[str]) -> None:
            Messages.close_busy(scan_dialog, force=True)
            self.main.loading.setVisible(False)
            if not paths:
                return
            dialog = SeriesImportDialog(root_dir, paths, self.main)
            if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
                return
            self._append_paths_to_series(dialog.selected_paths())

        def on_error(error_tuple) -> None:
            Messages.close_busy(scan_dialog, force=True)
            self.main.default_error_handler(error_tuple)

        def on_finished() -> None:
            Messages.close_busy(scan_dialog, force=True)
            self.main.loading.setVisible(False)

        self.main.run_threaded(
            scan_series_source_files,
            on_result,
            on_error,
            on_finished,
            root_dir,
        )

    def _append_paths_to_series(self, paths: list[str]) -> None:
        if not self.series_file:
            return
        paths = self._filter_appendable_paths(paths)
        if not paths:
            return
        root_dir = str(self.series_manifest.get("root_dir") or os.path.dirname(paths[0]))
        global_settings = normalize_series_global_settings(self.series_manifest.get("global_settings"))
        self.main.loading.setVisible(True)
        busy_dialog = Messages.show_busy(
            self.main,
            self.main.tr("Adding files to series..."),
            title=self.main.tr("Series Project"),
            minimum_visible_ms=300,
        )

        def on_result(items: list[dict[str, object]]) -> None:
            loaded = load_series_project(self.series_file)
            self.series_manifest = dict(loaded["manifest"])
            self.series_items = list(loaded["items"])
            self._sync_paused_pending_runtime()
            self._apply_workspace_state()

        def on_error(error_tuple) -> None:
            Messages.close_busy(busy_dialog, force=True)
            self.main.default_error_handler(error_tuple)

        def on_finished() -> None:
            Messages.close_busy(busy_dialog)
            self.main.loading.setVisible(False)

        self.main.run_threaded(
            add_series_paths,
            on_result,
            on_error,
            on_finished,
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
        normalized = merge_series_global_settings(
            self.series_manifest.get("global_settings")
            if isinstance(self.series_manifest.get("global_settings"), dict)
            else None,
            values,
        )
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
        dialog = SeriesSettingsDialog(self.main)
        dialog.configure_options(
            **self._series_workspace_options(),
            fonts=self._series_font_options_from_main(),
            output_options=self._series_output_options_from_main(),
        )
        global_settings = normalize_series_global_settings(self.series_manifest.get("global_settings"))
        if not isinstance(global_settings.get("render_settings"), dict) or not global_settings.get("render_settings"):
            global_settings["render_settings"] = self._series_render_settings_from_main()
        if not isinstance(global_settings.get("export_settings"), dict) or not global_settings.get("export_settings"):
            global_settings["export_settings"] = self.main.settings_page.get_export_settings()
        dialog.set_payload(
            normalize_series_settings(self.series_manifest.get("series_settings")),
            global_settings,
        )
        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return
        series_settings, next_global_settings = dialog.payload()
        self.series_manifest = update_series_settings(self.series_file, series_settings)
        self.series_manifest = update_series_global_settings(self.series_file, next_global_settings)
        loaded = load_series_project(self.series_file)
        self.series_manifest = dict(loaded["manifest"])
        self.series_items = list(loaded["items"])
        self._apply_series_globals_to_main()
        self._apply_workspace_state()

    def sync_active_child_to_series(self) -> None:
        if not self.is_child_project_active() or not self.series_file:
            return
        payload = self.prepare_active_child_sync()
        if payload is None:
            return
        self.write_active_child_sync(payload)
        self.finalize_active_child_sync()

    def prepare_active_child_sync(self) -> dict[str, str] | None:
        """1단계 (메인 스레드): UI 상태를 수집하고 쓰기에 필요한 값만 뽑는다.

        `save_current_state` 는 화면의 편집 결과를 페이지 상태로 옮기는
        일이라 반드시 메인 스레드여야 한다. 나머지 단계는 값만 있으면 된다.
        """
        if not self.is_child_project_active() or not self.series_file:
            return None
        self.main.project_ctrl.save_current_state()
        return {
            "series_file": str(self.series_file),
            "child_project_path": str(self.active_child_project_path),
            "series_item_id": str(self.active_child_item_id),
        }

    def write_active_child_sync(
        self,
        payload: dict[str, str],
        *,
        series_target_file: str | None = None,
    ) -> None:
        """2단계 (워커 가능): 자식 프로젝트를 쓰고 시리즈에 임베드한다.

        예전에는 `main.project_file` 을 자식 경로로 잠시 바꿔치기했다. 저장을
        워커로 옮기면 그 전역 조작이 메인 스레드와 경합하므로,
        `source_project_file` 인자로 대체했다.

        `series_target_file` 을 주면 원본 대신 그 파일에 임베드한다. 자동저장이
        꺼져 있을 때 원본 시리즈 파일을 건드리지 않기 위한 경로다.
        """
        child_project_path = payload["child_project_path"]
        save_state_to_proj_file(
            self.main,
            child_project_path,
            source_project_file=child_project_path,
        )
        update_series_child_from_file(
            series_target_file or payload["series_file"],
            series_item_id=payload["series_item_id"],
            child_project_path=child_project_path,
        )

    def finalize_active_child_sync(self) -> None:
        """3단계 (메인 스레드): 갱신된 매니페스트를 다시 읽고 UI 를 맞춘다."""
        if not self.series_file:
            return
        loaded = load_series_project(self.series_file)
        self.series_manifest = dict(loaded["manifest"])
        self.series_items = list(loaded["items"])
        self._child_unsynced_dirty = False
        self._apply_workspace_state()
        self._set_series_window_title(self._active_child_display_name())

    def thread_save_series(self, target_path: str | None = None, post_save_callback: Callable[[], None] | None = None) -> bool:
        if not self.series_file:
            return False
        target = ensure_project_extension(
            target_path or self.series_file,
            SERIES_PROJECT_FILE_EXT,
        )
        self.main.loading.setVisible(True)
        busy_dialog = Messages.show_busy(
            self.main,
            self.main.tr("Saving series project file..."),
            title=self.main.tr("Project File"),
        )
        if self.is_child_project_active():
            try:
                self.sync_active_child_to_series()
            except Exception as exc:
                Messages.close_busy(busy_dialog, force=True)
                self.main.loading.setVisible(False)
                QtWidgets.QMessageBox.warning(
                    self.main,
                    self.main.tr("Save Series Project"),
                    self.main.tr("Failed to synchronize the active child project before saving.\n\n{error}").format(
                        error=str(exc)
                    ),
                )
                return False

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
            self._clear_recovery_loaded()
            self.main.set_project_clean()
            self.main.project_ctrl.clear_recovery_checkpoint()

        def on_error(error_tuple) -> None:
            Messages.close_busy(busy_dialog, force=True)
            self.main.default_error_handler(error_tuple)

        def on_finished() -> None:
            Messages.close_busy(busy_dialog)
            self.main.loading.setVisible(False)
            if post_save_callback is not None:
                post_save_callback()

        self.main.run_threaded(worker, on_result, on_error, on_finished)
        return True

    def _update_queue_runtime(
        self,
        *,
        queue_state: str | object = _UNSET,
        pause_requested: bool | object = _UNSET,
        pending_item_ids: list[str] | object = _UNSET,
        active_item_id: str | None | object = _UNSET,
        failed_item_ids: list[str] | object = _UNSET,
        skipped_item_ids: list[str] | object = _UNSET,
        failed_item_id: str | None | object = _UNSET,
        completed_item_ids: list[str] | object = _UNSET,
        retry_remaining_by_item: dict[str, int] | object = _UNSET,
        last_run_started_at: str | None | object = _UNSET,
        last_run_finished_at: str | None | object = _UNSET,
        last_run_summary: dict[str, object] | object = _UNSET,
    ) -> None:
        if not self.series_file:
            return
        self.series_manifest = update_series_queue_runtime(
            self.series_file,
            queue_state=queue_state,
            pause_requested=pause_requested,
            pending_item_ids=pending_item_ids,
            active_item_id=active_item_id,
            failed_item_ids=failed_item_ids,
            skipped_item_ids=skipped_item_ids,
            failed_item_id=failed_item_id,
            completed_item_ids=completed_item_ids,
            retry_remaining_by_item=retry_remaining_by_item,
            last_run_started_at=last_run_started_at,
            last_run_finished_at=last_run_finished_at,
            last_run_summary=last_run_summary,
        )
        # `update_series_queue_runtime` 이 이미 갱신된 매니페스트를 돌려주고,
        # 큐 런타임은 manifest 필드라 items 는 바뀌지 않는다. 예전에는 여기서
        # 파일을 통째로 다시 읽어, 큐 아이템이 넘어갈 때마다 GUI 스레드에서
        # load 가 두 번씩 돌았다.
        self.series_manifest = dict(self.series_manifest)
        # 큐 상태가 바뀌면 표시줄의 히스토리 잠금 사유도 함께 바뀐다.
        self._refresh_breadcrumb()

    def pause_queue_translation(self) -> None:
        if not self.series_file or not self._queue_active:
            return
        if self._pause_requested:
            return
        self._pause_requested = True
        self.main.pipeline_status_panel.set_series_queue_pause_visible(True, pause_requested=True)
        queue_runtime = self.active_queue_runtime()
        self._update_queue_runtime(
            queue_state=queue_runtime.get("queue_state") or "running",
            pause_requested=True,
            pending_item_ids=list(self._queue_pending_ids),
            active_item_id=queue_runtime.get("active_item_id"),
            failed_item_ids=list(self._queue_failed_ids),
            skipped_item_ids=list(self._queue_skipped_ids),
            failed_item_id=queue_runtime.get("failed_item_id"),
            completed_item_ids=list(self._queue_completed_ids),
            retry_remaining_by_item=dict(self._queue_retry_remaining),
            last_run_started_at=queue_runtime.get("last_run_started_at"),
            last_run_finished_at=queue_runtime.get("last_run_finished_at"),
            last_run_summary=queue_runtime.get("last_run_summary") or {},
        )
        request_batch_pause = getattr(self.main, "request_current_batch_pause", None)
        if callable(request_batch_pause):
            request_batch_pause()
        self._apply_workspace_state()

    def resume_queue_translation(self) -> None:
        if not self.series_file or self._queue_active:
            return
        queue_runtime = self.active_queue_runtime()
        if queue_runtime.get("queue_state") != "paused":
            return
        self._pause_requested = False
        self._queue_active = True
        pending_ids = [
            item_id
            for item_id in pending_series_item_ids(self.series_items)
            if str(item_id or "").strip()
        ]
        completed_ids = list(queue_runtime.get("completed_item_ids") or [])
        failed_ids = list(queue_runtime.get("failed_item_ids") or [])
        skipped_ids = list(queue_runtime.get("skipped_item_ids") or [])
        retry_remaining = dict(queue_runtime.get("retry_remaining_by_item") or {})
        self._queue_pending_ids = list(pending_ids)
        self._queue_completed_ids = list(completed_ids)
        self._queue_failed_ids = list(failed_ids)
        self._queue_skipped_ids = list(skipped_ids)
        self._queue_retry_remaining = dict(retry_remaining)
        self.main.pipeline_status_panel.set_series_queue_pause_visible(True, pause_requested=False)
        self._update_queue_runtime(
            queue_state="running",
            pause_requested=False,
            pending_item_ids=list(self._queue_pending_ids),
            active_item_id=None,
            failed_item_ids=list(self._queue_failed_ids),
            skipped_item_ids=list(self._queue_skipped_ids),
            failed_item_id=queue_runtime.get("failed_item_id"),
            completed_item_ids=list(self._queue_completed_ids),
            retry_remaining_by_item=dict(self._queue_retry_remaining),
            last_run_started_at=queue_runtime.get("last_run_started_at") or "",
            last_run_finished_at=None,
            last_run_summary=queue_runtime.get("last_run_summary") or {},
        )
        self._apply_workspace_state()
        self._run_next_queue_item()

    def open_last_failed_item(self) -> None:
        if not self.series_file or self._queue_active:
            return
        failed_item_id = str(self.active_queue_runtime().get("failed_item_id") or "").strip()
        if not failed_item_id:
            return
        self.request_open_item(failed_item_id)

    def open_active_queue_item(self) -> None:
        if not self.series_file:
            return
        active_item_id = str(self.active_queue_runtime().get("active_item_id") or "").strip()
        if not active_item_id:
            return
        if self.is_child_project_active() and str(self.active_child_item_id) == active_item_id:
            self.main.show_main_page()
            self._set_series_window_title(self._active_child_display_name())
            return
        self._open_item(active_item_id, push_history=False)

    def show_board_during_queue(self) -> None:
        if not self.series_file:
            return
        if self._queue_active and self.is_child_project_active():
            self._apply_workspace_state()
            self.main.show_series_page()
            self._set_series_window_title()
            return
        self.request_show_board()

    def _finalize_queue_summary(self, *, failed_count: int = 0, skipped_count: int = 0) -> dict[str, object]:
        queue_runtime = self.active_queue_runtime()
        started_at = str(queue_runtime.get("last_run_started_at") or "").strip() or None
        finished_at = QtCore.QDateTime.currentDateTime().toString(QtCore.Qt.DateFormat.ISODate)
        return build_series_run_summary(
            done_count=len(self._queue_completed_ids),
            failed_count=max(0, len(self._queue_failed_ids) + int(failed_count or 0)),
            skipped_count=max(0, len(self._queue_skipped_ids) + int(skipped_count or 0)),
            started_at=started_at,
            finished_at=finished_at,
        )

    def on_batch_process_finished(
        self,
        *,
        was_cancelled: bool,
        failed: bool,
        was_paused: bool = False,
    ) -> None:
        if not self.is_child_project_active():
            return

        try:
            self.sync_active_child_to_series()
        except Exception:
            # 배치 흐름 자체는 계속 살려 둔다. 다만 예전에는 아무 흔적도 남기지
            # 않고 빠져나가서, 방금 번역한 결과가 시리즈에 반영되지 않았다는
            # 사실을 사용자도 로그도 알 수 없었다.
            logger.warning(
                "Series sync after the batch run failed; the chapter result is not "
                "written back to the series project yet.",
                exc_info=True,
            )
            self._warn_child_sync_failed(self.active_child_temp_dir)
            return

        if not self._queue_active:
            return

        current_item_id = str(self.active_child_item_id or "")
        series_settings = normalize_series_settings(self.series_manifest.get("series_settings"))
        queue_runtime = self.active_queue_runtime()

        if was_paused:
            if current_item_id:
                self.series_items = update_series_item_status(
                    self.series_file,
                    series_item_id=current_item_id,
                    status="pending",
                )
                loaded = load_series_project(self.series_file)
                self.series_manifest = dict(loaded["manifest"])
                self.series_items = list(loaded["items"])
            pending_ids = pending_series_item_ids(self.series_items)
            self._queue_pending_ids = list(pending_ids)
            self._queue_active = False
            self._pause_requested = False
            self.main.pipeline_status_panel.set_series_queue_pause_visible(False, pause_requested=False)
            self._update_queue_runtime(
                queue_state="paused",
                pause_requested=False,
                pending_item_ids=list(self._queue_pending_ids),
                active_item_id=None,
                failed_item_ids=list(self._queue_failed_ids),
                skipped_item_ids=list(self._queue_skipped_ids),
                failed_item_id=queue_runtime.get("failed_item_id"),
                completed_item_ids=list(self._queue_completed_ids),
                retry_remaining_by_item=dict(self._queue_retry_remaining),
                last_run_started_at=queue_runtime.get("last_run_started_at"),
                last_run_finished_at=None,
                last_run_summary=queue_runtime.get("last_run_summary") or {},
            )
            self._apply_workspace_state()
            QtCore.QTimer.singleShot(0, self.main, lambda: self._show_board(push_history=False))
            return

        if was_cancelled:
            self._queue_active = False
            self._pause_requested = False
            self.main.pipeline_status_panel.set_series_queue_pause_visible(False, pause_requested=False)
            summary = self._finalize_queue_summary()
            self._update_queue_runtime(
                queue_state="idle",
                pause_requested=False,
                pending_item_ids=list(self._queue_pending_ids),
                active_item_id=None,
                failed_item_ids=list(self._queue_failed_ids),
                skipped_item_ids=list(self._queue_skipped_ids),
                failed_item_id=current_item_id or None,
                completed_item_ids=list(self._queue_completed_ids),
                retry_remaining_by_item=dict(self._queue_retry_remaining),
                last_run_started_at=queue_runtime.get("last_run_started_at"),
                last_run_finished_at=summary.get("finished_at"),
                last_run_summary=summary,
            )
            self._apply_workspace_state()
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
                self._update_queue_runtime(
                    queue_state="running",
                    pause_requested=bool(self._pause_requested),
                    pending_item_ids=list(self._queue_pending_ids),
                    active_item_id=current_item_id,
                    failed_item_ids=list(self._queue_failed_ids),
                    skipped_item_ids=list(self._queue_skipped_ids),
                    failed_item_id=None,
                    completed_item_ids=list(self._queue_completed_ids),
                    retry_remaining_by_item=dict(self._queue_retry_remaining),
                    last_run_started_at=queue_runtime.get("last_run_started_at"),
                    last_run_finished_at=None,
                    last_run_summary=queue_runtime.get("last_run_summary") or {},
                )
                QtCore.QTimer.singleShot(0, self.main, self._start_batch_for_active_child)
                return

            self.series_items = update_series_item_status(
                self.series_file,
                series_item_id=current_item_id,
                status="failed",
            )
            loaded = load_series_project(self.series_file)
            self.series_manifest = dict(loaded["manifest"])
            self.series_items = list(loaded["items"])

            if self._pause_requested:
                self._queue_failed_ids.append(current_item_id)
                self._queue_active = False
                self._pause_requested = False
                self.main.pipeline_status_panel.set_series_queue_pause_visible(False, pause_requested=False)
                self._update_queue_runtime(
                    queue_state="paused",
                    pause_requested=False,
                    pending_item_ids=pending_series_item_ids(self.series_items),
                    active_item_id=None,
                    failed_item_ids=list(self._queue_failed_ids),
                    skipped_item_ids=list(self._queue_skipped_ids),
                    failed_item_id=current_item_id,
                    completed_item_ids=list(self._queue_completed_ids),
                    retry_remaining_by_item=dict(self._queue_retry_remaining),
                    last_run_started_at=queue_runtime.get("last_run_started_at"),
                    last_run_finished_at=None,
                    last_run_summary=queue_runtime.get("last_run_summary") or {},
                )
                self._apply_workspace_state()
                QtCore.QTimer.singleShot(0, self.main, lambda: self._show_board(push_history=False))
                return

            if str(series_settings.get("queue_failure_policy")) == "skip":
                self._queue_skipped_ids.append(current_item_id)
                summary_runtime = self.active_queue_runtime()
                self._update_queue_runtime(
                    queue_state="running",
                    pause_requested=bool(self._pause_requested),
                    pending_item_ids=list(self._queue_pending_ids),
                    active_item_id=None,
                    failed_item_ids=list(self._queue_failed_ids),
                    skipped_item_ids=list(self._queue_skipped_ids),
                    failed_item_id=current_item_id,
                    completed_item_ids=list(self._queue_completed_ids),
                    retry_remaining_by_item=dict(self._queue_retry_remaining),
                    last_run_started_at=summary_runtime.get("last_run_started_at"),
                    last_run_finished_at=None,
                    last_run_summary=summary_runtime.get("last_run_summary") or {},
                )
                QtCore.QTimer.singleShot(0, self.main, self._run_next_queue_item)
                return

            self._queue_failed_ids.append(current_item_id)
            self._queue_active = False
            self._pause_requested = False
            self.main.pipeline_status_panel.set_series_queue_pause_visible(False, pause_requested=False)
            summary = self._finalize_queue_summary()
            self._update_queue_runtime(
                queue_state="paused",
                pause_requested=False,
                pending_item_ids=pending_series_item_ids(self.series_items),
                active_item_id=None,
                failed_item_ids=list(self._queue_failed_ids),
                skipped_item_ids=list(self._queue_skipped_ids),
                failed_item_id=current_item_id,
                completed_item_ids=list(self._queue_completed_ids),
                retry_remaining_by_item=dict(self._queue_retry_remaining),
                last_run_started_at=queue_runtime.get("last_run_started_at"),
                last_run_finished_at=summary.get("finished_at"),
                last_run_summary=summary,
            )
            self._apply_workspace_state()
            QtCore.QTimer.singleShot(0, self.main, lambda: self._show_board(push_history=False))
            return

        self._queue_completed_ids.append(current_item_id)
        self.series_items = update_series_item_status(
            self.series_file,
            series_item_id=current_item_id,
            status="done",
        )
        loaded = load_series_project(self.series_file)
        self.series_manifest = dict(loaded["manifest"])
        self.series_items = list(loaded["items"])
        if self._pause_requested:
            self._queue_active = False
            self._pause_requested = False
            self.main.pipeline_status_panel.set_series_queue_pause_visible(False, pause_requested=False)
            self._update_queue_runtime(
                queue_state="paused",
                pause_requested=False,
                pending_item_ids=pending_series_item_ids(self.series_items),
                active_item_id=None,
                failed_item_ids=list(self._queue_failed_ids),
                skipped_item_ids=list(self._queue_skipped_ids),
                failed_item_id=queue_runtime.get("failed_item_id"),
                completed_item_ids=list(self._queue_completed_ids),
                retry_remaining_by_item=dict(self._queue_retry_remaining),
                last_run_started_at=queue_runtime.get("last_run_started_at"),
                last_run_finished_at=None,
                last_run_summary=queue_runtime.get("last_run_summary") or {},
            )
            self._apply_workspace_state()
            QtCore.QTimer.singleShot(0, self.main, lambda: self._show_board(push_history=False))
            return
        self._update_queue_runtime(
            queue_state="running",
            pause_requested=False,
            pending_item_ids=list(self._queue_pending_ids),
            active_item_id=None,
            failed_item_ids=list(self._queue_failed_ids),
            skipped_item_ids=list(self._queue_skipped_ids),
            failed_item_id=None,
            completed_item_ids=list(self._queue_completed_ids),
            retry_remaining_by_item=dict(self._queue_retry_remaining),
            last_run_started_at=queue_runtime.get("last_run_started_at"),
            last_run_finished_at=None,
            last_run_summary=queue_runtime.get("last_run_summary") or {},
        )
        QtCore.QTimer.singleShot(0, self.main, self._run_next_queue_item)

    def start_queue_translation(self) -> None:
        if not self.series_file or self._queue_active or self.is_queue_paused():
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

        busy_dialog = Messages.show_busy(
            self.main,
            self.main.tr("Preparing automatic translation..."),
            title=self.main.tr("Series Project"),
            minimum_visible_ms=300,
        )
        self._queue_active = True
        try:
            self._pause_requested = False
            self.main.pipeline_status_panel.set_series_queue_pause_visible(True, pause_requested=False)
            self._queue_pending_ids = pending_ids
            self._queue_completed_ids = []
            self._queue_failed_ids = []
            self._queue_skipped_ids = []
            self._queue_retry_remaining = {}
            started_at = QtCore.QDateTime.currentDateTime().toString(QtCore.Qt.DateFormat.ISODate)
            self._update_queue_runtime(
                queue_state="running",
                pause_requested=False,
                pending_item_ids=list(self._queue_pending_ids),
                active_item_id=None,
                failed_item_ids=[],
                skipped_item_ids=[],
                failed_item_id=None,
                completed_item_ids=list(self._queue_completed_ids),
                retry_remaining_by_item={},
                last_run_started_at=started_at,
                last_run_finished_at=None,
                last_run_summary=self.active_queue_runtime().get("last_run_summary") or {},
            )
            self._apply_workspace_state()
            self._run_next_queue_item()
        finally:
            Messages.close_busy(busy_dialog)

    def _run_next_queue_item(self) -> None:
        if not self._queue_active:
            return
        if not self._queue_pending_ids:
            self._queue_active = False
            self._pause_requested = False
            self.main.pipeline_status_panel.set_series_queue_pause_visible(False, pause_requested=False)
            queue_runtime = self.active_queue_runtime()
            summary = self._finalize_queue_summary()
            self._update_queue_runtime(
                queue_state="idle",
                pause_requested=False,
                pending_item_ids=[],
                active_item_id=None,
                failed_item_ids=list(self._queue_failed_ids),
                skipped_item_ids=list(self._queue_skipped_ids),
                failed_item_id=None,
                completed_item_ids=list(self._queue_completed_ids),
                retry_remaining_by_item=dict(self._queue_retry_remaining),
                last_run_started_at=queue_runtime.get("last_run_started_at"),
                last_run_finished_at=summary.get("finished_at"),
                last_run_summary=summary,
            )
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
        loaded = load_series_project(self.series_file)
        self.series_manifest = dict(loaded["manifest"])
        self.series_items = list(loaded["items"])
        queue_runtime = self.active_queue_runtime()
        self._update_queue_runtime(
            queue_state="running",
            pause_requested=bool(self._pause_requested),
            pending_item_ids=list(self._queue_pending_ids),
            active_item_id=next_item_id,
            failed_item_ids=list(self._queue_failed_ids),
            skipped_item_ids=list(self._queue_skipped_ids),
            failed_item_id=None,
            completed_item_ids=list(self._queue_completed_ids),
            retry_remaining_by_item=dict(self._queue_retry_remaining),
            last_run_started_at=queue_runtime.get("last_run_started_at")
            or QtCore.QDateTime.currentDateTime().toString(QtCore.Qt.DateFormat.ISODate),
            last_run_finished_at=None,
            last_run_summary=queue_runtime.get("last_run_summary") or {},
        )
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

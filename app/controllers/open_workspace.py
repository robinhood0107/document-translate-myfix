from __future__ import annotations

import logging
import os
import shutil
import tempfile
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Callable

import numpy as np

from modules.utils.file_handler import FileHandler

logger = logging.getLogger(__name__)


@dataclass
class PreparedWorkspace:
    file_handler: FileHandler
    file_paths: list[str]
    first_image: np.ndarray | None
    _committed: bool = False
    _success_callbacks: list[Callable[[], None]] = field(default_factory=list)

    def cleanup(self) -> None:
        if not self._committed:
            self.file_handler.cleanup()

    def transfer_ownership(self, main: Any) -> None:
        self._committed = True
        main._workspace_owned_temp_roots = []

    def defer_success(self, callback: Callable[[], None]) -> None:
        self._success_callbacks.append(callback)

    def commit_success(self) -> None:
        for callback in self._success_callbacks:
            try:
                callback()
            except Exception:
                logger.warning("Workspace success cleanup failed.", exc_info=True)
        self._success_callbacks.clear()


class _StagedBatchReport:
    def __init__(self):
        self.latest_report: dict = {}

    def import_latest_report_from_project(self, report, refresh=False) -> None:
        self.latest_report = dict(report or {})


@dataclass
class ProjectLoadSnapshot:
    state: Any
    saved_context: str
    _committed: bool = False
    _success_callbacks: list[Callable[[], None]] = field(default_factory=list)

    def cleanup(self) -> None:
        if not self._committed:
            shutil.rmtree(str(self.state.temp_dir), ignore_errors=True)

    def transfer_ownership(self, main: Any) -> None:
        self._committed = True
        root = os.path.abspath(str(self.state.temp_dir))
        main._workspace_owned_temp_roots = [root]

    def defer_success(self, callback: Callable[[], None]) -> None:
        self._success_callbacks.append(callback)

    def commit_success(self) -> None:
        for callback in self._success_callbacks:
            try:
                callback()
            except Exception:
                logger.warning("Project success cleanup failed.", exc_info=True)
        self._success_callbacks.clear()

    def apply(self, main: Any) -> None:
        for attribute in (
            "curr_img_idx",
            "webtoon_mode",
            "image_files",
            "export_source_by_path",
            "image_states",
            "image_data",
            "image_history",
            "in_memory_history",
            "current_history_index",
            "displayed_images",
            "loaded_images",
            "image_patches",
            "project_output_preferences",
            "project_checkpoint_reference",
            "project_checkpoint_reference_persisted",
            "project_checkpoint_warning",
        ):
            if hasattr(self.state, attribute):
                setattr(main, attribute, getattr(self.state, attribute))
        main.image_viewer.webtoon_view_state = dict(
            getattr(self.state.image_viewer, "webtoon_view_state", {}) or {}
        )
        main.batch_report_ctrl.import_latest_report_from_project(
            getattr(self.state.batch_report_ctrl, "latest_report", {}),
            refresh=False,
        )


_WORKSPACE_STATE_ATTRIBUTES = (
    "curr_img_idx",
    "webtoon_mode",
    "image_files",
    "export_source_by_path",
    "image_states",
    "image_data",
    "image_history",
    "in_memory_history",
    "current_history_index",
    "displayed_images",
    "loaded_images",
    "image_patches",
    "in_memory_patches",
    "image_cards",
    "current_card",
    "blk_list",
    "undo_stacks",
    "project_file",
    "project_kind",
    "project_output_preferences",
    "project_checkpoint_reference",
    "project_checkpoint_reference_persisted",
    "project_checkpoint_warning",
    "_manual_dirty",
    "_dirty_revision",
)
_MUTATED_BY_CLEAR_STATE = frozenset(
    {
        "image_states",
        "image_data",
        "image_history",
        "in_memory_history",
        "current_history_index",
        "displayed_images",
        "image_patches",
        "in_memory_patches",
        "image_cards",
        "undo_stacks",
    }
)


class _WorkspaceCommitTransaction:
    """Preserve the current workspace until the staged UI commit succeeds."""

    def __init__(self, main: Any, prepared: Any):
        self.main = main
        self.prepared = prepared
        self.old_handler = getattr(main, "file_handler", None)
        self.old_values = {
            name: getattr(main, name)
            for name in _WORKSPACE_STATE_ATTRIBUTES
            if hasattr(main, name)
        }
        self.old_owned_roots = list(
            getattr(main, "_workspace_owned_temp_roots", []) or []
        )
        try:
            self.old_title = str(main.windowTitle())
        except Exception:
            self.old_title = ""
        self._participant_rollback = None
        capture = getattr(prepared, "capture_for_commit", None)
        if callable(capture):
            self._participant_rollback = capture(main)

    def isolate_mutable_state(self) -> None:
        # clear_state() clears several containers in place. Give it fresh
        # containers so rollback can restore the previous objects without a
        # full-resolution deep copy.
        for name in _MUTATED_BY_CLEAR_STATE:
            if name not in self.old_values:
                continue
            value = self.old_values[name]
            if isinstance(value, dict):
                replacement = {}
            elif isinstance(value, set):
                replacement = set()
            else:
                replacement = []
            setattr(self.main, name, replacement)

    def _restore_ui_best_effort(self) -> None:
        try:
            if self.old_title:
                self.main.setWindowTitle(self.old_title)
        except Exception:
            pass
        try:
            self.main.image_ctrl.update_image_cards()
        except Exception:
            pass
        try:
            index = int(getattr(self.main, "curr_img_idx", -1))
            if index >= 0:
                self.main.page_list.setCurrentRow(index)
                files = list(getattr(self.main, "image_files", []) or [])
                if index < len(files):
                    image = (getattr(self.main, "image_data", {}) or {}).get(files[index])
                    if image is not None:
                        self.main.image_ctrl.display_image_from_loaded(
                            image,
                            index,
                            switch_page=False,
                        )
        except Exception:
            pass
        try:
            self.main.batch_report_ctrl.refresh_action_buttons()
        except Exception:
            pass

    def rollback(self) -> None:
        current_handler = getattr(self.main, "file_handler", None)
        for name, value in self.old_values.items():
            setattr(self.main, name, value)
        self.main.file_handler = self.old_handler
        self.main._workspace_owned_temp_roots = list(self.old_owned_roots)
        if callable(self._participant_rollback):
            self._participant_rollback()
        self._restore_ui_best_effort()
        cleanup = getattr(self.prepared, "cleanup", None)
        if callable(cleanup):
            cleanup()
        elif current_handler is not None and current_handler is not self.old_handler:
            try:
                current_handler.cleanup()
            except Exception:
                logger.debug("Could not clean the failed staged file handler.", exc_info=True)

    def commit(self) -> None:
        transfer = getattr(self.prepared, "transfer_ownership", None)
        if callable(transfer):
            transfer(self.main)

        current_handler = getattr(self.main, "file_handler", None)
        if self.old_handler is not None and current_handler is not self.old_handler:
            try:
                self.old_handler.cleanup()
            except Exception:
                logger.warning("Could not clean the previous file workspace.", exc_info=True)

        old_undo_stacks = self.old_values.get("undo_stacks")
        current_undo_stacks = getattr(self.main, "undo_stacks", None)
        if isinstance(old_undo_stacks, dict) and current_undo_stacks is not old_undo_stacks:
            detach = getattr(self.main, "_detach_undo_stack", None)
            if callable(detach):
                for stack in list(old_undo_stacks.values()):
                    try:
                        detach(stack)
                    except Exception:
                        logger.debug("Could not detach a previous undo stack.", exc_info=True)
            old_undo_stacks.clear()

        retained_roots = {
            os.path.abspath(str(root))
            for root in (getattr(self.main, "_workspace_owned_temp_roots", []) or [])
        }
        for root in self.old_owned_roots:
            resolved = os.path.abspath(str(root))
            if resolved not in retained_roots:
                shutil.rmtree(resolved, ignore_errors=True)

        commit_success = getattr(self.prepared, "commit_success", None)
        if callable(commit_success):
            commit_success()


class OpenWorkspaceCoordinator:
    """Own the embedded open-progress surface and one active open job."""

    def __init__(self, main: Any):
        self.main = main
        self.overlay = main.open_workspace_overlay
        self.active = False
        self._failed = False
        self._transaction: _WorkspaceCommitTransaction | None = None
        self._prepared_result: Any = None
        self.overlay.close_requested.connect(self.close_failure)

    def begin(self, message: str) -> bool:
        if self.active:
            return False
        self.active = True
        self._failed = False
        self.main._set_project_navigation_enabled(False)
        self.main.set_runtime_editing_locked(True)
        self.main._update_runtime_surface_geometry()
        self.overlay.start(message)
        return True

    def progress(self, event: dict) -> None:
        if not self.active:
            return
        self.overlay.update_progress(dict(event or {}))

    def complete(self) -> None:
        transaction = self._transaction
        self._transaction = None
        self._prepared_result = None
        if transaction is not None:
            transaction.commit()
        self.active = False
        self._failed = False
        self.overlay.hide()
        self.main.set_runtime_editing_locked(False)
        self.main._set_project_navigation_enabled(True)

    def fail(self, error_tuple: tuple) -> None:
        _exctype, value, traceback_text = error_tuple
        logger.error("Workspace open failed: %s\n%s", value, traceback_text)
        transaction = self._transaction
        self._transaction = None
        prepared = self._prepared_result
        self._prepared_result = None
        if transaction is not None:
            transaction.rollback()
        elif prepared is not None:
            cleanup = getattr(prepared, "cleanup", None)
            if callable(cleanup):
                cleanup()
        self.active = False
        self._failed = True
        self.main.set_runtime_editing_locked(False)
        self.main._set_project_navigation_enabled(True)
        message = str(value or self.main.tr("The selected workspace could not be opened."))
        first_line = next((line.strip() for line in message.splitlines() if line.strip()), message)
        self.overlay.show_failure(
            first_line,
            self.main.tr("See the application log for technical details."),
        )

    def close_failure(self) -> None:
        if self._failed:
            self._failed = False
            self.overlay.hide()

    def run(
        self,
        *,
        message: str,
        prepare: Callable,
        commit: Callable,
    ) -> bool:
        if not self.begin(message):
            return False

        prepared_result: dict[str, Any] = {"value": None}

        def on_result(result: Any) -> None:
            prepared_result["value"] = result
            self._prepared_result = result
            self.progress(
                {
                    "stage": "apply",
                    "message": self.main.tr("Building the workspace..."),
                }
            )
            try:
                self._transaction = _WorkspaceCommitTransaction(self.main, result)
                self._transaction.isolate_mutable_state()
                commit_result = commit(result)
            except Exception as exc:
                import traceback

                self.fail((type(exc), exc, traceback.format_exc()))
                return
            if commit_result is not False:
                self.complete()

        def on_error(error_tuple: tuple) -> None:
            value = prepared_result.get("value")
            self._prepared_result = value
            self.fail(error_tuple)

        self.main.run_threaded_with_progress(
            prepare,
            self.progress,
            on_result,
            on_error,
            None,
        )
        return True

    @staticmethod
    def prepare_regular_files(
        report_progress: Callable[[dict], None],
        main: Any,
        paths: list[str],
    ) -> PreparedWorkspace:
        staged_handler = FileHandler()
        try:
            report_progress(
                {"stage": "validate", "message": main.tr("Checking selected files...")}
            )
            file_paths = staged_handler.prepare_files(
                paths,
                progress_callback=report_progress,
            )
            first_image = None
            if file_paths:
                report_progress(
                    {
                        "stage": "decode",
                        "message": main.tr("Preparing the first page preview..."),
                        "detail": os.path.basename(file_paths[0]),
                    }
                )
                from app.path_materialization import ensure_path_materialized
                import imkit as imk

                ensure_path_materialized(file_paths[0])
                first_image = imk.read_image(file_paths[0])
                if first_image is None:
                    raise RuntimeError(main.tr("The first page could not be decoded."))
            return PreparedWorkspace(staged_handler, file_paths, first_image)
        except Exception:
            staged_handler.cleanup()
            raise

    @staticmethod
    def prepare_project(
        report_progress: Callable[[dict], None],
        main: Any,
        file_name: str,
    ) -> ProjectLoadSnapshot:
        from app.projects.project_state import load_state_from_proj_file

        report_progress(
            {
                "stage": "validate",
                "message": main.tr("Checking the project file..."),
                "detail": os.path.basename(file_name),
            }
        )
        staging_root = tempfile.mkdtemp(prefix="open_project_", dir=main.temp_dir)
        staging = SimpleNamespace(
            curr_img_idx=-1,
            webtoon_mode=False,
            image_files=[],
            export_source_by_path={},
            image_states={},
            image_data={},
            image_history={},
            in_memory_history={},
            current_history_index={},
            displayed_images=set(),
            loaded_images=[],
            image_patches={},
            project_output_preferences={},
            project_checkpoint_reference={},
            project_checkpoint_reference_persisted=False,
            project_checkpoint_warning="",
            temp_dir=staging_root,
            image_viewer=SimpleNamespace(webtoon_view_state={}),
            batch_report_ctrl=_StagedBatchReport(),
            settings_page=main.settings_page,
        )
        try:
            report_progress(
                {
                    "stage": "index",
                    "message": main.tr("Reading project pages..."),
                    "detail": os.path.basename(file_name),
                }
            )
            saved_context = load_state_from_proj_file(staging, file_name)
            return ProjectLoadSnapshot(staging, saved_context)
        except Exception:
            shutil.rmtree(staging_root, ignore_errors=True)
            raise

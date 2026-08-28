from __future__ import annotations

import logging
import os
import shutil
import tempfile
from dataclasses import dataclass
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

    def cleanup(self) -> None:
        self.file_handler.cleanup()


class _StagedBatchReport:
    def __init__(self):
        self.latest_report: dict = {}

    def import_latest_report_from_project(self, report, refresh=False) -> None:
        self.latest_report = dict(report or {})


@dataclass
class ProjectLoadSnapshot:
    state: Any
    saved_context: str

    def cleanup(self) -> None:
        shutil.rmtree(str(self.state.temp_dir), ignore_errors=True)

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


class OpenWorkspaceCoordinator:
    """Own the embedded open-progress surface and one active open job."""

    def __init__(self, main: Any):
        self.main = main
        self.overlay = main.open_workspace_overlay
        self.active = False
        self._failed = False
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
        self.active = False
        self._failed = False
        self.overlay.hide()
        self.main.set_runtime_editing_locked(False)
        self.main._set_project_navigation_enabled(True)

    def fail(self, error_tuple: tuple) -> None:
        _exctype, value, traceback_text = error_tuple
        logger.error("Workspace open failed: %s\n%s", value, traceback_text)
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
            self.progress(
                {
                    "stage": "apply",
                    "message": self.main.tr("Building the workspace..."),
                }
            )
            try:
                commit_result = commit(result)
            except Exception as exc:
                cleanup = getattr(result, "cleanup", None)
                if callable(cleanup):
                    cleanup()
                import traceback

                self.fail((type(exc), exc, traceback.format_exc()))
                return
            if commit_result is not False:
                self.complete()

        def on_error(error_tuple: tuple) -> None:
            value = prepared_result.get("value")
            cleanup = getattr(value, "cleanup", None)
            if callable(cleanup):
                cleanup()
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

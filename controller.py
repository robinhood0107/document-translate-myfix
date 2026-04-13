import logging
import os
import requests
import numpy as np
import shutil
import tempfile
import traceback
from typing import Any, Callable, Tuple

from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import QCoreApplication, QThreadPool
from PySide6.QtGui import QUndoGroup, QUndoStack, QIcon

from app.ui.dayu_widgets.qt import MPixmap
from app.ui.main_window import ComicTranslateUI
from app.ui.messages import Messages

from modules.ocr.local_runtime import LocalOCRRuntimeManager
from modules.translation.local_runtime import LocalGemmaRuntimeManager
from modules.ocr.selection import OCR_MODE_BEST_LOCAL, normalize_ocr_mode, resolve_ocr_engine
from app.ui.canvas.text_item import TextBlockItem
from app.ui.commands.box import DeleteBoxesCommand

from modules.utils.textblock import TextBlock
from modules.utils.file_handler import FileHandler
from modules.utils.pipeline_config import validate_settings
from modules.utils.automatic_progress import AutomaticProgressTracker
from modules.utils.download import set_download_callback
from modules.utils.notification_sound import SYSTEM_SOUND_MODE, notify_pipeline_event, play_completion_sound
from modules.utils.txt_md_exchange import (
    apply_translation_pages,
    collect_page_entries,
    dump_exchange_text,
    find_duplicate_page_names,
    page_name_from_path,
    parse_translation_exchange_file,
)
from pipeline.main_pipeline import ComicTranslatePipeline

from app.controllers.image import ImageStateController
from app.controllers.rect_item import RectItemController
from app.controllers.projects import ProjectController
from app.controllers.text import TextController
from app.controllers.webtoons import WebtoonController
from app.controllers.search_replace import SearchReplaceController
from app.controllers.shortcuts import ShortcutController
from app.controllers.task_runner import TaskRunnerController
from app.controllers.batch_report import BatchReportController
from app.controllers.manual_workflow import ManualWorkflowController
from modules.utils.exceptions import (
    LocalServiceError,
    LocalServiceConnectionError,
    LocalServiceSetupError,
    OperationCancelledError,
)


logger = logging.getLogger(__name__)


def _env_enabled(name: str) -> bool:
    value = str(os.environ.get(name, "") or "").strip().lower()
    return value in {"1", "true", "yes", "on"}


# Toggle memory logging / benchmark diagnostics from the environment.
ENABLE_MEMLOGGER = _env_enabled("CT_ENABLE_MEMLOG") or _env_enabled("CT_ENABLE_GPU_BENCH")
DISABLE_BACKGROUND_UPDATE_CHECK = _env_enabled("CT_DISABLE_UPDATE_CHECK")

class ComicTranslate(ComicTranslateUI):
    image_processed = QtCore.Signal(int, object, str)
    patches_processed = QtCore.Signal(list, str)
    progress_update = QtCore.Signal(int, int, int, int, bool)
    runtime_progress_update = QtCore.Signal(dict)
    image_skipped = QtCore.Signal(str, str, str)
    blk_rendered = QtCore.Signal(str, int, object, str)
    render_state_ready = QtCore.Signal(str)
    download_event = QtCore.Signal(str, str)  # status, name

    def __init__(self, parent=None):
        super(ComicTranslate, self).__init__(parent)
        self.setWindowTitle("Project1.ctpr[*]")

        # Memory logging toggle for local diagnostics.
        # Start as early as possible after QWidget init so we can attribute idle RSS.
        self._memlogger = None
        if ENABLE_MEMLOGGER:
            try:
                from modules.utils.memlog import MemLogger

                self._memlogger = MemLogger(self)
                self._memlogger.start()
                self._memlogger.emit("after_super_init")
            except Exception:
                self._memlogger = None

        # Explicitly set window icon to ensure it persists after splash screen
        current_file_dir = os.path.dirname(os.path.abspath(__file__))
        icon_path = os.path.join(current_file_dir, 'resources', 'icons', 'icon.ico')
        self.setWindowIcon(QIcon(icon_path))

        self.blk_list: list[TextBlock] = []   
        self.curr_tblock: TextBlock = None
        self.curr_tblock_item: TextBlockItem = None     

        self.image_files = []
        self.selected_batch = []
        self.curr_img_idx = -1
        self.image_states = {}
        self.image_data = {}  # Store the latest version of each image
        self.image_history = {}  # Store file path history for all images
        self.in_memory_history = {}  # Store image history for recent images
        self.current_history_index = {}  # Current position in the history for each image
        self.displayed_images = set()  # Set to track displayed images
        self.image_patches = {}  # Store patches for each image
        self.in_memory_patches = {}  # Store patches in memory for each image
        self.image_cards = []
        self.current_card = None
        self.max_images_in_memory = 5
        self.loaded_images = []

        self.undo_group = QUndoGroup(self)
        self.undo_stacks: dict[str, QUndoStack] = {}
        self.project_file = None
        self.temp_dir = tempfile.mkdtemp()
        self._manual_dirty = False
        self._dirty_revision = 0
        self._skip_close_prompt = False

        self.pipeline = ComicTranslatePipeline(self)
        try:
            if self._memlogger is not None:
                self._memlogger.emit("after_pipeline_init")
        except Exception:
            pass
        self.file_handler = FileHandler()
        self.threadpool = QThreadPool()
        self.current_worker = None
        self._batch_active = False
        self._batch_cancel_requested = False
        self._batch_failed = False
        self._current_batch_run_type = None
        self._last_batch_request_paths = []
        self._last_batch_run_type = "batch"
        self._automatic_progress_settings_target = None
        self._last_runtime_preview_path = ""
        self._last_batch_output_root = ""
        self._automatic_progress_tracker = AutomaticProgressTracker()
        self.local_ocr_runtime_manager = LocalOCRRuntimeManager()
        self.local_translation_runtime_manager = LocalGemmaRuntimeManager()

        self.image_ctrl = ImageStateController(self)
        self.rect_item_ctrl = RectItemController(self)
        self.project_ctrl = ProjectController(self)
        self.text_ctrl = TextController(self)
        self.webtoon_ctrl = WebtoonController(self)
        self.search_ctrl = SearchReplaceController(self)
        self.shortcut_ctrl = ShortcutController(self)
        self.task_runner_ctrl = TaskRunnerController(self)
        self.batch_report_ctrl = BatchReportController(self)
        self.manual_workflow_ctrl = ManualWorkflowController(self)
        try:
            if self._memlogger is not None:
                self._memlogger.emit("after_controllers_init")
        except Exception:
            pass

        self.image_skipped.connect(self.image_ctrl.on_image_skipped)
        self.image_processed.connect(self.image_ctrl.on_image_processed)
        self.patches_processed.connect(self.image_ctrl.on_inpaint_patches_processed)
        self.progress_update.connect(self.update_progress)
        self.blk_rendered.connect(self.text_ctrl.on_blk_rendered)
        self.render_state_ready.connect(self.image_ctrl.on_render_state_ready)
        self.render_state_ready.connect(self.project_ctrl._on_batch_page_done)
        self.download_event.connect(self.on_download_event)
        self.runtime_progress_update.connect(self.on_runtime_progress_update)

        self.connect_ui_elements()
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)

        self.project_ctrl.load_main_page_settings()
        self.settings_page.load_settings()
        self.refresh_inpaint_tool_ui()
        self.project_ctrl.initialize_autosave()

        # Populate the home screen with any previously-saved recent projects
        self.startup_home.populate(self.project_ctrl.get_recent_projects())
        
        # Check for updates in background
        if not DISABLE_BACKGROUND_UPDATE_CHECK:
            self.settings_page.check_for_updates(is_background=True)

        self._processing_page_change = False  # Flag to prevent recursive page change handling

        # Hook the global download callback so utils can notify us
        def _dl_cb(status: str, name: str):
            # Ensure cross-thread safe emit
            try:
                self.download_event.emit(status, name)
            except Exception:
                pass
        set_download_callback(_dl_cb)

    def emit_memlog(self, tag: str, **extra):
        try:
            if self._memlogger is not None:
                self._memlogger.emit(tag, extra=extra or None)
        except Exception:
            pass

    def connect_ui_elements(self):
        # Browsers
        self.image_browser_button.sig_files_changed.connect(self.image_ctrl.thread_load_images)
        self.document_browser_button.sig_files_changed.connect(self.image_ctrl.thread_load_images)
        self.archive_browser_button.sig_files_changed.connect(self.image_ctrl.thread_load_images)
        self.comic_browser_button.sig_files_changed.connect(self.image_ctrl.thread_load_images)
        self.psd_browser_button.sig_files_changed.connect(self.image_ctrl.thread_load_images)
        self.project_browser_button.sig_file_changed.connect(self.project_ctrl.thread_load_project)
        self.insert_browser_button.sig_files_changed.connect(self.image_ctrl.thread_insert)

        self.save_browser.sig_file_changed.connect(self.image_ctrl.save_current_image)
        self.save_all_browser.sig_file_changed.connect(self.project_ctrl.save_and_make)
        self.save_project_button.clicked.connect(self.project_ctrl.thread_save_project)
        self.save_as_project_button.clicked.connect(self.project_ctrl.thread_save_as_project)
        self.drag_browser.sig_files_changed.connect(self._guarded_thread_load_images)
       
        self.manual_radio.clicked.connect(self.manual_mode_selected)
        self.automatic_radio.clicked.connect(self.batch_mode_selected)
        
        # Webtoon mode toggle
        self.webtoon_toggle.clicked.connect(self.webtoon_ctrl.toggle_webtoon_mode)

        # Connect buttons from button_groups
        self.hbutton_group.get_button_group().buttons()[0].clicked.connect(lambda: self.block_detect())
        self.hbutton_group.get_button_group().buttons()[1].clicked.connect(self.ocr)
        self.hbutton_group.get_button_group().buttons()[2].clicked.connect(self.translate_image)
        self.hbutton_group.get_button_group().buttons()[3].clicked.connect(self.load_segmentation_points)
        self.hbutton_group.get_button_group().buttons()[4].clicked.connect(self.inpaint_and_set)
        self.hbutton_group.get_button_group().buttons()[5].clicked.connect(self.text_ctrl.render_text)

        self.undo_tool_group.get_button_group().buttons()[0].clicked.connect(self.undo_group.undo)
        self.undo_tool_group.get_button_group().buttons()[1].clicked.connect(self.undo_group.redo)

        # Connect other buttons and widgets
        self.translate_button.clicked.connect(self.start_batch_process)
        self.cancel_button.clicked.connect(self.cancel_current_task)
        self.batch_report_button.clicked.connect(self.show_latest_batch_report)
        self.retry_failed_button.clicked.connect(self.retry_failed_batch_pages)
        self.one_page_auto_button.clicked.connect(self.start_one_page_auto_process)
        self.pipeline_status_panel.cancel_requested.connect(self._on_automatic_progress_cancel)
        self.pipeline_status_panel.retry_requested.connect(self._on_automatic_progress_retry)
        self.pipeline_status_panel.open_settings_requested.connect(self._on_automatic_progress_open_settings)
        self.pipeline_status_panel.report_requested.connect(self.show_latest_batch_report)
        self.pipeline_status_panel.open_output_requested.connect(self._open_latest_batch_output)
        self.set_all_button.clicked.connect(self.text_ctrl.set_src_trg_all)
        self.clear_rectangles_button.clicked.connect(self.image_viewer.clear_rectangles)
        self.clear_brush_strokes_button.clicked.connect(self.image_viewer.clear_brush_strokes)
        self.export_source_txt_button.clicked.connect(lambda: self.export_source_exchange(".txt"))
        self.import_translation_txt_button.clicked.connect(lambda: self.import_translation_exchange(".txt"))
        self.export_source_md_button.clicked.connect(lambda: self.export_source_exchange(".md"))
        self.import_translation_md_button.clicked.connect(lambda: self.import_translation_exchange(".md"))
        self.draw_blklist_blks.clicked.connect(self.restore_text_blocks)
        self.change_all_blocks_size_dec.clicked.connect(lambda: self.text_ctrl.change_all_blocks_size(-int(self.change_all_blocks_size_diff.text())))
        self.change_all_blocks_size_inc.clicked.connect(lambda: self.text_ctrl.change_all_blocks_size(int(self.change_all_blocks_size_diff.text())))
        self.delete_button.clicked.connect(self.delete_selected_box)
        for checkbox in (
            self.auto_export_source_txt_checkbox,
            self.auto_export_source_md_checkbox,
            self.auto_export_translation_txt_checkbox,
            self.auto_export_translation_md_checkbox,
        ):
            checkbox.stateChanged.connect(self.settings_page._save_settings_if_not_loading)

        # Connect text edit widgets
        self.s_text_edit.textChanged.connect(self.text_ctrl.update_text_block)
        self.t_text_edit.textChanged.connect(self.text_ctrl.update_text_block_from_edit)

        self.s_combo.currentTextChanged.connect(self.text_ctrl.save_src_trg)
        self.t_combo.currentTextChanged.connect(self.text_ctrl.save_src_trg)

        # Connect image viewer signals for both modes
        self.image_viewer.rectangle_selected.connect(self.rect_item_ctrl.handle_rectangle_selection)
        self.image_viewer.rectangle_created.connect(self.rect_item_ctrl.handle_rectangle_creation)
        self.image_viewer.rectangle_deleted.connect(self.rect_item_ctrl.handle_rectangle_deletion)
        self.image_viewer.command_emitted.connect(self.push_command)
        self.image_viewer.restore_stroke_requested.connect(self.image_ctrl.apply_restore_stroke)
        self.image_viewer.connect_rect_item.connect(self.rect_item_ctrl.connect_rect_item_signals)
        self.image_viewer.connect_text_item.connect(self.text_ctrl.connect_text_item_signals)
        self.image_viewer.page_changed.connect(self.webtoon_ctrl.on_page_changed)
        self.image_viewer.page_changed.connect(lambda _page_index: self.refresh_inpaint_tool_ui())
        self.image_viewer.clear_text_edits.connect(self.text_ctrl.clear_text_edits)

        try:
            if self._memlogger is not None:
                self._memlogger.emit("after_signal_wiring")
        except Exception:
            pass

        # Rendering
        self.font_dropdown.currentTextChanged.connect(self.text_ctrl.on_font_dropdown_change)
        self.font_size_dropdown.currentTextChanged.connect(self.text_ctrl.on_font_size_change)
        self.line_spacing_dropdown.currentTextChanged.connect(self.text_ctrl.on_line_spacing_change)
        self.block_font_color_button.clicked.connect(self.text_ctrl.on_font_color_change)
        self.alignment_tool_group.get_button_group().buttons()[0].clicked.connect(self.text_ctrl.left_align)
        self.alignment_tool_group.get_button_group().buttons()[1].clicked.connect(self.text_ctrl.center_align)
        self.alignment_tool_group.get_button_group().buttons()[2].clicked.connect(self.text_ctrl.right_align)
        self.vertical_alignment_tool_group.sig_checked_changed.connect(self.text_ctrl.on_vertical_alignment_changed)
        self.bold_button.clicked.connect(self.text_ctrl.bold)
        self.italic_button.clicked.connect(self.text_ctrl.italic)
        self.underline_button.clicked.connect(self.text_ctrl.underline)
        self.outline_font_color_button.clicked.connect(self.text_ctrl.on_outline_color_change)
        self.outline_width_dropdown.currentTextChanged.connect(self.text_ctrl.on_outline_width_change)
        self.outline_checkbox.stateChanged.connect(self.text_ctrl.toggle_outline_settings)
        self.outline_checkbox.stateChanged.connect(self.text_ctrl.sync_outline_mode_group)
        self.outline_mode_group.sig_checked_changed.connect(self.text_ctrl.on_outline_mode_group_changed)
        self.text_ctrl.sync_outline_mode_group(2 if self.outline_checkbox.isChecked() else 0)

        # Page List
        self.page_list.currentItemChanged.connect(self.image_ctrl.on_card_selected)
        self.page_list.selection_changed.connect(self.image_ctrl.on_selection_changed)
        self.page_list.order_changed.connect(self.image_ctrl.handle_image_reorder)
        self.page_list.del_img.connect(self.image_ctrl.handle_image_deletion)
        self.page_list.insert_browser.sig_files_changed.connect(self.image_ctrl.thread_insert)
        self.page_list.toggle_skip_img.connect(self.image_ctrl.handle_toggle_skip_images)
        self.page_list.translate_imgs.connect(self.batch_translate_selected)
        self.page_list.sort_requested.connect(self.image_ctrl.sort_images)

        # New project and safety confirmations
        self.new_project_button.clicked.connect(self._on_new_project_clicked)

        # Home screen signals
        self.startup_home.sig_open_files.connect(self._guarded_thread_load_images)
        self.startup_home.sig_open_project.connect(self._open_project_from_home)
        self.startup_home._sig_remove_one.connect(self._on_home_remove_recent)
        self.startup_home._sig_clear_all.connect(self._on_home_clear_recent)
        self.startup_home._sig_pin.connect(
            lambda path, pinned: self.project_ctrl.toggle_pin_project(path, pinned)
        )
        self.title_bar.project_target_requested.connect(self.project_ctrl.thread_change_project_file)

    def _guarded_thread_load_images(self, paths: list[str]):
        """Wrap thread_load_images with unsaved-project confirmation and clear state."""
        if not paths:
            # Empty list = "New Project" action from the home screen
            self._on_new_project_clicked()
            return
        if not self._confirm_start_new_project():
            return
        self.image_ctrl.thread_load_images(paths)

    def _on_new_project_clicked(self):
        """Clear the app to initial state after confirmation."""
        if not self._confirm_start_new_project():
            return
        self.project_ctrl.clear_recovery_checkpoint()
        # Clear state and switch to the main editor showing the drag area
        self.image_ctrl.clear_state()
        self.central_stack.setCurrentWidget(self.drag_browser)
        self.show_main_page()
        self.project_ctrl.ensure_autosave_project_file_for_new_project()
        # Reset webtoon mode UI state
        if self.webtoon_mode:
            self.webtoon_toggle.setChecked(False)
        self.webtoon_mode = False

    # Home screen helper methods

    def _open_project_from_home(self, path: str):
        """Load a .ctpr project selected on the home screen."""
        if not self._confirm_start_new_project():
            return
        if not path or not path.lower().endswith(".ctpr"):
            # Treat as generic files
            self._guarded_thread_load_images([path])
            return
        self.project_ctrl.thread_load_project(path)
        self.show_main_page()

    def _on_home_remove_recent(self, path: str):
        """Persist removal of one entry from the recent list."""
        self.project_ctrl.remove_recent_project(path)

    def _on_home_clear_recent(self):
        """Persist clearing of the entire recent list."""
        self.project_ctrl.clear_recent_projects()

    def connect_rect_item_signals(self, rect_item, force_reconnect: bool = False): return self.rect_item_ctrl.connect_rect_item_signals(rect_item, force_reconnect=force_reconnect)
    def apply_inpaint_patches(self, patches): return self.image_ctrl.apply_inpaint_patches(patches)
    def render_settings(self): return self.text_ctrl.render_settings()
    def load_image(self, file_path: str) -> np.ndarray: return self.image_ctrl.load_image(file_path)
    def get_selected_page_paths(self) -> list[str]:
        selected_paths: list[str] = []
        seen: set[str] = set()
        for path in self.page_list.selected_file_paths():
            if isinstance(path, str) and path in self.image_files and path not in seen:
                selected_paths.append(path)
                seen.add(path)
        return selected_paths

    def _any_undo_dirty(self) -> bool:
        for stack in self.undo_stacks.values():
            try:
                if stack and not stack.isClean():
                    return True
            except Exception:
                continue
        return False

    def has_unsaved_changes(self) -> bool:
        return bool(self._manual_dirty) or self._any_undo_dirty()

    def _bump_dirty_revision(self, *_):
        self._dirty_revision += 1
        try:
            self.project_ctrl.notify_project_dirty_revision_changed()
        except Exception:
            pass

    def mark_project_dirty(self):
        self._bump_dirty_revision()
        self._manual_dirty = True
        self._update_window_modified()

    def set_project_clean(self):
        self._manual_dirty = False
        for stack in self.undo_stacks.values():
            try:
                stack.setClean()
            except Exception:
                continue
        self._update_window_modified()

    def _update_window_modified(self):
        try:
            self.setWindowModified(self.has_unsaved_changes())
        except Exception:
            pass

    def _finish_close_after_save(self):
        self._skip_close_prompt = True
        self.close()

    def push_command(self, command):
        if self.undo_group.activeStack():
            self.undo_group.activeStack().push(command)

    def delete_selected_box(self):
        if self.curr_tblock:
            # Create and push the delete command
            command = DeleteBoxesCommand(
                self,
                self.image_viewer.selected_rect,
                self.curr_tblock_item,
                self.curr_tblock,
                self.blk_list,
            )
            self.undo_group.activeStack().push(command)

    def restore_text_blocks(self):
        if not self.webtoon_mode:
            if self.blk_list:
                self.pipeline.load_box_coords(self.blk_list)
            return

        manager = getattr(self.image_viewer, "webtoon_manager", None)
        page_idx = self.curr_img_idx
        if manager is None or not (0 <= page_idx < len(self.image_files)):
            if self.blk_list:
                self.pipeline.load_box_coords(self.blk_list)
            return

        page_y = manager.image_positions[page_idx]
        page_bottom = page_y + manager.image_heights[page_idx]

        current_page_blocks = []
        for blk in self.blk_list:
            if blk.xyxy is None or len(blk.xyxy) < 4:
                continue
            blk_y = blk.xyxy[1]
            blk_bottom = blk.xyxy[3]
            if (
                (blk_y >= page_y and blk_y < page_bottom)
                or (blk_bottom > page_y and blk_bottom <= page_bottom)
                or (blk_y < page_y and blk_bottom > page_bottom)
            ):
                current_page_blocks.append(blk)

        if current_page_blocks:
            self.pipeline.load_box_coords(current_page_blocks)

    def batch_mode_selected(self):
        self.disable_hbutton_group()
        self.translate_button.setEnabled(True)
        self.cancel_button.setEnabled(True)
        self.batch_report_ctrl.refresh_action_buttons()

    def manual_mode_selected(self):
        self.enable_hbutton_group()
        self.translate_button.setEnabled(False)
        self.cancel_button.setEnabled(False)
        self.batch_report_ctrl.refresh_action_buttons()

    def on_manual_finished(self):
        self.loading.setVisible(False)
        self.enable_hbutton_group()

    def run_threaded(self, callback: Callable, result_callback: Callable=None,
                    error_callback: Callable=None, finished_callback: Callable=None,
                    *args, **kwargs):
        return self.task_runner_ctrl.run_threaded(
            callback, result_callback, error_callback, finished_callback, *args, **kwargs
        )

    def run_threaded_immediate(self, callback: Callable, result_callback: Callable=None,
                              error_callback: Callable=None, finished_callback: Callable=None,
                              *args, **kwargs):
        return self.task_runner_ctrl.run_threaded_immediate(
            callback, result_callback, error_callback, finished_callback, *args, **kwargs
        )

    def clear_operation_queue(self):
        self.task_runner_ctrl.clear_operation_queue()

    def cancel_current_task(self):
        self.task_runner_ctrl.cancel_current_task()

    def is_current_task_cancelled(self) -> bool:
        worker = getattr(self, "current_worker", None)
        return bool(self._batch_cancel_requested or (worker and worker.is_cancelled))

    def report_runtime_progress(self, payload: dict[str, Any]):
        if not isinstance(payload, dict):
            return
        try:
            self.runtime_progress_update.emit(dict(payload))
        except Exception:
            logger.debug("Failed to queue runtime progress update.", exc_info=True)

    def _ensure_automatic_progress_dialog(self):
        return self.pipeline_status_panel

    def _show_automatic_progress_dialog(self, selected_paths: list[str], run_type: str):
        self._automatic_progress_tracker.reset(page_total=len(selected_paths), run_type=run_type)
        self._last_runtime_preview_path = ""
        self._last_batch_output_root = ""
        panel = self._ensure_automatic_progress_dialog()
        panel.set_output_root("")
        panel.set_minimized(False)
        self.set_pipeline_overlay_active(True)
        self.on_runtime_progress_update({
            "phase": "gemma_startup",
            "service": "gemma",
            "status": "starting",
            "step_key": "queue",
            "message": self.tr("Gemma와 OCR 준비를 확인하는 중..."),
            "page_total": len(selected_paths),
            "page_index": 0,
            "image_name": os.path.basename(selected_paths[0]) if selected_paths else "",
            "panel_state": "running",
            "panel_message_level": "info",
        })
        panel.show()
        panel.raise_()

    @QtCore.Slot(dict)
    def on_runtime_progress_update(self, payload: dict):
        if not isinstance(payload, dict):
            return
        event = self._automatic_progress_tracker.enrich(payload)
        preview_path = str(event.get("preview_path") or "").strip()
        if preview_path:
            self._last_runtime_preview_path = preview_path
        self._ensure_automatic_progress_dialog().update_event(event)
        self._log_runtime_progress(event)

    def _log_runtime_progress(self, event: dict):
        logger.info(
            "Runtime progress: phase=%s service=%s step=%s status=%s page=%s/%s image=%s elapsed=%s eta=%s finish_at=%s",
            event.get("phase", ""),
            event.get("service", ""),
            event.get("step_key", ""),
            event.get("status", ""),
            (int(event.get("page_index", 0)) + 1) if event.get("page_index") is not None else "-",
            event.get("page_total", "-"),
            event.get("image_name", ""),
            event.get("elapsed_text", ""),
            event.get("eta_text", ""),
            event.get("eta_finish_at_local", ""),
        )

    def _on_automatic_progress_cancel(self):
        self.cancel_current_task()
        self._ensure_automatic_progress_dialog().show_passive_message(
            "info",
            self.tr("취소 중..."),
            duration=None,
            closable=False,
            source="pipeline",
        )

    def _on_automatic_progress_retry(self):
        if self._batch_active or not self._last_batch_request_paths:
            return
        self._start_batch_process_for_paths(list(self._last_batch_request_paths), run_type=self._last_batch_run_type)

    def _on_automatic_progress_open_settings(self):
        self.show_settings_page()
        try:
            ui = self.settings_page.ui
            page_map = {
                self.tr("PaddleOCR VL Settings"): 2,
                self.tr("HunyuanOCR Settings"): 3,
                self.tr("Gemma Local Server Settings"): 4,
            }
            target_index = page_map.get(self._automatic_progress_settings_target, 4)
            if len(ui.nav_cards) > target_index:
                ui.on_nav_clicked(target_index, ui.nav_cards[target_index])
        except Exception:
            logger.debug("Failed to focus local service settings page.", exc_info=True)

    def run_finish_only(self, finished_callback: Callable, error_callback: Callable = None):
        self.task_runner_ctrl.run_finish_only(finished_callback, error_callback)

    def default_error_handler(self, error_tuple: Tuple):
        exctype, value, traceback_str = error_tuple

        if issubclass(exctype, OperationCancelledError):
            self.pipeline_status_panel.update_event({
                "phase": "done",
                "status": "cancelled",
                "service": "batch",
                "message": self.tr("작업이 취소되었습니다."),
                "panel_state": "cancelled",
            })
            self.set_pipeline_overlay_active(False)
            self.loading.setVisible(False)
            return

        if self._batch_active:
            self._batch_failed = True
            service_name = getattr(value, "service_name", "Gemma") if isinstance(value, BaseException) else "Gemma"
            self._automatic_progress_settings_target = getattr(value, "settings_page_name", self.tr("Gemma Local Server Settings"))
            service_key = "gemma" if "gemma" in service_name.lower() else ("paddleocr_vl" if "paddle" in service_name.lower() else "batch")
            self.on_runtime_progress_update({
                "phase": "error",
                "service": service_key,
                "status": "failed",
                "step_key": "error",
                "message": self.tr("자동번역 준비 또는 실행에 실패했습니다."),
                "detail": str(value),
                "page_total": len(self._last_batch_request_paths),
                "page_index": 0,
                "image_name": os.path.basename(self._last_batch_request_paths[0]) if self._last_batch_request_paths else "",
                "panel_state": "failed",
                "panel_message_level": "error",
            })
            self.loading.setVisible(False)
            return

        if issubclass(exctype, LocalServiceError):
            if issubclass(exctype, LocalServiceSetupError):
                error_kind = "setup"
            elif issubclass(exctype, LocalServiceConnectionError):
                error_kind = "connection"
            else:
                error_kind = "response"
            Messages.show_local_service_error(
                self,
                details=str(value),
                service_name=getattr(value, "service_name", "PaddleOCR VL"),
                settings_page_name=getattr(
                    value,
                    "settings_page_name",
                    self.tr("PaddleOCR VL Settings"),
                ),
                error_kind=error_kind,
            )

        # Handle HTTP Errors (Server-side)
        elif issubclass(exctype, requests.exceptions.HTTPError):
            response = value.response
            if response is not None:
                status_code = response.status_code
                
                # Content Flagged / Moderation Blocked
                if status_code == 400:
                    try:
                        detail = response.json().get('detail', {})
                        err_type = detail.get('type') if isinstance(detail, dict) else ""
                        if err_type == 'CONTENT_FLAGGED_UNSAFE':
                            Messages.show_content_flagged_error(self)
                            self.loading.setVisible(False)
                            self.enable_hbutton_group()
                            return
                    except Exception:
                        pass # Fall through if parsing fails
                        
                # Server Errors (5xx)
                if 500 <= status_code < 600:
                    # Try to determine context from error type for better messaging
                    context = None
                    try:
                        detail = response.json().get('detail', {})
                        if isinstance(detail, dict):
                            err_type = detail.get('type', '')
                            if 'OCR' in err_type:
                                context = 'ocr'
                            elif 'TRANSLAT' in err_type:
                                context = 'translation'
                    except Exception:
                        pass
                    
                    Messages.show_server_error(self, status_code, context)
                    self.loading.setVisible(False)
                    self.enable_hbutton_group()
                    return

            # If not handled above, fall through to generic error (with traceback)
            error_msg = f"An error occurred:\n{exctype.__name__}: {value}"
            error_msg_trcbk = f"An error occurred:\n{exctype.__name__}: {value}\n\nTraceback:\n{traceback_str}"
            Messages.show_error_with_copy(self, self.tr("Error"), error_msg, error_msg_trcbk)

        # Handle Network Errors (Connection, Timeout, etc.)
        elif issubclass(exctype, requests.exceptions.RequestException):
            Messages.show_network_error(self)

        else:
            error_msg = f"An error occurred:\n{exctype.__name__}: {value}"
            error_msg_trcbk = f"An error occurred:\n{exctype.__name__}: {value}\n\nTraceback:\n{traceback_str}"
            print(error_msg_trcbk)
            Messages.show_error_with_copy(self, self.tr("Error"), error_msg, error_msg_trcbk)

        self.loading.setVisible(False)
        self.enable_hbutton_group()

    def _start_batch_report(self, batch_paths: list[str], run_type: str = "batch"):
        self.batch_report_ctrl.start_batch_report(batch_paths, run_type=run_type)

    def _finalize_batch_report(self, was_cancelled: bool):
        return self.batch_report_ctrl.finalize_batch_report(was_cancelled)

    def show_latest_batch_report(self):
        self.batch_report_ctrl.show_latest_batch_report()

    def register_batch_skip(self, image_path: str, skip_reason: str, error: str):
        self.batch_report_ctrl.register_batch_skip(image_path, skip_reason, error)

    def _sync_txt_md_project_state(self) -> None:
        if self.webtoon_mode:
            manager = getattr(self.image_viewer, "webtoon_manager", None)
            scene_mgr = getattr(manager, "scene_item_manager", None) if manager is not None else None
            if scene_mgr is not None:
                scene_mgr.save_all_scene_items_to_states()
        else:
            self.image_ctrl.save_current_image_state()

    def _ensure_txt_md_ready(self) -> bool:
        if not self.image_files:
            Messages.show_info(
                self,
                self.tr("No pages are loaded for TXT/MD import or export."),
                duration=5,
                closable=True,
                source="txt_md",
            )
            return False

        duplicates = find_duplicate_page_names(self.image_files)
        if duplicates:
            Messages.show_warning(
                self,
                self.tr(
                    "TXT/MD import and export require unique page file names.\nRename duplicate pages first.\nDuplicates:\n{names}"
                ).format(names="\n".join(duplicates)),
                duration=None,
                closable=True,
                source="txt_md",
            )
            return False
        return True

    def _txt_md_default_dir(self) -> str:
        return self.project_ctrl._get_default_export_dir()

    def _txt_md_bundle_name(self) -> str:
        return self.project_ctrl._get_export_bundle_name()

    def _txt_md_save_path(self, target: str, suffix: str) -> str:
        return os.path.join(
            self._txt_md_default_dir(),
            f"{self._txt_md_bundle_name()}_{target}{suffix}",
        )

    def _write_txt_md_exchange(
        self,
        target: str,
        suffix: str,
        page_paths: list[str],
    ) -> str:
        page_entries = collect_page_entries(page_paths, self.image_states, target)
        save_path = self._txt_md_save_path(target, suffix)
        return dump_exchange_text(save_path, page_entries)

    def export_source_exchange(self, suffix: str) -> None:
        if not self._ensure_txt_md_ready():
            return

        self._sync_txt_md_project_state()
        try:
            save_path = self._write_txt_md_exchange("source", suffix, list(self.image_files))
        except Exception as exc:
            error_text = traceback.format_exc()
            Messages.show_error_with_copy(
                self,
                self.tr("TXT/MD Export Failed"),
                self.tr("Failed to export source text."),
                error_text,
            )
            logger.exception("Failed to export source exchange file: %s", exc)
            return

        Messages.show_success(
            self,
            self.tr("Exported source text to:\n{path}").format(path=save_path),
            duration=None,
            closable=True,
            source="txt_md",
        )

    def _refresh_after_translation_import(self, matched_paths: list[str]) -> None:
        if not matched_paths:
            return

        current_file = None
        if 0 <= self.curr_img_idx < len(self.image_files):
            current_file = self.image_files[self.curr_img_idx]

        for file_path in matched_paths:
            stack = self.undo_stacks.get(file_path)
            if stack is not None:
                stack.clear()
                stack.setClean()

        if current_file not in matched_paths:
            return

        if self.webtoon_mode:
            manager = getattr(self.image_viewer, "webtoon_manager", None)
            scene_mgr = getattr(manager, "scene_item_manager", None) if manager is not None else None
            if (
                scene_mgr is not None
                and manager is not None
                and 0 <= self.curr_img_idx < len(self.image_files)
                and self.curr_img_idx in manager.loaded_pages
            ):
                scene_mgr.unload_page_scene_items(self.curr_img_idx)
                scene_mgr.load_page_scene_items(self.curr_img_idx)
                self.text_ctrl.clear_text_edits()
        else:
            self.image_ctrl.on_render_state_ready(current_file)

    def _build_translation_import_message(self, all_matched: bool, match_result: dict[str, list[str]]) -> str:
        if all_matched:
            return self.tr("Translation imported and matched successfully.")

        lines = [
            self.tr(
                "Imported TXT/MD content was only partially matched. Make sure the file follows the exported exchange format."
            )
        ]
        if match_result["missing_pages"]:
            lines.append("")
            lines.append(self.tr("Missing pages:"))
            lines.extend(match_result["missing_pages"])
        if match_result["unexpected_pages"]:
            lines.append("")
            lines.append(self.tr("Unexpected pages:"))
            lines.extend(match_result["unexpected_pages"])
        if match_result["unmatched_pages"]:
            lines.append("")
            lines.append(self.tr("Unmatched pages:"))
            lines.extend(match_result["unmatched_pages"])
        return "\n".join(lines).strip()

    def import_translation_exchange(self, suffix: str) -> None:
        if not self._ensure_txt_md_ready():
            return

        self._sync_txt_md_project_state()
        file_filter = (
            self.tr("TXT Files (*.txt *.TXT)")
            if suffix == ".txt"
            else self.tr("Markdown Files (*.md *.MD)")
        )
        selected_file, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            self.tr("Import Translation"),
            self._txt_md_default_dir(),
            file_filter,
        )
        if not selected_file:
            return

        try:
            parsed_pages = parse_translation_exchange_file(selected_file)
            page_name_to_blocks: dict[str, list[TextBlock]] = {}
            page_name_to_path: dict[str, str] = {}
            for file_path in self.image_files:
                state = self.image_ctrl.ensure_page_state(file_path)
                page_name = page_name_from_path(file_path)
                page_name_to_blocks[page_name] = state.get("blk_list", []) or []
                page_name_to_path[page_name] = file_path

            all_matched, match_result = apply_translation_pages(
                parsed_pages,
                page_name_to_blocks,
                translation_rules=self.settings_page.get_translation_result_dictionary_rules(),
            )
            matched_paths = [
                page_name_to_path[name]
                for name in match_result["matched_pages"]
                if name in page_name_to_path
            ]
            if matched_paths:
                self.text_ctrl.rebuild_text_items_state_for_paths(matched_paths)
                self._refresh_after_translation_import(matched_paths)
                self.mark_project_dirty()

            message = self._build_translation_import_message(all_matched, match_result)
            icon = (
                QtWidgets.QMessageBox.Icon.Information
                if all_matched
                else QtWidgets.QMessageBox.Icon.Warning
            )
            msg_box = QtWidgets.QMessageBox(self)
            msg_box.setIcon(icon)
            msg_box.setWindowTitle(self.tr("Import Translation"))
            msg_box.setText(message)
            msg_box.exec()
        except Exception:
            Messages.show_error_with_copy(
                self,
                self.tr("TXT/MD Import Failed"),
                self.tr("Failed to import translation text."),
                traceback.format_exc(),
            )

    def _run_txt_md_auto_exports(self, page_paths: list[str]) -> None:
        if not page_paths:
            return
        if not self._ensure_txt_md_ready():
            return

        export_settings = self.settings_page.get_export_settings()
        targets = []
        if export_settings.get("auto_export_source_txt"):
            targets.append(("source", ".txt"))
        if export_settings.get("auto_export_source_md"):
            targets.append(("source", ".md"))
        if export_settings.get("auto_export_translation_txt"):
            targets.append(("translation", ".txt"))
        if export_settings.get("auto_export_translation_md"):
            targets.append(("translation", ".md"))
        for target, suffix in targets:
            try:
                self._write_txt_md_exchange(target, suffix, page_paths)
            except Exception:
                logger.exception("Automatic TXT/MD export failed for %s%s", target, suffix)
                Messages.show_warning(
                    self,
                    self.tr("Automatic TXT/MD export failed for {target}{suffix}.").format(
                        target=target,
                        suffix=suffix,
                    ),
                    duration=None,
                    closable=True,
                    source="txt_md",
                )
                return

    def _start_batch_process_for_paths(self, selected_paths: list[str], run_type: str = "batch") -> bool:
        if not selected_paths:
            return False

        selected_paths = [path for path in selected_paths if path in self.image_files]
        if not selected_paths:
            return False

        self._last_batch_request_paths = list(selected_paths)
        self._last_batch_run_type = run_type
        self._batch_failed = False
        ocr_preflight_cache: dict[str, str] = {}

        for path in selected_paths:
            page_state = self.image_ctrl.ensure_page_state(path)
            tgt = page_state['target_lang']
            src = page_state['source_lang']
            if not validate_settings(self, tgt, source_lang=src, preflight_cache=ocr_preflight_cache):
                return False

        self.image_ctrl.clear_page_skip_errors_for_paths(selected_paths)
        self._start_batch_report(selected_paths, run_type=run_type)
        self.selected_batch = selected_paths
        self._current_batch_run_type = run_type

        if self.manual_radio.isChecked():
            self.automatic_radio.setChecked(True)
            self.batch_mode_selected()
        self._batch_active = True
        self._batch_cancel_requested = False
        self.translate_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.save_as_project_button.setEnabled(False)
        self.webtoon_toggle.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.batch_report_ctrl.refresh_action_buttons()
        self._show_automatic_progress_dialog(selected_paths, run_type)

        if self.webtoon_mode:
            self.run_threaded(
                lambda: self.pipeline.webtoon_batch_process(selected_paths),
                None,
                self.default_error_handler,
                self.on_batch_process_finished
            )
        else:
            self.run_threaded(
                lambda: self.pipeline.batch_process(selected_paths),
                None,
                self.default_error_handler,
                self.on_batch_process_finished
            )
        return True

    def _confirm_and_apply_auto_languages(self, selected_paths: list[str], run_type: str) -> bool:
        selected_paths = [path for path in selected_paths if path in self.image_files]
        if not selected_paths:
            return False

        source_lang = self.s_combo.currentText()
        target_lang = self.t_combo.currentText()
        ocr_mode = self.settings_page.get_tool_selection("ocr")
        source_lang_english = self.lang_mapping.get(source_lang, source_lang)
        resolved_ocr = resolve_ocr_engine(ocr_mode, source_lang_english)
        resolved_ocr_label = (
            resolved_ocr if normalize_ocr_mode(ocr_mode) == OCR_MODE_BEST_LOCAL else None
        )
        run_label = (
            self.tr("One-Page Auto")
            if run_type == "one_page_auto"
            else self.tr("Translate All")
        )
        confirmed = Messages.confirm_automatic_run(
            self,
            run_label=run_label,
            page_count=len(selected_paths),
            source_lang=source_lang,
            target_lang=target_lang,
            ocr_mode_label=self.settings_page.get_ocr_mode_label(ocr_mode),
            resolved_ocr_label=resolved_ocr_label,
        )
        if not confirmed:
            return False

        self.image_ctrl.apply_languages_to_paths(selected_paths, source_lang, target_lang)
        return True

    def start_batch_process(self):
        try:
            if self._memlogger is not None:
                self._memlogger.emit("batch_start_all")
        except Exception:
            pass
        if not self._confirm_and_apply_auto_languages(self.image_files, "batch"):
            return
        self._start_batch_process_for_paths(self.image_files, run_type="batch")

    def batch_translate_selected(self, selected_paths: list[str]):
        try:
            if self._memlogger is not None:
                self._memlogger.emit("batch_start_selected")
        except Exception:
            pass
        selected_paths = [p for p in selected_paths if p in self.image_files]
        self._start_batch_process_for_paths(selected_paths, run_type="batch")

    def retry_failed_batch_pages(self):
        if self._batch_active:
            return

        retry_paths = self.batch_report_ctrl.get_latest_retry_paths()
        if not retry_paths:
            Messages.show_info(
                self,
                self.tr("No failed pages from the latest batch are available to retry."),
                duration=5,
                closable=True,
                source="batch",
            )
            return

        self._start_batch_process_for_paths(retry_paths, run_type="retry_failed")

    def start_one_page_auto_process(self):
        if self._batch_active:
            return
        if not (0 <= self.curr_img_idx < len(self.image_files)):
            Messages.show_info(
                self,
                self.tr("No current page is available for automatic processing."),
                duration=5,
                closable=True,
                source="batch",
            )
            return
        current_path = self.image_files[self.curr_img_idx]
        if not self._confirm_and_apply_auto_languages([current_path], "one_page_auto"):
            return
        self._start_batch_process_for_paths([current_path], run_type="one_page_auto")

    def on_batch_process_finished(self):
        try:
            if self._memlogger is not None:
                self._memlogger.emit("batch_finished")
        except Exception:
            pass
        was_cancelled = self._batch_cancel_requested
        failed = self._batch_failed
        total_images = len(self.selected_batch)
        completed_batch_paths = list(self.selected_batch)
        self._batch_active = False
        self._batch_cancel_requested = False
        self._batch_failed = False
        self._current_batch_run_type = None
        report = self._finalize_batch_report(was_cancelled)
        self._last_batch_output_root = self._find_latest_batch_output_root(completed_batch_paths)
        self.progress_bar.setVisible(False)
        self.translate_button.setEnabled(True)
        self.cancel_button.setEnabled(True)
        self.save_as_project_button.setEnabled(True)
        self.webtoon_toggle.setEnabled(True)
        self.selected_batch = []
        self.set_pipeline_overlay_active(False)

        panel = self.pipeline_status_panel
        panel.set_output_root(self._last_batch_output_root)
        if was_cancelled:
            panel.update_event({
                "phase": "done",
                "service": "batch",
                "status": "cancelled",
                "step_key": "done",
                "message": self.tr("작업이 취소되었습니다."),
                "panel_state": "cancelled",
            })
        elif failed:
            panel.show()
            panel.raise_()
        else:
            self._automatic_progress_tracker.record_batch_completion(success=True, total_images=total_images)
            self.on_runtime_progress_update({
                "phase": "done",
                "service": "batch",
                "status": "completed",
                "step_key": "done",
                "message": self.tr("자동번역이 완료되었습니다."),
                "page_total": total_images,
                "page_index": max(total_images - 1, 0),
                "image_name": "",
                "preview_path": self._last_runtime_preview_path,
                "panel_state": "done",
                "panel_message_level": "success",
            })
            self._play_completion_sound_if_enabled()
            # Reserved hook for future ntfy integration. Keep best-effort and non-blocking.
            notify_pipeline_event(
                {
                    "event_type": "pipeline_completed",
                    "run_type": self._last_batch_run_type,
                    "success": True,
                    "image_count": total_images,
                    "output_root": self._last_batch_output_root,
                    "message": self.tr("자동번역이 완료되었습니다."),
                }
            )

        if report and report["skipped_count"] > 0:
            Messages.show_batch_skipped_summary(self, report["skipped_count"])
        elif not was_cancelled and not failed:
            self._run_txt_md_auto_exports(completed_batch_paths)
            Messages.show_translation_complete(self)

        # Drop cached models/sessions after batch to keep RAM bounded.
        try:
            if self.pipeline is not None:
                self.pipeline.release_model_caches()
            if self._memlogger is not None:
                self._memlogger.emit("model_caches_released")
        except Exception:
            pass
        self.batch_report_ctrl.refresh_action_buttons()

    def _find_latest_batch_output_root(self, page_paths: list[str]) -> str:
        for file_path in reversed(list(page_paths or [])):
            state = self.image_states.get(file_path, {})
            summary = state.get("processing_summary", {}) if isinstance(state, dict) else {}
            export_root = str(summary.get("export_root", "")).strip()
            if export_root:
                return export_root
        return ""

    def _open_latest_batch_output(self) -> None:
        output_root = self._last_batch_output_root or self._find_latest_batch_output_root(self._last_batch_request_paths)
        if not output_root:
            return
        QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(output_root))

    def _play_completion_sound_if_enabled(self) -> None:
        settings = self.settings_page.get_notification_settings()
        if not bool(settings.get("enable_completion_sound", True)):
            return
        play_completion_sound(
            str(settings.get("completion_sound_mode") or SYSTEM_SOUND_MODE),
            str(settings.get("completion_sound_file") or ""),
        )

    def disable_hbutton_group(self):
        for button in self.hbutton_group.get_button_group().buttons():
            button.setEnabled(False)

    def enable_hbutton_group(self):
        for button in self.hbutton_group.get_button_group().buttons():
            button.setEnabled(True)

    def block_detect(self, load_rects: bool = True):
        self.manual_workflow_ctrl.block_detect(load_rects)

    def finish_ocr_translate(self, single_block=False):
        self.manual_workflow_ctrl.finish_ocr_translate(single_block)

    def ocr(self, single_block=False):
        self.manual_workflow_ctrl.ocr(single_block)

    def translate_image(self, single_block=False):
        self.manual_workflow_ctrl.translate_image(single_block)

    def _get_visible_text_items(self):
        return self.manual_workflow_ctrl._get_visible_text_items()

    def update_translated_text_items(self, single_blk: bool):
        self.manual_workflow_ctrl.update_translated_text_items(single_blk)

    def inpaint_and_set(self):
        self.manual_workflow_ctrl.inpaint_and_set()

    def blk_detect_segment(self, result): 
        self.manual_workflow_ctrl.blk_detect_segment(result)

    def load_segmentation_points(self):
        self.manual_workflow_ctrl.load_segmentation_points()
                
    def _on_segmentation_bboxes_ready(self, results):
        self.manual_workflow_ctrl._on_segmentation_bboxes_ready(results)

    def update_progress(self, index: int, total_images: int, step: int, total_steps: int, change_name: bool):
        if self._batch_cancel_requested:
            return

        # Assign weights to image processing and archiving (adjust as needed)
        image_processing_weight = 0.9
        archiving_weight = 0.1

        archive_info_list = self.file_handler.archive_info
        total_archives = len(archive_info_list)
        image_list = self.selected_batch if self.selected_batch else self.image_files

        if change_name:
            if index < total_images:
                im_path = image_list[index]
                im_name = os.path.basename(im_path)
                self.progress_bar.setFormat(QCoreApplication.translate('Messages', 'Processing:') + f" {im_name} . . . %p%")
            else:
                archive_index = index - total_images
                self.progress_bar.setFormat(QCoreApplication.translate('Messages', 'Archiving:') + f" {archive_index + 1}/{total_archives} . . . %p%")

        if index < total_images:
            # Image processing progress
            task_progress = (index / total_images) * image_processing_weight
            step_progress = (step / total_steps) * (1 / total_images) * image_processing_weight
        else:
            # Archiving progress
            archive_index = index - total_images
            task_progress = image_processing_weight + (archive_index / total_archives) * archiving_weight
            step_progress = (step / total_steps) * (1 / total_archives) * archiving_weight

        progress = (task_progress + step_progress) * 100 
        self.progress_bar.setValue(int(progress))

    def on_download_event(self, status: str, name: str):
        """Show a loading-type MMessage while models/files are being downloaded."""
        # Keep a counter of active downloads to handle multiple files
        if not hasattr(self, "_active_downloads"):
            self._active_downloads = 0

        if status == 'start':
            self._active_downloads += 1
            filename = os.path.basename(name)
            self.set_download_status(self.tr(f"Downloading model file: {filename}"))
        elif status == 'end':
            self._active_downloads = max(0, self._active_downloads - 1)
            if self._active_downloads == 0:
                self.set_download_status(None)
                # Do not change the main window loading spinner here; it's managed by the running task lifecycle

    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_Left:
            self.image_ctrl.navigate_images(-1)
        elif event.key() == QtCore.Qt.Key_Right:
            self.image_ctrl.navigate_images(1)
        else:
            super().keyPressEvent(event)

    def closeEvent(self, event):
        try:
            self.text_ctrl._commit_pending_text_command()
        except Exception:
            pass
        if not getattr(self, "_skip_close_prompt", False):
            if self.has_unsaved_changes():
                msg_box = QtWidgets.QMessageBox(self)
                msg_box.setIcon(QtWidgets.QMessageBox.Question)
                msg_box.setWindowTitle(self.tr("Unsaved Changes"))
                msg_box.setText(self.tr("Save changes to this file?"))
                save_btn = msg_box.addButton(self.tr("Save"), QtWidgets.QMessageBox.ButtonRole.AcceptRole)
                msg_box.addButton(self.tr("Don't Save"), QtWidgets.QMessageBox.ButtonRole.DestructiveRole)
                cancel_btn = msg_box.addButton(self.tr("Cancel"), QtWidgets.QMessageBox.ButtonRole.RejectRole)
                msg_box.setDefaultButton(save_btn)
                msg_box.exec()
                clicked = msg_box.clickedButton()

                if clicked == save_btn:
                    self.project_ctrl.thread_save_project(
                        post_save_callback=self._finish_close_after_save
                    )
                    event.ignore()
                    return
                if clicked == cancel_btn or clicked is None:
                    event.ignore()
                    return
        else:
            self._skip_close_prompt = False

        self.project_ctrl.shutdown_autosave(clear_recovery=True)
        self.shutdown()

        # Save all settings when the application is closed
        self.settings_page.save_settings()
        self.project_ctrl.save_main_page_settings()
        self.image_ctrl.cleanup()
        
        # Delete temp archive folders
        for archive in self.file_handler.archive_info:
            temp_dir = archive['temp_dir']
            if os.path.exists(temp_dir): 
                shutil.rmtree(temp_dir)  

        for root, dirs, files in os.walk(self.temp_dir, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))
        os.rmdir(self.temp_dir)

        super().closeEvent(event)

    def shutdown(self):
        if getattr(self, "_is_shutting_down", False):
            return
        self._is_shutting_down = True

        self.batch_report_ctrl.shutdown()

        try:
            self.cancel_current_task()
        except Exception:
            pass

        try:
            self.threadpool.clear()
            self.threadpool.waitForDone(2000)
        except Exception:
            pass

        try:
            self.settings_page.shutdown()
        except Exception:
            pass
        try:
            self.pipeline_status_panel.hide()
            self.set_pipeline_overlay_active(False)
        except Exception:
            pass
        try:
            self.local_ocr_runtime_manager.shutdown()
        except Exception:
            pass

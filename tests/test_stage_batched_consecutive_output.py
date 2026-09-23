from __future__ import annotations

import threading
import unittest
from contextlib import nullcontext
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import mock

import numpy as np
from PIL import Image
from PySide6 import QtCore

from app.controllers.series import SeriesController
from app.projects.series_state_v1 import (
    create_series_project,
    load_series_project,
    load_series_project_blob,
)
from modules.utils.exceptions import OperationCancelledError
from pipeline.stage_batched_processor import StageBatchedProcessor, StagePageContext


class ConsecutiveStageBatchedOutputTests(unittest.TestCase):
    def _processor(self, output_root: Path) -> StageBatchedProcessor:
        processor = object.__new__(StageBatchedProcessor)
        cancelled = {"value": False}
        processor.main_page = SimpleNamespace(
            image_files=[],
            file_handler=SimpleNamespace(should_pre_materialize=lambda _paths: False),
            settings_page=object(),
            project_ctrl=None,
            is_current_task_cancelled=lambda: cancelled["value"],
        )
        processor._test_cancelled = cancelled
        processor._render_cancel_event = threading.Event()
        processor._render_executor = None
        processor._pending_render_jobs = []
        processor._recent_page_durations = []
        processor._benchmark_stage_ceiling = lambda: None
        processor._measure_performance = lambda **_kwargs: nullcontext()
        processor._load_page_contexts = lambda paths: [
            StagePageContext(
                image_path=path,
                image_name=Path(path).name,
                source_lang="Japanese",
                target_lang="Korean",
                directory=str(output_root),
                image=np.zeros((8, 8, 3), dtype=np.uint8),
                blk_list=[],
                mask=np.zeros((8, 8), dtype=np.uint8),
                no_text_detected=True,
            )
            for path in paths
        ]
        processor._ensure_stage_policy = lambda _pages: {"primary_ocr_engine": "Default"}
        processor._ocr_runtime_service_name = lambda _key: "ocr"
        processor._effective_export_settings = lambda _settings: {}
        for method in (
            "_emit_benchmark_event",
            "_record_performance_workload",
            "_reset_prewarm_lifecycle",
            "_shutdown_managed_runtimes",
            "_start_ocr_container_prewarm",
            "_detect_all",
            "_await_ocr_container_prewarm",
            "_start_ocr_prewarm",
            "_start_gemma_page_cache_prefetch",
            "_ocr_all",
            "_translate_all",
            "_release_gemma_before_inpainter",
            "_inpaint_all",
            "_sample_performance_resources",
            "_record_stage_memory_telemetry",
            "_persist_stage_rates",
            "_shutdown_page_cache_executor",
            "_shutdown_prewarm_executor",
            "_write_run_report",
        ):
            setattr(processor, method, mock.Mock())
        return processor

    def test_same_processor_writes_distinct_pixels_on_two_runs(self) -> None:
        with TemporaryDirectory() as temporary:
            output_root = Path(temporary)
            processor = self._processor(output_root)
            values = iter((31, 219))

            def render(pages: list[StagePageContext]) -> None:
                if processor._render_cancel_event.is_set():
                    raise OperationCancelledError("A retired render token was reused")
                value = next(values)
                for page in pages:
                    target = output_root / f"{Path(page.image_path).stem}-translated.png"
                    Image.new("RGB", (8, 8), (value, 0, 0)).save(target)
                    page.output_path = str(target)

            processor._render_all = render
            with mock.patch(
                "pipeline.stage_batched_processor.open_project_stage_checkpoint_store",
                return_value=None,
            ):
                processor.batch_process(["chapter-a.png"])
                retired_event = processor._render_cancel_event
                processor.batch_process(["chapter-b.png"])

            self.assertTrue(retired_event.is_set())
            self.assertIsNot(retired_event, processor._render_cancel_event)
            for name, value in (("chapter-a", 31), ("chapter-b", 219)):
                target = output_root / f"{name}-translated.png"
                self.assertTrue(target.is_file(), name)
                with Image.open(target) as image:
                    self.assertEqual(image.getpixel((0, 0)), (value, 0, 0))
            self.assertEqual(processor._write_run_report.call_count, 2)

    def test_missing_output_is_reported_as_failure_not_success(self) -> None:
        with TemporaryDirectory() as temporary:
            processor = self._processor(Path(temporary))
            processor._render_all = lambda _pages: None
            processor._write_fallback_export = mock.Mock(return_value=False)
            processor._release_page_buffers = mock.Mock()
            with mock.patch(
                "pipeline.stage_batched_processor.open_project_stage_checkpoint_store",
                return_value=None,
            ), self.assertRaisesRegex(RuntimeError, "0/1"):
                processor.batch_process(["chapter-a.png"])

            self.assertEqual(
                processor._write_run_report.call_args.kwargs["output_summary"]["missing"],
                ["chapter-a.png"],
            )
            tags = [call.args[0] for call in processor._emit_benchmark_event.call_args_list]
            self.assertIn("batch_run_failed", tags)
            self.assertNotIn("batch_run_done", tags)

    def test_series_queue_two_children_save_pixels_and_done_states(self) -> None:
        class SeriesMain(QtCore.QObject):
            def __init__(self) -> None:
                super().__init__()
                self.pipeline_status_panel = SimpleNamespace(
                    set_series_queue_pause_visible=mock.Mock()
                )
                self.project_ctrl = SimpleNamespace(save_current_state=mock.Mock())

        with TemporaryDirectory() as temporary:
            output_root = Path(temporary)
            series_file = output_root / "example.seriesctpr"
            items = [
                {
                    "series_item_id": f"item-{index}",
                    "queue_index": index,
                    "display_name": f"chapter-{index}.ctpr",
                    "source_kind": "ctpr_import",
                    "source_origin_path": str(output_root / f"chapter-{index}.ctpr"),
                    "source_origin_relpath": f"chapter-{index}.ctpr",
                    "imported_at": "2026-01-01T00:00:00",
                    "updated_at": "2026-01-01T00:00:00",
                    "status": "pending",
                    "embedded_project_blob_hash": f"hash-{index}",
                    "child_page_count": 1,
                }
                for index in (1, 2)
            ]
            create_series_project(
                str(series_file),
                root_dir=str(output_root),
                items=items,
                embedded_projects=[
                    {
                        "project_hash": f"hash-{index}",
                        "display_name": f"chapter-{index}.ctpr",
                        "project_size": 7,
                        "project_blob": b"project",
                    }
                    for index in (1, 2)
                ],
            )
            controller = SeriesController(SeriesMain())
            state = load_series_project(str(series_file))
            controller.series_file = str(series_file)
            controller.series_manifest = dict(state["manifest"])
            controller.series_items = list(state["items"])
            controller._queue_active = True
            controller._queue_pending_ids = ["item-1", "item-2"]
            controller._queue_completed_ids = []
            controller._queue_failed_ids = []
            controller._queue_skipped_ids = []
            controller._queue_retry_remaining = {}
            controller._apply_workspace_state = mock.Mock()
            controller._show_board = mock.Mock()
            controller._set_series_window_title = mock.Mock()
            processor = self._processor(output_root)

            def render(pages: list[StagePageContext]) -> None:
                if processor._render_cancel_event.is_set():
                    raise OperationCancelledError("A retired render token was reused")
                for page in pages:
                    target = output_root / f"{Path(page.image_path).stem}-translated.png"
                    value = 35 if "chapter-1" in page.image_path else 225
                    Image.new("RGB", (8, 8), (value, 0, 0)).save(target)
                    page.output_path = str(target)

            processor._render_all = render

            def open_child(item_id: str, *, push_history: bool, after_loaded) -> None:
                self.assertFalse(push_history)
                controller.active_child_item_id = item_id
                controller.active_child_project_path = str(output_root / f"{item_id}.ctpr")
                after_loaded()

            def process_child() -> None:
                item_id = str(controller.active_child_item_id)
                processor.batch_process([f"chapter-{item_id[-1]}.png"])
                controller.on_batch_process_finished(was_cancelled=False, failed=False)

            controller._open_item = open_child
            controller._start_batch_for_active_child = process_child

            def save_child(_main, child_path: str, **_kwargs) -> None:
                index = int(Path(child_path).stem[-1])
                output = output_root / f"chapter-{index}-translated.png"
                Path(child_path).write_bytes(output.read_bytes())

            with (
                mock.patch(
                    "pipeline.stage_batched_processor.open_project_stage_checkpoint_store",
                    return_value=None,
                ),
                mock.patch(
                    "app.controllers.series.QtCore.QTimer.singleShot",
                    side_effect=lambda _delay, _receiver, callback: callback(),
                ),
                mock.patch("app.controllers.series.save_state_to_proj_file", save_child),
            ):
                controller._run_next_queue_item()

            reloaded = load_series_project(str(series_file))
            self.assertEqual([item["status"] for item in reloaded["items"]], ["done", "done"])
            self.assertEqual(reloaded["manifest"]["series_queue_runtime"]["queue_state"], "idle")
            self.assertEqual(processor._write_run_report.call_count, 2)
            for index, value in ((1, 35), (2, 225)):
                with Image.open(output_root / f"chapter-{index}-translated.png") as image:
                    self.assertEqual(image.getpixel((0, 0)), (value, 0, 0))
                item = reloaded["items"][index - 1]
                blob = load_series_project_blob(
                    str(series_file), item["embedded_project_blob_hash"]
                )
                with Image.open(BytesIO(blob)) as embedded:
                    self.assertEqual(embedded.getpixel((0, 0)), (value, 0, 0))

    def test_cancelled_run_does_not_poison_the_next_run(self) -> None:
        with TemporaryDirectory() as temporary:
            output_root = Path(temporary)
            processor = self._processor(output_root)
            processor._test_cancelled["value"] = True

            def render(pages: list[StagePageContext]) -> None:
                self.assertFalse(processor._render_cancel_event.is_set())
                target = output_root / "retry-translated.png"
                Image.new("RGB", (8, 8), (83, 0, 0)).save(target)
                pages[0].output_path = str(target)

            processor._render_all = render
            with mock.patch(
                "pipeline.stage_batched_processor.open_project_stage_checkpoint_store",
                return_value=None,
            ):
                processor.batch_process(["cancelled.png"])
                processor._test_cancelled["value"] = False
                processor.batch_process(["retry.png"])

            with Image.open(output_root / "retry-translated.png") as image:
                self.assertEqual(image.getpixel((0, 0)), (83, 0, 0))
            self.assertEqual(processor._write_run_report.call_count, 1)

    def test_failed_run_can_retry_with_a_new_render_token(self) -> None:
        with TemporaryDirectory() as temporary:
            output_root = Path(temporary)
            processor = self._processor(output_root)
            attempts = 0

            def render(pages: list[StagePageContext]) -> None:
                nonlocal attempts
                attempts += 1
                self.assertFalse(processor._render_cancel_event.is_set())
                if attempts == 1:
                    raise RuntimeError("synthetic render failure")
                target = output_root / "retry-translated.png"
                Image.new("RGB", (8, 8), (97, 0, 0)).save(target)
                pages[0].output_path = str(target)

            processor._render_all = render
            with mock.patch(
                "pipeline.stage_batched_processor.open_project_stage_checkpoint_store",
                return_value=None,
            ):
                with self.assertRaisesRegex(RuntimeError, "synthetic render failure"):
                    processor.batch_process(["first.png"])
                processor.batch_process(["retry.png"])

            self.assertEqual(attempts, 2)
            with Image.open(output_root / "retry-translated.png") as image:
                self.assertEqual(image.getpixel((0, 0)), (97, 0, 0))

    def test_unrequested_render_cancellation_is_a_failure(self) -> None:
        with TemporaryDirectory() as temporary:
            processor = self._processor(Path(temporary))
            processor._render_all = mock.Mock(
                side_effect=OperationCancelledError("orphaned render token")
            )
            with mock.patch(
                "pipeline.stage_batched_processor.open_project_stage_checkpoint_store",
                return_value=None,
            ), self.assertRaisesRegex(RuntimeError, "orphaned render token"):
                processor.batch_process(["chapter-a.png"])

            tags = [call.args[0] for call in processor._emit_benchmark_event.call_args_list]
            self.assertIn("batch_run_failed", tags)
            self.assertNotIn("batch_run_cancelled", tags)

    def test_checkpoint_only_run_followed_by_render_miss_uses_fresh_token(self) -> None:
        with TemporaryDirectory() as temporary:
            output_root = Path(temporary)
            processor = self._processor(output_root)
            cached = output_root / "cached-translated.png"
            Image.new("RGB", (8, 8), (17, 0, 0)).save(cached)

            def render(pages: list[StagePageContext]) -> None:
                page = pages[0]
                if page.image_path == "cached.png":
                    # No render job was submitted: the checkpoint materialized
                    # a pre-existing, valid output on the first run.
                    page.output_path = str(cached)
                    return
                if processor._render_cancel_event.is_set():
                    raise OperationCancelledError("retired token after checkpoint hit")
                target = output_root / "fresh-translated.png"
                Image.new("RGB", (8, 8), (117, 0, 0)).save(target)
                page.output_path = str(target)

            processor._render_all = render
            with mock.patch(
                "pipeline.stage_batched_processor.open_project_stage_checkpoint_store",
                return_value=None,
            ):
                processor.batch_process(["cached.png"])
                processor.batch_process(["fresh.png"])

            with Image.open(output_root / "fresh-translated.png") as image:
                self.assertEqual(image.getpixel((0, 0)), (117, 0, 0))
            self.assertEqual(processor._write_run_report.call_count, 2)

    def test_checkpoint_hit_then_real_qt_pool_job_keeps_new_token(self) -> None:
        with TemporaryDirectory() as temporary:
            output_root = Path(temporary)
            processor = self._processor(output_root)
            cached = output_root / "cached-translated.png"
            Image.new("RGB", (8, 8), (27, 0, 0)).save(cached)
            processor.main_page.curr_img_idx = -1
            processor.main_page.button_to_alignment = {1: QtCore.Qt.AlignmentFlag.AlignCenter}
            processor.main_page.button_to_vertical_alignment = {1: "center"}
            processor._lazy_render_context = lambda _ctx: (SimpleNamespace(alignment_id=1), "ko")
            processor._set_current_image = mock.Mock()
            processor._write_json_exports = mock.Mock()
            processor._ensure_page_state = lambda _path: {"viewer_state": {}}
            processor._restore_render_project_state = mock.Mock()
            processor._finish_render_checkpoint_hit = lambda ctx, **_kwargs: setattr(
                ctx, "output_path", str(cached)
            )
            processor._reserve_render_output_path = lambda ctx, **_kwargs: (
                str(output_root / f"{Path(ctx.image_path).stem}-translated.png"),
                str(output_root),
                "png",
            )
            processor._render_worker_count = lambda: 1
            processor._record_performance_detail = mock.Mock()

            def finish_render(pending, *, result, exc) -> None:
                if exc is not None:
                    raise exc
                pending.ctx.output_path = result.final_output_path

            def prepare_checkpoint(ctx, **_kwargs):
                if ctx.image_path == "cached.png":
                    return SimpleNamespace(
                        output_root=str(output_root), output_exists=True
                    ), str(output_root)
                return None, str(output_root)

            def inpaint_and_submit(pages: list[StagePageContext]) -> None:
                for index, page in enumerate(pages):
                    processor._submit_or_inline_render(
                        page, index=index, total_images=len(pages), export_settings={}
                    )

            processor._finish_render_page_bookkeeping = finish_render
            processor._prepare_render_checkpoint = prepare_checkpoint
            processor._inpaint_all = inpaint_and_submit
            processor._render_all = lambda pages: processor._drain_render_futures(block=True)
            seen_tokens: list[bool] = []

            def render_job(job):
                seen_tokens.append(bool(job.is_cancelled()))
                if job.is_cancelled():
                    raise OperationCancelledError("retired token reached Qt pool")
                Image.new("RGB", (8, 8), (127, 0, 0)).save(job.output_path)
                return SimpleNamespace(final_output_path=job.output_path)

            with (
                mock.patch(
                    "pipeline.stage_batched_processor.open_project_stage_checkpoint_store",
                    return_value=None,
                ),
                mock.patch(
                    "pipeline.stage_batched_processor.materialize_render_checkpoint_output",
                    return_value=str(cached),
                ),
                mock.patch("pipeline.stage_batched_processor.run_render_job", render_job),
            ):
                processor.batch_process(["cached.png"])
                retired = processor._render_cancel_event
                processor.batch_process(["fresh.png"])

            self.assertTrue(retired.is_set())
            self.assertIsNot(retired, processor._render_cancel_event)
            self.assertEqual(seen_tokens, [False])
            with Image.open(output_root / "fresh-translated.png") as image:
                self.assertEqual(image.getpixel((0, 0)), (127, 0, 0))


if __name__ == "__main__":
    unittest.main()

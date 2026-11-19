from __future__ import annotations

import threading
import traceback
import unittest
from concurrent.futures import Future, wait as future_wait
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules.utils.exceptions import OperationCancelledError  # noqa: E402
from modules.utils.textblock import TextBlock  # noqa: E402
from pipeline.render_worker import RenderJobResult  # noqa: E402
from pipeline.stage_batched_processor import (  # noqa: E402
    StageBatchedProcessor,
    StagePageContext,
    _PriorRenderBookkeepingError,
)


class _ControlledRenderPool:
    def __init__(self) -> None:
        self.futures: list[Future] = []
        self.second_submitted = threading.Event()
        self.third_submitted = threading.Event()
        self.fourth_submitted = threading.Event()

    def submit(self, _fn, *_args) -> Future:
        future = Future()
        self.futures.append(future)
        if len(self.futures) >= 2:
            self.second_submitted.set()
        if len(self.futures) >= 3:
            self.third_submitted.set()
        if len(self.futures) >= 4:
            self.fourth_submitted.set()
        return future


def _processor() -> tuple[StageBatchedProcessor, _ControlledRenderPool]:
    processor = StageBatchedProcessor.__new__(StageBatchedProcessor)
    executor = _ControlledRenderPool()
    processor.main_page = SimpleNamespace(
        image_ctrl=SimpleNamespace(
            update_processing_summary=mock.Mock(),
            mark_processing_stage=mock.Mock(),
        ),
        render_state_ready=SimpleNamespace(emit=mock.Mock()),
        curr_img_idx=-1,
        image_files=[],
        button_to_alignment={1: 1},
        button_to_vertical_alignment={1: 1},
        is_current_task_cancelled=lambda: processor._render_cancel_event.is_set(),
    )
    processor._pending_render_jobs = []
    processor._render_cancel_event = threading.Event()
    processor._ensure_render_executor = lambda: executor
    processor._lazy_render_context = lambda _ctx: (
        SimpleNamespace(alignment_id=1),
        "ko",
    )
    processor._set_current_image = mock.Mock()
    processor._write_json_exports = mock.Mock()
    processor._emit_benchmark_event = mock.Mock()
    processor._prepare_render_checkpoint = mock.Mock(return_value=(None, ""))
    processor._reserve_render_output_path = mock.Mock(
        side_effect=lambda ctx, **_kwargs: (f"{ctx.image_path}.out.png", "output", "png")
    )
    processor._ensure_page_state = lambda _path: {"viewer_state": {}}
    processor._mark_page_failed = mock.Mock()
    processor._record_performance_detail = mock.Mock()
    processor._log_page_done = mock.Mock()
    processor.emit_progress = mock.Mock()
    processor._released_page_buffer_bytes = 0
    return processor, executor


def _page(index: int) -> StagePageContext:
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    return StagePageContext(
        image_path=f"example_source_{index}.png",
        image_name=f"example_source_{index}.png",
        source_lang="Japanese",
        target_lang="Korean",
        image=image,
        inpaint_input_img=image.copy(),
        mask=np.zeros((10, 10), dtype=np.uint8),
    )


class RenderBackpressureTests(unittest.TestCase):
    def test_inpaint_branch_propagates_prior_bookkeeping_error_unchanged(self) -> None:
        for no_text_detected in (False, True):
            with self.subTest(no_text_detected=no_text_detected):
                processor, executor = _processor()
                for index in range(2):
                    processor._submit_or_inline_render(
                        _page(index), index=index, total_images=3, export_settings={}
                    )
                first_future = executor.futures[0]
                block = TextBlock(
                    text_bbox=np.array([1, 1, 8, 8], dtype=np.int32),
                    text="protected",
                    text_class="sfx",
                    block_id="stable",
                )
                image = np.arange(10 * 10 * 3, dtype=np.uint8).reshape(10, 10, 3)
                current_page = StagePageContext(
                    image_path="example_source_current.png",
                    image_name="example_source_current.png",
                    source_lang="Japanese",
                    target_lang="Korean",
                    image=image,
                    blk_list=[block],
                    no_text_detected=no_text_detected,
                )
                processor.main_page.settings_page = SimpleNamespace(
                    get_hd_strategy_settings=lambda: {"strategy": "Original"},
                    get_mask_refiner_settings=lambda: {},
                    get_tool_selection=lambda _tool: "AOT",
                    ui=SimpleNamespace(value_mappings={"Original": "Original"}),
                )
                processor._effective_export_settings = mock.Mock(return_value={})
                processor._project_checkpoint_store = None
                processor._finish_inpaint_page = mock.Mock()
                processor._release_inpainter_before_render = mock.Mock()
                bookkeeping_cause = ValueError("example bookkeeping cause")
                bookkeeping_error = RuntimeError("example bookkeeping failure")
                bookkeeping_error.__cause__ = bookkeeping_cause
                processor._finish_render_page_bookkeeping = mock.Mock(
                    side_effect=bookkeeping_error
                )
                real_wait = future_wait

                def complete_at_capacity_wait(*args, **kwargs):
                    if kwargs.get("timeout") == 0.2 and not first_future.done():
                        first_future.set_result(
                            _result("example_previous_output.png")
                        )
                    return real_wait(*args, **kwargs)

                with mock.patch.object(StageBatchedProcessor, "_render_worker_count", return_value=1), mock.patch(
                    "pipeline.stage_batched_processor.psutil.virtual_memory",
                    return_value=SimpleNamespace(available=10**9),
                ), mock.patch(
                    "pipeline.stage_batched_processor.wait",
                    side_effect=complete_at_capacity_wait,
                ), mock.patch(
                    "pipeline.stage_batched_processor.get_inpainter_runtime",
                    return_value={"key": "AOT", "backend": "torch", "device": "cpu"},
                ), mock.patch(
                    "pipeline.stage_batched_processor.registered_inpainter_model_identity",
                    return_value={},
                ), mock.patch(
                    "pipeline.stage_batched_processor.get_config",
                    return_value={},
                ), mock.patch(
                    "pipeline.stage_batched_processor.split_inpaint_protected_ocr_blocks",
                    return_value=([], [block]),
                ):
                    caught = None
                    traceback_frames: set[str] = set()
                    try:
                        processor._inpaint_all([current_page])
                    except RuntimeError as exc:
                        caught = exc
                        traceback_frames = {
                            frame.name for frame in traceback.extract_tb(exc.__traceback__)
                        }
                    else:
                        self.fail("bookkeeping failure was swallowed")

                self.assertIsNotNone(caught)
                self.assertIs(caught, bookkeeping_error)
                self.assertIs(caught.__cause__, bookkeeping_cause)
                self.assertIn(
                    "_resolve_render_future",
                    traceback_frames,
                )
                self.assertEqual(current_page.failed_stage, "")
                processor._mark_page_failed.assert_not_called()
                self.assertEqual(len(executor.futures), 2)

    def test_prior_bookkeeping_error_propagates_without_failing_current_page(self) -> None:
        processor, executor = _processor()
        for index in range(2):
            processor._submit_or_inline_render(
                _page(index), index=index, total_images=3, export_settings={}
            )
        executor.futures[0].set_result(_result("example_output_0.png"))
        bookkeeping_error = RuntimeError("example bookkeeping failure")
        processor._finish_render_page_bookkeeping = mock.Mock(
            side_effect=bookkeeping_error
        )
        current_page = _page(2)

        with mock.patch.object(StageBatchedProcessor, "_render_worker_count", return_value=1), mock.patch(
            "pipeline.stage_batched_processor.psutil.virtual_memory",
            return_value=SimpleNamespace(available=10**9),
        ):
            with self.assertRaises(
                _PriorRenderBookkeepingError
            ) as caught:
                processor._submit_or_inline_render(
                    current_page,
                    index=2,
                    total_images=3,
                    export_settings={},
                )

        self.assertEqual(len(executor.futures), 2)
        self.assertEqual(len(processor._pending_render_jobs), 1)
        self.assertIs(caught.exception.original, bookkeeping_error)
        self.assertIs(caught.exception.__cause__, bookkeeping_error)
        processor._mark_page_failed.assert_not_called()
        self.assertEqual(current_page.failed_stage, "")

    def test_model_forward_drain_preserves_bookkeeping_failures_and_cancellation(
        self,
    ) -> None:
        for cancel_before_drain in (False, True):
            with self.subTest(cancel_before_drain=cancel_before_drain):
                processor, executor = _processor()
                processor._submit_or_inline_render(
                    _page(0), index=0, total_images=2, export_settings={}
                )
                previous_future = executor.futures[0]
                current_page = _page(1)
                block = TextBlock(
                    text_bbox=np.array([1, 1, 8, 8], dtype=np.int32),
                    text="protected",
                    text_class="sfx",
                    block_id="stable",
                )
                current_page.blk_list = [block]
                processor.main_page.settings_page = SimpleNamespace(
                    get_hd_strategy_settings=lambda: {"strategy": "Original"},
                    get_mask_refiner_settings=lambda: {},
                    get_tool_selection=lambda _tool: "AOT",
                    ui=SimpleNamespace(value_mappings={"Original": "Original"}),
                )
                processor._effective_export_settings = mock.Mock(return_value={})
                processor._ensure_source_image = lambda ctx: ctx.image
                processor._finish_inpaint_page = mock.Mock()
                processor._release_inpainter_before_render = mock.Mock()
                processor._measure_performance = mock.Mock(
                    return_value=mock.MagicMock()
                )
                processor._sample_performance_resources = mock.Mock()

                def model_ready() -> None:
                    if cancel_before_drain:
                        processor._render_cancel_event.set()
                    previous_future.set_result(
                        _result("example_previous_output.png")
                    )

                processor._ensure_inpainter = mock.Mock(side_effect=model_ready)
                processor.inpainting = SimpleNamespace(
                    inpaint_with_blocks=mock.Mock(),
                )
                bookkeeping_cause = ValueError("example bookkeeping cause")
                bookkeeping_error = RuntimeError("example bookkeeping failure")
                bookkeeping_error.__cause__ = bookkeeping_cause
                processor._finish_render_page_bookkeeping = mock.Mock(
                    side_effect=None if cancel_before_drain else bookkeeping_error
                )

                with mock.patch.object(StageBatchedProcessor, "_render_worker_count", return_value=1), mock.patch(
                    "pipeline.stage_batched_processor.psutil.virtual_memory",
                    return_value=SimpleNamespace(available=10**9),
                ), mock.patch(
                    "pipeline.stage_batched_processor.get_inpainter_runtime",
                    return_value={"key": "AOT", "backend": "torch", "device": "cpu"},
                ), mock.patch(
                    "pipeline.stage_batched_processor.registered_inpainter_model_identity",
                    return_value={},
                ), mock.patch(
                    "pipeline.stage_batched_processor.get_config",
                    return_value={},
                ), mock.patch(
                    "pipeline.stage_batched_processor.split_inpaint_protected_ocr_blocks",
                    return_value=([block], []),
                ), mock.patch(
                    "pipeline.stage_batched_processor.generate_mask",
                    return_value={
                        "final_mask": np.ones((10, 10), dtype=np.uint8),
                        "raw_mask": np.ones((10, 10), dtype=np.uint8),
                    },
                ):
                    if cancel_before_drain:
                        with self.assertRaises(OperationCancelledError):
                            processor._inpaint_all([current_page])
                        processor._finish_render_page_bookkeeping.assert_not_called()
                        processor.main_page.render_state_ready.emit.assert_not_called()
                        self.assertEqual(len(processor._pending_render_jobs), 1)
                    else:
                        try:
                            processor._inpaint_all([current_page])
                        except RuntimeError as exc:
                            caught = exc
                        else:
                            processor._mark_page_failed.assert_called_once()
                            self.assertIs(
                                processor._mark_page_failed.call_args.args[0],
                                current_page,
                            )
                            self.fail(
                                "prior render bookkeeping failure was attributed to the current page"
                            )

                        self.assertIs(caught, bookkeeping_error)
                        self.assertIs(caught.__cause__, bookkeeping_cause)
                        traceback_frames = {
                            frame.name
                            for frame in traceback.extract_tb(caught.__traceback__)
                        }
                        self.assertIn("_resolve_render_future", traceback_frames)
                        self.assertIn("_drain_render_futures", traceback_frames)
                        self.assertEqual(processor._pending_render_jobs, [])

                processor._mark_page_failed.assert_not_called()
                self.assertEqual(current_page.failed_stage, "")
                processor._release_inpainter_before_render.assert_called_once_with(
                    [current_page], handoff_outcome="aborted"
                )
                processor.inpainting.inpaint_with_blocks.assert_not_called()

    def test_outstanding_limit_waits_for_failed_future_then_submits(self) -> None:
        processor, executor = _processor()
        wait_started = threading.Event()
        thread_errors: list[BaseException] = []
        real_wait = future_wait

        def observed_wait(*args, **kwargs):
            if kwargs.get("timeout") == 0.2:
                wait_started.set()
            return real_wait(*args, **kwargs)

        with mock.patch.object(StageBatchedProcessor, "_render_worker_count", return_value=1):
            with mock.patch(
                "pipeline.stage_batched_processor.psutil.virtual_memory",
                return_value=SimpleNamespace(available=10**9),
            ), mock.patch(
                "pipeline.stage_batched_processor.wait",
                side_effect=observed_wait,
            ):
                for index in range(2):
                    processor._submit_or_inline_render(
                        _page(index),
                        index=index,
                        total_images=3,
                        export_settings={},
                    )

                def submit_third() -> None:
                    try:
                        processor._submit_or_inline_render(
                            _page(2),
                            index=2,
                            total_images=3,
                            export_settings={},
                        )
                    except BaseException as exc:
                        thread_errors.append(exc)

                thread = threading.Thread(target=submit_third)
                thread.start()
                self.assertTrue(wait_started.wait(2))
                self.assertEqual(len(executor.futures), 2)
                executor.futures[0].set_exception(RuntimeError("example render failure"))
                self.assertTrue(executor.third_submitted.wait(2))
                thread.join(timeout=2)

        self.assertFalse(thread.is_alive())
        self.assertEqual(thread_errors, [])
        self.assertEqual(len(processor._pending_render_jobs), 2)
        processor._mark_page_failed.assert_called_once()
        self.assertEqual(
            processor._mark_page_failed.call_args.args[0].image_path,
            "example_source_0.png",
        )
        self.assertEqual(
            processor._mark_page_failed.call_args.kwargs["stage"],
            "render",
        )
        for future in executor.futures[1:]:
            future.set_exception(RuntimeError("example cleanup failure"))
        processor._drain_render_futures(block=True)
        self.assertEqual(processor._pending_render_jobs, [])

    def test_byte_budget_waits_until_existing_input_buffers_finish(self) -> None:
        processor, executor = _processor()
        wait_started = threading.Event()
        thread_errors: list[BaseException] = []
        real_wait = future_wait

        def observed_wait(*args, **kwargs):
            if kwargs.get("timeout") == 0.2:
                wait_started.set()
            return real_wait(*args, **kwargs)

        def small_page(index: int) -> StagePageContext:
            return StagePageContext(
                image_path=f"example_source_{index}.png",
                image_name=f"example_source_{index}.png",
                source_lang="Japanese",
                target_lang="Korean",
                image=np.zeros((10, 10), dtype=np.uint8),
            )

        with mock.patch.object(StageBatchedProcessor, "_render_worker_count", return_value=1):
            with mock.patch(
                "pipeline.stage_batched_processor.psutil.virtual_memory",
                return_value=SimpleNamespace(available=300),
            ), mock.patch(
                "pipeline.stage_batched_processor.wait",
                side_effect=observed_wait,
            ):
                processor._submit_or_inline_render(
                    small_page(0), index=0, total_images=2, export_settings={}
                )

                def submit_second() -> None:
                    try:
                        processor._submit_or_inline_render(
                            small_page(1),
                            index=1,
                            total_images=2,
                            export_settings={},
                        )
                    except BaseException as exc:
                        thread_errors.append(exc)

                thread = threading.Thread(target=submit_second)
                thread.start()
                self.assertTrue(wait_started.wait(2))
                self.assertEqual(len(executor.futures), 1)
                executor.futures[0].set_exception(RuntimeError("example render failure"))
                self.assertTrue(executor.second_submitted.wait(2))
                thread.join(timeout=2)

        self.assertFalse(thread.is_alive())
        self.assertEqual(thread_errors, [])
        self.assertEqual(len(executor.futures), 2)

    def test_three_configured_workers_keep_three_outstanding_slots(self) -> None:
        processor, executor = _processor()
        wait_started = threading.Event()
        thread_errors: list[BaseException] = []
        real_wait = future_wait

        def observed_wait(*args, **kwargs):
            if kwargs.get("timeout") == 0.2:
                wait_started.set()
            return real_wait(*args, **kwargs)

        with mock.patch.object(StageBatchedProcessor, "_render_worker_count", return_value=3):
            with mock.patch(
                "pipeline.stage_batched_processor.psutil.virtual_memory",
                return_value=SimpleNamespace(available=10**9),
            ), mock.patch(
                "pipeline.stage_batched_processor.wait",
                side_effect=observed_wait,
            ):
                for index in range(3):
                    processor._submit_or_inline_render(
                        _page(index), index=index, total_images=4, export_settings={}
                    )

                def submit_fourth() -> None:
                    try:
                        processor._submit_or_inline_render(
                            _page(3), index=3, total_images=4, export_settings={}
                        )
                    except BaseException as exc:
                        thread_errors.append(exc)

                thread = threading.Thread(target=submit_fourth)
                thread.start()
                self.assertTrue(wait_started.wait(2))
                self.assertEqual(len(executor.futures), 3)
                executor.futures[0].set_exception(RuntimeError("example render failure"))
                self.assertTrue(executor.fourth_submitted.wait(2))
                thread.join(timeout=2)

        self.assertFalse(thread.is_alive())
        self.assertEqual(thread_errors, [])
        self.assertEqual(len(executor.futures), 4)
        self.assertEqual(len(processor._pending_render_jobs), 3)
        for future in executor.futures[1:]:
            future.set_exception(RuntimeError("example cleanup failure"))
        processor._drain_render_futures(block=True)

    def test_large_input_is_allowed_when_no_render_is_pending(self) -> None:
        processor, executor = _processor()

        with mock.patch.object(StageBatchedProcessor, "_render_worker_count", return_value=1), mock.patch(
            "pipeline.stage_batched_processor.psutil.virtual_memory",
            return_value=SimpleNamespace(available=10),
        ):
            processor._submit_or_inline_render(
                _page(0), index=0, total_images=1, export_settings={}
            )

        self.assertEqual(len(executor.futures), 1)

    def test_cancellation_after_capacity_wait_prevents_success_bookkeeping(self) -> None:
        processor, executor = _processor()
        wait_started = threading.Event()
        thread_errors: list[BaseException] = []
        real_wait = future_wait

        def observed_wait(*args, **kwargs):
            if kwargs.get("timeout") == 0.2:
                wait_started.set()
            return real_wait(*args, **kwargs)

        with mock.patch.object(StageBatchedProcessor, "_render_worker_count", return_value=1):
            with mock.patch(
                "pipeline.stage_batched_processor.psutil.virtual_memory",
                return_value=SimpleNamespace(available=10**9),
            ), mock.patch(
                "pipeline.stage_batched_processor.wait",
                side_effect=observed_wait,
            ):
                for index in range(2):
                    processor._submit_or_inline_render(
                        _page(index), index=index, total_images=3, export_settings={}
                    )

                def submit_third() -> None:
                    try:
                        processor._submit_or_inline_render(
                            _page(2), index=2, total_images=3, export_settings={}
                        )
                    except BaseException as exc:
                        thread_errors.append(exc)

                thread = threading.Thread(target=submit_third)
                thread.start()
                self.assertTrue(wait_started.wait(2))
                processor._render_cancel_event.set()
                executor.futures[0].set_result(_result("example_output_0.png"))
                thread.join(timeout=2)

        self.assertFalse(thread.is_alive())
        self.assertEqual(len(thread_errors), 1)
        self.assertIsInstance(thread_errors[0], OperationCancelledError)
        self.assertEqual(len(executor.futures), 2)
        processor.main_page.image_ctrl.update_processing_summary.assert_not_called()

    def test_checkpoint_hit_skips_capacity_wait(self) -> None:
        processor, executor = _processor()
        for index in range(2):
            processor._submit_or_inline_render(
                _page(index), index=index, total_images=3, export_settings={}
            )
        hit = SimpleNamespace(output_root="output", output_exists=True)
        processor._prepare_render_checkpoint.return_value = (hit, "output")
        processor._restore_render_project_state = mock.Mock()
        processor._finish_render_checkpoint_hit = mock.Mock()

        with mock.patch(
            "pipeline.stage_batched_processor.materialize_render_checkpoint_output",
            return_value="example_cached_output.png",
        ):
            processor._submit_or_inline_render(
                _page(2), index=2, total_images=3, export_settings={}
            )

        self.assertEqual(len(executor.futures), 2)
        processor._finish_render_checkpoint_hit.assert_called_once()

    def test_out_of_order_futures_keep_their_page_outputs(self) -> None:
        processor, executor = _processor()
        pages = [_page(0), _page(1)]
        states: dict[str, dict] = {}
        processor._ensure_page_state = lambda path: states.setdefault(
            path, {"viewer_state": {}}
        )

        with mock.patch.object(StageBatchedProcessor, "_render_worker_count", return_value=2), mock.patch(
            "pipeline.stage_batched_processor.psutil.virtual_memory",
            return_value=SimpleNamespace(available=10**9),
        ):
            for index, ctx in enumerate(pages):
                processor._submit_or_inline_render(
                    ctx, index=index, total_images=2, export_settings={}
                )

            executor.futures[1].set_result(_result("example_output_1.png"))
            processor._drain_render_futures(block=False)
            executor.futures[0].set_result(_result("example_output_0.png"))
            processor._drain_render_futures(block=True)

        self.assertEqual(pages[0].output_path, "example_output_0.png")
        self.assertEqual(pages[1].output_path, "example_output_1.png")
        self.assertEqual(processor._pending_render_jobs, [])


def _result(path: str) -> RenderJobResult:
    return RenderJobResult(
        viewer_state={"text_items_state": []},
        blk_rendered_events=[],
        restore_applied=False,
        restore_stats={},
        inpaint_input_img=None,
        mask=None,
        patches=[],
        final_output_path=path,
    )


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import os
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest import mock

import numpy as np

from modules.ocr.factory import OCRFactory
from modules.ocr.local_runtime import LocalOCRRuntimeManager
from modules.ocr.ocr_paddle_VL import PaddleOCRVLEngine
from modules.translation.local_runtime import LocalGemmaRuntimeManager
from modules.utils.textblock import TextBlock
from pipeline.stage_batched_processor import StageBatchedProcessor, StagePageContext
from modules.utils.exceptions import OperationCancelledError


class StageBatchedCancellationTests(unittest.TestCase):
    def _processor(self, *, cancelled: bool) -> StageBatchedProcessor:
        processor = object.__new__(StageBatchedProcessor)
        processor.main_page = SimpleNamespace(
            is_current_task_cancelled=lambda: cancelled,
            settings_page=object(),
        )
        processor.block_detection = SimpleNamespace(block_detector_cache=object())
        return processor

    def test_cancel_check_raises_operation_cancelled_error(self) -> None:
        processor = self._processor(cancelled=True)

        with self.assertRaises(OperationCancelledError):
            processor._raise_if_cancelled()

    def test_detect_stage_stops_before_processing_page_when_cancelled(self) -> None:
        processor = self._processor(cancelled=True)
        page = StagePageContext(
            image_path="page.png",
            image_name="page.png",
            source_lang="Japanese",
            target_lang="Korean",
        )

        with self.assertRaises(OperationCancelledError):
            processor._detect_all([page])

    def test_prewarm_fallback_is_not_called_after_cancel(self) -> None:
        processor = self._processor(cancelled=True)
        processor._prewarm_jobs = {}
        called = False

        def fallback() -> None:
            nonlocal called
            called = True

        with self.assertRaises(OperationCancelledError):
            processor._await_prewarm_or_run("ocr", "OCR", "hunyuanocr", fallback)

        self.assertFalse(called)

    def test_paddle_persistent_cache_defers_ocr_prewarm_until_lookup(self) -> None:
        processor = self._processor(cancelled=False)
        processor.main_page.local_ocr_runtime_manager = LocalOCRRuntimeManager()
        processor.main_page.settings_page = SimpleNamespace(
            get_paddleocr_vl_settings=lambda: {
                "persistent_cache_enabled": True,
            }
        )
        processor._get_paddleocr_cache_store = mock.Mock(
            return_value=SimpleNamespace(
                stats=lambda: {
                    "enabled": True,
                    "item_count": 1,
                }
            )
        )
        processor._start_prewarm = mock.Mock()

        processor._start_ocr_prewarm(
            {"primary_ocr_engine": "PaddleOCR VL"}
        )

        processor._start_prewarm.assert_not_called()

    def test_empty_paddle_cache_preserves_cold_runtime_prewarm(self) -> None:
        processor = self._processor(cancelled=False)
        processor.main_page.local_ocr_runtime_manager = LocalOCRRuntimeManager()
        processor.main_page.settings_page = SimpleNamespace(
            get_paddleocr_vl_settings=lambda: {
                "persistent_cache_enabled": True,
            }
        )
        processor._get_paddleocr_cache_store = mock.Mock(
            return_value=SimpleNamespace(
                stats=lambda: {
                    "enabled": True,
                    "item_count": 0,
                }
            )
        )
        processor._start_prewarm = mock.Mock()

        processor._start_ocr_prewarm(
            {"primary_ocr_engine": "PaddleOCR VL"}
        )

        processor._start_prewarm.assert_called_once()

    def test_paddle_all_hit_lookup_skips_runtime_start(self) -> None:
        processor = self._processor(cancelled=False)
        runtime_manager = LocalOCRRuntimeManager()
        processor.main_page.local_ocr_runtime_manager = runtime_manager
        processor.main_page.settings_page = SimpleNamespace(
            get_paddleocr_vl_settings=lambda: {
                "persistent_cache_enabled": True,
            },
            is_gpu_enabled=lambda: False,
            get_tool_selection=lambda _tool: "PaddleOCR VL",
        )
        processor.main_page.image_ctrl = SimpleNamespace(
            mark_processing_stage=mock.Mock(),
        )
        processor._paddleocr_cache_store = None
        processor._paddleocr_cache_identity = None
        processor._await_ocr_runtime = mock.Mock()
        processor._prepare_paddleocr_cache_plans = mock.Mock(
            return_value=False
        )
        processor._set_current_image = mock.Mock()
        processor.emit_progress = mock.Mock()
        processor._emit_benchmark_event = mock.Mock()
        processor._log_ocr_quality = mock.Mock()
        processor._persist_ocr_state = mock.Mock()
        processor._shutdown_runtime_with_retry = mock.Mock()
        processor._raise_if_cancelled = mock.Mock()
        processor._run_primary_ocr = mock.Mock(
            return_value={
                "quality": {"non_empty": 1, "low_quality": False},
                "metrics": {"ocr_non_empty_block_count": 1},
                "cache_status": "hit",
                "attempt_count": 1,
                "page_profile": {
                    "persistent_cache": {
                        "hit_count": 1,
                        "miss_count": 0,
                    }
                },
                "engine_name": "PaddleOCRVLEngine",
            }
        )
        page = StagePageContext(
            image_path="page.png",
            image_name="page.png",
            source_lang="Japanese",
            target_lang="Korean",
            image=object(),
            blk_list=[object()],
        )
        identity = {"runtime_fingerprint": "runtime"}

        with mock.patch.object(
            runtime_manager,
            "get_ocr_cache_identity",
            return_value=identity,
        ):
            processor._ocr_all(
                [page],
                {
                    "primary_ocr_engine": "PaddleOCR VL",
                    "normalized_ocr_mode": "best_local",
                },
            )

        processor._prepare_paddleocr_cache_plans.assert_called_once_with(
            [page],
            {
                "primary_ocr_engine": "PaddleOCR VL",
                "normalized_ocr_mode": "best_local",
            },
            identity,
        )
        processor._await_ocr_runtime.assert_not_called()
        processor._run_primary_ocr.assert_called_once()

    def test_paddle_cache_hit_applies_current_dictionary_once(self) -> None:
        processor = self._processor(cancelled=False)
        processor.main_page.lang_mapping = {"Japanese": "Japanese"}
        processor.main_page.settings_page = SimpleNamespace(
            is_gpu_enabled=lambda: False,
            get_ocr_result_dictionary_rules=lambda: [
                {"source": "raw", "target": "dict"}
            ],
        )
        processor.cache_manager = mock.Mock()
        processor._paddleocr_cache_store = None
        processor._paddleocr_cache_identity = {"runtime": "identity"}
        block = TextBlock(
            text_bbox=np.array([10, 10, 100, 100], dtype=np.int32),
            text_class="text_bubble",
            source_lang="ja",
        )
        engine = PaddleOCRVLEngine()
        plan = SimpleNamespace(
            lookup_disabled=False,
            all_hit=True,
            hit_count=1,
            requires_runtime=False,
        )

        def restore_raw(_plan) -> None:
            block.text = "raw"
            block.ocr_status = "ok"

        engine.process_persistent_cache_plan = mock.Mock(
            side_effect=restore_raw
        )
        engine.build_persistent_cache_records = mock.Mock(return_value=[])
        engine.last_page_profile = {}
        page = StagePageContext(
            image_path="page.png",
            image_name="page.png",
            source_lang="Japanese",
            target_lang="Korean",
            image=np.zeros((120, 120, 3), dtype=np.uint8),
            blk_list=[block],
            paddleocr_cache_plan=plan,
            paddleocr_cache_engine=engine,
        )

        def apply_dictionary(blocks, _rules) -> None:
            for item in blocks:
                item.text = item.text.replace("raw", "dict")

        with mock.patch(
            "pipeline.stage_batched_processor.apply_ocr_result_dictionary",
            side_effect=apply_dictionary,
        ) as dictionary:
            result = processor._run_primary_ocr(
                page,
                {
                    "primary_ocr_engine": "PaddleOCR VL",
                    "normalized_ocr_mode": "best_local",
                },
            )

        dictionary.assert_called_once()
        self.assertEqual(block.text, "dict")
        self.assertEqual(result["cache_status"], "hit")

    def test_paddle_folder_path_never_uses_sampled_fuzzy_memory_cache(self) -> None:
        processor = self._processor(cancelled=False)
        processor.main_page.lang_mapping = {"Japanese": "Japanese"}
        processor.main_page.settings_page = SimpleNamespace(
            is_gpu_enabled=lambda: False,
            get_ocr_result_dictionary_rules=lambda: [],
        )
        processor.cache_manager = mock.Mock()
        processor._paddleocr_cache_store = None
        processor._paddleocr_cache_identity = None
        block = TextBlock(
            text_bbox=np.array([10, 10, 100, 100], dtype=np.int32),
            text_class="text_bubble",
            source_lang="ja",
        )
        engine = PaddleOCRVLEngine()

        def process(_image, blocks) -> None:
            blocks[0].text = "raw"
            blocks[0].ocr_status = "ok"

        engine.process_image = mock.Mock(side_effect=process)
        engine.last_page_profile = {}
        page = StagePageContext(
            image_path="page.png",
            image_name="page.png",
            source_lang="Japanese",
            target_lang="Korean",
            image=np.zeros((120, 120, 3), dtype=np.uint8),
            blk_list=[block],
        )

        with mock.patch.object(OCRFactory, "create_engine", return_value=engine):
            processor._run_primary_ocr(
                page,
                {
                    "primary_ocr_engine": "PaddleOCR VL",
                    "normalized_ocr_mode": "best_local",
                },
            )

        processor.cache_manager._get_ocr_cache_key.assert_not_called()
        processor.cache_manager._can_serve_all_blocks_from_ocr_cache.assert_not_called()
        processor.cache_manager._cache_ocr_results.assert_not_called()

    def test_paddle_cache_plan_failure_is_page_scoped(self) -> None:
        processor = self._processor(cancelled=False)
        processor.main_page.lang_mapping = {"Japanese": "Japanese"}
        processor.main_page.settings_page = object()
        processor._get_paddleocr_cache_store = mock.Mock(return_value=object())
        processor._paddleocr_cache_store = None
        processor._paddleocr_cache_identity = None
        processor._mark_page_failed = mock.Mock(
            side_effect=lambda page, **_kwargs: setattr(page, "failed_stage", "ocr")
        )
        first_engine = PaddleOCRVLEngine()
        second_engine = PaddleOCRVLEngine()
        first_engine.prepare_persistent_cache = mock.Mock(
            side_effect=RuntimeError("cache-plan-failure")
        )
        second_plan = SimpleNamespace(requires_runtime=False)
        second_engine.prepare_persistent_cache = mock.Mock(
            return_value=second_plan
        )
        pages = [
            StagePageContext(
                image_path="page-1.png",
                image_name="page-1.png",
                source_lang="Japanese",
                target_lang="Korean",
                image=np.zeros((120, 120, 3), dtype=np.uint8),
                blk_list=[
                    TextBlock(
                        text_bbox=np.array([10, 10, 100, 100], dtype=np.int32)
                    )
                ],
            ),
            StagePageContext(
                image_path="page-2.png",
                image_name="page-2.png",
                source_lang="Japanese",
                target_lang="Korean",
                image=np.zeros((120, 120, 3), dtype=np.uint8),
                blk_list=[
                    TextBlock(
                        text_bbox=np.array([10, 10, 100, 100], dtype=np.int32)
                    )
                ],
            ),
        ]

        with mock.patch.object(
            OCRFactory,
            "create_engine",
            side_effect=[first_engine, second_engine],
        ):
            requires_runtime = processor._prepare_paddleocr_cache_plans(
                pages,
                {
                    "primary_ocr_engine": "PaddleOCR VL",
                    "normalized_ocr_mode": "best_local",
                },
                {"runtime_fingerprint": "runtime"},
            )

        self.assertFalse(requires_runtime)
        self.assertEqual(pages[0].failed_stage, "ocr")
        self.assertIsNone(pages[0].paddleocr_cache_plan)
        self.assertIs(pages[1].paddleocr_cache_engine, second_engine)
        self.assertIs(pages[1].paddleocr_cache_plan, second_plan)
        processor._mark_page_failed.assert_called_once()

    def test_batch_cleanup_stops_both_managed_runtimes_after_cancel(self) -> None:
        processor = self._processor(cancelled=True)
        ocr_manager = LocalOCRRuntimeManager()
        gemma_manager = LocalGemmaRuntimeManager()
        processor.main_page.local_ocr_runtime_manager = ocr_manager
        processor.main_page.local_translation_runtime_manager = gemma_manager

        with mock.patch.object(ocr_manager, "shutdown") as shutdown_ocr, \
             mock.patch.object(gemma_manager, "shutdown") as shutdown_gemma:
            processor._shutdown_managed_runtimes()

        shutdown_ocr.assert_called_once_with()
        shutdown_gemma.assert_called_once_with()

    def test_batch_cleanup_still_stops_gemma_when_ocr_shutdown_fails(self) -> None:
        processor = self._processor(cancelled=True)
        ocr_manager = LocalOCRRuntimeManager()
        gemma_manager = LocalGemmaRuntimeManager()
        processor.main_page.local_ocr_runtime_manager = ocr_manager
        processor.main_page.local_translation_runtime_manager = gemma_manager

        with mock.patch.object(
            ocr_manager,
            "shutdown",
            side_effect=RuntimeError("stop failed"),
        ) as shutdown_ocr, \
             mock.patch.object(gemma_manager, "shutdown") as shutdown_gemma, \
             self.assertLogs("pipeline.stage_batched_processor", level="WARNING"):
            processor._shutdown_managed_runtimes()

        self.assertEqual(shutdown_ocr.call_count, 2)
        shutdown_gemma.assert_called_once_with()

    def test_batch_startup_preflight_stops_both_runtimes_then_fails_closed(self) -> None:
        processor = self._processor(cancelled=False)
        ocr_manager = LocalOCRRuntimeManager()
        gemma_manager = LocalGemmaRuntimeManager()
        processor.main_page.local_ocr_runtime_manager = ocr_manager
        processor.main_page.local_translation_runtime_manager = gemma_manager

        with mock.patch.object(
            ocr_manager,
            "shutdown",
            side_effect=RuntimeError("retained OCR runtime"),
        ) as shutdown_ocr, mock.patch.object(
            gemma_manager,
            "shutdown",
        ) as shutdown_gemma, self.assertLogs(
            "pipeline.stage_batched_processor",
            level="WARNING",
        ), self.assertRaisesRegex(RuntimeError, "retained OCR runtime"):
            processor._shutdown_managed_runtimes(
                context="batch startup preflight",
                raise_on_failure=True,
            )

        self.assertEqual(shutdown_ocr.call_count, 2)
        shutdown_gemma.assert_called_once_with()

    def test_batch_ocr_stage_ceiling_skips_later_stages(self) -> None:
        processor = self._processor(cancelled=False)
        processor.main_page.image_files = ["page.png"]
        processor.main_page.file_handler = SimpleNamespace(
            should_pre_materialize=lambda _paths: False,
        )
        processor.main_page.reset_automatic_output_reservations = mock.Mock()
        processor._recent_page_durations = []
        pages = [object()]
        processor._emit_benchmark_event = mock.Mock()
        processor._reset_prewarm_lifecycle = mock.Mock()
        processor._load_page_contexts = mock.Mock(return_value=pages)
        processor._ensure_stage_policy = mock.Mock(return_value={"primary_ocr_engine": "PaddleOCR VL"})
        processor._raise_if_cancelled = mock.Mock()
        processor._shutdown_managed_runtimes = mock.Mock()
        processor._start_ocr_prewarm = mock.Mock()
        processor._detect_all = mock.Mock()
        processor._ocr_all = mock.Mock()
        processor._complete_ocr_stage_ceiling = mock.Mock()
        processor._inpaint_all = mock.Mock()
        processor._translate_all = mock.Mock()
        processor._render_all = mock.Mock()
        processor._shutdown_prewarm_executor = mock.Mock()

        with mock.patch.dict(os.environ, {"CT_BENCH_STAGE_CEILING": "ocr"}):
            processor.batch_process()

        processor._detect_all.assert_called_once_with(pages)
        processor._ocr_all.assert_called_once_with(
            pages,
            {"primary_ocr_engine": "PaddleOCR VL"},
        )
        processor._complete_ocr_stage_ceiling.assert_called_once_with(pages)
        processor._inpaint_all.assert_not_called()
        processor._translate_all.assert_not_called()
        processor._render_all.assert_not_called()
        processor._emit_benchmark_event.assert_any_call(
            "batch_run_done",
            total_images=1,
            stage_ceiling="ocr",
        )

    def test_complete_ocr_stage_ceiling_marks_only_successful_pages_done(self) -> None:
        processor = self._processor(cancelled=False)
        processor.main_page.image_ctrl = SimpleNamespace(
            update_processing_summary=mock.Mock(),
        )
        processor._emit_benchmark_event = mock.Mock()
        processor._log_page_done = mock.Mock()
        processor.emit_progress = mock.Mock()
        successful = StagePageContext(
            image_path="success.png",
            image_name="success.png",
            source_lang="Japanese",
            target_lang="Korean",
            blk_list=[object()],
            page_ocr_metrics={"ocr_non_empty_block_count": 1},
        )
        failed = StagePageContext(
            image_path="failed.png",
            image_name="failed.png",
            source_lang="Japanese",
            target_lang="Korean",
            failed_stage="ocr",
        )

        processor._complete_ocr_stage_ceiling([successful, failed])

        processor.main_page.image_ctrl.update_processing_summary.assert_called_once_with(
            "success.png",
            {"benchmark_stage_ceiling": "ocr"},
        )
        processor._emit_benchmark_event.assert_called_once_with(
            "page_done",
            image_path="success.png",
            image_index=0,
            total_images=2,
            block_count=1,
            patch_count=0,
            stage_ceiling="ocr",
            ocr_non_empty_block_count=1,
        )
        processor._log_page_done.assert_called_once_with(0, 2, "success.png")
        processor.emit_progress.assert_called_once_with(0, 2, 10, 10, False)

    def test_stage_transition_continues_after_verified_stop_retry(self) -> None:
        processor = self._processor(cancelled=False)
        for label, manager, next_stage in (
            ("OCR", LocalOCRRuntimeManager(), "inpaint"),
            ("Gemma", LocalGemmaRuntimeManager(), "render"),
        ):
            reached: list[str] = []
            with self.subTest(label=label), mock.patch.object(
                manager,
                "shutdown",
                side_effect=[RuntimeError("first stop failed"), None],
            ) as shutdown, self.assertLogs(
                "pipeline.stage_batched_processor",
                level="WARNING",
            ):
                processor._shutdown_runtime_with_retry(
                    label,
                    manager,
                    context=f"before {next_stage}",
                    raise_on_failure=True,
                )
                reached.append(next_stage)

            self.assertEqual(shutdown.call_count, 2)
            self.assertEqual(reached, [next_stage])

    def test_stage_transition_fails_closed_after_two_stop_failures(self) -> None:
        processor = self._processor(cancelled=False)
        manager = LocalGemmaRuntimeManager()

        with mock.patch.object(
            manager,
            "shutdown",
            side_effect=RuntimeError("stop failed"),
        ) as shutdown, self.assertLogs(
            "pipeline.stage_batched_processor",
            level="WARNING",
        ), self.assertRaisesRegex(RuntimeError, "stop failed"):
            processor._shutdown_runtime_with_retry(
                "Gemma",
                manager,
                context="translation-to-render handoff",
                raise_on_failure=True,
            )

        self.assertEqual(shutdown.call_count, 2)

    def test_prewarm_shutdown_waits_and_cancels_queued_late_start(self) -> None:
        processor = self._processor(cancelled=False)
        processor._prewarm_cancel_event = threading.Event()
        processor._prewarm_executor = ThreadPoolExecutor(max_workers=1)
        processor._prewarm_jobs = {}
        blocker_started = threading.Event()
        release_blocker = threading.Event()
        late_start_called = threading.Event()

        def blocker() -> None:
            blocker_started.set()
            release_blocker.wait(timeout=5.0)

        def late_start() -> None:
            late_start_called.set()

        blocker_job = processor._prewarm_executor.submit(blocker)
        late_job = processor._prewarm_executor.submit(late_start)
        processor._prewarm_jobs = {
            "ocr": blocker_job,
            "gemma": late_job,
        }
        self.assertTrue(blocker_started.wait(timeout=2.0))

        shutdown_done = threading.Event()

        def shutdown() -> None:
            processor._shutdown_prewarm_executor()
            shutdown_done.set()

        shutdown_thread = threading.Thread(target=shutdown)
        shutdown_thread.start()
        deadline = time.monotonic() + 2.0
        while not processor._prewarm_cancel_event.is_set() and time.monotonic() < deadline:
            time.sleep(0.01)

        self.assertTrue(processor._prewarm_cancel_event.is_set())
        self.assertFalse(shutdown_done.is_set())
        release_blocker.set()
        shutdown_thread.join(timeout=2.0)

        self.assertTrue(shutdown_done.is_set())
        self.assertFalse(late_start_called.is_set())
        self.assertTrue(late_job.cancelled())
        self.assertEqual(processor._prewarm_jobs, {})

    def test_running_prewarm_cannot_report_ready_after_shutdown(self) -> None:
        processor = self._processor(cancelled=False)
        processor._prewarm_cancel_event = threading.Event()
        processor._prewarm_executor = None
        processor._prewarm_jobs = {}
        progress_statuses: list[str] = []
        processor._prewarm_progress = lambda **payload: progress_statuses.append(
            str(payload.get("status") or "")
        )
        started = threading.Event()
        release = threading.Event()

        def prewarm() -> None:
            started.set()
            release.wait(timeout=5.0)

        processor._start_prewarm("gemma", "Gemma", "gemma", prewarm)
        self.assertTrue(started.wait(timeout=2.0))
        shutdown_thread = threading.Thread(target=processor._shutdown_prewarm_executor)
        shutdown_thread.start()
        deadline = time.monotonic() + 2.0
        while not processor._prewarm_cancel_event.is_set() and time.monotonic() < deadline:
            time.sleep(0.01)
        release.set()
        shutdown_thread.join(timeout=2.0)

        self.assertFalse(shutdown_thread.is_alive())
        self.assertEqual(progress_statuses, ["starting"])

    def test_all_hit_folder_skips_gemma_prewarm_and_readiness_wait(self) -> None:
        processor = self._processor(cancelled=False)
        processor.main_page.settings_page = SimpleNamespace(
            get_llm_settings=lambda: {"extra_context": "context"},
            get_tool_selection=lambda _tool: "Custom Local Server(Gemma)",
            get_translation_result_dictionary_rules=lambda: [],
        )
        processor._inpainter_release_gate = {
            "required": True,
            "observed": False,
        }
        processor._start_gemma_prewarm = mock.Mock()
        processor._await_gemma_runtime = mock.Mock()
        processor._set_current_image = mock.Mock()
        processor.emit_progress = mock.Mock()
        processor._report_runtime_progress = mock.Mock()
        processor._emit_benchmark_event = mock.Mock()
        processor._persist_translation_state = mock.Mock()
        pages = [
            StagePageContext(
                image_path=f"page-{index}.png",
                image_name=f"page-{index}.png",
                source_lang="Japanese",
                target_lang="Korean",
                image=object(),
                blk_list=[SimpleNamespace(text=f"source-{index}", translation="")],
            )
            for index in range(2)
        ]

        class _AllHitTranslator:
            def __init__(self, *_args, **_kwargs) -> None:
                self.engine = SimpleNamespace(last_benchmark_stats={})
                self.translation_cache_status = "persistent-hit"
                self.uses_persistent_translation_memory = True

            @staticmethod
            def prepare_translation(_blocks, _extra_context) -> bool:
                return False

            @staticmethod
            def translate_with_cache_manager(
                blocks,
                _image,
                _extra_context,
                _cache_manager,
            ):
                for block in blocks:
                    block.translation = "cached"
                return blocks, "persistent-hit"

        processor.cache_manager = object()
        with mock.patch(
            "pipeline.stage_batched_processor.Translator",
            _AllHitTranslator,
        ), mock.patch(
            "pipeline.stage_batched_processor.apply_translation_result_dictionary",
        ):
            processor._translate_all(pages)

        processor._start_gemma_prewarm.assert_not_called()
        processor._await_gemma_runtime.assert_not_called()
        self.assertEqual(
            [page.blk_list[0].translation for page in pages],
            ["cached", "cached"],
        )

    def test_cache_state_change_cannot_bypass_unobserved_vram_release_gate(self) -> None:
        processor = self._processor(cancelled=False)
        processor.main_page.settings_page = SimpleNamespace(
            get_llm_settings=lambda: {"extra_context": "context"},
            get_tool_selection=lambda _tool: "Custom Local Server(Gemma)",
            get_translation_result_dictionary_rules=lambda: [],
        )
        processor._inpainter_release_gate = {
            "required": True,
            "observed": False,
        }
        processor.cache_manager = object()
        processor._start_gemma_prewarm = mock.Mock()
        processor._await_gemma_runtime = mock.Mock()
        processor._set_current_image = mock.Mock()
        processor.emit_progress = mock.Mock()
        processor._report_runtime_progress = mock.Mock()
        processor._emit_benchmark_event = mock.Mock()
        processor._persist_translation_state = mock.Mock()
        page = StagePageContext(
            image_path="page.png",
            image_name="page.png",
            source_lang="Japanese",
            target_lang="Korean",
            image=object(),
            blk_list=[SimpleNamespace(text="source", translation="")],
        )

        class _ChangingCacheTranslator:
            uses_persistent_translation_memory = True

            def __init__(self, *_args, **_kwargs) -> None:
                self.engine = SimpleNamespace(last_benchmark_stats={})
                self._prepare_results = iter((False, True))

            def prepare_translation(self, _blocks, _extra_context) -> bool:
                return next(self._prepare_results)

            @staticmethod
            def translate_with_cache_manager(*_args, **_kwargs):
                raise AssertionError("translation must not start before the VRAM gate")

        with mock.patch(
            "pipeline.stage_batched_processor.Translator",
            _ChangingCacheTranslator,
        ), self.assertRaisesRegex(RuntimeError, "VRAM release"):
            processor._translate_all([page])

        processor._start_gemma_prewarm.assert_not_called()
        processor._await_gemma_runtime.assert_not_called()


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest import mock

from modules.ocr.local_runtime import LocalOCRRuntimeManager
from modules.translation.local_runtime import LocalGemmaRuntimeManager
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


if __name__ == "__main__":
    unittest.main()

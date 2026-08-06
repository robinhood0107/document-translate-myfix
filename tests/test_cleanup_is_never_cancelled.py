"""런타임 정리는 어떤 상황에서도 취소되어서는 안 된다.

정리는 실행이 끝났거나 취소된 뒤에 돈다. `_shutdown_prewarm_executor`가 취소
이벤트를 설정한 **직후** 정리가 시작되므로, 일반 취소 검사기를 그대로 넘기면
컨테이너 정지가 자기 자신을 취소한다. 실제로 다음과 같이 끝났다.

    OperationCancelledError: Router operation cancelled before Router container stop.
    RouterReleaseError: Router terminal cleanup failed: ...

그러면 컨테이너와 GPU 메모리가 그대로 남는다. 정리 경로에는 취소가 없어야 한다.
"""

from __future__ import annotations

import os
import threading
import unittest
from types import SimpleNamespace
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from modules.ocr.local_runtime import LocalOCRRuntimeManager  # noqa: E402
from pipeline.stage_batched_processor import StageBatchedProcessor  # noqa: E402


class CleanupCancellationTests(unittest.TestCase):
    def _processor(self) -> StageBatchedProcessor:
        processor = object.__new__(StageBatchedProcessor)
        processor.main_page = SimpleNamespace(
            is_current_task_cancelled=lambda: True,
        )
        # 정리 시작 직전 상태를 그대로 재현한다.
        processor._prewarm_cancel_event = threading.Event()
        processor._prewarm_cancel_event.set()
        processor._record_runtime_transition = mock.Mock()
        processor._record_runtime_performance = mock.Mock()
        processor._sample_performance_resources = mock.Mock()
        processor._emit_benchmark_event = mock.Mock()
        processor._runtime_gpu_release_services = lambda: set()
        processor._runtime_gpu_baselines = lambda: {}
        processor._capture_runtime_gpu_start_baseline = mock.Mock()
        return processor

    def test_the_normal_checker_reports_cancelled_at_cleanup_time(self) -> None:
        """전제 확인: 정리 시점에는 일반 검사기가 이미 취소를 보고한다."""

        processor = self._processor()
        self.assertTrue(processor._prewarm_cancel_checker())

    def test_the_cleanup_checker_never_reports_cancelled(self) -> None:
        processor = self._processor()
        self.assertFalse(processor._cleanup_cancel_checker())

    def test_terminal_shutdown_passes_a_checker_that_never_cancels(self) -> None:
        processor = self._processor()
        manager = mock.Mock(spec=LocalOCRRuntimeManager)
        manager.shutdown = mock.Mock(
            return_value={"runtime_state": "stopped", "gpu_release_expected": False}
        )
        processor.main_page.local_ocr_runtime_manager = manager
        processor.main_page.local_translation_runtime_manager = None

        with mock.patch.object(
            StageBatchedProcessor, "_router_runtime_is_active", return_value=True
        ), mock.patch.object(
            StageBatchedProcessor,
            "_verify_managed_runtime_gpu_release",
            return_value={"required": False, "observed": True},
        ), mock.patch.object(
            StageBatchedProcessor, "_runtime_resource_arbiter"
        ) as arbiter, mock.patch(
            "pipeline.stage_batched_processor.query_cuda_handoff_metrics",
            return_value={},
        ):
            arbiter.return_value.model_release = mock.MagicMock()
            processor._shutdown_managed_runtimes(ocr_service="paddleocr_vl")

        manager.shutdown.assert_called_once()
        checker = manager.shutdown.call_args.kwargs["cancel_checker"]
        self.assertFalse(
            checker(),
            "종료 정리에 취소를 보고하는 검사기가 넘어갔습니다.",
        )


if __name__ == "__main__":
    unittest.main()

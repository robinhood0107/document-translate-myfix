"""OCR Router 컨테이너 기동은 검출 sweep 과 겹친다.

PR #242 는 OCR 예열을 검출 뒤로 미뤘다. 이유는 검출기의 ONNX 세션이 GPU 에 남아 있는
동안 **모델 적재** baseline 을 잡으면 그 값이 오염되고, 나중의 반환 검증이 어긋나기
때문이다. 그 결정은 유효하다.

다만 Router v2 는 `--models-max 1 --no-models-autoload` 로 뜬다. 컨테이너 기동 자체는
어떤 모델도 올리지 않으므로 baseline 과 무관하다. 그래서 그 부분만 앞으로 당긴다.
모델 적재는 예전 그대로 검출이 끝난 뒤다.
"""

from __future__ import annotations

import inspect
import os
import unittest
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from modules.ocr.local_runtime import LocalOCRRuntimeManager  # noqa: E402
from modules.utils.local_llama_router.contracts import (  # noqa: E402
    _ROUTER_PAIRS,
    expected_router_server_args,
)
from pipeline.stage_batched_processor import StageBatchedProcessor  # noqa: E402


class ContainerStartupOverlapsDetectionTests(unittest.TestCase):
    def test_the_container_prewarm_runs_before_the_detect_sweep(self) -> None:
        source = inspect.getsource(StageBatchedProcessor.batch_process)
        self.assertLess(
            source.index("_start_ocr_container_prewarm(policy)"),
            source.index("self._detect_all(pages, policy)"),
        )

    def test_the_model_load_still_waits_for_detection(self) -> None:
        """PR #242 의 결정은 그대로다. 모델 적재는 검출 뒤에 남는다."""

        source = inspect.getsource(StageBatchedProcessor.batch_process)
        self.assertLess(
            source.index("self._detect_all(pages, policy)"),
            source.index("_start_ocr_prewarm(policy)"),
        )

    def test_the_model_load_waits_for_the_container(self) -> None:
        """같은 코디네이터를 두 스레드가 동시에 만지면 안 된다."""

        source = inspect.getsource(StageBatchedProcessor.batch_process)
        self.assertLess(
            source.index("_await_ocr_container_prewarm()"),
            source.index("_start_ocr_prewarm(policy)"),
        )


class ContainerStartupLoadsNoModelTests(unittest.TestCase):
    """이 전제가 깨지면 검출과 겹치는 순간 baseline 이 오염된다."""

    def test_the_router_is_launched_without_model_autoload(self) -> None:
        for pair in _ROUTER_PAIRS:
            with self.subTest(pair=getattr(pair, "kind", pair)):
                args = expected_router_server_args(pair)
                self.assertIn("--no-models-autoload", args)
                self.assertIn("--models-max", args)

    def test_the_prepare_path_never_loads_an_alias(self) -> None:
        source = inspect.getsource(LocalOCRRuntimeManager.prepare_engine_container)
        self.assertIn("coordinator.prepare(", source)
        self.assertNotIn("coordinator.load(", source)


class ContainerPrewarmFailsOpenTests(unittest.TestCase):
    """사전 기동은 최적화일 뿐이다. 실패가 배치를 멈춰서는 안 된다."""

    def _processor(self, manager) -> StageBatchedProcessor:
        processor = object.__new__(StageBatchedProcessor)
        processor.main_page = SimpleNamespace(
            local_ocr_runtime_manager=manager,
            settings_page=object(),
        )
        processor._prewarm_cancel_checker = lambda: False
        processor._emit_benchmark_event = lambda *_a, **_k: None
        return processor

    def test_a_non_router_manager_is_a_no_op(self) -> None:
        processor = self._processor(object())
        processor._start_ocr_container_prewarm({"primary_ocr_engine": "PaddleOCR VL"})
        self.assertIsNone(getattr(processor, "_ocr_container_prewarm_job", None))

    def test_awaiting_without_a_job_is_a_no_op(self) -> None:
        processor = self._processor(object())
        processor._await_ocr_container_prewarm()

    def test_an_empty_engine_key_is_a_no_op(self) -> None:
        processor = self._processor(object())
        processor._start_ocr_container_prewarm({})
        self.assertIsNone(getattr(processor, "_ocr_container_prewarm_job", None))

    def test_a_failed_prepare_returns_false_instead_of_raising(self) -> None:
        manager = object.__new__(LocalOCRRuntimeManager)
        manager._lock = __import__("threading").RLock()
        manager._router_coordinator = SimpleNamespace(prepare=lambda *a, **k: None)
        manager._router_pair = None

        def explode(*_args, **_kwargs):
            raise RuntimeError("boom")

        manager.router_pair_for_engine = explode
        self.assertFalse(
            manager.prepare_engine_container("PaddleOCR VL", object())
        )

    def test_an_owned_pair_is_left_to_the_real_path(self) -> None:
        """이미 쌍을 소유했는데 여기서 전환하면 예열이 아니라 본 작업이 된다."""

        manager = object.__new__(LocalOCRRuntimeManager)
        manager._lock = __import__("threading").RLock()
        prepared: list[object] = []
        manager._router_coordinator = SimpleNamespace(
            prepare=lambda *a, **k: prepared.append(a)
        )
        manager._router_pair = object()
        manager.router_pair_for_engine = lambda *_a, **_k: object()

        self.assertFalse(
            manager.prepare_engine_container("PaddleOCR VL", object())
        )
        self.assertEqual(prepared, [])

    def test_the_teardown_cancels_a_queued_container_prewarm(self) -> None:
        source = inspect.getsource(
            StageBatchedProcessor._shutdown_prewarm_executor
        )
        self.assertIn("_ocr_container_prewarm_job", source)


if __name__ == "__main__":
    unittest.main()

"""Gemma 적재는 페이지 캐시 프리페치 뒤에 숨는다.

llama.cpp 는 GGUF 를 mmap 으로 읽으므로 적재 시간은 페이지 폴트가 디스크를 때리느냐
캐시에 맞느냐로 갈린다. 실측(13.58 GB, Docker Desktop WSL VM):

* 캐시 미적중 순차 읽기 7.99초 (1,825 MB/s)
* 캐시 적중 순차 읽기 0.72~0.88초 (약 20 GB/s)
* 실제 Gemma 적재는 첫 실행 43.95초, 이후 4.53~12.28초

적재가 순차 읽기보다 5배 넘게 느린 이유는 mmap 폴트의 접근 패턴이 순차가 아니기
때문이다. 그래서 순차 읽기로 캐시를 먼저 채운다. 이 작업은 디스크에서 RAM 으로만
옮기므로 OCR sweep 과 겹쳐도 VRAM 을 다투지 않는다.
"""

from __future__ import annotations

import inspect
import os
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from modules.translation.local_runtime import (  # noqa: E402
    GEMMA_MODEL_SIZE_PROBE_CONTAINER,
    GEMMA_PAGE_CACHE_PREFETCH_CONTAINER,
    LocalGemmaRuntimeManager,
    _available_page_cache_bytes,
    _volume_model_size_bytes,
)
from modules.utils.exceptions import OperationCancelledError  # noqa: E402
from pipeline.stage_batched_processor import StageBatchedProcessor  # noqa: E402


class PrefetchOverlapsTheOcrSweepTests(unittest.TestCase):
    def test_the_prefetch_starts_with_the_ocr_prewarm(self) -> None:
        """OCR sweep 71초 뒤에 숨어야 이득이 생긴다."""

        source = inspect.getsource(StageBatchedProcessor.batch_process)
        self.assertIn("_start_gemma_page_cache_prefetch()", source)
        self.assertLess(
            source.index("_start_ocr_prewarm(policy)"),
            source.index("_start_gemma_page_cache_prefetch()"),
        )
        self.assertLess(
            source.index("_start_gemma_page_cache_prefetch()"),
            source.index("self._ocr_all(pages, policy)"),
        )

    def test_the_load_waits_for_the_prefetch(self) -> None:
        """같은 파일을 프리페치와 mmap 이 동시에 읽으면 디스크를 두 배로 두드린다."""

        source = inspect.getsource(StageBatchedProcessor._translate_all)
        self.assertLess(
            source.index("_await_gemma_page_cache_prefetch()"),
            source.index("_start_gemma_prewarm()"),
        )

    def test_the_prefetch_never_blocks_runtime_commands(self) -> None:
        """런타임 명령 실행기는 max_workers=1 이다. 8초를 붙들면 안 된다."""

        source = inspect.getsource(
            StageBatchedProcessor._start_gemma_page_cache_prefetch
        )
        self.assertIn("_page_cache_executor", source)
        self.assertNotIn("_ensure_prewarm_executor", source)

    def test_the_prefetch_takes_no_gpu_lease(self) -> None:
        """디스크에서 RAM 으로만 옮긴다. arbiter 를 잡으면 OCR 과 다툰다."""

        source = inspect.getsource(
            StageBatchedProcessor._start_gemma_page_cache_prefetch
        )
        self.assertNotIn("arbiter", source)
        self.assertNotIn("model_start", source)


class PrefetchFailsOpenTests(unittest.TestCase):
    """프리페치는 최적화일 뿐이다. 실패가 배치를 멈춰서는 안 된다."""

    def _processor(self, manager) -> StageBatchedProcessor:
        processor = object.__new__(StageBatchedProcessor)
        processor.main_page = SimpleNamespace(
            local_translation_runtime_manager=manager,
            settings_page=object(),
        )
        processor._prewarm_cancel_checker = lambda: False
        processor._emit_benchmark_event = lambda *_a, **_k: None
        return processor

    def test_an_exception_is_swallowed(self) -> None:
        manager = mock.create_autospec(LocalGemmaRuntimeManager, instance=True)
        manager.prefetch_model_into_page_cache.side_effect = RuntimeError("boom")
        processor = self._processor(manager)

        processor._start_gemma_page_cache_prefetch()
        # 예외가 여기서 다시 나오면 배치가 멈춘다.
        processor._await_gemma_page_cache_prefetch()
        processor._shutdown_page_cache_executor()

    def test_a_missing_manager_is_a_no_op(self) -> None:
        processor = self._processor(object())
        processor._start_gemma_page_cache_prefetch()
        self.assertIsNone(getattr(processor, "_gemma_prefetch_job", None))

    def test_awaiting_without_a_job_is_a_no_op(self) -> None:
        processor = self._processor(object())
        processor._await_gemma_page_cache_prefetch()

    def test_teardown_survives_a_partially_built_processor(self) -> None:
        """정리는 어떤 상황에서도 실패하면 안 된다."""

        processor = object.__new__(StageBatchedProcessor)
        processor._shutdown_page_cache_executor()
        processor._shutdown_page_cache_executor()

    def test_the_next_batch_resets_the_prefetch_cancel_signal(self) -> None:
        processor = object.__new__(StageBatchedProcessor)
        processor._prewarm_cancel_event = threading.Event()
        processor._prewarm_cancel_event.set()
        processor._runtime_progress_lock = threading.RLock()
        processor._runtime_progress_started = {}

        processor._reset_prewarm_lifecycle()

        self.assertFalse(processor._prewarm_cancel_event.is_set())

    def test_a_completed_prefetch_can_be_awaited_and_torn_down(self) -> None:
        manager = mock.create_autospec(LocalGemmaRuntimeManager, instance=True)
        manager.prefetch_model_into_page_cache.return_value = {"performed": True}
        processor = self._processor(manager)

        processor._start_gemma_page_cache_prefetch()
        processor._await_gemma_page_cache_prefetch()
        processor._shutdown_page_cache_executor()

        manager.prefetch_model_into_page_cache.assert_called_once()
        self.assertIsNone(processor._page_cache_executor)

    def test_teardown_cancels_a_queued_prefetch(self) -> None:
        processor = object.__new__(StageBatchedProcessor)
        processor._page_cache_executor = ThreadPoolExecutor(max_workers=1)
        processor._gemma_prefetch_job = None
        processor._shutdown_page_cache_executor()
        self.assertIsNone(processor._page_cache_executor)

    def test_the_batch_finalizer_shuts_the_executor_down(self) -> None:
        source = inspect.getsource(StageBatchedProcessor._finalize_batch_run)
        self.assertIn(
            'run_cleanup_step("shutdown_page_cache_executor", '
            "self._shutdown_page_cache_executor)",
            source,
        )


class PageCacheTeardownTests(unittest.TestCase):
    def test_batch_finalization_waits_for_a_cancelled_running_prefetch(self) -> None:
        started = threading.Event()
        cancel_observed = threading.Event()
        allow_worker_exit = threading.Event()
        worker_finished = threading.Event()
        finalizer_finished = threading.Event()
        finalizer_errors: list[BaseException] = []

        def blocked_prefetch(_settings_page, *, cancel_checker=None):
            started.set()
            try:
                while not cancel_checker():
                    cancel_observed.wait(0.01)
                cancel_observed.set()
                allow_worker_exit.wait()
                return {"performed": False, "reason": "cancelled"}
            finally:
                worker_finished.set()

        manager = mock.create_autospec(LocalGemmaRuntimeManager, instance=True)
        manager.prefetch_model_into_page_cache.side_effect = blocked_prefetch
        processor = object.__new__(StageBatchedProcessor)
        processor.main_page = SimpleNamespace(
            local_translation_runtime_manager=manager,
            settings_page=object(),
            is_current_task_cancelled=lambda: False,
        )
        processor._prewarm_cancel_event = threading.Event()
        processor._page_cache_executor = None
        processor._gemma_prefetch_job = None
        processor._prewarm_executor = None
        processor._prewarm_jobs = {}
        processor._ocr_container_prewarm_job = None
        processor._render_executor = None
        processor._render_cancel_event = threading.Event()
        processor._pending_render_jobs = []
        processor._persist_stage_rates = mock.Mock()
        processor._write_run_report = mock.Mock()

        def finalize() -> None:
            try:
                processor._finalize_batch_run(
                    pages=[],
                    policy={},
                    batch_completed=False,
                    processing_status="cancelled",
                    processing_error=None,
                    output_summary={},
                    autosave_deferral_started=False,
                    runtime_cleanup_needed=False,
                )
            except BaseException as exc:
                finalizer_errors.append(exc)
            finally:
                finalizer_finished.set()

        finalizer_thread = None
        try:
            processor._start_gemma_page_cache_prefetch()
            self.assertTrue(started.wait(2))
            finalizer_thread = threading.Thread(target=finalize, daemon=True)
            finalizer_thread.start()

            self.assertTrue(cancel_observed.wait(2))
            self.assertFalse(finalizer_finished.wait(0.2))
        finally:
            processor._prewarm_cancel_event.set()
            allow_worker_exit.set()
            if finalizer_thread is not None:
                finalizer_thread.join(2)

        self.assertTrue(finalizer_finished.is_set())
        self.assertTrue(worker_finished.is_set())
        self.assertFalse(finalizer_errors)
        self.assertEqual(
            processor._write_run_report.call_args.kwargs["processing_status"],
            "cancelled",
        )


class GemmaNamedContainerCleanupTests(unittest.TestCase):
    def _manager(self) -> LocalGemmaRuntimeManager:
        manager = object.__new__(LocalGemmaRuntimeManager)
        manager._lock = threading.RLock()
        manager.router_credentials = lambda _settings: (
            "http://127.0.0.1:18080/v1",
            "gemma-4-26B-IQ4_NL.gguf",
        )
        manager._configured_volume_name = lambda: "comic-translate-gemma-models-v2"
        return manager

    def _run_with_fake_container(self, operation, *, error=None):
        live_containers: set[str] = set()
        created_containers: list[str] = []

        def fake_run_docker_command(command, **_kwargs):
            container_name = command[command.index("--name") + 1]
            live_containers.add(container_name)
            created_containers.append(container_name)
            if error is not None:
                raise error
            return SimpleNamespace(returncode=0, stdout="100", stderr="")

        with mock.patch(
            "modules.translation.local_runtime.run_docker_command",
            side_effect=fake_run_docker_command,
        ), mock.patch(
            "modules.utils.llama_cpp_runtime.remove_named_container",
            side_effect=live_containers.discard,
        ):
            if error is None:
                result = operation()
            else:
                with self.assertRaises(type(error)):
                    operation()
                result = None

        self.assertTrue(created_containers)
        self.assertFalse(live_containers)
        return result, created_containers

    def test_prefetch_removes_its_container_on_normal_error_and_cancel(self) -> None:
        manager = self._manager()
        outcomes = (
            None,
            RuntimeError("docker failed"),
            OperationCancelledError("cancelled"),
        )
        for error in outcomes:
            with self.subTest(error=type(error).__name__ if error else "success"):
                with mock.patch(
                    "modules.translation.local_runtime._available_page_cache_bytes",
                    return_value=1000,
                ), mock.patch(
                    "modules.translation.local_runtime._volume_model_size_bytes",
                    return_value=100,
                ):
                    result, created = self._run_with_fake_container(
                        lambda: manager.prefetch_model_into_page_cache(
                            object(),
                            cancel_checker=lambda: False,
                        ),
                        error=error,
                    )
                self.assertEqual(created, [GEMMA_PAGE_CACHE_PREFETCH_CONTAINER])
                if error is None:
                    self.assertTrue(result["performed"])

    def test_named_page_cache_probes_remove_their_containers_on_all_exits(
        self,
    ) -> None:
        probes = (
            (
                "memory",
                lambda: _available_page_cache_bytes(
                    cancel_checker=lambda: False
                ),
                f"{GEMMA_MODEL_SIZE_PROBE_CONTAINER}-mem",
            ),
            (
                "model size",
                lambda: _volume_model_size_bytes(
                    volume_name="comic-translate-gemma-models-v2",
                    model_name="gemma-4-26B-IQ4_NL.gguf",
                    image_ref="pinned-image",
                    cancel_checker=lambda: False,
                ),
                GEMMA_MODEL_SIZE_PROBE_CONTAINER,
            ),
        )
        outcomes = (
            None,
            RuntimeError("docker failed"),
            OperationCancelledError("cancelled"),
        )
        for probe_name, probe, expected_container in probes:
            for error in outcomes:
                with self.subTest(
                    probe=probe_name,
                    error=type(error).__name__ if error else "success",
                ):
                    result, created = self._run_with_fake_container(
                        probe,
                        error=error,
                    )
                    self.assertEqual(created, [expected_container])
                    if error is None:
                        self.assertGreater(result, 0)


class PrefetchRuntimeContractTests(unittest.TestCase):
    def test_the_container_names_are_fixed_and_distinct(self) -> None:
        """임의 이름 컨테이너가 생기면 안 되고, 두 용도가 서로를 지워도 안 된다."""

        for name in (
            GEMMA_PAGE_CACHE_PREFETCH_CONTAINER,
            GEMMA_MODEL_SIZE_PROBE_CONTAINER,
        ):
            with self.subTest(name=name):
                self.assertTrue(name.startswith("comic-translate-"))
        self.assertNotEqual(
            GEMMA_PAGE_CACHE_PREFETCH_CONTAINER,
            GEMMA_MODEL_SIZE_PROBE_CONTAINER,
        )

    def test_the_prefetch_reuses_the_pinned_image(self) -> None:
        """새 베이스 이미지를 들이지 않는다. 볼륨 프로브와 같은 기법을 쓴다."""

        source = inspect.getsource(
            LocalGemmaRuntimeManager.prefetch_model_into_page_cache
        )
        self.assertIn("DEFAULT_GEMMA_LLAMA_CPP_IMAGE", source)
        self.assertIn('"--pull",', source)
        self.assertIn('"never",', source)
        self.assertIn("readonly", source)
        self.assertIn("remove_named_container", source)

    def test_a_docker_failure_returns_instead_of_raising(self) -> None:
        manager = object.__new__(LocalGemmaRuntimeManager)
        manager._lock = __import__("threading").RLock()
        manager.router_credentials = staticmethod(
            lambda _page: ("http://127.0.0.1:18080/v1", "gemma-4-26B-IQ4_NL.gguf")
        )
        manager._configured_volume_name = lambda: "comic-translate-gemma-models-v2"

        with mock.patch(
            "modules.translation.local_runtime._available_page_cache_bytes",
            return_value=64 * 1024**3,
        ), mock.patch(
            "modules.translation.local_runtime._volume_model_size_bytes",
            return_value=14_585_439_872,
        ), mock.patch(
            "modules.translation.local_runtime.run_docker_command",
            return_value=SimpleNamespace(returncode=1, stdout="", stderr="nope"),
        ):
            result = manager.prefetch_model_into_page_cache(object())

        self.assertFalse(result["performed"])
        self.assertEqual(result["reason"], "docker-failed")

    def test_the_prefetch_is_skipped_when_memory_is_short(self) -> None:
        """여유보다 큰 파일을 읽으면 다른 단계가 쓰던 캐시를 밀어낸다."""

        manager = object.__new__(LocalGemmaRuntimeManager)
        manager._lock = __import__("threading").RLock()
        manager.router_credentials = staticmethod(
            lambda _page: ("http://127.0.0.1:18080/v1", "gemma-4-26B-IQ4_NL.gguf")
        )
        manager._configured_volume_name = lambda: "comic-translate-gemma-models-v2"

        with mock.patch(
            "modules.translation.local_runtime._available_page_cache_bytes",
            return_value=4 * 1024**3,
        ), mock.patch(
            "modules.translation.local_runtime._volume_model_size_bytes",
            return_value=14_585_439_872,
        ), mock.patch(
            "modules.translation.local_runtime.run_docker_command",
        ) as docker:
            result = manager.prefetch_model_into_page_cache(object())

        self.assertFalse(result["performed"])
        self.assertEqual(result["reason"], "insufficient-memory")
        docker.assert_not_called()


class DeadPrewarmHookIsGoneTests(unittest.TestCase):
    """순서가 번역 → 인페인팅으로 바뀌며 도달할 수 없게 된 훅."""

    def test_the_inpainter_release_no_longer_starts_gemma(self) -> None:
        source = inspect.getsource(
            StageBatchedProcessor._release_inpainter_before_render
        )
        self.assertNotIn("_start_gemma_prewarm", source)

    def test_the_dead_parameter_is_gone(self) -> None:
        signature = inspect.signature(
            StageBatchedProcessor._release_inpainter_before_render
        )
        self.assertNotIn("start_gemma", signature.parameters)


if __name__ == "__main__":
    unittest.main()

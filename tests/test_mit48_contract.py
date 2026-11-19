from __future__ import annotations

import copy
import hashlib
import io
import json
import stat
import struct
import subprocess
import threading
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import cv2
import numpy as np
import pytest

from modules.ocr import mit48_engine, mit48_worker
from modules.ocr.mit48_assets import (
    MIT48_AR_CHECKPOINT_SHA256,
    MIT48_AR_DICTIONARY_SHA256,
    MIT48_AR_MODEL_IDENTITY,
    MIT48_CTC_ARCHIVE_SHA256,
    MIT48_CTC_CHECKPOINT_SHA256,
    MIT48_CTC_DICTIONARY_SHA256,
    MIT48_CTC_MODEL_IDENTITY,
    MIT48_MODEL_SOURCE_SHA256,
    MIT48_RELEASE_TAG,
    MIT48_RUNTIME_IDENTITY,
    MIT48_UPSTREAM_COMMIT,
    MIT48_UPSTREAM_LICENSE_SHA256,
    MIT48_XPOS_LICENSE_COMMIT,
    MIT48_XPOS_LICENSE_SHA256,
)
from modules.ocr.mit48_engine import Mit48OCREngine
from modules.ocr.local_runtime import (
    LocalOCRRuntimeManager,
    NativeOCRWorkerReleaseError,
    _NativeOCRWorkerLease,
)
from modules.ocr.mit48_worker import _infer, _projected_line_geometry, _transform_line
from modules.ocr.factory import OCRFactory
from modules.ocr.processor import OCRProcessor
from pipeline.cache_manager import CacheManager
from scripts import windows_install_state
from modules.utils.exceptions import OperationCancelledError
from pipeline.runtime_resource_arbiter import (
    RuntimeLeaseConflictError,
    RuntimeModelState,
    RuntimeResourceArbiter,
)
from pipeline.stage_batched_processor import StageBatchedProcessor, StagePageContext
from modules.utils import download as download_module
from modules.utils.download import (
    CORE_APPLICATION_MODELS,
    MIT48_AR_APPLICATION_MODELS,
    MIT48_CTC_APPLICATION_MODELS,
    FULL_ONLY_APPLICATION_MODELS,
    OPTIONAL_APPLICATION_MODELS,
    ModelID,
    ModelDownloader,
    ModelSpec,
    application_model_profile,
)
from modules.utils.textblock import TextBlock
from modules.utils import windows_installation
from modules.utils.windows_installation import DEFAULT_OCR_APPLICATION_MODEL_IDS
from modules.ocr.mit48_assets import (
    MIT48_CTC_ARCHIVE,
    MIT48_CTC_CHECKPOINT,
    MIT48_CTC_DICTIONARY,
    MIT48_CTC_EXTRACT_DIRECTORY,
)


class _FakeStdin:
    def __init__(self, process: "_FakeProcess") -> None:
        self.process = process
        self.closed = False

    def write(self, value: bytes) -> int:
        if value == b'{"type":"shutdown"}\n' and self.process.graceful_exit:
            self.process.returncode = 0
        return len(value)

    def flush(self) -> None:
        return None

    def close(self) -> None:
        self.closed = True


class _FakeProcess:
    def __init__(
        self,
        events: list[str],
        *,
        output: bytes = b"",
        graceful_exit: bool = False,
    ) -> None:
        self.events = events
        self.pid = 12345
        self.returncode: int | None = None
        self.graceful_exit = graceful_exit
        self.stdin = _FakeStdin(self)
        self.stdout = io.BytesIO(output)
        self.stderr = io.BytesIO()

    def poll(self) -> int | None:
        return self.returncode

    def wait(self, timeout: float | None = None) -> int:
        self.events.append("wait")
        if self.returncode is None:
            raise subprocess.TimeoutExpired("mit48-worker", timeout)
        return self.returncode

    def terminate(self) -> None:
        self.events.append("terminate")
        self.returncode = -15

    def kill(self) -> None:
        self.events.append("kill")
        self.returncode = -9


class _RequestRaceStdin(_FakeStdin):
    def write(self, value: bytes) -> int:
        if value != b'{"type":"shutdown"}\n':
            self.process.request_received.set()
        return super().write(value)


class _RequestRaceStdout:
    def __init__(self, process: "_RequestRaceProcess", output: bytes) -> None:
        self.process = process
        self.lines = iter(output.splitlines(keepends=True))
        self.closed = False

    def readline(self, limit: int = -1) -> bytes:
        line = next(self.lines, b"")
        if line and not self.process.request_received.wait(1.0):
            return b""
        return line if limit < 0 else line[:limit]

    def close(self) -> None:
        self.closed = True


class _RequestRaceProcess(_FakeProcess):
    def __init__(self, events: list[str], output: bytes) -> None:
        super().__init__(events, graceful_exit=True)
        self.request_received = threading.Event()
        self.stdin = _RequestRaceStdin(self)
        self.stdout = _RequestRaceStdout(self, output)


class _StalledPipe:
    def __init__(self, release: threading.Event, *, block_write: bool = False) -> None:
        self.release = release
        self.block_write = block_write
        self.closed = False

    def write(self, value: bytes) -> int:
        if self.block_write:
            self.release.wait()
        return len(value)

    def flush(self) -> None:
        return None

    def readline(self, _limit: int = -1) -> bytes:
        self.release.wait()
        return b""

    def close(self) -> None:
        self.closed = True
        self.release.set()


class _StalledWorkerProcess:
    def __init__(self, *, block_request: bool) -> None:
        self.release = threading.Event()
        self.returncode: int | None = None
        self.stdin = _StalledPipe(self.release, block_write=block_request)
        self.stdout = _StalledPipe(self.release)
        self.stderr = _StalledPipe(self.release)

    def poll(self) -> int | None:
        return self.returncode

    def wait(self, timeout: float | None = None) -> int:
        if self.returncode is None:
            raise subprocess.TimeoutExpired("mit48-worker", timeout)
        return self.returncode

    def terminate(self) -> None:
        self.returncode = -15
        self.release.set()

    def kill(self) -> None:
        self.returncode = -9
        self.release.set()


class _UnreapedWorkerProcess(_StalledWorkerProcess):
    def __init__(self) -> None:
        super().__init__(block_request=False)
        self.pid = 12345
        self.events: list[str] = []
        self.wait_timeouts: list[float | None] = []

    def wait(self, timeout: float | None = None) -> int:
        self.events.append("wait")
        self.wait_timeouts.append(timeout)
        raise subprocess.TimeoutExpired("mit48-worker", timeout)

    def terminate(self) -> None:
        self.events.append("terminate")
        self.release.set()

    def kill(self) -> None:
        self.events.append("kill")
        self.release.set()


class _FakeLease:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.release_report = {"required": False, "observed": True}
        self.process: _FakeProcess | None = None

    def __enter__(self):
        self.events.append("lease_enter")
        return self

    def mark_worker_started(self, _pid: int) -> None:
        self.events.append("worker_started")

    def mark_model_loaded(self, _runtime: dict) -> None:
        self.events.append("model_loaded")

    def capture_pre_stop(self) -> None:
        self.events.append("capture_pre_stop")

    def mark_worker_reaped(self) -> None:
        self.events.append("worker_reaped")

    def __exit__(self, _exc_type, _exc, _traceback) -> bool:
        self.events.append("lease_exit")
        if self.process is not None:
            assert self.process.poll() is not None
            assert "worker_reaped" in self.events
        return False


@pytest.mark.parametrize("malformed_field", ["confidence", "region_bbox"])
def test_malformed_later_block_does_not_partially_update_earlier_blocks(
    monkeypatch,
    malformed_field: str,
) -> None:
    engine = Mit48OCREngine("ar", "cpu", "English")
    blocks = [
        TextBlock(
            text_bbox=np.asarray([1, 1, 8, 8], dtype=np.int32),
            text="old one",
            texts=["old one"],
            block_id="block-1",
        ),
        TextBlock(
            text_bbox=np.asarray([8, 8, 15, 15], dtype=np.int32),
            text="old two",
            texts=["old two"],
            block_id="block-2",
        ),
    ]
    response = {
        "type": "result",
        "runtime": {
            "identity": MIT48_RUNTIME_IDENTITY,
            "device": "cpu",
            "torch_version": "2.11.0",
            "worker_pid": 22222,
        },
        "worker_pid": 22222,
        "blocks": [
            {
                "block_id": "block-1",
                "text": "new one",
                "confidence": 0.9,
                "empty_reason": "",
                "regions": [
                    {
                        "line_index": 0,
                        "text": "new one",
                        "confidence": 0.9,
                        "bbox": [1, 1, 7, 7],
                        "geometry_source": "native_line_quad",
                        "empty_reason": "",
                    }
                ],
            },
            {
                "block_id": "block-2",
                "text": "new two",
                "confidence": "invalid" if malformed_field == "confidence" else 0.9,
                "empty_reason": "",
                "regions": (
                    [
                        {
                            "line_index": 0,
                            "text": "new two",
                            "confidence": 0.9,
                            "bbox": [8, 8, 8, 15],
                            "geometry_source": "bbox_projection_v1",
                            "empty_reason": "",
                        }
                    ]
                    if malformed_field == "region_bbox"
                    else []
                ),
            },
        ],
    }
    monkeypatch.setattr(engine, "_run_worker", lambda _image, _blocks: response)
    before = [
        {
            "block_id": block.block_id,
            "text": block.text,
            "texts": copy.deepcopy(block.texts),
            "ocr_regions": copy.deepcopy(block.ocr_regions),
            "ocr_geometry_provenance": copy.deepcopy(block.ocr_geometry_provenance),
            "processing_action": block.processing_action,
            "processing_decision_reasons": copy.deepcopy(
                block.processing_decision_reasons
            ),
            "xyxy": block.xyxy.copy(),
        }
        for block in blocks
    ]

    with pytest.raises(RuntimeError, match="invalid confidence|invalid region data"):
        engine.process_image(np.zeros((20, 20, 3), dtype=np.uint8), blocks)

    for block, snapshot in zip(blocks, before):
        assert block.block_id == snapshot["block_id"]
        assert block.text == snapshot["text"]
        assert block.texts == snapshot["texts"]
        assert block.ocr_regions == snapshot["ocr_regions"]
        assert block.ocr_geometry_provenance == snapshot["ocr_geometry_provenance"]
        assert block.processing_action == snapshot["processing_action"]
        assert block.processing_decision_reasons == snapshot[
            "processing_decision_reasons"
        ]
        assert np.array_equal(block.xyxy, snapshot["xyxy"])


def test_non_object_worker_response_is_a_typed_protocol_failure(monkeypatch) -> None:
    engine = Mit48OCREngine("ar", "cpu", "English")
    monkeypatch.setattr(engine, "_run_worker", lambda _image, _blocks: [])
    block = TextBlock(
        text_bbox=np.asarray([1, 1, 8, 8], dtype=np.int32),
        text="old",
        texts=["old"],
        block_id="block-1",
    )

    with pytest.raises(RuntimeError, match="unexpected response"):
        engine.process_image(np.zeros((20, 20, 3), dtype=np.uint8), [block])

    assert block.text == "old"
    assert block.texts == ["old"]


@pytest.mark.parametrize(
    ("engine_key", "variant"),
    [("MIT 48px AR", "ar"), ("MIT 48px CTC", "ctc")],
)
def test_ocr_factory_routes_mit48_modes_to_the_native_engine(
    monkeypatch,
    engine_key: str,
    variant: str,
) -> None:
    monkeypatch.setattr(
        mit48_engine.Mit48OCREngine,
        "initialize",
        lambda self, **_kwargs: None,
    )
    monkeypatch.setattr(
        "modules.ocr.factory.resolve_device",
        lambda _use_gpu, _backend: "cpu",
    )
    settings = SimpleNamespace(is_gpu_enabled=lambda: False)

    engine = OCRFactory._create_new_engine(
        settings,
        "English",
        engine_key,
    )

    assert isinstance(engine, Mit48OCREngine)
    assert engine.variant == variant


def test_native_worker_lease_keeps_cpu_noop_and_existing_router_path() -> None:
    manager = LocalOCRRuntimeManager()
    cpu_lease = manager.router_inference_lease(
        "MIT 48px AR",
        None,
        device="cpu",
    )
    with cpu_lease as state:
        assert state is None

    pair = SimpleNamespace(ocr_alias="paddle-alias")
    coordinator = SimpleNamespace(inference_lease=mock.Mock(return_value=object()))
    manager = LocalOCRRuntimeManager(router_coordinator=coordinator)
    manager._router_pair = pair
    manager.router_pair_for_engine = mock.Mock(return_value=pair)
    expected = coordinator.inference_lease.return_value
    result = manager.router_inference_lease(
        "PaddleOCR VL",
        object(),
        resource_arbiter=object(),
        device="cuda",
    )

    assert result is expected
    coordinator.inference_lease.assert_called_once_with(
        pair=pair,
        model_alias="paddle-alias",
    )


def test_legacy_ocr_processor_publishes_a_shared_native_gpu_arbiter() -> None:
    processor = OCRProcessor()
    processor.main_page = SimpleNamespace()

    arbiter = processor._runtime_resource_arbiter()

    assert processor.main_page.runtime_resource_arbiter is arbiter


def test_stage_and_legacy_ocr_use_the_same_main_page_arbiter() -> None:
    owner = SimpleNamespace(runtime_resource_arbiter=RuntimeResourceArbiter())
    legacy = OCRProcessor()
    legacy.main_page = owner
    stage = object.__new__(StageBatchedProcessor)
    stage.main_page = owner
    stage._runtime_resource_arbiter_instance = RuntimeResourceArbiter()

    assert stage._runtime_resource_arbiter() is legacy._runtime_resource_arbiter()


def _acquire_external_model_from_thread(
    arbiter: RuntimeResourceArbiter,
    service: str,
) -> BaseException | None:
    result: list[BaseException | None] = []
    completed = threading.Event()

    def acquire() -> None:
        try:
            arbiter.acquire_external_model(service)
            arbiter.release_external_model(service, release_succeeded=True)
            result.append(None)
        except BaseException as exc:
            result.append(exc)
        finally:
            completed.set()

    thread = threading.Thread(target=acquire, daemon=True)
    thread.start()
    assert completed.wait(1.0), "arbiter command lock remained held"
    thread.join()
    return result[0]


def test_failed_external_release_fences_same_service_reacquisition() -> None:
    arbiter = RuntimeResourceArbiter()
    arbiter.acquire_external_model("mit48_ar")
    arbiter.release_external_model("mit48_ar", release_succeeded=False)

    try:
        assert isinstance(
            _acquire_external_model_from_thread(arbiter, "mit48_ar"),
            RuntimeLeaseConflictError,
        )
        assert isinstance(
            _acquire_external_model_from_thread(arbiter, "mit48_ctc"),
            RuntimeLeaseConflictError,
        )
    finally:
        if arbiter.external_model_held_by_current_thread("mit48_ar"):
            arbiter.release_external_model(
                "mit48_ar",
                release_succeeded=True,
            )


def test_baseline_sampling_failure_releases_arbiter_command_lock(monkeypatch) -> None:
    arbiter = RuntimeResourceArbiter()
    monkeypatch.setattr(
        "modules.ocr.local_runtime.query_cuda_handoff_metrics",
        mock.Mock(side_effect=OSError("baseline sampling failed")),
    )
    lease = _NativeOCRWorkerLease(arbiter, "mit48_ar", "cuda")

    with pytest.raises(OSError, match="baseline sampling failed"):
        lease.__enter__()

    assert arbiter.snapshot().active_model is None
    assert _acquire_external_model_from_thread(arbiter, "mit48_ctc") is None


def _stage_native_ocr(monkeypatch, engine_key: str, cancel_checker=lambda: False):
    settings = SimpleNamespace(
        is_gpu_enabled=lambda: True,
        get_credentials=lambda _key: {},
        get_ocr_generic_settings=lambda: {},
        get_ocr_result_dictionary_rules=lambda: [],
    )
    monkeypatch.setattr(OCRFactory, "_engines", {})
    monkeypatch.setattr("modules.ocr.factory.resolve_device", lambda *_args: "cuda")
    monkeypatch.setattr("pipeline.stage_batched_processor.resolve_device", lambda *_args: "cuda")
    monkeypatch.setattr(Mit48OCREngine, "initialize", lambda self, **_kwargs: None)
    monkeypatch.setattr("pipeline.stage_batched_processor.summarize_ocr_quality", lambda _blocks: {"low_quality": False})
    monkeypatch.setattr("pipeline.stage_batched_processor.drop_embedded_ui_ocr_blocks", lambda blocks, _shape: (blocks, []))
    transitions = []
    arbiter = RuntimeResourceArbiter(
        transition_callback=lambda service, state, _old, _outcome: transitions.append((service, state.value))
    )
    manager = LocalOCRRuntimeManager()
    manager.router_inference_lease = mock.Mock(wraps=manager.router_inference_lease)
    processor = object.__new__(StageBatchedProcessor)
    processor._runtime_resource_arbiter_instance = arbiter
    processor.main_page = SimpleNamespace(
        settings_page=settings,
        local_ocr_runtime_manager=manager,
        runtime_resource_arbiter=arbiter,
        is_current_task_cancelled=cancel_checker,
        lang_mapping={"Japanese": "Japanese"},
    )
    processor._shutdown_runtime_with_retry = mock.Mock()
    processor.cache_manager = SimpleNamespace(
        _get_ocr_cache_key=mock.Mock(return_value="example-cache"),
        _can_serve_all_blocks_from_ocr_cache=mock.Mock(return_value=False),
        _cache_ocr_results=mock.Mock(),
    )
    processor._paddleocr_cache_store = None
    processor._paddleocr_cache_identity = None
    cached = OCRFactory.create_engine(settings, "Japanese", engine_key)
    assert processor._runtime_resource_arbiter() is arbiter
    assert cached.device == "cuda"
    block = TextBlock(
        text_bbox=np.array([1, 1, 8, 8], dtype=np.int32),
        text_class="text_bubble", block_id="example-block",
    )
    page = StagePageContext(
        image_path="example-page", image_name="example-page", source_lang="Japanese",
        target_lang="Korean", image=np.zeros((20, 20, 3), dtype=np.uint8), blk_list=[block],
    )
    policy = {"primary_ocr_engine": engine_key, "normalized_ocr_mode": engine_key}
    return processor, arbiter, transitions, cached, page, policy


def _successful_native_worker(events: list[str]) -> _FakeProcess:
    runtime = {
        "identity": MIT48_RUNTIME_IDENTITY, "device": "cuda",
        "torch_version": "2.11.0", "worker_pid": 12345,
    }
    messages = [
        {"type": "loaded", "runtime": runtime, "worker_pid": 12345},
        {"type": "result", "runtime": runtime, "worker_pid": 12345, "blocks": [{
            "block_id": "example-block", "text": "example", "confidence": 0.9,
            "empty_reason": "", "regions": [{
                "line_index": 0, "text": "example", "confidence": 0.9,
                "bbox": [1, 1, 8, 8], "geometry_source": "native_line_quad", "empty_reason": "",
            }],
        }]},
    ]
    output = b"".join(json.dumps(message).encode("utf-8") + b"\n" for message in messages)
    return _FakeProcess(events, output=output, graceful_exit=True)


@pytest.mark.parametrize("engine_key", ["MIT 48px AR", "MIT 48px CTC"])
@pytest.mark.parametrize("release_observed", [True, False])
def test_stage_factory_worker_uses_shared_arbiter_and_reaps_before_release(
    monkeypatch, engine_key: str, release_observed: bool,
) -> None:
    processor, arbiter, transitions, cached, page, policy = _stage_native_ocr(monkeypatch, engine_key)
    events: list[str] = []
    process = _successful_native_worker(events)
    service = "mit48_ar" if engine_key.endswith("AR") else "mit48_ctc"

    def popen(*_args, **_kwargs):
        lease_manager = processor.main_page.local_ocr_runtime_manager
        assert lease_manager.router_inference_lease.call_args.kwargs[
            "resource_arbiter"
        ] is arbiter
        assert arbiter.snapshot().active_model == service
        assert arbiter.snapshot().states[service] == "model_loading"
        return process

    def verify_release(_before, **_kwargs):
        assert process.poll() == 0
        assert (service, "model_ready") in transitions
        return {"required": True, "observed": release_observed}

    monkeypatch.setattr(mit48_engine.subprocess, "Popen", popen)
    monkeypatch.setattr("modules.ocr.local_runtime.query_cuda_handoff_metrics", lambda: {"sample": "neutral"})
    monkeypatch.setattr("modules.ocr.local_runtime.wait_for_global_vram_release", verify_release)
    if release_observed:
        result = processor._run_primary_ocr(page, policy)
        assert result["engine_name"] == "Mit48OCREngine"
        assert page.blk_list[0].text == "example"
        assert arbiter.snapshot().active_model is None
        assert arbiter.snapshot().states[service] == "stopped"
    else:
        with pytest.raises(RuntimeError, match="release was not confirmed"):
            processor._run_primary_ocr(page, policy)
        assert arbiter.snapshot().active_model == service
        assert arbiter.snapshot().states[service] == "release_failed"
        with pytest.raises(RuntimeLeaseConflictError):
            arbiter.acquire_external_model("gemma")
    assert process.poll() == 0
    assert "wait" in events
    assert cached.inference_lease_factory is None
    assert cached.cancel_checker is None


def test_stage_native_worker_failure_before_process_start_releases_arbiter(monkeypatch) -> None:
    processor, arbiter, _transitions, cached, page, policy = _stage_native_ocr(monkeypatch, "MIT 48px AR")
    monkeypatch.setattr("modules.ocr.local_runtime.query_cuda_handoff_metrics", lambda: {"sample": "baseline"})
    monkeypatch.setattr(mit48_engine.subprocess, "Popen", mock.Mock(side_effect=OSError("start failed")))
    with pytest.raises(OSError, match="start failed"):
        processor._run_primary_ocr(page, policy)
    assert arbiter.snapshot().active_model is None
    assert arbiter.snapshot().states["mit48_ar"] == "stopped"
    assert cached.inference_lease_factory is None


def test_stage_native_worker_cancellation_reaps_and_releases(monkeypatch) -> None:
    checks = iter([False, True])
    processor, arbiter, _transitions, cached, page, policy = _stage_native_ocr(
        monkeypatch, "MIT 48px AR", cancel_checker=lambda: next(checks, True),
    )
    events: list[str] = []
    process = _FakeProcess(events, graceful_exit=True)
    monkeypatch.setattr(mit48_engine.subprocess, "Popen", lambda *_args, **_kwargs: process)
    monkeypatch.setattr("modules.ocr.local_runtime.query_cuda_handoff_metrics", lambda: {"sample": "neutral"})
    monkeypatch.setattr("modules.ocr.local_runtime.wait_for_global_vram_release", lambda *_a, **_k: {"observed": True})
    with pytest.raises(OperationCancelledError):
        processor._run_primary_ocr(page, policy)
    assert process.poll() is not None
    assert arbiter.snapshot().active_model is None
    assert arbiter.snapshot().states["mit48_ar"] == "stopped"
    assert cached.cancel_checker is None


@pytest.mark.parametrize("engine_key", ["MIT 48px AR", "MIT 48px CTC"])
def test_native_ocr_skips_empty_docker_prewarm_and_does_not_claim_arbiter(engine_key: str) -> None:
    arbiter = RuntimeResourceArbiter()
    manager = LocalOCRRuntimeManager()
    processor = object.__new__(StageBatchedProcessor)
    processor.main_page = SimpleNamespace(
        settings_page=object(), local_ocr_runtime_manager=manager,
        runtime_resource_arbiter=arbiter, is_current_task_cancelled=lambda: False,
    )
    processor._start_prewarm = mock.Mock()
    processor._await_prewarm_or_run = mock.Mock()
    processor._ensure_prewarm_executor = mock.Mock()
    policy = {"primary_ocr_engine": engine_key}
    processor._start_ocr_container_prewarm(policy)
    processor._start_ocr_prewarm(policy)
    processor._await_ocr_runtime(policy)
    processor._start_prewarm.assert_not_called()
    processor._await_prewarm_or_run.assert_not_called()
    processor._ensure_prewarm_executor.assert_not_called()
    assert arbiter.snapshot().active_model is None


def test_native_ocr_initialization_failure_does_not_leave_phantom_lease(monkeypatch) -> None:
    processor, arbiter, _transitions, _cached, page, policy = _stage_native_ocr(monkeypatch, "MIT 48px AR")
    monkeypatch.setattr(OCRFactory, "_engines", {})
    monkeypatch.setattr(Mit48OCREngine, "initialize", mock.Mock(side_effect=RuntimeError("not prepared")))
    with pytest.raises(RuntimeError, match="not prepared"):
        processor._run_primary_ocr(page, policy)
    assert arbiter.snapshot().active_model is None


def test_legacy_native_processor_uses_execution_copy_for_lease(monkeypatch) -> None:
    stage, arbiter, _transitions, cached, page, _policy = _stage_native_ocr(monkeypatch, "MIT 48px AR")
    monkeypatch.setattr("modules.ocr.local_runtime.query_cuda_handoff_metrics", lambda: {"sample": "baseline"})

    def process(self, _image, blocks):
        with self._inference_lease():
            assert arbiter.snapshot().active_model == "mit48_ar"
        return blocks

    monkeypatch.setattr(Mit48OCREngine, "process_image", process)
    processor = OCRProcessor()
    processor.main_page = stage.main_page
    processor.settings = stage.main_page.settings_page
    processor.source_lang_english = "Japanese"
    processor.ocr_key = "MIT 48px AR"
    processor.ocr_mode = "MIT 48px AR"
    processor.last_device = "cuda"
    processor.process(page.image, page.blk_list)
    assert arbiter.snapshot().active_model is None
    assert cached.inference_lease_factory is None
    assert cached.cancel_checker is None


def test_batch_ocr_propagates_native_release_failure_instead_of_continuing(monkeypatch) -> None:
    processor, arbiter, _transitions, _cached, page, policy = _stage_native_ocr(monkeypatch, "MIT 48px AR")
    with arbiter.model_start(arbiter.token("mit48_ar")):
        pass
    with pytest.raises(RuntimeError, match="unconfirmed"):
        with arbiter.model_release("mit48_ar"):
            raise RuntimeError("unconfirmed")
    processor._project_checkpoint_store = None
    processor._canonicalize_ocr_inputs = mock.Mock()
    processor._set_current_image = mock.Mock()
    processor.emit_progress = mock.Mock()
    processor._emit_benchmark_event = mock.Mock()
    processor._ensure_source_image = mock.Mock(return_value=page.image)
    processor._release_source_image = mock.Mock()
    processor._mark_page_failed = mock.Mock()
    processor._run_primary_ocr = mock.Mock(
        side_effect=NativeOCRWorkerReleaseError(
            "worker GPU release was not confirmed"
        )
    )
    with pytest.raises(RuntimeError, match="release was not confirmed"):
        processor._ocr_all([page], policy)
    processor._mark_page_failed.assert_not_called()
    processor._release_source_image.assert_called_once_with(page)
    processor._shutdown_runtime_with_retry.assert_not_called()
    assert arbiter.snapshot().active_model == "mit48_ar"
    assert arbiter.snapshot().states["mit48_ar"] == "release_failed"


def test_worker_cleanup_finishes_before_gpu_lease_exit(monkeypatch) -> None:
    events: list[str] = []
    lease = _FakeLease(events)
    engine = Mit48OCREngine("ar", "cpu", "English")
    engine.set_inference_lease_factory(lambda: lease)
    engine.set_cancel_checker(lambda: True)
    process = _FakeProcess(events)
    lease.process = process
    monkeypatch.setattr(mit48_engine.subprocess, "Popen", lambda *_a, **_k: process)

    with pytest.raises(OperationCancelledError):
        engine._run_worker(np.zeros((8, 8, 3), dtype=np.uint8), [])

    assert events.index("capture_pre_stop") < events.index("terminate")
    assert events.index("terminate") < events.index("worker_reaped")
    assert events.index("worker_reaped") < events.index("lease_exit")


def test_worker_protocol_failure_reaps_child_inside_gpu_lease(monkeypatch) -> None:
    events: list[str] = []
    lease = _FakeLease(events)
    engine = Mit48OCREngine("ar", "cpu", "English")
    engine.set_inference_lease_factory(lambda: lease)
    process = _FakeProcess(events, output=b"[]\n")
    lease.process = process
    monkeypatch.setattr(mit48_engine.subprocess, "Popen", lambda *_a, **_k: process)

    with pytest.raises(RuntimeError, match="non-object message"):
        engine._run_worker(np.zeros((8, 8, 3), dtype=np.uint8), [])

    assert events.index("worker_reaped") < events.index("lease_exit")


def test_worker_start_failure_exits_lease_without_child(monkeypatch) -> None:
    events: list[str] = []
    lease = _FakeLease(events)
    engine = Mit48OCREngine("ar", "cpu", "English")
    engine.set_inference_lease_factory(lambda: lease)

    def fail_start(*_args, **_kwargs):
        raise OSError("worker could not start")

    monkeypatch.setattr(mit48_engine.subprocess, "Popen", fail_start)
    with pytest.raises(OSError, match="could not start"):
        engine._run_worker(np.zeros((8, 8, 3), dtype=np.uint8), [])
    assert events == ["lease_enter", "lease_exit"]


@pytest.mark.parametrize("observed", [True, False])
def test_native_gpu_lease_uses_spawn_baseline_and_live_child_pre_stop(
    monkeypatch,
    observed: bool,
) -> None:
    baseline = {"sample": "before-spawn"}
    pre_stop = {"sample": "child-loaded-and-alive"}
    metrics = iter((baseline, pre_stop))
    calls: list[dict] = []
    monkeypatch.setattr(
        "modules.ocr.local_runtime.query_cuda_handoff_metrics",
        lambda: next(metrics),
    )

    def verify_release(before, **kwargs):
        calls.append({"before": before, **kwargs})
        return {"required": True, "observed": observed, "status": "observed" if observed else "timeout"}

    monkeypatch.setattr(
        "modules.ocr.local_runtime.wait_for_global_vram_release",
        verify_release,
    )
    arbiter = RuntimeResourceArbiter()
    lease = _NativeOCRWorkerLease(arbiter, "mit48_ar", "cuda")

    if observed:
        with lease:
            lease.mark_worker_started(12345)
            lease.mark_model_loaded({"device": "cuda"})
            lease.capture_pre_stop()
            lease.mark_worker_reaped()
        assert arbiter.snapshot().active_model is None
        assert arbiter.snapshot().states["mit48_ar"] == RuntimeModelState.STOPPED.value
    else:
        with pytest.raises(RuntimeError, match="release was not confirmed"):
            with lease:
                lease.mark_worker_started(12345)
                lease.mark_model_loaded({"device": "cuda"})
                lease.capture_pre_stop()
                lease.mark_worker_reaped()
        assert arbiter.snapshot().active_model == "mit48_ar"
        assert arbiter.snapshot().states["mit48_ar"] == RuntimeModelState.RELEASE_FAILED.value
        with pytest.raises(RuntimeLeaseConflictError, match="lease is held"):
            arbiter.acquire_external_model("gemma")

    assert calls == [
        {
            "before": pre_stop,
            "gpu_release_expected": True,
            "driver_baseline": baseline,
            "min_drop_mb": 0.0,
            "expected_drop_ratio": 0.9,
        }
    ]
    assert lease.release_report["observed"] is observed


def test_native_gpu_lease_preserves_owner_when_pre_stop_metrics_are_missing(
    monkeypatch,
) -> None:
    baseline = {"sample": "before-spawn"}
    samples = iter((baseline, RuntimeError("pre-stop NVML sample unavailable")))
    wait_called = False

    def query_metrics():
        value = next(samples)
        if isinstance(value, Exception):
            raise value
        return value

    def unexpected_wait(*_args, **_kwargs):
        nonlocal wait_called
        wait_called = True
        return {"required": True, "observed": True}

    monkeypatch.setattr(
        "modules.ocr.local_runtime.query_cuda_handoff_metrics",
        query_metrics,
    )
    monkeypatch.setattr(
        "modules.ocr.local_runtime.wait_for_global_vram_release",
        unexpected_wait,
    )
    arbiter = RuntimeResourceArbiter()
    lease = _NativeOCRWorkerLease(arbiter, "mit48_ctc", "cuda")

    with pytest.raises(RuntimeError, match="release was not confirmed"):
        with lease:
            lease.mark_worker_started(23456)
            lease.mark_model_loaded({"device": "cuda"})
            lease.capture_pre_stop()
            lease.mark_worker_reaped()

    assert not wait_called
    assert lease.release_report["status"] == "measurement_unavailable"
    assert lease.release_report["model_loaded"] is True
    assert arbiter.snapshot().active_model == "mit48_ctc"
    assert arbiter.snapshot().states["mit48_ctc"] == RuntimeModelState.RELEASE_FAILED.value
    with pytest.raises(RuntimeLeaseConflictError, match="lease is held"):
        arbiter.acquire_external_model("gemma")


@pytest.mark.parametrize("gray", [180, 210, 230])
def test_projection_splits_two_gray_horizontal_rows_without_dropping_thin_marks(
    gray: int,
) -> None:
    image = np.full((100, 240, 3), 255, dtype=np.uint8)
    color = (gray, gray, gray)
    cv2.putText(image, "Ii-7.", (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, 1, cv2.LINE_8)
    cv2.putText(image, "i-7.I", (12, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, 1, cv2.LINE_8)

    polygons, vertical = _projected_line_geometry(
        image,
        [0, 0, 240, 100],
        vertical=False,
        source_lang="en",
    )

    assert not vertical
    assert len(polygons) == 2
    for y, x in zip(*np.where(image[:, :, 0] == gray)):
        assert any(
            polygon[0][0] <= x <= polygon[1][0]
            and polygon[0][1] <= y <= polygon[2][1]
            for polygon in polygons
        )


def test_projection_identifies_two_vertical_columns_on_dark_background() -> None:
    image = np.full((150, 100, 3), 35, dtype=np.uint8)
    for x in (22, 62):
        for y, text in zip((28, 62, 96, 130), ("I", "I", "7", "7")):
            cv2.putText(
                image,
                text,
                (x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (230, 230, 230),
                2,
                cv2.LINE_8,
            )

    polygons, vertical = _projected_line_geometry(
        image,
        [8, 5, 94, 145],
        vertical=None,
        source_lang="ja",
    )

    assert vertical
    assert len(polygons) == 2
    assert sum(point[0] for point in polygons[0]) > sum(
        point[0] for point in polygons[1]
    )


def test_directionless_tall_english_block_stays_horizontal() -> None:
    image = np.full((140, 80, 3), 255, dtype=np.uint8)
    cv2.putText(
        image,
        "TALL",
        (5, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        (20, 20, 20),
        2,
        cv2.LINE_8,
    )

    polygons, vertical = _projected_line_geometry(
        image,
        [0, 0, 80, 140],
        vertical=None,
        source_lang="en",
    )

    assert not vertical
    assert len(polygons) == 1


def test_textless_panel_line_is_marked_for_geometry_review() -> None:
    image = np.full((80, 80, 3), 255, dtype=np.uint8)
    cv2.rectangle(image, (8, 8), (71, 71), (0, 0, 0), 1)
    response = _infer(
        {
            "variant": "ar",
            "device": "cpu",
            "source_language": "en",
            "image": image,
            "blocks": [
                {
                    "block_id": "panel-edge",
                    "xyxy": [0, 0, 80, 80],
                    "lines": None,
                    "segment_points": None,
                    "direction": "horizontal",
                }
            ],
        },
        model=None,
        torch=SimpleNamespace(__version__="no-model"),
    )

    assert response["blocks"][0]["empty_reason"] == "geometry_review"
    assert response["blocks"][0]["regions"] == []


def test_mit48_assets_are_full_only_and_not_required_by_default_ocr() -> None:
    mit48_ids = {
        ModelID.MIT48_AR,
        ModelID.MIT48_CTC_ARCHIVE,
        ModelID.MIT48_CTC,
        ModelID.MIT48_SOURCE,
    }
    assert not mit48_ids.intersection(OPTIONAL_APPLICATION_MODELS)
    assert mit48_ids.issubset(FULL_ONLY_APPLICATION_MODELS)
    assert not mit48_ids.intersection(application_model_profile("core"))
    assert mit48_ids.issubset(application_model_profile("full"))
    assert not mit48_ids.intersection(DEFAULT_OCR_APPLICATION_MODEL_IDS)


def test_full_install_seal_reuses_registry_digest_without_hashing_model_body(
    monkeypatch,
    tmp_path,
) -> None:
    model_dir = tmp_path / "models" / "ocr" / "mit48" / "ar"
    model_dir.mkdir(parents=True)
    model_file = model_dir / "small-test-model.bin"
    model_file.write_bytes(b"model bytes are not rehashed here")
    expected_digest = "a" * 64
    monkeypatch.setattr(windows_install_state, "ROOT", tmp_path)
    monkeypatch.setattr(
        windows_install_state,
        "application_model_profile",
        lambda _profile: (ModelID.MIT48_AR,),
    )

    def fail_hash(_path):
        raise AssertionError("model was rehashed")

    monkeypatch.setattr(
        windows_install_state,
        "_sha256_file",
        fail_hash,
    )
    monkeypatch.setitem(
        ModelDownloader.registry,
        ModelID.MIT48_AR,
        ModelSpec(
            id=ModelID.MIT48_AR,
            url="",
            files=[model_file.name],
            sha256=[expected_digest],
            save_dir=str(model_dir),
        ),
    )

    records = windows_install_state._model_records("full")

    assert records[0]["files"][0]["digest"] == expected_digest
    assert records[0]["files"][0]["bytes"] == model_file.stat().st_size


@pytest.mark.parametrize(
    (
        "previous_tier",
        "previous_ids",
        "requested_profile",
        "requested_tier",
        "expected_ids",
        "expected_additions",
        "expected_tier",
    ),
    [
        (
            "full",
            frozenset(application_model_profile("full"))
            - frozenset(FULL_ONLY_APPLICATION_MODELS),
            "core",
            "core",
            frozenset(application_model_profile("full"))
            - frozenset(FULL_ONLY_APPLICATION_MODELS),
            frozenset(),
            "full",
        ),
        (
            "full",
            frozenset(application_model_profile("full")),
            "core",
            "core",
            frozenset(application_model_profile("full")),
            frozenset(),
            "full",
        ),
        (
            "full",
            frozenset(application_model_profile("full"))
            - frozenset(FULL_ONLY_APPLICATION_MODELS),
            "full",
            "full",
            frozenset(application_model_profile("full")),
            frozenset(FULL_ONLY_APPLICATION_MODELS),
            "full",
        ),
        (
            None,
            frozenset(),
            "core",
            "core",
            frozenset(application_model_profile("core")),
            frozenset(application_model_profile("core")),
            "core",
        ),
    ],
    ids=("legacy-full18-core", "full22-core", "explicit-full22", "fresh-core"),
)
def test_setup_preserves_existing_full_seal_and_adds_only_requested_models(
    monkeypatch,
    tmp_path,
    previous_tier,
    previous_ids,
    requested_profile,
    requested_tier,
    expected_ids,
    expected_additions,
    expected_tier,
) -> None:
    state_dir = tmp_path / "install-state"
    state_dir.mkdir()
    monkeypatch.setattr(windows_install_state, "STATE_DIR", state_dir)
    previous_state = None
    if previous_tier is not None:
        previous_state = {
            "schema_version": 1,
            "runtime": "cuda13",
            "provisioned_tier": previous_tier,
            "application_models": [
                {"id": model_id.value, "files": [{}]}
                for model_id in ModelID
                if model_id in previous_ids
            ],
        }
        windows_install_state._state_path("cuda13").write_text(
            json.dumps(previous_state),
            encoding="utf-8",
        )

    provision = mock.Mock()
    with (
        mock.patch.object(
            windows_install_state,
            "_validate_application_models",
        ),
        mock.patch.object(windows_install_state, "provision_model_ids", provision),
    ):
        windows_install_state.command_provision(
            SimpleNamespace(
                runtime="cuda13",
                profile=requested_profile,
                requirements="requirements-cuda13.txt",
            )
        )

    if expected_additions:
        provision.assert_called_once()
        assert set(provision.call_args.args[0]) == set(expected_additions)
    else:
        provision.assert_not_called()

    image_ref = "ghcr.io/ggml-org/llama.cpp:server-cuda13"
    managed_path = tmp_path / "managed-runtime-state.json"
    managed_runtimes = [
        {
            "name": f"runtime-{index}",
            "runtime_name": runtime_name,
            "preparation_version": 1,
        }
        for index, runtime_name in enumerate(
            sorted(
                windows_install_state.FULL_MANAGED_RUNTIMES
                if expected_tier == "full"
                else windows_install_state.CORE_MANAGED_RUNTIMES
            )
        )
    ]
    managed_path.write_text(
        json.dumps(
            {
                "image_ref": image_ref,
                "image_id": "sha256:test-image",
                "volumes": managed_runtimes,
            }
        ),
        encoding="utf-8",
    )
    recorded_ids: set[ModelID] = set()

    def model_records(_profile, *, model_ids=None):
        selected_ids = set(model_ids or ())
        recorded_ids.update(selected_ids)
        return [
            {"id": model_id.value, "files": [{}]}
            for model_id in ModelID
            if model_id in selected_ids
        ]

    with (
        mock.patch.object(
            windows_install_state,
            "_assert_runtime_image",
        ),
        mock.patch.object(
            windows_install_state,
            "_requirements_record",
            return_value={"path": "requirements-cuda13.txt", "sha256": "a" * 64},
        ),
        mock.patch.object(windows_install_state, "_validate_application_models"),
        mock.patch.object(windows_install_state, "_model_records", side_effect=model_records),
    ):
        windows_install_state.command_write(
            SimpleNamespace(
                runtime="cuda13",
                tier=requested_tier,
                profile=requested_profile,
                requirements="requirements-cuda13.txt",
                image_ref=image_ref,
                image_id="sha256:test-image",
                required_cuda="13.1",
                managed_state=str(managed_path),
            )
        )

    sealed_state = json.loads(windows_install_state._state_path("cuda13").read_text())
    sealed_ids = {ModelID(record["id"]) for record in sealed_state["application_models"]}
    assert sealed_ids == set(expected_ids)
    assert recorded_ids == set(expected_ids)
    assert sealed_state["provisioned_tier"] == expected_tier


def test_preflight_rejects_a_missing_model_in_the_declared_seal(
    monkeypatch,
    tmp_path,
) -> None:
    missing_id = application_model_profile("core")[0]
    model_records = []
    for model_id in application_model_profile("core"):
        relative = Path("models") / model_id.value / "model.bin"
        model_path = tmp_path / relative
        if model_id != missing_id:
            model_path.parent.mkdir(parents=True)
            model_path.write_bytes(b"x")
            stat = model_path.stat()
            size = stat.st_size
            mtime_ns = stat.st_mtime_ns
        else:
            size = 1
            mtime_ns = 1
        model_records.append(
            {
                "id": model_id.value,
                "files": [
                    {
                        "path": relative.as_posix(),
                        "bytes": size,
                        "mtime_ns": mtime_ns,
                    }
                ],
            }
        )
    payload = {
        "runtime": "cuda13",
        "provisioned_tier": "core",
        "requirements": {"path": "requirements-cuda13.txt", "sha256": "a" * 64},
        "application_models": model_records,
        "llama_image": {"ref": "ghcr.io/ggml-org/llama.cpp:server-cuda13"},
    }
    monkeypatch.setattr(windows_install_state, "ROOT", tmp_path)
    args = SimpleNamespace(
        runtime="cuda13",
        requirements="requirements-cuda13.txt",
        emit_cmd=False,
    )

    with (
        mock.patch.object(windows_install_state, "_read_state", return_value=payload),
        mock.patch.object(
            windows_install_state,
            "_requirements_record",
            return_value=payload["requirements"],
        ),
        mock.patch.object(windows_install_state, "_validate_docker_state") as validate_docker,
        pytest.raises(windows_install_state.InstallStateError, match="Application model is missing"),
    ):
        windows_install_state.command_preflight(args)

    validate_docker.assert_not_called()


def test_full_preflight_points_a_corrupt_mit48_asset_to_setup_full(
    monkeypatch,
) -> None:
    model_ids = (
        set(application_model_profile("full"))
        - set(FULL_ONLY_APPLICATION_MODELS)
        | {ModelID.MIT48_AR}
    )
    payload = {
        "runtime": "cuda13",
        "provisioned_tier": "full",
        "requirements": {"path": "requirements-cuda13.txt", "sha256": "a" * 64},
        "application_models": [
            {"id": model_id.value, "files": [{"path": "sealed-model.bin"}]}
            for model_id in model_ids
        ],
        "llama_image": {"ref": "ghcr.io/ggml-org/llama.cpp:server-cuda13"},
    }
    args = SimpleNamespace(
        runtime="cuda13",
        requirements="requirements-cuda13.txt",
        emit_cmd=False,
    )

    def validate_models(_payload, *, model_ids, allow_digest_fallback):
        if model_ids == {ModelID.MIT48_AR.value}:
            raise windows_install_state.InstallStateError("MIT48 asset changed")
        assert allow_digest_fallback

    with (
        mock.patch.object(windows_install_state, "_read_state", return_value=payload),
        mock.patch.object(
            windows_install_state,
            "_requirements_record",
            return_value=payload["requirements"],
        ),
        mock.patch.object(
            windows_install_state,
            "_validate_application_models",
            side_effect=validate_models,
        ),
        mock.patch.object(windows_install_state, "_validate_docker_state") as validate_docker,
        pytest.raises(windows_install_state.InstallStateError, match="setup_full"),
    ):
        windows_install_state.command_preflight(args)

    validate_docker.assert_not_called()


@pytest.mark.parametrize(
    ("engine", "missing_id"),
    [
        ("MIT 48px AR", ModelID.MIT48_AR.value),
        ("MIT 48px CTC", ModelID.MIT48_CTC_ARCHIVE.value),
    ],
)
def test_unsealed_mit48_selection_requires_full_setup(
    monkeypatch,
    engine: str,
    missing_id: str,
) -> None:
    state = {
        "provisioned_tier": "core",
        "application_models": [
            {"id": model_id.value, "files": [{"path": model_id.value}]}
            for model_id in CORE_APPLICATION_MODELS
        ],
    }
    monkeypatch.setattr(
        windows_installation,
        "active_windows_runtime",
        lambda: "cuda13",
    )
    monkeypatch.setattr(
        windows_installation,
        "active_windows_install_state",
        lambda: state,
    )

    def get_tool_selection(name: str) -> str:
        return {"ocr": engine, "inpainter": "LaMa", "translator": ""}.get(
            name,
            "",
        )

    settings = SimpleNamespace(get_tool_selection=get_tool_selection)
    with pytest.raises(RuntimeError, match="setup_full") as exc_info:
        windows_installation.assert_selected_windows_models_installed(
            settings,
            "English",
        )
    assert missing_id in str(exc_info.value)


@pytest.mark.parametrize(
    ("engine", "missing_id"),
    [
        ("MIT 48px AR", ModelID.MIT48_AR.value),
        ("MIT 48px CTC", ModelID.MIT48_CTC_ARCHIVE.value),
    ],
)
def test_legacy_full_seal_requires_selected_mit48_assets_before_page_work(
    monkeypatch,
    engine: str,
    missing_id: str,
) -> None:
    legacy_full_ids = set(application_model_profile("full")) - set(
        FULL_ONLY_APPLICATION_MODELS
    )
    state = {
        "schema_version": windows_install_state.SCHEMA_VERSION,
        "provisioned_tier": "full",
        "application_models": [
            {"id": model_id.value, "files": [{"path": "sealed-model.bin"}]}
            for model_id in ModelID
            if model_id in legacy_full_ids
        ],
    }
    monkeypatch.setattr(windows_installation, "active_windows_runtime", lambda: "cuda13")
    monkeypatch.setattr(
        windows_installation,
        "active_windows_install_state",
        lambda: state,
    )
    settings = SimpleNamespace(
        get_tool_selection=lambda name: {
            "ocr": engine,
            "inpainter": "LaMa",
            "translator": "",
        }.get(name, "")
    )

    with pytest.raises(RuntimeError, match="setup_full") as exc_info:
        windows_installation.assert_selected_windows_models_installed(
            settings,
            "English",
        )

    assert missing_id in str(exc_info.value)


@pytest.mark.parametrize(
    ("engine", "required_ids"),
    [
        ("MIT 48px AR", MIT48_AR_APPLICATION_MODELS),
        ("MIT 48px CTC", MIT48_CTC_APPLICATION_MODELS),
    ],
)
def test_selected_mit48_corrupt_asset_requires_full_setup(
    monkeypatch,
    tmp_path,
    engine: str,
    required_ids: tuple[ModelID, ...],
) -> None:
    records = [
        {"id": model_id.value, "files": [{"path": model_id.value}]}
        for model_id in CORE_APPLICATION_MODELS
    ]
    corrupt_id = required_ids[0]
    for model_id in required_ids:
        relative = Path("models") / model_id.value / "model.bin"
        model_path = tmp_path / relative
        model_path.parent.mkdir(parents=True)
        model_path.write_bytes(b"x")
        stat = model_path.stat()
        monkeypatch.setitem(
            ModelDownloader.registry,
            model_id,
            ModelSpec(
                id=model_id,
                url="",
                files=[model_path.name],
                sha256=["b" * 64],
                save_dir=str(model_path.parent),
            ),
        )
        records.append(
            {
                "id": model_id.value,
                "files": [
                    {
                        "path": relative.as_posix(),
                        "bytes": 2 if model_id == corrupt_id else 1,
                        "mtime_ns": stat.st_mtime_ns,
                        "digest_algorithm": "sha256",
                        "digest": "b" * 64,
                    }
                ],
            }
        )
    state = {
        "schema_version": windows_install_state.SCHEMA_VERSION,
        "provisioned_tier": "full",
        "application_models": records,
    }
    monkeypatch.setattr(windows_install_state, "ROOT", tmp_path)
    monkeypatch.setattr(windows_installation, "active_windows_runtime", lambda: "cuda13")
    monkeypatch.setattr(
        windows_installation,
        "active_windows_install_state",
        lambda: state,
    )
    settings = SimpleNamespace(
        get_tool_selection=lambda name: {
            "ocr": engine,
            "inpainter": "LaMa",
            "translator": "",
        }.get(name, "")
    )

    with pytest.raises(RuntimeError, match="setup_full") as exc_info:
        windows_installation.assert_selected_windows_models_installed(
            settings,
            "English",
        )

    assert "mit48" in str(exc_info.value).lower()


def test_ctc_archive_extracts_only_pinned_members_atomically(
    monkeypatch,
    tmp_path,
) -> None:
    archive_root = tmp_path / "ctc"
    archive_root.mkdir()
    checkpoint = b"test-checkpoint-bytes"
    dictionary = b"<blank>\nhello\n"
    archive_path = archive_root / MIT48_CTC_ARCHIVE
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(MIT48_CTC_CHECKPOINT, checkpoint)
        archive.writestr(MIT48_CTC_DICTIONARY, dictionary)
    checkpoint_sha = hashlib.sha256(checkpoint).hexdigest()
    dictionary_sha = hashlib.sha256(dictionary).hexdigest()
    destination = archive_root / MIT48_CTC_EXTRACT_DIRECTORY
    monkeypatch.setenv("COMIC_MODEL_DOWNLOAD_POLICY", "forbid")
    monkeypatch.setitem(
        ModelDownloader.registry,
        ModelID.MIT48_CTC_ARCHIVE,
        ModelSpec(
            id=ModelID.MIT48_CTC_ARCHIVE,
            url="",
            files=[MIT48_CTC_ARCHIVE],
            sha256=[hashlib.sha256(archive_path.read_bytes()).hexdigest()],
            save_dir=str(archive_root),
        ),
    )
    monkeypatch.setitem(
        ModelDownloader.registry,
        ModelID.MIT48_CTC,
        ModelSpec(
            id=ModelID.MIT48_CTC,
            url="",
            files=[MIT48_CTC_CHECKPOINT, MIT48_CTC_DICTIONARY],
            sha256=[checkpoint_sha, dictionary_sha],
            save_dir=str(destination),
        ),
    )

    download_module._prepare_mit48_ctc_members()

    assert (destination / MIT48_CTC_CHECKPOINT).read_bytes() == checkpoint
    assert (destination / MIT48_CTC_DICTIONARY).read_bytes() == dictionary
    assert not list(archive_root.glob(".mit48-ctc-*"))


def test_ctc_archive_path_traversal_is_rejected_without_extraction(
    monkeypatch,
    tmp_path,
) -> None:
    archive_root = tmp_path / "ctc"
    archive_root.mkdir()
    archive_path = archive_root / MIT48_CTC_ARCHIVE
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("../outside.txt", b"outside")
    destination = archive_root / MIT48_CTC_EXTRACT_DIRECTORY
    monkeypatch.setenv("COMIC_MODEL_DOWNLOAD_POLICY", "forbid")
    monkeypatch.setitem(
        ModelDownloader.registry,
        ModelID.MIT48_CTC_ARCHIVE,
        ModelSpec(
            id=ModelID.MIT48_CTC_ARCHIVE,
            url="",
            files=[MIT48_CTC_ARCHIVE],
            sha256=[hashlib.sha256(archive_path.read_bytes()).hexdigest()],
            save_dir=str(archive_root),
        ),
    )
    monkeypatch.setitem(
        ModelDownloader.registry,
        ModelID.MIT48_CTC,
        ModelSpec(
            id=ModelID.MIT48_CTC,
            url="",
            files=["../outside.txt"],
            sha256=[hashlib.sha256(b"outside").hexdigest()],
            save_dir=str(destination),
        ),
    )

    with pytest.raises(RuntimeError, match="unsafe member"):
        download_module._prepare_mit48_ctc_members()

    assert not (tmp_path / "outside.txt").exists()
    assert not destination.exists()
    assert not list(archive_root.glob(".mit48-ctc-*"))


@pytest.mark.parametrize(
    ("archive_case", "expected_error"),
    [
        ("unexpected_member", "members do not match"),
        ("symlink", "unsafe member"),
        ("encrypted", "unsafe member"),
        ("oversized", "size limit"),
        ("high_ratio", "ratio limit"),
        ("sha_mismatch", "failed verification"),
    ],
)
def test_ctc_archive_rejects_unsafe_members_without_publishing(
    monkeypatch,
    tmp_path,
    archive_case: str,
    expected_error: str,
) -> None:
    archive_root = tmp_path / "ctc"
    archive_root.mkdir()
    checkpoint = b"a" * (100_000 if archive_case == "high_ratio" else 20)
    dictionary = b"neutral fixture\n"
    archive_path = archive_root / MIT48_CTC_ARCHIVE
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        checkpoint_info = zipfile.ZipInfo(MIT48_CTC_CHECKPOINT)
        checkpoint_info.compress_type = zipfile.ZIP_DEFLATED
        if archive_case == "symlink":
            checkpoint_info.external_attr = (stat.S_IFLNK | 0o777) << 16
        archive.writestr(checkpoint_info, checkpoint)
        archive.writestr(MIT48_CTC_DICTIONARY, dictionary)
        if archive_case == "unexpected_member":
            archive.writestr("extra.bin", b"extra")

    if archive_case in {"encrypted", "oversized"}:
        archive_bytes = bytearray(archive_path.read_bytes())
        central_entry = archive_bytes.find(b"PK\x01\x02")
        assert central_entry >= 0
        if archive_case == "encrypted":
            flags = struct.unpack_from("<H", archive_bytes, central_entry + 8)[0]
            struct.pack_into("<H", archive_bytes, central_entry + 8, flags | 1)
        else:
            struct.pack_into(
                "<I",
                archive_bytes,
                central_entry + 24,
                1024**3 + 1,
            )
        archive_path.write_bytes(archive_bytes)

    destination = archive_root / MIT48_CTC_EXTRACT_DIRECTORY
    monkeypatch.setenv("COMIC_MODEL_DOWNLOAD_POLICY", "forbid")
    monkeypatch.setitem(
        ModelDownloader.registry,
        ModelID.MIT48_CTC_ARCHIVE,
        ModelSpec(
            id=ModelID.MIT48_CTC_ARCHIVE,
            url="",
            files=[MIT48_CTC_ARCHIVE],
            sha256=[hashlib.sha256(archive_path.read_bytes()).hexdigest()],
            save_dir=str(archive_root),
        ),
    )
    monkeypatch.setitem(
        ModelDownloader.registry,
        ModelID.MIT48_CTC,
        ModelSpec(
            id=ModelID.MIT48_CTC,
            url="",
            files=[MIT48_CTC_CHECKPOINT, MIT48_CTC_DICTIONARY],
            sha256=[
                (
                    "0" * 64
                    if archive_case == "sha_mismatch"
                    else hashlib.sha256(checkpoint).hexdigest()
                ),
                hashlib.sha256(dictionary).hexdigest(),
            ],
            save_dir=str(destination),
        ),
    )

    with pytest.raises(RuntimeError, match=expected_error):
        download_module._prepare_mit48_ctc_members()

    assert not destination.exists()
    assert not list(archive_root.glob(".mit48-ctc-*"))


def _write_ctc_repair_fixture(
    monkeypatch,
    archive_root: Path,
    destination: Path,
    checkpoint: bytes,
    dictionary: bytes,
    *,
    expected_checkpoint_sha: str | None = None,
) -> Path:
    archive_root.mkdir(parents=True, exist_ok=True)
    archive_path = archive_root / MIT48_CTC_ARCHIVE
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(MIT48_CTC_CHECKPOINT, checkpoint)
        archive.writestr(MIT48_CTC_DICTIONARY, dictionary)
    monkeypatch.setenv("COMIC_MODEL_DOWNLOAD_POLICY", "forbid")
    monkeypatch.setitem(
        ModelDownloader.registry,
        ModelID.MIT48_CTC_ARCHIVE,
        ModelSpec(
            id=ModelID.MIT48_CTC_ARCHIVE,
            url="",
            files=[MIT48_CTC_ARCHIVE],
            sha256=[hashlib.sha256(archive_path.read_bytes()).hexdigest()],
            save_dir=str(archive_root),
        ),
    )
    monkeypatch.setitem(
        ModelDownloader.registry,
        ModelID.MIT48_CTC,
        ModelSpec(
            id=ModelID.MIT48_CTC,
            url="",
            files=[MIT48_CTC_CHECKPOINT, MIT48_CTC_DICTIONARY],
            sha256=[
                expected_checkpoint_sha or hashlib.sha256(checkpoint).hexdigest(),
                hashlib.sha256(dictionary).hexdigest(),
            ],
            save_dir=str(destination),
        ),
    )
    return archive_path


def test_setup_full_repairs_ctc_directory_and_preserves_original(monkeypatch, tmp_path) -> None:
    archive_root = tmp_path / "ctc"
    destination = archive_root / MIT48_CTC_EXTRACT_DIRECTORY
    destination.mkdir(parents=True)
    (destination / "preserve.marker").write_bytes(b"old extracted data")
    checkpoint = b"new neutral checkpoint"
    dictionary = b"neutral dictionary\n"
    _write_ctc_repair_fixture(
        monkeypatch,
        archive_root,
        destination,
        checkpoint,
        dictionary,
    )

    download_module._prepare_mit48_ctc_members()

    assert (destination / MIT48_CTC_CHECKPOINT).read_bytes() == checkpoint
    assert (destination / MIT48_CTC_DICTIONARY).read_bytes() == dictionary
    backups = list(archive_root.glob(".mit48-ctc-preserved-*"))
    assert len(backups) == 1
    assert (backups[0] / destination.name / "preserve.marker").read_bytes() == (
        b"old extracted data"
    )


def test_setup_full_restores_original_ctc_directory_if_publish_fails(
    monkeypatch,
    tmp_path,
) -> None:
    archive_root = tmp_path / "ctc"
    destination = archive_root / MIT48_CTC_EXTRACT_DIRECTORY
    destination.mkdir(parents=True)
    marker = destination / "preserve.marker"
    marker.write_bytes(b"old extracted data")
    _write_ctc_repair_fixture(
        monkeypatch,
        archive_root,
        destination,
        b"new neutral checkpoint",
        b"neutral dictionary\n",
    )
    original_replace = download_module.os.replace

    def fail_publish(source, target):
        if Path(target) == destination and Path(source).name.startswith(
            ".mit48-ctc-extract-"
        ):
            raise OSError("simulated publish failure")
        return original_replace(source, target)

    monkeypatch.setattr(download_module.os, "replace", fail_publish)
    with pytest.raises(OSError, match="simulated publish failure"):
        download_module._prepare_mit48_ctc_members()

    assert marker.read_bytes() == b"old extracted data"
    assert not (destination / MIT48_CTC_CHECKPOINT).exists()
    assert not list(archive_root.glob(".mit48-ctc-extract-*"))


def test_setup_full_does_not_move_ctc_directory_before_fresh_hashes_pass(
    monkeypatch,
    tmp_path,
) -> None:
    archive_root = tmp_path / "ctc"
    destination = archive_root / MIT48_CTC_EXTRACT_DIRECTORY
    destination.mkdir(parents=True)
    marker = destination / "preserve.marker"
    marker.write_bytes(b"old extracted data")
    _write_ctc_repair_fixture(
        monkeypatch,
        archive_root,
        destination,
        b"new neutral checkpoint",
        b"neutral dictionary\n",
        expected_checkpoint_sha="0" * 64,
    )

    with pytest.raises(RuntimeError, match="failed verification"):
        download_module._prepare_mit48_ctc_members()

    assert marker.read_bytes() == b"old extracted data"
    assert not list(archive_root.glob(".mit48-ctc-preserved-*"))
    assert not list(archive_root.glob(".mit48-ctc-extract-*"))


@pytest.mark.parametrize("stall", ["request", "response"])
def test_native_worker_request_and_response_stalls_are_bounded(
    monkeypatch,
    stall: str,
) -> None:
    monkeypatch.setattr(
        mit48_engine,
        "_WORKER_REQUEST_WRITE_TIMEOUT_SEC",
        0.05,
        raising=False,
    )
    monkeypatch.setattr(
        mit48_engine,
        "_WORKER_RESPONSE_STALL_TIMEOUT_SEC",
        0.05,
        raising=False,
    )
    process = _StalledWorkerProcess(block_request=stall == "request")
    monkeypatch.setattr(mit48_engine.subprocess, "Popen", lambda *_args, **_kwargs: process)
    cancelled = threading.Event()
    engine = Mit48OCREngine("ar", "cpu", "English")
    engine.set_cancel_checker(cancelled.is_set)
    done = threading.Event()
    outcome: dict[str, BaseException] = {}

    def run_worker() -> None:
        try:
            engine._run_worker(np.zeros((8, 8, 3), dtype=np.uint8), [])
        except BaseException as exc:
            outcome["error"] = exc
        finally:
            done.set()

    thread = threading.Thread(target=run_worker, daemon=True)
    thread.start()
    completed_before_cancel = done.wait(0.4)
    if not completed_before_cancel:
        cancelled.set()
        process.terminate()
        done.wait(2.0)

    assert done.is_set()
    assert isinstance(outcome.get("error"), TimeoutError)
    assert process.poll() is not None


def test_native_worker_waits_for_writer_completion_notification(monkeypatch) -> None:
    events: list[str] = []
    output = _successful_native_worker(events).stdout.getvalue()
    process = _RequestRaceProcess(events, output)
    lease = _FakeLease(events)
    lease.process = process
    engine = Mit48OCREngine("ar", "cuda", "English")
    engine.set_inference_lease_factory(lambda: lease)
    monkeypatch.setattr(
        mit48_engine.subprocess,
        "Popen",
        lambda *_args, **_kwargs: process,
    )

    original_event = threading.Event
    event_count = 0
    delayed = []

    class DelayedEvent:
        def __init__(self) -> None:
            self.event = original_event()
            self.release = original_event()
            self.wait_called = original_event()

        def set(self) -> None:
            self.release.wait(2.0)
            self.event.set()

        def is_set(self) -> bool:
            return self.event.is_set()

        def wait(self, timeout=None) -> bool:
            self.wait_called.set()
            self.release.set()
            return self.event.wait(timeout)

    def event_factory():
        nonlocal event_count
        event_count += 1
        if event_count == 4:
            event = DelayedEvent()
            delayed.append(event)
            return event
        return original_event()

    monkeypatch.setattr(mit48_engine.threading, "Event", event_factory)
    try:
        response = engine._run_worker(np.zeros((8, 8, 3), dtype=np.uint8), [])
    finally:
        for event in delayed:
            event.release.set()

    assert response["type"] == "result"
    assert delayed and delayed[0].wait_called.is_set()
    assert process.poll() == 0
    assert "worker_reaped" in events


def test_unreaped_worker_uses_bounded_kill_and_keeps_failed_lease(
    monkeypatch,
) -> None:
    checks = iter([False, True])
    processor, arbiter, _transitions, _cached, page, policy = _stage_native_ocr(
        monkeypatch,
        "MIT 48px AR",
        cancel_checker=lambda: next(checks, True),
    )
    lease_manager = processor.main_page.local_ocr_runtime_manager
    router_lease = lease_manager.router_inference_lease
    leases = []

    def capture_lease(*args, **kwargs):
        lease = router_lease(*args, **kwargs)
        leases.append(lease)
        return lease

    lease_manager.router_inference_lease = capture_lease
    process = _UnreapedWorkerProcess()
    monkeypatch.setattr(
        mit48_engine.subprocess,
        "Popen",
        lambda *_args, **_kwargs: process,
    )
    monkeypatch.setattr(
        "modules.ocr.local_runtime.query_cuda_handoff_metrics",
        lambda: {"sample": "neutral"},
    )

    with pytest.raises(NativeOCRWorkerReleaseError, match="release was not confirmed"):
        processor._run_primary_ocr(page, policy)

    assert process.poll() is None
    assert process.wait_timeouts == [2.0, 2.0]
    assert process.events == ["terminate", "wait", "kill", "wait"]
    assert leases[0].release_report["status"] == "worker_not_reaped"
    assert arbiter.snapshot().active_model == "mit48_ar"
    assert arbiter.snapshot().states["mit48_ar"] == RuntimeModelState.RELEASE_FAILED.value
    assert isinstance(
        _acquire_external_model_from_thread(arbiter, "mit48_ctc"),
        RuntimeLeaseConflictError,
    )


def test_over_limit_line_geometry_is_reviewed_before_warp_allocation(
    monkeypatch,
) -> None:
    image = np.zeros((2, 200, 3), dtype=np.uint8)
    warp = mock.Mock(return_value=SimpleNamespace(shape=(48, 9_600, 3)))
    monkeypatch.setattr(mit48_worker.cv2, "warpPerspective", warp)
    monkeypatch.setattr(mit48_worker, "_infer_lines", lambda *_args, **_kwargs: [])
    request = {
        "variant": "ar",
        "device": "cpu",
        "source_language": "en",
        "image": image,
        "blocks": [
            {
                "block_id": "neutral-block",
                "xyxy": [0, 0, 200, 2],
                "lines": [[[0, 0], [200, 0], [200, 1], [0, 1]]],
                "direction": "horizontal",
                "source_lang": "en",
            }
        ],
    }

    response = _infer(request, object(), SimpleNamespace(__version__="fixture"))

    block = response["blocks"][0]
    assert block["empty_reason"] == "geometry_review"
    assert "8192" in block.get("geometry_error", "")
    assert "native_line_quad" in block.get("geometry_error", "")
    warp.assert_not_called()


def test_valid_native_line_crop_keeps_its_source_bounds() -> None:
    image = np.zeros((12, 40, 3), dtype=np.uint8)
    polygon = [[2, 2], [32, 2], [32, 7], [2, 7]]

    region, width, bbox = _transform_line(image, polygon, vertical=False)

    assert region.shape == (48, 288, 3)
    assert width == 288
    assert bbox == [2, 2, 33, 8]








def _patch_core_mit48_windows_state(monkeypatch) -> dict:
    state = {
        "schema_version": windows_install_state.SCHEMA_VERSION,
        "provisioned_tier": "core",
        "application_models": [
            {"id": model_id.value, "files": [{"path": model_id.value}]}
            for model_id in CORE_APPLICATION_MODELS
        ],
    }
    monkeypatch.setattr(windows_installation, "active_windows_runtime", lambda: "cuda13")
    monkeypatch.setattr(
        windows_installation,
        "active_windows_install_state",
        lambda: state,
    )
    return state


def _mit48_page_settings():
    return SimpleNamespace(
        get_tool_selection=lambda key: {
            "ocr": "mit48px",
            "inpainter": "LaMa",
            "translator": "",
        }.get(key, ""),
        is_gpu_enabled=lambda: False,
    )


def _patch_valid_mit48_windows_state(
    monkeypatch,
    tmp_path,
    *,
    invalid_field: str | None = None,
    invalid_value=None,
) -> dict:
    state = _patch_core_mit48_windows_state(monkeypatch)
    state["provisioned_tier"] = "full"
    for model_id in MIT48_AR_APPLICATION_MODELS:
        model_dir = tmp_path / "models" / "mit48-fixture" / model_id.value
        model_dir.mkdir(parents=True)
        model_file = model_dir / "asset.bin"
        model_file.write_bytes(f"neutral-{model_id.value}".encode())
        digest = hashlib.sha256(model_file.read_bytes()).hexdigest()
        monkeypatch.setitem(
            ModelDownloader.registry,
            model_id,
            ModelSpec(
                id=model_id,
                url="",
                files=[model_file.name],
                sha256=[digest],
                save_dir=str(model_dir),
            ),
        )
        stat = model_file.stat()
        file_record = {
            "path": model_file.relative_to(tmp_path).as_posix(),
            "bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "digest_algorithm": "sha256",
            "digest": digest,
        }
        if invalid_field and model_id == MIT48_AR_APPLICATION_MODELS[0]:
            file_record[invalid_field] = invalid_value
        state["application_models"].append(
            {"id": model_id.value, "files": [file_record]}
        )
    monkeypatch.setattr(windows_install_state, "ROOT", tmp_path)
    state["managed_runtimes"] = [
        {"name": "PaddleOCR VL", "runtime_name": "PaddleOCR-VL-llama.cpp"}
    ]
    return state


def test_normal_batch_preflights_selected_mit48_before_dispatch(monkeypatch) -> None:
    from pipeline.main_pipeline import ComicTranslatePipeline

    _patch_core_mit48_windows_state(monkeypatch)
    settings = _mit48_page_settings()
    settings.get_workflow_mode = lambda: "legacy_page_pipeline"
    page = SimpleNamespace(
        settings_page=settings,
        s_combo=SimpleNamespace(currentText=lambda: "English"),
        lang_mapping={"English": "English"},
        image_files=[],
    )
    pipeline = object.__new__(ComicTranslatePipeline)
    pipeline.main_page = page
    pipeline.batch_processor = SimpleNamespace(batch_process=mock.Mock())
    pipeline.stage_batched_processor = SimpleNamespace(batch_process=mock.Mock())

    with pytest.raises(RuntimeError, match="setup_full"):
        pipeline.batch_process(["example_page_a"])

    pipeline.batch_processor.batch_process.assert_not_called()
    pipeline.stage_batched_processor.batch_process.assert_not_called()


def test_webtoon_batch_preflights_selected_mit48_before_dispatch(monkeypatch) -> None:
    from pipeline.main_pipeline import ComicTranslatePipeline

    _patch_core_mit48_windows_state(monkeypatch)
    page = SimpleNamespace(
        settings_page=_mit48_page_settings(),
        s_combo=SimpleNamespace(currentText=lambda: "English"),
        lang_mapping={"English": "English"},
    )
    pipeline = object.__new__(ComicTranslatePipeline)
    pipeline.main_page = page
    pipeline.webtoon_batch_processor = SimpleNamespace(
        webtoon_batch_process=mock.Mock()
    )

    with pytest.raises(RuntimeError, match="setup_full"):
        pipeline.webtoon_batch_process(["example_page_a"])

    pipeline.webtoon_batch_processor.webtoon_batch_process.assert_not_called()


def test_ocr_cache_ingress_preflights_selected_mit48_before_lookup(monkeypatch) -> None:
    from pipeline.ocr_handler import OCRHandler

    _patch_core_mit48_windows_state(monkeypatch)
    pixel_loader = mock.Mock(side_effect=AssertionError("pixel materialization ran"))
    viewer = SimpleNamespace(
        hasPhoto=lambda: True,
        rectangles=[object()],
        get_image_array=pixel_loader,
    )
    main = SimpleNamespace(
        s_combo=SimpleNamespace(currentText=lambda: "English"),
        image_viewer=viewer,
        settings_page=_mit48_page_settings(),
        lang_mapping={"English": "English"},
    )
    handler = object.__new__(OCRHandler)
    handler.main_page = main
    handler.cache_manager = SimpleNamespace(
        _get_ocr_cache_key=mock.Mock(side_effect=AssertionError("cache lookup ran"))
    )

    with pytest.raises(RuntimeError, match="setup_full"):
        handler.OCR_image()

    viewer.get_image_array.assert_not_called()
    handler.cache_manager._get_ocr_cache_key.assert_not_called()


def test_ocr_cache_ingress_ignores_unselected_aot_and_gemma_seals(
    monkeypatch,
) -> None:
    from pipeline.ocr_handler import OCRHandler

    state = _patch_core_mit48_windows_state(monkeypatch)
    state["managed_runtimes"] = [
        {"name": "PaddleOCR VL", "runtime_name": "PaddleOCR-VL-llama.cpp"}
    ]
    settings = SimpleNamespace(
        get_tool_selection=lambda key: {
            "ocr": "PaddleOCR VL",
            "inpainter": "AOT",
            "translator": "Custom Local Server(Gemma)",
        }.get(key, ""),
        is_gpu_enabled=lambda: False,
    )
    pixel_loader = mock.Mock(return_value=np.zeros((8, 8, 3), dtype=np.uint8))
    viewer = SimpleNamespace(
        hasPhoto=lambda: True,
        rectangles=[object()],
        get_image_array=pixel_loader,
    )
    main = SimpleNamespace(
        s_combo=SimpleNamespace(currentText=lambda: "English"),
        image_viewer=viewer,
        settings_page=settings,
        lang_mapping={"English": "English"},
        blk_list=[],
    )
    handler = object.__new__(OCRHandler)
    handler.main_page = main
    handler.cache_manager = SimpleNamespace(
        _get_ocr_cache_key=mock.Mock(return_value="neutral-cache"),
        _can_serve_all_blocks_from_ocr_cache=mock.Mock(return_value=True),
        _apply_cached_ocr_to_blocks=mock.Mock(),
    )
    handler._apply_ocr_corrections = mock.Mock()
    handler._persist_current_page_ocr_state = mock.Mock()

    handler.OCR_image()

    pixel_loader.assert_called_once()
    handler.cache_manager._get_ocr_cache_key.assert_called_once()


def test_webtoon_visible_ocr_preflights_before_materializing_visible_image(
    monkeypatch,
) -> None:
    from pipeline.ocr_handler import OCRHandler

    _patch_core_mit48_windows_state(monkeypatch)
    viewer = SimpleNamespace(
        hasPhoto=lambda: True,
        get_visible_area_image=mock.Mock(side_effect=AssertionError("page work ran")),
    )
    main = SimpleNamespace(
        s_combo=SimpleNamespace(currentText=lambda: "English"),
        image_viewer=viewer,
        webtoon_mode=True,
        settings_page=_mit48_page_settings(),
        lang_mapping={"English": "English"},
    )
    handler = object.__new__(OCRHandler)
    handler.main_page = main

    with pytest.raises(RuntimeError, match="setup_full"):
        handler.OCR_webtoon_visible_area()

    viewer.get_visible_area_image.assert_not_called()


@pytest.mark.parametrize("engine", ["PaddleOCR VL", "MIT 48px AR"])
def test_manual_ocr_ignores_unused_aot_and_gemma_seals(
    monkeypatch,
    tmp_path,
    engine: str,
) -> None:
    from app.controllers import manual_workflow

    if engine == "MIT 48px AR":
        _patch_valid_mit48_windows_state(monkeypatch, tmp_path)
    else:
        state = _patch_core_mit48_windows_state(monkeypatch)
        state["managed_runtimes"] = [
            {"name": "PaddleOCR VL", "runtime_name": "PaddleOCR-VL-llama.cpp"}
        ]
    settings = SimpleNamespace(
        get_tool_selection=lambda key: {
            "ocr": engine,
            "inpainter": "AOT",
            "translator": "Custom Local Server(Gemma)",
        }.get(key, ""),
        is_gpu_enabled=lambda: False,
        get_ocr_result_dictionary_rules=lambda: [],
    )
    paths = ["example_page_a", "example_page_b"]
    blocks = {
        path: [
            TextBlock(
                text_bbox=np.asarray([1, 1, 8, 8], dtype=np.int32),
                text="neutral",
                block_id=f"manual-{index}",
            )
        ]
        for index, path in enumerate(paths)
    }
    cache_manager = SimpleNamespace(
        _get_ocr_cache_key=mock.Mock(return_value="neutral-cache"),
        _can_serve_all_blocks_from_ocr_cache=mock.Mock(return_value=True),
        _apply_cached_ocr_to_blocks=mock.Mock(),
    )
    main = SimpleNamespace(
        tr=lambda text: text,
        settings_page=settings,
        s_combo=SimpleNamespace(currentText=lambda: "English"),
        lang_mapping={"English": "English"},
        image_states={
            path: {"blk_list": blocks[path], "source_lang": "English"}
            for path in paths
        },
        pipeline=SimpleNamespace(cache_manager=cache_manager),
        loading=SimpleNamespace(setVisible=mock.Mock()),
        disable_hbutton_group=mock.Mock(),
        run_threaded=lambda work, *_callbacks: work(),
    )
    workflow = object.__new__(manual_workflow.ManualWorkflowController)
    workflow.main = main
    workflow._selected_page_paths = mock.Mock(return_value=paths)
    workflow._prepare_multi_page_context = mock.Mock(
        return_value={"current_file": paths[0], "current_page_unloaded": False}
    )
    workflow._show_manual_busy = mock.Mock(return_value=object())
    workflow._load_page_image = mock.Mock(
        return_value=np.zeros((8, 8, 3), dtype=np.uint8)
    )

    class FakeOCR:
        last_engine_name = "neutral-fake"

        def initialize(self, *_args, **_kwargs) -> None:
            return None

    monkeypatch.setattr(manual_workflow, "OCRProcessor", FakeOCR)
    monkeypatch.setattr(manual_workflow, "validate_ocr", lambda _main: True)
    monkeypatch.setattr(manual_workflow, "apply_ocr_result_dictionary", lambda *_: None)
    monkeypatch.setattr(
        manual_workflow,
        "summarize_ocr_quality",
        lambda _blocks: {
            "block_count": 1,
            "non_empty": 1,
            "empty": 0,
            "single_char_like": 0,
        },
    )

    workflow.ocr()

    assert workflow._load_page_image.call_count == len(paths)
    assert cache_manager._get_ocr_cache_key.call_count == len(paths)


@pytest.mark.parametrize(("field", "value"), [("bytes", []), ("mtime_ns", {})])
def test_ocr_cache_gate_wraps_malformed_mit48_file_metadata(
    monkeypatch,
    tmp_path,
    field: str,
    value,
) -> None:
    from pipeline.ocr_handler import OCRHandler

    _patch_valid_mit48_windows_state(
        monkeypatch,
        tmp_path,
        invalid_field=field,
        invalid_value=value,
    )
    pixel_loader = mock.Mock(side_effect=AssertionError("pixel materialization ran"))
    viewer = SimpleNamespace(
        hasPhoto=lambda: True,
        rectangles=[object()],
        get_image_array=pixel_loader,
    )
    main = SimpleNamespace(
        s_combo=SimpleNamespace(currentText=lambda: "English"),
        image_viewer=viewer,
        settings_page=_mit48_page_settings(),
        lang_mapping={"English": "English"},
    )
    handler = object.__new__(OCRHandler)
    handler.main_page = main
    handler.cache_manager = SimpleNamespace(
        _get_ocr_cache_key=mock.Mock(side_effect=AssertionError("cache lookup ran"))
    )

    with pytest.raises(RuntimeError, match="setup_full"):
        handler.OCR_image()

    pixel_loader.assert_not_called()
    handler.cache_manager._get_ocr_cache_key.assert_not_called()


@pytest.mark.parametrize(
    "raw_state",
    [b"\xff", b'{"schema_version":[],"application_models":[]}'],
)
def test_ocr_cache_gate_handles_malformed_install_state(
    monkeypatch,
    tmp_path,
    raw_state: bytes,
) -> None:
    from pipeline.ocr_handler import OCRHandler

    state_path = tmp_path / ".comic-bootstrap" / "install-state-cuda13.json"
    state_path.parent.mkdir()
    state_path.write_bytes(raw_state)
    monkeypatch.setenv("COMIC_WINDOWS_RUNTIME", "cuda13")
    monkeypatch.setattr(windows_installation, "get_repo_root", lambda: tmp_path)
    pixel_loader = mock.Mock(side_effect=AssertionError("pixel materialization ran"))
    viewer = SimpleNamespace(
        hasPhoto=lambda: True,
        rectangles=[object()],
        get_image_array=pixel_loader,
    )
    main = SimpleNamespace(
        s_combo=SimpleNamespace(currentText=lambda: "English"),
        image_viewer=viewer,
        settings_page=_mit48_page_settings(),
        lang_mapping={"English": "English"},
    )
    handler = object.__new__(OCRHandler)
    handler.main_page = main
    handler.cache_manager = SimpleNamespace(
        _get_ocr_cache_key=mock.Mock(side_effect=AssertionError("cache lookup ran"))
    )

    with pytest.raises(RuntimeError, match="setup_full"):
        handler.OCR_image()

    pixel_loader.assert_not_called()


def test_manual_batch_preflights_mit48_before_loading_pages(monkeypatch) -> None:
    from app.controllers import manual_workflow

    _patch_core_mit48_windows_state(monkeypatch)
    paths = ["example_page_a", "example_page_b"]
    block = TextBlock(
        text_bbox=np.asarray([1, 1, 8, 8], dtype=np.int32),
        text="neutral",
        block_id="manual-block",
    )
    main = SimpleNamespace(
        tr=lambda text: text,
        settings_page=_mit48_page_settings(),
        s_combo=SimpleNamespace(currentText=lambda: "English"),
        lang_mapping={"English": "English"},
        image_states={path: {"blk_list": [block], "source_lang": "English"} for path in paths},
        pipeline=SimpleNamespace(cache_manager=object()),
        loading=SimpleNamespace(setVisible=mock.Mock()),
        disable_hbutton_group=mock.Mock(),
        run_threaded=lambda work, *_callbacks: work(),
    )
    workflow = object.__new__(manual_workflow.ManualWorkflowController)
    workflow.main = main
    workflow._selected_page_paths = mock.Mock(return_value=paths)
    workflow._prepare_multi_page_context = mock.Mock(
        return_value={"current_file": paths[0], "current_page_unloaded": False}
    )
    workflow._show_manual_busy = mock.Mock(return_value=object())
    workflow._load_page_image = mock.Mock(side_effect=AssertionError("page load ran"))
    monkeypatch.setattr(manual_workflow, "validate_ocr", lambda _main: True)

    with pytest.raises(RuntimeError, match="setup_full"):
        workflow.ocr()

    workflow._load_page_image.assert_not_called()




def _write_core_setup_seal(monkeypatch, tmp_path, old_state: str | bytes) -> dict:
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    monkeypatch.setattr(windows_install_state, "STATE_DIR", state_dir)
    old_path = windows_install_state._state_path("cuda13")
    old_path.write_bytes(old_state if isinstance(old_state, bytes) else old_state.encode())
    managed_path = tmp_path / "managed-state.json"
    managed_path.write_text(
        json.dumps({
            "image_ref": "ghcr.io/ggml-org/llama.cpp:server-cuda13",
            "image_id": "sha256:test-image",
            "volumes": [
                {"name": f"runtime-{index}", "runtime_name": name, "preparation_version": 1}
                for index, name in enumerate(
                    sorted(windows_install_state.CORE_MANAGED_RUNTIMES)
                )
            ],
        }),
        encoding="utf-8",
    )
    args = SimpleNamespace(
        runtime="cuda13",
        tier="core",
        profile="core",
        requirements="requirements-cuda13.txt",
        image_ref="ghcr.io/ggml-org/llama.cpp:server-cuda13",
        image_id="sha256:test-image",
        required_cuda="13.1",
        managed_state=str(managed_path),
    )
    with (
        mock.patch.object(windows_install_state, "_assert_runtime_image"),
        mock.patch.object(
            windows_install_state,
            "_requirements_record",
            return_value={"path": args.requirements, "sha256": "a" * 64},
        ),
        mock.patch.object(
            windows_install_state,
            "_model_records",
            return_value=[
                {"id": model_id.value, "files": [{}]}
                for model_id in CORE_APPLICATION_MODELS
            ],
        ),
        mock.patch.object(windows_install_state, "_validate_managed_runtime_records"),
    ):
        windows_install_state.command_write(args)

    return json.loads(windows_install_state._state_path("cuda13").read_text())


@pytest.mark.parametrize(
    "old_state",
    [
        "not-json",
        '{"schema_version":0}',
        '{"schema_version":[]}',
        b"\xff",
    ],
)
def test_setup_write_recovers_malformed_old_seal(
    monkeypatch,
    tmp_path,
    old_state: str | bytes,
) -> None:
    repaired = _write_core_setup_seal(monkeypatch, tmp_path, old_state)

    assert repaired["provisioned_tier"] == "core"
    assert {record["id"] for record in repaired["application_models"]} == {
        model_id.value for model_id in CORE_APPLICATION_MODELS
    }


@pytest.mark.parametrize(("field", "value"), [("bytes", []), ("mtime_ns", {})])
def test_setup_write_drops_malformed_mit48_record_metadata(
    monkeypatch,
    tmp_path,
    field: str,
    value,
) -> None:
    model_dir = tmp_path / "models" / "mit48" / "ar"
    model_dir.mkdir(parents=True)
    model_file = model_dir / "checkpoint.bin"
    model_file.write_bytes(b"neutral fixture")
    digest = hashlib.sha256(model_file.read_bytes()).hexdigest()
    monkeypatch.setattr(windows_install_state, "ROOT", tmp_path)
    monkeypatch.setitem(
        ModelDownloader.registry,
        ModelID.MIT48_AR,
        ModelSpec(
            id=ModelID.MIT48_AR,
            url="",
            files=[model_file.name],
            sha256=[digest],
            save_dir=str(model_dir),
        ),
    )
    stat = model_file.stat()
    file_record = {
        "path": model_file.relative_to(tmp_path).as_posix(),
        "bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "digest_algorithm": "sha256",
        "digest": digest,
    }
    file_record[field] = value
    old_state = json.dumps({
        "schema_version": windows_install_state.SCHEMA_VERSION,
        "application_models": [
            {"id": ModelID.MIT48_AR.value, "files": [file_record]}
        ],
    })

    repaired = _write_core_setup_seal(monkeypatch, tmp_path, old_state)

    assert {record["id"] for record in repaired["application_models"]} == {
        model_id.value for model_id in CORE_APPLICATION_MODELS
    }










def test_ar_model_identity_is_pinned_to_source_and_checkpoint() -> None:
    assert MIT48_RELEASE_TAG == "beta-0.3"
    assert MIT48_UPSTREAM_COMMIT == "a9ef04da0adda877413afe40b7656aebe22dc4be"
    assert MIT48_AR_CHECKPOINT_SHA256 == "29daa46d080818bb4ab239a518a88338cbccff8f901bef8c9db191a7cb97671d"
    assert MIT48_AR_DICTIONARY_SHA256 == "f5722368146aa0fbcc9f4726866e4efc3203318ebb66c811d8cbbe915576538a"
    assert MIT48_CTC_ARCHIVE_SHA256 == "fc61c52f7a811bc72c54f6be85df814c6b60f63585175db27cb94a08e0c30101"
    assert MIT48_CTC_CHECKPOINT_SHA256 == "8b0837a24da5fde96c23ca47bb7abd590cd5b185c307e348c6e0b7238178ed89"
    assert MIT48_CTC_DICTIONARY_SHA256 == "c1295ae1962e69e35b5b225a0405d1f3432e368c9941d23bfd3acda12654da33"
    assert MIT48_MODEL_SOURCE_SHA256 == {
        "model_48px.py": "6f5a8aee4af67223f2b11a9421dd873a18ce78d2deebd6a75be81b5db18ae453",
        "model_48px_ctc.py": "dcb3d7e830f9fb188e7f0b10b6a7bd2713ad36df205706dbb86dfff29cdb05d6",
        "xpos_relative_position.py": "27420e6068ab687a084878a4be6c59a245290927fd51057dc9f65838e24e694f",
        "LICENSE-GPL-3.0.txt": "3972dc9744f6499f0f9b2dbf76696f2ae7ad8af9b23dde66d6af86c9dfb36986",
        "LICENSE-MIT-XPOS.txt": "c2cfccb812fe482101a8f04597dfc5a9991a6b2748266c47ac91b6a5aae15383",
    }
    assert MIT48_UPSTREAM_LICENSE_SHA256 == "3972dc9744f6499f0f9b2dbf76696f2ae7ad8af9b23dde66d6af86c9dfb36986"
    assert MIT48_XPOS_LICENSE_COMMIT == "47a4c0adc0f9328ff72f241e7067b4d1a95401cd"
    assert MIT48_XPOS_LICENSE_SHA256 == "c2cfccb812fe482101a8f04597dfc5a9991a6b2748266c47ac91b6a5aae15383"
    assert MIT48_AR_MODEL_IDENTITY == (
        f"mit48-ar:{MIT48_UPSTREAM_COMMIT}:"
        f"{MIT48_AR_CHECKPOINT_SHA256}:{MIT48_AR_DICTIONARY_SHA256}"
    )
    assert MIT48_CTC_MODEL_IDENTITY == (
        f"mit48-ctc:{MIT48_UPSTREAM_COMMIT}:"
        f"{MIT48_CTC_CHECKPOINT_SHA256}:{MIT48_CTC_DICTIONARY_SHA256}"
    )
    assert MIT48_AR_MODEL_IDENTITY in CacheManager.MIT48_AR_CACHE_VERSION
    assert MIT48_CTC_MODEL_IDENTITY in CacheManager.MIT48_CTC_CACHE_VERSION
    assert MIT48_RUNTIME_IDENTITY in CacheManager.MIT48_AR_CACHE_VERSION
    assert MIT48_RUNTIME_IDENTITY in CacheManager.MIT48_CTC_CACHE_VERSION
    assert CacheManager._default_ocr_cache_version("mit48px") == (
        CacheManager.MIT48_AR_CACHE_VERSION
    )
    assert CacheManager._default_ocr_cache_version("mit48px_ctc") == (
        CacheManager.MIT48_CTC_CACHE_VERSION
    )

    cache = CacheManager()
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    ar_key = cache._get_ocr_cache_key(image, "English", "mit48px", "cpu")
    ctc_key = cache._get_ocr_cache_key(image, "English", "mit48px_ctc", "cpu")
    ar_engine_key = cache._get_ocr_cache_key(
        image,
        "English",
        "MIT 48px AR",
        "cpu",
        resolved_ocr_model="MIT 48px AR",
    )
    ctc_engine_key = cache._get_ocr_cache_key(
        image,
        "English",
        "MIT 48px CTC",
        "cpu",
        resolved_ocr_model="MIT 48px CTC",
    )
    default_key = cache._get_ocr_cache_key(image, "English", "Default", "cpu")
    existing_key = cache._get_ocr_cache_key(image, "English", "Microsoft OCR", "cpu")
    hunyuan_key = cache._get_ocr_cache_key(image, "English", "HunyuanOCR", "cpu")
    paddle_key = cache._get_ocr_cache_key(image, "English", "PaddleOCR VL", "cpu")

    assert ar_key != ctc_key
    assert ar_key[-1] == CacheManager.MIT48_AR_CACHE_VERSION
    assert ctc_key[-1] == CacheManager.MIT48_CTC_CACHE_VERSION
    assert ar_engine_key[-1] == CacheManager.MIT48_AR_CACHE_VERSION
    assert ctc_engine_key[-1] == CacheManager.MIT48_CTC_CACHE_VERSION
    assert len(default_key) == len(existing_key) == 4
    assert hunyuan_key[-1] == CacheManager.HUNYUAN_OCR_CACHE_VERSION
    assert paddle_key[-1] == CacheManager.PADDLEOCR_VL_CACHE_VERSION


def test_stage_and_legacy_paths_reuse_the_main_page_arbiter() -> None:
    owner = SimpleNamespace(runtime_resource_arbiter=RuntimeResourceArbiter())
    legacy = OCRProcessor()
    legacy.main_page = owner
    stage = object.__new__(StageBatchedProcessor)
    stage.main_page = owner
    stage._runtime_resource_arbiter_instance = RuntimeResourceArbiter()

    assert stage._runtime_resource_arbiter() is legacy._runtime_resource_arbiter()






def test_pre_stop_sampling_failure_is_preserved_in_release_diagnostics(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "modules.ocr.local_runtime.query_cuda_handoff_metrics",
        mock.Mock(side_effect=[{"sample": "baseline"}, OSError("pre-stop sampling failed")]),
    )
    arbiter = RuntimeResourceArbiter()
    lease = _NativeOCRWorkerLease(arbiter, "mit48_ar", "cuda")
    lease.__enter__()
    lease.mark_worker_started(321)
    lease.mark_model_loaded({})
    lease.capture_pre_stop()
    lease.mark_worker_reaped()

    with pytest.raises(NativeOCRWorkerReleaseError, match="release was not confirmed"):
        lease.__exit__(None, None, None)

    assert "pre-stop sampling failed" in lease.release_report["measurement_error"]
    assert arbiter.snapshot().active_model == "mit48_ar"
    assert arbiter.snapshot().states["mit48_ar"] == RuntimeModelState.RELEASE_FAILED.value
    assert isinstance(
        _acquire_external_model_from_thread(arbiter, "mit48_ctc"),
        RuntimeLeaseConflictError,
    )


def test_engine_keeps_native_worker_release_diagnostics(monkeypatch) -> None:
    monkeypatch.setattr(
        "modules.ocr.local_runtime.query_cuda_handoff_metrics",
        mock.Mock(side_effect=[{"sample": "baseline"}, OSError("pre-stop sampling failed")]),
    )
    engine = Mit48OCREngine("ar", "cuda", "English")
    lease = _NativeOCRWorkerLease(RuntimeResourceArbiter(), "mit48_ar", "cuda")
    engine.set_inference_lease_factory(lambda: lease)

    with pytest.raises(NativeOCRWorkerReleaseError):
        with engine._inference_lease() as state:
            state.mark_worker_started(321)
            state.mark_model_loaded({})
            state.capture_pre_stop()
            state.mark_worker_reaped()

    assert "pre-stop sampling failed" in engine.last_worker_release_report[
        "measurement_error"
    ]


@pytest.mark.parametrize("pin_mismatch", ["digest", "path", "extra_file"])
def test_preflight_rejects_a_mit48_seal_that_does_not_match_current_registry(
    monkeypatch,
    tmp_path,
    pin_mismatch: str,
) -> None:
    model_dir = tmp_path / "models" / "mit48" / "ar"
    model_dir.mkdir(parents=True)
    model_file = model_dir / "checkpoint.bin"
    model_file.write_bytes(b"neutral fixture")
    stat = model_file.stat()
    monkeypatch.setattr(windows_install_state, "ROOT", tmp_path)
    monkeypatch.setitem(
        ModelDownloader.registry,
        ModelID.MIT48_AR,
        ModelSpec(
            id=ModelID.MIT48_AR,
            url="",
            files=[model_file.name],
            sha256=["b" * 64],
            save_dir=str(model_dir),
        ),
    )
    record = {
        "path": (
            (Path("models") / "mit48" / "ar" / "old.bin").as_posix()
            if pin_mismatch == "path"
            else (Path("models") / "mit48" / "ar" / model_file.name).as_posix()
        ),
        "bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "digest_algorithm": "sha256",
        "digest": "a" * 64 if pin_mismatch == "digest" else "b" * 64,
    }
    if pin_mismatch == "path":
        old_path = model_dir / "old.bin"
        old_path.write_bytes(model_file.read_bytes())
        record["bytes"] = old_path.stat().st_size
        record["mtime_ns"] = old_path.stat().st_mtime_ns
    files = [record]
    if pin_mismatch == "extra_file":
        extra_path = model_dir / "extra.bin"
        extra_path.write_bytes(b"extra")
        extra_stat = extra_path.stat()
        files.append({
            "path": (Path("models") / "mit48" / "ar" / extra_path.name).as_posix(),
            "bytes": extra_stat.st_size,
            "mtime_ns": extra_stat.st_mtime_ns,
            "digest_algorithm": "sha256",
            "digest": "b" * 64,
        })
    payload = {"application_models": [{"id": ModelID.MIT48_AR.value, "files": files}]}

    with pytest.raises(windows_install_state.InstallStateError):
        windows_install_state._validate_application_models(
            payload,
            model_ids={ModelID.MIT48_AR.value},
            allow_digest_fallback=True,
        )


def test_legacy_full_preflight_requires_the_unchanged_generic_full_profile(
    monkeypatch,
) -> None:
    payload = {
        "runtime": "cuda13",
        "provisioned_tier": "full",
        "requirements": {"path": "requirements-cuda13.txt", "sha256": "a" * 64},
        "application_models": [
            {"id": model_id.value, "files": [{"path": model_id.value}]}
            for model_id in CORE_APPLICATION_MODELS
        ],
        "llama_image": {"ref": "ghcr.io/ggml-org/llama.cpp:server-cuda13"},
    }
    args = SimpleNamespace(
        runtime="cuda13",
        requirements="requirements-cuda13.txt",
        emit_cmd=False,
    )
    with (
        mock.patch.object(windows_install_state, "_read_state", return_value=payload),
        mock.patch.object(
            windows_install_state,
            "_requirements_record",
            return_value=payload["requirements"],
        ),
        mock.patch.object(windows_install_state, "_validate_application_models"),
        mock.patch.object(windows_install_state, "_validate_docker_state") as docker,
        pytest.raises(
            windows_install_state.InstallStateError,
            match="pororo-onnx",
        ),
    ):
        windows_install_state.command_preflight(args)

    docker.assert_not_called()

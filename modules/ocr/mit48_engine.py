from __future__ import annotations

from collections import deque
from contextlib import contextmanager, nullcontext
import json
import logging
import math
import os
import pickle
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

from modules.ocr.base import OCREngine
from modules.ocr.common.result_contract import (
    PROCESSING_ACTION_REVIEW,
    SEMANTIC_ROLE_AMBIGUOUS,
    assign_ocr_processing_contract,
    initialize_ocr_result_contract,
)
from modules.ocr.mit48_assets import (
    MIT48_AR_MODEL_IDENTITY,
    MIT48_CTC_MODEL_IDENTITY,
    MIT48_RUNTIME_IDENTITY,
)
from modules.utils.download import (
    ModelID,
    ModelNotPreparedError,
    models_base_dir,
    require_prepared_model,
)
from modules.utils.exceptions import OperationCancelledError
from modules.utils.language_utils import is_no_space_lang, language_codes
from modules.utils.ocr_debug import (
    OCR_STATUS_EMPTY_INITIAL,
    OCR_STATUS_OK,
    set_block_ocr_diagnostics,
)
from modules.utils.textblock import TextBlock, new_text_block_id


logger = logging.getLogger(__name__)
_ROOT = Path(__file__).resolve().parents[2]
_REQUEST_SIZE_LIMIT = 512 * 1024**2
_PROCESS_POLL_SEC = 0.1
_WORKER_REQUEST_WRITE_TIMEOUT_SEC = 60.0
_WORKER_RESPONSE_STALL_TIMEOUT_SEC = 300.0
_WORKER_CONTROL_WRITE_TIMEOUT_SEC = 2.0
_WORKER_RESPONSE_LINE_LIMIT = 64 * 1024**2


class Mit48OCREngine(OCREngine):
    def __init__(
        self,
        variant: str,
        device: str,
        source_lang_english: str,
    ) -> None:
        self.variant = str(variant)
        self.device = str(device)
        self.source_lang = language_codes.get(source_lang_english, "en")
        self.cancel_checker: Callable[[], bool] | None = None
        self.inference_lease_factory: Callable[[], Any] | None = None
        self.last_page_profile: dict[str, Any] = {}
        self.last_worker_release_report: dict[str, Any] = {}
        self.model_root = Path(models_base_dir) / "ocr" / "mit48"
        self.model_identity = (
            MIT48_AR_MODEL_IDENTITY
            if self.variant == "ar"
            else MIT48_CTC_MODEL_IDENTITY
        )

    def initialize(self, **_kwargs) -> None:
        if self.variant not in {"ar", "ctc"}:
            raise ValueError(f"Unsupported MIT 48px variant: {self.variant}")
        if self.device not in {"cpu", "cuda", "mps", "xpu"}:
            raise ValueError(f"Unsupported MIT 48px device: {self.device}")
        model_ids = (
            (ModelID.MIT48_AR, ModelID.MIT48_SOURCE)
            if self.variant == "ar"
            else (
                ModelID.MIT48_CTC_ARCHIVE,
                ModelID.MIT48_CTC,
                ModelID.MIT48_SOURCE,
            )
        )
        for model_id in model_ids:
            try:
                require_prepared_model(model_id)
            except ModelNotPreparedError as exc:
                raise RuntimeError(
                    f"MIT 48px {self.variant.upper()} is not prepared. "
                    "Run the matching setup_full BAT before page work."
                ) from exc

    def set_cancel_checker(
        self,
        cancel_checker: Callable[[], bool] | None,
    ) -> None:
        self.cancel_checker = cancel_checker

    def set_inference_lease_factory(
        self,
        factory: Callable[[], Any] | None,
    ) -> None:
        self.inference_lease_factory = factory

    def _is_cancelled(self) -> bool:
        try:
            return bool(self.cancel_checker and self.cancel_checker())
        except Exception:
            return True

    def _raise_if_cancelled(self) -> None:
        if self._is_cancelled():
            raise OperationCancelledError("MIT 48px OCR was cancelled.")

    @staticmethod
    def _json_geometry(value: Any) -> Any:
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, (list, tuple)):
            return [Mit48OCREngine._json_geometry(item) for item in value]
        if isinstance(value, (np.integer, np.floating)):
            return value.item()
        return value

    def _worker_blocks(self, blocks: list[TextBlock]) -> list[dict[str, Any]]:
        payloads: list[dict[str, Any]] = []
        seen_ids: set[str] = set()
        for block in blocks:
            block_id = str(getattr(block, "block_id", "") or "").strip()
            if not block_id:
                block_id = new_text_block_id()
            if block_id in seen_ids:
                raise RuntimeError("MIT 48px OCR received duplicate text block IDs.")
            seen_ids.add(block_id)
            payloads.append(
                {
                    "block_id": block_id,
                    "xyxy": self._json_geometry(getattr(block, "xyxy", None)),
                    "lines": self._json_geometry(getattr(block, "lines", None)),
                    "segment_points": self._json_geometry(
                        getattr(block, "segm_pts", None)
                    ),
                    "direction": str(getattr(block, "direction", "") or ""),
                    "source_lang": str(
                        getattr(block, "source_lang", "") or self.source_lang
                    ),
                }
            )
        return payloads

    def _inference_lease(self):
        factory = self.inference_lease_factory
        if callable(factory):
            context = factory()

            @contextmanager
            def track_release_report():
                lease_state = None
                try:
                    with context as lease_state:
                        yield lease_state
                finally:
                    report = getattr(
                        lease_state if lease_state is not None else context,
                        "release_report",
                        None,
                    )
                    if isinstance(report, dict):
                        self.last_worker_release_report = dict(report)

            return track_release_report()
        if self.device != "cpu":
            raise RuntimeError(
                "MIT 48px GPU inference requires the shared runtime resource arbiter."
            )
        return nullcontext(None)

    def _run_worker(
        self,
        image: np.ndarray,
        blocks: list[dict[str, Any]],
    ) -> dict[str, Any]:
        command = [
            sys.executable,
            "-m",
            "modules.ocr.mit48_worker",
            "--variant",
            self.variant,
            "--device",
            self.device,
            "--model-root",
            str(self.model_root),
        ]
        creationflags = (
            getattr(subprocess, "CREATE_NO_WINDOW", 0)
            if os.name == "nt"
            else 0
        )
        request = pickle.dumps(
            {
                "variant": self.variant,
                "device": self.device,
                "source_language": self.source_lang,
                "image": image,
                "blocks": blocks,
            },
            protocol=pickle.HIGHEST_PROTOCOL,
        )
        if len(request) > _REQUEST_SIZE_LIMIT:
            raise ValueError("MIT 48px OCR request exceeds the in-memory IPC limit.")
        messages: queue.Queue[Any] = queue.Queue()
        stderr_lines: deque[str] = deque(maxlen=32)
        stderr_lock = threading.Lock()
        process: subprocess.Popen[bytes] | None = None
        stdout_thread: threading.Thread | None = None
        stderr_thread: threading.Thread | None = None
        request_thread: threading.Thread | None = None
        request_done = threading.Event()
        request_errors: list[BaseException] = []
        control_threads: list[threading.Thread] = []
        pending_result: dict[str, Any] | None = None
        completed_response: dict[str, Any] | None = None
        worker_error: str | None = None
        lease_context = self._inference_lease()
        pre_stop_captured = False
        worker_started = False
        graceful_shutdown = False

        def start_stdin_write(
            value: bytes,
        ) -> tuple[threading.Thread, threading.Event, list[BaseException]]:
            if process is None or process.stdin is None or process.stdin.closed:
                raise BrokenPipeError("MIT 48px worker stdin is unavailable.")
            stream = process.stdin
            finished = threading.Event()
            errors: list[BaseException] = []

            def write() -> None:
                try:
                    remaining = memoryview(value)
                    while remaining:
                        written = stream.write(remaining)
                        if not written:
                            raise BrokenPipeError("MIT 48px worker stdin stopped accepting data.")
                        remaining = remaining[written:]
                    stream.flush()
                except BaseException as exc:
                    errors.append(exc)
                finally:
                    finished.set()

            thread = threading.Thread(target=write, daemon=True)
            thread.start()
            return thread, finished, errors

        def send_control(kind: str) -> bool:
            if process is None or process.stdin is None or process.stdin.closed:
                return False
            thread, finished, errors = start_stdin_write(
                json.dumps({"type": kind}, separators=(",", ":")).encode("utf-8")
                + b"\n"
            )
            control_threads.append(thread)
            if not finished.wait(_WORKER_CONTROL_WRITE_TIMEOUT_SEC):
                return False
            return not errors

        def terminate_worker() -> None:
            if process is None or process.poll() is not None:
                return
            try:
                process.terminate()
            except OSError:
                pass
            try:
                process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                try:
                    process.kill()
                except OSError:
                    pass
                try:
                    process.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    return

        def capture_pre_stop() -> None:
            nonlocal pre_stop_captured
            if pre_stop_captured or process is None or process.poll() is not None:
                return
            marker = getattr(lease_state, "capture_pre_stop", None)
            if callable(marker):
                marker()
            pre_stop_captured = True

        def mark_worker_reaped() -> None:
            marker = getattr(lease_state, "mark_worker_reaped", None)
            if callable(marker):
                marker()

        def cleanup_worker() -> None:
            if process is None:
                return
            if process.poll() is None:
                capture_pre_stop()
                if (
                    not graceful_shutdown
                    or not request_done.is_set()
                    or request_errors
                    or not send_control("shutdown")
                ):
                    terminate_worker()
                elif process.stdin is not None and not process.stdin.closed:
                    process.stdin.close()
                    try:
                        process.wait(timeout=2.0)
                    except subprocess.TimeoutExpired:
                        terminate_worker()
            if process.stdin is not None and not process.stdin.closed:
                process.stdin.close()
            if process.poll() is not None and worker_started:
                mark_worker_reaped()
            if stdout_thread is not None:
                stdout_thread.join(timeout=1.0)
            if stderr_thread is not None:
                stderr_thread.join(timeout=1.0)
            if request_thread is not None:
                request_thread.join(timeout=1.0)
            for thread in control_threads:
                thread.join(timeout=1.0)
            if process.stdout is not None:
                process.stdout.close()
            if process.stderr is not None:
                process.stderr.close()
            if process.poll() is None:
                raise TimeoutError(
                    "MIT 48px worker did not exit before the termination deadline."
                )
            if any(
                thread is not None and thread.is_alive()
                for thread in (stdout_thread, stderr_thread, request_thread, *control_threads)
            ):
                raise RuntimeError("MIT 48px worker IPC thread did not stop after child exit.")

        with lease_context as lease_state:
            try:
                process = subprocess.Popen(
                    command,
                    cwd=_ROOT,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    bufsize=0,
                    creationflags=creationflags,
                )
                worker_started = True
                marker = getattr(lease_state, "mark_worker_started", None)
                if callable(marker):
                    marker(process.pid)

                def read_stdout() -> None:
                    assert process is not None and process.stdout is not None
                    try:
                        while True:
                            raw_line = process.stdout.readline(
                                _WORKER_RESPONSE_LINE_LIMIT + 1
                            )
                            if not raw_line:
                                break
                            if len(raw_line) > _WORKER_RESPONSE_LINE_LIMIT:
                                messages.put({
                                    "type": "protocol_error",
                                    "error": "MIT 48px worker response exceeded its line limit.",
                                })
                                break
                            try:
                                payload = json.loads(raw_line.decode("utf-8"))
                            except (UnicodeDecodeError, json.JSONDecodeError):
                                payload = {
                                    "type": "protocol_error",
                                    "error": "MIT 48px worker returned invalid JSON.",
                                }
                            messages.put(payload)
                    finally:
                        messages.put(None)

                def read_stderr() -> None:
                    assert process is not None and process.stderr is not None
                    for raw_line in iter(
                        lambda: process.stderr.readline(4096),
                        b"",
                    ):
                        line = raw_line.decode("utf-8", errors="replace").rstrip()
                        with stderr_lock:
                            stderr_lines.append(line)

                stdout_thread = threading.Thread(target=read_stdout, daemon=True)
                stderr_thread = threading.Thread(target=read_stderr, daemon=True)
                stdout_thread.start()
                stderr_thread.start()
                request_thread, request_done, request_errors = start_stdin_write(
                    len(request).to_bytes(8, "big") + request
                )
                request_deadline = time.monotonic() + _WORKER_REQUEST_WRITE_TIMEOUT_SEC
                response_deadline: float | None = None

                saw_eof = False
                model_loaded = False
                while pending_result is None and worker_error is None:
                    self._raise_if_cancelled()
                    now = time.monotonic()
                    if not request_done.is_set():
                        if now >= request_deadline:
                            raise TimeoutError(
                                "MIT 48px worker request write stalled."
                            )
                    elif request_errors:
                        raise RuntimeError(
                            f"MIT 48px worker request write failed: {request_errors[0]}"
                        ) from request_errors[0]
                    else:
                        if response_deadline is None:
                            response_deadline = now + _WORKER_RESPONSE_STALL_TIMEOUT_SEC
                        if now >= response_deadline:
                            raise TimeoutError(
                                "MIT 48px worker response stalled."
                            )
                    wait_timeout = _PROCESS_POLL_SEC
                    if not request_done.is_set():
                        wait_timeout = min(wait_timeout, max(0.0, request_deadline - now))
                    elif response_deadline is not None:
                        wait_timeout = min(wait_timeout, max(0.0, response_deadline - now))
                    try:
                        message = messages.get(timeout=wait_timeout)
                    except queue.Empty:
                        if process.poll() is not None and messages.empty():
                            break
                        continue
                    if message is None:
                        saw_eof = True
                        break
                    if not isinstance(message, dict):
                        worker_error = "MIT 48px worker returned a non-object message."
                        break
                    if not request_done.is_set():
                        while not request_done.is_set():
                            self._raise_if_cancelled()
                            remaining = request_deadline - time.monotonic()
                            if remaining <= 0.0:
                                break
                            request_done.wait(min(_PROCESS_POLL_SEC, remaining))
                    if request_errors:
                        worker_error = "MIT 48px worker responded before receiving its request."
                        break
                    if not request_done.is_set():
                        worker_error = "MIT 48px worker responded before receiving its request."
                        break
                    response_deadline = time.monotonic() + _WORKER_RESPONSE_STALL_TIMEOUT_SEC
                    message_type = str(message.get("type") or "")
                    if message_type == "loaded":
                        if model_loaded:
                            worker_error = "MIT 48px worker sent duplicate load messages."
                            break
                        runtime = message.get("runtime")
                        if not self._valid_runtime_identity(runtime, message):
                            worker_error = (
                                "MIT 48px worker reported an invalid runtime identity."
                            )
                            break
                        marker = getattr(lease_state, "mark_model_loaded", None)
                        if callable(marker):
                            marker(runtime)
                        model_loaded = True
                    elif message_type == "result":
                        if not model_loaded:
                            worker_error = "MIT 48px worker returned a result before load."
                            break
                        pending_result = message
                        graceful_shutdown = True
                    elif message_type == "error":
                        worker_error = str(message.get("error") or "worker error")
                        traceback_text = str(message.get("traceback") or "")
                        if traceback_text:
                            worker_error = f"{worker_error}\n{traceback_text}"
                    else:
                        worker_error = str(
                            message.get("error") or "MIT 48px worker protocol failed."
                        )

                if pending_result is None and worker_error is None:
                    detail = "MIT 48px worker exited without a result."
                    if saw_eof:
                        detail += " Its response stream closed early."
                    with stderr_lock:
                        diagnostic = "\n".join(stderr_lines)
                    if diagnostic:
                        detail += "\n" + diagnostic[-4000:]
                    raise RuntimeError(detail)
                if worker_error is not None:
                    raise RuntimeError(f"MIT 48px worker failed: {worker_error}")
                completed_response = pending_result
            finally:
                cleanup_worker()

        report = getattr(lease_state, "release_report", None)
        self.last_worker_release_report = (
            dict(report) if isinstance(report, dict) else {}
        )

        if process is not None and process.returncode != 0:
            with stderr_lock:
                detail = "\n".join(stderr_lines)
            if detail:
                logger.error("MIT 48px worker stderr: %s", detail[-4000:])
            raise RuntimeError(
                f"MIT 48px worker exited with code {process.returncode}."
            )
        if completed_response is None:
            raise RuntimeError("MIT 48px worker produced no validated result.")
        return completed_response

    def _valid_runtime_identity(
        self,
        runtime: Any,
        message: dict[str, Any],
    ) -> bool:
        if not isinstance(runtime, dict):
            return False
        runtime_pid = runtime.get("worker_pid")
        message_pid = message.get("worker_pid", runtime_pid)
        return bool(
            runtime.get("identity") == MIT48_RUNTIME_IDENTITY
            and runtime.get("device") == self.device
            and isinstance(runtime.get("torch_version"), str)
            and runtime.get("torch_version", "").strip()
            and isinstance(runtime_pid, int)
            and not isinstance(runtime_pid, bool)
            and runtime_pid > 0
            and isinstance(message_pid, int)
            and not isinstance(message_pid, bool)
            and message_pid == runtime_pid
        )

    @staticmethod
    def _validate_response(
        response: Any,
        expected_ids: list[str],
        *,
        device: str,
        source_lang: str,
        image_shape: tuple[int, int],
    ) -> tuple[list[dict[str, Any]], dict[str, Any], int]:
        if not isinstance(response, dict) or response.get("type") != "result":
            raise RuntimeError("MIT 48px worker returned an unexpected response.")
        runtime = response.get("runtime")
        if not isinstance(runtime, dict):
            raise RuntimeError("MIT 48px worker did not report its runtime identity.")
        runtime_pid = runtime.get("worker_pid")
        worker_pid = response.get("worker_pid")
        if (
            runtime.get("identity") != MIT48_RUNTIME_IDENTITY
            or runtime.get("device") != device
            or not isinstance(runtime.get("torch_version"), str)
            or not runtime["torch_version"].strip()
            or isinstance(runtime_pid, bool)
            or not isinstance(runtime_pid, int)
            or runtime_pid <= 0
            or isinstance(worker_pid, bool)
            or not isinstance(worker_pid, int)
            or worker_pid != runtime_pid
        ):
            raise RuntimeError("MIT 48px worker returned an invalid runtime identity.")
        results = response.get("blocks")
        if not isinstance(results, list) or len(results) != len(expected_ids):
            raise RuntimeError("MIT 48px worker returned an incomplete block list.")
        if any(not isinstance(item, dict) for item in results):
            raise RuntimeError("MIT 48px worker returned an invalid block record.")
        if any(not isinstance(item.get("block_id"), str) for item in results):
            raise RuntimeError("MIT 48px worker returned an invalid block identity.")
        actual_ids = [item["block_id"] for item in results]
        if actual_ids != expected_ids or len(set(actual_ids)) != len(actual_ids):
            raise RuntimeError("MIT 48px worker changed text block identity or order.")
        image_height, image_width = image_shape
        normalized_results: list[dict[str, Any]] = []
        for result in results:
            if not isinstance(result.get("text"), str):
                raise RuntimeError("MIT 48px worker returned non-text OCR output.")
            if not isinstance(result.get("empty_reason", ""), str):
                raise RuntimeError("MIT 48px worker returned an invalid empty reason.")
            confidence = result.get("confidence")
            if (
                isinstance(confidence, bool)
                or not isinstance(confidence, (int, float))
                or not math.isfinite(float(confidence))
                or not 0.0 <= float(confidence) <= 1.0
            ):
                raise RuntimeError("MIT 48px worker returned invalid confidence.")
            regions = result.get("regions")
            if not isinstance(regions, list):
                raise RuntimeError("MIT 48px worker returned invalid region provenance.")
            normalized_regions: list[dict[str, Any]] = []
            previous_line_index = -1
            for region in regions:
                if not isinstance(region, dict):
                    raise RuntimeError("MIT 48px worker returned an invalid region.")
                line_index = region.get("line_index")
                region_confidence = region.get("confidence")
                bbox = region.get("bbox")
                if (
                    isinstance(line_index, bool)
                    or not isinstance(line_index, int)
                    or line_index <= previous_line_index
                    or not isinstance(region.get("text"), str)
                    or region.get("geometry_source")
                    not in {
                        "native_line_quad",
                        "native_segment_quad",
                        "bbox_projection_v1",
                    }
                    or line_index != len(normalized_regions)
                    or not isinstance(region.get("empty_reason", ""), str)
                    or isinstance(region_confidence, bool)
                    or not isinstance(region_confidence, (int, float))
                    or not math.isfinite(float(region_confidence))
                    or not 0.0 <= float(region_confidence) <= 1.0
                    or not isinstance(bbox, list)
                    or len(bbox) != 4
                    or any(
                        isinstance(value, bool) or not isinstance(value, int)
                        for value in bbox
                    )
                    or bbox[0] < 0
                    or bbox[1] < 0
                    or bbox[2] <= bbox[0]
                    or bbox[3] <= bbox[1]
                    or bbox[2] > image_width
                    or bbox[3] > image_height
                ):
                    raise RuntimeError("MIT 48px worker returned invalid region data.")
                previous_line_index = line_index
                normalized_regions.append(
                    {
                        "line_index": line_index,
                        "text": region["text"],
                        "confidence": float(region_confidence),
                        "bbox": list(bbox),
                        "geometry_source": region["geometry_source"],
                        "empty_reason": region.get("empty_reason", ""),
                    }
                )
            fragments = [region["text"] for region in normalized_regions if region["text"]]
            expected_text = ("" if is_no_space_lang(source_lang) else " ").join(fragments)
            expected_confidence = (
                sum(region["confidence"] for region in normalized_regions)
                / len(normalized_regions)
                if normalized_regions
                else 0.0
            )
            if result["text"] != expected_text or not math.isclose(
                float(confidence), expected_confidence, rel_tol=0.0, abs_tol=1e-6
            ):
                raise RuntimeError("MIT 48px worker returned inconsistent block data.")
            normalized_results.append(
                {
                    "block_id": result["block_id"],
                    "text": result["text"],
                    "confidence": float(confidence),
                    "empty_reason": result.get("empty_reason", ""),
                    "regions": normalized_regions,
                }
            )
        return normalized_results, runtime, runtime_pid

    def process_image(
        self,
        img: np.ndarray,
        blk_list: list[TextBlock],
    ) -> list[TextBlock]:
        self._raise_if_cancelled()
        if not blk_list:
            return blk_list
        image = np.asarray(img)
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("MIT 48px OCR requires a uint8 RGB image.")
        blocks = self._worker_blocks(blk_list)
        ids = [block["block_id"] for block in blocks]
        response = self._run_worker(np.ascontiguousarray(image), blocks)
        self._raise_if_cancelled()
        results, runtime, worker_pid = self._validate_response(
            response,
            ids,
            device=self.device,
            source_lang=self.source_lang,
            image_shape=image.shape[:2],
        )
        self._raise_if_cancelled()

        updates: list[dict[str, Any]] = []
        for result in results:
            text = result["text"]
            regions = result["regions"]
            boxes = [region["bbox"] for region in regions]
            updates.append(
                {
                    "text": text,
                    "reason": result.get("empty_reason", ""),
                    "confidence": result["confidence"],
                    "texts": [region["text"] for region in regions if region["text"]],
                    "regions": regions,
                    "geometry_sources": sorted(
                        {region["geometry_source"] for region in regions}
                    ),
                    "crop_bbox": (
                        [
                            min(box[0] for box in boxes),
                            min(box[1] for box in boxes),
                            max(box[2] for box in boxes),
                            max(box[3] for box in boxes),
                        ]
                        if boxes
                        else None
                    ),
                }
            )

        for block, block_id, update in zip(blk_list, ids, updates):
            if not str(getattr(block, "block_id", "") or "").strip():
                block.block_id = block_id
            initialize_ocr_result_contract(block)
            text = update["text"]
            set_block_ocr_diagnostics(
                block,
                text=text,
                confidence=update["confidence"],
                status=OCR_STATUS_OK if text else OCR_STATUS_EMPTY_INITIAL,
                empty_reason=update["reason"],
                attempt_count=1,
                raw_text=text,
                sanitized_text=text,
            )
            block.texts = update["texts"]
            block.ocr_regions = update["regions"]
            if update["crop_bbox"] is not None:
                block.ocr_crop_bbox = update["crop_bbox"]
            block.ocr_strategy = f"mit48_{self.variant}"
            block.ocr_model_identity = self.model_identity
            block.ocr_runtime_identity = (
                f"{MIT48_RUNTIME_IDENTITY}:torch={runtime.get('torch_version', '')}:"
                f"device={runtime.get('device', self.device)}"
            )
            if update["reason"] == "geometry_review":
                assign_ocr_processing_contract(
                    block,
                    semantic_role=SEMANTIC_ROLE_AMBIGUOUS,
                    processing_action=PROCESSING_ACTION_REVIEW,
                    decision_source="mit48_geometry_review",
                    reasons=["mit48_line_geometry_unresolved"],
                )

        self.last_page_profile = {
            "variant": self.variant,
            "device": self.device,
            "model_identity": self.model_identity,
            "runtime_identity": MIT48_RUNTIME_IDENTITY,
            "block_count": len(blk_list),
            "non_empty_blocks": sum(bool(result["text"]) for result in results),
            "worker_pid": worker_pid,
            "geometry_sources": sorted(
                {
                    source
                    for update in updates
                    for source in update["geometry_sources"]
                }
            ),
            "gpu_release": dict(self.last_worker_release_report),
        }
        return blk_list

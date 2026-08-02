#!/usr/bin/env python3
"""Lab-only one-router runtime adapters for managed OCR plus Gemma.

Nothing in this module is imported by the product.  It replaces the two
already-created product managers on one offscreen benchmark window, keeps the
existing ``RuntimeResourceArbiter`` ownership boundary, and starts only a
uniquely named router container.  The router itself stays alive across the
pipeline while each model is explicitly loaded and unloaded.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import threading
import time
from typing import Any, Callable, Iterator, Mapping
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlsplit
from urllib.request import Request, urlopen

from modules.ocr.local_runtime import LocalOCRRuntimeManager
from modules.translation.local_runtime import LocalGemmaRuntimeManager
from modules.utils.exceptions import OperationCancelledError
from modules.utils.gpu_metrics import query_cuda_handoff_metrics, query_gpu_metrics
from modules.utils.llama_cpp_runtime import resolve_docker_compose_command


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_VERSION = "llamacpp-router-handoff-v1"
PINNED_LLAMA_IMAGE = (
    "ghcr.io/ggml-org/llama.cpp@sha256:"
    "22e0e3bfe967af4fd1df6a918022abbfd4e72e4d40a4769e616a4176790acbcb"
)
PINNED_LLAMA_REVISION = "ff067f76dd8e9e05f0528056f1274adf01a54d70"
PINNED_LLAMA_BUILD = "b10133"
GEMMA_ALIAS = "gemma-4-26B-IQ4_NL.gguf"
GEMMA_VOLUME = "comic-translate-gemma-models-v2"
GEMMA_SHA256 = "768a89b94209243b333b2e074b928fe51ea208ebdad6424a510bd73e5cb4d0b8"
LAB_CONTAINER_PREFIX = "ct-router-lab-"
LAB_LABEL_PROTOCOL = "com.comictranslate.benchmark-protocol"
LAB_LABEL_PAIR = "com.comictranslate.benchmark-pair"
LAB_LABEL_OWNER = "com.comictranslate.benchmark-owner"
MODEL_IDENTITY_HELPER_IMAGE = "alpine:3.22"
_SAFE_DOCKER_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class RouterHandoffLabError(RuntimeError):
    """Raised when a lab-only router contract cannot be proven."""


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True)
class RouterPair:
    key: str
    engine_key: str
    ocr_alias: str
    ocr_model: str
    ocr_mmproj: str
    ocr_model_sha256: str
    ocr_mmproj_sha256: str
    ocr_volume: str
    ocr_port: int
    compose_file: Path
    preset_file: Path
    source_lang: str
    baseline_bind_dir: Path | None = None

    def validate(self) -> None:
        if not self.key or not _SAFE_DOCKER_NAME.fullmatch(self.key):
            raise RouterHandoffLabError(f"Unsafe pair key: {self.key!r}")
        if not self.engine_key or not self.ocr_alias:
            raise RouterHandoffLabError("Router pair needs engine and alias.")
        if not _SAFE_DOCKER_NAME.fullmatch(self.ocr_volume):
            raise RouterHandoffLabError("Router OCR volume is unsafe.")
        if not 1024 <= int(self.ocr_port) <= 65535:
            raise RouterHandoffLabError("Router OCR port is invalid.")
        if not self.compose_file.is_file() or not self.preset_file.is_file():
            raise RouterHandoffLabError(
                f"Router pair assets are missing for {self.key}."
            )
        for value in (self.ocr_model_sha256, self.ocr_mmproj_sha256):
            if not _SHA256_RE.fullmatch(value):
                raise RouterHandoffLabError("Router model SHA must be lowercase hex.")


def pair_catalog() -> dict[str, RouterPair]:
    base = ROOT / "benchmarks" / "llamacpp_router_handoff"
    pairs = (
        RouterPair(
            key="paddle-crop",
            engine_key="PaddleOCR VL",
            ocr_alias="PaddleOCR-VL-1.6-0.9B",
            ocr_model="PaddleOCR-VL-1.6-GGUF.gguf",
            ocr_mmproj="PaddleOCR-VL-1.6-GGUF-mmproj.gguf",
            ocr_model_sha256=(
                "f3ae46ec885050acf4b3d31944431e1fd90d50664fb09126af4a3c050ba14ee8"
            ),
            ocr_mmproj_sha256=(
                "204d757d7610d9b3faab10d506d69e5b244e32bf765e2bab2d0167e65e0a058a"
            ),
            ocr_volume="comic-translate-paddleocr-vl-llamacpp-models-v1",
            ocr_port=18000,
            compose_file=base / "compose" / "paddle-crop.router.yaml",
            preset_file=base / "presets" / "paddle-crop.ini",
            source_lang="Japanese",
        ),
        RouterPair(
            key="paddle-spotting",
            engine_key="PaddleOCR VL Spotting",
            ocr_alias="PaddleOCR-VL-1.6-Spotting",
            ocr_model="PaddleOCR-VL-1.6-Spotting-GGUF.gguf",
            ocr_mmproj="PaddleOCR-VL-1.6-Spotting-mmproj.gguf",
            ocr_model_sha256=(
                "f3ae46ec885050acf4b3d31944431e1fd90d50664fb09126af4a3c050ba14ee8"
            ),
            ocr_mmproj_sha256=(
                "8e011479092c5e82c8c1c2d85d52b9ac48df12183c5c7bc3190190732259db09"
            ),
            ocr_volume="comic-translate-paddleocr-vl-spotting-llamacpp-models-v2",
            ocr_port=18002,
            compose_file=base / "compose" / "paddle-spotting.router.yaml",
            preset_file=base / "presets" / "paddle-spotting.ini",
            source_lang="Japanese",
        ),
        RouterPair(
            key="hunyuanocr",
            engine_key="HunyuanOCR",
            ocr_alias="HunyuanOCR.Q8_0.gguf",
            ocr_model="HunyuanOCR.Q8_0.gguf",
            ocr_mmproj="HunyuanOCR.mmproj-Q8_0.gguf",
            ocr_model_sha256=(
                "cdafc794cafeae377868d7a40a70e282a737e39abe77c0d8b73614447b364a21"
            ),
            ocr_mmproj_sha256=(
                "b77913164ff73d4c0dc4d994e236ed72bacbbe5c5db1ec9b2828627b46c32804"
            ),
            ocr_volume="comic-translate-hunyuanocr-models-v1",
            ocr_port=28080,
            compose_file=base / "compose" / "hunyuanocr.router.yaml",
            preset_file=base / "presets" / "hunyuanocr.ini",
            source_lang="Chinese",
            # Baseline still uses the legacy product bind mount.  The lab
            # verifies it byte-for-byte against the router named volume; it
            # does not alter that product Compose until a user-approved
            # promotion.
            baseline_bind_dir=ROOT / "testmodel",
        ),
        RouterPair(
            key="mangalmm",
            engine_key="MangaLMM",
            ocr_alias="MangaLMM.Q8_0.gguf",
            ocr_model="MangaLMM.Q8_0.gguf",
            ocr_mmproj="MangaLMM.mmproj-Q8_0.gguf",
            ocr_model_sha256=(
                "55e42d513ee22ab1a301b5fa8f04a2812b69d6b351e7d34efdff2b8d8e8fa01a"
            ),
            ocr_mmproj_sha256=(
                "24f43da26996b54bf5764177a954e49b24ec38a53de34d8231764747b0dcd8d7"
            ),
            ocr_volume="comic-translate-mangalmm-models-v2",
            ocr_port=28081,
            compose_file=base / "compose" / "mangalmm.router.yaml",
            preset_file=base / "presets" / "mangalmm.ini",
            source_lang="Japanese",
        ),
    )
    catalog = {pair.key: pair for pair in pairs}
    for pair in catalog.values():
        pair.validate()
    return catalog


def _run(
    command: list[str],
    *,
    env: Mapping[str, str] | None = None,
    cwd: Path | None = None,
    timeout: float = 120.0,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        command,
        cwd=str(cwd or ROOT),
        env=dict(env) if env is not None else None,
        capture_output=True,
        text=True,
        errors="replace",
        check=False,
        timeout=timeout,
    )
    if check and completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "").strip()
        raise RouterHandoffLabError(
            f"Command failed ({completed.returncode}): {' '.join(command[:3])}\n"
            f"{detail[-4096:]}"
        )
    return completed


def _gpu_used_mib() -> int | None:
    metrics = query_gpu_metrics()
    primary = metrics.get("primary") if isinstance(metrics, Mapping) else None
    try:
        return int(primary["memory_used_mb"])  # type: ignore[index]
    except (KeyError, TypeError, ValueError):
        return None


def _json_response(
    url: str,
    *,
    payload: Mapping[str, Any] | None = None,
    timeout: float = 30.0,
) -> tuple[int, dict[str, Any]]:
    body = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = Request(
        url,
        data=body,
        headers=headers,
        method="POST" if payload is not None else "GET",
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8", errors="replace")
            try:
                parsed = json.loads(raw) if raw else {}
            except json.JSONDecodeError:
                parsed = {"raw": raw}
            return int(response.status), parsed if isinstance(parsed, dict) else {"value": parsed}
    except HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        try:
            parsed = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            parsed = {"raw": raw}
        return int(exc.code), parsed if isinstance(parsed, dict) else {"value": parsed}
    except (URLError, OSError) as exc:
        raise RouterHandoffLabError(f"Router HTTP request failed: {url}: {exc}") from exc


def _text_response(url: str, *, timeout: float = 15.0) -> tuple[int, str]:
    request = Request(url, headers={"Accept": "text/plain"})
    try:
        with urlopen(request, timeout=timeout) as response:
            return int(response.status), response.read().decode("utf-8", errors="replace")
    except HTTPError as exc:
        return int(exc.code), exc.read().decode("utf-8", errors="replace")
    except (URLError, OSError) as exc:
        raise RouterHandoffLabError(f"Router endpoint failed: {url}: {exc}") from exc


def _models_by_alias(payload: Mapping[str, Any]) -> dict[str, str]:
    rows = payload.get("data")
    if not isinstance(rows, list):
        rows = payload.get("models")
    result: dict[str, str] = {}
    if isinstance(rows, Mapping):
        rows = [
            {"id": key, **(value if isinstance(value, Mapping) else {})}
            for key, value in rows.items()
        ]
    if not isinstance(rows, list):
        return result
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        alias = str(row.get("id") or row.get("name") or row.get("model") or "")
        # router mode exposes the state as ``value``.  Depending on the
        # image build it is either a peer of ``id`` or a nested object under
        # ``state``.  Preserve the alias key while accepting those two
        # documented response shapes; every other value remains unknown and
        # therefore fails closed.
        state_value: Any = row.get("state") or row.get("status") or row.get("value")
        if isinstance(state_value, Mapping):
            state_value = (
                state_value.get("value")
                or state_value.get("state")
                or state_value.get("status")
            )
        state = str(state_value or "unknown").lower()
        if alias:
            result[alias] = state
    return result


def _loaded_count(states: Mapping[str, str]) -> int:
    return sum(1 for state in states.values() if state != "unloaded")


def _is_safe_loopback_router_url(value: str, *, expected_port: int) -> bool:
    """Accept only the exact loopback OpenAI-style endpoint for one model.

    The adapter observes OCR requests that contain page image bytes.  A text
    substring check is not a routing boundary: ``127.0.0.1:18000@host`` is an
    external host.  Parse the authority and reject userinfo, non-loopback
    hosts, a mismatched port, and non-``/v1/`` API paths before a proxy ever
    forwards the request.
    """

    try:
        parsed = urlsplit(str(value or ""))
        port = parsed.port
    except ValueError:
        return False
    return bool(
        parsed.scheme == "http"
        and parsed.username is None
        and parsed.password is None
        and parsed.hostname == "127.0.0.1"
        and port == int(expected_port)
        and parsed.path.startswith("/v1/")
    )


@dataclass
class RouterLabSession:
    pair: RouterPair
    artifact_dir: Path
    container_name: str
    image_ref: str = PINNED_LLAMA_IMAGE
    gemma_port: int = 18080
    release_residual_mib: int = 128

    def __post_init__(self) -> None:
        self.pair.validate()
        if not self.container_name.startswith(LAB_CONTAINER_PREFIX):
            raise RouterHandoffLabError("Router lab container must use the lab prefix.")
        if not _SAFE_DOCKER_NAME.fullmatch(self.container_name):
            raise RouterHandoffLabError("Router lab container name is unsafe.")
        self._lock = threading.RLock()
        self._prepared = False
        self._active_alias = ""
        self._events: list[dict[str, Any]] = []
        self._last_states: dict[str, str] = {}
        self._pre_start_gpu_mib: int | None = None
        self._first_request: dict[str, float] = {}
        self._load_completed_at: dict[str, float] = {}
        self._captured_ocr_request: tuple[str, dict[str, Any]] | None = None
        self._round_trip_verified = False
        self._live_contract: dict[str, Any] = {}

    @property
    def endpoint(self) -> str:
        return f"http://127.0.0.1:{self.gemma_port}"

    @property
    def project_name(self) -> str:
        suffix = hashlib.sha256(self.container_name.encode("utf-8")).hexdigest()[:10]
        return f"ct-router-lab-{suffix}"

    @property
    def owner_token(self) -> str:
        """Return the arm-specific token required before container cleanup.

        A predictable container name is useful for diagnostics, but it is not
        ownership proof.  Bind the label to the managed private arm directory
        so a failed runner can clean up only the container it created.
        """

        return hashlib.sha256(
            str(self.artifact_dir.resolve()).encode("utf-8")
        ).hexdigest()[:16]

    @property
    def config_dir(self) -> Path:
        return self.artifact_dir / "router-config"

    def prepare_process(
        self,
        *,
        cancel_checker: Callable[[], bool] | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            if self._prepared:
                return {"status": "reused", "elapsed_ms": 0.0}
            self._raise_if_cancelled(cancel_checker)
            if self._container_exists():
                raise RouterHandoffLabError(
                    f"Refusing to reuse an existing lab container: {self.container_name}"
                )
            self.config_dir.mkdir(parents=True, exist_ok=True)
            preset_destination = self.config_dir / "models.ini"
            shutil.copyfile(self.pair.preset_file, preset_destination)
            self._pre_start_gpu_mib = _gpu_used_mib()
            started = time.perf_counter()
            try:
                _run(
                    [
                        *resolve_docker_compose_command(),
                        "-f",
                        str(self.pair.compose_file),
                        "--project-name",
                        self.project_name,
                        "up",
                        "-d",
                        "--no-build",
                    ],
                    env=self._compose_environment(),
                    cwd=self.pair.compose_file.parent,
                    timeout=90.0,
                )
                self._wait_router_ready(cancel_checker=cancel_checker)
                self._live_contract = self._verify_live_container_contract()
                states = self.model_states()
                if _loaded_count(states):
                    raise RouterHandoffLabError(
                        "Router loaded a model before the explicit load contract."
                )
                self._assert_unloaded_request_does_not_autoload(states)
            except BaseException:
                try:
                    # ``compose up`` can create the named container before
                    # returning a failure.  This cleanup is deliberately by
                    # exact owned name only; it never calls ``compose down``.
                    self.stop_process()
                except BaseException as cleanup_error:
                    raise RouterHandoffLabError(
                        "Router startup failed and the owned container could not be removed."
                    ) from cleanup_error
                raise
            self._prepared = True
            event = {
                "event": "router_prepare",
                "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 3),
                "preset_sha256": hashlib.sha256(preset_destination.read_bytes()).hexdigest(),
                "states": states,
            }
            self._events.append(event)
            return dict(event)

    def load_model(
        self,
        alias: str,
        *,
        cancel_checker: Callable[[], bool] | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            self.prepare_process(cancel_checker=cancel_checker)
            if self._active_alias == alias:
                states = self.model_states()
                if states.get(alias) == "loaded" and _loaded_count(states) == 1:
                    return {"status": "reused", "model": alias, "elapsed_ms": 0.0}
                self._active_alias = ""
            if self._active_alias:
                raise RouterHandoffLabError(
                    f"Cannot load {alias}; {self._active_alias} is still active."
                )
            self._raise_if_cancelled(cancel_checker)
            before = self.model_states()
            if _loaded_count(before):
                raise RouterHandoffLabError("Router has a pre-existing loaded model.")
            started = time.perf_counter()
            status, payload = _json_response(
                f"{self.endpoint}/models/load", payload={"model": alias}, timeout=90.0
            )
            if not 200 <= status < 300:
                raise RouterHandoffLabError(f"Router load failed for {alias}: HTTP {status}")
            states = self._wait_model_state(alias, expected="loaded", cancel_checker=cancel_checker)
            if _loaded_count(states) != 1:
                raise RouterHandoffLabError("Router loaded count is not exactly one.")
            self._verify_loaded_identity(alias)
            self._active_alias = alias
            self._load_completed_at[alias] = time.perf_counter()
            event = {
                "event": "router_load",
                "model": alias,
                "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 3),
                "response": payload,
                "states": states,
            }
            self._events.append(event)
            return dict(event)

    def unload_model(
        self,
        alias: str,
        *,
        cancel_checker: Callable[[], bool] | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            if not self._prepared:
                return {"runtime_state": "stopped", "gpu_release_expected": False}
            if self._active_alias and self._active_alias != alias:
                raise RouterHandoffLabError(
                    f"Cannot unload {alias}; active model is {self._active_alias}."
                )
            states = self.model_states()
            if alias not in states or states.get(alias) == "unloaded":
                self._active_alias = ""
                return {"runtime_state": "stopped", "gpu_release_expected": False}
            started = time.perf_counter()
            status, payload = _json_response(
                f"{self.endpoint}/models/unload", payload={"model": alias}, timeout=90.0
            )
            if not 200 <= status < 300:
                raise RouterHandoffLabError(f"Router unload failed for {alias}: HTTP {status}")
            states = self._wait_model_state(alias, expected="unloaded", cancel_checker=cancel_checker)
            if _loaded_count(states):
                raise RouterHandoffLabError("Router still has a loaded model after unload.")
            self._active_alias = ""
            event = {
                "event": "router_unload",
                "model": alias,
                "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 3),
                "response": payload,
                "states": states,
                "model_state_unloaded": True,
                # The product StageBatchedProcessor owns the authoritative
                # driver-global return gate.  It captured the per-service
                # baseline immediately before this model load, while this
                # router only knows its process start baseline (which also
                # includes RT-DETR/torch residency).  Returning ``sleeping``
                # means "server stays alive but the model is unloaded" and
                # invokes that existing 85-percent/512-MiB CUDA-context gate.
                "gpu_return_gate": "stage_batched_processor",
            }
            self._events.append(event)
            return {
                "runtime_state": "sleeping",
                "gpu_release_expected": True,
                "router_release": event,
            }

    def begin_http_request(self, alias: str) -> float:
        with self._lock:
            if alias != self._active_alias:
                raise RouterHandoffLabError(
                    f"HTTP request routed to {alias} while active model is {self._active_alias or 'none'}."
                )
            return time.perf_counter()

    def finish_http_request(
        self,
        alias: str,
        *,
        started: float,
        successful: bool,
    ) -> None:
        with self._lock:
            if alias != self._active_alias:
                raise RouterHandoffLabError(
                    f"HTTP request completed for {alias} while active model is {self._active_alias or 'none'}."
                )
            if not successful:
                self._events.append(
                    {
                        "event": "router_http_failure",
                        "model": alias,
                        "elapsed_ms": round(
                            max(0.0, (time.perf_counter() - started) * 1000.0),
                            3,
                        ),
                    }
                )
                return
            if alias not in self._first_request:
                request_ms = max(0.0, (time.perf_counter() - started) * 1000.0)
                load_to_request_ms = max(
                    0.0,
                    (started - self._load_completed_at.get(alias, started)) * 1000.0,
                )
                self._first_request[alias] = round(request_ms, 3)
                self._events.append(
                    {
                        "event": "router_first_request",
                        "model": alias,
                        "elapsed_ms": round(request_ms, 3),
                        "load_to_request_ms": round(load_to_request_ms, 3),
                    }
                )

    def capture_ocr_request(self, url: str, payload: Mapping[str, Any]) -> None:
        """Retain one lab-only OCR request for the required post-Gemma relay."""

        with self._lock:
            if self._captured_ocr_request is None:
                copied = json.loads(json.dumps(dict(payload), ensure_ascii=False))
                self._captured_ocr_request = (str(url), copied)
                self._events.append(
                    {
                        "event": "ocr_request_captured",
                        "payload_sha256": canonical_sha256(copied),
                    }
                )

    def execute_captured_ocr_round_trip_request(
        self,
        *,
        cancel_checker: Callable[[], bool] | None = None,
    ) -> None:
        """Run the real captured OCR request after the Arbiter acquired OCR.

        ``InstalledRouterHandoffLabRuntime`` deliberately owns the surrounding
        model-start/model-release and GPU-return proof.  Keeping this method to
        the HTTP request only prevents a post-E2E relay from bypassing the
        product Arbiter.
        """

        self._raise_if_cancelled(cancel_checker)
        with self._lock:
            if self._active_alias != self.pair.ocr_alias:
                raise RouterHandoffLabError(
                    "OCR re-load probe requires the OCR model to be explicitly loaded."
                )
            if self._captured_ocr_request is None:
                raise RouterHandoffLabError(
                    "OCR re-load probe has no captured real OCR request."
                )
            url, payload = self._captured_ocr_request
            if not _is_safe_loopback_router_url(
                url,
                expected_port=self.pair.ocr_port,
            ):
                raise RouterHandoffLabError("Captured OCR request is not a safe router endpoint.")
            payload = json.loads(json.dumps(payload, ensure_ascii=False))
        started = self.begin_http_request(self.pair.ocr_alias)
        try:
            import requests

            response = requests.post(url, json=payload, timeout=180)
        except BaseException:
            self.finish_http_request(
                self.pair.ocr_alias,
                started=started,
                successful=False,
            )
            raise
        successful = 200 <= int(response.status_code) < 300
        self.finish_http_request(
            self.pair.ocr_alias,
            started=started,
            successful=successful,
        )
        # ``requests`` cannot be interrupted safely from another thread, but
        # a cancellation raised while it was in flight must still be observed
        # before the caller decides whether the relay passed.  The caller's
        # ``finally`` then performs the unconditional unload/VRAM proof.
        self._raise_if_cancelled(cancel_checker)
        if not successful:
            raise RouterHandoffLabError(
                "OCR re-load request failed: HTTP " f"{response.status_code}"
            )

    def mark_ocr_round_trip_verified(
        self,
        *,
        elapsed_ms: float,
        gpu_return_gate: Mapping[str, Any],
    ) -> dict[str, Any]:
        with self._lock:
            self._round_trip_verified = True
            event = {
                "event": "ocr_reload_request_unload",
                "elapsed_ms": round(max(0.0, elapsed_ms), 3),
                "verified": True,
                "gpu_return_gate": dict(gpu_return_gate),
            }
            self._events.append(event)
            return dict(event)

    def record_command_queue_wait(self, operation: str, started: float) -> None:
        with self._lock:
            self._events.append(
                {
                    "event": "arbiter_command_enter",
                    "operation": operation,
                    "queue_wait_ms": round(
                        max(0.0, (time.perf_counter() - started) * 1000.0),
                        3,
                    ),
                }
            )

    def model_states(self) -> dict[str, str]:
        status, payload = _json_response(f"{self.endpoint}/models")
        if not 200 <= status < 300:
            raise RouterHandoffLabError(f"Router /models failed: HTTP {status}")
        states = _models_by_alias(payload)
        if not states:
            raise RouterHandoffLabError("Router /models returned no model states.")
        supported_states = {"loaded", "unloaded", "loading", "unloading"}
        unexpected = {
            alias: state
            for alias, state in states.items()
            if state not in supported_states
        }
        if unexpected:
            raise RouterHandoffLabError(
                f"Router /models returned unsupported states: {unexpected}"
            )
        self._last_states = dict(states)
        return states

    def stop_process(self) -> dict[str, Any]:
        with self._lock:
            started = time.perf_counter()
            labels = self._container_labels()
            if labels is None:
                self._prepared = False
                self._active_alias = ""
                event = {
                    "event": "router_process_stop",
                    "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 3),
                    "attempted": False,
                    "orphan": False,
                    "ownership_verified": True,
                    "gpu_return_gate": "already-verified-by-stage",
                }
                self._events.append(event)
                return event
            if not self._labels_match_owner(labels):
                event = {
                    "event": "router_process_stop_ownership_unverified",
                    "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 3),
                    "attempted": False,
                    "orphan": False,
                    "ownership_verified": False,
                }
                self._events.append(event)
                raise RouterHandoffLabError(
                    "Router lab cleanup refuses a container without this arm's ownership labels."
                )
            active = self._active_alias
            if active:
                try:
                    self.unload_model(active)
                except Exception as exc:
                    self._events.append({"event": "unload_during_stop_failed", "error": str(exc)})
            _run(["docker", "stop", "--timeout", "10", self.container_name], check=False, timeout=30.0)
            _run(["docker", "rm", self.container_name], check=False, timeout=30.0)
            self._prepared = False
            self._active_alias = ""
            orphan = self._container_exists()
            event = {
                "event": "router_process_stop",
                "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 3),
                "attempted": True,
                "orphan": orphan,
                "ownership_verified": True,
                "gpu_return_gate": "already-verified-by-stage",
            }
            self._events.append(event)
            if orphan:
                raise RouterHandoffLabError("Router lab container orphan remains after cleanup.")
            return event

    def evidence(self) -> dict[str, Any]:
        inspection = _run(
            ["docker", "inspect", self.container_name, "--format", "{{json .}}"],
            check=False,
            timeout=20.0,
        )
        return {
            "protocol_version": PROTOCOL_VERSION,
            "pair": self.pair.key,
            "image": self.image_ref,
            "image_revision": PINNED_LLAMA_REVISION,
            "image_build": PINNED_LLAMA_BUILD,
            "container_name": self.container_name,
            "active_alias": self._active_alias,
            "last_states": dict(self._last_states),
            "first_request_ms": dict(self._first_request),
            "ocr_round_trip_verified": self._round_trip_verified,
            "events": list(self._events),
            "live_contract": dict(self._live_contract),
            "container_present": inspection.returncode == 0,
        }

    def _compose_environment(self) -> dict[str, str]:
        environment = os.environ.copy()
        environment.update(
            {
                "LLAMA_CPP_IMAGE": self.image_ref,
                "ROUTER_CONTAINER_NAME": self.container_name,
                "ROUTER_PROTOCOL_VERSION": PROTOCOL_VERSION,
                "ROUTER_OWNER_TOKEN": self.owner_token,
                "ROUTER_CONFIG_DIR": str(self.config_dir.resolve()),
                "OCR_HOST_PORT": str(self.pair.ocr_port),
                "GEMMA_HOST_PORT": str(self.gemma_port),
                "OCR_MODEL_VOLUME": self.pair.ocr_volume,
                "GEMMA_MODEL_VOLUME": GEMMA_VOLUME,
            }
        )
        return environment

    def _container_exists(self) -> bool:
        completed = _run(
            ["docker", "inspect", self.container_name], check=False, timeout=20.0
        )
        return completed.returncode == 0

    def _container_labels(self) -> dict[str, str] | None:
        """Return labels for this exact name, without claiming ownership.

        ``None`` means the container is absent.  Any malformed inspect result
        is an error rather than a reason to issue a broad cleanup command.
        """

        completed = _run(
            ["docker", "inspect", self.container_name, "--format", "{{json .Config.Labels}}"],
            check=False,
            timeout=20.0,
        )
        if completed.returncode != 0:
            return None
        try:
            labels = json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            raise RouterHandoffLabError("Router lab container labels were invalid.") from exc
        if not isinstance(labels, Mapping):
            raise RouterHandoffLabError("Router lab container labels were not an object.")
        return {str(key): str(value) for key, value in labels.items()}

    def _labels_match_owner(self, labels: Mapping[str, str]) -> bool:
        return (
            labels.get(LAB_LABEL_PROTOCOL) == PROTOCOL_VERSION
            and labels.get(LAB_LABEL_PAIR) == self.pair.key
            and labels.get(LAB_LABEL_OWNER) == self.owner_token
        )

    def _verify_live_container_contract(self) -> dict[str, Any]:
        inspected = _run(
            ["docker", "inspect", self.container_name, "--format", "{{json .}}"],
            timeout=20.0,
        )
        try:
            payload = json.loads(inspected.stdout)
        except json.JSONDecodeError as exc:
            raise RouterHandoffLabError("Router live container inspect was invalid.") from exc
        if not isinstance(payload, Mapping):
            raise RouterHandoffLabError("Router live container inspect was not an object.")
        config = payload.get("Config") if isinstance(payload.get("Config"), Mapping) else {}
        labels = config.get("Labels") if isinstance(config.get("Labels"), Mapping) else {}
        if not self._labels_match_owner(
            {str(key): str(value) for key, value in labels.items()}
        ):
            raise RouterHandoffLabError("Router live container labels do not match the lab contract.")
        host_config = (
            payload.get("HostConfig")
            if isinstance(payload.get("HostConfig"), Mapping)
            else {}
        )
        if host_config.get("NetworkMode") != "bridge":
            raise RouterHandoffLabError(
                "Router lab must use Docker's built-in bridge, not a per-arm Compose network."
            )
        command = [str(item) for item in (config.get("Cmd") or [])]
        required = ["--models-max", "1", "--no-models-autoload", "--models-preset", "/config/models.ini"]
        if not all(item in command for item in required):
            raise RouterHandoffLabError("Router live command does not match models-max/no-autoload contract.")
        mounts = payload.get("Mounts") if isinstance(payload.get("Mounts"), list) else []
        by_destination = {
            str(item.get("Destination") or ""): item
            for item in mounts
            if isinstance(item, Mapping)
        }
        expected = {
            "/models/ocr": self.pair.ocr_volume,
            "/models/gemma": GEMMA_VOLUME,
            "/config": None,
        }
        for destination, source_name in expected.items():
            mount = by_destination.get(destination)
            if not isinstance(mount, Mapping) or mount.get("RW") is not False:
                raise RouterHandoffLabError(
                    f"Router {destination} is missing or writable."
                )
            if source_name is not None and str(mount.get("Name") or "") != source_name:
                raise RouterHandoffLabError(
                    f"Router {destination} uses an unexpected volume."
                )
        image = _run(
            ["docker", "image", "inspect", self.image_ref, "--format", "{{json .}}"],
            timeout=20.0,
        )
        try:
            image_payload = json.loads(image.stdout)
        except json.JSONDecodeError as exc:
            raise RouterHandoffLabError("Pinned llama image digest inspect was invalid.") from exc
        digests = image_payload.get("RepoDigests") if isinstance(image_payload, Mapping) else None
        expected_image_id = str(image_payload.get("Id") or "") if isinstance(image_payload, Mapping) else ""
        if not isinstance(digests, list) or not any(
            str(digest).endswith(PINNED_LLAMA_IMAGE.split("@", 1)[-1])
            for digest in digests
        ):
            raise RouterHandoffLabError("Pinned llama image digest does not match the live image.")
        if str(payload.get("Image") or "") != expected_image_id:
            raise RouterHandoffLabError("Live router container does not use the pinned image ID.")
        model_identities = self._verify_model_volume_identities()
        return {
            "verified": True,
            "command": command,
            "image_repo_digests": [str(value) for value in digests],
            "mount_destinations": sorted(by_destination),
            "model_identities": model_identities,
        }

    def _verify_model_volume_identities(self) -> dict[str, Any]:
        """Bind the actual read-only model bytes to this router arm.

        The router API reports aliases, not file digests.  Verify both model
        volumes with a network-less, read-only helper before a model can be
        accepted as the configured alias.  This is evidence for the private
        arm artifact only; no source paths or model bytes are published.
        """

        return verify_pair_model_identities(self.pair)

    @staticmethod
    def _verify_volume_files(
        volume: str,
        expected_files: Mapping[str, str],
    ) -> dict[str, Any]:
        if not _SAFE_DOCKER_NAME.fullmatch(str(volume or "")):
            raise RouterHandoffLabError("Router model volume name is unsafe.")
        expected: dict[str, str] = {}
        for filename, digest in expected_files.items():
            name = str(filename or "")
            if Path(name).name != name or not name or not _SHA256_RE.fullmatch(
                str(digest or "")
            ):
                raise RouterHandoffLabError("Router model identity manifest is invalid.")
            expected[name] = str(digest)
        if not expected:
            raise RouterHandoffLabError("Router model identity manifest is empty.")
        quoted_files = " ".join(f'"/models/{name}"' for name in sorted(expected))
        completed = _run(
            [
                "docker",
                "run",
                "--rm",
                "--pull",
                "never",
                "--network",
                "none",
                "--read-only",
                "--mount",
                f"type=volume,source={volume},target=/models,readonly",
                MODEL_IDENTITY_HELPER_IMAGE,
                "sh",
                "-ec",
                f"sha256sum {quoted_files}",
            ],
            timeout=900.0,
        )
        observed: dict[str, str] = {}
        for line in completed.stdout.splitlines():
            columns = line.strip().split(maxsplit=1)
            if len(columns) != 2:
                continue
            digest, path = columns
            filename = Path(path.strip()).name
            if filename in expected and _SHA256_RE.fullmatch(digest):
                observed[filename] = digest
        if observed != expected:
            raise RouterHandoffLabError(
                "Router model volume bytes do not match the pinned identity manifest."
            )
        return {
            "volume": volume,
            "files": {
                filename: {"sha256": digest}
                for filename, digest in sorted(observed.items())
            },
        }

    @staticmethod
    def _verify_bind_files(
        directory: Path,
        expected_files: Mapping[str, str],
    ) -> dict[str, Any]:
        """Verify the legacy baseline bind mount without mutating it."""

        root = directory.resolve()
        if not root.is_dir():
            raise RouterHandoffLabError("Baseline model bind directory is unavailable.")
        files: dict[str, dict[str, str]] = {}
        for filename, expected in sorted(expected_files.items()):
            name = str(filename or "")
            if Path(name).name != name or not _SHA256_RE.fullmatch(
                str(expected or "")
            ):
                raise RouterHandoffLabError("Baseline model identity manifest is invalid.")
            path = root / name
            if not path.is_file():
                raise RouterHandoffLabError("Baseline model bind identity is incomplete.")
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            observed = digest.hexdigest()
            if observed != expected:
                raise RouterHandoffLabError(
                    "Baseline model bind bytes do not match the pinned identity manifest."
                )
            files[name] = {"sha256": observed}
        return {"bind": "verified", "files": files}

    def _wait_router_ready(self, *, cancel_checker: Callable[[], bool] | None) -> None:
        deadline = time.monotonic() + 120.0
        last_error = ""
        while time.monotonic() < deadline:
            self._raise_if_cancelled(cancel_checker)
            try:
                status, _payload = _json_response(f"{self.endpoint}/health", timeout=3.0)
                if 200 <= status < 300:
                    return
            except RouterHandoffLabError as exc:
                last_error = str(exc)
            time.sleep(0.2)
        raise RouterHandoffLabError(f"Router health timed out: {last_error}")

    def _wait_model_state(
        self,
        alias: str,
        *,
        expected: str,
        cancel_checker: Callable[[], bool] | None,
    ) -> dict[str, str]:
        deadline = time.monotonic() + 420.0
        last_states: dict[str, str] = {}
        while time.monotonic() < deadline:
            self._raise_if_cancelled(cancel_checker)
            states = self.model_states()
            last_states = states
            if states.get(alias) == expected:
                return states
            time.sleep(0.1 if time.monotonic() + 3.0 < deadline else 0.5)
        raise RouterHandoffLabError(
            f"Router model state timed out: model={alias} expected={expected} states={last_states}"
        )

    def _assert_unloaded_request_does_not_autoload(self, before: Mapping[str, str]) -> None:
        alias = self.pair.ocr_alias
        query = urlencode({"model": alias, "autoload": "false"})
        status, _payload = _json_response(
            f"{self.endpoint}/v1/chat/completions?{query}",
            payload={
                "model": alias,
                "messages": [{"role": "user", "content": "router contract probe"}],
                "max_tokens": 1,
            },
            timeout=20.0,
        )
        after = self.model_states()
        if status < 400 or dict(before) != after or _loaded_count(after):
            raise RouterHandoffLabError(
                "Unloaded request was accepted or triggered model autoload."
            )
        self._events.append({"event": "no_autoload_probe", "status": status, "states": after})

    def _verify_loaded_identity(self, alias: str) -> None:
        query = urlencode({"model": alias, "autoload": "false"})
        status, models_payload = _json_response(f"{self.endpoint}/v1/models?{query}")
        model_rows = _models_by_alias(models_payload)
        if not 200 <= status < 300 or alias not in model_rows:
            raise RouterHandoffLabError(
                f"Router v1/models identity failed for {alias}: HTTP {status}"
            )
        for endpoint in ("props", "slots"):
            status, _payload = _json_response(f"{self.endpoint}/{endpoint}?{query}")
            if not 200 <= status < 300:
                raise RouterHandoffLabError(
                    f"Router {endpoint} readiness failed for {alias}: HTTP {status}"
                )
        status, metrics = _text_response(f"{self.endpoint}/metrics?{query}")
        if not 200 <= status < 300 or not metrics.strip():
            raise RouterHandoffLabError(f"Router metrics readiness failed for {alias}.")

    def _wait_gpu_release(self, *, before_gpu: int | None, started: float) -> dict[str, Any]:
        baseline = self._pre_start_gpu_mib
        if before_gpu is None or baseline is None:
            return {
                "observed": False,
                "status": "gpu_metric_unavailable",
                "before_gpu_mib": before_gpu,
                "baseline_gpu_mib": baseline,
            }
        deadline = time.monotonic() + 45.0
        threshold = int(baseline) + self.release_residual_mib
        last = before_gpu
        while time.monotonic() < deadline:
            current = _gpu_used_mib()
            if current is not None:
                last = current
                if current <= threshold:
                    return {
                        "observed": True,
                        "status": "returned",
                        "before_gpu_mib": before_gpu,
                        "baseline_gpu_mib": baseline,
                        "after_gpu_mib": current,
                        "threshold_mib": threshold,
                        "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 3),
                    }
            time.sleep(0.25)
        return {
            "observed": False,
            "status": "timeout",
            "before_gpu_mib": before_gpu,
            "baseline_gpu_mib": baseline,
            "after_gpu_mib": last,
            "threshold_mib": threshold,
            "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 3),
        }

    @staticmethod
    def _raise_if_cancelled(cancel_checker: Callable[[], bool] | None) -> None:
        if callable(cancel_checker) and cancel_checker():
            raise OperationCancelledError("Router handoff lab was cancelled.")


def verify_pair_model_identities(pair: RouterPair) -> dict[str, Any]:
    """Verify the bytes used by both baseline and router arms for one pair."""

    pair.validate()
    ocr_expected = {
        pair.ocr_model: pair.ocr_model_sha256,
        pair.ocr_mmproj: pair.ocr_mmproj_sha256,
    }
    router_ocr = RouterLabSession._verify_volume_files(
        pair.ocr_volume,
        ocr_expected,
    )
    baseline_ocr = (
        RouterLabSession._verify_bind_files(pair.baseline_bind_dir, ocr_expected)
        if pair.baseline_bind_dir is not None
        else router_ocr
    )
    return {
        "router_ocr": router_ocr,
        "baseline_ocr": baseline_ocr,
        "gemma": RouterLabSession._verify_volume_files(
            GEMMA_VOLUME,
            {GEMMA_ALIAS: GEMMA_SHA256},
        ),
    }


class RouterLabOCRRuntimeManager(LocalOCRRuntimeManager):
    def __init__(self, *, inner: LocalOCRRuntimeManager, session: RouterLabSession) -> None:
        super().__init__()
        self.inner = inner
        self.session = session

    def validate_engine(self, engine_key: str, _settings_page: Any) -> None:
        if engine_key != self.session.pair.engine_key:
            raise RouterHandoffLabError(f"Unexpected router OCR engine: {engine_key}")

    def should_manage_engine(self, engine_key: str, _settings_page: Any) -> bool:
        return engine_key == self.session.pair.engine_key

    def preflight_cache_key(self, engine_key: str, _settings_page: Any) -> str | None:
        return f"router-lab|{self.session.pair.key}" if self.should_manage_engine(engine_key, _settings_page) else None

    def get_ocr_cache_identity(self, engine_key: str, _settings_page: Any) -> dict[str, Any] | None:
        if not self.should_manage_engine(engine_key, _settings_page):
            return None
        pair = self.session.pair
        return {
            "managed": True,
            "router_lab": True,
            "pair": pair.key,
            "model_name": pair.ocr_model,
            "model_sha256": pair.ocr_model_sha256,
            "mmproj_sha256": pair.ocr_mmproj_sha256,
            "runtime_fingerprint": canonical_sha256(
                {"protocol": PROTOCOL_VERSION, "pair": pair.key, "model": pair.ocr_alias}
            ),
        }

    def probe_managed_engine(self, engine_key: str, _settings_page: Any) -> str:
        return "healthy" if self.session._prepared and engine_key == self.session.pair.engine_key else "unavailable"

    def ensure_engine(
        self,
        engine_key: str,
        _settings_page: Any,
        *,
        timeout_sec: int = 420,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
        cancel_checker: Callable[[], bool] | None = None,
    ) -> None:
        del timeout_sec
        self.validate_engine(engine_key, _settings_page)
        event = self.session.load_model(self.session.pair.ocr_alias, cancel_checker=cancel_checker)
        if callable(progress_callback):
            progress_callback(
                {
                    "phase": "router_handoff_lab",
                    "service": self.session.pair.key,
                    "status": "ready",
                    "step_key": "router_ocr_load",
                    "detail": json.dumps(event, ensure_ascii=False, sort_keys=True),
                }
            )

    def release_for_handoff(self) -> dict[str, Any]:
        return self.session.unload_model(self.session.pair.ocr_alias)

    def shutdown(self) -> dict[str, Any]:
        return self.session.unload_model(self.session.pair.ocr_alias)


class RouterLabGemmaRuntimeManager(LocalGemmaRuntimeManager):
    def __init__(self, *, inner: LocalGemmaRuntimeManager, session: RouterLabSession) -> None:
        super().__init__()
        self.inner = inner
        self.session = session

    def validate_server(self, _settings_page: Any) -> None:
        return None

    def should_manage_server(self, _settings_page: Any) -> bool:
        return True

    def get_translation_cache_identity(self, _settings_page: Any) -> dict[str, Any]:
        return {
            "managed": True,
            "router_lab": True,
            "pair": self.session.pair.key,
            "model_name": GEMMA_ALIAS,
            "model_sha256": GEMMA_SHA256,
            "runtime_image_ref": PINNED_LLAMA_IMAGE,
            "runtime_image_revision": PINNED_LLAMA_REVISION,
            "runtime_fingerprint": canonical_sha256(
                {"protocol": PROTOCOL_VERSION, "pair": self.session.pair.key, "model": GEMMA_ALIAS}
            ),
        }

    def ensure_server(
        self,
        _settings_page: Any,
        *,
        timeout_sec: int = 420,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
        cancel_checker: Callable[[], bool] | None = None,
    ) -> None:
        del timeout_sec
        event = self.session.load_model(GEMMA_ALIAS, cancel_checker=cancel_checker)
        if callable(progress_callback):
            progress_callback(
                {
                    "phase": "router_handoff_lab",
                    "service": "gemma",
                    "status": "ready",
                    "step_key": "router_gemma_load",
                    "detail": json.dumps(event, ensure_ascii=False, sort_keys=True),
                }
            )

    def shutdown(self) -> dict[str, Any]:
        return self.session.unload_model(GEMMA_ALIAS)


class _RequestsProxy:
    def __init__(self, original: Any, *, session: RouterLabSession, alias: str, inject_model: bool) -> None:
        self._original = original
        self._session = session
        self._alias = alias
        self._inject_model = inject_model
        self.exceptions = original.exceptions

    def post(self, *args: Any, **kwargs: Any) -> Any:
        return self._post_via(self._original.post, *args, **kwargs)

    def _post_via(self, sender: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        url = str(args[0] if args else kwargs.get("url", ""))
        if not self._is_router_url(url):
            # These proxies replace only the lab's managed OCR/Gemma modules.
            # Letting an endpoint that merely resembles loopback fall through
            # would allow a page-image payload to escape the lab boundary.
            raise RouterHandoffLabError("Router lab request does not target its exact loopback endpoint.")
        payload = kwargs.get("json")
        if not isinstance(payload, Mapping):
            raise RouterHandoffLabError("Router request body must be a JSON object.")
        copied = dict(payload)
        if self._inject_model:
            existing = str(copied.get("model", "") or "")
            if existing and existing != self._alias:
                raise RouterHandoffLabError(
                    f"Unexpected router model: {existing} (expected {self._alias})"
                )
            copied["model"] = self._alias
        elif str(copied.get("model", "") or "") != self._alias:
            raise RouterHandoffLabError(
                f"Router request omitted or changed model {self._alias}."
            )
        kwargs = dict(kwargs)
        kwargs["json"] = copied
        if self._alias == self._session.pair.ocr_alias:
            self._session.capture_ocr_request(url, copied)
        request_started = self._session.begin_http_request(self._alias)
        try:
            response = sender(*args, **kwargs)
        except BaseException:
            self._session.finish_http_request(
                self._alias,
                started=request_started,
                successful=False,
            )
            raise
        self._session.finish_http_request(
            self._alias,
            started=request_started,
            successful=200 <= int(getattr(response, "status_code", 0)) < 300,
        )
        return response

    def Session(self, *args: Any, **kwargs: Any) -> Any:
        return _SessionPostProxy(self._original.Session(*args, **kwargs), self)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._original, name)

    def _is_router_url(self, value: str) -> bool:
        expected_port = (
            self._session.pair.ocr_port
            if self._alias == self._session.pair.ocr_alias
            else self._session.gemma_port
        )
        return _is_safe_loopback_router_url(value, expected_port=expected_port)


class _SessionPostProxy:
    """Preserve session semantics while routing every post through its guard."""

    def __init__(self, original: Any, parent: _RequestsProxy) -> None:
        self._original = original
        self._parent = parent

    def post(self, *args: Any, **kwargs: Any) -> Any:
        return self._parent._post_via(self._original.post, *args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._original, name)


class InstalledRouterHandoffLabRuntime:
    def __init__(
        self,
        *,
        window: Any,
        session: RouterLabSession,
        original_ocr: LocalOCRRuntimeManager,
        original_gemma: LocalGemmaRuntimeManager,
        restore_requests: list[tuple[Any, Any]],
        original_start_gemma_prewarm: Callable[..., Any],
        arbiter: Any,
        original_arbiter_methods: tuple[Callable[..., Any], Callable[..., Any]],
    ) -> None:
        self.window = window
        self.session = session
        self.original_ocr = original_ocr
        self.original_gemma = original_gemma
        self.restore_requests = restore_requests
        self.original_start_gemma_prewarm = original_start_gemma_prewarm
        self.arbiter = arbiter
        self.original_arbiter_methods = original_arbiter_methods
        self.closed = False

    def prepare_process(self) -> dict[str, Any]:
        return self.session.prepare_process()

    def verify_round_trip(self) -> dict[str, Any]:
        """Prove the required OCR -> Gemma -> OCR relay under the Arbiter.

        This probe is intentionally outside the measured E2E timing, but it is
        still a real GPU model transition.  It therefore uses the exact same
        lease and driver-global release proof as the product stage path.
        """

        if self.session._round_trip_verified:
            return {"status": "reused"}
        processor = self.window.pipeline.stage_batched_processor
        # ``batch_process`` always closes its prewarm generation before it
        # returns.  This is not a user cancellation: it prevents late
        # background prewarms from touching a completed page.  The explicit
        # post-run relay below needs its own fresh Arbiter generation, but only
        # after proving that the completed pipeline left no active or failed
        # lease behind.
        processor._raise_if_cancelled()
        completed_snapshot = self.arbiter.snapshot()
        if completed_snapshot.active_model is not None:
            raise RouterHandoffLabError(
                "Router round-trip requires no active model after pipeline completion."
            )
        if any(
            state == "release_failed"
            for state in completed_snapshot.states.values()
        ):
            raise RouterHandoffLabError(
                "Router round-trip refuses to reset after a failed GPU release."
            )
        self.arbiter.reset()
        service = processor._ocr_runtime_service_name(self.session.pair.engine_key)
        # This check runs after the page pipeline has completed.  At that
        # point the product prewarm checker intentionally reports the finished
        # task as cancelled, which is correct for a new prewarm but would make
        # this already-owned, cleanup-protected lab probe fail before it can
        # prove the final unload.  Build a fresh user-cancellation probe from
        # the processor instead.  It remains active during the new load and
        # relay; only the cleanup path deliberately ignores cancellation so it
        # can always unload the model it owns.
        def cancel_checker() -> bool:
            try:
                processor._raise_if_cancelled()
            except OperationCancelledError:
                return True
            return False

        token = self.arbiter.token(service)
        started = time.perf_counter()

        def release_probe() -> dict[str, Any]:
            release_before = query_cuda_handoff_metrics()
            with self.arbiter.model_release(service) as release_context:
                release_report = self.session.unload_model(
                    self.session.pair.ocr_alias,
                    cancel_checker=None,
                )
                gate = processor._verify_managed_runtime_gpu_release(
                    service,
                    release_report,
                    before=release_before,
                )
                if bool(gate.get("required")) and not bool(gate.get("observed")):
                    raise RouterHandoffLabError(
                        "Router OCR re-load probe did not prove GPU return."
                    )
            return gate

        # If model load or the cancellation checker fails, the Arbiter keeps
        # ownership until this cleanup has unloaded the router model and the
        # same product gate has observed VRAM return.
        with self.arbiter.model_start(
            token,
            cancel_checker=cancel_checker,
            stale_cleanup=release_probe,
        ):
            processor._capture_runtime_gpu_start_baseline(service)
            self.session.load_model(
                self.session.pair.ocr_alias,
                cancel_checker=cancel_checker,
            )
        gate: dict[str, Any] = {}
        try:
            self.session.execute_captured_ocr_round_trip_request(
                cancel_checker=cancel_checker,
            )
        finally:
            gate = release_probe()
        return self.session.mark_ocr_round_trip_verified(
            elapsed_ms=(time.perf_counter() - started) * 1000.0,
            gpu_return_gate=gate,
        )

    def validate_window_contract(self) -> None:
        ui = self.window.settings_page.ui
        gemma_url = str(
            ui.credential_widgets["Custom Local Server(Gemma)_api_url"].text()
            or ""
        ).strip().rstrip("/")
        if gemma_url != f"http://127.0.0.1:{self.session.gemma_port}/v1":
            raise RouterHandoffLabError("Router lab rejects a custom Gemma endpoint.")
        gemma_model = str(
            ui.credential_widgets["Custom Local Server(Gemma)_model"].text()
            or ""
        ).strip()
        if gemma_model != GEMMA_ALIAS:
            raise RouterHandoffLabError("Router lab Gemma model identity drifted.")
        urls = {
            "PaddleOCR VL": str(ui.paddleocr_vl_server_url_input.text() or ""),
            "PaddleOCR VL Spotting": str(
                ui.paddleocr_vl_spotting_server_url_input.text() or ""
            ),
            "HunyuanOCR": str(ui.hunyuan_ocr_server_url_input.text() or ""),
            "MangaLMM": str(ui.mangalmm_ocr_server_url_input.text() or ""),
        }
        expected_paths = {
            "PaddleOCR VL": "/v1/chat/completions",
            "PaddleOCR VL Spotting": "/v1/chat/completions",
            "HunyuanOCR": "/v1",
            "MangaLMM": "/v1",
        }
        expected = (
            f"http://127.0.0.1:{self.session.pair.ocr_port}"
            f"{expected_paths[self.session.pair.engine_key]}"
        )
        observed = urls.get(self.session.pair.engine_key, "").strip().rstrip("/")
        if observed != expected:
            raise RouterHandoffLabError("Router lab OCR endpoint drifted.")

    def close(self) -> None:
        if self.closed:
            return
        try:
            self.session.stop_process()
        finally:
            self.window.local_ocr_runtime_manager = self.original_ocr
            self.window.local_translation_runtime_manager = self.original_gemma
            self.window.pipeline.stage_batched_processor._start_gemma_prewarm = self.original_start_gemma_prewarm
            self.arbiter.model_start, self.arbiter.model_release = (
                self.original_arbiter_methods
            )
            for module, original in reversed(self.restore_requests):
                module.requests = original
            self.closed = True


def _install_request_proxies(session: RouterLabSession) -> list[tuple[Any, Any]]:
    from modules.ocr import hunyuan_ocr
    from modules.ocr import ocr_paddle_VL
    from modules.ocr.paddle_crop import engine as paddle_crop_engine
    from modules.ocr.paddle_spotting import engine as spotting_engine
    from modules.ocr.mangalmm_full_page import engine as mangalmm_engine
    from modules.translation.llm import custom_local_gemma

    modules: list[tuple[Any, str, bool]] = [
        (custom_local_gemma, GEMMA_ALIAS, False),
        (ocr_paddle_VL, session.pair.ocr_alias, False),
        (paddle_crop_engine, session.pair.ocr_alias, False),
        (spotting_engine, session.pair.ocr_alias, False),
        (hunyuan_ocr, session.pair.ocr_alias, True),
        (mangalmm_engine, session.pair.ocr_alias, True),
    ]
    restored: list[tuple[Any, Any]] = []
    for module, alias, inject_model in modules:
        original = module.requests
        module.requests = _RequestsProxy(
            original,
            session=session,
            alias=alias,
            inject_model=inject_model,
        )
        restored.append((module, original))
    return restored


def _instrument_arbiter_commands(processor: Any, session: RouterLabSession) -> tuple[Any, tuple[Callable[..., Any], Callable[..., Any]]]:
    """Record only time spent waiting to enter the existing command gates."""

    arbiter = processor._runtime_resource_arbiter()
    original_start = arbiter.model_start
    original_release = arbiter.model_release

    def observed_start(*args: Any, **kwargs: Any):
        @contextmanager
        def gate() -> Iterator[None]:
            queued_at = time.perf_counter()
            with original_start(*args, **kwargs):
                session.record_command_queue_wait("model_start", queued_at)
                yield

        return gate()

    def observed_release(*args: Any, **kwargs: Any):
        @contextmanager
        def gate() -> Iterator[Any]:
            queued_at = time.perf_counter()
            with original_release(*args, **kwargs) as context:
                session.record_command_queue_wait("model_release", queued_at)
                yield context

        return gate()

    arbiter.model_start = observed_start
    arbiter.model_release = observed_release
    return arbiter, (original_start, original_release)


def install_router_handoff_lab_runtime_adapter(
    window: Any,
    config: Mapping[str, Any],
) -> InstalledRouterHandoffLabRuntime:
    pair_key = str(config.get("pair", "") or "").strip()
    catalog = pair_catalog()
    if pair_key not in catalog:
        raise RouterHandoffLabError(f"Unknown router lab pair: {pair_key!r}")
    artifact_dir = Path(str(config.get("artifact_dir", "") or "")).expanduser().resolve()
    if not artifact_dir.is_dir():
        raise RouterHandoffLabError("Router lab artifact directory does not exist.")
    container_name = str(config.get("container_name", "") or "").strip()
    if not container_name:
        raise RouterHandoffLabError("Router lab requires an exact container name.")
    original_ocr = getattr(window, "local_ocr_runtime_manager", None)
    original_gemma = getattr(window, "local_translation_runtime_manager", None)
    if not isinstance(original_ocr, LocalOCRRuntimeManager) or not isinstance(
        original_gemma, LocalGemmaRuntimeManager
    ):
        raise RouterHandoffLabError("Router lab requires the two product runtime managers.")
    session = RouterLabSession(
        pair=catalog[pair_key], artifact_dir=artifact_dir, container_name=container_name
    )
    owner_token = str(config.get("owner_token", "") or "").strip()
    if owner_token != session.owner_token:
        raise RouterHandoffLabError("Router lab owner token does not match the managed arm.")
    processor = window.pipeline.stage_batched_processor
    original_start_gemma_prewarm = processor._start_gemma_prewarm
    restore_requests: list[tuple[Any, Any]] = []
    arbiter = None
    original_arbiter_methods: tuple[Callable[..., Any], Callable[..., Any]] | None = None
    try:
        window.local_ocr_runtime_manager = RouterLabOCRRuntimeManager(
            inner=original_ocr, session=session
        )
        window.local_translation_runtime_manager = RouterLabGemmaRuntimeManager(
            inner=original_gemma, session=session
        )
        processor._start_gemma_prewarm = lambda: None
        restore_requests = _install_request_proxies(session)
        arbiter, original_arbiter_methods = _instrument_arbiter_commands(processor, session)
    except BaseException:
        window.local_ocr_runtime_manager = original_ocr
        window.local_translation_runtime_manager = original_gemma
        processor._start_gemma_prewarm = original_start_gemma_prewarm
        if arbiter is not None and original_arbiter_methods is not None:
            arbiter.model_start, arbiter.model_release = original_arbiter_methods
        for module, original in reversed(restore_requests):
            module.requests = original
        raise
    assert arbiter is not None and original_arbiter_methods is not None
    return InstalledRouterHandoffLabRuntime(
        window=window,
        session=session,
        original_ocr=original_ocr,
        original_gemma=original_gemma,
        restore_requests=restore_requests,
        original_start_gemma_prewarm=original_start_gemma_prewarm,
        arbiter=arbiter,
        original_arbiter_methods=original_arbiter_methods,
    )

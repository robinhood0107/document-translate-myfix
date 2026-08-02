"""Product-only llama.cpp Router lifecycle coordination.

The Router owns one managed Paddle/Gemma pair.  Model command calls are
serialized here, while OCR and translation HTTP requests remain in their
existing clients and execute outside this command lock.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlparse
from urllib.request import Request, urlopen

from modules.ocr.paddle_crop.runtime import (
    DEFAULT_PADDLE_LLAMA_CPP_IMAGE,
    PADDLE_LLAMA_MMPROJ_NAME,
    PADDLE_LLAMA_MODEL_ALIAS,
    PADDLE_LLAMA_MODEL_NAME,
    PADDLE_LLAMA_MODEL_SPECS,
)
from modules.ocr.paddle_spotting.runtime import (
    DEFAULT_PADDLE_SPOTTING_MODEL_VOLUME,
    PADDLE_SPOTTING_MMPROJ_NAME,
    PADDLE_SPOTTING_MODEL_ALIAS,
    PADDLE_SPOTTING_MODEL_NAME,
    PADDLE_SPOTTING_MODEL_SPECS,
)
from modules.translation.gemma_runtime_contract import (
    DEFAULT_GEMMA_MODEL_VOLUME,
    DEFAULT_GEMMA_PREPARED_MODEL,
    GEMMA_MODEL_SPECS,
)
from modules.utils.exceptions import OperationCancelledError
from modules.utils.llama_cpp_runtime import (
    DEFAULT_MANAGED_RUNTIME_STOP_TIMEOUT_SEC,
    resolve_docker_compose_command,
    run_docker_command,
)


ROOT_DIR = Path(__file__).resolve().parents[2]
ROUTER_PROTOCOL_VERSION = "llamacpp-router-product-v1"
ROUTER_CONTAINER_NAME = "comic-translate-paddle-router"
ROUTER_GEMMA_HOST_PORT = 18080
ROUTER_INTERNAL_PORT = 8080
ROUTER_GEMMA_ALIAS = DEFAULT_GEMMA_PREPARED_MODEL
ROUTER_GEMMA_VOLUME = DEFAULT_GEMMA_MODEL_VOLUME
ROUTER_GEMMA_MODEL_SHA256 = str(GEMMA_MODEL_SPECS[ROUTER_GEMMA_ALIAS]["sha256"])

ROUTER_LABEL_PROTOCOL = "comic-translate.router-protocol"
ROUTER_LABEL_PAIR = "comic-translate.router-pair"
ROUTER_LABEL_FINGERPRINT = "comic-translate.router-fingerprint"
ROUTER_LABEL_OCR_VOLUME = "comic-translate.router-ocr-volume"
ROUTER_LABEL_GEMMA_VOLUME = "comic-translate.router-gemma-volume"
ROUTER_LABEL_OCR_MODEL_SHA256 = "comic-translate.router-ocr-model-sha256"
ROUTER_LABEL_OCR_MMPROJ_SHA256 = "comic-translate.router-ocr-mmproj-sha256"
ROUTER_LABEL_OCR_MANIFEST_SHA256 = "comic-translate.router-ocr-manifest-sha256"
ROUTER_LABEL_GEMMA_MODEL_SHA256 = "comic-translate.router-gemma-model-sha256"
ROUTER_LABEL_GEMMA_MANIFEST_SHA256 = "comic-translate.router-gemma-manifest-sha256"
ROUTER_LABEL_COMMAND_SHA256 = "comic-translate.router-command-sha256"

DEFAULT_ROUTER_OCR_OPTIONS = {
    "context": "4096",
    "parallel": "1",
    "threads": "10",
    "batch": "2048",
    "ubatch": "512",
    "gpu_layers": "all",
    "sleep_idle_seconds": "5",
}
DEFAULT_ROUTER_GEMMA_OPTIONS = {
    "context": "4096",
    "parallel": "1",
    "threads": "10",
    "batch": "2048",
    "ubatch": "512",
    "gpu_layers": "23",
    "cache_type_k": "f16",
    "cache_type_v": "f16",
    "kv_offload": True,
    "swa_full": True,
    "jinja": True,
    "reasoning": "off",
    "cache_ram_mib": "0",
    "spec_type": "none",
    "spec_draft_n_max": "8",
}


class RouterRuntimeError(RuntimeError):
    """Raised when the product Router contract cannot be proven."""


@dataclass(frozen=True)
class RouterPairSpec:
    key: str
    engine_key: str
    ocr_alias: str
    ocr_model: str
    ocr_mmproj: str
    ocr_model_sha256: str
    ocr_mmproj_sha256: str
    ocr_port: int
    ocr_volume: str
    compose_file: Path
    preset_file: Path

    @property
    def endpoint(self) -> str:
        return f"http://127.0.0.1:{self.ocr_port}"


@dataclass(frozen=True)
class RouterRuntimeSnapshot:
    generation: int
    pair: str | None
    prepared: bool
    container_running: bool
    active_model: str | None
    loaded_count: int
    states: dict[str, str]
    fingerprint: str
    release_failed: bool


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalize_url(value: str) -> str:
    parsed = urlparse(str(value or "").strip())
    path = parsed.path.rstrip("/") or "/"
    return parsed._replace(path=path, params="", query="", fragment="").geturl()


def _exact_loopback_url(value: str, expected_port: int, expected_path: str) -> bool:
    try:
        parsed = urlparse(str(value or "").strip())
        port = parsed.port
    except ValueError:
        return False
    return bool(
        parsed.scheme == "http"
        and parsed.hostname == "127.0.0.1"
        and port == int(expected_port)
        and parsed.username is None
        and parsed.password is None
        and _normalize_url(value) == _normalize_url(
            f"http://127.0.0.1:{expected_port}{expected_path}"
        )
    )


def _json_http(
    url: str,
    *,
    payload: Mapping[str, Any] | None = None,
    timeout_sec: float = 30.0,
) -> tuple[int, Any]:
    data = None
    headers = {"Accept": "application/json"}
    method = "GET"
    if payload is not None:
        data = json.dumps(dict(payload), ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json"
        method = "POST"
    request = Request(url, data=data, headers=headers, method=method)
    try:
        with urlopen(request, timeout=timeout_sec) as response:
            raw = response.read().decode("utf-8", errors="replace")
            try:
                return int(response.status), json.loads(raw) if raw else {}
            except json.JSONDecodeError:
                return int(response.status), raw
    except HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        try:
            body: Any = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            body = raw
        return int(exc.code), body
    except (URLError, OSError, ValueError) as exc:
        raise RouterRuntimeError(f"Router HTTP request failed: {url}: {exc}") from exc


def _model_states(payload: Any) -> dict[str, str]:
    rows: Any = payload.get("data") if isinstance(payload, Mapping) else None
    if not isinstance(rows, list):
        rows = payload.get("models") if isinstance(payload, Mapping) else None
    if isinstance(rows, Mapping):
        rows = [
            {"id": key, **(value if isinstance(value, Mapping) else {})}
            for key, value in rows.items()
        ]
    states: dict[str, str] = {}
    if not isinstance(rows, list):
        return states
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        alias = str(row.get("id") or row.get("name") or row.get("model") or "").strip()
        state: Any = row.get("state") or row.get("status") or row.get("value")
        if isinstance(state, Mapping):
            state = state.get("value") or state.get("state") or state.get("status")
        if alias:
            states[alias] = str(state or "unknown").lower()
    return states


def _loaded_count(states: Mapping[str, str]) -> int:
    return sum(1 for value in states.values() if value != "unloaded")


def _model_ids(payload: Any) -> set[str]:
    rows: Any = payload.get("data") if isinstance(payload, Mapping) else None
    if not isinstance(rows, list):
        rows = payload.get("models") if isinstance(payload, Mapping) else None
    if isinstance(rows, Mapping):
        rows = [
            {"id": key, **(value if isinstance(value, Mapping) else {})}
            for key, value in rows.items()
        ]
    if not isinstance(rows, list):
        return set()
    result: set[str] = set()
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        for key in ("id", "name", "model"):
            value = str(row.get(key) or "").strip()
            if value:
                result.add(value)
                break
    return result


def _expected_command() -> tuple[str, ...]:
    return (
        "--host",
        "0.0.0.0",
        "--port",
        str(ROUTER_INTERNAL_PORT),
        "--models-preset",
        "/config/models.ini",
        "--models-max",
        "1",
        "--no-models-autoload",
        "--metrics",
        "--props",
        "--slots",
    )


def _pair_catalog() -> dict[str, RouterPairSpec]:
    paddle_root = ROOT_DIR / "paddleocr_vl_docker_files"
    spotting_root = ROOT_DIR / "paddleocr_vl_spotting_docker_files"
    return {
        "paddle-crop": RouterPairSpec(
            key="paddle-crop",
            engine_key="PaddleOCR VL",
            ocr_alias=PADDLE_LLAMA_MODEL_ALIAS,
            ocr_model=PADDLE_LLAMA_MODEL_NAME,
            ocr_mmproj=PADDLE_LLAMA_MMPROJ_NAME,
            ocr_model_sha256=str(PADDLE_LLAMA_MODEL_SPECS[PADDLE_LLAMA_MODEL_NAME]["sha256"]),
            ocr_mmproj_sha256=str(PADDLE_LLAMA_MODEL_SPECS[PADDLE_LLAMA_MMPROJ_NAME]["sha256"]),
            ocr_port=18000,
            ocr_volume="comic-translate-paddleocr-vl-llamacpp-models-v1",
            compose_file=paddle_root / "docker-compose.router.yaml",
            preset_file=paddle_root / "models.ini",
        ),
        "paddle-spotting": RouterPairSpec(
            key="paddle-spotting",
            engine_key="PaddleOCR VL Spotting",
            ocr_alias=PADDLE_SPOTTING_MODEL_ALIAS,
            ocr_model=PADDLE_SPOTTING_MODEL_NAME,
            ocr_mmproj=PADDLE_SPOTTING_MMPROJ_NAME,
            ocr_model_sha256=str(PADDLE_SPOTTING_MODEL_SPECS[PADDLE_SPOTTING_MODEL_NAME]["sha256"]),
            ocr_mmproj_sha256=str(
                PADDLE_SPOTTING_MODEL_SPECS[PADDLE_SPOTTING_MMPROJ_NAME]["sha256"]
            ),
            ocr_port=18002,
            ocr_volume=DEFAULT_PADDLE_SPOTTING_MODEL_VOLUME,
            compose_file=spotting_root / "docker-compose.router.yaml",
            preset_file=spotting_root / "models.ini",
        ),
    }


class LocalLlamaRouterCoordinator:
    """Coordinate one explicit-load Router pair for the product runtime."""

    def __init__(self) -> None:
        self._command_lock = threading.RLock()
        self._generation = 0
        self._gemma_identity_provider: Callable[[Any], Mapping[str, Any] | None] | None = None
        self._pair_key: str | None = None
        self._pending_ocr_identity: dict[str, Any] | None = None
        self._expected_contract: dict[str, Any] | None = None
        self._prepared = False
        self._container_running = False
        self._active_model: str | None = None
        self._states: dict[str, str] = {}
        self._fingerprint = ""
        self._autoload_probe_generation: int | None = None
        self._release_failed = False

    @property
    def generation(self) -> int:
        with self._command_lock:
            return self._generation

    def register_gemma_identity_provider(
        self,
        provider: Callable[[Any], Mapping[str, Any] | None],
    ) -> None:
        with self._command_lock:
            self._gemma_identity_provider = provider

    def begin_generation(self) -> int:
        with self._command_lock:
            if self._release_failed or self._active_model or _loaded_count(self._states):
                raise RouterRuntimeError(
                    "Router generation cannot advance while model release is unresolved."
                )
            self._generation += 1
            self._autoload_probe_generation = None
            return self._generation

    def snapshot(self) -> RouterRuntimeSnapshot:
        with self._command_lock:
            return RouterRuntimeSnapshot(
                generation=self._generation,
                pair=self._pair_key,
                prepared=self._prepared,
                container_running=self._container_running,
                active_model=self._active_model,
                loaded_count=_loaded_count(self._states),
                states=dict(self._states),
                fingerprint=self._fingerprint,
                release_failed=self._release_failed,
            )

    def has_active_pair(self) -> bool:
        with self._command_lock:
            return bool(self._pair_key or self._prepared or self._container_running)

    def is_router_ocr_candidate(self, engine_key: str, settings_page: Any) -> bool:
        pair = self._pair_for_engine(engine_key)
        if pair is None:
            return False
        if not _exact_loopback_url(
            self._ocr_url(engine_key, settings_page),
            pair.ocr_port,
            "/v1/chat/completions",
        ):
            return False
        try:
            if settings_page.get_tool_selection("translator") != "Custom Local Server(Gemma)":
                return False
            creds = settings_page.get_credentials("Custom Local Server(Gemma)") or {}
        except (AttributeError, KeyError):
            return False
        return (
            _normalize_url(str(creds.get("api_url", "")))
            == _normalize_url("http://127.0.0.1:18080/v1")
            and str(creds.get("model", "")).strip() == ROUTER_GEMMA_ALIAS
        )

    def is_router_gemma_candidate(self, settings_page: Any) -> bool:
        with self._command_lock:
            pair_selected = self._pair_key is not None
        if not pair_selected:
            return False
        try:
            if settings_page.get_tool_selection("translator") != "Custom Local Server(Gemma)":
                return False
            creds = settings_page.get_credentials("Custom Local Server(Gemma)") or {}
        except (AttributeError, KeyError):
            return False
        return (
            _normalize_url(str(creds.get("api_url", "")))
            == _normalize_url("http://127.0.0.1:18080/v1")
            and str(creds.get("model", "")).strip() == ROUTER_GEMMA_ALIAS
        )

    def cache_identity_for_ocr(
        self,
        engine_key: str,
        settings_page: Any,
        ocr_identity: Mapping[str, Any],
    ) -> dict[str, Any] | None:
        if not self.is_router_ocr_candidate(engine_key, settings_page):
            return None
        gemma_identity = self._gemma_identity(settings_page)
        if gemma_identity is None:
            raise RouterRuntimeError(
                "Router OCR routing requires a complete Gemma runtime identity."
            )
        pair = self._pair_for_engine(engine_key)
        if pair is None:
            raise RouterRuntimeError(f"No product Router pair exists for {engine_key!r}.")
        contract = self._build_contract(pair, ocr_identity, gemma_identity)
        with self._command_lock:
            self._select_pair_locked(pair, ocr_identity)
            generation = self._generation
        return {
            "identity_schema_version": 4,
            "managed": True,
            "router": True,
            "pair": pair.key,
            "engine": engine_key,
            "backend": "llama.cpp-router",
            "endpoint": _normalize_url(self._ocr_url(engine_key, settings_page)),
            "model_name": pair.ocr_alias,
            "model_file": pair.ocr_model,
            "model_sha256": pair.ocr_model_sha256,
            "mmproj_file": pair.ocr_mmproj,
            "mmproj_sha256": pair.ocr_mmproj_sha256,
            "model_volume": str(ocr_identity.get("volume", pair.ocr_volume)),
            "ready_manifest_sha256": str(ocr_identity.get("manifest_sha256", "")),
            "llama_image_ref": str(contract["image_ref"]),
            "llama_image_id": str(contract["image_id"]),
            "compose_sha256": str(contract["compose_sha256"]),
            "preset_sha256": str(contract["preset_sha256"]),
            "command_sha256": str(contract["command_sha256"]),
            "runtime_fingerprint": str(contract["fingerprint"]),
            "router_generation": generation,
        }

    def ensure_ocr_model(
        self,
        engine_key: str,
        settings_page: Any,
        ocr_identity: Mapping[str, Any],
        *,
        cancel_checker: Callable[[], bool] | None = None,
    ) -> dict[str, Any] | None:
        if not self.is_router_ocr_candidate(engine_key, settings_page):
            return None
        pair = self._pair_for_engine(engine_key)
        gemma_identity = self._gemma_identity(settings_page)
        if pair is None or gemma_identity is None:
            raise RouterRuntimeError(
                "Router OCR startup requires a complete Paddle/Gemma pair identity."
            )
        contract = self._prepare_pair(
            pair,
            ocr_identity,
            gemma_identity,
            cancel_checker=cancel_checker,
        )
        self._probe_no_autoload_if_needed(pair, cancel_checker=cancel_checker)
        with self._command_lock:
            self._load_model_locked(pair.ocr_alias, cancel_checker=cancel_checker)
            return {"router": True, "model": pair.ocr_alias, **contract}

    def ensure_gemma_model(
        self,
        settings_page: Any,
        gemma_identity: Mapping[str, Any],
        *,
        cancel_checker: Callable[[], bool] | None = None,
    ) -> dict[str, Any] | None:
        if not self.is_router_gemma_candidate(settings_page):
            return None
        with self._command_lock:
            pair = self._pair_catalog().get(self._pair_key or "")
            ocr_identity = dict(self._pending_ocr_identity or {})
        if pair is None or not ocr_identity:
            raise RouterRuntimeError(
                "Gemma Router startup requires a selected Paddle OCR pair."
            )
        contract = self._prepare_pair(
            pair,
            ocr_identity,
            gemma_identity,
            cancel_checker=cancel_checker,
        )
        self._probe_no_autoload_if_needed(pair, cancel_checker=cancel_checker)
        with self._command_lock:
            self._load_model_locked(ROUTER_GEMMA_ALIAS, cancel_checker=cancel_checker)
            return {"router": True, "model": ROUTER_GEMMA_ALIAS, **contract}

    def unload_model(
        self,
        alias: str,
        *,
        cancel_checker: Callable[[], bool] | None = None,
    ) -> dict[str, Any]:
        with self._command_lock:
            if not self._prepared:
                return {"runtime_state": "stopped", "gpu_release_expected": False}
            if self._active_model and self._active_model != alias:
                raise RouterRuntimeError(
                    f"Cannot unload {alias}; Router model is {self._active_model}."
                )
            states = self._model_states_locked()
            if alias not in states or states.get(alias) == "unloaded":
                if _loaded_count(states):
                    self._release_failed = True
                    raise RouterRuntimeError(
                        f"Router model {alias} is not loaded but another model remains loaded: {states}"
                    )
                self._active_model = None
                self._states = states
                self._release_failed = False
                return {"runtime_state": "sleeping", "gpu_release_expected": False}
            started = time.perf_counter()
            status, payload = _json_http(
                f"{self._endpoint}/models/unload",
                payload={"model": alias},
                timeout_sec=90.0,
            )
            if not 200 <= status < 300:
                self._release_failed = True
                raise RouterRuntimeError(
                    f"Router unload failed for {alias}: HTTP {status} {payload}"
                )
            states = self._wait_model_state_locked(
                alias,
                expected="unloaded",
                cancel_checker=cancel_checker,
            )
            if _loaded_count(states):
                self._release_failed = True
                raise RouterRuntimeError(
                    f"Router still has a loaded model after unloading {alias}."
                )
            self._active_model = None
            self._states = states
            self._release_failed = False
            return {
                "runtime_state": "sleeping",
                "gpu_release_expected": True,
                "router_model": alias,
                "router_unload_elapsed_ms": round(
                    (time.perf_counter() - started) * 1000.0,
                    3,
                ),
                "router_response": payload,
                "router_states": states,
            }

    def finish_pair(self, *, keep_container: bool) -> dict[str, Any]:
        with self._command_lock:
            if not self._prepared and not self._container_exists_locked():
                return {"runtime_state": "stopped", "container_present": False}
            if self._active_model:
                try:
                    self.unload_model(self._active_model)
                except Exception:
                    self._release_failed = True
                    self._stop_container_locked()
                    raise
            states = self._model_states_locked() if self._prepared else {}
            if _loaded_count(states):
                self._release_failed = True
                raise RouterRuntimeError(
                    "Router finish refused while a model remains loaded."
                )
            if keep_container:
                return {
                    "runtime_state": "sleeping",
                    "container_present": True,
                    "loaded_count": 0,
                    "router_states": states,
                }
            self._stop_container_locked()
            self._clear_pair_locked()
            return {"runtime_state": "stopped", "container_present": False}

    def stop_pair(self) -> dict[str, Any]:
        return self.finish_pair(keep_container=False)

    def _prepare_pair(
        self,
        pair: RouterPairSpec,
        ocr_identity: Mapping[str, Any],
        gemma_identity: Mapping[str, Any],
        *,
        cancel_checker: Callable[[], bool] | None,
    ) -> dict[str, Any]:
        with self._command_lock:
            self._select_pair_locked(pair, ocr_identity)
            contract = self._build_contract(pair, ocr_identity, gemma_identity)
            if self._release_failed:
                raise RouterRuntimeError(
                    "Router model release failed; no later model may be loaded."
                )
            if self._prepared and self._fingerprint != contract["fingerprint"]:
                self._stop_container_locked()
                self._clear_pair_locked(clear_selection=False)
            if not self._prepared:
                self._expected_contract = dict(contract)
                self._start_container_locked(pair, contract, cancel_checker=cancel_checker)
                self._prepared = True
                self._fingerprint = str(contract["fingerprint"])
                self._states = self._model_states_locked()
                if _loaded_count(self._states):
                    raise RouterRuntimeError(
                        "Router loaded a model before the explicit load contract."
                    )
            return dict(contract)

    def _probe_no_autoload_if_needed(
        self,
        pair: RouterPairSpec,
        *,
        cancel_checker: Callable[[], bool] | None,
    ) -> None:
        with self._command_lock:
            if self._autoload_probe_generation == self._generation:
                return
            if not self._prepared:
                raise RouterRuntimeError("Router is not prepared before no-autoload probe.")
            before = dict(self._states or self._model_states_locked())
        if cancel_checker and cancel_checker():
            raise OperationCancelledError("Cancelled before Router no-autoload probe.")
        query = urlencode({"model": pair.ocr_alias, "autoload": "false"})
        status, _payload = _json_http(
            f"{self._endpoint}/v1/chat/completions?{query}",
            payload={
                "model": pair.ocr_alias,
                "messages": [{"role": "user", "content": "router contract probe"}],
                "max_tokens": 1,
            },
            timeout_sec=20.0,
        )
        with self._command_lock:
            after = self._model_states_locked()
            if status < 400 or before != after or _loaded_count(after):
                self._release_failed = True
                raise RouterRuntimeError(
                    "Router accepted an unloaded request or implicitly loaded a model."
                )
            self._autoload_probe_generation = self._generation
            self._states = after

    def _load_model_locked(
        self,
        alias: str,
        *,
        cancel_checker: Callable[[], bool] | None,
    ) -> None:
        pair = self._pair_catalog().get(self._pair_key or "")
        allowed = {pair.ocr_alias, ROUTER_GEMMA_ALIAS} if pair else set()
        if alias not in allowed:
            raise RouterRuntimeError(f"Router model alias is not allowed: {alias!r}")
        if self._active_model == alias and self._states.get(alias) == "loaded":
            return
        states = self._model_states_locked()
        if _loaded_count(states):
            raise RouterRuntimeError(
                f"Cannot load {alias}; Router already has a model: {states}."
            )
        if cancel_checker and cancel_checker():
            raise OperationCancelledError(f"Cancelled before Router load of {alias}.")
        status, payload = _json_http(
            f"{self._endpoint}/models/load",
            payload={"model": alias},
            timeout_sec=90.0,
        )
        if not 200 <= status < 300:
            raise RouterRuntimeError(
                f"Router load failed for {alias}: HTTP {status} {payload}"
            )
        self._active_model = alias
        states = self._wait_model_state_locked(
            alias,
            expected="loaded",
            cancel_checker=cancel_checker,
        )
        if _loaded_count(states) != 1 or states.get(alias) != "loaded":
            raise RouterRuntimeError(
                f"Router loaded count is not exactly one after loading {alias}: {states}"
            )
        self._states = states
        self._verify_loaded_identity_locked(alias)

    def _wait_model_state_locked(
        self,
        alias: str,
        *,
        expected: str,
        cancel_checker: Callable[[], bool] | None,
    ) -> dict[str, str]:
        deadline = time.monotonic() + 420.0
        states: dict[str, str] = {}
        while time.monotonic() < deadline:
            if cancel_checker and cancel_checker():
                raise OperationCancelledError(
                    f"Cancelled while waiting for Router model {alias} {expected}."
                )
            states = self._model_states_locked()
            if states.get(alias) == expected:
                return states
            time.sleep(0.1)
        raise RouterRuntimeError(
            f"Router model state timed out: alias={alias} expected={expected} states={states}"
        )

    def _model_states_locked(self) -> dict[str, str]:
        status, payload = _json_http(f"{self._endpoint}/models", timeout_sec=15.0)
        if not 200 <= status < 300:
            raise RouterRuntimeError(f"Router /models failed: HTTP {status} {payload}")
        states = _model_states(payload)
        if not states:
            raise RouterRuntimeError("Router /models returned no model states.")
        allowed = {"loaded", "unloaded", "loading", "unloading"}
        unexpected = {key: value for key, value in states.items() if value not in allowed}
        if unexpected:
            raise RouterRuntimeError(f"Router returned unsupported model states: {unexpected}")
        return states

    def _verify_loaded_identity_locked(self, alias: str) -> None:
        query = urlencode({"model": alias, "autoload": "false"})
        status, payload = _json_http(
            f"{self._endpoint}/v1/models?{query}",
            timeout_sec=15.0,
        )
        if not 200 <= status < 300 or alias not in _model_ids(payload):
            self._release_failed = True
            raise RouterRuntimeError(
                f"Router loaded-model identity failed for {alias}: HTTP {status} {payload}"
            )
        for endpoint in ("props", "slots"):
            status, payload = _json_http(
                f"{self._endpoint}/{endpoint}?{query}",
                timeout_sec=15.0,
            )
            if not 200 <= status < 300:
                self._release_failed = True
                raise RouterRuntimeError(
                    f"Router {endpoint} readiness failed for {alias}: HTTP {status} {payload}"
                )
        status, payload = _json_http(
            f"{self._endpoint}/metrics?{query}",
            timeout_sec=15.0,
        )
        if not 200 <= status < 300 or not str(payload).strip():
            self._release_failed = True
            raise RouterRuntimeError(
                f"Router metrics readiness failed for {alias}: HTTP {status}"
            )

    def _start_container_locked(
        self,
        pair: RouterPairSpec,
        contract: Mapping[str, Any],
        *,
        cancel_checker: Callable[[], bool] | None,
    ) -> None:
        if not pair.compose_file.is_file() or not pair.preset_file.is_file():
            raise RouterRuntimeError(f"Router assets are missing for {pair.key}.")
        existing = self._inspect_container_locked()
        if existing is not None:
            if not self._container_matches_locked(existing, contract):
                raise RouterRuntimeError(
                    "Refusing to reuse a Router container with a foreign or stale contract."
                )
            state = existing.get("State") if isinstance(existing.get("State"), Mapping) else {}
            if not bool(state.get("Running")):
                run_docker_command(
                    ["docker", "start", ROUTER_CONTAINER_NAME],
                    timeout_sec=60.0,
                    cancel_checker=cancel_checker,
                )
            self._container_running = True
            self._wait_health_locked(cancel_checker=cancel_checker)
            self._verify_live_contract_locked(contract)
            return

        environment = self._compose_environment(pair, contract)
        try:
            run_docker_command(
                [
                    *resolve_docker_compose_command(cancel_checker=cancel_checker),
                    "-f",
                    str(pair.compose_file),
                    "up",
                    "-d",
                    "--no-build",
                ],
                cwd=pair.compose_file.parent,
                env=environment,
                timeout_sec=600.0,
                cancel_checker=cancel_checker,
            )
            self._container_running = True
            self._wait_health_locked(cancel_checker=cancel_checker)
            self._verify_live_contract_locked(contract)
        except BaseException:
            try:
                self._stop_container_locked()
            except BaseException as cleanup_error:
                raise RouterRuntimeError(
                    "Router startup failed and its owned container could not be cleaned up."
                ) from cleanup_error
            raise

    def _wait_health_locked(self, *, cancel_checker: Callable[[], bool] | None) -> None:
        deadline = time.monotonic() + 420.0
        last_error = ""
        while time.monotonic() < deadline:
            if cancel_checker and cancel_checker():
                raise OperationCancelledError("Cancelled while waiting for Router health.")
            try:
                status, _payload = _json_http(
                    f"{self._endpoint}/health",
                    timeout_sec=3.0,
                )
                if 200 <= status < 300:
                    return
                last_error = f"HTTP {status}"
            except RouterRuntimeError as exc:
                last_error = str(exc)
            time.sleep(0.25)
        raise RouterRuntimeError(f"Router health timed out: {last_error}")

    def _verify_live_contract_locked(self, contract: Mapping[str, Any]) -> None:
        inspected = self._inspect_container_locked()
        if inspected is None or not self._container_matches_locked(inspected, contract):
            raise RouterRuntimeError("Live Router container does not match its fingerprint contract.")

    def _inspect_container_locked(self) -> dict[str, Any] | None:
        completed = run_docker_command(
            ["docker", "inspect", ROUTER_CONTAINER_NAME, "--format", "{{json .}}"],
            check=False,
            timeout_sec=20.0,
        )
        if completed.returncode != 0:
            return None
        try:
            payload = json.loads(completed.stdout or "")
        except json.JSONDecodeError as exc:
            raise RouterRuntimeError("Router Docker inspection was invalid.") from exc
        if not isinstance(payload, Mapping):
            raise RouterRuntimeError("Router Docker inspection was not an object.")
        return dict(payload)

    def _container_exists_locked(self) -> bool:
        return self._inspect_container_locked() is not None

    def _container_matches_locked(
        self,
        inspected: Mapping[str, Any],
        contract: Mapping[str, Any],
    ) -> bool:
        config = inspected.get("Config") if isinstance(inspected.get("Config"), Mapping) else {}
        labels = config.get("Labels") if isinstance(config.get("Labels"), Mapping) else {}
        expected_labels = self._expected_labels(contract)
        if any(str(labels.get(key, "")) != value for key, value in expected_labels.items()):
            return False
        if str(inspected.get("Image", "")) != str(contract.get("image_id", "")):
            return False
        if tuple(str(item) for item in (config.get("Cmd") or [])) != _expected_command():
            return False
        if _canonical_sha256(list(config.get("Cmd") or [])) != str(
            contract.get("command_sha256", "")
        ):
            return False
        host_config = inspected.get("HostConfig") if isinstance(inspected.get("HostConfig"), Mapping) else {}
        if str(host_config.get("NetworkMode", "")) != "bridge":
            return False
        pair = self._pair_catalog().get(self._pair_key or "")
        if pair is None:
            return False
        network_settings = (
            inspected.get("NetworkSettings")
            if isinstance(inspected.get("NetworkSettings"), Mapping)
            else {}
        )
        port_bindings = network_settings.get("Ports")
        bindings = port_bindings.get("8080/tcp") if isinstance(port_bindings, Mapping) else None
        observed_ports = {
            (
                str(item.get("HostIp") or ""),
                str(item.get("HostPort") or ""),
            )
            for item in bindings
            if isinstance(item, Mapping)
        } if isinstance(bindings, list) else set()
        if {
            ("127.0.0.1", str(pair.ocr_port)),
            ("127.0.0.1", str(ROUTER_GEMMA_HOST_PORT)),
        } - observed_ports:
            return False
        mounts = inspected.get("Mounts") if isinstance(inspected.get("Mounts"), list) else []
        by_destination = {
            str(item.get("Destination", "")): item
            for item in mounts
            if isinstance(item, Mapping)
        }
        expected_mounts = {
            "/models/ocr": str(contract["ocr_volume"]),
            "/models/gemma": ROUTER_GEMMA_VOLUME,
            "/config/models.ini": None,
        }
        for destination, source in expected_mounts.items():
            mount = by_destination.get(destination)
            if not isinstance(mount, Mapping) or bool(mount.get("RW", True)):
                return False
            if source is not None and str(mount.get("Name", "")) != source:
                return False
        return True

    def _stop_container_locked(self) -> None:
        inspected = self._inspect_container_locked()
        if inspected is None:
            self._container_running = False
            return
        contract = self._expected_contract
        if contract is None or not self._container_matches_locked(inspected, contract):
            raise RouterRuntimeError(
                "Refusing to stop a Router container without the current ownership contract."
            )
        run_docker_command(
            [
                "docker",
                "stop",
                "--timeout",
                str(DEFAULT_MANAGED_RUNTIME_STOP_TIMEOUT_SEC),
                ROUTER_CONTAINER_NAME,
            ],
            check=False,
            timeout_sec=DEFAULT_MANAGED_RUNTIME_STOP_TIMEOUT_SEC + 15.0,
        )
        run_docker_command(
            ["docker", "rm", ROUTER_CONTAINER_NAME],
            check=False,
            timeout_sec=30.0,
        )
        if self._inspect_container_locked() is not None:
            raise RouterRuntimeError("Router container orphan remains after cleanup.")
        self._container_running = False

    def _select_pair_locked(
        self,
        pair: RouterPairSpec,
        ocr_identity: Mapping[str, Any],
    ) -> None:
        if self._release_failed:
            raise RouterRuntimeError(
                "Router ownership is retained after a release failure; pair change is blocked."
            )
        if self._pair_key == pair.key:
            self._pending_ocr_identity = dict(ocr_identity)
            return
        if self._active_model or _loaded_count(self._states):
            raise RouterRuntimeError(
                "Router pair change requires the current model to be unloaded first."
            )
        if self._pair_key or self._prepared or self._container_running:
            self._stop_container_locked()
        self._generation += 1
        self._pair_key = pair.key
        self._pending_ocr_identity = dict(ocr_identity)
        self._expected_contract = None
        self._prepared = False
        self._active_model = None
        self._states = {}
        self._fingerprint = ""
        self._autoload_probe_generation = None

    def _clear_pair_locked(self, *, clear_selection: bool = True) -> None:
        self._prepared = False
        self._container_running = False
        self._active_model = None
        self._states = {}
        self._fingerprint = ""
        self._expected_contract = None
        self._autoload_probe_generation = None
        if clear_selection:
            self._pair_key = None
            self._pending_ocr_identity = None

    def _build_contract(
        self,
        pair: RouterPairSpec,
        ocr_identity: Mapping[str, Any],
        gemma_identity: Mapping[str, Any],
    ) -> dict[str, Any]:
        image_ref = str(ocr_identity.get("image_ref") or DEFAULT_PADDLE_LLAMA_CPP_IMAGE)
        image_id = str(ocr_identity.get("image_id") or "")
        gemma_image_ref = str(gemma_identity.get("image_ref") or image_ref)
        gemma_image_id = str(gemma_identity.get("image_id") or image_id)
        if (
            not image_id
            or image_id != gemma_image_id
            or image_ref != gemma_image_ref
            or image_ref != DEFAULT_PADDLE_LLAMA_CPP_IMAGE
        ):
            raise RouterRuntimeError("OCR and Gemma Router image identities do not match.")
        if str(ocr_identity.get("model_sha256", "")) != pair.ocr_model_sha256:
            raise RouterRuntimeError("Router OCR model identity does not match the selected pair.")
        if str(ocr_identity.get("mmproj_sha256", "")) != pair.ocr_mmproj_sha256:
            raise RouterRuntimeError("Router OCR projector identity does not match the selected pair.")
        if str(gemma_identity.get("model_name", "")) != ROUTER_GEMMA_ALIAS:
            raise RouterRuntimeError("Router Gemma model alias does not match the product default.")
        if str(gemma_identity.get("model_sha256", "")) != ROUTER_GEMMA_MODEL_SHA256:
            raise RouterRuntimeError("Router Gemma model identity does not match the product default.")
        for label, value in (
            ("OCR model manifest", ocr_identity.get("manifest_sha256")),
            ("Gemma model manifest", gemma_identity.get("manifest_sha256")),
        ):
            normalized = str(value or "").strip().lower()
            if len(normalized) != 64 or any(
                char not in "0123456789abcdef" for char in normalized
            ):
                raise RouterRuntimeError(f"{label} SHA-256 is missing or invalid.")
        ocr_options = dict(ocr_identity.get("runtime_options") or {})
        gemma_options = dict(gemma_identity.get("runtime_options") or {})
        if ocr_options and not self._ocr_options_match(ocr_options):
            raise RouterRuntimeError("Non-default Paddle runtime options are not eligible for the Router.")
        if gemma_options and not self._gemma_options_match(gemma_options):
            raise RouterRuntimeError("Non-default Gemma runtime options are not eligible for the Router.")
        compose_sha256 = _file_sha256(pair.compose_file)
        preset_sha256 = _file_sha256(pair.preset_file)
        command = list(_expected_command())
        command_sha256 = _canonical_sha256(command)
        fingerprint_payload = {
            "protocol": ROUTER_PROTOCOL_VERSION,
            "pair": pair.key,
            "image_ref": image_ref,
            "image_digest": image_ref.rsplit("@", 1)[-1],
            "image_id": image_id,
            "ocr": {
                "alias": pair.ocr_alias,
                "model": pair.ocr_model,
                "model_sha256": pair.ocr_model_sha256,
                "mmproj": pair.ocr_mmproj,
                "mmproj_sha256": pair.ocr_mmproj_sha256,
                "manifest_sha256": str(ocr_identity.get("manifest_sha256", "")),
                "volume": str(ocr_identity.get("volume") or pair.ocr_volume),
            },
            "gemma": {
                "alias": ROUTER_GEMMA_ALIAS,
                "model_sha256": ROUTER_GEMMA_MODEL_SHA256,
                "manifest_sha256": str(gemma_identity.get("manifest_sha256", "")),
                "volume": str(gemma_identity.get("volume") or ROUTER_GEMMA_VOLUME),
            },
            "compose_sha256": compose_sha256,
            "preset_sha256": preset_sha256,
            "command_sha256": command_sha256,
        }
        return {
            "fingerprint": _canonical_sha256(fingerprint_payload),
            "fingerprint_payload": fingerprint_payload,
            "image_ref": image_ref,
            "image_id": image_id,
            "image_digest": image_ref.rsplit("@", 1)[-1],
            "ocr_volume": str(ocr_identity.get("volume") or pair.ocr_volume),
            "gemma_volume": str(gemma_identity.get("volume") or ROUTER_GEMMA_VOLUME),
            "ocr_model_sha256": pair.ocr_model_sha256,
            "ocr_mmproj_sha256": pair.ocr_mmproj_sha256,
            "ocr_manifest_sha256": str(ocr_identity.get("manifest_sha256", "")),
            "gemma_model_sha256": ROUTER_GEMMA_MODEL_SHA256,
            "gemma_manifest_sha256": str(gemma_identity.get("manifest_sha256", "")),
            "compose_sha256": compose_sha256,
            "preset_sha256": preset_sha256,
            "command_sha256": command_sha256,
            "command": command,
        }

    @staticmethod
    def _ocr_options_match(options: Mapping[str, Any]) -> bool:
        aliases = {
            "PADDLEOCR_LLAMA_CTX_SIZE": "context",
            "PADDLEOCR_LLAMA_PARALLEL": "parallel",
            "PADDLEOCR_LLAMA_THREADS": "threads",
            "PADDLEOCR_LLAMA_BATCH_SIZE": "batch",
            "PADDLEOCR_LLAMA_UBATCH_SIZE": "ubatch",
            "PADDLEOCR_LLAMA_GPU_LAYERS": "gpu_layers",
            "PADDLEOCR_LLAMA_SLEEP_IDLE_SECONDS": "sleep_idle_seconds",
            "PADDLEOCR_SPOTTING_LLAMA_CTX_SIZE": "context",
            "PADDLEOCR_SPOTTING_LLAMA_PARALLEL": "parallel",
            "PADDLEOCR_SPOTTING_LLAMA_THREADS": "threads",
            "PADDLEOCR_SPOTTING_LLAMA_BATCH_SIZE": "batch",
            "PADDLEOCR_SPOTTING_LLAMA_UBATCH_SIZE": "ubatch",
            "PADDLEOCR_SPOTTING_LLAMA_GPU_LAYERS": "gpu_layers",
            "PADDLEOCR_SPOTTING_LLAMA_SLEEP_IDLE_SECONDS": "sleep_idle_seconds",
        }
        for key, name in aliases.items():
            if str(options.get(key, DEFAULT_ROUTER_OCR_OPTIONS[name])) != str(
                DEFAULT_ROUTER_OCR_OPTIONS[name]
            ):
                return False
        return True

    @staticmethod
    def _gemma_options_match(options: Mapping[str, Any]) -> bool:
        aliases = {
            "LLAMA_CTX_SIZE": "context",
            "LLAMA_N_PARALLEL": "parallel",
            "LLAMA_THREADS": "threads",
            "LLAMA_BATCH_SIZE": "batch",
            "LLAMA_UBATCH_SIZE": "ubatch",
            "LLAMA_N_GPU_LAYERS": "gpu_layers",
            "LLAMA_CACHE_TYPE_K": "cache_type_k",
            "LLAMA_CACHE_TYPE_V": "cache_type_v",
            "LLAMA_CACHE_RAM_MIB": "cache_ram_mib",
            "LLAMA_SPEC_TYPE": "spec_type",
            "LLAMA_SPEC_DRAFT_N_MAX": "spec_draft_n_max",
        }
        for key, name in aliases.items():
            if str(options.get(key, DEFAULT_ROUTER_GEMMA_OPTIONS[name])) != str(DEFAULT_ROUTER_GEMMA_OPTIONS[name]):
                return False
        return True

    def _compose_environment(
        self,
        pair: RouterPairSpec,
        contract: Mapping[str, Any],
    ) -> dict[str, str]:
        environment = dict(os.environ)
        environment.update(
            {
                "ROUTER_LLAMA_CPP_IMAGE": str(contract["image_ref"]),
                "ROUTER_CONTAINER_NAME": ROUTER_CONTAINER_NAME,
                "ROUTER_PROTOCOL_VERSION": ROUTER_PROTOCOL_VERSION,
                "ROUTER_PAIR": pair.key,
                "ROUTER_PAIR_FINGERPRINT": str(contract["fingerprint"]),
                "ROUTER_OCR_MODEL_VOLUME": str(contract["ocr_volume"]),
                "ROUTER_GEMMA_MODEL_VOLUME": str(contract["gemma_volume"]),
                "ROUTER_OCR_MODEL_SHA256": str(contract["ocr_model_sha256"]),
                "ROUTER_OCR_MMPROJ_SHA256": str(contract["ocr_mmproj_sha256"]),
                "ROUTER_OCR_MANIFEST_SHA256": str(contract["ocr_manifest_sha256"]),
                "ROUTER_GEMMA_MODEL_SHA256": str(contract["gemma_model_sha256"]),
                "ROUTER_GEMMA_MANIFEST_SHA256": str(contract["gemma_manifest_sha256"]),
                "ROUTER_COMMAND_SHA256": str(contract["command_sha256"]),
                "ROUTER_OCR_HOST_PORT": str(pair.ocr_port),
                "ROUTER_GEMMA_HOST_PORT": str(ROUTER_GEMMA_HOST_PORT),
            }
        )
        return environment

    def _expected_labels(self, contract: Mapping[str, Any]) -> dict[str, str]:
        return {
            ROUTER_LABEL_PROTOCOL: ROUTER_PROTOCOL_VERSION,
            ROUTER_LABEL_PAIR: str(self._pair_key or ""),
            ROUTER_LABEL_FINGERPRINT: str(contract["fingerprint"]),
            ROUTER_LABEL_OCR_VOLUME: str(contract["ocr_volume"]),
            ROUTER_LABEL_GEMMA_VOLUME: str(contract["gemma_volume"]),
            ROUTER_LABEL_OCR_MODEL_SHA256: str(contract["ocr_model_sha256"]),
            ROUTER_LABEL_OCR_MMPROJ_SHA256: str(contract["ocr_mmproj_sha256"]),
            ROUTER_LABEL_OCR_MANIFEST_SHA256: str(contract["ocr_manifest_sha256"]),
            ROUTER_LABEL_GEMMA_MODEL_SHA256: str(contract["gemma_model_sha256"]),
            ROUTER_LABEL_GEMMA_MANIFEST_SHA256: str(contract["gemma_manifest_sha256"]),
            ROUTER_LABEL_COMMAND_SHA256: str(contract["command_sha256"]),
        }

    def _gemma_identity(self, settings_page: Any) -> dict[str, Any] | None:
        provider = self._gemma_identity_provider
        if not callable(provider):
            return None
        try:
            value = provider(settings_page)
        except (OperationCancelledError, RouterRuntimeError):
            raise
        except Exception as exc:
            raise RouterRuntimeError(
                "Gemma Router identity provider failed."
            ) from exc
        return dict(value) if isinstance(value, Mapping) else None

    @staticmethod
    def _pair_catalog() -> dict[str, RouterPairSpec]:
        return _pair_catalog()

    @classmethod
    def _pair_for_engine(cls, engine_key: str) -> RouterPairSpec | None:
        return next(
            (pair for pair in cls._pair_catalog().values() if pair.engine_key == engine_key),
            None,
        )

    @staticmethod
    def _ocr_url(engine_key: str, settings_page: Any) -> str:
        if engine_key == "PaddleOCR VL":
            return str(settings_page.get_paddleocr_vl_settings().get("server_url", ""))
        if engine_key == "PaddleOCR VL Spotting":
            return str(
                settings_page.get_paddleocr_vl_spotting_settings().get("server_url", "")
            )
        return ""

    @property
    def _endpoint(self) -> str:
        return f"http://127.0.0.1:{ROUTER_GEMMA_HOST_PORT}"


__all__ = [
    "LocalLlamaRouterCoordinator",
    "RouterPairSpec",
    "RouterRuntimeError",
    "RouterRuntimeSnapshot",
    "ROUTER_CONTAINER_NAME",
    "ROUTER_GEMMA_ALIAS",
    "ROUTER_GEMMA_HOST_PORT",
    "ROUTER_GEMMA_VOLUME",
]

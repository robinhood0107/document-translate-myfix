#!/usr/bin/env python3
"""Compare Paddle crop OCR through PaddleX relay and direct llama.cpp.

This benchmark-only runner reuses frozen detector snapshots.  Both transports
receive the same product JPEG crop; the direct adapter reproduces PaddleX's
official PNG conversion and image-first llama.cpp request.  Results must be
written outside Git.
"""

from __future__ import annotations

import argparse
import base64
import copy
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

import cv2
import numpy as np
import requests


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import benchmark_ocr_three_way_human_truth as truth_tools
from modules.ocr.paddle_crop.engine import PaddleOCRVLEngine
from modules.ocr.paddle_crop.response_parser import normalize_output_text
from modules.utils.exceptions import (
    LocalServiceConnectionError,
    LocalServiceResponseError,
)
from modules.utils.textblock import TextBlock


PROTOCOL_VERSION = "paddle-crop-transport-v1"
DIRECT_ENDPOINT = "http://127.0.0.1:18000/v1/chat/completions"
RELAY_ENDPOINT = "http://127.0.0.1:28118/layout-parsing"
DIRECT_HEALTH_URL = "http://127.0.0.1:18000/health"
RELAY_HEALTH_URL = "http://127.0.0.1:28118/docs"
MODEL_ALIAS = "PaddleOCR-VL-1.6-0.9B"
CONTAINERS = ("paddleocr-server", "paddleocr-llamacpp")
TRANSPORTS = ("relay", "direct")
EXPECTED_LLAMA_IMAGE_ID = (
    "sha256:22e0e3bfe967af4fd1df6a918022abbfd4e72e4d40a4769e616a4176790acbcb"
)
EXPECTED_RELAY_IMAGE_ID = (
    "sha256:d0d32c04a2119613d25a0a4c292e165ccc107954b74580613cf59e378037f8f5"
)
EXPECTED_MODEL_SHA256 = (
    "f3ae46ec885050acf4b3d31944431e1fd90d50664fb09126af4a3c050ba14ee8"
)
EXPECTED_MMPROJ_SHA256 = (
    "204d757d7610d9b3faab10d506d69e5b244e32bf765e2bab2d0167e65e0a058a"
)
EXPECTED_MODEL_VOLUME = "comic-translate-paddleocr-vl-llamacpp-models-v1"


class TransportContractError(ValueError):
    """Raised when benchmark input or runtime evidence is invalid."""


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _external_empty_directory(path: Path, *, label: str) -> Path:
    output = truth_tools.require_external_path(path, label=label)
    if output.exists() and any(output.iterdir()):
        raise TransportContractError(f"{label} must be empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    return output


def _extract_chat_content(payload: Mapping[str, Any]) -> tuple[str, str]:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise LocalServiceResponseError(
            "Direct Paddle llama.cpp response did not include choices.",
            service_name="PaddleOCR VL",
            settings_page_name="PaddleOCR VL Settings",
        )
    choice = choices[0]
    if not isinstance(choice, dict):
        raise LocalServiceResponseError(
            "Direct Paddle llama.cpp response choice is invalid.",
            service_name="PaddleOCR VL",
            settings_page_name="PaddleOCR VL Settings",
        )
    message = choice.get("message")
    if not isinstance(message, dict):
        raise LocalServiceResponseError(
            "Direct Paddle llama.cpp response has no message payload.",
            service_name="PaddleOCR VL",
            settings_page_name="PaddleOCR VL Settings",
        )
    content = message.get("content", "")
    if isinstance(content, list):
        content = "\n".join(
            str(item.get("text", "") or "")
            for item in content
            if isinstance(item, dict) and item.get("type") == "text"
        )
    return (
        str(content or "").strip(),
        str(choice.get("finish_reason", "") or "").strip().lower(),
    )


def _direct_payload(image_b64: str, *, max_tokens: int) -> dict[str, Any]:
    return {
        "model": MODEL_ALIAS,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/png;base64," + image_b64
                        },
                    },
                    {"type": "text", "text": "OCR:"},
                ],
            }
        ],
        "temperature": 0,
        "max_tokens": int(max_tokens),
        "stream": False,
    }


def _relay_compatible_text(content: str) -> str:
    paragraph_lines = re.sub(r"(?<!\n)\n(?!\n)", "\n\n", content)
    return normalize_output_text(paragraph_lines)


class DirectPaddleOCRVLEngine(PaddleOCRVLEngine):
    """Product crop/guard/parser behavior with the official direct OCR call."""

    def __init__(self, endpoint: str = DIRECT_ENDPOINT) -> None:
        super().__init__()
        self.direct_endpoint = str(endpoint)

    def _request_ocr_text_from_encoded(self, image_bytes: bytes) -> str:
        record = self._current_request_record()
        self._add_request_metric(record, "logical_request_count", 1)
        self._add_request_metric(record, "request_bytes", len(image_bytes))
        started = time.perf_counter()
        decoded = cv2.imdecode(
            np.frombuffer(image_bytes, dtype=np.uint8),
            cv2.IMREAD_COLOR,
        )
        if decoded is None:
            raise LocalServiceResponseError(
                "Unable to decode the product JPEG for direct Paddle OCR.",
                service_name="PaddleOCR VL",
                settings_page_name="PaddleOCR VL Settings",
            )
        encoded, png = cv2.imencode(".png", decoded)
        if not encoded:
            raise LocalServiceResponseError(
                "Unable to encode the official direct Paddle PNG.",
                service_name="PaddleOCR VL",
                settings_page_name="PaddleOCR VL Settings",
            )
        image_b64 = base64.b64encode(png.tobytes()).decode("ascii")
        self._add_request_metric(
            record,
            "base64_ms",
            (time.perf_counter() - started) * 1000.0,
        )
        self._add_request_metric(record, "base64_chars", len(image_b64))
        started = time.perf_counter()
        payload = _direct_payload(
            image_b64,
            max_tokens=self.max_new_tokens,
        )
        self._add_request_metric(
            record,
            "payload_build_ms",
            (time.perf_counter() - started) * 1000.0,
        )

        response_payload = self._send_direct_request(
            payload,
            telemetry_record=record,
        )
        parse_started = time.perf_counter()
        content, finish_reason = _extract_chat_content(response_payload)
        self._add_request_metric(
            record,
            "parse_sanitize_ms",
            (time.perf_counter() - parse_started) * 1000.0,
        )
        if finish_reason == "length":
            raise LocalServiceResponseError(
                "Direct Paddle llama.cpp OCR response was truncated.",
                service_name="PaddleOCR VL",
                settings_page_name="PaddleOCR VL Settings",
            )
        return _relay_compatible_text(content)

    def _send_direct_request(
        self,
        payload: dict[str, Any],
        *,
        telemetry_record: dict[str, Any] | None,
    ) -> dict[str, Any]:
        record = telemetry_record or self._current_request_record()
        last_status = 0
        last_detail = ""
        for attempt_index in range(self.REQUEST_RETRY_TOTAL_ATTEMPTS):
            self._raise_if_cancelled()
            request_started = time.perf_counter()
            self._add_request_metric(record, "http_attempt_count", 1)
            try:
                response = requests.post(
                    self.direct_endpoint,
                    json=payload,
                    headers={"Content-Type": "application/json"},
                    timeout=self.REQUEST_TIMEOUT_SECONDS,
                )
            except requests.exceptions.RequestException as exc:
                self._add_request_metric(
                    record,
                    "request_wall_ms",
                    (time.perf_counter() - request_started) * 1000.0,
                )
                if self._should_retry_request(attempt_index):
                    self._add_request_metric(record, "http_retry_count", 1)
                    backoff_started = time.perf_counter()
                    self._sleep_before_retry(attempt_index, exc)
                    self._add_request_metric(
                        record,
                        "retry_backoff_ms",
                        (time.perf_counter() - backoff_started) * 1000.0,
                    )
                    continue
                raise LocalServiceConnectionError(
                    "Unable to reach the direct Paddle llama.cpp service.",
                    service_name="PaddleOCR VL",
                    settings_page_name="PaddleOCR VL Settings",
                ) from exc
            self._add_request_metric(
                record,
                "request_wall_ms",
                (time.perf_counter() - request_started) * 1000.0,
            )
            last_status = int(response.status_code)
            if response.status_code == 200:
                decode_started = time.perf_counter()
                try:
                    decoded = response.json()
                except ValueError as exc:
                    raise LocalServiceResponseError(
                        "Direct Paddle llama.cpp returned invalid JSON.",
                        service_name="PaddleOCR VL",
                        settings_page_name="PaddleOCR VL Settings",
                    ) from exc
                finally:
                    self._add_request_metric(
                        record,
                        "response_decode_ms",
                        (time.perf_counter() - decode_started) * 1000.0,
                    )
                if not isinstance(decoded, dict):
                    raise LocalServiceResponseError(
                        "Direct Paddle llama.cpp response must be an object.",
                        service_name="PaddleOCR VL",
                        settings_page_name="PaddleOCR VL Settings",
                    )
                return decoded
            last_detail = (response.text or "").strip()[:1000]
            if (
                response.status_code in self.TRANSIENT_HTTP_STATUS_CODES
                and self._should_retry_request(attempt_index)
            ):
                self._add_request_metric(record, "http_retry_count", 1)
                backoff_started = time.perf_counter()
                self._sleep_before_retry(attempt_index, response)
                self._add_request_metric(
                    record,
                    "retry_backoff_ms",
                    (time.perf_counter() - backoff_started) * 1000.0,
                )
                continue
            break
        raise LocalServiceResponseError(
            (
                f"Direct Paddle llama.cpp returned HTTP {last_status}."
                + (f" {last_detail}" if last_detail else "")
            ),
            service_name="PaddleOCR VL",
            settings_page_name="PaddleOCR VL Settings",
        )


class _Settings:
    def __init__(self, *, server_url: str, workers: int, max_tokens: int) -> None:
        self.server_url = server_url
        self.workers = workers
        self.max_tokens = max_tokens

    def get_paddleocr_vl_settings(self) -> dict[str, Any]:
        return {
            "server_url": self.server_url,
            "parallel_workers": self.workers,
            "max_new_tokens": self.max_tokens,
            "prettify_markdown": False,
            "visualize": False,
        }

    @staticmethod
    def value(_key: str, default: Any = None, type: Any = None) -> Any:
        del type
        return default


def _text_blocks(
    records: Sequence[Mapping[str, Any]],
    *,
    language: str,
) -> list[TextBlock]:
    blocks: list[TextBlock] = []
    for index, record in enumerate(records):
        bbox = truth_tools._bbox(
            record.get("bbox_xyxy"),
            label=f"detector block {index}",
            width=10**9,
            height=10**9,
        )
        bubble = record.get("bubble_xyxy")
        bubble_bbox = None
        if bubble is not None:
            bubble_bbox = np.asarray(
                truth_tools._bbox(
                    bubble,
                    label=f"detector bubble {index}",
                    width=10**9,
                    height=10**9,
                ),
                dtype=np.int32,
            )
        blocks.append(
            TextBlock(
                text_bbox=np.asarray(bbox, dtype=np.int32),
                bubble_bbox=bubble_bbox,
                text_class=str(record.get("text_class", "") or ""),
                source_lang=language,
                direction=str(record.get("direction", "") or ""),
                block_id=str(record.get("block_id", "") or ""),
                semantic_role=str(record.get("semantic_role", "") or ""),
                processing_action=str(
                    record.get("processing_action", "") or ""
                ),
            )
        )
    return blocks


def _select_manifest_pages(
    pages: Sequence[Mapping[str, Any]],
    page_ids: Sequence[str],
) -> list[Mapping[str, Any]]:
    selected = {str(value) for value in page_ids}
    selected_pages = [
        page
        for page in pages
        if not selected or str(page["page_id"]) in selected
    ]
    found = {str(page["page_id"]) for page in selected_pages}
    if selected and selected != found:
        raise TransportContractError(
            f"Unknown page IDs: {sorted(selected - found)}"
        )
    if not selected_pages:
        raise TransportContractError("Transport run selected no pages.")
    return selected_pages


def _serialize_block(block: TextBlock) -> dict[str, Any]:
    bubble = getattr(block, "bubble_xyxy", None)
    return {
        "block_id": str(getattr(block, "block_id", "") or ""),
        "xyxy": [int(float(value)) for value in block.xyxy],
        "bubble_xyxy": (
            [int(float(value)) for value in bubble]
            if bubble is not None
            else None
        ),
        "text_class": str(getattr(block, "text_class", "") or ""),
        "source_lang": str(getattr(block, "source_lang", "") or ""),
        "direction": str(getattr(block, "direction", "") or ""),
        "text": str(getattr(block, "text", "") or ""),
        "ocr_status": str(getattr(block, "ocr_status", "") or ""),
        "ocr_empty_reason": str(
            getattr(block, "ocr_empty_reason", "") or ""
        ),
        "ocr_raw_text": str(getattr(block, "ocr_raw_text", "") or ""),
        "ocr_effective_crop_xyxy": list(
            getattr(block, "ocr_effective_crop_xyxy", []) or []
        ),
        "ocr_crop_source": str(
            getattr(block, "ocr_crop_source", "") or ""
        ),
        "ocr_geometry_provenance": copy.deepcopy(
            getattr(block, "ocr_geometry_provenance", {}) or {}
        ),
        "semantic_role": str(
            getattr(block, "semantic_role", "") or ""
        ),
        "processing_action": str(
            getattr(block, "processing_action", "") or ""
        ),
    }


def _run_page(
    *,
    engine: PaddleOCRVLEngine,
    page: Mapping[str, Any],
) -> dict[str, Any]:
    source = Path(str(page["source_image"]["path"]))
    if truth_tools.sha256_file(source) != page["source_image"]["sha256"]:
        raise TransportContractError(f"Source image changed: {source}")
    image = cv2.imread(str(source), cv2.IMREAD_COLOR)
    if image is None:
        raise TransportContractError(f"Unable to decode source image: {source}")
    expected_shape = (
        int(page["source_image"]["height"]),
        int(page["source_image"]["width"]),
    )
    if tuple(image.shape[:2]) != expected_shape:
        raise TransportContractError(f"Source dimensions changed: {source}")
    detector = truth_tools._detector_blocks_for_page(page)
    blocks = _text_blocks(detector, language=str(page["language"]))
    started = time.perf_counter()
    status = "success"
    error = ""
    try:
        engine.process_image(image, blocks)
    except Exception as exc:
        status = "failure"
        error = f"{type(exc).__name__}: {exc}"
    elapsed = time.perf_counter() - started
    return {
        "protocol_version": PROTOCOL_VERSION,
        "page_id": str(page["page_id"]),
        "source_path": str(source),
        "source_sha256": str(page["source_image"]["sha256"]),
        "shape_hw": list(expected_shape),
        "request_seconds": round(elapsed, 6),
        "status": status,
        "error": error,
        "block_count": len(blocks),
        "non_empty_count": sum(bool(block.text) for block in blocks),
        "page_profile": copy.deepcopy(engine.last_page_profile),
        "blocks": [_serialize_block(block) for block in blocks],
    }


def _docker(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["docker", *args],
        check=check,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


def _stop_runtime() -> None:
    existing: list[str] = []
    for name in CONTAINERS:
        inspected = _docker("inspect", name, check=False)
        if inspected.returncode == 0:
            existing.append(name)
    if existing:
        _docker("stop", *existing)


def _health_ready(url: str) -> bool:
    try:
        with urlopen(url, timeout=2) as response:
            return 200 <= int(getattr(response, "status", 200)) < 400
    except (HTTPError, URLError, OSError, ValueError):
        return False


def _start_runtime(transport: str, *, timeout_sec: int) -> float:
    if transport not in TRANSPORTS:
        raise TransportContractError(f"Unknown transport: {transport}")
    names = ["paddleocr-llamacpp"]
    health_urls = [DIRECT_HEALTH_URL]
    if transport == "relay":
        names.append("paddleocr-server")
        health_urls.append(RELAY_HEALTH_URL)
    started = time.perf_counter()
    _docker("start", *names)
    deadline = time.monotonic() + max(1, int(timeout_sec))
    while time.monotonic() < deadline:
        if all(_health_ready(url) for url in health_urls):
            return time.perf_counter() - started
        time.sleep(0.5)
    raise TransportContractError(
        f"{transport} runtime did not become healthy within {timeout_sec}s."
    )


def _runtime_snapshot(transport: str) -> dict[str, Any]:
    names = ["paddleocr-llamacpp"]
    if transport == "relay":
        names.append("paddleocr-server")
    containers: list[dict[str, Any]] = []
    for name in names:
        inspected = _docker("inspect", name)
        try:
            decoded = json.loads(inspected.stdout)
        except json.JSONDecodeError as exc:
            raise TransportContractError(
                f"docker inspect returned invalid JSON for {name}."
            ) from exc
        if not isinstance(decoded, list) or len(decoded) != 1:
            raise TransportContractError(
                f"Unexpected docker inspect result for {name}."
            )
        record = decoded[0]
        config = record.get("Config") if isinstance(record, dict) else None
        state = record.get("State") if isinstance(record, dict) else None
        labels = config.get("Labels") if isinstance(config, dict) else None
        mounts = record.get("Mounts") if isinstance(record, dict) else None
        containers.append(
            {
                "name": name,
                "container_id": str(record.get("Id", "") or ""),
                "image_id": str(record.get("Image", "") or ""),
                "configured_image": str(
                    config.get("Image", "") if isinstance(config, dict) else ""
                ),
                "entrypoint": copy.deepcopy(
                    config.get("Entrypoint") if isinstance(config, dict) else None
                ),
                "command": copy.deepcopy(
                    config.get("Cmd") if isinstance(config, dict) else None
                ),
                "state": str(
                    state.get("Status", "") if isinstance(state, dict) else ""
                ),
                "runtime_labels": {
                    str(key): str(value)
                    for key, value in (labels or {}).items()
                    if str(key).startswith("com.comictranslate.")
                },
                "mounts": [
                    {
                        "type": str(mount.get("Type", "") or ""),
                        "name": str(mount.get("Name", "") or ""),
                        "destination": str(
                            mount.get("Destination", "") or ""
                        ),
                        "mode": str(mount.get("Mode", "") or ""),
                        "rw": bool(mount.get("RW", False)),
                    }
                    for mount in (mounts or [])
                    if isinstance(mount, dict)
                ],
            }
        )
    return {"captured_at": utc_now(), "containers": containers}


def _validate_runtime_snapshot(
    snapshot: Mapping[str, Any],
    *,
    transport: str,
) -> None:
    raw_containers = snapshot.get("containers")
    if not isinstance(raw_containers, list):
        raise TransportContractError("Runtime snapshot has no containers.")
    containers = {
        str(record.get("name", "")): record
        for record in raw_containers
        if isinstance(record, dict)
    }
    expected_names = {"paddleocr-llamacpp"}
    if transport == "relay":
        expected_names.add("paddleocr-server")
    if set(containers) != expected_names:
        raise TransportContractError(
            "Runtime container set does not match the transport contract."
        )

    llama = containers["paddleocr-llamacpp"]
    if llama.get("image_id") != EXPECTED_LLAMA_IMAGE_ID:
        raise TransportContractError("Paddle llama.cpp image digest changed.")
    if llama.get("state") != "running":
        raise TransportContractError("Paddle llama.cpp container is not running.")
    labels = llama.get("runtime_labels")
    expected_labels = {
        "com.comictranslate.paddleocr-model-sha256": EXPECTED_MODEL_SHA256,
        "com.comictranslate.paddleocr-mmproj-sha256": EXPECTED_MMPROJ_SHA256,
        "com.comictranslate.paddleocr-model-volume": EXPECTED_MODEL_VOLUME,
    }
    if not isinstance(labels, dict) or any(
        labels.get(key) != value for key, value in expected_labels.items()
    ):
        raise TransportContractError("Paddle model runtime labels changed.")
    mounts = llama.get("mounts")
    if not isinstance(mounts, list) or not any(
        isinstance(mount, dict)
        and mount.get("type") == "volume"
        and mount.get("name") == EXPECTED_MODEL_VOLUME
        and mount.get("destination") == "/models"
        and mount.get("rw") is False
        for mount in mounts
    ):
        raise TransportContractError("Paddle named-volume mount contract changed.")
    command = llama.get("command")
    required_command_tokens = {
        "/models/PaddleOCR-VL-1.6-GGUF.gguf",
        "/models/PaddleOCR-VL-1.6-GGUF-mmproj.gguf",
        MODEL_ALIAS,
        "--n-gpu-layers",
        "all",
    }
    if not isinstance(command, list) or not required_command_tokens.issubset(
        {str(value) for value in command}
    ):
        raise TransportContractError("Paddle llama.cpp command contract changed.")

    if transport == "relay":
        relay = containers["paddleocr-server"]
        if relay.get("image_id") != EXPECTED_RELAY_IMAGE_ID:
            raise TransportContractError("PaddleX relay image digest changed.")
        relay_command = relay.get("command")
        if (
            relay.get("state") != "running"
            or not isinstance(relay_command, list)
            or not any("paddlex --serve" in str(value) for value in relay_command)
        ):
            raise TransportContractError("PaddleX relay command contract changed.")


@dataclass(frozen=True)
class RunConfig:
    transport: str
    workers: int
    max_tokens: int
    timeout_sec: int


def run_transport(
    *,
    manifest_path: Path,
    output_dir: Path,
    config: RunConfig,
    page_ids: Sequence[str] = (),
) -> dict[str, Any]:
    manifest = truth_tools.validate_corpus_manifest(manifest_path)
    output = _external_empty_directory(output_dir, label="transport output")
    pages = _select_manifest_pages(manifest["pages"], page_ids)

    engine: PaddleOCRVLEngine
    endpoint = RELAY_ENDPOINT
    if config.transport == "direct":
        endpoint = DIRECT_ENDPOINT
        engine = DirectPaddleOCRVLEngine(endpoint)
    elif config.transport == "relay":
        engine = PaddleOCRVLEngine()
    else:
        raise TransportContractError(
            f"Unknown transport: {config.transport}"
        )
    engine.initialize(
        _Settings(
            server_url=endpoint,
            workers=config.workers,
            max_tokens=config.max_tokens,
        )
    )

    _stop_runtime()
    startup_seconds = 0.0
    run_started = time.perf_counter()
    results: list[dict[str, Any]] = []
    runtime_snapshot: dict[str, Any] = {}
    try:
        startup_seconds = _start_runtime(
            config.transport,
            timeout_sec=config.timeout_sec,
        )
        runtime_snapshot = _runtime_snapshot(config.transport)
        _validate_runtime_snapshot(
            runtime_snapshot,
            transport=config.transport,
        )
        for page in pages:
            page_result = _run_page(engine=engine, page=page)
            page_dir = output / str(page["page_id"])
            _write_json(page_dir / "result.json", page_result)
            results.append(page_result)
            if page_result["status"] != "success":
                raise TransportContractError(
                    f"OCR failed on {page['page_id']}: {page_result['error']}"
                )
    finally:
        _stop_runtime()
    wall_seconds = time.perf_counter() - run_started
    summary = {
        "schema_version": 1,
        "protocol_version": PROTOCOL_VERSION,
        "created_at": utc_now(),
        "transport": config.transport,
        "backend": "llama.cpp",
        "endpoint": endpoint,
        "model_alias": MODEL_ALIAS,
        "prompt": "OCR:",
        "special_tokens": False,
        "llama_cpp_image_format": "PNG",
        "message_content_order": ["image_url", "text"],
        "workers": config.workers,
        "max_tokens": config.max_tokens,
        "manifest_path": str(manifest_path),
        "manifest_sha256": truth_tools.sha256_file(manifest_path),
        "page_ids": [str(page["page_id"]) for page in pages],
        "page_count": len(results),
        "block_count": sum(len(page["blocks"]) for page in results),
        "startup_seconds": round(startup_seconds, 6),
        "request_seconds": round(
            sum(float(page["request_seconds"]) for page in results),
            6,
        ),
        "wall_seconds": round(wall_seconds, 6),
        "http_attempt_count": sum(
            int(
                page.get("page_profile", {})
                .get("performance", {})
                .get("http_attempt_count", 0)
                or 0
            )
            for page in results
        ),
        "http_retry_count": sum(
            int(
                page.get("page_profile", {})
                .get("performance", {})
                .get("http_retry_count", 0)
                or 0
            )
            for page in results
        ),
        "runtime_snapshot": runtime_snapshot,
    }
    _write_json(output / "summary.json", summary)
    return summary


def _result_blocks(root: Path, page_id: str) -> dict[str, dict[str, Any]]:
    payload = truth_tools.read_json(root / page_id / "result.json")
    blocks = payload.get("blocks")
    if not isinstance(blocks, list):
        raise TransportContractError(f"Missing blocks for {page_id}: {root}")
    result: dict[str, dict[str, Any]] = {}
    for block in blocks:
        if not isinstance(block, dict):
            raise TransportContractError(f"Invalid block for {page_id}: {root}")
        block_id = str(block.get("block_id", "") or "")
        if not block_id or block_id in result:
            raise TransportContractError(
                f"Invalid or duplicate block ID for {page_id}: {block_id}"
            )
        result[block_id] = block
    return result


def _selected_result_pages(root: Path) -> set[str] | None:
    summary_path = root / "summary.json"
    if not summary_path.is_file():
        return None
    summary = truth_tools.read_json(summary_path)
    page_ids = summary.get("page_ids")
    if page_ids is None:
        inferred = {
            child.name
            for child in root.iterdir()
            if child.is_dir() and (child / "result.json").is_file()
        }
        if not inferred:
            raise TransportContractError(
                f"Legacy result root has no page results: {root}"
            )
        return inferred
    if not isinstance(page_ids, list) or not page_ids:
        raise TransportContractError(
            f"Transport summary has invalid page IDs: {summary_path}"
        )
    normalized = {str(value) for value in page_ids}
    if len(normalized) != len(page_ids) or any(not value for value in normalized):
        raise TransportContractError(
            f"Transport summary has invalid page IDs: {summary_path}"
        )
    return normalized


def build_manifest_from_snapshots(
    *,
    snapshots_path: Path,
    output_dir: Path,
    suite_id: str,
    language: str,
    split: str,
) -> dict[str, Any]:
    snapshots_file = truth_tools.require_external_path(
        snapshots_path, label="page snapshots"
    )
    output = _external_empty_directory(
        output_dir, label="snapshot manifest output"
    )
    if language not in truth_tools.ALLOWED_LANGUAGES:
        raise TransportContractError(f"Unsupported language: {language}")
    if split not in truth_tools.ALLOWED_SPLITS:
        raise TransportContractError(f"Unsupported split: {split}")
    safe_suite_id = truth_tools.require_safe_id(suite_id, label="suite ID")
    payload = truth_tools.read_json(snapshots_file)
    raw_pages = payload.get("pages")
    if not isinstance(raw_pages, list) or not raw_pages:
        raise TransportContractError("page_snapshots.json has no pages.")

    manifest_pages: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw_page in raw_pages:
        if not isinstance(raw_page, dict):
            raise TransportContractError("Invalid page snapshot record.")
        page_id = truth_tools.require_safe_id(
            raw_page.get("image_stem"), label="snapshot page ID"
        )
        if page_id in seen:
            raise TransportContractError(f"Duplicate snapshot page: {page_id}")
        seen.add(page_id)
        source = truth_tools.require_external_path(
            Path(str(raw_page.get("image_path", ""))),
            label=f"snapshot source {page_id}",
        )
        if not source.is_file():
            raise TransportContractError(f"Missing snapshot source: {source}")
        image = cv2.imread(str(source), cv2.IMREAD_COLOR)
        if image is None:
            raise TransportContractError(f"Unable to decode snapshot source: {source}")
        height, width = [int(value) for value in image.shape[:2]]
        source_sha = truth_tools.sha256_file(source)
        raw_blocks = raw_page.get("blocks")
        if not isinstance(raw_blocks, list) or not raw_blocks:
            raise TransportContractError(
                f"Snapshot page has no OCR blocks: {page_id}"
            )
        detector_blocks: list[dict[str, Any]] = []
        for index, raw_block in enumerate(raw_blocks):
            if not isinstance(raw_block, dict):
                raise TransportContractError(
                    f"Invalid snapshot block {page_id}:{index}."
                )
            detector_blocks.append(
                {
                    "block_id": f"detector-{index:04d}",
                    "xyxy": truth_tools._bbox(
                        raw_block.get("xyxy"),
                        label=f"{page_id}:detector-{index:04d}",
                        width=width,
                        height=height,
                    ),
                    "bubble_xyxy": (
                        truth_tools._bbox(
                            raw_block.get("bubble_xyxy"),
                            label=f"{page_id}:detector-{index:04d}:bubble",
                            width=width,
                            height=height,
                        )
                        if raw_block.get("bubble_xyxy")
                        else None
                    ),
                    "text_class": str(raw_block.get("text_class", "") or ""),
                    "direction": str(raw_block.get("direction", "") or ""),
                }
            )
        detector_path = output / "detector" / page_id / "result.json"
        _write_json(
            detector_path,
            {
                "source_sha256": source_sha,
                "shape_hw": [height, width],
                "blocks": detector_blocks,
            },
        )
        manifest_pages.append(
            {
                "page_id": page_id,
                "source_page_id": page_id,
                "split": split,
                "language": language,
                "source_image": {
                    "path": str(source.resolve()),
                    "sha256": source_sha,
                    "width": width,
                    "height": height,
                },
                "detector_snapshot": {
                    "path": str(detector_path.resolve()),
                    "sha256": truth_tools.sha256_file(detector_path),
                },
            }
        )
    manifest: dict[str, Any] = {
        "schema_version": truth_tools.CORPUS_SCHEMA_VERSION,
        "protocol_version": truth_tools.PROTOCOL_VERSION,
        "suite_id": safe_suite_id,
        "created_at": utc_now(),
        "source_page_snapshots": str(snapshots_file.resolve()),
        "source_page_snapshots_sha256": truth_tools.sha256_file(snapshots_file),
        "pages": manifest_pages,
    }
    manifest["manifest_sha256"] = truth_tools.canonical_sha256(manifest)
    manifest_path = output / "corpus-manifest.json"
    _write_json(manifest_path, manifest)
    truth_tools.validate_corpus_manifest(manifest_path)
    return {
        "manifest_path": str(manifest_path),
        "manifest_sha256": truth_tools.sha256_file(manifest_path),
        "page_count": len(manifest_pages),
        "block_count": sum(
            len(page.get("blocks", []))
            for page in raw_pages
            if isinstance(page, dict)
        ),
    }


def _timing_summary(root: Path) -> dict[str, float]:
    summary_path = root / "summary.json"
    if not summary_path.is_file():
        return {}
    summary = truth_tools.read_json(summary_path)
    result: dict[str, float] = {}
    for key in ("startup_seconds", "request_seconds", "wall_seconds"):
        value = summary.get(key)
        if isinstance(value, (int, float)) and float(value) >= 0:
            result[key] = float(value)
    return result


def _improvement_percent(baseline: float, candidate: float) -> float | None:
    if baseline <= 0:
        return None
    return (baseline - candidate) / baseline * 100.0


def compare_exact_transports(
    *,
    baseline_dir: Path,
    candidate_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    baseline = truth_tools.require_external_path(
        baseline_dir, label="baseline results"
    )
    candidate = truth_tools.require_external_path(
        candidate_dir, label="candidate results"
    )
    baseline_pages = _selected_result_pages(baseline)
    candidate_pages = _selected_result_pages(candidate)
    if baseline_pages is None or candidate_pages is None:
        raise TransportContractError(
            "Exact transport comparison requires locked page IDs."
        )
    if baseline_pages != candidate_pages:
        raise TransportContractError(
            "Baseline and candidate result page sets differ."
        )
    output = _external_empty_directory(output_dir, label="exact comparison output")
    rows: list[dict[str, Any]] = []
    block_count = 0
    normalized_changed = 0
    raw_changed = 0
    contract_changed = 0
    for page_id in sorted(baseline_pages):
        baseline_blocks = _result_blocks(baseline, page_id)
        candidate_blocks = _result_blocks(candidate, page_id)
        if set(baseline_blocks) != set(candidate_blocks):
            raise TransportContractError(
                f"Detector block set changed on {page_id}."
            )
        for block_id in sorted(baseline_blocks):
            baseline_block = baseline_blocks[block_id]
            candidate_block = candidate_blocks[block_id]
            baseline_text = str(baseline_block.get("text", "") or "")
            candidate_text = str(candidate_block.get("text", "") or "")
            raw_differs = baseline_text != candidate_text
            normalized_differs = (
                truth_tools.normalize_text(baseline_text)
                != truth_tools.normalize_text(candidate_text)
            )
            contract_differs = baseline_block != candidate_block
            block_count += 1
            raw_changed += int(raw_differs)
            normalized_changed += int(normalized_differs)
            contract_changed += int(contract_differs)
            if contract_differs:
                rows.append(
                    {
                        "page_id": page_id,
                        "block_id": block_id,
                        "baseline_text": baseline_text,
                        "candidate_text": candidate_text,
                        "raw_text_changed": raw_differs,
                        "normalized_text_changed": normalized_differs,
                        "baseline_block": baseline_block,
                        "candidate_block": candidate_block,
                    }
                )
    baseline_timing = _timing_summary(baseline)
    candidate_timing = _timing_summary(candidate)
    improvements = {
        key: _improvement_percent(value, candidate_timing.get(key, 0.0))
        for key, value in baseline_timing.items()
        if key in candidate_timing
    }
    result = {
        "schema_version": 1,
        "protocol_version": PROTOCOL_VERSION,
        "created_at": utc_now(),
        "page_ids": sorted(baseline_pages),
        "page_count": len(baseline_pages),
        "block_count": block_count,
        "raw_text_changed_count": raw_changed,
        "normalized_text_changed_count": normalized_changed,
        "block_contract_changed_count": contract_changed,
        "baseline_timing": baseline_timing,
        "candidate_timing": candidate_timing,
        "improvement_percent": improvements,
        "changed_rows": rows,
    }
    _write_json(output / "comparison.json", result)
    return result


def compare_transports(
    *,
    truth_dir: Path,
    baseline_dir: Path,
    candidate_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    _lock, truth_pages = truth_tools.validate_locked_truth(truth_dir)
    baseline = truth_tools.require_external_path(
        baseline_dir, label="baseline results"
    )
    candidate = truth_tools.require_external_path(
        candidate_dir, label="candidate results"
    )
    baseline_pages = _selected_result_pages(baseline)
    candidate_pages = _selected_result_pages(candidate)
    if (
        baseline_pages is not None
        and candidate_pages is not None
        and baseline_pages != candidate_pages
    ):
        raise TransportContractError(
            "Baseline and candidate result page sets differ."
        )
    selected_pages = baseline_pages or candidate_pages
    output = _external_empty_directory(output_dir, label="comparison output")
    rows: list[dict[str, Any]] = []
    baseline_exact = 0
    candidate_exact = 0
    baseline_accuracy = 0.0
    candidate_accuracy = 0.0
    detector_truth_count = 0
    candidate_only_regressions = 0
    candidate_only_improvements = 0
    changed_count = 0

    for page in truth_pages:
        page_id = str(page["page_id"])
        if selected_pages is not None and page_id not in selected_pages:
            continue
        baseline_blocks = _result_blocks(baseline, page_id)
        candidate_blocks = _result_blocks(candidate, page_id)
        if set(baseline_blocks) != set(candidate_blocks):
            raise TransportContractError(
                f"Detector block set changed on {page_id}."
            )
        for region in page["regions"]:
            detector_ids = [
                str(value) for value in region.get("detector_block_ids", [])
            ]
            if len(detector_ids) != 1:
                continue
            block_id = detector_ids[0]
            if block_id not in baseline_blocks:
                raise TransportContractError(
                    f"Truth detector block is missing on {page_id}: {block_id}"
                )
            truth_text = str(region.get("transcription", "") or "")
            baseline_text = str(
                baseline_blocks[block_id].get("text", "") or ""
            )
            candidate_text = str(
                candidate_blocks[block_id].get("text", "") or ""
            )
            truth_normalized = truth_tools.normalize_text(truth_text)
            baseline_normalized = truth_tools.normalize_text(baseline_text)
            candidate_normalized = truth_tools.normalize_text(candidate_text)
            baseline_is_exact = baseline_normalized == truth_normalized
            candidate_is_exact = candidate_normalized == truth_normalized
            changed = baseline_normalized != candidate_normalized
            detector_truth_count += 1
            baseline_exact += int(baseline_is_exact)
            candidate_exact += int(candidate_is_exact)
            baseline_accuracy += truth_tools.normalized_character_accuracy(
                truth_text, baseline_text
            )
            candidate_accuracy += truth_tools.normalized_character_accuracy(
                truth_text, candidate_text
            )
            candidate_only_regressions += int(
                baseline_is_exact and not candidate_is_exact
            )
            candidate_only_improvements += int(
                candidate_is_exact and not baseline_is_exact
            )
            changed_count += int(changed)
            if changed:
                rows.append(
                    {
                        "page_id": page_id,
                        "truth_region_id": str(region["truth_region_id"]),
                        "block_id": block_id,
                        "truth": truth_text,
                        "baseline": baseline_text,
                        "candidate": candidate_text,
                        "baseline_exact": baseline_is_exact,
                        "candidate_exact": candidate_is_exact,
                        "baseline_accuracy": round(
                            truth_tools.normalized_character_accuracy(
                                truth_text, baseline_text
                            ),
                            6,
                        ),
                        "candidate_accuracy": round(
                            truth_tools.normalized_character_accuracy(
                                truth_text, candidate_text
                            ),
                            6,
                        ),
                        "semantic_review": "",
                        "notes": "",
                        "crop_asset": str(
                            truth_dir / str(region.get("crop_asset", ""))
                        ),
                    }
                )
    if not detector_truth_count:
        raise TransportContractError("Locked truth contains no detector regions.")
    result = {
        "schema_version": 1,
        "protocol_version": PROTOCOL_VERSION,
        "created_at": utc_now(),
        "page_ids": sorted(selected_pages or {str(page["page_id"]) for page in truth_pages}),
        "detector_truth_count": detector_truth_count,
        "changed_output_count": changed_count,
        "baseline": {
            "normalized_exact_count": baseline_exact,
            "normalized_exact_rate": baseline_exact / detector_truth_count,
            "mean_character_accuracy": baseline_accuracy
            / detector_truth_count,
        },
        "candidate": {
            "normalized_exact_count": candidate_exact,
            "normalized_exact_rate": candidate_exact / detector_truth_count,
            "mean_character_accuracy": candidate_accuracy
            / detector_truth_count,
        },
        "candidate_only_exact_regression_count": candidate_only_regressions,
        "candidate_only_exact_improvement_count": candidate_only_improvements,
        "changed_rows": rows,
    }
    _write_json(output / "comparison.json", result)
    lines = [
        "# Paddle crop relay/direct 비교",
        "",
        f"- detector truth: `{detector_truth_count}`",
        f"- changed outputs: `{changed_count}`",
        f"- relay normalized exact: `{baseline_exact}/{detector_truth_count}`",
        f"- direct normalized exact: `{candidate_exact}/{detector_truth_count}`",
        (
            "- relay mean character accuracy: "
            f"`{baseline_accuracy / detector_truth_count:.4%}`"
        ),
        (
            "- direct mean character accuracy: "
            f"`{candidate_accuracy / detector_truth_count:.4%}`"
        ),
        f"- direct-only exact regressions: `{candidate_only_regressions}`",
        f"- direct-only exact improvements: `{candidate_only_improvements}`",
        "",
        "문자열 자동 통계는 의미 품질 판정이 아닙니다. changed_rows의 원본 crop을 직접 검수해야 합니다.",
    ]
    (output / "summary-ko.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--transport", choices=TRANSPORTS, required=True)
    run_parser.add_argument("--manifest", type=Path, required=True)
    run_parser.add_argument("--output", type=Path, required=True)
    run_parser.add_argument("--page-id", action="append", default=[])
    run_parser.add_argument("--workers", type=int, default=8)
    run_parser.add_argument("--max-tokens", type=int, default=1024)
    run_parser.add_argument("--startup-timeout", type=int, default=120)

    compare_parser = subparsers.add_parser("compare")
    compare_parser.add_argument("--truth-dir", type=Path, required=True)
    compare_parser.add_argument("--baseline", type=Path, required=True)
    compare_parser.add_argument("--candidate", type=Path, required=True)
    compare_parser.add_argument("--output", type=Path, required=True)

    exact_parser = subparsers.add_parser("compare-exact")
    exact_parser.add_argument("--baseline", type=Path, required=True)
    exact_parser.add_argument("--candidate", type=Path, required=True)
    exact_parser.add_argument("--output", type=Path, required=True)

    manifest_parser = subparsers.add_parser("build-manifest")
    manifest_parser.add_argument("--snapshots", type=Path, required=True)
    manifest_parser.add_argument("--output", type=Path, required=True)
    manifest_parser.add_argument("--suite-id", required=True)
    manifest_parser.add_argument(
        "--language", choices=sorted(truth_tools.ALLOWED_LANGUAGES), required=True
    )
    manifest_parser.add_argument(
        "--split",
        choices=sorted(truth_tools.ALLOWED_SPLITS),
        default="development",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "run":
            result = run_transport(
                manifest_path=args.manifest,
                output_dir=args.output,
                config=RunConfig(
                    transport=args.transport,
                    workers=int(args.workers),
                    max_tokens=int(args.max_tokens),
                    timeout_sec=int(args.startup_timeout),
                ),
                page_ids=args.page_id,
            )
        elif args.command == "compare":
            result = compare_transports(
                truth_dir=args.truth_dir,
                baseline_dir=args.baseline,
                candidate_dir=args.candidate,
                output_dir=args.output,
            )
        elif args.command == "compare-exact":
            result = compare_exact_transports(
                baseline_dir=args.baseline,
                candidate_dir=args.candidate,
                output_dir=args.output,
            )
        elif args.command == "build-manifest":
            result = build_manifest_from_snapshots(
                snapshots_path=args.snapshots,
                output_dir=args.output,
                suite_id=args.suite_id,
                language=args.language,
                split=args.split,
            )
        else:  # pragma: no cover
            raise TransportContractError(f"Unknown command: {args.command}")
    except (
        OSError,
        subprocess.SubprocessError,
        TransportContractError,
        truth_tools.ContractError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

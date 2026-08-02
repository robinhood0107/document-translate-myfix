#!/usr/bin/env python3
"""Benchmark llama.cpp serving and Paddle crop scheduling candidates.

The runner is deliberately lab-only.  It launches a short-lived, isolated
llama.cpp container against the prepared read-only Paddle named volume and
drives the real offscreen product pipeline up to the OCR ceiling.  Candidate
results, page snapshots, local paths and runtime logs are always written via
the managed private-artifact harness.

The product container is never removed and Docker Compose ``down`` is never
used. Runtime candidates are compared with alternating baseline/candidate
order. Byte-exact candidates can auto-promote; changed snapshots are retained
for a fixed quality-review window but cannot auto-promote.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import copy
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import random
import statistics
import subprocess
import sys
import threading
import time
from typing import Any, Iterable, Mapping, Sequence
from urllib.error import URLError
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "scripts"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from validation_artifact_harness import (  # noqa: E402
    ManagedArtifactRun,
    select_managed_output_directory,
)


PROTOCOL_VERSION = "serving-scheduler-matrix-v1"
FAMILY_NAME = "serving-scheduler-matrix"
ARTIFACT_CATEGORY = "90-cross-cutting"
PINNED_LLAMA_IMAGE = (
    "ghcr.io/ggml-org/llama.cpp@sha256:"
    "22e0e3bfe967af4fd1df6a918022abbfd4e72e4d40a4769e616a4176790acbcb"
)
PINNED_LLAMA_REVISION = "ff067f76dd8e9e05f0528056f1274adf01a54d70"
PINNED_LLAMA_BUILD = "b10133"
PADDLE_MODEL_VOLUME = "comic-translate-paddleocr-vl-llamacpp-models-v1"
PADDLE_READY_MANIFEST = (
    ".comic-translate-paddleocr-vl-llamacpp-ready-v1.json"
)
PADDLE_RUNTIME_NAME = "PaddleOCR-VL-llama.cpp"
PADDLE_PREPARATION_VERSION = 1
PADDLE_MODEL_FILE = "PaddleOCR-VL-1.6-GGUF.gguf"
PADDLE_MMPROJ_FILE = "PaddleOCR-VL-1.6-GGUF-mmproj.gguf"
PADDLE_MODEL_ALIAS = "PaddleOCR-VL-1.6-0.9B"
PADDLE_SLOT_CONTEXT = 4096
PADDLE_MODEL_SHA256 = (
    "f3ae46ec885050acf4b3d31944431e1fd90d50664fb09126af4a3c050ba14ee8"
)
PADDLE_MODEL_BYTES = 935_769_056
PADDLE_MMPROJ_SHA256 = (
    "204d757d7610d9b3faab10d506d69e5b244e32bf765e2bab2d0167e65e0a058a"
)
PADDLE_MMPROJ_BYTES = 881_770_560
GEMMA_MODEL_FILE = "gemma-4-26B-IQ4_NL.gguf"
GEMMA_MODEL_ALIAS = GEMMA_MODEL_FILE
LAB_CONTAINER_PREFIX = "ct-serving-matrix-paddle"
DEFAULT_PORT = 18001
DEFAULT_MAX_ROUNDS = 7
DEFAULT_INITIAL_ROUNDS = 2
GPU_BACKGROUND_LIMIT_MIB = 2048
WINDOWS_AVAILABLE_LIMIT_BYTES = 6 * 1024**3
WINDOWS_EMERGENCY_LIMIT_BYTES = 1 * 1024**3
RESIDENCY_PREFLIGHT_RATIO = 0.95
PADDLE_MEASURED_PEAK_MIB = 2_407
GEMMA_MEASURED_PEAK_MIB = 11_634

REQUIRED_LLAMA_OPTIONS = frozenset(
    {
        "--parallel",
        "--threads-http",
        "--poll",
        "--poll-batch",
        "--batch-size",
        "--ubatch-size",
        "--metrics",
        "--slots",
        "--sleep-idle-seconds",
        "--models-preset",
        "--models-max",
        "--models-autoload",
    }
)

OCR_SNAPSHOT_BLOCK_KEYS = (
    "xyxy",
    "bubble_xyxy",
    "angle",
    "text_class",
    "text",
    "normalized_text",
    "ocr_status",
    "ocr_empty_reason",
    "ocr_attempt_count",
    "ocr_raw_text",
    "ocr_sanitized_text",
    "ocr_reject_reason",
    "ui_panel_mode",
    "mask_decision",
    "mask_reject_reason",
)
FULL_SNAPSHOT_BLOCK_KEYS = OCR_SNAPSHOT_BLOCK_KEYS + (
    "translation",
    "normalized_translation",
    "render_text",
    "render_area_source",
    "render_source_xyxy",
    "render_anchor_xyxy",
    "render_bubble_xyxy",
    "render_normalization_applied",
    "render_normalization_reasons",
    "render_normalization_replacements",
    "render_centered_layout",
    "render_layout_reasons",
    "text_fit_status",
    "text_fit_metrics",
    "block_final_mask_pixel_count",
    "block_mask_iou",
    "block_mask_span_coverage",
    "block_mask_bbox",
    "block_mask_source",
    "block_mask_decision",
    "render_restore_applied",
)


class BenchmarkContractError(RuntimeError):
    """Raised when a run would violate the fixed benchmark contract."""


@dataclass(frozen=True)
class ServingCandidate:
    key: str
    axis: str
    n_parallel: int = 1
    client_workers: int = 8
    threads_http: int = -1
    poll: int = 50
    poll_batch: int = 1
    batch_size: int = 2048
    ubatch_size: int = 512
    sleep_idle_seconds: int = 5
    scheduler_mode: str = "fixed_area_desc"

    def validate(self) -> None:
        if not self.key or any(ch in self.key for ch in "/\\"):
            raise BenchmarkContractError(f"Unsafe candidate key: {self.key!r}")
        if self.n_parallel not in {1, 2, 4}:
            raise BenchmarkContractError("n_parallel must be 1, 2, or 4.")
        if self.client_workers not in {2, 4, 6, 8}:
            raise BenchmarkContractError("client_workers must be 2, 4, 6, or 8.")
        if self.threads_http not in {-1, 2, 4, 8}:
            raise BenchmarkContractError("threads_http must be -1, 2, 4, or 8.")
        if not 0 <= self.poll <= 100:
            raise BenchmarkContractError("poll must be between 0 and 100.")
        if self.poll_batch not in {0, 1}:
            raise BenchmarkContractError("poll_batch must be 0 or 1.")
        if self.ubatch_size > self.batch_size:
            raise BenchmarkContractError("ubatch_size may not exceed batch_size.")
        if self.sleep_idle_seconds < 1:
            raise BenchmarkContractError("sleep_idle_seconds must be positive.")


BASELINE = ServingCandidate(key="baseline", axis="baseline")


def candidate_catalog() -> dict[str, ServingCandidate]:
    candidates = [
        BASELINE,
        ServingCandidate(
            key="idle1",
            axis="handoff",
            sleep_idle_seconds=1,
        ),
    ]
    candidates.extend(
        ServingCandidate(
            key=f"np{n_parallel}-w{workers}",
            axis="parallel",
            n_parallel=n_parallel,
            client_workers=workers,
        )
        for n_parallel, workers in (
            (2, 2),
            (2, 4),
            (2, 6),
            (2, 8),
            (4, 4),
            (4, 6),
            (4, 8),
        )
    )
    candidates.extend(
        ServingCandidate(
            key=f"http{threads_http}",
            axis="http",
            threads_http=threads_http,
        )
        for threads_http in (2, 4, 8)
    )
    candidates.extend(
        ServingCandidate(
            key=f"np4-w6-http{threads_http}",
            axis="http",
            n_parallel=4,
            client_workers=6,
            threads_http=threads_http,
        )
        for threads_http in (2, 4, 8)
    )
    candidates.extend(
        (
            ServingCandidate(key="poll0", axis="poll", poll=0, poll_batch=0),
            ServingCandidate(key="poll100", axis="poll", poll=100, poll_batch=1),
            ServingCandidate(
                key="poll50-batch0",
                axis="poll",
                poll=50,
                poll_batch=0,
            ),
        )
    )
    candidates.extend(
        (
            ServingCandidate(
                key="np4-w6-poll0",
                axis="poll",
                n_parallel=4,
                client_workers=6,
                poll=0,
                poll_batch=0,
            ),
            ServingCandidate(
                key="np4-w6-poll100",
                axis="poll",
                n_parallel=4,
                client_workers=6,
                poll=100,
                poll_batch=1,
            ),
            ServingCandidate(
                key="np4-w6-poll50-batch0",
                axis="poll",
                n_parallel=4,
                client_workers=6,
                poll=50,
                poll_batch=0,
            ),
        )
    )
    catalog = {candidate.key: candidate for candidate in candidates}
    for candidate in catalog.values():
        candidate.validate()
    return catalog


def staged_candidate_keys() -> dict[str, tuple[str, ...]]:
    """Return the one-axis-at-a-time execution plan.

    Batch/ubatch and folder-global queue candidates are intentionally absent:
    their current direct llama.cpp CUDA decisions already exhausted seven
    paired rounds without proving a positive speed gain.
    """

    return {
        "handoff": ("idle1",),
        "parallel": (
            "np2-w2",
            "np2-w4",
            "np2-w6",
            "np2-w8",
            "np4-w4",
            "np4-w6",
            "np4-w8",
        ),
        "http": (
            "np4-w6-http2",
            "np4-w6-http4",
            "np4-w6-http8",
        ),
        "poll": (
            "np4-w6-poll0",
            "np4-w6-poll100",
            "np4-w6-poll50-batch0",
        ),
    }


def build_paddle_server_command(candidate: ServingCandidate) -> list[str]:
    candidate.validate()
    command = [
        "-m",
        f"/models/{PADDLE_MODEL_FILE}",
        "--mmproj",
        f"/models/{PADDLE_MMPROJ_FILE}",
        "--alias",
        PADDLE_MODEL_ALIAS,
        "--host",
        "0.0.0.0",
        "--port",
        "8080",
        "-c",
        str(PADDLE_SLOT_CONTEXT * candidate.n_parallel),
        "-np",
        str(candidate.n_parallel),
        "-t",
        "10",
        "-b",
        str(candidate.batch_size),
        "-ub",
        str(candidate.ubatch_size),
        "--n-gpu-layers",
        "all",
        "--fit",
        "off",
        "--flash-attn",
        "on",
        "--temp",
        "0",
        "--metrics",
        "--slots",
        "--poll",
        str(candidate.poll),
        "--poll-batch",
        str(candidate.poll_batch),
        "--sleep-idle-seconds",
        str(candidate.sleep_idle_seconds),
    ]
    if candidate.threads_http >= 0:
        command.extend(["--threads-http", str(candidate.threads_http)])
    return command


def build_router_preset() -> str:
    """Build the official INI contract for a maximum-one-model router."""

    return "\n".join(
        [
            "version = 1",
            "",
            f"[{PADDLE_MODEL_ALIAS}]",
            f"model = /models/paddle/{PADDLE_MODEL_FILE}",
            f"mmproj = /models/paddle/{PADDLE_MMPROJ_FILE}",
            "c = 4096",
            "np = 1",
            "t = 10",
            "b = 2048",
            "ub = 512",
            "n-gpu-layers = all",
            "fit = off",
            "flash-attn = on",
            "temp = 0",
            "metrics = true",
            "slots = true",
            "sleep-idle-seconds = 1",
            "load-on-startup = false",
            "",
            f"[{GEMMA_MODEL_ALIAS}]",
            f"model = /models/gemma/{GEMMA_MODEL_FILE}",
            "c = 4096",
            "np = 1",
            "t = 10",
            "b = 2048",
            "ub = 512",
            "n-gpu-layers = 23",
            "fit = off",
            "flash-attn = on",
            "cache-type-k = f16",
            "cache-type-v = f16",
            "kv-offload = true",
            "swa-full = true",
            "jinja = true",
            "reasoning = off",
            "reasoning-budget = 0",
            "reasoning-format = none",
            "metrics = true",
            "slots = true",
            "load-on-startup = false",
            "",
        ]
    )


def build_router_command(preset_path: str = "/config/models.ini") -> list[str]:
    return [
        "--host",
        "0.0.0.0",
        "--port",
        "8080",
        "--models-preset",
        preset_path,
        "--models-max",
        "1",
        "--no-models-autoload",
        "--metrics",
        "--slots",
    ]


def parse_supported_options(help_text: str) -> set[str]:
    return {
        option
        for option in REQUIRED_LLAMA_OPTIONS
        if option in str(help_text or "")
    }


def missing_required_options(help_text: str) -> list[str]:
    return sorted(REQUIRED_LLAMA_OPTIONS - parse_supported_options(help_text))


def residency_preflight(
    *,
    physical_mib: int,
    paddle_peak_mib: int,
    gemma_peak_mib: int,
    threshold: float = RESIDENCY_PREFLIGHT_RATIO,
) -> dict[str, Any]:
    if physical_mib <= 0:
        raise BenchmarkContractError("physical_mib must be positive.")
    combined = max(0, paddle_peak_mib) + max(0, gemma_peak_mib)
    ratio = combined / float(physical_mib)
    return {
        "physical_mib": int(physical_mib),
        "paddle_peak_mib": int(paddle_peak_mib),
        "gemma_peak_mib": int(gemma_peak_mib),
        "combined_peak_mib": int(combined),
        "combined_ratio": round(ratio, 6),
        "threshold": float(threshold),
        "may_run_dual_model": bool(ratio <= threshold),
    }


def slot_context_report(
    *,
    props: Mapping[str, Any],
    slots: Mapping[str, Any],
    expected_parallel: int,
) -> dict[str, Any]:
    raw_slots = slots.get("value", slots.get("slots", []))
    slot_rows = raw_slots if isinstance(raw_slots, list) else []
    slot_contexts = [
        int(row.get("n_ctx", 0) or 0)
        for row in slot_rows
        if isinstance(row, Mapping)
    ]
    total_slots = int(props.get("total_slots", 0) or len(slot_contexts))
    default_settings = props.get("default_generation_settings")
    total_context = (
        int(default_settings.get("n_ctx", 0) or 0)
        if isinstance(default_settings, Mapping)
        else 0
    )
    inferred_per_slot = (
        total_context // total_slots
        if total_context > 0 and total_slots > 0
        else 0
    )
    minimum_context = min(slot_contexts) if slot_contexts else inferred_per_slot
    failures = []
    if total_slots != int(expected_parallel):
        failures.append("parallel_slot_count_mismatch")
    if minimum_context < PADDLE_SLOT_CONTEXT:
        failures.append("slot_context_below_4096")
    return {
        "expected_parallel": int(expected_parallel),
        "observed_total_slots": total_slots,
        "observed_total_context": total_context,
        "observed_slot_contexts": slot_contexts,
        "inferred_per_slot_context": inferred_per_slot,
        "minimum_slot_context": minimum_context,
        "failures": failures,
        "passed": not failures,
    }


def canonical_ocr_snapshot(payload: Mapping[str, Any]) -> dict[str, Any]:
    pages = payload.get("pages", [])
    if not isinstance(pages, list):
        raise BenchmarkContractError("page snapshot must contain a pages list.")
    canonical_pages: list[dict[str, Any]] = []
    for page in pages:
        if not isinstance(page, Mapping):
            raise BenchmarkContractError("page snapshot entries must be objects.")
        blocks = page.get("blocks", [])
        if not isinstance(blocks, list):
            raise BenchmarkContractError("page blocks must be a list.")
        canonical_blocks = []
        for block in blocks:
            if not isinstance(block, Mapping):
                raise BenchmarkContractError("snapshot blocks must be objects.")
            canonical_blocks.append(
                {key: block.get(key) for key in OCR_SNAPSHOT_BLOCK_KEYS}
            )
        canonical_pages.append(
            {
                "source_lang": page.get("source_lang"),
                "target_lang": page.get("target_lang"),
                "page_failed": bool(page.get("page_failed", False)),
                "page_failed_reason": str(page.get("page_failed_reason", "") or ""),
                "ocr_quality": page.get("ocr_quality", {}),
                "blocks": canonical_blocks,
            }
        )
    return {"page_count": len(canonical_pages), "pages": canonical_pages}


def canonical_snapshot_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        canonical_ocr_snapshot(payload),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def freeze_ocr_workload(
    snapshot_path: Path,
    *,
    output_path: Path,
) -> dict[str, Any]:
    """Freeze detector geometry once so serving candidates see identical crops."""

    payload = json.loads(snapshot_path.read_text(encoding="utf-8"))
    raw_pages = payload.get("pages")
    if not isinstance(raw_pages, list) or not raw_pages:
        raise BenchmarkContractError("Frozen OCR capture contains no pages.")
    pages: list[dict[str, Any]] = []
    for page_index, page in enumerate(raw_pages):
        if not isinstance(page, Mapping):
            raise BenchmarkContractError("Frozen OCR page must be an object.")
        source = Path(str(page.get("image_path", "") or "")).resolve()
        if not source.is_file():
            raise BenchmarkContractError(
                f"Frozen OCR source is missing for page {page_index}."
            )
        raw_blocks = page.get("blocks")
        if not isinstance(raw_blocks, list):
            raise BenchmarkContractError("Frozen OCR blocks must be a list.")
        blocks: list[dict[str, Any]] = []
        for block_index, block in enumerate(raw_blocks):
            if not isinstance(block, Mapping):
                raise BenchmarkContractError(
                    "Frozen OCR block must be an object."
                )
            xyxy = block.get("xyxy")
            if not isinstance(xyxy, list) or len(xyxy) != 4:
                raise BenchmarkContractError(
                    f"Frozen OCR block has invalid geometry: {page_index}:{block_index}"
                )
            bubble = block.get("bubble_xyxy")
            if bubble is not None and (
                not isinstance(bubble, list) or len(bubble) != 4
            ):
                raise BenchmarkContractError(
                    "Frozen OCR bubble geometry must contain four values."
                )
            blocks.append(
                {
                    "block_id": f"page-{page_index:04d}-block-{block_index:04d}",
                    "xyxy": [int(float(value)) for value in xyxy],
                    "bubble_xyxy": (
                        [int(float(value)) for value in bubble]
                        if bubble is not None
                        else None
                    ),
                    "angle": int(float(block.get("angle", 0) or 0)),
                    "text_class": str(block.get("text_class", "") or ""),
                }
            )
        pages.append(
            {
                "page_index": page_index,
                "image_name": str(page.get("image_name", "") or ""),
                "source_path": str(source),
                "source_sha256": _sha256_file(source),
                "source_lang": str(page.get("source_lang", "") or "Japanese"),
                "target_lang": str(page.get("target_lang", "") or "Korean"),
                "blocks": blocks,
            }
        )
    workload = {
        "protocol_version": PROTOCOL_VERSION,
        "kind": "frozen-paddle-crop-workload",
        "source_snapshot_sha256": _sha256_file(snapshot_path),
        "page_count": len(pages),
        "block_count": sum(len(page["blocks"]) for page in pages),
        "pages": pages,
    }
    _json_write(output_path, workload)
    return workload


class _FrozenReplaySettings:
    def __init__(self, candidate: ServingCandidate, *, endpoint: str) -> None:
        self.candidate = candidate
        self.endpoint = endpoint

    def get_paddleocr_vl_settings(self) -> dict[str, Any]:
        return {
            "server_url": self.endpoint,
            "parallel_workers": self.candidate.client_workers,
            "max_new_tokens": 1024,
            "prettify_markdown": False,
            "visualize": False,
        }

    def get_ocr_generic_settings(self) -> dict[str, Any]:
        return {
            "paddleocr_vl_scheduler_mode": self.candidate.scheduler_mode,
        }

    @staticmethod
    def value(_key: str, default: Any = None, type: Any = None) -> Any:
        del type
        return default


def _aggregate_request_profiles(profiles: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    totals: dict[str, float] = {
        "logical_request_count": 0,
        "http_attempt_count": 0,
        "http_retry_count": 0,
        "queue_wait_ms": 0.0,
        "request_wall_ms": 0.0,
        "encode_ms": 0.0,
        "payload_build_ms": 0.0,
        "response_decode_ms": 0.0,
        "parse_sanitize_ms": 0.0,
    }
    for profile in profiles:
        records = profile.get("request_records")
        if not isinstance(records, list):
            continue
        for record in records:
            if not isinstance(record, Mapping):
                continue
            for key in totals:
                value = record.get(key, 0)
                if isinstance(value, (int, float)):
                    totals[key] += float(value)
    return {
        "page_count": len(profiles),
        **{
            key: int(value) if key.endswith("count") else round(value, 6)
            for key, value in totals.items()
        },
    }


def _frozen_block_snapshot(block: Any) -> dict[str, Any]:
    bubble = getattr(block, "bubble_xyxy", None)
    text = str(getattr(block, "text", "") or "")
    return {
        "xyxy": [int(float(value)) for value in block.xyxy],
        "bubble_xyxy": (
            [int(float(value)) for value in bubble]
            if bubble is not None
            else None
        ),
        "angle": int(float(getattr(block, "angle", 0) or 0)),
        "text_class": str(getattr(block, "text_class", "") or ""),
        "text": text,
        "normalized_text": " ".join(text.split()),
        "ocr_status": str(getattr(block, "ocr_status", "") or ""),
        "ocr_empty_reason": str(
            getattr(block, "ocr_empty_reason", "") or ""
        ),
        "ocr_attempt_count": int(
            getattr(block, "ocr_attempt_count", 0) or 0
        ),
        "ocr_raw_text": str(getattr(block, "ocr_raw_text", "") or ""),
        "ocr_sanitized_text": str(
            getattr(block, "ocr_sanitized_text", "") or ""
        ),
        "ocr_reject_reason": str(
            getattr(block, "ocr_reject_reason", "") or ""
        ),
        "ui_panel_mode": str(getattr(block, "ui_panel_mode", "") or ""),
        "mask_decision": str(getattr(block, "mask_decision", "") or ""),
        "mask_reject_reason": str(
            getattr(block, "mask_reject_reason", "") or ""
        ),
    }


def _monitor_callable_resources(
    action,
    *,
    cancel_event: threading.Event,
    sample_interval_sec: float = 1.0,
) -> tuple[Any, dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    stop_event = threading.Event()

    def monitor() -> None:
        emergency_count = 0
        started = time.monotonic()
        while not stop_event.is_set():
            available = _windows_available_bytes()
            samples.append(
                {
                    "elapsed_sec": round(time.monotonic() - started, 3),
                    "windows_available_bytes": available,
                }
            )
            if available is not None and available < WINDOWS_EMERGENCY_LIMIT_BYTES:
                emergency_count += 1
            else:
                emergency_count = 0
            if emergency_count >= 3:
                cancel_event.set()
                return
            stop_event.wait(max(0.1, float(sample_interval_sec)))

    thread = threading.Thread(target=monitor, name="ct-host-resource-monitor", daemon=True)
    thread.start()
    try:
        result = action()
    finally:
        stop_event.set()
        thread.join(timeout=5)
    values = [
        int(row["windows_available_bytes"])
        for row in samples
        if row.get("windows_available_bytes") is not None
    ]
    minimum = min(values) if values else None
    return result, {
        "sample_interval_sec": float(sample_interval_sec),
        "sample_count": len(samples),
        "windows_available_min_bytes": minimum,
        "windows_available_gate_bytes": WINDOWS_AVAILABLE_LIMIT_BYTES,
        "windows_available_gate_pass": bool(
            minimum is not None and minimum >= WINDOWS_AVAILABLE_LIMIT_BYTES
        ),
        "emergency_limit_bytes": WINDOWS_EMERGENCY_LIMIT_BYTES,
        "emergency_terminated": cancel_event.is_set(),
        "timed_out": False,
        "samples": samples,
    }


def run_frozen_ocr_replay(
    *,
    workload: Mapping[str, Any],
    candidate: ServingCandidate,
    endpoint: str,
) -> dict[str, Any]:
    import cv2
    import numpy as np

    from modules.ocr.paddle_crop.engine import PaddleOCRVLEngine
    from modules.utils.textblock import TextBlock

    raw_pages = workload.get("pages")
    if not isinstance(raw_pages, list) or not raw_pages:
        raise BenchmarkContractError("Frozen OCR workload contains no pages.")
    engine = PaddleOCRVLEngine()
    engine.initialize(_FrozenReplaySettings(candidate, endpoint=endpoint))
    cancel_event = threading.Event()
    engine.set_cancel_checker(cancel_event.is_set)

    def action() -> dict[str, Any]:
        started = time.perf_counter()
        output_pages: list[dict[str, Any]] = []
        profiles: list[dict[str, Any]] = []
        for page in raw_pages:
            if not isinstance(page, Mapping):
                raise BenchmarkContractError("Frozen OCR page must be an object.")
            source = Path(str(page.get("source_path", "") or ""))
            if not source.is_file() or _sha256_file(source) != page.get(
                "source_sha256"
            ):
                raise BenchmarkContractError("Frozen OCR source changed.")
            image = cv2.imread(str(source), cv2.IMREAD_COLOR)
            if image is None:
                raise BenchmarkContractError("Unable to decode frozen OCR source.")
            raw_blocks = page.get("blocks")
            if not isinstance(raw_blocks, list):
                raise BenchmarkContractError("Frozen OCR blocks must be a list.")
            blocks = []
            for block in raw_blocks:
                if not isinstance(block, Mapping):
                    raise BenchmarkContractError("Frozen OCR block is invalid.")
                bubble = block.get("bubble_xyxy")
                blocks.append(
                    TextBlock(
                        text_bbox=np.asarray(block["xyxy"], dtype=np.int32),
                        bubble_bbox=(
                            np.asarray(bubble, dtype=np.int32)
                            if bubble is not None
                            else None
                        ),
                        angle=int(block.get("angle", 0) or 0),
                        text_class=str(block.get("text_class", "") or ""),
                        source_lang=str(page.get("source_lang", "") or "Japanese"),
                        target_lang=str(page.get("target_lang", "") or "Korean"),
                        block_id=str(block.get("block_id", "") or ""),
                    )
                )
            engine.process_image(image, blocks)
            profiles.append(copy.deepcopy(engine.last_page_profile))
            output_pages.append(
                {
                    "source_lang": str(page.get("source_lang", "") or ""),
                    "target_lang": str(page.get("target_lang", "") or ""),
                    "page_failed": False,
                    "page_failed_reason": "",
                    "ocr_quality": {
                        "block_count": len(blocks),
                        "non_empty": sum(bool(block.text) for block in blocks),
                    },
                    "blocks": [_frozen_block_snapshot(block) for block in blocks],
                }
            )
        wall_sec = time.perf_counter() - started
        snapshot = {"pages": output_pages}
        return {
            "wall_sec": round(wall_sec, 6),
            "snapshot": snapshot,
            "snapshot_sha256": canonical_snapshot_sha256(snapshot),
            "ocr_request_metrics": _aggregate_request_profiles(profiles),
            "page_profiles": profiles,
        }

    result, monitor = _monitor_callable_resources(
        action,
        cancel_event=cancel_event,
    )
    result["host_resource_monitor"] = monitor
    return result


def canonical_full_snapshot(payload: Mapping[str, Any]) -> dict[str, Any]:
    pages = payload.get("pages", [])
    if not isinstance(pages, list):
        raise BenchmarkContractError("page snapshot must contain a pages list.")
    canonical_pages: list[dict[str, Any]] = []
    for page in pages:
        if not isinstance(page, Mapping):
            raise BenchmarkContractError("page snapshot entries must be objects.")
        blocks = page.get("blocks", [])
        if not isinstance(blocks, list):
            raise BenchmarkContractError("page blocks must be a list.")
        canonical_pages.append(
            {
                "source_lang": page.get("source_lang"),
                "target_lang": page.get("target_lang"),
                "page_failed": bool(page.get("page_failed", False)),
                "page_failed_reason": str(page.get("page_failed_reason", "") or ""),
                "ocr_quality": page.get("ocr_quality", {}),
                "translated_image_exists": bool(
                    page.get("translated_image_exists", False)
                ),
                "translated_image_sha256": str(
                    page.get("translated_image_sha256", "") or ""
                ),
                "translated_image_decoded_pixel_sha256": str(
                    page.get("translated_image_decoded_pixel_sha256", "") or ""
                ),
                "blocks": [
                    {key: block.get(key) for key in FULL_SNAPSHOT_BLOCK_KEYS}
                    for block in blocks
                    if isinstance(block, Mapping)
                ],
            }
        )
    return {"page_count": len(canonical_pages), "pages": canonical_pages}


def canonical_full_snapshot_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        canonical_full_snapshot(payload),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def paired_improvements(
    baseline_seconds: Sequence[float],
    candidate_seconds: Sequence[float],
) -> list[float]:
    if len(baseline_seconds) != len(candidate_seconds) or not baseline_seconds:
        raise BenchmarkContractError("paired timings must be non-empty and equal length.")
    improvements = []
    for baseline, candidate in zip(baseline_seconds, candidate_seconds):
        baseline_value = float(baseline)
        candidate_value = float(candidate)
        if baseline_value <= 0 or candidate_value <= 0:
            raise BenchmarkContractError("paired timings must be positive.")
        improvements.append((baseline_value - candidate_value) / baseline_value * 100.0)
    return improvements


def bootstrap_lower_bound(
    values: Sequence[float],
    *,
    confidence: float = 0.95,
    resamples: int = 100_000,
    seed: int = 20260801,
) -> float:
    samples = [float(value) for value in values]
    if not samples:
        raise BenchmarkContractError("bootstrap values cannot be empty.")
    if len(samples) == 1:
        return samples[0]
    rng = random.Random(seed)
    means = []
    count = len(samples)
    for _ in range(max(1, int(resamples))):
        means.append(
            sum(samples[rng.randrange(count)] for _ in range(count)) / count
        )
    means.sort()
    tail = max(0.0, min(1.0, 1.0 - float(confidence)))
    index = min(len(means) - 1, max(0, int(math.floor(tail * len(means)))))
    return float(means[index])


def summarize_pairs(
    baseline_seconds: Sequence[float],
    candidate_seconds: Sequence[float],
) -> dict[str, Any]:
    improvements = paired_improvements(baseline_seconds, candidate_seconds)
    return {
        "pair_count": len(improvements),
        "baseline_median_sec": round(statistics.median(baseline_seconds), 6),
        "candidate_median_sec": round(statistics.median(candidate_seconds), 6),
        "improvements_percent": [round(value, 6) for value in improvements],
        "mean_improvement_percent": round(statistics.fmean(improvements), 6),
        "median_improvement_percent": round(statistics.median(improvements), 6),
        "one_sided_95_bootstrap_lower_percent": round(
            bootstrap_lower_bound(improvements),
            6,
        ),
        "candidate_wins": sum(1 for value in improvements if value > 0),
        "candidate_losses": sum(1 for value in improvements if value < 0),
    }


def should_continue_adaptive(summary: Mapping[str, Any], *, rounds: int) -> bool:
    if rounds >= DEFAULT_MAX_ROUNDS:
        return False
    lower = float(summary.get("one_sided_95_bootstrap_lower_percent", 0.0) or 0.0)
    wins = int(summary.get("candidate_wins", 0) or 0)
    losses = int(summary.get("candidate_losses", 0) or 0)
    return lower <= 0.0 or (wins > 0 and losses > 0)


def should_stop_pair_matrix(
    *,
    summary: Mapping[str, Any],
    axis_summary: Mapping[str, Any],
    rounds: int,
    initial_rounds: int,
    snapshot_mismatches: Sequence[Mapping[str, Any]],
) -> bool:
    """Stop only after enough evidence exists for speed or quality review.

    A changed snapshot is never eligible for automatic promotion, but one
    changed run is not enough to tell whether the alternative output is stable.
    Collect the fixed initial AB/BA window, then stop without spending adaptive
    rounds on a quality-review-only candidate.
    """

    if rounds < initial_rounds:
        return False
    if snapshot_mismatches:
        return True
    return not should_continue_adaptive(
        summary,
        rounds=rounds,
    ) and not should_continue_adaptive(
        axis_summary,
        rounds=rounds,
    )


def _json_write(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _run(
    command: Sequence[str],
    *,
    cwd: Path = ROOT,
    env: Mapping[str, str] | None = None,
    check: bool = True,
    timeout: float | None = None,
) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        list(command),
        cwd=str(cwd),
        env=dict(env) if env is not None else None,
        check=False,
        capture_output=True,
        text=True,
        errors="replace",
        timeout=timeout,
    )
    if check and completed.returncode != 0:
        raise BenchmarkContractError(
            f"Command failed ({completed.returncode}): {Path(command[0]).name}\n"
            + (completed.stderr or completed.stdout or "")[-4096:]
        )
    return completed


def _run_pipeline_monitored(
    command: Sequence[str],
    *,
    stdout_path: Path,
    stderr_path: Path,
    env: Mapping[str, str],
    timeout: float,
    sample_interval_sec: float = 1.0,
) -> tuple[subprocess.CompletedProcess[str], dict[str, Any]]:
    """Run a Windows pipeline child while tracking host RAM pressure.

    The 6 GiB contract is a promotion gate.  A sustained drop below 1 GiB is
    an emergency stop so an invalid benchmark cannot starve the desktop while
    the synchronous child is still running.
    """

    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_path.parent.mkdir(parents=True, exist_ok=True)
    samples: list[dict[str, Any]] = []
    emergency_count = 0
    emergency_terminated = False
    timed_out = False
    started = time.monotonic()
    with stdout_path.open("w", encoding="utf-8") as stdout_file, stderr_path.open(
        "w", encoding="utf-8"
    ) as stderr_file:
        process = subprocess.Popen(
            list(command),
            cwd=str(ROOT),
            env=dict(env),
            stdout=stdout_file,
            stderr=stderr_file,
            text=True,
        )
        while process.poll() is None:
            elapsed = time.monotonic() - started
            available = _windows_available_bytes()
            samples.append(
                {
                    "elapsed_sec": round(elapsed, 3),
                    "windows_available_bytes": available,
                }
            )
            if (
                available is not None
                and available < WINDOWS_EMERGENCY_LIMIT_BYTES
            ):
                emergency_count += 1
            else:
                emergency_count = 0
            if emergency_count >= 3:
                emergency_terminated = True
                process.terminate()
                break
            if elapsed >= timeout:
                timed_out = True
                process.terminate()
                break
            time.sleep(max(0.1, float(sample_interval_sec)))
        try:
            returncode = process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            process.kill()
            returncode = process.wait(timeout=15)

    stdout = stdout_path.read_text(encoding="utf-8", errors="replace")
    stderr = stderr_path.read_text(encoding="utf-8", errors="replace")
    valid_values = [
        int(item["windows_available_bytes"])
        for item in samples
        if item.get("windows_available_bytes") is not None
    ]
    minimum = min(valid_values) if valid_values else None
    report = {
        "sample_interval_sec": float(sample_interval_sec),
        "sample_count": len(samples),
        "windows_available_min_bytes": minimum,
        "windows_available_gate_bytes": WINDOWS_AVAILABLE_LIMIT_BYTES,
        "windows_available_gate_pass": bool(
            minimum is not None and minimum >= WINDOWS_AVAILABLE_LIMIT_BYTES
        ),
        "emergency_limit_bytes": WINDOWS_EMERGENCY_LIMIT_BYTES,
        "emergency_terminated": emergency_terminated,
        "timed_out": timed_out,
        "samples": samples,
    }
    return (
        subprocess.CompletedProcess(
            list(command),
            returncode,
            stdout=stdout,
            stderr=stderr,
        ),
        report,
    )


def _http_json(url: str, *, timeout: float = 3.0) -> dict[str, Any]:
    request = Request(url, headers={"Accept": "application/json"})
    with urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return payload if isinstance(payload, dict) else {"value": payload}


def adaptive_wait_for_health(
    url: str,
    *,
    timeout_sec: float = 90.0,
    monotonic=time.monotonic,
    sleeper=time.sleep,
) -> dict[str, Any]:
    started = monotonic()
    attempts = 0
    intervals: list[float] = []
    last_error = ""
    while monotonic() - started < timeout_sec:
        attempts += 1
        try:
            payload = _http_json(url)
            return {
                "status": "ready",
                "attempts": attempts,
                "elapsed_sec": round(monotonic() - started, 6),
                "intervals": intervals,
                "payload": payload,
            }
        except (OSError, URLError, ValueError, json.JSONDecodeError) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
        elapsed = monotonic() - started
        interval = 0.1 if elapsed < 2.0 else (0.25 if elapsed < 10.0 else 1.0)
        intervals.append(interval)
        sleeper(interval)
    raise BenchmarkContractError(
        f"Timed out waiting for {url}: {last_error or 'not ready'}"
    )


def _gpu_snapshot() -> dict[str, Any]:
    completed = _run(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        check=False,
    )
    line = (completed.stdout or "").strip().splitlines()
    if completed.returncode != 0 or not line:
        return {"available": False, "error": (completed.stderr or "")[-512:]}
    fields = [value.strip() for value in line[0].split(",")]
    if len(fields) < 5:
        return {"available": False, "raw": line[0]}
    return {
        "available": True,
        "name": fields[0],
        "total_mib": int(fields[1]),
        "used_mib": int(fields[2]),
        "free_mib": int(fields[3]),
        "utilization_percent": int(fields[4]),
    }


def _wsl_swap_used_bytes() -> int | None:
    try:
        if os.name == "nt":
            completed = _run(
                [
                    "wsl.exe",
                    "-e",
                    "sh",
                    "-lc",
                    "awk '/^SwapTotal:/{t=$2}/^SwapFree:/{f=$2}END{print (t-f)*1024}' /proc/meminfo",
                ],
                check=False,
            )
            raw = (completed.stdout or "").strip().splitlines()
            return int(raw[-1]) if completed.returncode == 0 and raw else None
        fields: dict[str, int] = {}
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if line.startswith(("SwapTotal:", "SwapFree:")):
                key, value, *_rest = line.split()
                fields[key.rstrip(":")] = int(value) * 1024
        return fields.get("SwapTotal", 0) - fields.get("SwapFree", 0)
    except (OSError, ValueError):
        return None


def _windows_available_bytes() -> int | None:
    try:
        if os.name == "nt":
            import psutil

            return int(psutil.virtual_memory().available)
        completed = _run(
            [
                "powershell.exe",
                "-NoProfile",
                "-NonInteractive",
                "-Command",
                "[int64](Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory * 1KB",
            ],
            check=False,
        )
        raw = (completed.stdout or "").strip().splitlines()
        return int(raw[-1]) if completed.returncode == 0 and raw else None
    except (ImportError, OSError, ValueError):
        return None


def resource_preflight() -> dict[str, Any]:
    gpu = _gpu_snapshot()
    windows_available = _windows_available_bytes()
    swap_used = _wsl_swap_used_bytes()
    failures = []
    if not gpu.get("available"):
        failures.append("gpu_metrics_unavailable")
    elif int(gpu.get("used_mib", 0)) > GPU_BACKGROUND_LIMIT_MIB:
        failures.append("gpu_background_above_2gib")
    if windows_available is None:
        failures.append("windows_available_ram_unavailable")
    elif windows_available < WINDOWS_AVAILABLE_LIMIT_BYTES:
        failures.append("windows_available_ram_below_6gib")
    if swap_used is None:
        failures.append("wsl_swap_metrics_unavailable")
    return {
        "gpu": gpu,
        "windows_available_bytes": windows_available,
        "wsl_swap_used_bytes": swap_used,
        "failures": failures,
        "passed": not failures,
    }


def _docker_image_help() -> str:
    completed = _run(
        [
            "docker",
            "run",
            "--rm",
            "--entrypoint",
            "/app/llama-server",
            PINNED_LLAMA_IMAGE,
            "--help",
        ],
        timeout=60,
    )
    return completed.stdout or ""


def validate_paddle_volume_probe(
    *,
    labels: Mapping[str, Any],
    manifest_bytes: bytes,
    manifest_sha256: str,
    observed_file_bytes: Mapping[str, int],
) -> dict[str, Any]:
    """Validate the prepared volume without rehashing the large GGUF files."""

    failures: list[str] = []
    expected_labels = {
        "comic-translate.runtime": PADDLE_RUNTIME_NAME,
        "comic-translate.preparation-version": str(
            PADDLE_PREPARATION_VERSION
        ),
    }
    if any(str(labels.get(key, "")) != value for key, value in expected_labels.items()):
        failures.append("volume_label_mismatch")

    normalized_manifest_sha = str(manifest_sha256 or "").strip().lower()
    actual_manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
    if normalized_manifest_sha != actual_manifest_sha:
        failures.append("ready_manifest_sha256_mismatch")

    try:
        manifest = json.loads(manifest_bytes.decode("utf-8-sig"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        manifest = None
        failures.append("ready_manifest_invalid_json")

    expected_image_id = PINNED_LLAMA_IMAGE.rsplit("@", maxsplit=1)[-1]
    required_header = {
        "schema_version": 1,
        "runtime": PADDLE_RUNTIME_NAME,
        "preparation_version": PADDLE_PREPARATION_VERSION,
        "volume_name": PADDLE_MODEL_VOLUME,
        "ready": True,
        "source_image_ref": PINNED_LLAMA_IMAGE,
        "source_image_id": expected_image_id,
    }
    if isinstance(manifest, Mapping):
        if any(manifest.get(key) != value for key, value in required_header.items()):
            failures.append("ready_manifest_header_mismatch")
        smoke_test = manifest.get("smoke_test")
        if not isinstance(smoke_test, Mapping) or smoke_test.get("passed") is not True:
            failures.append("ready_manifest_smoke_missing")

        raw_files = manifest.get("files")
        file_entries = {
            str(item.get("name", "")): item
            for item in raw_files
            if isinstance(item, Mapping)
        } if isinstance(raw_files, list) else {}
        expected_files = {
            PADDLE_MODEL_FILE: {
                "bytes": PADDLE_MODEL_BYTES,
                "sha256": PADDLE_MODEL_SHA256,
                "role": "vlm",
            },
            PADDLE_MMPROJ_FILE: {
                "bytes": PADDLE_MMPROJ_BYTES,
                "sha256": PADDLE_MMPROJ_SHA256,
                "role": "vision-projector",
            },
        }
        if set(file_entries) != set(expected_files):
            failures.append("ready_manifest_file_registry_mismatch")
        else:
            for name, expected in expected_files.items():
                entry = file_entries[name]
                if (
                    int(entry.get("bytes", -1)) != expected["bytes"]
                    or str(entry.get("sha256", "")).lower()
                    != expected["sha256"]
                    or str(entry.get("role", "")) != expected["role"]
                    or int(observed_file_bytes.get(name, -1))
                    != expected["bytes"]
                ):
                    failures.append(f"volume_file_contract_mismatch:{name}")

    return {
        "volume_name": PADDLE_MODEL_VOLUME,
        "ready_manifest": PADDLE_READY_MANIFEST,
        "ready_manifest_sha256": normalized_manifest_sha,
        "volume_labels": dict(labels),
        "observed_file_bytes": {
            str(key): int(value) for key, value in observed_file_bytes.items()
        },
        "failures": failures,
        "passed": not failures,
    }


def paddle_volume_contract_snapshot() -> dict[str, Any]:
    labels_result = _run(
        [
            "docker",
            "volume",
            "inspect",
            "--format",
            "{{json .Labels}}",
            PADDLE_MODEL_VOLUME,
        ]
    )
    try:
        labels = json.loads((labels_result.stdout or "").strip() or "{}")
    except json.JSONDecodeError as exc:
        raise BenchmarkContractError(
            "Unable to parse the prepared Paddle volume labels."
        ) from exc
    if not isinstance(labels, Mapping):
        raise BenchmarkContractError(
            "Prepared Paddle volume labels must be a JSON object."
        )

    probe_script = """
set -eu
manifest_path="/models/$READY_MANIFEST"
model_path="/models/$MODEL_FILE"
mmproj_path="/models/$MMPROJ_FILE"
test -f "$manifest_path"
test -f "$model_path"
test -f "$mmproj_path"
printf 'manifest_sha256=%s\\n' "$(sha256sum "$manifest_path" | cut -d ' ' -f 1)"
printf 'manifest_base64='
base64 -w 0 "$manifest_path"
printf '\\nmodel_bytes=%s\\n' "$(stat -c %s "$model_path")"
printf 'mmproj_bytes=%s\\n' "$(stat -c %s "$mmproj_path")"
""".strip()
    completed = _run(
        [
            "docker",
            "run",
            "--rm",
            "--pull",
            "never",
            "-e",
            f"READY_MANIFEST={PADDLE_READY_MANIFEST}",
            "-e",
            f"MODEL_FILE={PADDLE_MODEL_FILE}",
            "-e",
            f"MMPROJ_FILE={PADDLE_MMPROJ_FILE}",
            "--mount",
            (
                f"type=volume,source={PADDLE_MODEL_VOLUME},"
                "target=/models,readonly"
            ),
            "--entrypoint",
            "/bin/sh",
            PINNED_LLAMA_IMAGE,
            "-ec",
            probe_script,
        ]
    )
    values: dict[str, str] = {}
    for line in (completed.stdout or "").splitlines():
        key, separator, value = line.partition("=")
        if separator:
            values[key.strip()] = value.strip()
    try:
        manifest_bytes = base64.b64decode(
            values["manifest_base64"], validate=True
        )
        observed = {
            PADDLE_MODEL_FILE: int(values["model_bytes"]),
            PADDLE_MMPROJ_FILE: int(values["mmproj_bytes"]),
        }
    except (KeyError, ValueError, binascii.Error) as exc:
        raise BenchmarkContractError(
            "Unable to parse the prepared Paddle volume probe output."
        ) from exc
    return validate_paddle_volume_probe(
        labels=labels,
        manifest_bytes=manifest_bytes,
        manifest_sha256=values.get("manifest_sha256", ""),
        observed_file_bytes=observed,
    )


def runtime_contract_snapshot() -> dict[str, Any]:
    image = _run(
        [
            "docker",
            "image",
            "inspect",
            PINNED_LLAMA_IMAGE,
            "--format",
            "{{json .}}",
        ]
    )
    image_payload = json.loads(image.stdout)
    help_text = _docker_image_help()
    missing = missing_required_options(help_text)
    labels = ((image_payload.get("Config") or {}).get("Labels") or {})
    revision = str(labels.get("org.opencontainers.image.revision", "") or "")
    build = str(labels.get("org.opencontainers.image.version", "") or "")
    volume_contract = paddle_volume_contract_snapshot()
    failures = []
    if revision != PINNED_LLAMA_REVISION:
        failures.append("llama_revision_mismatch")
    if build != PINNED_LLAMA_BUILD:
        failures.append("llama_build_mismatch")
    if missing:
        failures.append("missing_required_options")
    if not volume_contract["passed"]:
        failures.extend(volume_contract["failures"])
    return {
        "protocol_version": PROTOCOL_VERSION,
        "image_ref": PINNED_LLAMA_IMAGE,
        "image_id": image_payload.get("Id"),
        "revision": revision,
        "build": build,
        "supported_options": sorted(parse_supported_options(help_text)),
        "missing_required_options": missing,
        "model_volume": PADDLE_MODEL_VOLUME,
        "model_sha256": PADDLE_MODEL_SHA256,
        "mmproj_sha256": PADDLE_MMPROJ_SHA256,
        "volume_contract": volume_contract,
        "failures": failures,
        "passed": not failures,
    }


def _safe_container_name(candidate_key: str, round_index: int, role: str) -> str:
    digest = hashlib.sha256(
        f"{os.getpid()}:{candidate_key}:{round_index}:{role}".encode("utf-8")
    ).hexdigest()[:10]
    return f"{LAB_CONTAINER_PREFIX}-{digest}"


def _stop_exact_container(name: str) -> None:
    _run(
        ["docker", "stop", "--timeout", "10", name],
        check=False,
        timeout=30,
    )


def _container_swap_peak_bytes(name: str) -> int | None:
    completed = _run(
        [
            "docker",
            "exec",
            name,
            "/bin/sh",
            "-ec",
            (
                "if test -r /sys/fs/cgroup/memory.swap.peak; then "
                "cat /sys/fs/cgroup/memory.swap.peak; "
                "elif test -r /sys/fs/cgroup/memory.swap.current; then "
                "cat /sys/fs/cgroup/memory.swap.current; "
                "else exit 44; fi"
            ),
        ],
        check=False,
        timeout=15,
    )
    raw = (completed.stdout or "").strip().splitlines()
    if completed.returncode != 0 or not raw:
        return None
    try:
        return max(0, int(raw[-1]))
    except ValueError:
        return None


def swap_gate_report(
    *,
    global_delta_bytes: int | None,
    cgroup_peak_bytes: Sequence[int | None],
) -> dict[str, Any]:
    observed_peaks = [int(value) for value in cgroup_peak_bytes if value is not None]
    cgroup_pass = (
        all(value == 0 for value in observed_peaks)
        if observed_peaks
        else None
    )
    global_pass = global_delta_bytes == 0
    return {
        "global_delta_bytes": global_delta_bytes,
        "global_pass": global_pass,
        "cgroup_peak_bytes": observed_peaks,
        "cgroup_pass": cgroup_pass,
        "source": "cgroup-v2+global-wsl" if observed_peaks else "global-wsl-fallback",
        "passed": bool(global_pass and cgroup_pass is not False),
    }


def _remove_exact_lab_container(name: str) -> None:
    if not str(name).startswith(f"{LAB_CONTAINER_PREFIX}-"):
        raise BenchmarkContractError(
            f"Refusing to remove a non-lab container: {name!r}"
        )
    _stop_exact_container(name)
    _run(
        ["docker", "rm", name],
        check=False,
        timeout=30,
    )


def _start_paddle_container(
    candidate: ServingCandidate,
    *,
    name: str,
    port: int,
) -> dict[str, Any]:
    command = [
        "docker",
        "run",
        "--detach",
        "--name",
        name,
        "--pull",
        "never",
        "--gpus",
        "all",
        "--publish",
        f"127.0.0.1:{port}:8080",
        "--mount",
        f"type=volume,source={PADDLE_MODEL_VOLUME},target=/models,readonly",
        "--label",
        f"com.comictranslate.benchmark-protocol={PROTOCOL_VERSION}",
        PINNED_LLAMA_IMAGE,
        *build_paddle_server_command(candidate),
    ]
    started = time.perf_counter()
    completed = _run(command, timeout=60)
    readiness = adaptive_wait_for_health(f"http://127.0.0.1:{port}/health")
    return {
        "container_id": (completed.stdout or "").strip(),
        "start_to_health_sec": round(time.perf_counter() - started, 6),
        "readiness": readiness,
        "command": build_paddle_server_command(candidate),
    }


def _build_candidate_preset(
    candidate: ServingCandidate,
    *,
    endpoint_port: int,
) -> dict[str, Any]:
    baseline_path = (
        ROOT
        / "benchmarks"
        / "cold_cache_finalization"
        / "presets"
        / "product-v1.1.0-baseline.json"
    )
    payload = json.loads(baseline_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise BenchmarkContractError("Product baseline preset must be an object.")
    preset = copy.deepcopy(payload)
    preset["name"] = f"{PROTOCOL_VERSION}-{candidate.key}"
    preset["description"] = "Private serving scheduler candidate"
    preset.setdefault("ocr_client", {})["server_url"] = (
        f"http://127.0.0.1:{endpoint_port}/v1/chat/completions"
    )
    preset["ocr_client"]["parallel_workers"] = candidate.client_workers
    preset.setdefault("ocr_generic", {})["paddleocr_vl_scheduler_mode"] = (
        candidate.scheduler_mode
    )
    preset.setdefault("benchmark_http", {})["gemma_seed"] = 20260801
    # The historical v1.1.0 preset predates the direct llama.cpp crop
    # runtime. Preserve that source fixture, but replace its retired PaddleX
    # relay contract in this private benchmark copy so staged product-managed
    # runs exercise the current product runtime exactly.
    preset["ocr_runtime"] = {
        "kind": "paddleocr_vl",
        "llama_cpp_image": PINNED_LLAMA_IMAGE,
        "pull_policy": "never",
        "model_path": f"/models/{PADDLE_MODEL_FILE}",
        "mmproj_path": f"/models/{PADDLE_MMPROJ_FILE}",
        "model_alias": PADDLE_MODEL_ALIAS,
        "context_size": 4096,
        "n_parallel": candidate.n_parallel,
        "threads": 10,
        "batch_size": candidate.batch_size,
        "ubatch_size": candidate.ubatch_size,
        "n_gpu_layers": "all",
        "sleep_idle_seconds": candidate.sleep_idle_seconds,
        "model_volume": PADDLE_MODEL_VOLUME,
    }
    preset.setdefault("benchmark_cache_policy", {}).update(
        {
            "paddleocr_persistent": False,
            "translation_persistent": False,
            "exact_tm": False,
            "project_checkpoint": False,
        }
    )
    preset.setdefault("export", {}).update(
        {
            "export_raw_text": False,
            "export_translated_text": False,
            "export_inpainted_image": False,
            "export_detector_overlay": False,
            "export_raw_mask": False,
            "export_mask_overlay": False,
            "export_cleanup_mask_delta": False,
            "export_debug_metadata": False,
        }
    )
    return preset


def _extract_ocr_request_metrics(metrics_path: Path) -> dict[str, Any]:
    totals = {
        "logical_request_count": 0,
        "http_attempt_count": 0,
        "http_retry_count": 0,
        "queue_wait_ms": 0.0,
        "request_wall_ms": 0.0,
        "encode_ms": 0.0,
        "payload_build_ms": 0.0,
        "response_decode_ms": 0.0,
        "parse_sanitize_ms": 0.0,
    }
    page_count = 0
    if not metrics_path.is_file():
        return {"page_count": 0, **totals}
    for raw_line in metrics_path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(raw_line)
        except json.JSONDecodeError:
            continue
        if not isinstance(row, dict) or str(row.get("tag", "")) != "ocr_end":
            continue
        profile = row.get("ocr_page_profile")
        if not isinstance(profile, dict):
            continue
        page_count += 1
        for record in profile.get("request_records", []) or []:
            if not isinstance(record, dict):
                continue
            for key in totals:
                value = record.get(key, 0)
                if isinstance(value, (int, float)):
                    totals[key] += value
    return {
        "page_count": page_count,
        **{
            key: int(value) if key.endswith("count") else round(float(value), 6)
            for key, value in totals.items()
        },
    }


def build_product_pipeline_command(
    *,
    preset_path: Path,
    run_dir: Path,
    sample_dir: Path,
    sample_count: int,
    python_executable: str,
    stage_ceiling: str,
    runtime_mode: str,
    runtime_services: str,
    product_managed_runtime: bool,
) -> list[str]:
    command = [
        python_executable,
        str(ROOT / "scripts" / "benchmark_pipeline.py"),
        "--preset",
        str(preset_path),
        "--mode",
        "batch",
        "--repeat",
        "1",
        "--runtime-mode",
        runtime_mode,
        "--runtime-services",
        runtime_services,
        "--sample-dir",
        str(sample_dir),
        "--sample-count",
        str(sample_count),
        "--source-lang",
        "Japanese",
        "--target-lang",
        "Korean",
        "--export-page-snapshots",
        "--stage-ceiling",
        stage_ceiling,
        "--output-dir",
        str(run_dir),
    ]
    if product_managed_runtime:
        command.append("--product-managed-runtime")
    return command


def _run_product_pipeline(
    *,
    candidate: ServingCandidate,
    run_dir: Path,
    sample_dir: Path,
    sample_count: int,
    endpoint_port: int,
    python_executable: str,
    stage_ceiling: str = "ocr",
    runtime_mode: str = "attach-running",
    runtime_services: str = "ocr-only",
    product_managed_runtime: bool = False,
    extra_environment: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    preset_path = run_dir / "preset.json"
    _json_write(
        preset_path,
        _build_candidate_preset(candidate, endpoint_port=endpoint_port),
    )
    command = build_product_pipeline_command(
        preset_path=preset_path,
        run_dir=run_dir,
        sample_dir=sample_dir,
        sample_count=sample_count,
        python_executable=python_executable,
        stage_ceiling=stage_ceiling,
        runtime_mode=runtime_mode,
        runtime_services=runtime_services,
        product_managed_runtime=product_managed_runtime,
    )
    environment = os.environ.copy()
    environment.update(
        {
            "QT_QPA_PLATFORM": "offscreen",
            "CT_DISABLE_UPDATE_CHECK": "1",
            "CT_ENABLE_MEMLOG": "1",
            "CT_ENABLE_GPU_BENCH": "1",
            "CT_MEMLOG_INTERVAL_SEC": "1",
            "COMIC_SKIP_STARTUP_MODELS": "1",
        }
    )
    if extra_environment:
        environment.update(
            {str(key): str(value) for key, value in extra_environment.items()}
        )
    started = time.perf_counter()
    completed, resource_monitor = _run_pipeline_monitored(
        command,
        stdout_path=run_dir / "runner.stdout.log",
        stderr_path=run_dir / "runner.stderr.log",
        env=environment,
        timeout=900,
    )
    wall_sec = time.perf_counter() - started
    _json_write(run_dir / "host-resource-monitor.json", resource_monitor)
    if completed.returncode != 0:
        raise BenchmarkContractError(
            f"Product OCR ceiling failed for {candidate.key}: "
            f"exit={completed.returncode} "
            f"emergency={resource_monitor['emergency_terminated']} "
            f"timeout={resource_monitor['timed_out']}"
        )
    summary_path = run_dir / "summary.json"
    snapshot_path = run_dir / "page_snapshots.json"
    if not summary_path.is_file() or not snapshot_path.is_file():
        raise BenchmarkContractError(
            f"Missing benchmark outputs for {candidate.key}."
        )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    snapshot_sha = (
        canonical_full_snapshot_sha256(snapshot)
        if stage_ceiling != "ocr"
        else canonical_snapshot_sha256(snapshot)
    )
    return {
        "wall_sec": round(wall_sec, 6),
        "summary": summary,
        "snapshot_sha256": snapshot_sha,
        "stage_ceiling": stage_ceiling,
        "host_resource_monitor": resource_monitor,
        "ocr_request_metrics": _extract_ocr_request_metrics(
            run_dir / "metrics.jsonl"
        ),
    }


def _runtime_evidence(container_name: str, port: int) -> dict[str, Any]:
    inspect = _run(
        ["docker", "inspect", container_name, "--format", "{{json .}}"],
        check=False,
    )
    payload = {}
    if inspect.returncode == 0:
        try:
            payload = json.loads(inspect.stdout)
        except json.JSONDecodeError:
            payload = {}
    evidence = {
        "container": {
            "id": payload.get("Id"),
            "image": payload.get("Image"),
            "command": (payload.get("Config") or {}).get("Cmd"),
            "labels": (payload.get("Config") or {}).get("Labels"),
            "mounts": payload.get("Mounts"),
        },
        "gpu": _gpu_snapshot(),
    }
    for endpoint, key in (
        ("props", "props"),
        ("slots", "slots"),
    ):
        try:
            evidence[key] = _http_json(f"http://127.0.0.1:{port}/{endpoint}")
        except Exception as exc:
            evidence[key] = {"error": f"{type(exc).__name__}: {exc}"}
    try:
        request = Request(f"http://127.0.0.1:{port}/metrics")
        with urlopen(request, timeout=3) as response:
            evidence["metrics_text"] = response.read().decode("utf-8")
    except Exception as exc:
        evidence["metrics_text"] = f"error: {type(exc).__name__}: {exc}"
    return evidence


def execute_candidate_once(
    candidate: ServingCandidate,
    *,
    run_dir: Path,
    sample_dir: Path,
    sample_count: int,
    round_index: int,
    python_executable: str,
    port: int = DEFAULT_PORT,
    full_auto: bool = False,
    frozen_workload: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    preflight = resource_preflight()
    _json_write(run_dir / "resource-preflight.json", preflight)
    if not preflight["passed"]:
        raise BenchmarkContractError(
            "Resource preflight failed: " + ", ".join(preflight["failures"])
        )
    swap_before = preflight.get("wsl_swap_used_bytes")
    name = _safe_container_name(candidate.key, round_index, run_dir.name)
    start_evidence: dict[str, Any] = {}
    pipeline_result: dict[str, Any] = {}
    if full_auto:
        cgroup_swap_peaks: list[int | None] = []
        for container_name in ("paddleocr-llamacpp", "gemma-local-server"):
            _stop_exact_container(container_name)
        try:
            pipeline_result = _run_product_pipeline(
                candidate=candidate,
                run_dir=run_dir,
                sample_dir=sample_dir,
                sample_count=sample_count,
                endpoint_port=18000,
                python_executable=python_executable,
                stage_ceiling="render",
                runtime_mode="attach-running",
                runtime_services="full",
                product_managed_runtime=True,
                extra_environment={
                    "PADDLEOCR_LLAMA_SLEEP_IDLE_SECONDS": str(
                        candidate.sleep_idle_seconds
                    )
                },
            )
        finally:
            for container_name in (
                "gemma-local-server",
                "paddleocr-llamacpp",
            ):
                cgroup_swap_peaks.append(
                    _container_swap_peak_bytes(container_name)
                )
                _stop_exact_container(container_name)
        swap_after = _wsl_swap_used_bytes()
        swap_delta = (
            None
            if swap_before is None or swap_after is None
            else max(0, int(swap_after) - int(swap_before))
        )
        swap_gate = swap_gate_report(
            global_delta_bytes=swap_delta,
            cgroup_peak_bytes=cgroup_swap_peaks,
        )
        result = {
            "candidate": asdict(candidate),
            "execution_mode": "managed-full-auto",
            "pipeline": pipeline_result,
            "wsl_swap_before_bytes": swap_before,
            "wsl_swap_after_bytes": swap_after,
            "wsl_swap_delta_bytes": swap_delta,
            "swap_gate": swap_gate,
            "swap_gate_pass": swap_gate["passed"],
            "windows_ram_gate_pass": bool(
                (pipeline_result.get("host_resource_monitor") or {}).get(
                    "windows_available_gate_pass", False
                )
            ),
        }
        _json_write(run_dir / "run-result.json", result)
        return result

    try:
        start_evidence = _start_paddle_container(
            candidate,
            name=name,
            port=port,
        )
        runtime_ready = _runtime_evidence(name, port)
        context_report = slot_context_report(
            props=(runtime_ready.get("props") or {}),
            slots=(runtime_ready.get("slots") or {}),
            expected_parallel=candidate.n_parallel,
        )
        runtime_ready["slot_context_report"] = context_report
        _json_write(run_dir / "runtime-ready.json", runtime_ready)
        if not context_report["passed"]:
            raise BenchmarkContractError(
                "Paddle slot context contract failed: "
                + ", ".join(context_report["failures"])
            )
        if frozen_workload is None:
            pipeline_result = _run_product_pipeline(
                candidate=candidate,
                run_dir=run_dir,
                sample_dir=sample_dir,
                sample_count=sample_count,
                endpoint_port=port,
                python_executable=python_executable,
            )
        else:
            pipeline_result = run_frozen_ocr_replay(
                workload=frozen_workload,
                candidate=candidate,
                endpoint=f"http://127.0.0.1:{port}/v1/chat/completions",
            )
            _json_write(run_dir / "frozen-replay.json", pipeline_result)
        runtime_after = _runtime_evidence(name, port)
        runtime_after["container"]["cgroup_swap_peak_bytes"] = (
            _container_swap_peak_bytes(name)
        )
        _json_write(run_dir / "runtime-after.json", runtime_after)
    finally:
        logs = _run(
            ["docker", "logs", "--tail", "500", name],
            check=False,
        )
        (run_dir / "container.log").write_text(
            (logs.stdout or "") + (logs.stderr or ""), encoding="utf-8"
        )
        _remove_exact_lab_container(name)
    swap_after = _wsl_swap_used_bytes()
    swap_delta = (
        None
        if swap_before is None or swap_after is None
        else max(0, int(swap_after) - int(swap_before))
    )
    swap_gate = swap_gate_report(
        global_delta_bytes=swap_delta,
        cgroup_peak_bytes=[
            ((runtime_after.get("container") or {}).get("cgroup_swap_peak_bytes"))
        ],
    )
    result = {
        "candidate": asdict(candidate),
        "execution_mode": (
            "isolated-frozen-ocr-replay"
            if frozen_workload is not None
            else "isolated-ocr-ceiling"
        ),
        "start": start_evidence,
        "pipeline": pipeline_result,
        "wsl_swap_before_bytes": swap_before,
        "wsl_swap_after_bytes": swap_after,
        "wsl_swap_delta_bytes": swap_delta,
        "swap_gate": swap_gate,
        "swap_gate_pass": swap_gate["passed"],
        "windows_ram_gate_pass": bool(
            (pipeline_result.get("host_resource_monitor") or {}).get(
                "windows_available_gate_pass", False
            )
        ),
    }
    _json_write(run_dir / "run-result.json", result)
    return result


def _pipeline_wall_value(result: Mapping[str, Any]) -> float:
    pipeline = result.get("pipeline")
    if not isinstance(pipeline, Mapping):
        raise BenchmarkContractError("Missing pipeline timing result.")
    value = float(pipeline.get("wall_sec", 0.0) or 0.0)
    if str(result.get("execution_mode", "")).startswith("isolated-"):
        start = result.get("start")
        if isinstance(start, Mapping):
            value += float(start.get("start_to_health_sec", 0.0) or 0.0)
    if value <= 0:
        raise BenchmarkContractError("Pipeline wall time must be positive.")
    return value


def _runtime_wall_value(
    result: Mapping[str, Any],
    *,
    runtime_name: str,
    metric_name: str,
) -> float:
    pipeline = result.get("pipeline")
    if not isinstance(pipeline, Mapping):
        raise BenchmarkContractError("Missing pipeline timing result.")
    summary = pipeline.get("summary")
    stats = summary.get("performance_stats") if isinstance(summary, Mapping) else None
    runtime = stats.get("runtime") if isinstance(stats, Mapping) else None
    runtime_stats = runtime.get(runtime_name) if isinstance(runtime, Mapping) else None
    milliseconds = (
        runtime_stats.get(metric_name)
        if isinstance(runtime_stats, Mapping)
        else None
    )
    value = float(milliseconds or 0.0) / 1000.0
    if value <= 0:
        raise BenchmarkContractError(
            f"Missing positive runtime metric: {runtime_name}.{metric_name}"
        )
    return value


def _axis_timing_value(
    result: Mapping[str, Any],
    *,
    axis: str,
) -> float:
    if axis == "handoff":
        return _runtime_wall_value(
            result,
            runtime_name="paddleocr_vl",
            metric_name="release_wall_ms",
        )
    pipeline = result.get("pipeline")
    if not isinstance(pipeline, Mapping):
        raise BenchmarkContractError("Missing pipeline timing result.")
    request = pipeline.get("ocr_request_metrics")
    if isinstance(request, Mapping):
        milliseconds = float(request.get("request_wall_ms", 0.0) or 0.0)
        if milliseconds > 0:
            return milliseconds / 1000.0
    raise BenchmarkContractError("Missing positive OCR request-wall metric.")


def execute_pair_matrix(
    candidate: ServingCandidate,
    *,
    reference_candidate: ServingCandidate = BASELINE,
    output_dir: Path,
    sample_dir: Path,
    sample_count: int,
    initial_rounds: int,
    max_rounds: int,
    python_executable: str,
    shared_frozen_workload: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if candidate.key == reference_candidate.key:
        raise BenchmarkContractError("Candidate must differ from the reference.")
    max_rounds = max(2, min(DEFAULT_MAX_ROUNDS, int(max_rounds)))
    initial_rounds = max(2, min(max_rounds, int(initial_rounds)))
    pairs: list[dict[str, Any]] = []
    reference_sha = ""
    snapshot_mismatches: list[dict[str, Any]] = []
    full_auto = candidate.axis == "handoff"
    frozen_workload: Mapping[str, Any] | None = shared_frozen_workload
    if not full_auto:
        if frozen_workload is None:
            capture_dir = output_dir / candidate.key / "frozen-capture"
            capture_dir.mkdir(parents=True, exist_ok=False)
            execute_candidate_once(
                BASELINE,
                run_dir=capture_dir,
                sample_dir=sample_dir,
                sample_count=sample_count,
                round_index=0,
                python_executable=python_executable,
            )
            frozen_workload = freeze_ocr_workload(
                capture_dir / "page_snapshots.json",
                output_path=output_dir / candidate.key / "frozen-workload.json",
            )
    for round_index in range(1, max_rounds + 1):
        order = (
            (reference_candidate, candidate)
            if round_index % 2
            else (candidate, reference_candidate)
        )
        round_results: dict[str, dict[str, Any]] = {}
        for profile in order:
            run_dir = output_dir / candidate.key / f"round-{round_index}" / profile.key
            run_dir.mkdir(parents=True, exist_ok=False)
            result = execute_candidate_once(
                profile,
                run_dir=run_dir,
                sample_dir=sample_dir,
                sample_count=sample_count,
                round_index=round_index,
                python_executable=python_executable,
                full_auto=full_auto,
                frozen_workload=frozen_workload,
            )
            snapshot_sha = str(
                ((result.get("pipeline") or {}).get("snapshot_sha256")) or ""
            )
            if not reference_sha:
                reference_sha = snapshot_sha
            if snapshot_sha != reference_sha:
                snapshot_mismatches.append(
                    {
                        "round": round_index,
                        "profile": profile.key,
                        "expected_sha256": reference_sha,
                        "actual_sha256": snapshot_sha,
                    }
                )
            round_results[profile.key] = result
        pair = {
            "round": round_index,
            "order": [profile.key for profile in order],
            "baseline_sec": _pipeline_wall_value(
                round_results[reference_candidate.key]
            ),
            "candidate_sec": _pipeline_wall_value(
                round_results[candidate.key]
            ),
            "baseline_axis_sec": _axis_timing_value(
                round_results[reference_candidate.key], axis=candidate.axis
            ),
            "candidate_axis_sec": _axis_timing_value(
                round_results[candidate.key], axis=candidate.axis
            ),
            "execution_mode": (
                "managed-full-auto"
                if full_auto
                else "isolated-frozen-ocr-replay"
            ),
            "snapshot_sha256": reference_sha,
            "baseline": round_results[reference_candidate.key],
            "candidate": round_results[candidate.key],
        }
        pairs.append(pair)
        summary = summarize_pairs(
            [item["baseline_sec"] for item in pairs],
            [item["candidate_sec"] for item in pairs],
        )
        axis_summary = summarize_pairs(
            [item["baseline_axis_sec"] for item in pairs],
            [item["candidate_axis_sec"] for item in pairs],
        )
        _json_write(
            output_dir / candidate.key / "pair-summary.json",
            {
                "protocol_version": PROTOCOL_VERSION,
                "reference_candidate": asdict(reference_candidate),
                "candidate": asdict(candidate),
                "snapshot_sha256": reference_sha,
                "rounds": pairs,
                "statistics": summary,
                "axis_statistics": axis_summary,
                "snapshot_mismatches": snapshot_mismatches,
            },
        )
        if should_stop_pair_matrix(
            summary=summary,
            axis_summary=axis_summary,
            rounds=round_index,
            initial_rounds=initial_rounds,
            snapshot_mismatches=snapshot_mismatches,
        ):
            break
    final_statistics = summarize_pairs(
        [item["baseline_sec"] for item in pairs],
        [item["candidate_sec"] for item in pairs],
    )
    final_axis_statistics = summarize_pairs(
        [item["baseline_axis_sec"] for item in pairs],
        [item["candidate_axis_sec"] for item in pairs],
    )
    resource_ok = all(
        bool(item[role].get("swap_gate_pass", False))
        and bool(item[role].get("windows_ram_gate_pass", False))
        for item in pairs
        for role in ("baseline", "candidate")
    )
    return {
        "protocol_version": PROTOCOL_VERSION,
        "reference_candidate": asdict(reference_candidate),
        "candidate": asdict(candidate),
        "snapshot_sha256": reference_sha,
        "round_count": len(pairs),
        "statistics": final_statistics,
        "axis_statistics": final_axis_statistics,
        "quality_exact": not snapshot_mismatches,
        "snapshot_mismatches": snapshot_mismatches,
        "resource_gate_pass": resource_ok,
        "promotion_eligible": bool(
            not snapshot_mismatches
            and resource_ok
            and float(
                final_statistics["one_sided_95_bootstrap_lower_percent"]
            )
            > 0.0
            and float(
                final_axis_statistics[
                    "one_sided_95_bootstrap_lower_percent"
                ]
            )
            > 0.0
        ),
    }


def render_public_plan() -> str:
    catalog = candidate_catalog()
    lines = [
        "# Serving scheduler matrix",
        "",
        f"- protocol: `{PROTOCOL_VERSION}`",
        f"- llama.cpp: `{PINNED_LLAMA_BUILD}` / `{PINNED_LLAMA_REVISION[:12]}`",
        f"- residency preflight threshold: `{RESIDENCY_PREFLIGHT_RATIO:.0%}`",
        "- exact OCR snapshot required: `true`",
        "- maximum adaptive rounds: `7`",
        "",
        "| phase | candidate | np | workers | http | poll | poll-batch | idle |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for phase, keys in staged_candidate_keys().items():
        for key in keys:
            item = catalog[key]
            lines.append(
                f"| {phase} | {key} | {item.n_parallel} | "
                f"{item.client_workers} | {item.threads_http} | {item.poll} | "
                f"{item.poll_batch} | {item.sleep_idle_seconds} |"
            )
    lines.extend(
        [
            "",
            "Batch/ubatch, completion token and folder-global queue are omitted "
            "because their direct llama.cpp CUDA decisions already exhausted the "
            "adaptive comparison without proving a positive gain.",
            "",
        ]
    )
    return "\n".join(lines)


def _selected_candidates(values: Sequence[str]) -> list[ServingCandidate]:
    catalog = candidate_catalog()
    requested = list(values or [])
    if not requested:
        requested = [key for keys in staged_candidate_keys().values() for key in keys]
    missing = sorted(set(requested) - set(catalog))
    if missing:
        raise BenchmarkContractError(
            "Unknown candidate(s): " + ", ".join(missing)
        )
    if BASELINE.key in requested:
        raise BenchmarkContractError("Do not select baseline as a candidate.")
    return [catalog[key] for key in requested]


def _selected_reference(value: str) -> ServingCandidate:
    catalog = candidate_catalog()
    if value not in catalog:
        raise BenchmarkContractError(f"Unknown reference candidate: {value}")
    return catalog[value]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the private serving/scheduler matrix."
    )
    parser.add_argument(
        "--mode",
        choices=("plan", "preflight", "paddle-matrix"),
        default="plan",
    )
    parser.add_argument("--candidate", action="append", default=[])
    parser.add_argument("--reference-candidate", default=BASELINE.key)
    parser.add_argument("--sample-dir", type=Path, default=ROOT / "Sample" / "japan")
    parser.add_argument("--sample-count", type=int, default=6)
    parser.add_argument("--initial-rounds", type=int, default=DEFAULT_INITIAL_ROUNDS)
    parser.add_argument("--max-rounds", type=int, default=DEFAULT_MAX_ROUNDS)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--output-dir", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.mode == "plan":
        print(render_public_plan())
        return 0

    output_dir, managed_run = select_managed_output_directory(
        family=FAMILY_NAME,
        category=ARTIFACT_CATEGORY,
        explicit_output_directory=args.output_dir,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        contract = runtime_contract_snapshot()
        resources = resource_preflight()
        gpu_resources = resources.get("gpu")
        physical_mib = (
            int(gpu_resources.get("total_mib", 0) or 0)
            if isinstance(gpu_resources, Mapping)
            else 0
        )
        if physical_mib <= 0:
            raise BenchmarkContractError(
                "Unable to determine physical GPU memory for residency preflight."
            )
        residency = residency_preflight(
            physical_mib=physical_mib,
            paddle_peak_mib=PADDLE_MEASURED_PEAK_MIB,
            gemma_peak_mib=GEMMA_MEASURED_PEAK_MIB,
        )
        preflight = {
            "protocol_version": PROTOCOL_VERSION,
            "runtime_contract": contract,
            "resources": resources,
            "residency": residency,
            "router_preset_sha256": hashlib.sha256(
                build_router_preset().encode("utf-8")
            ).hexdigest(),
            "router_command": build_router_command(),
        }
        _json_write(output_dir / "preflight.json", preflight)
        (output_dir / "matrix-plan.md").write_text(
            render_public_plan(), encoding="utf-8"
        )
        if not contract["passed"]:
            raise BenchmarkContractError(
                "Pinned llama.cpp runtime contract failed: "
                + ", ".join(contract["failures"])
            )
        if not resources["passed"]:
            raise BenchmarkContractError(
                "Resource preflight failed: "
                + ", ".join(resources["failures"])
            )
        if args.mode == "preflight":
            result = {"status": "passed", **preflight}
        else:
            sample_dir = args.sample_dir.resolve()
            if not sample_dir.is_dir():
                raise BenchmarkContractError(
                    f"Sample directory does not exist: {sample_dir}"
                )
            candidates = _selected_candidates(args.candidate)
            reference_candidate = _selected_reference(args.reference_candidate)
            if any(candidate.key == reference_candidate.key for candidate in candidates):
                raise BenchmarkContractError(
                    "A candidate cannot also be the reference candidate."
                )
            shared_frozen_workload: Mapping[str, Any] | None = None
            if any(candidate.axis != "handoff" for candidate in candidates):
                capture_dir = output_dir / "shared-frozen-capture"
                capture_dir.mkdir(parents=True, exist_ok=False)
                execute_candidate_once(
                    BASELINE,
                    run_dir=capture_dir,
                    sample_dir=sample_dir,
                    sample_count=max(1, int(args.sample_count)),
                    round_index=0,
                    python_executable=str(args.python),
                )
                shared_frozen_workload = freeze_ocr_workload(
                    capture_dir / "page_snapshots.json",
                    output_path=output_dir / "shared-frozen-workload.json",
                )
            results = []
            for candidate in candidates:
                results.append(
                    execute_pair_matrix(
                        candidate,
                        reference_candidate=reference_candidate,
                        output_dir=output_dir,
                        sample_dir=sample_dir,
                        sample_count=max(1, int(args.sample_count)),
                        initial_rounds=args.initial_rounds,
                        max_rounds=args.max_rounds,
                        python_executable=str(args.python),
                        shared_frozen_workload=(
                            shared_frozen_workload
                            if candidate.axis != "handoff"
                            else None
                        ),
                    )
                )
                _json_write(
                    output_dir / "matrix-summary.json",
                    {
                        "protocol_version": PROTOCOL_VERSION,
                        "status": "running",
                        "results": results,
                    },
                )
            result = {
                "protocol_version": PROTOCOL_VERSION,
                "status": "completed",
                "results": results,
            }
            _json_write(output_dir / "matrix-summary.json", result)
        if managed_run is not None:
            managed_run.complete(
                metadata={
                    "protocol_version": PROTOCOL_VERSION,
                    "mode": args.mode,
                    "status": "completed",
                }
            )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except BaseException as exc:
        if managed_run is not None:
            managed_run.fail(
                exc,
                metadata={
                    "protocol_version": PROTOCOL_VERSION,
                    "mode": args.mode,
                },
            )
        raise


if __name__ == "__main__":
    raise SystemExit(main())

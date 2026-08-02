#!/usr/bin/env python3
"""Run exactly one ABBA screen for each single-router OCR/Gemma pair.

This is intentionally a small lab runner, not a scheduler replacement.  It
never retries or expands a pair beyond ``baseline -> router -> router ->
baseline``.  Source images, request/response ledgers, rendered pages, resource
samples, and semantic-review inputs stay under the managed private archive.
Only an aggregate report is written below ``docs/``.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import threading
import time
from typing import Any, Mapping, Sequence
import re

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import benchmark_translation_semantic_review as semantic_review
import validation_artifact_harness as artifact_harness
from modules.utils.gpu_metrics import query_gpu_metrics


FAMILY = "llamacpp-router-handoff"
CATEGORY = "60-runtime-release"
FIXTURE_SCHEMA = "llamacpp-router-handoff-fixtures-v1"
ARMS = ("baseline", "router", "router", "baseline")
SEED = 20260801
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
LAB_LABEL_PROTOCOL = "com.comictranslate.benchmark-protocol"
LAB_LABEL_PAIR = "com.comictranslate.benchmark-pair"
LAB_LABEL_OWNER = "com.comictranslate.benchmark-owner"
_SHA256_LENGTH = 64
_OWNED_ROUTER_CONTAINER_RE = re.compile(r"^ct-router-lab-[A-Za-z0-9][A-Za-z0-9_.-]*$")
_SAFE_DOCKER_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_PRE_TRANSLATION_BLOCK_KEYS = (
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
    "block_final_mask_pixel_count",
    "block_mask_iou",
    "block_mask_span_coverage",
    "block_mask_bbox",
    "block_mask_source",
    "block_mask_decision",
)
_PRIVATE_STAGE_CONTRACT_HASH_KEYS = (
    "detection_sha256",
    "ocr_raw_results_sha256",
    "ocr_page_profile_sha256",
    "inpaint_decoded_pixel_sha256",
    "inpaint_diagnostics_sha256",
)
_VOLATILE_STAGE_KEYS = frozenset({"updated_at", "started_at", "ended_at", "elapsed_sec", "cache_status"})
_KNOWN_RUNTIME_CONTAINERS = frozenset(
    {
        "gemma-local-server",
        "paddleocr-llamacpp",
        "paddleocr-spotting-llamacpp",
        "hunyuanocr-local-server",
        "mangalmm-local-server",
    }
)


def _pair_catalog() -> dict[str, Any]:
    # Keep the runner importable in the lightweight test environment.  The
    # adapter itself imports the product OCR stack only when an actual router
    # arm is launched by the supported Windows runtime.
    from router_handoff_lab_runtime import pair_catalog

    return pair_catalog()


class RouterHandoffBenchmarkError(RuntimeError):
    """The fixed one-pass router benchmark could not prove its contract."""


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_read(path: Path) -> dict[str, Any]:
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RouterHandoffBenchmarkError(f"Invalid private JSON artifact: {path.name}") from exc
    if not isinstance(parsed, dict):
        raise RouterHandoffBenchmarkError(f"Private JSON artifact is not an object: {path.name}")
    return parsed


def _json_write(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _run(command: Sequence[str], *, timeout: float = 120.0, check: bool = True) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        list(command),
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        errors="replace",
        check=False,
        timeout=timeout,
    )
    if check and completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "").strip()
        raise RouterHandoffBenchmarkError(
            f"Command failed ({completed.returncode}): {Path(command[0]).name}\n{detail[-4096:]}"
        )
    return completed


def _normalise_response(value: Any) -> dict[str, Any]:
    choices = value.get("choices") if isinstance(value, Mapping) else None
    normalized: list[dict[str, Any]] = []
    for index, choice in enumerate(choices if isinstance(choices, list) else []):
        if not isinstance(choice, Mapping):
            continue
        message = choice.get("message")
        content = message.get("content") if isinstance(message, Mapping) else choice.get("content")
        normalized.append(
            {
                "index": int(choice.get("index", index) or index),
                "content": str(content or ""),
                "finish_reason": str(choice.get("finish_reason", "") or ""),
            }
        )
    return {"choices": normalized}


def _build_request_ledger(*, preset: Mapping[str, Any], metrics_path: Path, record_path: Path) -> dict[str, Any]:
    translation_order: list[dict[str, Any]] = []
    if metrics_path.is_file():
        for raw in metrics_path.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                row = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if isinstance(row, Mapping) and row.get("tag") == "translate_start":
                translation_order.append(
                    {
                        "image_index": row.get("image_index"),
                        "block_count": row.get("block_count"),
                        "translator_key": row.get("translator_key"),
                    }
                )
    gemma = preset.get("gemma") if isinstance(preset.get("gemma"), Mapping) else {}
    fixed_contract = {
        "model": gemma.get("model"),
        "context_size": gemma.get("context_size"),
        "n_gpu_layers": gemma.get("n_gpu_layers"),
        "n_parallel": gemma.get("n_parallel"),
        "threads": gemma.get("threads"),
        "batch_size": gemma.get("batch_size"),
        "ubatch_size": gemma.get("ubatch_size"),
        "cache_type_k": gemma.get("cache_type_k"),
        "seed": (preset.get("benchmark_http") or {}).get("gemma_seed"),
    }
    parse_failures: list[str] = []
    attempts: list[dict[str, Any]] = []
    responses: list[dict[str, Any]] = []
    record_summary: Mapping[str, Any] | None = None
    if not record_path.is_file():
        parse_failures.append("gemma_http_record_missing")
    else:
        for line_number, raw in enumerate(
            record_path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1
        ):
            try:
                row = json.loads(raw)
            except json.JSONDecodeError:
                parse_failures.append(f"invalid_record_line_{line_number}")
                continue
            if not isinstance(row, Mapping):
                parse_failures.append(f"nonobject_record_line_{line_number}")
                continue
            if row.get("record_type") == "summary":
                if record_summary is not None:
                    parse_failures.append("duplicate_record_summary")
                record_summary = row
                continue
            request = row.get("request")
            response = row.get("response")
            if not isinstance(request, Mapping):
                parse_failures.append(f"request_missing_line_{line_number}")
                continue
            try:
                index = int(row.get("attempt_index", len(attempts)))
            except (TypeError, ValueError):
                index = len(attempts)
                parse_failures.append(f"invalid_attempt_index_line_{line_number}")
            if str(row.get("error", "") or ""):
                parse_failures.append(f"http_error_line_{line_number}")
            if request.get("model") != GEMMA_ALIAS:
                parse_failures.append(f"model_mismatch_line_{line_number}")
            if request.get("seed") != SEED:
                parse_failures.append(f"seed_mismatch_line_{line_number}")
            if not isinstance(request.get("messages"), list) or "response_format" not in request:
                parse_failures.append(f"prompt_or_schema_missing_line_{line_number}")
            status = row.get("status_code")
            if not isinstance(status, int) or not 200 <= status < 300:
                parse_failures.append(f"status_invalid_line_{line_number}")
            canonical_response = _normalise_response(response)
            try:
                semantic_review.validate_translation_response(canonical_response, index=index)
            except semantic_review.SemanticReviewError:
                parse_failures.append(f"response_contract_invalid_line_{line_number}")
            attempts.append(
                {
                    "attempt_index": index,
                    "model": str(request.get("model", "") or ""),
                    "prompt_sha256": _canonical_sha256(request.get("messages")),
                    "schema_sha256": _canonical_sha256(request.get("response_format")),
                    "seed": request.get("seed"),
                    "payload_sha256": _canonical_sha256(request),
                }
            )
            responses.append({"attempt_index": index, "canonical_response": canonical_response})
    attempts.sort(key=lambda row: int(row["attempt_index"]))
    responses.sort(key=lambda row: int(row["attempt_index"]))
    if not attempts:
        parse_failures.append("gemma_http_record_empty")
    if record_summary is None:
        parse_failures.append("gemma_http_summary_missing")
    else:
        try:
            summary_count = int(record_summary.get("attempt_count"))
        except (TypeError, ValueError):
            summary_count = -1
        if summary_count != len(attempts):
            parse_failures.append("gemma_http_summary_attempt_count_mismatch")
        if str(record_summary.get("write_error", "") or ""):
            parse_failures.append("gemma_http_writer_error")
        if [row["attempt_index"] for row in attempts] != list(range(max(0, summary_count))):
            parse_failures.append("gemma_http_attempt_order_invalid")
    payload = {
        "fixed_contract": fixed_contract,
        "translation_start_order": translation_order,
        "actual_http_attempts": attempts,
    }
    return {
        **payload,
        "response_ledger": {"rows": responses, "sha256": _canonical_sha256(responses)},
        "record_complete": not parse_failures,
        "record_failures": parse_failures,
        "sha256": _canonical_sha256(payload),
    }


def _pre_translation_snapshot_sha256(snapshot: Mapping[str, Any]) -> str:
    pages = snapshot.get("pages")
    if not isinstance(pages, list):
        raise RouterHandoffBenchmarkError("Page snapshot has no pages list.")
    canonical_pages: list[dict[str, Any]] = []
    for page in pages:
        if not isinstance(page, Mapping):
            raise RouterHandoffBenchmarkError("Page snapshot item is invalid.")
        stages = page.get("stage_status")
        stage_status: dict[str, Any] = {}
        if isinstance(stages, Mapping):
            for name in ("detect", "ocr", "inpaint"):
                stage = stages.get(name)
                if isinstance(stage, Mapping):
                    stage_status[name] = {
                        key: value for key, value in stage.items() if key not in _VOLATILE_STAGE_KEYS
                    }
        blocks = page.get("blocks")
        if not isinstance(blocks, list):
            raise RouterHandoffBenchmarkError("Page snapshot blocks are invalid.")
        private_contract = page.get("private_stage_contract")
        if not isinstance(private_contract, Mapping):
            raise RouterHandoffBenchmarkError(
                "Page snapshot has no private upstream contract."
            )
        contract_hashes: dict[str, str] = {}
        for key in _PRIVATE_STAGE_CONTRACT_HASH_KEYS:
            value = str(private_contract.get(key, "") or "")
            if len(value) != _SHA256_LENGTH:
                raise RouterHandoffBenchmarkError(
                    f"Page snapshot upstream contract hash is missing: {key}"
                )
            contract_hashes[key] = value
        canonical_pages.append(
            {
                "source_lang": page.get("source_lang"),
                "target_lang": page.get("target_lang"),
                "ocr_quality": page.get("ocr_quality", {}),
                "stage_status": stage_status,
                "private_stage_contract": contract_hashes,
                "blocks": [
                    {key: block.get(key) for key in _PRE_TRANSLATION_BLOCK_KEYS}
                    for block in blocks
                    if isinstance(block, Mapping)
                ],
            }
        )
    return _canonical_sha256({"page_count": len(canonical_pages), "pages": canonical_pages})


def _full_auto_quality_failures(snapshot: Mapping[str, Any], summary: Mapping[str, Any]) -> list[str]:
    failures: list[str] = []
    pages = snapshot.get("pages")
    if not isinstance(pages, list) or len(pages) != 1:
        return ["snapshot_page_count_mismatch"]
    if int(summary.get("page_failed_count", -1)) != 0:
        failures.append("summary_page_failed")
    if int(summary.get("page_done_count", -1)) != 1:
        failures.append("summary_page_done_count_mismatch")
    page = pages[0]
    if not isinstance(page, Mapping):
        return failures + ["snapshot_page_invalid"]
    if bool(page.get("page_failed", False)) or not bool(page.get("translated_image_exists", False)):
        failures.append("render_missing_or_failed")
    pixel_sha = str(page.get("translated_image_decoded_pixel_sha256", "") or "")
    if len(pixel_sha) != _SHA256_LENGTH:
        failures.append("render_pixel_hash_missing")
    private_contract = page.get("private_stage_contract")
    if not isinstance(private_contract, Mapping):
        failures.append("private_upstream_contract_missing")
    else:
        for key in _PRIVATE_STAGE_CONTRACT_HASH_KEYS:
            if len(str(private_contract.get(key, "") or "")) != _SHA256_LENGTH:
                failures.append(f"private_upstream_contract_hash_missing:{key}")
    return failures


def _snapshot_pixels(snapshot: Mapping[str, Any]) -> list[str]:
    pages = snapshot.get("pages")
    if not isinstance(pages, list):
        return []
    return [str(page.get("translated_image_decoded_pixel_sha256", "") or "") for page in pages if isinstance(page, Mapping)]


def _gpu_used_mib() -> int | None:
    metrics = query_gpu_metrics()
    primary = metrics.get("primary") if isinstance(metrics, Mapping) else None
    try:
        return int(primary["memory_used_mb"])  # type: ignore[index]
    except (KeyError, TypeError, ValueError):
        return None


def _wsl_swap_bytes() -> int | None:
    try:
        values = {
            key.rstrip(":"): int(value) * 1024
            for key, value, *_ in (
                line.split() for line in Path("/proc/meminfo").read_text().splitlines() if line.startswith(("SwapTotal:", "SwapFree:"))
            )
        }
        return max(0, values["SwapTotal"] - values["SwapFree"])
    except (OSError, KeyError, ValueError):
        return None


def _windows_available_bytes() -> int | None:
    executable = shutil.which("powershell.exe") or shutil.which("powershell")
    if not executable:
        return None
    completed = _run(
        [
            executable,
            "-NoProfile",
            "-Command",
            "[int64]((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory * 1KB)",
        ],
        timeout=20.0,
        check=False,
    )
    try:
        return int((completed.stdout or "").strip().splitlines()[-1]) if completed.returncode == 0 else None
    except (IndexError, ValueError):
        return None


def _shared_gpu_mib() -> float | None:
    executable = shutil.which("powershell.exe") or shutil.which("powershell")
    if not executable:
        return None
    command = (
        "$s=(Get-Counter '\\GPU Process Memory(*)\\Shared Usage' -ErrorAction Stop).CounterSamples;"
        "[math]::Round((($s|Measure-Object CookedValue -Sum).Sum / 1MB),3)"
    )
    completed = _run([executable, "-NoProfile", "-Command", command], timeout=20.0, check=False)
    try:
        return float((completed.stdout or "").strip().splitlines()[-1]) if completed.returncode == 0 else None
    except (IndexError, ValueError):
        return None


@dataclass
class ResourceSampler:
    samples: list[dict[str, Any]]

    def __init__(self) -> None:
        self.samples = []
        self._background: dict[str, Any] | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        # Capture a synchronous pre-spawn sample.  A background thread alone
        # can race Popen and turn the first measurement into a peak sample.
        self._background = self._sample("background")
        self.samples.append(dict(self._background))
        self._thread = threading.Thread(target=self._collect, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)

    def _collect(self) -> None:
        while not self._stop.is_set():
            self.samples.append(self._sample("interval"))
            self._stop.wait(1.0)

    @staticmethod
    def _sample(phase: str) -> dict[str, Any]:
        return {
            "phase": phase,
            "monotonic": round(time.monotonic(), 6),
            "gpu_used_mib": _gpu_used_mib(),
            "windows_available_bytes": _windows_available_bytes(),
            "shared_gpu_mib": _shared_gpu_mib(),
            "wsl_swap_bytes": _wsl_swap_bytes(),
        }

    def summary(self) -> dict[str, Any]:
        def values(key: str) -> list[float]:
            return [float(row[key]) for row in self.samples if isinstance(row.get(key), (int, float))]

        gpu = values("gpu_used_mib")
        ram = values("windows_available_bytes")
        shared = values("shared_gpu_mib")
        swap = values("wsl_swap_bytes")
        return {
            "sample_count": len(self.samples),
            "background": dict(self._background or {}),
            "peak_vram_mib": max(gpu) if gpu else None,
            "minimum_windows_available_bytes": min(ram) if ram else None,
            "peak_shared_gpu_mib": max(shared) if shared else None,
            "wsl_swap_delta_bytes": (max(swap) - min(swap)) if swap else None,
        }


def _fixture_for_pair(manifest: Mapping[str, Any], pair: Any) -> Path:
    if manifest.get("schema_version") != FIXTURE_SCHEMA:
        raise RouterHandoffBenchmarkError("Fixture manifest schema mismatch.")
    pairs = manifest.get("pairs")
    entry = pairs.get(pair.key) if isinstance(pairs, Mapping) else None
    if not isinstance(entry, Mapping):
        raise RouterHandoffBenchmarkError(f"Private fixture missing for {pair.key}.")
    path = Path(str(entry.get("path", "") or "")).expanduser().resolve()
    digest = str(entry.get("sha256", "") or "")
    if not path.is_file() or len(digest) != _SHA256_LENGTH or _sha256_file(path) != digest:
        raise RouterHandoffBenchmarkError(f"Private fixture integrity failed for {pair.key}.")
    return path


def _pair_preset(pair: Any, *, arm: str, arm_dir: Path) -> dict[str, Any]:
    source = ROOT / "benchmarks" / "cold_cache_finalization" / "presets" / "product-v1.1.0-baseline.json"
    preset = copy.deepcopy(_json_read(source))
    preset["name"] = f"{PROTOCOL_VERSION}-{pair.key}-{arm}"
    preset["description"] = "Private one-pass single-router handoff comparison."
    preset["benchmark_contract"] = {
        "protocol_version": PROTOCOL_VERSION,
        "workflow_mode": "stage_batched_pipeline",
        "request_mode": "contextual-single",
        "cache_mode": "disabled",
        "router_handoff": arm == "router",
    }
    app = preset.setdefault("app", {})
    app.update(
        {
            "workflow_mode": "stage_batched_pipeline",
            "translator": "Custom Local Server(Gemma)",
            "ocr": pair.engine_key,
            "detector": "RT-DETR-v2",
            "inpainter": "lama_large_512px",
            "use_gpu": True,
        }
    )
    gemma = preset.setdefault("gemma", {})
    gemma.update(
        {
            "image": PINNED_LLAMA_IMAGE,
            "pull_policy": "never",
            "endpoint_url": "http://127.0.0.1:18080/v1",
            "model": GEMMA_ALIAS,
            "model_path": f"/models/{GEMMA_ALIAS}",
            "model_sha256": GEMMA_SHA256,
            "context_size": 4096,
            "n_parallel": 1,
            "threads": 10,
            "batch_size": 2048,
            "ubatch_size": 512,
            "n_gpu_layers": 23,
            "chunk_size": 6,
            "max_completion_tokens": 512,
            "cache_type_k": "f16",
            "cache_type_v": "f16",
            "spec_type": "none",
        }
    )
    preset["benchmark_http"] = {
        "gemma_session": False,
        "paddle_thread_local_session": False,
        "gemma_seed": SEED,
        "gemma_request_record_path": str(arm_dir / "gemma-http-records.jsonl"),
    }
    preset["benchmark_gemma_pagecache"] = {"mode": "off", "reset_before_run": False}
    preset["benchmark_cache_policy"] = {
        "paddleocr_persistent": False,
        "translation_persistent": False,
        "exact_tm": False,
        "project_checkpoint": False,
    }
    preset["ocr_client"] = {
        "server_url": "http://127.0.0.1:18000/v1/chat/completions",
        # Keep the promoted crop transport's existing client concurrency;
        # this lab changes only container/process handoff, never Paddle's
        # request construction or scheduling intent.
        "parallel_workers": 8,
        "max_new_tokens": 1024,
        "prettify_markdown": False,
        "visualize": False,
    }
    preset["paddle_spotting_ocr_client"] = {
        "server_url": "http://127.0.0.1:18002/v1/chat/completions",
        # Preserve the promoted Spotting engine defaults.  The router may
        # change only model process handoff; its official request contract,
        # completion budget, and timeout remain untouched.
        "max_completion_tokens": 3000,
        "request_timeout_sec": 360,
    }
    preset["hunyuan_ocr_client"] = {
        "server_url": "http://127.0.0.1:28080/v1",
        "max_completion_tokens": 256,
        "parallel_workers": 1,
        "request_timeout_sec": 60,
        "raw_response_logging": False,
    }
    preset["mangalmm_ocr_client"] = {
        "server_url": "http://127.0.0.1:28081/v1",
        # Preserve the official full-page MangaLMM contract.  A smaller
        # completion ceiling or resize budget changes the OCR response itself
        # and is not a router handoff experiment.
        "max_completion_tokens": 4096,
        "parallel_workers": 1,
        "request_timeout_sec": 60,
        "raw_response_logging": False,
        "safe_resize": True,
        "max_pixels": 2116800,
        "max_long_side": 1728,
    }
    if arm == "router":
        safe_name = hashlib.sha256(
            str(arm_dir.resolve()).encode("utf-8")
        ).hexdigest()[:16]
        preset["benchmark_router_handoff"] = {
            "protocol_version": PROTOCOL_VERSION,
            "pair": pair.key,
            "artifact_dir": str(arm_dir),
            "container_name": f"ct-router-lab-{pair.key}-{safe_name}",
            "owner_token": safe_name,
        }
    return preset


def _stage_fixture(source: Path, arm_dir: Path) -> Path:
    fixture_dir = arm_dir / "fixture"
    fixture_dir.mkdir(parents=True, exist_ok=True)
    target = fixture_dir / source.name
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)
    return fixture_dir


def _running_known_containers() -> list[str]:
    completed = _run(["docker", "ps", "--format", "{{.Names}}"], check=False, timeout=20.0)
    if completed.returncode != 0:
        return ["docker_ps_failed"]
    names = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    return [name for name in names if name in _KNOWN_RUNTIME_CONTAINERS or name.startswith("ct-router-lab-")]


def _cleanup_owned_router_container(
    container_name: str | None,
    *,
    pair: str | None = None,
    owner_token: str | None = None,
) -> dict[str, Any]:
    """Stop/remove only a router container deterministically owned by this arm."""

    name = str(container_name or "")
    if not name:
        return {"attempted": False, "orphan": False, "ownership_verified": True}
    if not _OWNED_ROUTER_CONTAINER_RE.fullmatch(name):
        raise RouterHandoffBenchmarkError("Router cleanup received an unsafe container name.")
    if not pair or not _SAFE_DOCKER_NAME.fullmatch(str(pair)):
        raise RouterHandoffBenchmarkError("Router cleanup requires a safe pair identity.")
    if not owner_token or not re.fullmatch(r"[0-9a-f]{16}", str(owner_token)):
        raise RouterHandoffBenchmarkError("Router cleanup requires the exact arm owner token.")
    inspected = _run(
        ["docker", "inspect", name, "--format", "{{json .Config.Labels}}"],
        check=False,
        timeout=20.0,
    )
    if inspected.returncode != 0:
        return {"attempted": False, "orphan": False, "ownership_verified": True}
    try:
        labels = json.loads(inspected.stdout)
    except json.JSONDecodeError:
        labels = None
    expected = {
        LAB_LABEL_PROTOCOL: PROTOCOL_VERSION,
        LAB_LABEL_PAIR: str(pair),
        LAB_LABEL_OWNER: str(owner_token),
    }
    if not isinstance(labels, Mapping) or any(
        str(labels.get(key) or "") != value for key, value in expected.items()
    ):
        return {
            "attempted": False,
            "orphan": False,
            "ownership_verified": False,
            "foreign_container_present": True,
        }
    stop = _run(["docker", "stop", "--timeout", "10", name], check=False, timeout=30.0)
    remove = _run(["docker", "rm", name], check=False, timeout=30.0)
    present = _run(["docker", "inspect", name], check=False, timeout=20.0).returncode == 0
    return {
        "attempted": True,
        "stop_returncode": stop.returncode,
        "remove_returncode": remove.returncode,
        "orphan": present,
        "ownership_verified": True,
    }


def _run_pipeline(
    command: Sequence[str],
    *,
    arm_dir: Path,
    sampler: ResourceSampler,
    owned_router_identity: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    stdout_path = arm_dir / "runner.stdout.log"
    stderr_path = arm_dir / "runner.stderr.log"
    started = time.monotonic()
    timed_out = False
    returncode = -1
    cleanup: dict[str, Any] = {
        "attempted": False,
        "orphan": False,
        "ownership_verified": True,
    }
    try:
        with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
            process = subprocess.Popen(
                list(command),
                cwd=str(ROOT),
                env={
                    **os.environ,
                    "QT_QPA_PLATFORM": "offscreen",
                    "CT_DISABLE_UPDATE_CHECK": "1",
                    "CT_ENABLE_MEMLOG": "1",
                    "CT_ENABLE_GPU_BENCH": "1",
                    "CT_MEMLOG_INTERVAL_SEC": "1",
                    "COMIC_SKIP_STARTUP_MODELS": "1",
                },
                stdout=stdout,
                stderr=stderr,
                text=True,
            )
            while process.poll() is None:
                if time.monotonic() - started >= 1800.0:
                    timed_out = True
                    process.terminate()
                    break
                time.sleep(0.25)
            try:
                returncode = process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                process.kill()
                returncode = process.wait(timeout=20)
    finally:
        # A terminated runner cannot execute its adapter's finally block.
        # Always remove the exact arm-owned container rather than merely
        # reporting an orphan that would contaminate the next arm.
        if owned_router_identity is not None:
            cleanup = _cleanup_owned_router_container(
                owned_router_identity.get("container_name"),
                pair=owned_router_identity.get("pair"),
                owner_token=owned_router_identity.get("owner_token"),
            )
    return {
        "command": list(command),
        "returncode": returncode,
        "timed_out": timed_out,
        "wall_sec": round(time.monotonic() - started, 6),
        "resource_observation": sampler.summary(),
        "owned_router_cleanup": cleanup,
    }


def _router_runtime_failures(runtime: Mapping[str, Any]) -> list[str]:
    failures: list[str] = []
    live = runtime.get("live_contract")
    if not isinstance(live, Mapping) or live.get("verified") is not True:
        failures.append("router_live_identity_missing")
    if runtime.get("ocr_round_trip_verified") is not True:
        failures.append("router_round_trip_missing")
    events = runtime.get("events")
    if not isinstance(events, list):
        return failures + ["router_event_log_missing"]
    loads = [row for row in events if isinstance(row, Mapping) and row.get("event") == "router_load"]
    unloads = [row for row in events if isinstance(row, Mapping) and row.get("event") == "router_unload"]
    if not loads or not unloads:
        failures.append("router_transition_evidence_missing")
    if any(
        row.get("model_state_unloaded") is not True
        or any(state != "unloaded" for state in (row.get("states") or {}).values())
        for row in unloads
    ):
        failures.append("router_unload_state_unconfirmed")
    if any(
        isinstance(row, Mapping) and row.get("event") == "unload_during_stop_failed"
        for row in events
    ):
        failures.append("router_unload_during_stop_failed")
    if any(
        sum(1 for state in (row.get("states") or {}).values() if state == "loaded") > 1
        for row in loads
        if isinstance(row.get("states"), Mapping)
    ):
        failures.append("loaded_count_exceeded")
    if any(isinstance(row, Mapping) and row.get("event") == "router_http_failure" for row in events):
        failures.append("router_http_failure")
    round_trips = [
        row
        for row in events
        if isinstance(row, Mapping)
        and row.get("event") == "ocr_reload_request_unload"
    ]
    if len(round_trips) != 1:
        failures.append("router_round_trip_evidence_missing")
    else:
        gate = round_trips[0].get("gpu_return_gate")
        if not isinstance(gate, Mapping) or not bool(gate.get("observed")):
            failures.append("router_round_trip_gpu_return_unconfirmed")
    return failures


def _router_command_queue_observation(
    runtime: Mapping[str, Any],
    *,
    e2e_seconds: float,
) -> dict[str, Any]:
    """Summarise only the recorded load/unload command-gate waiting time."""

    events = runtime.get("events") if isinstance(runtime, Mapping) else None
    waits = []
    if isinstance(events, list):
        waits = [
            float(event.get("queue_wait_ms", 0.0) or 0.0)
            for event in events
            if isinstance(event, Mapping) and event.get("event") == "arbiter_command_enter"
        ]
    if not waits or e2e_seconds <= 0:
        return {"status": "missing", "sample_count": len(waits)}
    median_ms = statistics.median(waits)
    return {
        "status": "observed",
        "sample_count": len(waits),
        "median_queue_wait_ms": round(median_ms, 6),
        "max_queue_wait_ms": round(max(waits), 6),
        "e2e_percent": round(median_ms / (e2e_seconds * 1000.0) * 100.0, 9),
    }


def _ocr_runtime_service(pair: Any) -> str:
    return {
        "PaddleOCR VL": "paddleocr_vl",
        "PaddleOCR VL Spotting": "paddleocr_vl_spotting",
        "HunyuanOCR": "hunyuanocr",
        "MangaLMM": "mangalmm",
    }[pair.engine_key]


def _router_stage_gpu_return(
    *,
    summary: Mapping[str, Any],
    pair: Any,
) -> tuple[dict[str, Any], list[str]]:
    """Read the product Arbiter's driver-global release evidence.

    The router must not substitute its process-start sample for the stage
    processor's model-start baseline.  This extracts the latter's compact
    telemetry after the pipeline completes and makes a missing/failed return
    gate a candidate failure.
    """

    performance = summary.get("performance_stats")
    runtime = performance.get("runtime") if isinstance(performance, Mapping) else None
    expected = (_ocr_runtime_service(pair), "gemma")
    evidence: dict[str, Any] = {}
    failures: list[str] = []
    for service in expected:
        row = runtime.get(service) if isinstance(runtime, Mapping) else None
        if not isinstance(row, Mapping):
            evidence[service] = {"status": "missing"}
            failures.append(f"gpu_return_gate_missing:{service}")
            continue
        observed = int(row.get("vram_release_gate_observed_count", 0) or 0)
        timed_out = int(row.get("vram_release_gate_timeout_count", 0) or 0)
        unavailable = int(row.get("vram_release_gate_unavailable_count", 0) or 0)
        evidence[service] = {
            "status": "observed" if observed else "unconfirmed",
            "observed_count": observed,
            "timeout_count": timed_out,
            "unavailable_count": unavailable,
            "vram_return_ms": float(row.get("vram_release_gate_wall_ms", 0.0) or 0.0),
        }
        if observed < 1 or timed_out or unavailable:
            failures.append(f"gpu_return_unconfirmed:{service}")
    return evidence, failures


def _execute_arm(pair: Any, *, arm: str, arm_dir: Path, fixture: Path, python_executable: str) -> dict[str, Any]:
    if _running_known_containers():
        raise RouterHandoffBenchmarkError("A managed runtime is already active before an arm.")
    from router_handoff_lab_runtime import verify_pair_model_identities

    # The baseline and router use the locked named model volumes.  Recheck
    # their actual bytes immediately before every arm so an alias/file-name
    # match cannot stand in for runtime model identity.
    model_identities = verify_pair_model_identities(pair)
    arm_dir.mkdir(parents=True, exist_ok=False)
    fixture_dir = _stage_fixture(fixture, arm_dir)
    preset = _pair_preset(pair, arm=arm, arm_dir=arm_dir)
    preset_path = arm_dir / "preset.json"
    _json_write(preset_path, preset)
    command = [
        python_executable,
        str(ROOT / "scripts" / "benchmark_pipeline.py"),
        "--preset",
        str(preset_path),
        "--mode",
        "one-page",
        "--repeat",
        "1",
        "--runtime-mode",
        "attach-running",
        "--runtime-services",
        "full",
        "--sample-dir",
        str(fixture_dir),
        "--sample-count",
        "1",
        "--source-lang",
        pair.source_lang,
        "--target-lang",
        "Korean",
        "--export-page-snapshots",
        "--stage-ceiling",
        "render",
        "--output-dir",
        str(arm_dir),
    ]
    sampler = ResourceSampler()
    sampler.start()
    try:
        router_config = (
            preset.get("benchmark_router_handoff")
            if isinstance(preset.get("benchmark_router_handoff"), Mapping)
            else {}
        )
        process = _run_pipeline(
            command,
            arm_dir=arm_dir,
            sampler=sampler,
            owned_router_identity=(
                {
                    "container_name": str(router_config.get("container_name", "") or ""),
                    "pair": str(router_config.get("pair", "") or ""),
                    "owner_token": str(router_config.get("owner_token", "") or ""),
                }
                if arm == "router"
                else None
            ),
        )
    finally:
        sampler.stop()
    _json_write(arm_dir / "resource-samples.json", {"samples": sampler.samples})
    snapshot = _json_read(arm_dir / "page_snapshots.json") if (arm_dir / "page_snapshots.json").is_file() else {}
    summary = _json_read(arm_dir / "summary.json") if (arm_dir / "summary.json").is_file() else {}
    quality_failures = _full_auto_quality_failures(snapshot, summary) if snapshot and summary else ["pipeline_output_missing"]
    ledger = _build_request_ledger(
        preset=preset,
        metrics_path=arm_dir / "metrics.jsonl",
        record_path=arm_dir / "gemma-http-records.jsonl",
    )
    runtime = _json_read(arm_dir / "router_handoff_lab_runtime.json") if arm == "router" and (arm_dir / "router_handoff_lab_runtime.json").is_file() else {}
    runtime_failures = _router_runtime_failures(runtime) if arm == "router" else []
    stage_gpu_return, stage_gpu_failures = (
        _router_stage_gpu_return(summary=summary, pair=pair)
        if arm == "router" and summary
        else ({}, ["gpu_return_gate_missing:summary"] if arm == "router" else [])
    )
    logs = ""
    for log in (arm_dir / "runner.stdout.log", arm_dir / "runner.stderr.log"):
        if log.is_file():
            logs += log.read_text(encoding="utf-8", errors="replace")
    oom = any(token in logs.lower() for token in ("out of memory", "cuda error", "cuda oom", "oom-kill"))
    orphans = _running_known_containers()
    failures = [
        *quality_failures,
        *ledger.get("record_failures", []),
        *runtime_failures,
        *stage_gpu_failures,
    ]
    if process["returncode"] != 0:
        failures.append("pipeline_failed")
    if process["timed_out"]:
        failures.append("pipeline_timeout")
    cleanup = process.get("owned_router_cleanup")
    if isinstance(cleanup, Mapping) and cleanup.get("ownership_verified") is not True:
        failures.append("owned_router_cleanup_ownership_unverified")
    if isinstance(cleanup, Mapping) and bool(cleanup.get("orphan")):
        failures.append("owned_router_cleanup_failed")
    if oom:
        failures.append("oom")
    if orphans:
        failures.append("container_orphan")
    pages = snapshot.get("pages") if isinstance(snapshot, Mapping) else []
    result = {
        "candidate": {"key": arm, "pair": pair.key},
        "runtime": {
            "image": PINNED_LLAMA_IMAGE if arm == "router" else "product-separate",
            "image_revision": PINNED_LLAMA_REVISION if arm == "router" else "",
            "image_build": PINNED_LLAMA_BUILD if arm == "router" else "",
            "model": GEMMA_ALIAS,
            "model_sha256": GEMMA_SHA256,
            "model_identities": model_identities,
        },
        "status": "passed" if not failures else "rejected",
        "failures": sorted(set(failures)),
        "pipeline_wall_sec": float(process.get("wall_sec", 0.0) or 0.0),
        "summary_elapsed_sec": float(summary.get("elapsed_sec", 0.0) or 0.0),
        "process": process,
        "pre_translation_snapshot_sha256": _pre_translation_snapshot_sha256(snapshot) if snapshot else "",
        "page_output_sha256": _snapshot_pixels(snapshot),
        "request_ledger": ledger,
        "router_runtime": runtime,
        "stage_gpu_return": stage_gpu_return,
        "performance_stats": summary.get("performance_stats", {}) if isinstance(summary, Mapping) else {},
        "resource_observation": sampler.summary(),
        "oom_detected": oom,
        "orphan_containers": orphans,
        "render_page_count": len(pages) if isinstance(pages, list) else 0,
    }
    _json_write(arm_dir / "run-result.json", result)
    return result


def _paired_stats(baselines: Sequence[float], routers: Sequence[float]) -> dict[str, Any]:
    deltas = [base - router for base, router in zip(baselines, routers)]
    improvements = [(base - router) / base * 100.0 for base, router in zip(baselines, routers)]
    return {
        "baseline_median_sec": round(statistics.median(baselines), 6),
        "router_median_sec": round(statistics.median(routers), 6),
        "paired_delta_sec": [round(value, 6) for value in deltas],
        "improvement_percent": [round(value, 6) for value in improvements],
        "median_improvement_percent": round(statistics.median(improvements), 6),
        "descriptive_ci_note": "ABBA one-pass descriptive only; no significance claim.",
    }


def _both_directions_faster(
    *,
    baseline_ab: float,
    router_ab: float,
    router_ba: float,
    baseline_ba: float,
) -> bool:
    """The one-pass decision rule; resource observations do not enter here."""

    return router_ab < baseline_ab and router_ba < baseline_ba


def _baseline_fixture_not_eligible(results: Sequence[Mapping[str, Any]]) -> str:
    """Return the fixture precondition failure that makes ABBA meaningless.

    A representative page must reach translation in *both* independent
    baseline arms.  If current product behavior fails its own OCR-quality gate
    before any Gemma request, a router arm cannot establish an E2E result; it
    is neither a router regression nor evidence of a speed win.
    """

    if len(results) != len(ARMS):
        return ""
    baselines = (results[0], results[3])
    expected = {"gemma_http_record_empty", "summary_page_failed"}
    if all(
        expected.issubset(
            {
                str(failure)
                for failure in result.get("failures", [])
                if isinstance(failure, str)
            }
        )
        for result in baselines
    ):
        return "baseline_fixture_ocr_quality_gate_failed_before_translation"
    return ""


def _evaluate_pair(
    pair: Any,
    results: Sequence[Mapping[str, Any]],
    pair_dir: Path,
    *,
    semantic_approval: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    baseline_a, router_ab, router_ba, baseline_b = results
    ineligible_reason = _baseline_fixture_not_eligible(results)
    if ineligible_reason:
        return {
            "protocol_version": PROTOCOL_VERSION,
            "pair": pair.key,
            "arm_order": list(ARMS),
            "results": list(results),
            "statistics": {},
            "semantic_review": {
                "status": "NOT_APPLICABLE",
                "mismatch_count": 0,
                "comparison_count": 0,
                "unresolved_count": 0,
                "semantic_reject_count": 0,
                "bindings": [],
            },
            "failures": [],
            "reason": ineligible_reason,
            "status": "NOT_ELIGIBLE",
        }
    failures = [
        failure
        for result in results
        for failure in result.get("failures", [])
        if isinstance(failure, str)
    ]
    semantic_bindings: list[dict[str, Any]] = []
    semantic_errors: list[str] = []
    queue_observations: list[dict[str, Any]] = []
    for result in (router_ab, router_ba):
        queue = _router_command_queue_observation(
            result.get("router_runtime", {}),
            e2e_seconds=float(result.get("pipeline_wall_sec", 0.0) or 0.0),
        )
        queue_observations.append(queue)
        if queue.get("status") != "observed":
            failures.append("router_command_queue_observation_missing")
        elif float(queue["e2e_percent"]) > 1.0:
            failures.append("router_command_queue_exceeds_one_percent")
    for baseline, router in ((baseline_a, router_ab), (baseline_b, router_ba)):
        try:
            binding = semantic_review.build_full_auto_comparison(baseline=baseline, candidate=router)
            semantic_bindings.append(binding)
            if not binding["mismatch_indices"] and int(binding["final_output_mismatch_count"]) != 0:
                failures.append("translation_exact_but_render_pixel_changed")
        except semantic_review.SemanticReviewError as exc:
            semantic_errors.append(str(exc))
    try:
        repeat_binding = semantic_review.build_full_auto_comparison(baseline=router_ab, candidate=router_ba)
        if repeat_binding["mismatch_indices"] or int(repeat_binding["final_output_mismatch_count"]) != 0:
            failures.append("router_internal_reproducibility_failed")
    except semantic_review.SemanticReviewError as exc:
        semantic_errors.append(str(exc))
    semantic = semantic_review.evaluate_semantic_review(
        protocol_version=PROTOCOL_VERSION,
        stage=f"{pair.key}-abba",
        comparisons=semantic_bindings,
        approval=semantic_approval,
    )
    if semantic.get("template"):
        _json_write(pair_dir / "semantic-review-template.json", semantic["template"])
    timing = [float(result.get("pipeline_wall_sec", 0.0) or 0.0) for result in results]
    if any(value <= 0 for value in timing):
        failures.append("invalid_e2e_timing")
        stats: dict[str, Any] = {}
    else:
        stats = _paired_stats((timing[0], timing[3]), (timing[1], timing[2]))
        if not _both_directions_faster(
            baseline_ab=timing[0],
            router_ab=timing[1],
            router_ba=timing[2],
            baseline_ba=timing[3],
        ):
            failures.append("direction_not_consistently_faster")
    if semantic_errors:
        failures.extend(f"semantic_contract:{item}" for item in semantic_errors)
    if semantic["status"] == "REJECT":
        failures.append("semantic_regression")
    status = "PASS"
    if failures:
        status = "REJECT"
    elif semantic["status"] == "REVIEW_REQUIRED":
        status = "REVIEW_REQUIRED"
    return {
        "protocol_version": PROTOCOL_VERSION,
        "pair": pair.key,
        "arm_order": list(ARMS),
        "results": list(results),
        "statistics": stats,
        "semantic_review": semantic,
        "command_queue": queue_observations,
        "failures": sorted(set(failures)),
        "status": status,
    }


def _load_pair_approval(
    approval_dir: Path | None,
    pair: Any,
) -> Mapping[str, Any] | None:
    if approval_dir is None:
        return None
    path = approval_dir / f"{pair.key}.json"
    if not path.is_file():
        return None
    return semantic_review.load_semantic_approval(path)


def _review_existing_pair(
    *,
    pair: Any,
    source_artifact_dir: Path,
    audit_pair_dir: Path,
    approval: Mapping[str, Any] | None,
) -> dict[str, Any]:
    source_summary = source_artifact_dir / pair.key / "pair-summary.json"
    payload = _json_read(source_summary)
    results = payload.get("results")
    if not isinstance(results, list) or len(results) != len(ARMS):
        raise RouterHandoffBenchmarkError(
            f"Existing pair summary is incomplete: {pair.key}."
        )
    audit_pair_dir.mkdir(parents=True, exist_ok=False)
    reviewed = _evaluate_pair(
        pair,
        results,
        audit_pair_dir,
        semantic_approval=approval,
    )
    reviewed["reviewed_source_pair_summary_sha256"] = _sha256_file(source_summary)
    return reviewed


def _pair_eligibility(pair: Any) -> tuple[bool, str]:
    for volume in (pair.ocr_volume, GEMMA_VOLUME):
        result = _run(["docker", "volume", "inspect", volume], check=False, timeout=20.0)
        if result.returncode != 0:
            return False, f"named_volume_missing:{volume}"
    if pair.key == "hunyuanocr":
        result = _run(
            ["docker", "volume", "inspect", pair.ocr_volume, "--format", "{{json .Labels}}"],
            check=False,
            timeout=20.0,
        )
        try:
            labels = json.loads(result.stdout) if result.returncode == 0 else {}
        except json.JSONDecodeError:
            labels = {}
        if not isinstance(labels, Mapping) or labels.get("com.comictranslate.hunyuanocr.router-ready") != "v1":
            return False, "hunyuanocr_canonical_volume_manifest_missing"
    return True, ""


def _router_preflight(pair: Any, directory: Path) -> dict[str, Any]:
    """Start the exact lab router without a loaded model, then clean it up."""

    from router_handoff_lab_runtime import LAB_CONTAINER_PREFIX, RouterLabSession

    directory.mkdir(parents=True, exist_ok=False)
    suffix = hashlib.sha256(str(directory).encode("utf-8")).hexdigest()[:16]
    session = RouterLabSession(
        pair=pair,
        artifact_dir=directory,
        container_name=f"{LAB_CONTAINER_PREFIX}{pair.key}-{suffix}",
    )
    try:
        prepared = session.prepare_process()
        evidence = session.evidence()
    finally:
        try:
            session.stop_process()
        finally:
            evidence = session.evidence()
    payload = {"prepared": prepared, "evidence": evidence}
    _json_write(directory / "router-preflight.json", payload)
    return payload


_PUBLIC_NOT_ELIGIBLE_REASONS = {
    "baseline_fixture_ocr_quality_gate_failed_before_translation": (
        "baseline OCR quality gate failed before translation"
    ),
    "hunyuanocr_canonical_volume_manifest_missing": (
        "canonical OCR model identity was unavailable"
    ),
}


def _public_not_eligible_reason(value: object) -> str:
    """Map private diagnostics to a bounded public classification."""

    reason = str(value or "")
    if reason in _PUBLIC_NOT_ELIGIBLE_REASONS:
        return _PUBLIC_NOT_ELIGIBLE_REASONS[reason]
    if reason.startswith("named_volume_missing:"):
        return "required runtime volume was unavailable"
    return "preflight or runtime contract was unavailable"


def _can_publish_report(
    *,
    mode: str,
    selected: Sequence[str],
    catalog: Mapping[str, Any],
) -> bool:
    return mode == "review" and set(selected) == set(catalog)


def _render_report(outcome: Mapping[str, Any]) -> str:
    lines = ["# Single llama.cpp Router 빠른 Lab", "", f"- protocol: `{PROTOCOL_VERSION}`", "- ABBA: `A → B → B → A` exactly once per eligible pair", "- status: results below are descriptive only; no statistical significance claim.", "", "| Pair | Result | E2E median (A / B) | Direction | Quality |", "|---|---|---:|---|---|"]
    for row in outcome.get("pairs", []):
        if not isinstance(row, Mapping):
            continue
        pair = str(row.get("pair", ""))
        status = str(row.get("status", ""))
        if status == "NOT_ELIGIBLE":
            lines.append(
                "| {pair} | NOT ELIGIBLE | - | - | {reason} |".format(
                    pair=pair,
                    reason=_public_not_eligible_reason(row.get("reason")),
                )
            )
            continue
        stats = row.get("statistics") if isinstance(row.get("statistics"), Mapping) else {}
        direction = "both faster" if not any("direction_not_consistently_faster" == item for item in row.get("failures", [])) else "not consistent"
        quality = str((row.get("semantic_review") or {}).get("status", "hard gate"))
        lines.append(
            "| {pair} | {status} | {a} / {b} | {direction} | {quality} |".format(
                pair=pair,
                status=status,
                a=stats.get("baseline_median_sec", "-"),
                b=stats.get("router_median_sec", "-"),
                direction=direction,
                quality=quality,
            )
        )
    lines.extend(["", "Private archive retains raw requests, text review inputs, images, commands, and resource samples.", "", "Product integration is intentionally blocked pending user approval."])
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="One-pass ABBA lab for the single llama.cpp router candidate.")
    parser.add_argument("--mode", choices=("preflight", "abba", "review"), default="abba")
    parser.add_argument("--pair", action="append", choices=("paddle-crop", "paddle-spotting", "hunyuanocr", "mangalmm"), help="Repeat to select pairs; default is every pair.")
    parser.add_argument("--fixture-manifest", type=Path, help="Private fixture manifest; never add it to Git.")
    parser.add_argument(
        "--semantic-approval-dir",
        type=Path,
        help="Private directory containing <pair>.json text-review approvals.",
    )
    parser.add_argument(
        "--review-artifact-dir",
        type=Path,
        help="Existing private ABBA artifacts directory; required for --mode review.",
    )
    parser.add_argument("--python", default=sys.executable, help="Windows benchmark Python executable.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Private explicit output override.")
    parser.add_argument(
        "--publish-report",
        action="store_true",
        help="Write the tracked aggregate report only from a full all-pair review.",
    )
    args = parser.parse_args(argv)
    if args.mode != "review" and args.fixture_manifest is None:
        parser.error("--fixture-manifest is required for preflight and abba.")
    if args.mode == "review" and args.review_artifact_dir is None:
        parser.error("--review-artifact-dir is required for review.")
    fixture_manifest = (
        _json_read(args.fixture_manifest.expanduser().resolve())
        if args.fixture_manifest is not None
        else {}
    )
    approval_dir = (
        args.semantic_approval_dir.expanduser().resolve()
        if args.semantic_approval_dir is not None
        else None
    )
    source_artifact_dir = (
        args.review_artifact_dir.expanduser().resolve()
        if args.review_artifact_dir is not None
        else None
    )
    catalog = _pair_catalog()
    selected = args.pair or list(catalog)
    if args.publish_report and not _can_publish_report(
        mode=args.mode,
        selected=selected,
        catalog=catalog,
    ):
        parser.error(
            "--publish-report requires --mode review with every router pair selected."
        )
    output_root, managed = artifact_harness.select_managed_output_directory(
        family=FAMILY,
        category=CATEGORY,
        explicit_output_directory=args.output_dir,
    )
    outcome: dict[str, Any] = {"protocol_version": PROTOCOL_VERSION, "mode": args.mode, "pairs": []}
    try:
        for key in selected:
            pair = catalog[key]
            if args.mode == "review":
                assert source_artifact_dir is not None
                reviewed = _review_existing_pair(
                    pair=pair,
                    source_artifact_dir=source_artifact_dir,
                    audit_pair_dir=output_root / key,
                    approval=_load_pair_approval(approval_dir, pair),
                )
                _json_write(output_root / key / "pair-summary-reviewed.json", reviewed)
                outcome["pairs"].append(reviewed)
                continue
            eligible, reason = _pair_eligibility(pair)
            if not eligible:
                outcome["pairs"].append({"pair": key, "status": "NOT_ELIGIBLE", "reason": reason})
                continue
            if args.mode == "preflight":
                try:
                    _router_preflight(pair, output_root / key / "preflight")
                except BaseException as exc:
                    outcome["pairs"].append(
                        {"pair": key, "status": "NOT_ELIGIBLE", "reason": str(exc)[:512]}
                    )
                else:
                    outcome["pairs"].append({"pair": key, "status": "ELIGIBLE"})
                continue
            fixture = _fixture_for_pair(fixture_manifest, pair)
            pair_dir = output_root / key
            pair_dir.mkdir(parents=True, exist_ok=False)
            results = [
                _execute_arm(
                    pair,
                    arm=arm,
                    arm_dir=pair_dir / f"arm-{index:02d}-{arm}",
                    fixture=fixture,
                    python_executable=args.python,
                )
                for index, arm in enumerate(ARMS, start=1)
            ]
            summary = _evaluate_pair(
                pair,
                results,
                pair_dir,
                semantic_approval=_load_pair_approval(approval_dir, pair),
            )
            _json_write(pair_dir / "pair-summary.json", summary)
            outcome["pairs"].append(summary)
        _json_write(output_root / "router-handoff-summary.json", outcome)
        if args.publish_report:
            report = (
                ROOT
                / "docs"
                / "benchmark"
                / "llamacpp-router-handoff"
                / "generated"
                / "latest-report-ko.md"
            )
            report.parent.mkdir(parents=True, exist_ok=True)
            report.write_text(_render_report(outcome), encoding="utf-8")
        if managed is not None:
            managed.complete(metadata={"pair_count": len(outcome["pairs"]), "mode": args.mode})
    except BaseException as exc:
        if managed is not None:
            managed.fail(exc, metadata={"partial_pair_count": len(outcome["pairs"])})
        raise
    print(
        json.dumps(
            {
                "pairs": [
                    {"pair": row.get("pair"), "status": row.get("status")}
                    for row in outcome["pairs"]
                ]
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

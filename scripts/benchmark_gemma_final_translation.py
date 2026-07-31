#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import os
import posixpath
import random
import re
import shutil
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import PIL
import requests
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modules.translation.llm import custom_local_gemma as gemma_runtime  # noqa: E402
from modules.translation.llm.custom_local_gemma import (  # noqa: E402
    DEFAULT_GEMMA_PROMPT_PROFILE,
    DEFAULT_GEMMA_RESPONSE_FORMAT_MODE,
    DEFAULT_GEMMA_RESPONSE_SCHEMA_MODE,
    DEFAULT_GEMMA_THINK_BRIEFLY_PROMPT,
    DEFAULT_GEMMA_TRANSLATION_MIN_P,
    DEFAULT_GEMMA_TRANSLATION_TEMPERATURE,
    DEFAULT_GEMMA_TRANSLATION_TOP_K,
    DEFAULT_GEMMA_TRANSLATION_TOP_P,
    GEMMA_REQUEST_MODE_CONTEXTUAL_SINGLE,
    CustomLocalGemmaTranslation,
)
from modules.utils.gpu_metrics import query_gpu_metrics  # noqa: E402
from modules.utils.textblock import TextBlock  # noqa: E402


EXPECTED_PAGE_COUNT = 22
EXPECTED_BLOCK_COUNT = 292
EXPECTED_MAX_COMPLETION_TOKENS = 512
DEFAULT_GROUP_SIZE = 7
DEFAULT_BASELINE_GROUP_SIZE = 6
BENCHMARK_PROTOCOL_VERSION = 3
HISTORICAL_GROUPED_REPLAY_COMMIT = (
    "76b81c7b903bd9569d116b5eabc966135a13a1f5"
)
GEMMA_REQUEST_MODE_CONTEXTUAL_GROUPED = "contextual-grouped"
DEFAULT_IMAGE_ID = (
    "sha256:22e0e3bfe967af4fd1df6a918022abbfd4e72e4d40a4769e616a4176790acbcb"
)
DEFAULT_EXPECTED_INPUT_MANIFEST_SHA256 = (
    "63cdfa53fc7c48efa9e6f1f11aae3e86bb5ea0aadcba361491ee34bc94cc9b1e"
)
DEFAULT_EXPECTED_OCR_SNAPSHOT_SHA256 = (
    "22fd706b63da75a5a4c7f4175cec6d23f9b9e9a831e82365329fca42e1c84605"
)
DEFAULT_MODEL_NAME = (
    "Gemma4-26B-A4B-Uncensored-HauhauCS-Balanced-IQ4_XS.gguf"
)
DEFAULT_MODEL_SIZE = 13_917_726_048
DEFAULT_MODEL_VOLUME = "comic-translate-gemma-models"
DEFAULT_HASH_HELPER_IMAGE_ID = (
    "sha256:14358309a308569c32bdc37e2e0e9694be33a9d99e68afb0f5ff33cc1f695dce"
)
DEFAULT_HASH_HELPER_IMAGE = (
    "alpine@sha256:14358309a308569c32bdc37e2e0e9694be33a9d99e68afb0f5ff33cc1f695dce"
)

CANDIDATE_BASELINE = "current-contextual-single"
CANDIDATE_GROUPED_F16 = "grouped-f16"
CANDIDATE_GROUPED_Q8 = "grouped-q8"
CANDIDATE_KEYS = (
    CANDIDATE_BASELINE,
    CANDIDATE_GROUPED_F16,
    CANDIDATE_GROUPED_Q8,
)
ROUND_ORDERS = {
    1: CANDIDATE_KEYS,
    2: (
        CANDIDATE_GROUPED_F16,
        CANDIDATE_GROUPED_Q8,
        CANDIDATE_BASELINE,
    ),
    3: (
        CANDIDATE_GROUPED_Q8,
        CANDIDATE_BASELINE,
        CANDIDATE_GROUPED_F16,
    ),
}
DEFAULT_RUNTIME_CONFIG_FINGERPRINTS = {
    CANDIDATE_BASELINE: (
        "628535e4d71e1c9015cd3e5c0275113c621039e176c79a7b3db901688dab38ae"
    ),
    CANDIDATE_GROUPED_F16: (
        "e6d83a374327b5e242dc0fa1b7cf59fdbd5a89f2830484829639f75089816128"
    ),
    CANDIDATE_GROUPED_Q8: (
        "eb765b6b40cb9bbfbf4af7bdb6f8ba767e4cea75c965b10c9d237fa8f31d9a27"
    ),
}

STRUCTURAL_STATS = (
    "gemma_truncated_count",
    "gemma_empty_content_count",
    "gemma_missing_key_count",
    "gemma_repetition_guard_count",
    "gemma_nested_value_count",
)


def require_historical_grouped_runtime() -> None:
    """Refuse a mislabeled replay after the product grouped path retired."""

    live_mode = getattr(
        gemma_runtime,
        "GEMMA_REQUEST_MODE_CONTEXTUAL_GROUPED",
        None,
    )
    if live_mode != GEMMA_REQUEST_MODE_CONTEXTUAL_GROUPED:
        raise RuntimeError(
            "Gemma final-translation protocol v3 requires the retired live "
            "contextual-grouped product path. Do not run it from the current "
            "checkout because grouped candidates would execute as "
            "contextual-single. Reproduce the historical suite only from "
            f"commit {HISTORICAL_GROUPED_REPLAY_COMMIT}; use protocol v4 "
            "report-only tooling for the preserved results."
        )
CLEAN_RUN_STATS = (
    *STRUCTURAL_STATS,
    "gemma_json_retry_count",
    "gemma_chunk_retry_events",
    "gemma_request_retry_count",
    "gemma_http_retry_count",
    "gemma_reasoning_without_final_count",
    "gemma_schema_validation_fail_count",
    "gemma_contextual_merge_fallback_count",
    "gemma_parser_error_count",
    "gemma_duplicate_key_count",
    "gemma_trailing_content_count",
    "gemma_top_level_type_error_count",
    "gemma_invalid_value_count",
    "gemma_unexpected_key_count",
    "gemma_strict_grouped_retry_count",
    "gemma_strict_single_retry_count",
    "gemma_partial_response_count",
    "gemma_partial_fallback_block_count",
    "gemma_split_count",
    "gemma_context_capacity_split_count",
)
IMAGE_SUFFIXES = {
    ".bmp",
    ".gif",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}
JPEG_ALIGNMENT_MAX_MAE = 1.25
JPEG_ALIGNMENT_MAX_P99 = 5.0
JPEG_ALIGNMENT_MAX_ABS = 12
JPEG_ALIGNMENT_MIN_PSNR_DB = 43.0


def write_json(path: Path, payload: Mapping[str, Any] | list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path.name}")
    return payload


def sha256_file(path: Path, *, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_bytes), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validated_sha256(value: str, *, option_name: str) -> str:
    normalized = str(value or "").strip().casefold()
    if not re.fullmatch(r"[0-9a-f]{64}", normalized):
        raise ValueError(
            f"{option_name} must be a 64-character hexadecimal digest"
        )
    return normalized


def validate_api_base_url(value: str) -> str:
    raw = str(value or "").strip()
    parsed = urllib.parse.urlsplit(raw)
    if (
        parsed.scheme != "http"
        or parsed.hostname != "127.0.0.1"
        or parsed.port != 18080
        or parsed.path.rstrip("/") != "/v1"
        or parsed.username is not None
        or parsed.password is not None
        or bool(parsed.query)
        or bool(parsed.fragment)
    ):
        raise ValueError(
            "--api-base-url must be exactly the loopback HTTP endpoint "
            "http://127.0.0.1:18080/v1"
        )
    return "http://127.0.0.1:18080/v1"


def build_translation_behavior_contract(
    *,
    group_size: int,
) -> dict[str, Any]:
    behavior_files = (
        Path(__file__).resolve(),
        ROOT / "modules" / "translation" / "llm" / "custom_local_gemma.py",
        ROOT / "modules" / "translation" / "llm" / "base.py",
        ROOT / "modules" / "translation" / "base.py",
        ROOT / "modules" / "translation" / "translation_memory.py",
        ROOT / "modules" / "utils" / "repetition_guard.py",
        ROOT / "modules" / "utils" / "gpu_metrics.py",
        ROOT / "modules" / "utils" / "text_normalization.py",
        ROOT / "modules" / "utils" / "textblock.py",
        ROOT / "modules" / "utils" / "translator_utils.py",
    )
    source_contract = {
        path.relative_to(ROOT).as_posix(): sha256_file(path)
        for path in behavior_files
    }
    contract = {
        "benchmark_protocol_version": BENCHMARK_PROTOCOL_VERSION,
        "source_sha256": source_contract,
        "engine": {
            "source_language": "Japanese",
            "target_language": "Korean",
            "prompt_profile": DEFAULT_GEMMA_PROMPT_PROFILE,
            "response_format_mode": DEFAULT_GEMMA_RESPONSE_FORMAT_MODE,
            "response_schema_mode": DEFAULT_GEMMA_RESPONSE_SCHEMA_MODE,
            "think_briefly_prompt": DEFAULT_GEMMA_THINK_BRIEFLY_PROMPT,
            "temperature": DEFAULT_GEMMA_TRANSLATION_TEMPERATURE,
            "top_k": DEFAULT_GEMMA_TRANSLATION_TOP_K,
            "top_p": DEFAULT_GEMMA_TRANSLATION_TOP_P,
            "min_p": DEFAULT_GEMMA_TRANSLATION_MIN_P,
            "max_completion_tokens": EXPECTED_MAX_COMPLETION_TOKENS,
            "contextual_merge_input": True,
            "persistent_cache_enabled": False,
            "exact_tm_enabled": False,
        },
        "candidate_request_contract": {
            CANDIDATE_BASELINE: {
                "request_mode": GEMMA_REQUEST_MODE_CONTEXTUAL_SINGLE,
                "group_size": DEFAULT_BASELINE_GROUP_SIZE,
            },
            CANDIDATE_GROUPED_F16: {
                "request_mode": GEMMA_REQUEST_MODE_CONTEXTUAL_GROUPED,
                "group_size": int(group_size),
            },
            CANDIDATE_GROUPED_Q8: {
                "request_mode": GEMMA_REQUEST_MODE_CONTEXTUAL_GROUPED,
                "group_size": int(group_size),
            },
        },
        "candidate_orders": {
            str(round_index): list(order)
            for round_index, order in ROUND_ORDERS.items()
        },
        "environment": {
            "python": sys.version,
            "numpy": np.__version__,
            "pillow": PIL.__version__,
            "requests": requests.__version__,
        },
    }
    return {
        **contract,
        "contract_sha256": canonical_sha256(contract),
    }


def neutral_basename(value: str) -> str:
    parts = re.split(r"[\\/]", str(value or ""))
    return parts[-1] if parts else ""


def _image_files(input_root: Path) -> list[Path]:
    return sorted(
        (
            path
            for path in input_root.iterdir()
            if path.is_file() and path.suffix.casefold() in IMAGE_SUFFIXES
        ),
        key=lambda path: (path.name.casefold(), path.name),
    )


def decoded_image_contract(path: Path) -> dict[str, Any]:
    with Image.open(path) as image:
        decoded = np.asarray(image.convert("RGB"), dtype=np.uint8)
    return {
        "width": int(decoded.shape[1]),
        "height": int(decoded.shape[0]),
        "rgb_sha256": hashlib.sha256(decoded.tobytes()).hexdigest(),
    }


def build_input_manifest(input_root: Path) -> dict[str, Any]:
    files = _image_files(input_root)
    if len(files) != EXPECTED_PAGE_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_PAGE_COUNT} input images, found {len(files)}"
        )
    records = []
    for path in files:
        records.append(
            {
                "name": path.name,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "decoded_rgb": decoded_image_contract(path),
            }
        )
    return {
        "file_count": len(records),
        "files": records,
        "manifest_sha256": canonical_sha256(records),
    }


def _snapshot_page_name(page: Mapping[str, Any]) -> str:
    return str(
        page.get("image_name")
        or neutral_basename(str(page.get("image_path") or ""))
        or page.get("image_stem")
        or ""
    ).strip()


def _snapshot_image_path(
    page: Mapping[str, Any],
    *,
    snapshot_path: Path,
    page_name: str,
) -> Path:
    raw_path = str(page.get("image_path") or "").strip()
    if raw_path:
        candidate = Path(raw_path)
        if candidate.is_file():
            return candidate
    sibling = snapshot_path.parent / "corpus" / page_name
    if sibling.is_file():
        return sibling
    raise FileNotFoundError(
        f"Snapshot source image is unavailable for page {page_name}"
    )


def compare_snapshot_image(
    *,
    input_path: Path,
    snapshot_path: Path,
    input_record: Mapping[str, Any],
) -> dict[str, Any]:
    snapshot_size = snapshot_path.stat().st_size
    snapshot_sha256 = sha256_file(snapshot_path)
    raw_equal = (
        snapshot_size == int(input_record["size_bytes"])
        and snapshot_sha256 == str(input_record["sha256"])
    )
    if raw_equal:
        decoded = input_record.get("decoded_rgb") or {}
        return {
            "passed": True,
            "mode": "raw-sha256",
            "input_sha256": str(input_record["sha256"]),
            "snapshot_copy_sha256": snapshot_sha256,
            "width": int(decoded.get("width") or 0),
            "height": int(decoded.get("height") or 0),
            "pixel_equal": True,
            "mean_absolute_error": 0.0,
            "p99_absolute_error": 0.0,
            "max_absolute_error": 0,
            "psnr_db": None,
        }

    with Image.open(input_path) as input_image:
        input_pixels = np.asarray(input_image.convert("RGB"), dtype=np.int16)
    with Image.open(snapshot_path) as snapshot_image:
        snapshot_pixels = np.asarray(snapshot_image.convert("RGB"), dtype=np.int16)
    if input_pixels.shape != snapshot_pixels.shape:
        return {
            "passed": False,
            "mode": "dimension-mismatch",
            "input_sha256": str(input_record["sha256"]),
            "snapshot_copy_sha256": snapshot_sha256,
            "input_shape": list(input_pixels.shape),
            "snapshot_shape": list(snapshot_pixels.shape),
        }

    difference = np.abs(input_pixels - snapshot_pixels)
    pixel_equal = not bool(np.any(difference))
    mean_absolute_error = float(np.mean(difference))
    p99_absolute_error = float(np.percentile(difference, 99))
    max_absolute_error = int(np.max(difference))
    mse = float(
        np.mean((input_pixels - snapshot_pixels).astype(np.float64) ** 2)
    )
    psnr_db = 99.0 if mse == 0 else float(
        20.0 * np.log10(255.0 / np.sqrt(mse))
    )
    jpeg_pair = (
        input_path.suffix.casefold() in {".jpg", ".jpeg"}
        and snapshot_path.suffix.casefold() in {".jpg", ".jpeg"}
    )
    passed = pixel_equal or bool(
        jpeg_pair
        and mean_absolute_error <= JPEG_ALIGNMENT_MAX_MAE
        and p99_absolute_error <= JPEG_ALIGNMENT_MAX_P99
        and max_absolute_error <= JPEG_ALIGNMENT_MAX_ABS
        and psnr_db >= JPEG_ALIGNMENT_MIN_PSNR_DB
    )
    return {
        "passed": passed,
        "mode": (
            "decoded-pixel-sha256"
            if pixel_equal
            else "bounded-jpeg-reencode"
            if passed
            else "pixel-content-mismatch"
        ),
        "input_sha256": str(input_record["sha256"]),
        "snapshot_copy_sha256": snapshot_sha256,
        "width": int(input_pixels.shape[1]),
        "height": int(input_pixels.shape[0]),
        "pixel_equal": pixel_equal,
        "mean_absolute_error": round(mean_absolute_error, 6),
        "p99_absolute_error": round(p99_absolute_error, 6),
        "max_absolute_error": max_absolute_error,
        "psnr_db": round(psnr_db, 6),
    }


def load_frozen_corpus(
    *,
    input_root: Path,
    snapshot_path: Path,
    input_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    payload = read_json(snapshot_path)
    pages_raw = payload.get("pages")
    if not isinstance(pages_raw, list):
        raise ValueError("OCR snapshot must contain a pages list")
    if len(pages_raw) != EXPECTED_PAGE_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_PAGE_COUNT} snapshot pages, found {len(pages_raw)}"
        )

    input_files = list(input_manifest.get("files") or [])
    input_by_name = {
        str(item["name"]): item
        for item in input_files
        if isinstance(item, Mapping) and item.get("name")
    }
    input_order = [str(item["name"]) for item in input_files]
    pages: list[dict[str, Any]] = []
    source_languages: set[str] = set()
    target_languages: set[str] = set()
    total_blocks = 0

    for page_index, page_raw in enumerate(pages_raw, start=1):
        if not isinstance(page_raw, Mapping):
            raise ValueError(f"Snapshot page {page_index} is not an object")
        page_name = _snapshot_page_name(page_raw)
        if not page_name:
            raise ValueError(f"Snapshot page {page_index} has no neutral image name")
        blocks_raw = page_raw.get("blocks")
        if not isinstance(blocks_raw, list) or not blocks_raw:
            raise ValueError(f"Snapshot page {page_index} has no blocks")

        source_language = str(page_raw.get("source_lang") or "").strip()
        target_language = str(page_raw.get("target_lang") or "").strip()
        source_languages.add(source_language)
        target_languages.add(target_language)

        snapshot_image = _snapshot_image_path(
            page_raw,
            snapshot_path=snapshot_path,
            page_name=page_name,
        )
        input_record = input_by_name.get(page_name)
        if input_record is None:
            raise ValueError(
                f"Snapshot page is absent from the frozen input manifest: {page_name}"
            )
        input_image = input_root / page_name
        if (
            input_image.stat().st_size != int(input_record["size_bytes"])
            or sha256_file(input_image) != str(input_record["sha256"])
        ):
            raise ValueError(
                f"Input image changed after manifest creation: {page_name}"
            )
        alignment = compare_snapshot_image(
            input_path=input_image,
            snapshot_path=snapshot_image,
            input_record=input_record,
        )
        if not alignment["passed"]:
            raise ValueError(
                f"Snapshot source image does not match the input manifest: {page_name}"
            )

        blocks: list[dict[str, Any]] = []
        for block_index, block_raw in enumerate(blocks_raw, start=1):
            if not isinstance(block_raw, Mapping):
                raise ValueError(
                    f"Snapshot page {page_index} block {block_index} is not an object"
                )
            source_text = str(block_raw.get("text") or "")
            if not source_text.strip():
                raise ValueError(
                    f"Snapshot page {page_index} block {block_index} is empty"
                )
            xyxy_raw = block_raw.get("xyxy")
            xyxy = (
                list(xyxy_raw)
                if isinstance(xyxy_raw, (list, tuple)) and len(xyxy_raw) == 4
                else [0, block_index * 10, 100, block_index * 10 + 8]
            )
            blocks.append(
                {
                    "row_id": f"p{page_index:03d}-b{block_index:03d}",
                    "block_index": block_index,
                    "source": source_text,
                    "source_sha256": hashlib.sha256(
                        source_text.encode("utf-8", errors="surrogatepass")
                    ).hexdigest(),
                    "xyxy": xyxy,
                }
            )
        total_blocks += len(blocks)
        pages.append(
            {
                "page_index": page_index,
                "page_name": page_name,
                "source_language": source_language,
                "target_language": target_language,
                "block_count": len(blocks),
                "source_alignment": alignment,
                "blocks": blocks,
            }
        )

    page_order = [str(page["page_name"]) for page in pages]
    if page_order != input_order:
        raise ValueError(
            "Snapshot page order does not match the deterministic input order"
        )
    if total_blocks != EXPECTED_BLOCK_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_BLOCK_COUNT} snapshot blocks, found {total_blocks}"
        )
    if source_languages != {"Japanese"} or target_languages != {"Korean"}:
        raise ValueError(
            "Final corpus must be Japanese-to-Korean only; "
            f"source={sorted(source_languages)}, target={sorted(target_languages)}"
        )

    page_contract = [
        {
            "page_index": page["page_index"],
            "page_name": page["page_name"],
            "block_count": page["block_count"],
            "source_alignment": page["source_alignment"],
            "ordered_source_sha256": [
                block["source_sha256"] for block in page["blocks"]
            ],
        }
        for page in pages
    ]
    return {
        "snapshot_sha256": sha256_file(snapshot_path),
        "page_count": len(pages),
        "block_count": total_blocks,
        "source_language": "Japanese",
        "target_language": "Korean",
        "page_contract": page_contract,
        "contract_sha256": canonical_sha256(page_contract),
        "pages": pages,
    }


def validate_frozen_asset_digests(
    *,
    input_manifest: Mapping[str, Any],
    corpus: Mapping[str, Any],
    expected_input_manifest_sha256: str,
    expected_ocr_snapshot_sha256: str,
) -> None:
    expected_manifest = validated_sha256(
        expected_input_manifest_sha256,
        option_name="--expected-input-manifest-sha256",
    )
    expected_snapshot = validated_sha256(
        expected_ocr_snapshot_sha256,
        option_name="--expected-ocr-snapshot-sha256",
    )
    actual_manifest = str(input_manifest.get("manifest_sha256") or "").casefold()
    actual_snapshot = str(corpus.get("snapshot_sha256") or "").casefold()
    errors: list[str] = []
    if actual_manifest != expected_manifest:
        errors.append(
            "input manifest SHA-256 differs "
            f"(expected={expected_manifest}, actual={actual_manifest})"
        )
    if actual_snapshot != expected_snapshot:
        errors.append(
            "OCR snapshot SHA-256 differs "
            f"(expected={expected_snapshot}, actual={actual_snapshot})"
        )
    if errors:
        raise ValueError("Frozen final corpus contract failed: " + "; ".join(errors))


def expected_grouped_request_count(
    pages: Iterable[Mapping[str, Any]],
    group_size: int,
) -> int:
    return sum(
        (int(page["block_count"]) + group_size - 1) // group_size
        for page in pages
    )


def candidate_order(round_index: int) -> tuple[str, ...]:
    try:
        return ROUND_ORDERS[int(round_index)]
    except KeyError as exc:
        raise ValueError(f"Unsupported benchmark round: {round_index}") from exc


def _run_process(
    command: list[str],
    *,
    timeout_sec: float = 30.0,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=max(0.1, float(timeout_sec)),
        check=False,
    )
    if check and completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "").strip()
        raise RuntimeError(
            f"Command failed with exit {completed.returncode}: "
            f"{command[0]} {detail}"
        )
    return completed


def resolve_docker_executable(requested: str = "") -> str:
    if requested:
        resolved = shutil.which(requested)
        if resolved:
            return resolved
        raise FileNotFoundError(f"Docker executable not found: {requested}")
    for candidate in ("docker.exe", "docker"):
        resolved = shutil.which(candidate)
        if resolved:
            return resolved
    raise FileNotFoundError("docker.exe/docker is unavailable")


def docker_json(
    docker_executable: str,
    arguments: list[str],
    *,
    timeout_sec: float = 30.0,
) -> Any:
    completed = _run_process(
        [docker_executable, *arguments],
        timeout_sec=timeout_sec,
    )
    return json.loads(completed.stdout)


def expected_candidate_command(
    *,
    candidate: str,
    model_name: str,
) -> list[str]:
    kv = "q8_0" if candidate == CANDIDATE_GROUPED_Q8 else "f16"
    command = [
        "-m",
        f"/models/{model_name}",
        "--host",
        "0.0.0.0",
        "--port",
        "8080",
        "-c",
        "4096",
        "-np",
        "1",
        "-t",
        "10",
        "--n-gpu-layers",
        "23",
        "--fit",
        "off",
        "-fa",
        "on",
        "-ctk",
        kv,
        "-ctv",
        kv,
        "--kv-offload",
        "--swa-full",
        "--jinja",
        "--reasoning",
        "off",
        "--reasoning-budget",
        "0",
        "--reasoning-format",
        "none",
        "--metrics",
        "--perf",
        "--cache-ram",
        "0",
    ]
    if candidate == CANDIDATE_BASELINE:
        command.extend(
            [
                "--spec-type",
                "ngram-mod",
                "--spec-draft-n-max",
                "8",
            ]
        )
    return command


def inspect_candidate_runtime(
    *,
    docker_executable: str,
    container_name: str,
    candidate: str,
    expected_image_id: str,
    expected_model_name: str,
    expected_model_volume: str,
    expected_config_fingerprint: str,
) -> dict[str, Any]:
    payload = docker_json(
        docker_executable,
        ["inspect", container_name],
        timeout_sec=30,
    )
    if not isinstance(payload, list) or len(payload) != 1:
        raise ValueError(f"Unable to inspect candidate container: {container_name}")
    item = payload[0]
    config = item.get("Config") or {}
    state = item.get("State") or {}
    host_config = item.get("HostConfig") or {}
    network = item.get("NetworkSettings") or {}
    command = [str(value) for value in config.get("Cmd") or []]
    labels = config.get("Labels") or {}
    mounts = item.get("Mounts") or []

    errors: list[str] = []
    if str(item.get("Image") or "") != expected_image_id:
        errors.append("image ID differs from the pinned llama.cpp image")
    if str(labels.get("comic-translate.runtime") or "") != "gemma-probe":
        errors.append("runtime label is not gemma-probe")
    expected_command = expected_candidate_command(
        candidate=candidate,
        model_name=expected_model_name,
    )
    if command != expected_command:
        errors.append("full llama.cpp command differs from the pinned candidate")
    actual_fingerprint = str(
        labels.get("comic-translate.config-fingerprint") or ""
    ).casefold()
    if actual_fingerprint != expected_config_fingerprint.casefold():
        errors.append(
            "runtime config fingerprint differs "
            f"(actual={actual_fingerprint!r}, "
            f"expected={expected_config_fingerprint.casefold()!r})"
        )

    model_tree_mounts = [
        mount
        for mount in mounts
        if (
            posixpath.normpath(
                str(mount.get("Destination") or "")
            ) == "/models"
            or posixpath.normpath(
                str(mount.get("Destination") or "")
            ).startswith("/models/")
        )
    ]
    if len(model_tree_mounts) != 1:
        errors.append(
            "exactly one mount at /models and none below /models are required"
        )
    else:
        mount = model_tree_mounts[0]
        if (
            posixpath.normpath(
                str(mount.get("Destination") or "")
            )
            != "/models"
        ):
            errors.append("the single model mount must target exactly /models")
        elif str(mount.get("Type") or "") != "volume":
            errors.append("/models must be a Docker volume")
        elif str(mount.get("Name") or "") != expected_model_volume:
            errors.append("/models volume differs")
        elif bool(mount.get("RW")):
            errors.append("/models volume must be read-only")

    port_bindings = host_config.get("PortBindings") or {}
    bindings = port_bindings.get("8080/tcp") or []
    normalized_bindings = [
        {
            "host_ip": str(binding.get("HostIp") or ""),
            "host_port": str(binding.get("HostPort") or ""),
        }
        for binding in bindings
        if isinstance(binding, Mapping)
    ]
    if normalized_bindings != [
        {"host_ip": "127.0.0.1", "host_port": "18080"}
    ]:
        errors.append(
            "container must publish 8080 only as 127.0.0.1:18080"
        )
    if set(port_bindings) != {"8080/tcp"}:
        errors.append("container must not publish any port except 8080/tcp")
    if str(host_config.get("NetworkMode") or "") != "bridge":
        errors.append("container network mode must be bridge")
    if bool(host_config.get("PublishAllPorts")):
        errors.append("container must not publish all exposed ports")
    if bool(host_config.get("Privileged")):
        errors.append("container must not be privileged")
    if bool(host_config.get("AutoRemove")):
        errors.append("container must not auto-remove")
    restart_policy = host_config.get("RestartPolicy") or {}
    if str(restart_policy.get("Name") or "") != "no":
        errors.append("container restart policy must be no")
    if not bool((host_config.get("DeviceRequests") or [])):
        errors.append("GPU device request is absent")
    if errors:
        raise ValueError(
            f"Candidate runtime contract failed for {candidate}: "
            + "; ".join(errors)
        )

    return {
        "candidate": candidate,
        "container_name": container_name,
        "container_id": str(item.get("Id") or ""),
        "created_at": str(item.get("Created") or ""),
        "state": str(state.get("Status") or ""),
        "image_id": str(item.get("Image") or ""),
        "image_version": str(labels.get("org.opencontainers.image.version") or ""),
        "image_revision": str(labels.get("org.opencontainers.image.revision") or ""),
        "profile_label": str(labels.get("comic-translate.profile") or ""),
        "config_fingerprint": actual_fingerprint,
        "command": command,
        "model_volume": expected_model_volume,
        "model_mount_read_only": True,
        "network_ports": network.get("Ports") or {},
    }


def inspect_and_stop_candidate_runtimes(
    *,
    docker_executable: str,
    containers: Mapping[str, str],
    expected_image_id: str,
    expected_model_name: str,
    expected_model_volume: str,
    expected_runtime_fingerprints: Mapping[str, str],
) -> list[dict[str, Any]]:
    """Validate every runtime read-only before stopping any pinned container."""

    runtime_contracts = [
        inspect_candidate_runtime(
            docker_executable=docker_executable,
            container_name=containers[candidate],
            candidate=candidate,
            expected_image_id=expected_image_id,
            expected_model_name=expected_model_name,
            expected_model_volume=expected_model_volume,
            expected_config_fingerprint=expected_runtime_fingerprints[
                candidate
            ],
        )
        for candidate in CANDIDATE_KEYS
    ]
    runtime_by_candidate = {
        str(contract["candidate"]): contract
        for contract in runtime_contracts
    }
    for candidate in CANDIDATE_KEYS:
        contract = runtime_by_candidate[candidate]
        stop_container(
            docker_executable,
            str(contract["container_name"]),
            expected_container_id=str(contract["container_id"]),
        )
    unexpected = running_port_18080_containers(docker_executable)
    if unexpected:
        raise RuntimeError(
            "Port 18080 remains occupied after pinned candidate cleanup: "
            + ", ".join(unexpected)
        )
    return runtime_contracts


def inspect_model_volume(
    *,
    docker_executable: str,
    volume_name: str,
) -> dict[str, Any]:
    payload = docker_json(
        docker_executable,
        ["volume", "inspect", volume_name],
        timeout_sec=30,
    )
    if not isinstance(payload, list) or len(payload) != 1:
        raise ValueError(f"Unable to inspect model volume: {volume_name}")
    item = payload[0]
    return {
        "name": str(item.get("Name") or ""),
        "driver": str(item.get("Driver") or ""),
        "created_at": str(item.get("CreatedAt") or ""),
        "labels": item.get("Labels") or {},
    }


def inspect_helper_image(
    *,
    docker_executable: str,
    helper_image: str,
    expected_image_id: str,
) -> dict[str, Any]:
    payload = docker_json(
        docker_executable,
        ["image", "inspect", helper_image],
        timeout_sec=30,
    )
    if not isinstance(payload, list) or len(payload) != 1:
        raise ValueError(f"Unable to inspect hash helper image: {helper_image}")
    image_id = str(payload[0].get("Id") or "")
    if image_id != expected_image_id:
        raise ValueError(
            "Hash helper image ID differs "
            f"(expected={expected_image_id}, actual={image_id})"
        )
    return {
        "reference": helper_image,
        "image_id": image_id,
        "repo_digests": list(payload[0].get("RepoDigests") or []),
    }


def volume_model_size(
    *,
    docker_executable: str,
    helper_image: str,
    volume_name: str,
    model_name: str,
) -> int:
    completed = _run_process(
        [
            docker_executable,
            "run",
            "--rm",
            "--mount",
            f"type=volume,source={volume_name},target=/models,readonly",
            "--entrypoint",
            "stat",
            helper_image,
            "-c",
            "%s",
            f"/models/{model_name}",
        ],
        timeout_sec=60,
    )
    return int(completed.stdout.strip())


def volume_model_sha256(
    *,
    docker_executable: str,
    helper_image: str,
    volume_name: str,
    model_name: str,
) -> str:
    completed = _run_process(
        [
            docker_executable,
            "run",
            "--rm",
            "--mount",
            f"type=volume,source={volume_name},target=/models,readonly",
            "--entrypoint",
            "sha256sum",
            helper_image,
            f"/models/{model_name}",
        ],
        timeout_sec=1800,
    )
    digest = completed.stdout.strip().split()[0].casefold()
    if not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError("Volume model SHA-256 output is invalid")
    return digest


def prepare_model_contract(
    *,
    docker_executable: str,
    helper_image: str,
    expected_helper_image_id: str,
    model_source: Path,
    model_name: str,
    model_volume: str,
    expected_size: int,
    expected_sha256: str,
) -> dict[str, Any]:
    if model_source.name != model_name:
        raise ValueError("Host model filename differs from --model-name")
    source_stat = model_source.stat()
    if source_stat.st_size != expected_size:
        raise ValueError(
            f"Host model size is {source_stat.st_size}, expected {expected_size}"
        )
    helper_contract = inspect_helper_image(
        docker_executable=docker_executable,
        helper_image=helper_image,
        expected_image_id=expected_helper_image_id,
    )
    volume = inspect_model_volume(
        docker_executable=docker_executable,
        volume_name=model_volume,
    )
    copied_size = volume_model_size(
        docker_executable=docker_executable,
        helper_image=helper_image,
        volume_name=model_volume,
        model_name=model_name,
    )
    if copied_size != expected_size:
        raise ValueError(
            f"Volume model size is {copied_size}, expected {expected_size}"
        )

    expected_sha256 = expected_sha256.casefold()
    print("[preflight] hashing the host IQ4_XS model", flush=True)
    source_sha256 = sha256_file(model_source)
    print("[preflight] hashing the read-only Docker volume copy", flush=True)
    copied_sha256 = volume_model_sha256(
        docker_executable=docker_executable,
        helper_image=helper_image,
        volume_name=model_volume,
        model_name=model_name,
    )
    if source_sha256 != expected_sha256 or copied_sha256 != expected_sha256:
        raise ValueError(
            "Model SHA-256 contract failed: "
            f"expected={expected_sha256}, host={source_sha256}, volume={copied_sha256}"
        )

    return {
        "model_name": model_name,
        "expected_size_bytes": expected_size,
        "expected_sha256": expected_sha256,
        "source_size_bytes": source_stat.st_size,
        "source_mtime_ns": source_stat.st_mtime_ns,
        "source_sha256": source_sha256,
        "volume_name": model_volume,
        "volume_created_at": volume["created_at"],
        "volume_size_bytes": copied_size,
        "volume_sha256": copied_sha256,
        "model_copy_verified": True,
        "full_hash_reused": False,
        "hash_helper_image": helper_contract,
    }


def _api_json(url: str, *, timeout_sec: float = 10.0) -> dict[str, Any]:
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout_sec) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected an object from {url}")
    return payload


def wait_for_runtime(
    *,
    api_base_url: str,
    model_name: str,
    timeout_sec: float,
) -> str:
    base = api_base_url.rstrip("/")
    root = re.sub(r"/v1$", "", base)
    deadline = time.monotonic() + timeout_sec
    last_error = ""
    while time.monotonic() < deadline:
        try:
            health = _api_json(f"{root}/health", timeout_sec=5)
            models = _api_json(f"{base}/models", timeout_sec=10)
            loaded = [
                str(item.get("id") or "")
                for item in models.get("data") or []
                if isinstance(item, Mapping) and item.get("id")
            ]
            matching = [
                model
                for model in loaded
                if neutral_basename(model) == model_name
            ]
            if str(health.get("status") or "") == "ok" and len(matching) == 1:
                return matching[0]
            last_error = (
                f"health={health.get('status')!r}, loaded_models={len(loaded)}"
            )
        except (
            OSError,
            ValueError,
            json.JSONDecodeError,
            urllib.error.URLError,
        ) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
        time.sleep(2)
    raise TimeoutError(
        f"Gemma runtime was not ready within {timeout_sec:.0f}s: {last_error}"
    )


def running_port_18080_containers(docker_executable: str) -> list[str]:
    completed = _run_process(
        [
            docker_executable,
            "ps",
            "--filter",
            "publish=18080",
            "--format",
            "{{.Names}}",
        ],
        timeout_sec=30,
    )
    return [
        line.strip()
        for line in completed.stdout.splitlines()
        if line.strip()
    ]


class ExclusiveSuiteLock:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._stream: Any = None

    def __enter__(self) -> ExclusiveSuiteLock:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._stream = self.path.open("a+b")
        self._stream.seek(0, os.SEEK_END)
        if self._stream.tell() == 0:
            self._stream.write(b"\0")
            self._stream.flush()
        self._stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(self._stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(
                    self._stream.fileno(),
                    fcntl.LOCK_EX | fcntl.LOCK_NB,
                )
        except OSError as exc:
            self._stream.close()
            self._stream = None
            raise RuntimeError(
                "Another Gemma final translation suite owns the shared "
                "candidate containers"
            ) from exc
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if self._stream is None:
            return
        try:
            self._stream.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(self._stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(self._stream.fileno(), fcntl.LOCK_UN)
        finally:
            self._stream.close()
            self._stream = None


def inspect_container_id(
    *,
    docker_executable: str,
    container_reference: str,
    expected_container_id: str,
) -> dict[str, Any]:
    payload = docker_json(
        docker_executable,
        ["inspect", container_reference],
        timeout_sec=30,
    )
    if not isinstance(payload, list) or len(payload) != 1:
        raise ValueError(
            f"Unable to inspect candidate container: {container_reference}"
        )
    actual_id = str(payload[0].get("Id") or "")
    if actual_id != expected_container_id:
        raise RuntimeError(
            "Candidate container identity changed "
            f"(expected={expected_container_id}, actual={actual_id})"
        )
    return payload[0]


def stop_container(
    docker_executable: str,
    container_name: str,
    *,
    expected_container_id: str | None = None,
) -> None:
    container_reference = expected_container_id or container_name
    if expected_container_id:
        item = inspect_container_id(
            docker_executable=docker_executable,
            container_reference=container_reference,
            expected_container_id=expected_container_id,
        )
    else:
        inspect = docker_json(
            docker_executable,
            ["inspect", container_reference],
            timeout_sec=30,
        )
        item = inspect[0] if inspect else {}
    state = item.get("State") or {}
    if bool(state.get("Running")):
        _run_process(
            [docker_executable, "stop", container_reference],
            timeout_sec=120,
        )
    if expected_container_id:
        verified_item = inspect_container_id(
            docker_executable=docker_executable,
            container_reference=container_reference,
            expected_container_id=expected_container_id,
        )
    else:
        verified = docker_json(
            docker_executable,
            ["inspect", container_reference],
            timeout_sec=30,
        )
        verified_item = verified[0] if verified else {}
    verified_state = verified_item.get("State") or {}
    if bool(verified_state.get("Running")):
        raise RuntimeError(f"Container did not stop: {container_name}")


def start_container(
    *,
    docker_executable: str,
    container_name: str,
    expected_container_id: str,
) -> None:
    inspect_container_id(
        docker_executable=docker_executable,
        container_reference=expected_container_id,
        expected_container_id=expected_container_id,
    )
    running = running_port_18080_containers(docker_executable)
    if running:
        raise RuntimeError(
            "Host port 18080 is already in use: " + ", ".join(running)
        )
    _run_process(
        [docker_executable, "start", expected_container_id],
        timeout_sec=120,
    )
    inspect_container_id(
        docker_executable=docker_executable,
        container_reference=expected_container_id,
        expected_container_id=expected_container_id,
    )


def _parse_size_bytes(value: str) -> int | None:
    match = re.match(
        r"\s*([0-9]+(?:\.[0-9]+)?)\s*([kmgt]?i?b)\b",
        str(value or ""),
        re.IGNORECASE,
    )
    if not match:
        return None
    number = float(match.group(1))
    unit = match.group(2).casefold()
    factors = {
        "b": 1,
        "kb": 1000,
        "kib": 1024,
        "mb": 1000**2,
        "mib": 1024**2,
        "gb": 1000**3,
        "gib": 1024**3,
        "tb": 1000**4,
        "tib": 1024**4,
    }
    return int(number * factors[unit])


def query_docker_stats(
    docker_executable: str,
    container_name: str,
) -> dict[str, Any]:
    try:
        completed = _run_process(
            [
                docker_executable,
                "stats",
                "--no-stream",
                "--format",
                "{{json .}}",
                container_name,
            ],
            timeout_sec=10,
        )
        payload = json.loads(completed.stdout.strip())
        memory_usage = str(payload.get("MemUsage") or "").split("/", 1)[0].strip()
        return {
            "available": True,
            "memory_usage_bytes": _parse_size_bytes(memory_usage),
            "cpu_percent": str(payload.get("CPUPerc") or ""),
            "pids": payload.get("PIDs"),
        }
    except Exception as exc:
        return {
            "available": False,
            "reason": f"{type(exc).__name__}: {exc}",
        }


def query_process_rss_bytes() -> int | None:
    try:
        import psutil  # type: ignore

        return int(psutil.Process(os.getpid()).memory_info().rss)
    except Exception:
        return None


def query_wsl_memory() -> dict[str, Any]:
    executable = shutil.which("wsl.exe")
    if not executable:
        return {"available": False, "reason": "wsl.exe unavailable"}
    script = (
        "awk '/^(MemTotal|MemAvailable|SwapTotal|SwapFree):/ "
        "{print $1 $2}' /proc/meminfo"
    )
    try:
        completed = _run_process(
            [executable, "-e", "sh", "-lc", script],
            timeout_sec=15,
        )
        values: dict[str, int] = {}
        for line in completed.stdout.splitlines():
            key, separator, raw = line.partition(":")
            if not separator:
                continue
            values[key] = int(raw.strip()) * 1024
        required = {"MemTotal", "MemAvailable", "SwapTotal", "SwapFree"}
        if not required.issubset(values):
            raise ValueError("incomplete /proc/meminfo sample")
        return {
            "available": True,
            "memory_total_bytes": values["MemTotal"],
            "memory_used_bytes": values["MemTotal"] - values["MemAvailable"],
            "swap_total_bytes": values["SwapTotal"],
            "swap_used_bytes": values["SwapTotal"] - values["SwapFree"],
        }
    except Exception as exc:
        return {
            "available": False,
            "reason": f"{type(exc).__name__}: {exc}",
        }


def query_windows_gpu_adapter_memory() -> dict[str, Any]:
    executable = shutil.which("powershell.exe")
    if not executable:
        return {
            "available": False,
            "reason": "powershell.exe unavailable",
        }
    command = (
        "$ErrorActionPreference='Stop';"
        "$c=Get-Counter -Counter "
        "'\\GPU Adapter Memory(*)\\Shared Usage',"
        "'\\GPU Adapter Memory(*)\\Dedicated Usage';"
        "$c.CounterSamples|Select-Object Path,CookedValue|ConvertTo-Json -Compress"
    )
    try:
        completed = _run_process(
            [
                executable,
                "-NoProfile",
                "-NonInteractive",
                "-Command",
                command,
            ],
            timeout_sec=20,
        )
        payload = json.loads(completed.stdout)
        rows = payload if isinstance(payload, list) else [payload]
        shared = sum(
            float(row.get("CookedValue") or 0)
            for row in rows
            if isinstance(row, Mapping)
            and "shared usage" in str(row.get("Path") or "").casefold()
        )
        dedicated = sum(
            float(row.get("CookedValue") or 0)
            for row in rows
            if isinstance(row, Mapping)
            and "dedicated usage" in str(row.get("Path") or "").casefold()
        )
        return {
            "available": True,
            "shared_usage_bytes": int(shared),
            "dedicated_usage_bytes": int(dedicated),
            "adapter_counter_count": len(rows),
        }
    except Exception as exc:
        return {
            "available": False,
            "reason": f"{type(exc).__name__}: {exc}",
        }


def query_gpu_compute_processes() -> dict[str, Any]:
    try:
        completed = _run_process(
            [
                "nvidia-smi",
                (
                    "--query-compute-apps="
                    "pid,process_name,gpu_uuid,used_gpu_memory"
                ),
                "--format=csv,noheader,nounits",
            ],
            timeout_sec=15,
        )
        rows: list[dict[str, Any]] = []
        for raw_line in completed.stdout.splitlines():
            parts = [part.strip() for part in raw_line.split(",", 3)]
            if len(parts) != 4:
                continue
            try:
                pid = int(parts[0])
            except ValueError:
                continue
            try:
                memory_used_mb: float | None = float(parts[3])
            except ValueError:
                memory_used_mb = None
            rows.append(
                {
                    "pid": pid,
                    "process_name": parts[1],
                    "gpu_uuid": parts[2],
                    "memory_used_mb": memory_used_mb,
                }
            )
        return {
            "available": True,
            "rows": rows,
        }
    except Exception as exc:
        return {
            "available": False,
            "rows": [],
            "reason": f"{type(exc).__name__}: {exc}",
        }


def query_docker_processes(
    docker_executable: str,
    container_id: str,
) -> dict[str, Any]:
    try:
        completed = _run_process(
            [
                docker_executable,
                "top",
                container_id,
                "-eo",
                "pid,comm,args",
            ],
            timeout_sec=15,
        )
        rows: list[dict[str, Any]] = []
        for raw_line in completed.stdout.splitlines()[1:]:
            parts = raw_line.strip().split(maxsplit=2)
            if len(parts) < 2:
                continue
            try:
                pid = int(parts[0])
            except ValueError:
                continue
            rows.append(
                {
                    "pid": pid,
                    "command": parts[1],
                    "arguments": parts[2] if len(parts) == 3 else "",
                }
            )
        return {
            "available": True,
            "rows": rows,
            "pids": sorted({int(row["pid"]) for row in rows}),
        }
    except Exception as exc:
        return {
            "available": False,
            "rows": [],
            "pids": [],
            "reason": f"{type(exc).__name__}: {exc}",
        }


def query_resource_snapshot(
    *,
    docker_executable: str,
    container_id: str | None,
) -> dict[str, Any]:
    started = time.perf_counter()
    if container_id:
        docker_stats = query_docker_stats(docker_executable, container_id)
        docker_processes = query_docker_processes(
            docker_executable,
            container_id,
        )
    else:
        docker_stats = {
            "available": False,
            "reason": "candidate container is not running",
        }
        docker_processes = {
            "available": False,
            "rows": [],
            "pids": [],
            "reason": "candidate container is not running",
        }
    return {
        "sampled_at": time.time(),
        "gpu": query_gpu_metrics(),
        "gpu_compute_processes": query_gpu_compute_processes(),
        "docker": docker_stats,
        "docker_processes": docker_processes,
        "app_rss_bytes": query_process_rss_bytes(),
        "wsl": query_wsl_memory(),
        "windows_gpu_adapter": query_windows_gpu_adapter_memory(),
        "sample_wall_sec": round(time.perf_counter() - started, 3),
    }


def _gpu_process_map(
    snapshot: Mapping[str, Any],
) -> dict[tuple[int, str], Mapping[str, Any]]:
    compute = snapshot.get("gpu_compute_processes") or {}
    if not bool(compute.get("available")):
        raise ValueError(
            str(compute.get("reason") or "GPU process query unavailable")
        )
    rows = compute.get("rows") or []
    result: dict[tuple[int, str], Mapping[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        pid = int(row.get("pid") or 0)
        gpu_uuid = str(row.get("gpu_uuid") or "")
        if pid <= 0 or not gpu_uuid:
            continue
        key = (pid, gpu_uuid)
        if key in result:
            raise ValueError(f"duplicate GPU process identity: {key}")
        result[key] = row
    return result


def attribute_candidate_gpu_memory(
    *,
    idle_snapshot: Mapping[str, Any],
    before_translation: Mapping[str, Any],
    after_translation: Mapping[str, Any],
) -> dict[str, Any]:
    """Return candidate VRAM only when process ownership is provable."""

    try:
        idle = _gpu_process_map(idle_snapshot)
        before = _gpu_process_map(before_translation)
        after = _gpu_process_map(after_translation)
        idle_ids = set(idle)
        before_new = set(before) - idle_ids
        after_new = set(after) - idle_ids
        if before_new != after_new or len(before_new) != 1:
            raise ValueError(
                "exactly one new GPU process must persist across both samples"
            )
        candidate_identity = next(iter(before_new))
        if (set(before) - {candidate_identity}) != idle_ids:
            raise ValueError(
                "external GPU process inventory changed before translation"
            )
        if (set(after) - {candidate_identity}) != idle_ids:
            raise ValueError(
                "external GPU process inventory changed after translation"
            )
        candidate_pid = candidate_identity[0]
        for label, snapshot in (
            ("before", before_translation),
            ("after", after_translation),
        ):
            docker_processes = snapshot.get("docker_processes") or {}
            if not bool(docker_processes.get("available")):
                raise ValueError(
                    f"Docker process inventory unavailable {label} translation"
                )
            docker_pids = {
                int(value)
                for value in docker_processes.get("pids") or []
            }
            if candidate_pid not in docker_pids:
                raise ValueError(
                    f"GPU process PID is not owned by the candidate {label} "
                    "translation"
                )
        memory_samples = [
            before[candidate_identity].get("memory_used_mb"),
            after[candidate_identity].get("memory_used_mb"),
        ]
        if any(value is None for value in memory_samples):
            raise ValueError("candidate GPU process memory is unavailable")
        return {
            "available": True,
            "pid": candidate_pid,
            "gpu_uuid": candidate_identity[1],
            "process_name": str(
                before[candidate_identity].get("process_name") or ""
            ),
            "memory_samples_mb": [
                float(value) for value in memory_samples
            ],
            "memory_used_mb": max(float(value) for value in memory_samples),
            "attribution": (
                "single persistent new GPU process with stable external "
                "inventory and matching Docker PID"
            ),
        }
    except Exception as exc:
        return {
            "available": False,
            "memory_used_mb": None,
            "reason": f"{type(exc).__name__}: {exc}",
        }


def query_docker_server_identity(docker_executable: str) -> dict[str, Any]:
    try:
        completed = _run_process(
            [
                docker_executable,
                "version",
                "--format",
                "{{json .Server}}",
            ],
            timeout_sec=30,
        )
        payload = json.loads(completed.stdout)
        components = {
            str(item.get("Name") or ""): str(item.get("Version") or "")
            for item in payload.get("Components") or []
            if isinstance(item, Mapping) and item.get("Name")
        }
        return {
            "available": True,
            "version": str(payload.get("Version") or ""),
            "api_version": str(payload.get("ApiVersion") or ""),
            "git_commit": str(payload.get("GitCommit") or ""),
            "os": str(payload.get("Os") or ""),
            "arch": str(payload.get("Arch") or ""),
            "kernel_version": str(payload.get("KernelVersion") or ""),
            "components": components,
        }
    except Exception as exc:
        return {
            "available": False,
            "reason": f"{type(exc).__name__}: {exc}",
        }


def query_nvidia_driver_identity() -> dict[str, Any]:
    try:
        completed = _run_process(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,name,driver_version",
                "--format=csv,noheader,nounits",
            ],
            timeout_sec=15,
        )
        rows = []
        for raw_line in completed.stdout.splitlines():
            parts = [part.strip() for part in raw_line.split(",", 3)]
            if len(parts) != 4:
                continue
            rows.append(
                {
                    "index": int(parts[0]),
                    "uuid": parts[1],
                    "name": parts[2],
                    "driver_version": parts[3],
                }
            )
        return {
            "available": bool(rows),
            "gpus": rows,
        }
    except Exception as exc:
        return {
            "available": False,
            "reason": f"{type(exc).__name__}: {exc}",
        }


def validate_measurement_environment(
    measurement_environment: Mapping[str, Any],
) -> None:
    docker_server = measurement_environment.get("docker_server") or {}
    nvidia_driver = measurement_environment.get("nvidia_driver") or {}
    errors: list[str] = []
    required_docker_fields = (
        "version",
        "api_version",
        "git_commit",
        "os",
        "arch",
        "kernel_version",
    )
    if not bool(docker_server.get("available")):
        errors.append("Docker server identity is unavailable")
    else:
        missing_docker = [
            field
            for field in required_docker_fields
            if not str(docker_server.get(field) or "").strip()
        ]
        if missing_docker:
            errors.append(
                "Docker server identity is incomplete: "
                + ", ".join(missing_docker)
            )
    gpu_rows = nvidia_driver.get("gpus") or []
    if not bool(nvidia_driver.get("available")) or not gpu_rows:
        errors.append("NVIDIA driver identity is unavailable")
    else:
        for index, row in enumerate(gpu_rows):
            if not isinstance(row, Mapping):
                errors.append(f"NVIDIA GPU identity {index} is not an object")
                continue
            missing_gpu = [
                field
                for field in ("uuid", "name", "driver_version")
                if not str(row.get(field) or "").strip()
            ]
            if missing_gpu:
                errors.append(
                    f"NVIDIA GPU identity {index} is incomplete: "
                    + ", ".join(missing_gpu)
                )
    if errors:
        raise ValueError(
            "Measurement environment contract failed: " + "; ".join(errors)
        )


def _sum_stats(
    destination: dict[str, int | float],
    source: Mapping[str, Any],
) -> None:
    for key, value in source.items():
        if key in {
            "gemma_telemetry_schema_version",
            "gemma_max_requested_group_size",
        }:
            destination[key] = max(
                int(destination.get(key, 0)),
                int(value or 0),
            )
        elif key == "gemma_configured_group_size":
            configured = int(value or 0)
            previous = int(destination.get(key, 0))
            if previous not in (0, configured) and configured != 0:
                raise ValueError(
                    "Gemma configured group size changed within one result set"
                )
            destination[key] = configured or previous
        elif isinstance(value, float):
            destination[key] = float(destination.get(key, 0.0)) + float(value)
        else:
            destination[key] = int(destination.get(key, 0)) + int(value or 0)


def _is_structural_translation(value: str) -> bool:
    normalized = str(value or "").strip()
    if not normalized:
        return False
    if normalized in {"{", "}", "[", "]"}:
        return True
    if (
        normalized.startswith("{")
        and normalized.endswith("}")
    ) or (
        normalized.startswith("[")
        and normalized.endswith("]")
    ):
        try:
            decoded = json.loads(normalized)
        except json.JSONDecodeError:
            return True
        return isinstance(decoded, (dict, list))
    return False


def build_engine(
    *,
    api_base_url: str,
    loaded_model: str,
    request_mode: str,
    group_size: int,
    timeout_sec: int,
) -> CustomLocalGemmaTranslation:
    engine = CustomLocalGemmaTranslation()
    engine.api_base_url = api_base_url.rstrip("/")
    engine.model = loaded_model
    engine.source_lang = "Japanese"
    engine.target_lang = "Korean"
    engine.request_mode = request_mode
    engine.chunk_size = int(group_size)
    engine.max_tokens = EXPECTED_MAX_COMPLETION_TOKENS
    engine.timeout = int(timeout_sec)
    engine.raw_response_logging = False
    engine.prompt_profile = DEFAULT_GEMMA_PROMPT_PROFILE
    engine.response_format_mode = DEFAULT_GEMMA_RESPONSE_FORMAT_MODE
    engine.response_schema_mode = DEFAULT_GEMMA_RESPONSE_SCHEMA_MODE
    engine.think_briefly_prompt = DEFAULT_GEMMA_THINK_BRIEFLY_PROMPT
    engine.temperature = DEFAULT_GEMMA_TRANSLATION_TEMPERATURE
    engine.top_k = DEFAULT_GEMMA_TRANSLATION_TOP_K
    engine.top_p = DEFAULT_GEMMA_TRANSLATION_TOP_P
    engine.min_p = DEFAULT_GEMMA_TRANSLATION_MIN_P
    engine.contextual_merge_input = True
    engine.configure_translation_memory(
        None,
        {
            "persistent_cache_enabled": False,
            "exact_tm_enabled": False,
        },
    )
    return engine


def _text_blocks(page: Mapping[str, Any]) -> list[TextBlock]:
    blocks: list[TextBlock] = []
    for block in page["blocks"]:
        blocks.append(
            TextBlock(
                text_bbox=np.asarray(block["xyxy"], dtype=np.int32),
                text=str(block["source"]),
                source_lang="Japanese",
                target_lang="Korean",
            )
        )
    return blocks


def warm_runtime(engine: CustomLocalGemmaTranslation) -> dict[str, Any]:
    block = TextBlock(
        text_bbox=np.asarray([0, 0, 100, 20], dtype=np.int32),
        text="これはランタイムのウォームアップです。",
        source_lang="Japanese",
        target_lang="Korean",
    )
    started = time.perf_counter()
    engine.translate(
        [block],
        np.zeros((1, 1, 3), dtype=np.uint8),
        "",
    )
    elapsed = time.perf_counter() - started
    if not str(block.translation or "").strip():
        raise ValueError("Warm-up translation is empty")
    return {
        "elapsed_sec": round(elapsed, 3),
        "output_nonempty": True,
        "stats": dict(engine.last_benchmark_stats),
    }


def _candidate_settings(candidate: str, group_size: int) -> tuple[str, int]:
    if candidate == CANDIDATE_BASELINE:
        return GEMMA_REQUEST_MODE_CONTEXTUAL_SINGLE, DEFAULT_BASELINE_GROUP_SIZE
    if candidate in {CANDIDATE_GROUPED_F16, CANDIDATE_GROUPED_Q8}:
        return GEMMA_REQUEST_MODE_CONTEXTUAL_GROUPED, int(group_size)
    raise ValueError(f"Unknown candidate: {candidate}")


def expected_request_stats(
    *,
    candidate: str,
    corpus: Mapping[str, Any],
    group_size: int,
) -> dict[str, int]:
    if candidate == CANDIDATE_BASELINE:
        normal_requests = EXPECTED_BLOCK_COUNT
        contextual_single = normal_requests
        contextual_grouped = 0
        configured_group_size = DEFAULT_BASELINE_GROUP_SIZE
    elif candidate in {CANDIDATE_GROUPED_F16, CANDIDATE_GROUPED_Q8}:
        normal_requests = expected_grouped_request_count(
            corpus["pages"],
            int(group_size),
        )
        contextual_single = 0
        contextual_grouped = normal_requests
        configured_group_size = int(group_size)
    else:
        raise ValueError(f"Unknown candidate: {candidate}")
    return {
        "gemma_logical_request_count": normal_requests,
        "gemma_http_attempt_count": normal_requests,
        "gemma_contextual_single_request_count": contextual_single,
        "gemma_contextual_grouped_request_count": contextual_grouped,
        "gemma_isolated_single_request_count": 0,
        "gemma_direct_grouped_request_count": 0,
        "gemma_configured_group_size": configured_group_size,
        "gemma_tm_requested_block_count": EXPECTED_BLOCK_COUNT,
    }


def request_stat_mismatches(
    *,
    stats: Mapping[str, Any],
    expected_stats: Mapping[str, int],
) -> dict[str, dict[str, int]]:
    return {
        key: {
            "expected": expected,
            "actual": int(stats.get(key, 0) or 0),
        }
        for key, expected in expected_stats.items()
        if int(stats.get(key, 0) or 0) != expected
    }


def run_candidate(
    *,
    candidate: str,
    round_index: int,
    container_name: str,
    container_id: str,
    docker_executable: str,
    api_base_url: str,
    model_name: str,
    corpus: Mapping[str, Any],
    group_size: int,
    contract_fingerprint: str,
    start_timeout_sec: int,
    request_timeout_sec: int,
) -> dict[str, Any]:
    request_mode, configured_group_size = _candidate_settings(
        candidate,
        group_size,
    )
    result: dict[str, Any] = {
        "candidate": candidate,
        "round": round_index,
        "request_mode": request_mode,
        "configured_group_size": configured_group_size,
        "container_name": container_name,
        "container_id": container_id,
        "contract_fingerprint": contract_fingerprint,
        "status": "running",
        "started_at": time.time(),
        "page_count": 0,
        "block_count": 0,
        "outputs": [],
    }
    start_attempted = False
    engine: CustomLocalGemmaTranslation | None = None
    translation_started: float | None = None
    combined_stats: dict[str, int | float] = {}
    page_timings: list[dict[str, Any]] = []
    outputs: list[dict[str, Any]] = []
    try:
        result["resource_idle_before_start"] = query_resource_snapshot(
            docker_executable=docker_executable,
            container_id=None,
        )
        start_attempted = True
        start_container(
            docker_executable=docker_executable,
            container_name=container_name,
            expected_container_id=container_id,
        )
        loaded_model = wait_for_runtime(
            api_base_url=api_base_url,
            model_name=model_name,
            timeout_sec=start_timeout_sec,
        )
        engine = build_engine(
            api_base_url=api_base_url,
            loaded_model=loaded_model,
            request_mode=request_mode,
            group_size=configured_group_size,
            timeout_sec=request_timeout_sec,
        )
        result["warmup"] = warm_runtime(engine)
        result["resource_before_translation"] = query_resource_snapshot(
            docker_executable=docker_executable,
            container_id=container_id,
        )

        translation_started = time.perf_counter()
        for page in corpus["pages"]:
            blocks = _text_blocks(page)
            page_started = time.perf_counter()
            try:
                engine.translate(
                    blocks,
                    np.zeros((1, 1, 3), dtype=np.uint8),
                    "",
                )
            except BaseException:
                _sum_stats(combined_stats, engine.last_benchmark_stats)
                result["failed_page"] = {
                    "page_index": int(page["page_index"]),
                    "page_name": str(page["page_name"]),
                    "elapsed_sec": round(
                        time.perf_counter() - page_started,
                        3,
                    ),
                }
                raise
            page_elapsed = time.perf_counter() - page_started
            _sum_stats(combined_stats, engine.last_benchmark_stats)
            page_timings.append(
                {
                    "page_index": int(page["page_index"]),
                    "page_name": str(page["page_name"]),
                    "block_count": len(blocks),
                    "elapsed_sec": round(page_elapsed, 3),
                }
            )
            for source_block, translated_block in zip(page["blocks"], blocks):
                translation = str(translated_block.translation or "")
                outputs.append(
                    {
                        "row_id": source_block["row_id"],
                        "page_index": int(page["page_index"]),
                        "page_name": str(page["page_name"]),
                        "block_index": int(source_block["block_index"]),
                        "source": str(source_block["source"]),
                        "source_sha256": str(source_block["source_sha256"]),
                        "translation": translation,
                        "translation_sha256": hashlib.sha256(
                            translation.encode("utf-8", errors="surrogatepass")
                        ).hexdigest(),
                        "empty": not bool(translation.strip()),
                        "structural_output": _is_structural_translation(
                            translation
                        ),
                    }
                )
            result["translation_elapsed_sec"] = round(
                time.perf_counter() - translation_started,
                3,
            )
            result["page_timings"] = list(page_timings)
            result["stats"] = dict(combined_stats)
            result["outputs"] = list(outputs)
            result["page_count"] = len(page_timings)
            result["block_count"] = len(outputs)
        translation_elapsed = time.perf_counter() - translation_started
        result["translation_elapsed_sec"] = round(translation_elapsed, 3)
        result["page_timings"] = page_timings
        result["stats"] = combined_stats
        result["outputs"] = outputs
        result["page_count"] = len(page_timings)
        result["block_count"] = len(outputs)
        result["resource_after_translation"] = query_resource_snapshot(
            docker_executable=docker_executable,
            container_id=container_id,
        )

        expected_ids = [
            block["row_id"]
            for page in corpus["pages"]
            for block in page["blocks"]
        ]
        actual_ids = [str(output["row_id"]) for output in outputs]
        empty_count = sum(bool(output["empty"]) for output in outputs)
        structural_count = sum(
            bool(output["structural_output"]) for output in outputs
        )
        unresolved_failure_count = (
            abs(EXPECTED_BLOCK_COUNT - len(outputs))
            + empty_count
            + structural_count
        )
        structural_telemetry_count = sum(
            int(combined_stats.get(key, 0) or 0)
            for key in STRUCTURAL_STATS
        )
        clean_run_telemetry_count = sum(
            int(combined_stats.get(key, 0) or 0)
            for key in CLEAN_RUN_STATS
        )
        expected_stats = expected_request_stats(
            candidate=candidate,
            corpus=corpus,
            group_size=group_size,
        )
        request_mismatches = request_stat_mismatches(
            stats=combined_stats,
            expected_stats=expected_stats,
        )
        request_contract_passed = not request_mismatches
        result["gates"] = {
            "page_count_ok": len(page_timings) == EXPECTED_PAGE_COUNT,
            "block_count_ok": len(outputs) == EXPECTED_BLOCK_COUNT,
            "order_preserved": actual_ids == expected_ids,
            "empty_translation_count": empty_count,
            "structural_output_count": structural_count,
            "structural_telemetry_count": structural_telemetry_count,
            "unresolved_failure_count": unresolved_failure_count,
            "clean_run_telemetry_count": clean_run_telemetry_count,
            "expected_request_stats": expected_stats,
            "request_stat_mismatches": request_mismatches,
            "request_contract_passed": request_contract_passed,
            "hard_gate_passed": (
                len(page_timings) == EXPECTED_PAGE_COUNT
                and len(outputs) == EXPECTED_BLOCK_COUNT
                and actual_ids == expected_ids
                and unresolved_failure_count == 0
                and structural_telemetry_count == 0
                and clean_run_telemetry_count == 0
                and request_contract_passed
            ),
            "clean_run_passed": clean_run_telemetry_count == 0,
        }
        result["status"] = (
            "passed"
            if result["gates"]["hard_gate_passed"]
            else "failed"
        )
    except BaseException as exc:
        if translation_started is not None:
            result["translation_elapsed_sec"] = round(
                time.perf_counter() - translation_started,
                3,
            )
        result["page_timings"] = list(page_timings)
        result["stats"] = dict(combined_stats)
        result["outputs"] = list(outputs)
        result["page_count"] = len(page_timings)
        result["block_count"] = len(outputs)
        result["status"] = "failed"
        result["failure_reason"] = f"{type(exc).__name__}: {exc}"
    finally:
        if (
            start_attempted
            and "resource_after_translation" not in result
        ):
            try:
                result["resource_after_translation"] = (
                    query_resource_snapshot(
                        docker_executable=docker_executable,
                        container_id=container_id,
                    )
                )
            except Exception as exc:
                result["resource_after_translation"] = {
                    "available": False,
                    "reason": f"{type(exc).__name__}: {exc}",
                }
        if (
            result.get("resource_idle_before_start")
            and result.get("resource_before_translation")
            and result.get("resource_after_translation")
        ):
            result["candidate_gpu_memory"] = (
                attribute_candidate_gpu_memory(
                    idle_snapshot=result["resource_idle_before_start"],
                    before_translation=result[
                        "resource_before_translation"
                    ],
                    after_translation=result[
                        "resource_after_translation"
                    ],
                )
            )
        else:
            result["candidate_gpu_memory"] = {
                "available": False,
                "memory_used_mb": None,
                "reason": "required resource snapshots are incomplete",
            }
        if start_attempted:
            try:
                stop_container(
                    docker_executable,
                    container_name,
                    expected_container_id=container_id,
                )
                result["container_stopped"] = True
            except Exception as exc:
                result["container_stopped"] = False
                result["stop_failure_reason"] = f"{type(exc).__name__}: {exc}"
                result["status"] = "failed"
        result["resource_after_stop"] = query_resource_snapshot(
            docker_executable=docker_executable,
            container_id=None,
        )
        result["completed_at"] = time.time()
    return result


def timing_variation_percent(values: Iterable[float]) -> float:
    samples = [float(value) for value in values]
    if len(samples) < 2:
        return 0.0
    median = statistics.median(samples)
    if median <= 0:
        return 0.0
    return ((max(samples) - min(samples)) / median) * 100.0


def relative_difference_percent(first: float, second: float) -> float:
    denominator = min(abs(float(first)), abs(float(second)))
    if denominator <= 0:
        return 0.0
    return abs(float(first) - float(second)) / denominator * 100.0


def should_add_third_round(
    elapsed_by_candidate: Mapping[str, Iterable[float]],
    *,
    threshold_percent: float = 5.0,
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    medians: dict[str, float] = {}
    for candidate in CANDIDATE_KEYS:
        values = [float(value) for value in elapsed_by_candidate.get(candidate, [])]
        if len(values) < 2:
            return False, ["two complete rounds are required"]
        medians[candidate] = statistics.median(values)
        variation = timing_variation_percent(values)
        if variation > threshold_percent:
            reasons.append(
                f"{candidate} run variation {variation:.2f}% exceeds "
                f"{threshold_percent:.2f}%"
            )
    finalist_difference = relative_difference_percent(
        medians[CANDIDATE_GROUPED_F16],
        medians[CANDIDATE_GROUPED_Q8],
    )
    if finalist_difference < threshold_percent:
        reasons.append(
            f"grouped finalist difference {finalist_difference:.2f}% is below "
            f"{threshold_percent:.2f}%"
        )
    return bool(reasons), reasons


def _load_completed_results(
    *,
    output_dir: Path,
    state: Mapping[str, Any],
    contract_fingerprint: str,
    corpus: Mapping[str, Any],
    group_size: int,
) -> list[dict[str, Any]]:
    runs_root = (output_dir / "runs").resolve()
    expected_outputs = [
        (
            str(block["row_id"]),
            str(block["source_sha256"]),
            str(block["source"]),
        )
        for page in corpus["pages"]
        for block in page["blocks"]
    ]
    results: list[dict[str, Any]] = []
    for record in state.get("runs") or []:
        if not isinstance(record, Mapping):
            continue
        relative = str(record.get("result_file") or "")
        if not relative:
            continue
        relative_path = Path(relative)
        if relative_path.is_absolute():
            raise ValueError("Resume result path must be relative")
        path = (output_dir / relative_path).resolve()
        try:
            path.relative_to(runs_root)
        except ValueError as exc:
            raise ValueError(
                f"Resume result escapes the suite runs directory: {relative}"
            ) from exc
        if not path.is_file():
            raise FileNotFoundError(f"Resume result is missing: {relative}")
        expected_result_sha256 = validated_sha256(
            str(record.get("result_sha256") or ""),
            option_name=f"state result_sha256 for {relative}",
        )
        actual_result_sha256 = sha256_file(path)
        if actual_result_sha256 != expected_result_sha256:
            raise ValueError(f"Resume result SHA-256 differs: {relative}")
        result = read_json(path)
        record_round = int(record.get("round") or 0)
        record_candidate = str(record.get("candidate") or "")
        if (
            int(result.get("round") or 0) != record_round
            or str(result.get("candidate") or "") != record_candidate
        ):
            raise ValueError(f"Resume result identity differs: {relative}")
        if str(result.get("contract_fingerprint") or "") != contract_fingerprint:
            raise ValueError(f"Resume result protocol differs: {relative}")
        if str(result.get("status") or "") == "passed":
            outputs = result.get("outputs") or []
            actual_outputs = [
                (
                    str(output.get("row_id") or ""),
                    str(output.get("source_sha256") or ""),
                    str(output.get("source") or ""),
                )
                for output in outputs
                if isinstance(output, Mapping)
            ]
            translations_valid = (
                len(outputs) == EXPECTED_BLOCK_COUNT
                and all(
                    isinstance(output, Mapping)
                    and bool(str(output.get("translation") or "").strip())
                    and hashlib.sha256(
                        str(output.get("source") or "").encode(
                            "utf-8",
                            errors="surrogatepass",
                        )
                    ).hexdigest()
                    == str(output.get("source_sha256") or "")
                    and hashlib.sha256(
                        str(output.get("translation") or "").encode(
                            "utf-8",
                            errors="surrogatepass",
                        )
                    ).hexdigest()
                    == str(output.get("translation_sha256") or "")
                    and not _is_structural_translation(
                        str(output.get("translation") or "")
                    )
                    for output in outputs
                )
            )
            gates = result.get("gates") or {}
            expected_stats = expected_request_stats(
                candidate=record_candidate,
                corpus=corpus,
                group_size=group_size,
            )
            stats = result.get("stats") or {}
            request_stats_valid = all(
                int(stats.get(key, 0) or 0) == expected
                for key, expected in expected_stats.items()
            )
            if not (
                int(result.get("page_count") or 0) == EXPECTED_PAGE_COUNT
                and int(result.get("block_count") or 0)
                == EXPECTED_BLOCK_COUNT
                and actual_outputs == expected_outputs
                and translations_valid
                and bool(gates.get("hard_gate_passed"))
                and bool(gates.get("clean_run_passed"))
                and bool(gates.get("request_contract_passed"))
                and request_stats_valid
                and bool(result.get("container_stopped"))
            ):
                raise ValueError(
                    f"Resume result no longer passes the final contract: {relative}"
                )
        results.append(result)
    return results


def _completed_result_map(
    *,
    output_dir: Path,
    state: Mapping[str, Any],
    contract_fingerprint: str,
    corpus: Mapping[str, Any],
    group_size: int,
) -> dict[tuple[int, str], dict[str, Any]]:
    completed: dict[tuple[int, str], dict[str, Any]] = {}
    for result in _load_completed_results(
        output_dir=output_dir,
        state=state,
        contract_fingerprint=contract_fingerprint,
        corpus=corpus,
        group_size=group_size,
    ):
        key = (int(result["round"]), str(result["candidate"]))
        if key in completed:
            raise ValueError(
                f"Resume state contains duplicate result identity: {key}"
            )
        completed[key] = result
    return completed


def _result_record(
    *,
    output_dir: Path,
    result_path: Path,
    result: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "round": int(result["round"]),
        "candidate": str(result["candidate"]),
        "status": str(result.get("status") or ""),
        "translation_elapsed_sec": result.get("translation_elapsed_sec"),
        "hard_gate_passed": bool(
            (result.get("gates") or {}).get("hard_gate_passed")
        ),
        "clean_run_passed": bool(
            (result.get("gates") or {}).get("clean_run_passed")
        ),
        "result_file": result_path.relative_to(output_dir).as_posix(),
        "result_sha256": sha256_file(result_path),
    }


def _candidate_summaries(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for candidate in CANDIDATE_KEYS:
        candidate_results = [
            result
            for result in results
            if result.get("candidate") == candidate
            and result.get("status") == "passed"
        ]
        elapsed = [
            float(result["translation_elapsed_sec"])
            for result in candidate_results
        ]
        if not elapsed:
            continue
        median_elapsed = statistics.median(elapsed)
        representative = min(
            candidate_results,
            key=lambda result: (
                abs(float(result["translation_elapsed_sec"]) - median_elapsed),
                int(result["round"]),
            ),
        )
        attributed_gpu_memory = [
            float(
                (result.get("candidate_gpu_memory") or {}).get(
                    "memory_used_mb"
                )
            )
            for result in candidate_results
            if bool(
                (result.get("candidate_gpu_memory") or {}).get("available")
            )
            and (
                result.get("candidate_gpu_memory") or {}
            ).get("memory_used_mb") is not None
        ]
        combined_stats: dict[str, int | float] = {}
        for result in candidate_results:
            _sum_stats(combined_stats, result.get("stats") or {})
        summaries.append(
            {
                "candidate": candidate,
                "run_count": len(candidate_results),
                "elapsed_sec": [round(value, 3) for value in elapsed],
                "median_elapsed_sec": round(median_elapsed, 3),
                "timing_variation_percent": round(
                    timing_variation_percent(elapsed),
                    3,
                ),
                "representative_round": int(representative["round"]),
                "hard_gate_passed": all(
                    bool((result.get("gates") or {}).get("hard_gate_passed"))
                    for result in candidate_results
                ),
                "clean_run_passed": all(
                    bool((result.get("gates") or {}).get("clean_run_passed"))
                    for result in candidate_results
                ),
                "attributed_gpu_memory_samples_mb": attributed_gpu_memory,
                "median_attributed_gpu_memory_mb": (
                    round(statistics.median(attributed_gpu_memory), 3)
                    if attributed_gpu_memory
                    else None
                ),
                "gpu_memory_attribution_available_for_all_runs": (
                    len(attributed_gpu_memory) == len(candidate_results)
                ),
                "stats": combined_stats,
            }
        )
    return summaries


def build_suite_summary(
    *,
    results: list[dict[str, Any]],
    third_round_required: bool,
    third_round_reasons: list[str],
    expected_baseline_requests: int,
    expected_grouped_requests: int,
    q8_vram_materiality_mb: int,
) -> dict[str, Any]:
    candidate_summaries = _candidate_summaries(results)
    by_candidate = {
        item["candidate"]: item
        for item in candidate_summaries
    }
    complete = all(
        candidate in by_candidate
        and int(by_candidate[candidate]["run_count"])
        >= (3 if third_round_required else 2)
        for candidate in CANDIDATE_KEYS
    )
    baseline = by_candidate.get(CANDIDATE_BASELINE)
    grouped_f16 = by_candidate.get(CANDIDATE_GROUPED_F16)
    grouped_q8 = by_candidate.get(CANDIDATE_GROUPED_Q8)
    grouped_f16_improvement = None
    grouped_q8_improvement = None
    q8_speed_improvement = None
    q8_vram_savings = None
    if baseline and grouped_f16:
        grouped_f16_improvement = (
            (
                float(baseline["median_elapsed_sec"])
                - float(grouped_f16["median_elapsed_sec"])
            )
            / float(baseline["median_elapsed_sec"])
            * 100.0
        )
    if baseline and grouped_q8:
        grouped_q8_improvement = (
            (
                float(baseline["median_elapsed_sec"])
                - float(grouped_q8["median_elapsed_sec"])
            )
            / float(baseline["median_elapsed_sec"])
            * 100.0
        )
    if grouped_f16 and grouped_q8:
        q8_speed_improvement = (
            (
                float(grouped_f16["median_elapsed_sec"])
                - float(grouped_q8["median_elapsed_sec"])
            )
            / float(grouped_f16["median_elapsed_sec"])
            * 100.0
        )
        if (
            grouped_f16.get("gpu_memory_attribution_available_for_all_runs")
            and grouped_q8.get("gpu_memory_attribution_available_for_all_runs")
            and grouped_f16.get("median_attributed_gpu_memory_mb") is not None
            and grouped_q8.get("median_attributed_gpu_memory_mb") is not None
        ):
            q8_vram_savings = (
                float(grouped_f16["median_attributed_gpu_memory_mb"])
                - float(grouped_q8["median_attributed_gpu_memory_mb"])
            )

    all_hard_gates = bool(
        complete
        and len(candidate_summaries) == len(CANDIDATE_KEYS)
        and all(item["hard_gate_passed"] for item in candidate_summaries)
    )
    grouped_speed_gate = bool(
        grouped_f16_improvement is not None
        and grouped_f16_improvement >= 20.0
    )
    q8_runtime_gate = bool(
        q8_speed_improvement is not None
        and (
            q8_speed_improvement >= 3.0
            or (
                q8_vram_savings is not None
                and q8_vram_savings >= q8_vram_materiality_mb
            )
        )
    )
    return {
        "status": "awaiting_user_quality_review" if complete else "incomplete",
        "translation_scope_only": True,
        "full_pipeline_executed": False,
        "candidate_summaries": candidate_summaries,
        "third_round_required": third_round_required,
        "third_round_reasons": third_round_reasons,
        "expected_normal_requests_per_run": {
            CANDIDATE_BASELINE: expected_baseline_requests,
            CANDIDATE_GROUPED_F16: expected_grouped_requests,
            CANDIDATE_GROUPED_Q8: expected_grouped_requests,
        },
        "grouped_f16_improvement_percent": (
            round(grouped_f16_improvement, 3)
            if grouped_f16_improvement is not None
            else None
        ),
        "grouped_q8_improvement_percent": (
            round(grouped_q8_improvement, 3)
            if grouped_q8_improvement is not None
            else None
        ),
        "q8_speed_improvement_vs_f16_percent": (
            round(q8_speed_improvement, 3)
            if q8_speed_improvement is not None
            else None
        ),
        "q8_attributed_vram_savings_mb": (
            round(q8_vram_savings, 3)
            if q8_vram_savings is not None
            else None
        ),
        "q8_vram_materiality_threshold_mb": q8_vram_materiality_mb,
        "gates": {
            "all_structural_gates_passed": all_hard_gates,
            "grouped_f16_translation_improvement_at_least_20_percent": (
                grouped_speed_gate
            ),
            "q8_at_least_3_percent_faster_or_material_vram_savings": (
                q8_runtime_gate
            ),
            "user_quality_review_passed": False,
            "full_pipeline_promotion_allowed": False,
        },
    }


def _markdown_cell(value: str) -> str:
    escaped = html.escape(str(value or ""), quote=False)
    return (
        escaped.replace("|", "&#124;")
        .replace("\r\n", "<br>")
        .replace("\n", "<br>")
        .replace("\r", "<br>")
    )


def _csv_safe(value: str) -> str:
    raw = str(value or "")
    if raw.lstrip().startswith(("=", "+", "-", "@")):
        return "'" + raw
    return raw


def _representative_results(
    *,
    results: list[dict[str, Any]],
    summary: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    representative_rounds = {
        str(item["candidate"]): int(item["representative_round"])
        for item in summary.get("candidate_summaries") or []
    }
    selected: dict[str, dict[str, Any]] = {}
    for result in results:
        candidate = str(result.get("candidate") or "")
        if int(result.get("round") or 0) == representative_rounds.get(candidate):
            selected[candidate] = result
    if set(selected) != set(CANDIDATE_KEYS):
        raise ValueError("Representative candidate outputs are incomplete")
    return selected


def write_blind_review(
    *,
    output_dir: Path,
    results: list[dict[str, Any]],
    summary: Mapping[str, Any],
    existing_key: Mapping[str, Any] | None = None,
) -> None:
    if existing_key:
        label_to_candidate = {
            str(label): str(candidate)
            for label, candidate in (existing_key.get("label_to_candidate") or {}).items()
        }
    else:
        shuffled = list(CANDIDATE_KEYS)
        random.SystemRandom().shuffle(shuffled)
        label_to_candidate = dict(zip(("A", "B", "C"), shuffled))
    if set(label_to_candidate) != {"A", "B", "C"} or set(
        label_to_candidate.values()
    ) != set(CANDIDATE_KEYS):
        raise ValueError("Blind candidate key is invalid")
    write_json(
        output_dir / "blind_key.json",
        {
            "label_to_candidate": label_to_candidate,
            "disclosure_status": "keep_private_until_user_review",
        },
    )

    representatives = _representative_results(
        results=results,
        summary=summary,
    )
    outputs_by_candidate = {
        candidate: {
            str(output["row_id"]): output
            for output in result.get("outputs") or []
        }
        for candidate, result in representatives.items()
    }
    row_ids = [
        str(output["row_id"])
        for output in representatives[CANDIDATE_BASELINE].get("outputs") or []
    ]
    for candidate, outputs in outputs_by_candidate.items():
        if list(outputs) != row_ids:
            raise ValueError(
                f"Blind output order differs for candidate {candidate}"
            )

    markdown_lines = [
        "# Blind Translation Quality Review",
        "",
        "- 설정명과 속도는 숨겼습니다.",
        "- 자연스러운 표현 차이는 허용합니다.",
        "- 화자, 관계, 부정, 행동, 대상, 숫자, 고유명사, 명시적 의미가 달라지면 회귀로 표시합니다.",
        "- `decision`에는 `A`, `B`, `C`, `tie`, `all-regressed` 중 하나를 적습니다.",
        "",
        "| row | page | block | source | A | B | C | decision | notes |",
        "|---:|---|---:|---|---|---|---|---|---|",
    ]
    csv_path = output_dir / "blind_review.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "row",
                "page",
                "block",
                "source",
                "A",
                "B",
                "C",
                "decision",
                "notes",
            ]
        )
        for row_number, row_id in enumerate(row_ids, start=1):
            baseline_output = outputs_by_candidate[CANDIDATE_BASELINE][row_id]
            values = {
                label: str(
                    outputs_by_candidate[candidate][row_id]["translation"]
                )
                for label, candidate in label_to_candidate.items()
            }
            source = str(baseline_output["source"])
            page_name = str(baseline_output["page_name"])
            block_index = int(baseline_output["block_index"])
            markdown_lines.append(
                "| {row} | {page} | {block} | {source} | {a} | {b} | {c} |  |  |".format(
                    row=row_number,
                    page=_markdown_cell(page_name),
                    block=block_index,
                    source=_markdown_cell(source),
                    a=_markdown_cell(values["A"]),
                    b=_markdown_cell(values["B"]),
                    c=_markdown_cell(values["C"]),
                )
            )
            writer.writerow(
                [
                    row_number,
                    _csv_safe(page_name),
                    block_index,
                    _csv_safe(source),
                    _csv_safe(values["A"]),
                    _csv_safe(values["B"]),
                    _csv_safe(values["C"]),
                    "",
                    "",
                ]
            )
    (output_dir / "blind_review.md").write_text(
        "\n".join(markdown_lines) + "\n",
        encoding="utf-8",
    )


def render_summary_markdown(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Gemma Final Translation-Only Comparison",
        "",
        "이 결과는 번역 stage만 측정했다. 사용자 품질 승인 전에는 22페이지 전체 파이프라인을 실행하지 않는다.",
        "",
        "| candidate | runs | elapsed sec | median sec | variation | structural | clean | median attributed VRAM MiB |",
        "|---|---:|---|---:|---:|---|---|---:|",
    ]
    for item in summary.get("candidate_summaries") or []:
        lines.append(
            "| {candidate} | {runs} | {elapsed} | {median} | {variation:.3f}% | {hard} | {clean} | {vram} |".format(
                candidate=item["candidate"],
                runs=item["run_count"],
                elapsed=", ".join(
                    f"{float(value):.3f}"
                    for value in item["elapsed_sec"]
                ),
                median=item["median_elapsed_sec"],
                variation=float(item["timing_variation_percent"]),
                hard="PASS" if item["hard_gate_passed"] else "FAIL",
                clean="PASS" if item["clean_run_passed"] else "WARN",
                vram=(
                    item["median_attributed_gpu_memory_mb"]
                    if item["median_attributed_gpu_memory_mb"] is not None
                    else "unavailable"
                ),
            )
        )
    lines.extend(
        [
            "",
            "## Performance gates",
            "",
            f"- grouped F16 improvement: `{summary.get('grouped_f16_improvement_percent')}`%",
            f"- grouped Q8 improvement: `{summary.get('grouped_q8_improvement_percent')}`%",
            f"- Q8 speed improvement versus F16: `{summary.get('q8_speed_improvement_vs_f16_percent')}`%",
            "- Q8 attributed VRAM savings: "
            f"`{summary.get('q8_attributed_vram_savings_mb')}` MiB",
            "",
            "## Next gate",
            "",
            "- `blind_review.md` 또는 `blind_review.csv`에서 292개 행을 검수한다.",
            "- `blind_key.json`은 검수가 끝날 때까지 열지 않는다.",
            "- 사용자 승인 전에는 full-pipeline promotion을 금지한다.",
            "",
        ]
    )
    return "\n".join(lines)


def _ensure_results_are_untracked(output_dir: Path) -> None:
    try:
        relative = output_dir.resolve().relative_to(ROOT.resolve())
    except ValueError:
        return
    completed = _run_process(
        [
            "git",
            "-C",
            str(ROOT),
            "check-ignore",
            "--no-index",
            "--quiet",
            "--",
            relative.as_posix(),
        ],
        timeout_sec=15,
        check=False,
    )
    if completed.returncode != 0:
        raise ValueError(
            "Raw benchmark output must be outside Git or under an ignored path"
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the frozen 22-page/292-block Gemma translation-only A/B/C "
            "comparison through CustomLocalGemmaTranslation."
        )
    )
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--ocr-snapshot", required=True)
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--baseline-container", required=True)
    parser.add_argument("--f16-container", required=True)
    parser.add_argument("--q8-container", required=True)
    parser.add_argument("--model-source", required=True)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--model-volume", default=DEFAULT_MODEL_VOLUME)
    parser.add_argument("--model-size", type=int, default=DEFAULT_MODEL_SIZE)
    parser.add_argument("--expected-model-sha256", required=True)
    parser.add_argument(
        "--expected-input-manifest-sha256",
        default=DEFAULT_EXPECTED_INPUT_MANIFEST_SHA256,
    )
    parser.add_argument(
        "--expected-ocr-snapshot-sha256",
        default=DEFAULT_EXPECTED_OCR_SNAPSHOT_SHA256,
    )
    parser.add_argument("--expected-image-id", default=DEFAULT_IMAGE_ID)
    parser.add_argument("--hash-helper-image", default=DEFAULT_HASH_HELPER_IMAGE)
    parser.add_argument(
        "--expected-hash-helper-image-id",
        default=DEFAULT_HASH_HELPER_IMAGE_ID,
    )
    parser.add_argument(
        "--expected-baseline-config-fingerprint",
        default=DEFAULT_RUNTIME_CONFIG_FINGERPRINTS[CANDIDATE_BASELINE],
    )
    parser.add_argument(
        "--expected-f16-config-fingerprint",
        default=DEFAULT_RUNTIME_CONFIG_FINGERPRINTS[CANDIDATE_GROUPED_F16],
    )
    parser.add_argument(
        "--expected-q8-config-fingerprint",
        default=DEFAULT_RUNTIME_CONFIG_FINGERPRINTS[CANDIDATE_GROUPED_Q8],
    )
    parser.add_argument("--api-base-url", default="http://127.0.0.1:18080/v1")
    parser.add_argument("--group-size", type=int, default=DEFAULT_GROUP_SIZE)
    parser.add_argument(
        "--max-completion-tokens",
        type=int,
        default=EXPECTED_MAX_COMPLETION_TOKENS,
    )
    parser.add_argument("--start-timeout-sec", type=int, default=420)
    parser.add_argument("--request-timeout-sec", type=int, default=240)
    parser.add_argument("--third-round-threshold-percent", type=float, default=5.0)
    parser.add_argument("--q8-vram-materiality-mb", type=int, default=512)
    parser.add_argument("--docker-executable", default="")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args(argv)


def reset_state_for_execution(state: dict[str, Any]) -> None:
    state["status"] = "running"
    state.pop("completed_at", None)
    state.pop("failure_reason", None)


def validate_resume_then_reset_state(
    *,
    output_dir: Path,
    state: dict[str, Any],
    contract_fingerprint: str,
    corpus: Mapping[str, Any],
    group_size: int,
) -> dict[tuple[int, str], dict[str, Any]]:
    completed = _completed_result_map(
        output_dir=output_dir,
        state=state,
        contract_fingerprint=contract_fingerprint,
        corpus=corpus,
        group_size=group_size,
    )
    reset_state_for_execution(state)
    return completed


def execute_benchmark_round(
    *,
    round_index: int,
    completed: dict[tuple[int, str], dict[str, Any]],
    state: dict[str, Any],
    output_dir: Path,
    containers: Mapping[str, str],
    container_ids: Mapping[str, str],
    docker_executable: str,
    api_base_url: str,
    model_name: str,
    corpus: Mapping[str, Any],
    group_size: int,
    contract_fingerprint: str,
    start_timeout_sec: int,
    request_timeout_sec: int,
) -> bool:
    for candidate in candidate_order(round_index):
        key = (round_index, candidate)
        existing_result = completed.get(key)
        if existing_result and existing_result.get("status") == "passed":
            print(
                f"[resume] round={round_index} candidate={candidate}",
                flush=True,
            )
            continue
        print(
            f"[run] round={round_index} candidate={candidate}",
            flush=True,
        )
        result = run_candidate(
            candidate=candidate,
            round_index=round_index,
            container_name=containers[candidate],
            container_id=container_ids[candidate],
            docker_executable=docker_executable,
            api_base_url=api_base_url,
            model_name=model_name,
            corpus=corpus,
            group_size=group_size,
            contract_fingerprint=contract_fingerprint,
            start_timeout_sec=start_timeout_sec,
            request_timeout_sec=request_timeout_sec,
        )
        result_path = (
            output_dir
            / "runs"
            / f"round-{round_index}_{candidate}.json"
        )
        write_json(result_path, result)
        state["runs"] = [
            record
            for record in state.get("runs") or []
            if not (
                int(record.get("round") or 0) == round_index
                and str(record.get("candidate") or "") == candidate
            )
        ]
        state["runs"].append(
            _result_record(
                output_dir=output_dir,
                result_path=result_path,
                result=result,
            )
        )
        write_json(output_dir / "suite_state.json", state)
        completed[key] = result
        if result.get("status") != "passed":
            state["status"] = "failed"
            state["failure_reason"] = str(
                result.get("failure_reason")
                or result.get("gates")
                or "candidate hard gate failed"
            )
            write_json(output_dir / "suite_state.json", state)
            return False
    return True


def _main_unlocked(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    require_historical_grouped_runtime()
    if args.group_size < 2 or args.group_size > 12:
        raise ValueError("--group-size must be between 2 and 12")
    if args.max_completion_tokens != EXPECTED_MAX_COMPLETION_TOKENS:
        raise ValueError(
            f"--max-completion-tokens is fixed at {EXPECTED_MAX_COMPLETION_TOKENS}"
        )
    if args.third_round_threshold_percent <= 0:
        raise ValueError("--third-round-threshold-percent must be positive")
    if args.q8_vram_materiality_mb <= 0:
        raise ValueError("--q8-vram-materiality-mb must be positive")
    api_base_url = validate_api_base_url(args.api_base_url)

    input_root = Path(args.input_root).expanduser().resolve()
    snapshot_path = Path(args.ocr_snapshot).expanduser().resolve()
    results_root = Path(args.results_root).expanduser().resolve()
    model_source = Path(args.model_source).expanduser().resolve()
    if not input_root.is_dir():
        raise FileNotFoundError(f"Input root not found: {input_root}")
    if not snapshot_path.is_file():
        raise FileNotFoundError(f"OCR snapshot not found: {snapshot_path}")
    if not model_source.is_file():
        raise FileNotFoundError(f"Model source not found: {model_source}")

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else results_root / f"{timestamp}_gemma-final-translation"
    )
    _ensure_results_are_untracked(output_dir)
    if output_dir.exists() and not args.resume and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty; use --resume: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    state_path = output_dir / "suite_state.json"
    existing_state = (
        read_json(state_path)
        if args.resume and state_path.is_file()
        else None
    )

    print("[preflight] freezing 22 input hashes and the 292-block OCR snapshot")
    input_manifest = build_input_manifest(input_root)
    corpus = load_frozen_corpus(
        input_root=input_root,
        snapshot_path=snapshot_path,
        input_manifest=input_manifest,
    )
    validate_frozen_asset_digests(
        input_manifest=input_manifest,
        corpus=corpus,
        expected_input_manifest_sha256=args.expected_input_manifest_sha256,
        expected_ocr_snapshot_sha256=args.expected_ocr_snapshot_sha256,
    )
    docker_executable = resolve_docker_executable(args.docker_executable)
    containers = {
        CANDIDATE_BASELINE: str(args.baseline_container),
        CANDIDATE_GROUPED_F16: str(args.f16_container),
        CANDIDATE_GROUPED_Q8: str(args.q8_container),
    }
    if len(set(containers.values())) != len(containers):
        raise ValueError("Each candidate requires a distinct container")
    expected_runtime_fingerprints = {
        CANDIDATE_BASELINE: validated_sha256(
            args.expected_baseline_config_fingerprint,
            option_name="--expected-baseline-config-fingerprint",
        ),
        CANDIDATE_GROUPED_F16: validated_sha256(
            args.expected_f16_config_fingerprint,
            option_name="--expected-f16-config-fingerprint",
        ),
        CANDIDATE_GROUPED_Q8: validated_sha256(
            args.expected_q8_config_fingerprint,
            option_name="--expected-q8-config-fingerprint",
        ),
    }
    runtime_contracts = inspect_and_stop_candidate_runtimes(
        docker_executable=docker_executable,
        containers=containers,
        expected_image_id=args.expected_image_id,
        expected_model_name=args.model_name,
        expected_model_volume=args.model_volume,
        expected_runtime_fingerprints=expected_runtime_fingerprints,
    )
    runtime_by_candidate = {
        str(contract["candidate"]): contract
        for contract in runtime_contracts
    }
    container_ids = {
        candidate: str(runtime_by_candidate[candidate]["container_id"])
        for candidate in CANDIDATE_KEYS
    }
    expected_model_sha256 = validated_sha256(
        args.expected_model_sha256,
        option_name="--expected-model-sha256",
    )
    expected_helper_image_id = validated_sha256(
        args.expected_hash_helper_image_id.removeprefix("sha256:"),
        option_name="--expected-hash-helper-image-id",
    )
    model_contract = prepare_model_contract(
        docker_executable=docker_executable,
        helper_image=args.hash_helper_image,
        expected_helper_image_id=f"sha256:{expected_helper_image_id}",
        model_source=model_source,
        model_name=args.model_name,
        model_volume=args.model_volume,
        expected_size=int(args.model_size),
        expected_sha256=expected_model_sha256,
    )
    measurement_environment = {
        "docker_server": query_docker_server_identity(docker_executable),
        "nvidia_driver": query_nvidia_driver_identity(),
    }
    validate_measurement_environment(measurement_environment)

    public_asset_contract = {
        "input_manifest": input_manifest,
        "expected_input_manifest_sha256": (
            args.expected_input_manifest_sha256.casefold()
        ),
        "expected_ocr_snapshot_sha256": (
            args.expected_ocr_snapshot_sha256.casefold()
        ),
        "snapshot": {
            key: value
            for key, value in corpus.items()
            if key != "pages"
        },
    }
    write_json(output_dir / "frozen_assets.json", public_asset_contract)
    write_json(output_dir / "runtime_contracts.json", runtime_contracts)
    write_json(output_dir / "model_contract.json", model_contract)
    write_json(
        output_dir / "measurement_environment.json",
        measurement_environment,
    )
    translation_behavior_contract = build_translation_behavior_contract(
        group_size=int(args.group_size),
    )
    suite_contract = {
        "benchmark_protocol_version": BENCHMARK_PROTOCOL_VERSION,
        "input_manifest_sha256": input_manifest["manifest_sha256"],
        "snapshot_sha256": corpus["snapshot_sha256"],
        "snapshot_contract_sha256": corpus["contract_sha256"],
        "model_sha256": expected_model_sha256,
        "image_id": args.expected_image_id,
        "hash_helper_image_id": (
            model_contract["hash_helper_image"]["image_id"]
        ),
        "runtime_config_fingerprints": {
            candidate: runtime_by_candidate[candidate]["config_fingerprint"]
            for candidate in CANDIDATE_KEYS
        },
        "translation_behavior_contract_sha256": (
            translation_behavior_contract["contract_sha256"]
        ),
        "api_base_url": api_base_url,
        "group_size": int(args.group_size),
        "max_completion_tokens": int(args.max_completion_tokens),
        "start_timeout_sec": int(args.start_timeout_sec),
        "request_timeout_sec": int(args.request_timeout_sec),
        "measurement_environment": measurement_environment,
        "third_round_threshold_percent": float(
            args.third_round_threshold_percent
        ),
        "q8_vram_materiality_mb": int(args.q8_vram_materiality_mb),
    }
    contract_fingerprint = canonical_sha256(suite_contract)
    if existing_state and (
        str(existing_state.get("contract_fingerprint") or "")
        != contract_fingerprint
    ):
        raise ValueError("Resume contract differs from the existing suite")

    state: dict[str, Any] = (
        dict(existing_state)
        if existing_state
        else {
            "status": "running",
            "started_at": time.time(),
            "runs": [],
        }
    )
    state.update(
        {
            "contract_fingerprint": contract_fingerprint,
            "suite_contract": suite_contract,
            "translation_behavior_contract": translation_behavior_contract,
            "model_contract": model_contract,
            "group_size": int(args.group_size),
            "max_completion_tokens": int(args.max_completion_tokens),
            "quality_status": "pending_user_review",
            "full_pipeline_executed": False,
        }
    )
    completed: dict[tuple[int, str], dict[str, Any]] = {}
    if not args.preflight_only:
        completed = validate_resume_then_reset_state(
            output_dir=output_dir,
            state=state,
            contract_fingerprint=contract_fingerprint,
            corpus=corpus,
            group_size=int(args.group_size),
        )
    write_json(state_path, state)
    if args.preflight_only:
        state["status"] = "preflight_passed"
        state["completed_at"] = time.time()
        write_json(state_path, state)
        print(f"Preflight passed: {output_dir}")
        return 0

    third_round_required = False
    third_round_reasons: list[str] = []
    for round_index in (1, 2):
        if not execute_benchmark_round(
            round_index=round_index,
            completed=completed,
            state=state,
            output_dir=output_dir,
            containers=containers,
            container_ids=container_ids,
            docker_executable=docker_executable,
            api_base_url=api_base_url,
            model_name=args.model_name,
            corpus=corpus,
            group_size=int(args.group_size),
            contract_fingerprint=contract_fingerprint,
            start_timeout_sec=int(args.start_timeout_sec),
            request_timeout_sec=int(args.request_timeout_sec),
        ):
            return 1

    elapsed_by_candidate = {
        candidate: [
            float(result["translation_elapsed_sec"])
            for (round_index, result_candidate), result in completed.items()
            if round_index in (1, 2)
            and result_candidate == candidate
            and result.get("status") == "passed"
        ]
        for candidate in CANDIDATE_KEYS
    }
    third_round_required, third_round_reasons = should_add_third_round(
        elapsed_by_candidate,
        threshold_percent=float(args.third_round_threshold_percent),
    )
    if third_round_required:
        if not execute_benchmark_round(
            round_index=3,
            completed=completed,
            state=state,
            output_dir=output_dir,
            containers=containers,
            container_ids=container_ids,
            docker_executable=docker_executable,
            api_base_url=api_base_url,
            model_name=args.model_name,
            corpus=corpus,
            group_size=int(args.group_size),
            contract_fingerprint=contract_fingerprint,
            start_timeout_sec=int(args.start_timeout_sec),
            request_timeout_sec=int(args.request_timeout_sec),
        ):
            return 1

    results = _load_completed_results(
        output_dir=output_dir,
        state=state,
        contract_fingerprint=contract_fingerprint,
        corpus=corpus,
        group_size=int(args.group_size),
    )
    summary = build_suite_summary(
        results=results,
        third_round_required=third_round_required,
        third_round_reasons=third_round_reasons,
        expected_baseline_requests=EXPECTED_BLOCK_COUNT,
        expected_grouped_requests=expected_grouped_request_count(
            corpus["pages"],
            int(args.group_size),
        ),
        q8_vram_materiality_mb=int(args.q8_vram_materiality_mb),
    )
    write_json(output_dir / "summary.json", summary)
    (output_dir / "summary.md").write_text(
        render_summary_markdown(summary),
        encoding="utf-8",
    )
    existing_blind_key = (
        read_json(output_dir / "blind_key.json")
        if (output_dir / "blind_key.json").is_file()
        else None
    )
    write_blind_review(
        output_dir=output_dir,
        results=results,
        summary=summary,
        existing_key=existing_blind_key,
    )
    state["status"] = str(summary["status"])
    state["third_round_required"] = third_round_required
    state["third_round_reasons"] = third_round_reasons
    state["completed_at"] = time.time()
    write_json(state_path, state)
    for candidate, container_name in containers.items():
        stop_container(
            docker_executable,
            container_name,
            expected_container_id=container_ids[candidate],
        )
    if running_port_18080_containers(docker_executable):
        raise RuntimeError("A Gemma candidate remained active after the suite")
    print(f"Translation-only comparison complete: {output_dir}")
    return 0


def main(argv: list[str] | None = None) -> int:
    lock_path = ROOT / ".git" / "gemma-final-translation-suite.lock"
    with ExclusiveSuiteLock(lock_path):
        return _main_unlocked(argv)


if __name__ == "__main__":
    raise SystemExit(main())

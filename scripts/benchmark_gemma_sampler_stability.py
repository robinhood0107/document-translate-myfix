#!/usr/bin/env python3
"""Run the private, text-only Gemma sampler stability protocol.

The corpus manifest, request payloads, responses, and semantic judgments are
private validation artifacts.  This script deliberately writes them only
under ``banchmark_result_log/`` and keeps the tracked protocol free of raw
source text and translations.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time
import urllib.error
import urllib.request
import uuid
from typing import Any, Iterable, Mapping


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "scripts"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from benchmark_gemma_translation_only_matrix import (  # noqa: E402
    build_single_block_translation_request,
)
from modules.ocr.local_runtime import LocalOCRRuntimeManager  # noqa: E402
from modules.translation.gemma_runtime_contract import DEFAULT_GEMMA_MODEL_VOLUME  # noqa: E402
from modules.translation.llm.custom_local_gemma import (  # noqa: E402
    DEFAULT_GEMMA_LOCAL_ENDPOINT,
    DEFAULT_GEMMA_LOCAL_MODEL,
    DEFAULT_GEMMA_MAX_COMPLETION_TOKENS,
    DEFAULT_GEMMA_PROMPT_PROFILE,
    DEFAULT_GEMMA_RESPONSE_FORMAT_MODE,
    DEFAULT_GEMMA_RESPONSE_SCHEMA_MODE,
    DEFAULT_GEMMA_THINK_BRIEFLY_PROMPT,
    DEFAULT_GEMMA_TRANSLATION_MIN_P,
    DEFAULT_GEMMA_TRANSLATION_TEMPERATURE,
    DEFAULT_GEMMA_TRANSLATION_TOP_K,
    DEFAULT_GEMMA_TRANSLATION_TOP_P,
    CustomLocalGemmaTranslation,
)
from modules.translation.local_runtime import LocalGemmaRuntimeManager  # noqa: E402
from modules.utils.local_llama_router import (  # noqa: E402
    LocalLlamaRouterCoordinator,
    ROUTER_GEMMA_ALIAS,
    RouterRuntimeError,
)
from scripts.validation_artifact_harness import (  # noqa: E402
    ArtifactHarnessError,
    ManagedArtifactRun,
    select_managed_output_directory,
)


PROTOCOL_PATH = ROOT / "benchmarks" / "gemma_sampler_stability" / "protocol-v1.json"
ARCHIVE_ROOT = ROOT / "banchmark_result_log"
ARTIFACT_CATEGORY = "10-gemma-translation"
ARTIFACT_FAMILY = "gemma-sampler-stability"
DEFAULT_TIMEOUT_SEC = 180
DEFAULT_SEEDS = (20260801, 20260802, 20260803)
SAMPLER_KEYS = ("temperature", "top_p", "top_k", "min_p", "seed")
EXPECTED_CASE_COUNTS = {
    "ja-ko": 18,
    "en-ko": 4,
    "router_mismatch": 14,
    "ja_meaning_control": 4,
}
MASKING_RE = re.compile(
    r"(?:\*{2,}|█{2,}|<censored>|\[censored\]|번역\s*불가|검열)",
    re.IGNORECASE,
)
CHANNEL_TOKEN_RE = re.compile(r"<\|channel\>[^\r\n<]*|<channel\|>")


class SamplerProtocolError(ValueError):
    """Raised when a private manifest or response violates the protocol."""


class SamplerRunError(RuntimeError):
    """Raised when a live sampler run cannot continue safely."""


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def text_sha256(value: str) -> str:
    return hashlib.sha256(str(value).encode("utf-8", errors="surrogatepass")).hexdigest()


def _float_label(value: float) -> str:
    label = f"{float(value):.3f}".rstrip("0").rstrip(".")
    return label or "0"


def _require_float(value: Any, *, field: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise SamplerProtocolError(f"{field} must be numeric") from exc
    if parsed < 0:
        raise SamplerProtocolError(f"{field} must not be negative")
    return parsed


def load_protocol() -> dict[str, Any]:
    payload = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("protocol_version") != "gemma-sampler-stability-v1":
        raise SamplerProtocolError("Sampler protocol is missing or has an unexpected version")
    return payload


def load_corpus_manifest(path: Path) -> tuple[dict[str, Any], str]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SamplerProtocolError(f"Unable to read corpus manifest: {path}") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != "gemma-sampler-corpus-v1":
        raise SamplerProtocolError("Corpus manifest schema must be gemma-sampler-corpus-v1")
    if payload.get("protocol_version") != "gemma-sampler-stability-v1":
        raise SamplerProtocolError("Corpus manifest is not pinned to this sampler protocol")
    cases = payload.get("cases")
    if not isinstance(cases, list) or len(cases) != 22:
        raise SamplerProtocolError("Corpus manifest must contain exactly 22 cases")

    seen: set[str] = set()
    counts = {"ja-ko": 0, "en-ko": 0, "router_mismatch": 0, "ja_meaning_control": 0}
    for case in cases:
        if not isinstance(case, dict):
            raise SamplerProtocolError("Every corpus case must be an object")
        case_id = str(case.get("case_id") or "").strip()
        if not case_id or case_id in seen:
            raise SamplerProtocolError(f"Duplicate or empty case_id: {case_id!r}")
        seen.add(case_id)
        language_pair = str(case.get("language_pair") or "").strip()
        family = str(case.get("family") or "").strip()
        if language_pair not in {"ja-ko", "en-ko"}:
            raise SamplerProtocolError(f"Unsupported language_pair for {case_id}")
        if family not in {"router_mismatch", "meaning_control"}:
            raise SamplerProtocolError(f"Unsupported case family for {case_id}")
        if language_pair == "en-ko" and family != "meaning_control":
            raise SamplerProtocolError("English cases must be classified as meaning_control")
        for field in ("source_text", "canonical_ko", "required_meaning", "forbidden_changes", "allowed_style"):
            value = case.get(field)
            if isinstance(value, str):
                if not value.strip():
                    raise SamplerProtocolError(f"Empty {field} for {case_id}")
            elif isinstance(value, list):
                if not value or any(not str(item).strip() for item in value):
                    raise SamplerProtocolError(f"Empty {field} item for {case_id}")
            else:
                raise SamplerProtocolError(f"Missing {field} for {case_id}")
        for context_key in ("context_before", "context_after"):
            context = case.get(context_key, [])
            if not isinstance(context, list) or any(not str(item).strip() for item in context):
                raise SamplerProtocolError(f"Invalid {context_key} for {case_id}")
        text_material = json.dumps(
            {
                key: case.get(key)
                for key in ("source_text", "canonical_ko", "context_before", "context_after")
            },
            ensure_ascii=False,
        ).lower()
        if any(marker in text_material for marker in (".png", ".jpg", ".jpeg", ".webp", "image_path", "image_bytes")):
            raise SamplerProtocolError(f"Image material is not allowed in corpus case {case_id}")
        for forbidden_key in ("image", "image_path", "image_bytes", "image_data"):
            if forbidden_key in case:
                raise SamplerProtocolError(f"Image material is not allowed in corpus case {case_id}")
        counts[language_pair] += 1
        if family == "router_mismatch":
            counts[family] += 1
        elif language_pair == "ja-ko" and family == "meaning_control":
            counts["ja_meaning_control"] += 1
    if counts != EXPECTED_CASE_COUNTS:
        raise SamplerProtocolError(f"Corpus counts are {counts}, expected {EXPECTED_CASE_COUNTS}")
    return payload, canonical_sha256(payload)


@dataclass(frozen=True)
class SamplerArm:
    phase: str
    temperature: float
    top_p: float
    top_k: int
    min_p: float = 0.0

    @property
    def arm_id(self) -> str:
        return (
            f"sampler-t{_float_label(self.temperature)}"
            f"-p{_float_label(self.top_p)}-k{int(self.top_k)}"
            f"-m{_float_label(self.min_p)}"
        )

    def sampler_values(self, seed: int) -> dict[str, Any]:
        return {
            "temperature": float(self.temperature),
            "top_p": float(self.top_p),
            "top_k": int(self.top_k),
            "min_p": float(self.min_p),
            "seed": int(seed),
        }


def _validate_selected_temperature(value: float | None) -> float:
    if value is None:
        raise SamplerProtocolError("--selected-temperature is required for top-p/top-k phases")
    normalized = round(float(value), 1)
    if normalized < 0 or normalized > 1 or abs(normalized - float(value)) > 1e-9:
        raise SamplerProtocolError("selected temperature must be one decimal from 0.0 to 1.0")
    return normalized


def _validate_selected_top_p(value: float | None) -> float:
    if value is None:
        raise SamplerProtocolError("--selected-top-p is required for top-k phase")
    normalized = round(float(value), 2)
    if normalized not in {0.9, 0.95, 1.0}:
        raise SamplerProtocolError("selected top-p must be 0.90, 0.95, or 1.00")
    return normalized


def build_arms(
    phase: str,
    *,
    selected_temperature: float | None = None,
    selected_top_p: float | None = None,
) -> list[SamplerArm]:
    if phase not in {"temperature", "top-p", "top-k", "all"}:
        raise SamplerProtocolError(f"Unsupported phase: {phase}")
    arms: list[SamplerArm] = []
    if phase in {"temperature", "all"}:
        arms.extend(
            SamplerArm("temperature", round(index / 10, 1), 0.95, 64)
            for index in range(11)
        )
    if phase in {"top-p", "all"}:
        temperature = _validate_selected_temperature(selected_temperature)
        arms.extend(
            SamplerArm("top-p", temperature, top_p, 64)
            for top_p in (0.9, 0.95, 1.0)
        )
    if phase in {"top-k", "all"}:
        temperature = _validate_selected_temperature(selected_temperature)
        top_p = _validate_selected_top_p(selected_top_p)
        arms.extend(
            SamplerArm("top-k", temperature, top_p, top_k)
            for top_k in (32, 64, 128)
        )
    unique: dict[str, SamplerArm] = {}
    for arm in arms:
        unique.setdefault(arm.arm_id, arm)
    return list(unique.values())


def center_out_order(count: int) -> list[int]:
    if count <= 0:
        return []
    order: list[int] = []
    left = (count - 1) // 2
    right = left + 1
    while left >= 0 or right < count:
        if left >= 0:
            order.append(left)
            left -= 1
        if right < count:
            order.append(right)
            right += 1
    return order


def seed_case_order(seed_index: int, case_count: int) -> list[int]:
    if seed_index == 0:
        return list(range(case_count))
    if seed_index == 1:
        return list(reversed(range(case_count)))
    return center_out_order(case_count)


def build_engine(case: Mapping[str, Any], arm: SamplerArm) -> CustomLocalGemmaTranslation:
    engine = CustomLocalGemmaTranslation()
    engine.api_base_url = DEFAULT_GEMMA_LOCAL_ENDPOINT
    engine.model = DEFAULT_GEMMA_LOCAL_MODEL
    engine.source_lang = "Japanese" if case.get("language_pair") == "ja-ko" else "English"
    engine.target_lang = "Korean"
    engine.chunk_size = 6
    engine.max_tokens = DEFAULT_GEMMA_MAX_COMPLETION_TOKENS
    engine.temperature = arm.temperature
    engine.top_k = arm.top_k
    engine.top_p = arm.top_p
    engine.min_p = arm.min_p
    engine.prompt_profile = DEFAULT_GEMMA_PROMPT_PROFILE
    engine.response_format_mode = DEFAULT_GEMMA_RESPONSE_FORMAT_MODE
    engine.response_schema_mode = DEFAULT_GEMMA_RESPONSE_SCHEMA_MODE
    engine.think_briefly_prompt = DEFAULT_GEMMA_THINK_BRIEFLY_PROMPT
    engine.request_mode = "contextual-single"
    return engine


def build_case_payload(case: Mapping[str, Any], arm: SamplerArm, seed: int) -> tuple[dict[str, Any], list[str]]:
    context_before = [str(item) for item in case.get("context_before", [])]
    context_after = [str(item) for item in case.get("context_after", [])]
    texts = [*context_before, str(case["source_text"]), *context_after]
    target_index = len(context_before)
    engine = build_engine(case, arm)
    payload, expected_keys = build_single_block_translation_request(
        engine,
        texts,
        target_index,
    )
    payload["seed"] = int(seed)
    return payload, expected_keys


def request_contract_hash(payload: Mapping[str, Any]) -> str:
    base = {key: value for key, value in payload.items() if key not in SAMPLER_KEYS}
    return canonical_sha256(base)


def response_key(manifest_sha256: str, case_id: str, arm: SamplerArm, seed: int) -> str:
    return canonical_sha256(
        {
            "protocol_version": "gemma-sampler-stability-v1",
            "manifest_sha256": manifest_sha256,
            "case_id": case_id,
            "arm_id": arm.arm_id,
            "seed": int(seed),
        }
    )[:32]


def _private_path(path: Path) -> Path:
    resolved = path.resolve()
    archive = ARCHIVE_ROOT.resolve()
    if resolved == archive:
        raise ArtifactHarnessError("A run output must be below banchmark_result_log, not the archive root")
    try:
        resolved.relative_to(archive)
    except ValueError as exc:
        raise ArtifactHarnessError(
            f"Sampler artifacts must stay under {ARCHIVE_ROOT}: {resolved}"
        ) from exc
    return resolved


def choose_output_directory(explicit: Path | None) -> tuple[Path, ManagedArtifactRun | None]:
    if explicit is not None:
        output = _private_path(explicit)
        output.mkdir(parents=True, exist_ok=True)
        return output, None
    output, owner = select_managed_output_directory(
        family=ARTIFACT_FAMILY,
        category=ARTIFACT_CATEGORY,
    )
    output.mkdir(parents=True, exist_ok=True)
    return output, owner


def atomic_json_write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    encoded = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _json_body(raw: bytes) -> Any:
    decoded = raw.decode("utf-8", errors="replace")
    try:
        return json.loads(decoded) if decoded else {}
    except json.JSONDecodeError:
        return decoded


def post_json(url: str, payload: Mapping[str, Any], *, timeout_sec: int) -> tuple[int, Any, float]:
    data = json.dumps(dict(payload), ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Accept": "application/json", "Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout_sec) as response:
            body = _json_body(response.read())
            return int(response.status), body, time.perf_counter() - started
    except urllib.error.HTTPError as exc:
        return int(exc.code), _json_body(exc.read()), time.perf_counter() - started
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise SamplerRunError(f"Gemma request failed: {exc}") from exc


def validate_response(
    body: Any,
    *,
    expected_keys: Iterable[str],
) -> dict[str, Any]:
    expected = [str(key) for key in expected_keys]
    failures: list[str] = []
    translation: str | None = None
    finish_reason: Any = None
    content = ""
    cleaned_content = ""
    usage: Mapping[str, Any] = {}
    if not isinstance(body, dict):
        failures.append("response_not_object")
    else:
        choices = body.get("choices")
        if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
            failures.append("choice_count")
        else:
            choice = choices[0]
            finish_reason = choice.get("finish_reason")
            if finish_reason != "stop":
                failures.append(f"finish_reason:{finish_reason}")
            message = choice.get("message")
            if not isinstance(message, dict) or not isinstance(message.get("content"), str):
                failures.append("content_missing")
            else:
                content = str(message["content"])
                cleaned_content = CHANNEL_TOKEN_RE.sub("", content).strip()
            usage_value = body.get("usage")
            if isinstance(usage_value, dict):
                usage = usage_value
        if not content:
            failures.append("empty_content")
        else:
            try:
                parsed = json.loads(cleaned_content)
            except json.JSONDecodeError:
                parsed = None
                failures.append("json_parse")
            if not isinstance(parsed, dict):
                failures.append("schema_object")
            else:
                actual_keys = [str(key) for key in parsed.keys()]
                if actual_keys != expected:
                    failures.append("schema_order_or_count")
                if actual_keys == expected:
                    value = parsed.get(expected[0])
                    if not isinstance(value, str) or not value.strip():
                        failures.append("translation_value")
                    else:
                        translation = value
    return {
        "structural_status": "pass" if not failures else "fail",
        "structural_failures": failures,
        "finish_reason": finish_reason,
        "content_hash": text_sha256(content),
        "content_length": len(content),
        "channel_token_sanitized": cleaned_content != content.strip(),
        "translation": translation,
        "translation_hash": text_sha256(translation) if translation is not None else "",
        "prompt_tokens": int(usage.get("prompt_tokens", 0) or 0),
        "completion_tokens": int(usage.get("completion_tokens", 0) or 0),
        "total_tokens": int(usage.get("total_tokens", 0) or 0),
    }


def _source_numbers(value: str) -> list[str]:
    return re.findall(r"\d+(?:[,.]\d+)*", str(value or ""))


def automatic_quality(case: Mapping[str, Any], validation: Mapping[str, Any]) -> dict[str, Any]:
    if validation.get("structural_status") != "pass":
        return {
            "status": "fail",
            "hard_failures": ["json_finish_reason_schema_order_or_count_error"],
            "review_required": [],
            "naturalness": None,
        }
    translation = str(validation.get("translation") or "")
    hard_failures: list[str] = []
    if MASKING_RE.search(translation):
        hard_failures.append("censorship_or_deletion")
    source_numbers = _source_numbers(str(case.get("source_text") or ""))
    if source_numbers and any(number not in translation for number in source_numbers):
        hard_failures.append("number_change")
    if hard_failures:
        return {
            "status": "fail",
            "hard_failures": hard_failures,
            "review_required": [],
            "naturalness": None,
        }
    canonical = str(case.get("canonical_ko") or "").strip()
    if translation.strip() == canonical:
        return {
            "status": "pass",
            "hard_failures": [],
            "review_required": [],
            "naturalness": 5,
        }
    return {
        "status": "review_required",
        "hard_failures": [],
        "review_required": ["unique_output_requires_manual_semantic_review"],
        "naturalness": None,
    }


def load_judgments(path: Path | None) -> dict[tuple[str, str, str], dict[str, Any]]:
    if path is None:
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SamplerProtocolError(f"Unable to read judgment file: {path}") from exc
    rows = payload.get("judgments") if isinstance(payload, dict) else None
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != "gemma-sampler-judgment-v1"
        or not isinstance(rows, list)
    ):
        raise SamplerProtocolError("Judgment file schema must be gemma-sampler-judgment-v1")
    result: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise SamplerProtocolError("Every judgment must be an object")
        key = (
            str(row.get("case_id") or ""),
            str(row.get("arm_id") or ""),
            str(row.get("translation_hash") or ""),
        )
        if not all(key) or key in result:
            raise SamplerProtocolError("Judgment key is empty or duplicated")
        if row.get("status") not in {"pass", "fail", "review_required"}:
            raise SamplerProtocolError(f"Invalid judgment status for {key[0]}")
        if "naturalness" in row and row["naturalness"] is not None:
            score = int(row["naturalness"])
            if score < 1 or score > 5:
                raise SamplerProtocolError("Judgment naturalness must be 1..5")
        result[key] = dict(row)
    return result


def apply_quality_judgment(
    case: Mapping[str, Any],
    arm_id: str,
    validation: Mapping[str, Any],
    judgments: Mapping[tuple[str, str, str], Mapping[str, Any]],
) -> dict[str, Any]:
    automatic = automatic_quality(case, validation)
    translation_hash = str(validation.get("translation_hash") or "")
    manual = judgments.get((str(case.get("case_id")), arm_id, translation_hash))
    if manual is not None and not automatic.get("hard_failures"):
        return {
            "status": str(manual["status"]),
            "hard_failures": list(manual.get("hard_failures") or []),
            "review_required": list(manual.get("review_required") or []),
            "naturalness": manual.get("naturalness"),
            "judgment_source": "manual",
            "note": str(manual.get("note") or ""),
        }
    automatic["judgment_source"] = "automatic"
    return automatic


def response_path(output_dir: Path, response_id: str) -> Path:
    return output_dir / "responses" / f"{response_id}.json"


def read_completed_response(path: Path, *, expected_contract_hash: str, expected_response_id: str) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    if (
        payload.get("response_id") != expected_response_id
        or payload.get("contract_hash") != expected_contract_hash
        or payload.get("run_status") != "complete"
    ):
        return None
    return payload


def make_response_record(
    *,
    response_id: str,
    contract_hash: str,
    case: Mapping[str, Any],
    arm: SamplerArm,
    seed: int,
    payload: Mapping[str, Any],
    expected_keys: list[str],
    http_status: int,
    body: Any,
    elapsed_sec: float,
    judgments: Mapping[tuple[str, str, str], Mapping[str, Any]],
) -> dict[str, Any]:
    validation = validate_response(body, expected_keys=expected_keys)
    quality = apply_quality_judgment(case, arm.arm_id, validation, judgments)
    if http_status < 200 or http_status >= 300:
        validation["structural_status"] = "fail"
        validation["structural_failures"] = [*validation["structural_failures"], f"http_status:{http_status}"]
        quality = {
            "status": "fail",
            "hard_failures": ["json_finish_reason_schema_order_or_count_error"],
            "review_required": [],
            "naturalness": None,
            "judgment_source": "automatic",
        }
    return {
        "schema_version": "gemma-sampler-response-v1",
        "run_status": "complete",
        "response_id": response_id,
        "contract_hash": contract_hash,
        "case_id": str(case["case_id"]),
        "arm_id": arm.arm_id,
        "phase": arm.phase,
        "seed": int(seed),
        "sampler": arm.sampler_values(seed),
        "request_payload": dict(payload),
        "http_status": int(http_status),
        "response_payload": body,
        "elapsed_sec": round(float(elapsed_sec), 6),
        "validation": validation,
        "quality": quality,
    }


def iter_jobs(cases: list[dict[str, Any]], arms: list[SamplerArm]) -> Iterable[tuple[dict[str, Any], SamplerArm, int]]:
    for arm in arms:
        for seed_index, seed in enumerate(DEFAULT_SEEDS):
            order = seed_case_order(seed_index, len(cases))
            for case_index in order:
                yield cases[case_index], arm, int(seed)


def summarize_run(
    *,
    output_dir: Path,
    protocol: Mapping[str, Any],
    manifest: Mapping[str, Any],
    manifest_sha256: str,
    arms: list[SamplerArm],
    judgments: Mapping[tuple[str, str, str], Mapping[str, Any]],
) -> dict[str, Any]:
    cases = [case for case in manifest["cases"] if isinstance(case, dict)]
    cases_by_id = {str(case.get("case_id") or ""): case for case in cases}
    by_arm: dict[str, list[dict[str, Any]]] = {arm.arm_id: [] for arm in arms}
    response_files = sorted((output_dir / "responses").glob("*.json"))
    for path in response_files:
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(row, dict) or row.get("manifest_sha256") not in {None, manifest_sha256}:
            continue
        arm_id = str(row.get("arm_id") or "")
        if arm_id in by_arm:
            by_arm[arm_id].append(row)

    arm_summaries: list[dict[str, Any]] = []
    expected_per_arm = len(cases) * len(DEFAULT_SEEDS)
    for arm in arms:
        rows = by_arm[arm.arm_id]
        complete = [row for row in rows if row.get("run_status") == "complete"]
        structural_failures = [
            row for row in complete
            if ((row.get("validation") or {}).get("structural_status") != "pass")
        ]
        for row in complete:
            case = cases_by_id.get(str(row.get("case_id") or ""))
            if case is not None:
                row["quality"] = apply_quality_judgment(
                    case,
                    arm.arm_id,
                    row.get("validation") or {},
                    judgments,
                )
        clusters: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for row in complete:
            validation = row.get("validation") or {}
            output_hash = str(validation.get("translation_hash") or validation.get("content_hash") or "")
            clusters.setdefault((str(row.get("case_id") or ""), output_hash), []).append(row)
        unique_cluster_rows: list[dict[str, Any]] = []
        hard_failures = 0
        review_required = 0
        naturalness_values: list[int] = []
        latency_values: list[float] = []
        completion_token_values: list[int] = []
        for row in complete:
            elapsed = row.get("elapsed_sec")
            if isinstance(elapsed, (int, float)):
                latency_values.append(float(elapsed))
            completion_tokens = (row.get("validation") or {}).get("completion_tokens")
            if isinstance(completion_tokens, int):
                completion_token_values.append(int(completion_tokens))
        for (case_id, output_hash), cluster_rows in sorted(clusters.items()):
            quality_values = [row.get("quality") or {} for row in cluster_rows]
            statuses = {str(value.get("status") or "") for value in quality_values}
            status = "fail" if "fail" in statuses else "review_required" if "review_required" in statuses else "pass"
            hard = sorted({str(item) for value in quality_values for item in value.get("hard_failures", [])})
            review = sorted({str(item) for value in quality_values for item in value.get("review_required", [])})
            if hard:
                hard_failures += 1
            if status == "review_required":
                review_required += 1
            for value in quality_values:
                if isinstance(value.get("naturalness"), int):
                    naturalness_values.append(int(value["naturalness"]))
            unique_cluster_rows.append(
                {
                    "case_id": case_id,
                    "translation_hash": output_hash,
                    "seed_count": len(cluster_rows),
                    "status": status,
                    "hard_failure_count": len(hard),
                    "review_required_count": len(review),
                }
            )
        stable_cases = 0
        covered_cases = 0
        for case in cases:
            case_rows = [row for row in complete if row.get("case_id") == case.get("case_id")]
            hashes = {
                str((row.get("validation") or {}).get("translation_hash") or "")
                for row in case_rows
                if (row.get("validation") or {}).get("structural_status") == "pass"
            }
            if len(case_rows) == len(DEFAULT_SEEDS):
                covered_cases += 1
                if len(hashes) == 1:
                    stable_cases += 1
        arm_summaries.append(
            {
                "arm_id": arm.arm_id,
                "phase": arm.phase,
                "sampler": {
                    "temperature": arm.temperature,
                    "top_p": arm.top_p,
                    "top_k": arm.top_k,
                    "min_p": arm.min_p,
                },
                "planned_responses": expected_per_arm,
                "completed_responses": len(complete),
                "structural_failure_responses": len(structural_failures),
                "channel_token_sanitized_responses": sum(
                    1
                    for row in complete
                    if bool((row.get("validation") or {}).get("channel_token_sanitized"))
                ),
                "unique_output_clusters": len(unique_cluster_rows),
                "hard_failure_clusters": hard_failures,
                "review_required_clusters": review_required,
                "seed_stable_case_count": stable_cases,
                "seed_covered_case_count": covered_cases,
                "naturalness_mean": round(sum(naturalness_values) / len(naturalness_values), 3) if naturalness_values else None,
                "latency_mean_sec": round(sum(latency_values) / len(latency_values), 6) if latency_values else None,
                "completion_tokens_mean": round(
                    sum(completion_token_values) / len(completion_token_values),
                    3,
                )
                if completion_token_values
                else None,
                "clean_candidate": bool(
                    len(complete) == expected_per_arm
                    and not structural_failures
                    and hard_failures == 0
                    and review_required == 0
                ),
                "clusters": unique_cluster_rows,
            }
        )
    clean_arms = [row for row in arm_summaries if row["clean_candidate"]]
    ranked = sorted(
        arm_summaries,
        key=lambda row: (
            0 if row["clean_candidate"] else 1,
            int(row["hard_failure_clusters"]),
            int(row["review_required_clusters"]),
            -int(row["seed_stable_case_count"]),
            -(float(row["naturalness_mean"]) if row["naturalness_mean"] is not None else -1.0),
            float(row["latency_mean_sec"]) if row["latency_mean_sec"] is not None else float("inf"),
            float(row["completion_tokens_mean"])
            if row["completion_tokens_mean"] is not None
            else float("inf"),
        ),
    )
    return {
        "schema_version": "gemma-sampler-report-v1",
        "protocol_version": protocol["protocol_version"],
        "manifest_sha256": manifest_sha256,
        "case_counts": {
            "total": len(cases),
            "ja-ko": sum(1 for case in cases if case.get("language_pair") == "ja-ko"),
            "en-ko": sum(1 for case in cases if case.get("language_pair") == "en-ko"),
        },
        "seed_values": list(DEFAULT_SEEDS),
        "seed_case_orders": ["forward", "reverse", "center-out"],
        "response_files_seen": len(response_files),
        "clean_candidate_count": len(clean_arms),
        "automatic_product_apply_allowed": False,
        "winner": ranked[0]["arm_id"] if clean_arms and ranked else None,
        "arm_summaries": arm_summaries,
        "ranked_arm_ids": [row["arm_id"] for row in ranked],
    }


class _LabSettingsPage:
    """Small settings surface used only to exercise the product Router contract."""

    def get_tool_selection(self, _kind: str) -> str:
        return "Custom Local Server(Gemma)"

    def get_credentials(self, _tool: str) -> dict[str, str]:
        return {
            "api_url": DEFAULT_GEMMA_LOCAL_ENDPOINT,
            "model": DEFAULT_GEMMA_LOCAL_MODEL,
        }

    def get_paddleocr_vl_settings(self) -> dict[str, str]:
        return {"server_url": "http://127.0.0.1:18000/v1/chat/completions"}

    def get_paddleocr_vl_spotting_settings(self) -> dict[str, str]:
        return {"server_url": "http://127.0.0.1:18002/v1/chat/completions"}


def prepare_crop_router_for_sampler() -> tuple[LocalLlamaRouterCoordinator, dict[str, Any]]:
    coordinator = LocalLlamaRouterCoordinator()
    settings = _LabSettingsPage()
    ocr_manager = LocalOCRRuntimeManager(coordinator)
    gemma_manager = LocalGemmaRuntimeManager(coordinator)
    ocr_identity = ocr_manager._router_runtime_identity("PaddleOCR VL")
    gemma_identity = gemma_manager.router_runtime_identity(settings)
    if not isinstance(ocr_identity, dict) or not isinstance(gemma_identity, dict):
        raise SamplerRunError("Product Paddle/Gemma runtime identities could not be resolved")
    try:
        coordinator.ensure_ocr_model("PaddleOCR VL", settings, ocr_identity)
        coordinator.unload_model(str(ocr_identity["model_name"]))
        coordinator.ensure_gemma_model(settings, gemma_identity)
        snapshot = coordinator.snapshot()
        if (
            snapshot.pair != "paddle-crop"
            or snapshot.active_model != ROUTER_GEMMA_ALIAS
            or snapshot.loaded_count != 1
            or not snapshot.container_running
        ):
            raise SamplerRunError(f"Crop Router did not reach one-loaded-Gemma state: {snapshot}")
        return coordinator, {
            "pair": snapshot.pair,
            "generation": snapshot.generation,
            "active_model": snapshot.active_model,
            "loaded_count": snapshot.loaded_count,
            "container_running": snapshot.container_running,
            "fingerprint": snapshot.fingerprint,
            "ocr_model": ocr_identity["model_name"],
            "gemma_model": gemma_identity["model_name"],
            "gemma_volume": gemma_identity.get("volume", DEFAULT_GEMMA_MODEL_VOLUME),
        }
    except Exception:
        try:
            coordinator.stop_pair()
        except Exception:
            pass
        raise


def cleanup_router(coordinator: LocalLlamaRouterCoordinator | None) -> dict[str, Any]:
    if coordinator is None:
        return {"runtime_state": "not-started"}
    try:
        result = coordinator.stop_pair()
    except (RouterRuntimeError, OSError, RuntimeError) as exc:
        raise SamplerRunError(f"Crop Router cleanup failed: {exc}") from exc
    snapshot = coordinator.snapshot()
    if snapshot.loaded_count or snapshot.container_running:
        raise SamplerRunError(f"Crop Router cleanup did not prove release: {snapshot}")
    return {
        "result": result,
        "generation": snapshot.generation,
        "loaded_count": snapshot.loaded_count,
        "container_running": snapshot.container_running,
        "release_failed": snapshot.release_failed,
    }


def _write_plan(
    output_dir: Path,
    *,
    protocol: Mapping[str, Any],
    manifest_sha256: str,
    arms: list[SamplerArm],
    cases: list[dict[str, Any]],
) -> None:
    plan = {
        "schema_version": "gemma-sampler-plan-v1",
        "protocol_version": protocol["protocol_version"],
        "manifest_sha256": manifest_sha256,
        "case_count": len(cases),
        "arm_count": len(arms),
        "planned_response_count": len(arms) * len(cases) * len(DEFAULT_SEEDS),
        "arms": [
            {
                "arm_id": arm.arm_id,
                "phase": arm.phase,
                "sampler": {
                    "temperature": arm.temperature,
                    "top_p": arm.top_p,
                    "top_k": arm.top_k,
                    "min_p": arm.min_p,
                },
            }
            for arm in arms
        ],
        "seed_values": list(DEFAULT_SEEDS),
        "seed_case_orders": [
            [seed, seed_case_order(index, len(cases))]
            for index, seed in enumerate(DEFAULT_SEEDS)
        ],
    }
    atomic_json_write(output_dir / "plan.json", plan)


def run_protocol(args: argparse.Namespace) -> dict[str, Any]:
    protocol = load_protocol()
    manifest, manifest_sha256 = load_corpus_manifest(Path(args.corpus_manifest))
    arms = build_arms(
        args.phase,
        selected_temperature=args.selected_temperature,
        selected_top_p=args.selected_top_p,
    )
    output_dir, owner = choose_output_directory(Path(args.output_dir) if args.output_dir else None)
    cases = [case for case in manifest["cases"] if isinstance(case, dict)]
    judgments = load_judgments(Path(args.judgments) if args.judgments else None)
    _write_plan(output_dir, protocol=protocol, manifest_sha256=manifest_sha256, arms=arms, cases=cases)

    if args.report_only:
        previous_summary: dict[str, Any] = {}
        try:
            previous_payload = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            previous_payload = None
        if isinstance(previous_payload, dict):
            previous_summary = previous_payload
        summary = summarize_run(
            output_dir=output_dir,
            protocol=protocol,
            manifest=manifest,
            manifest_sha256=manifest_sha256,
            arms=arms,
            judgments=judgments,
        )
        for key in (
            "planned_in_this_invocation",
            "executed_in_this_invocation",
            "reused_in_this_invocation",
            "router_runtime",
            "router_cleanup",
        ):
            if key in previous_summary:
                summary[key] = previous_summary[key]
        for filename, key in (
            ("router-runtime.json", "router_runtime"),
            ("router-cleanup.json", "router_cleanup"),
        ):
            if key in summary:
                continue
            try:
                runtime_payload = json.loads((output_dir / filename).read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                runtime_payload = None
            if isinstance(runtime_payload, dict):
                summary[key] = runtime_payload
        atomic_json_write(output_dir / "summary.json", summary)
        if owner is not None:
            owner.complete(metadata={
                "protocol_version": protocol["protocol_version"],
                "manifest_sha256": manifest_sha256,
                "response_count": summary["response_files_seen"],
                "clean_candidate_count": summary["clean_candidate_count"],
            })
        return summary

    if args.dry_run:
        summary = {
            "schema_version": "gemma-sampler-dry-run-v1",
            "protocol_version": protocol["protocol_version"],
            "manifest_sha256": manifest_sha256,
            "planned_response_count": len(arms) * len(cases) * len(DEFAULT_SEEDS),
            "output_dir": str(output_dir),
        }
        atomic_json_write(output_dir / "summary.json", summary)
        if owner is not None:
            owner.complete(metadata={
                "protocol_version": protocol["protocol_version"],
                "manifest_sha256": manifest_sha256,
                "response_count": 0,
                "clean_candidate_count": 0,
            })
        return summary

    coordinator: LocalLlamaRouterCoordinator | None = None
    runtime_metadata: dict[str, Any] = {}
    summary: dict[str, Any] | None = None
    try:
        coordinator, runtime_metadata = prepare_crop_router_for_sampler()
        atomic_json_write(output_dir / "router-runtime.json", runtime_metadata)
        planned = 0
        executed = 0
        reused = 0
        for case, arm, seed in iter_jobs(cases, arms):
            if args.limit is not None and planned >= int(args.limit):
                break
            planned += 1
            payload, expected_keys = build_case_payload(case, arm, seed)
            contract_hash = request_contract_hash(payload)
            response_id = response_key(manifest_sha256, str(case["case_id"]), arm, seed)
            path = response_path(output_dir, response_id)
            if args.resume and read_completed_response(
                path,
                expected_contract_hash=contract_hash,
                expected_response_id=response_id,
            ) is not None:
                reused += 1
                continue
            try:
                http_status, body, elapsed_sec = post_json(
                    DEFAULT_GEMMA_LOCAL_ENDPOINT.rstrip("/") + "/chat/completions",
                    payload,
                    timeout_sec=int(args.request_timeout_sec),
                )
            except SamplerRunError as exc:
                atomic_json_write(
                    path,
                    {
                        "schema_version": "gemma-sampler-response-v1",
                        "run_status": "incomplete",
                        "response_id": response_id,
                        "contract_hash": contract_hash,
                        "case_id": case["case_id"],
                        "arm_id": arm.arm_id,
                        "seed": seed,
                        "error": str(exc),
                    },
                )
                raise
            record = make_response_record(
                response_id=response_id,
                contract_hash=contract_hash,
                case=case,
                arm=arm,
                seed=seed,
                payload=payload,
                expected_keys=expected_keys,
                http_status=http_status,
                body=body,
                elapsed_sec=elapsed_sec,
                judgments=judgments,
            )
            record["manifest_sha256"] = manifest_sha256
            atomic_json_write(path, record)
            executed += 1
        summary = summarize_run(
            output_dir=output_dir,
            protocol=protocol,
            manifest=manifest,
            manifest_sha256=manifest_sha256,
            arms=arms,
            judgments=judgments,
        )
        summary["planned_in_this_invocation"] = planned
        summary["executed_in_this_invocation"] = executed
        summary["reused_in_this_invocation"] = reused
        summary["router_runtime"] = runtime_metadata
    finally:
        if coordinator is not None:
            cleanup = cleanup_router(coordinator)
            atomic_json_write(output_dir / "router-cleanup.json", cleanup)
            if summary is not None:
                summary["router_cleanup"] = cleanup
                atomic_json_write(output_dir / "summary.json", summary)
    if summary is None:
        raise SamplerRunError("Sampler run ended without a summary")
    if owner is not None:
        owner.complete(metadata={
            "protocol_version": protocol["protocol_version"],
            "manifest_sha256": manifest_sha256,
            "response_count": summary["response_files_seen"],
            "clean_candidate_count": summary["clean_candidate_count"],
        })
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--phase", choices=("temperature", "top-p", "top-k", "all"), default="temperature")
    parser.add_argument("--selected-temperature", type=float)
    parser.add_argument("--selected-top-p", type=float)
    parser.add_argument("--judgments", type=Path)
    parser.add_argument("--request-timeout-sec", type=int, default=DEFAULT_TIMEOUT_SEC)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        summary = run_protocol(args)
    except (ArtifactHarnessError, SamplerProtocolError, SamplerRunError, OSError) as exc:
        print(f"sampler protocol failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

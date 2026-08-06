"""HunyuanOCR용 관리형 llama.cpp 런타임 계약.

다른 관리형 엔진과 같은 규약을 따른다. 즉 모델은 SHA-256으로 검증된 external
volume에 준비되고, 그 volume은 스모크 테스트를 통과한 ready manifest를 담으며,
런타임은 지원 목록에 있는 llama.cpp CUDA 서버 이미지로만 뜬다. 이 계약이 있어야
HunyuanOCR이 Router pair가 될 수 있다. Router는 OCR과 Gemma가 동일한
이미지를 쓰고 양쪽 모델이 검증된 volume에 있다는 증거를 요구한다.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from modules.utils.llama_cpp_runtime import (
    DEFAULT_LLAMA_CPP_IMAGE,
    is_supported_llama_cpp_image,
)


HUNYUAN_OCR_RUNTIME_MANIFEST_SCHEMA_VERSION = 1
HUNYUAN_OCR_RUNTIME_PREPARATION_VERSION = 1
DEFAULT_HUNYUAN_OCR_MODEL_VOLUME = "comic-translate-hunyuanocr-models-v2"
DEFAULT_HUNYUAN_OCR_READY_MANIFEST = ".comic-translate-hunyuanocr-ready-v1.json"
DEFAULT_HUNYUAN_OCR_LLAMA_CPP_IMAGE = DEFAULT_LLAMA_CPP_IMAGE

HUNYUAN_OCR_MODEL_NAME = "HunyuanOCR.Q8_0.gguf"
HUNYUAN_OCR_MMPROJ_NAME = "HunyuanOCR.mmproj-Q8_0.gguf"
# Router는 요청의 model 필드로만 대상을 고르고, HunyuanOCR 엔진은 파일명을 그대로
# 보낸다. 따라서 alias·preset 섹션명·엔진이 보내는 값이 모두 이 값이어야 한다.
HUNYUAN_OCR_MODEL_ALIAS = HUNYUAN_OCR_MODEL_NAME
HUNYUAN_OCR_MODEL_SPECS: dict[str, dict[str, Any]] = {
    HUNYUAN_OCR_MODEL_NAME: {
        "bytes": 577_949_408,
        "sha256": "cdafc794cafeae377868d7a40a70e282a737e39abe77c0d8b73614447b364a21",
        "role": "vlm",
    },
    HUNYUAN_OCR_MMPROJ_NAME: {
        "bytes": 732_938_240,
        "sha256": "b77913164ff73d4c0dc4d994e236ed72bacbbe5c5db1ec9b2828627b46c32804",
        "role": "vision-projector",
    },
}

# 과거 Compose는 Gemma와 같은 일반 이름(`LLAMA_CTX_SIZE` 등)을 읽어서 한쪽을
# 조정하면 다른 쪽까지 바뀌었다. 준비 볼륨 계약에서는 엔진별로 이름을 분리한다.
DEFAULT_HUNYUAN_OCR_RUNTIME_OPTIONS: dict[str, str] = {
    "HUNYUAN_OCR_LLAMA_CTX_SIZE": "4096",
    "HUNYUAN_OCR_LLAMA_PARALLEL": "1",
    "HUNYUAN_OCR_LLAMA_THREADS": "12",
    # 과거 런타임 매니저가 LLAMA_N_GPU_LAYERS=80을 주입했으므로 Compose 기본값
    # 99가 아니라 실제로 동작해 온 80이 이 계약의 기본값이다.
    "HUNYUAN_OCR_LLAMA_GPU_LAYERS": "80",
    "HUNYUAN_OCR_LLAMA_IMAGE_MAX_TOKENS": "1024",
}

HUNYUAN_OCR_RUNTIME_FINGERPRINT_LABEL = (
    "com.comictranslate.hunyuanocr-runtime-fingerprint"
)
HUNYUAN_OCR_RUNTIME_VOLUME_LABEL = "com.comictranslate.hunyuanocr-model-volume"
HUNYUAN_OCR_RUNTIME_MANIFEST_SHA_LABEL = (
    "com.comictranslate.hunyuanocr-ready-manifest-sha256"
)
HUNYUAN_OCR_RUNTIME_MODEL_SHA_LABEL = (
    "com.comictranslate.hunyuanocr-model-sha256"
)
HUNYUAN_OCR_RUNTIME_MMPROJ_SHA_LABEL = (
    "com.comictranslate.hunyuanocr-mmproj-sha256"
)

_SAFE_DOCKER_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class HunyuanOCRRuntimeContractError(ValueError):
    pass


@dataclass(frozen=True)
class HunyuanOCRRuntimeContract:
    volume_name: str
    ready_manifest_name: str
    ready_manifest_sha256: str
    preparation_version: int
    llama_image_ref: str
    llama_image_id: str
    compose_file_sha256: str
    command_sha256: str
    fingerprint: str
    command: tuple[str, ...]
    runtime_options: Mapping[str, str]

    def compose_environment(self) -> dict[str, str]:
        values = dict(self.runtime_options)
        values.update(
            {
                "HUNYUAN_OCR_LLAMA_CPP_IMAGE": self.llama_image_ref,
                "HUNYUAN_OCR_MODEL_VOLUME": self.volume_name,
                "HUNYUAN_OCR_MODEL_FILE": HUNYUAN_OCR_MODEL_NAME,
                "HUNYUAN_OCR_MMPROJ_FILE": HUNYUAN_OCR_MMPROJ_NAME,
                "HUNYUAN_OCR_MODEL_ALIAS": HUNYUAN_OCR_MODEL_ALIAS,
                "HUNYUAN_OCR_RUNTIME_FINGERPRINT": self.fingerprint,
                "HUNYUAN_OCR_READY_MANIFEST_SHA256": self.ready_manifest_sha256,
                "HUNYUAN_OCR_MODEL_SHA256": str(
                    HUNYUAN_OCR_MODEL_SPECS[HUNYUAN_OCR_MODEL_NAME]["sha256"]
                ),
                "HUNYUAN_OCR_MMPROJ_SHA256": str(
                    HUNYUAN_OCR_MODEL_SPECS[HUNYUAN_OCR_MMPROJ_NAME]["sha256"]
                ),
                "HUNYUAN_OCR_PREPARATION_VERSION": str(self.preparation_version),
            }
        )
        return values


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_hunyuan_ocr_volume_name(value: str) -> str:
    normalized = str(value or "").strip()
    if not _SAFE_DOCKER_NAME.fullmatch(normalized):
        raise HunyuanOCRRuntimeContractError(
            f"HunyuanOCR 모델 volume 이름이 올바르지 않습니다: {value!r}"
        )
    return normalized


def resolve_hunyuan_ocr_runtime_options(
    environment: Mapping[str, Any] | None = None,
) -> dict[str, str]:
    source = environment or {}
    values: dict[str, str] = {}
    for key, default in DEFAULT_HUNYUAN_OCR_RUNTIME_OPTIONS.items():
        raw_value = source.get(key, default)
        normalized = str(raw_value).strip() if raw_value is not None else ""
        values[key] = normalized or default

    numeric_ranges = {
        "HUNYUAN_OCR_LLAMA_CTX_SIZE": (2048, 16_384),
        "HUNYUAN_OCR_LLAMA_PARALLEL": (1, 2),
        "HUNYUAN_OCR_LLAMA_THREADS": (1, 64),
        "HUNYUAN_OCR_LLAMA_IMAGE_MAX_TOKENS": (256, 8192),
    }
    for key, (minimum, maximum) in numeric_ranges.items():
        try:
            parsed = int(values[key])
        except ValueError as exc:
            raise HunyuanOCRRuntimeContractError(
                f"{key}는 정수여야 합니다."
            ) from exc
        if parsed < minimum or parsed > maximum:
            raise HunyuanOCRRuntimeContractError(
                f"{key}는 {minimum}과 {maximum} 사이여야 합니다: {parsed}"
            )
        values[key] = str(parsed)

    gpu_layers = values["HUNYUAN_OCR_LLAMA_GPU_LAYERS"].lower()
    if gpu_layers != "all":
        try:
            parsed_gpu_layers = int(gpu_layers)
        except ValueError as exc:
            raise HunyuanOCRRuntimeContractError(
                "HUNYUAN_OCR_LLAMA_GPU_LAYERS는 'all' 또는 정수여야 합니다."
            ) from exc
        if parsed_gpu_layers < 0 or parsed_gpu_layers > 999:
            raise HunyuanOCRRuntimeContractError(
                "HUNYUAN_OCR_LLAMA_GPU_LAYERS는 0과 999 사이여야 합니다."
            )
        gpu_layers = str(parsed_gpu_layers)
    values["HUNYUAN_OCR_LLAMA_GPU_LAYERS"] = gpu_layers
    return values


def build_hunyuan_ocr_server_command(
    runtime_options: Mapping[str, str],
) -> tuple[str, ...]:
    return (
        "-m",
        f"/models/{HUNYUAN_OCR_MODEL_NAME}",
        "--mmproj",
        f"/models/{HUNYUAN_OCR_MMPROJ_NAME}",
        "--alias",
        HUNYUAN_OCR_MODEL_ALIAS,
        "--host",
        "0.0.0.0",
        "--port",
        "8080",
        "-c",
        runtime_options["HUNYUAN_OCR_LLAMA_CTX_SIZE"],
        "-np",
        runtime_options["HUNYUAN_OCR_LLAMA_PARALLEL"],
        "-t",
        runtime_options["HUNYUAN_OCR_LLAMA_THREADS"],
        "--cache-ram",
        "0",
        "--n-gpu-layers",
        runtime_options["HUNYUAN_OCR_LLAMA_GPU_LAYERS"],
        "--image-max-tokens",
        runtime_options["HUNYUAN_OCR_LLAMA_IMAGE_MAX_TOKENS"],
    )


def build_hunyuan_ocr_runtime_contract(
    *,
    manifest_bytes: bytes,
    manifest_sha256: str,
    observed_file_bytes: Mapping[str, int],
    volume_name: str,
    llama_image_ref: str,
    llama_image_id: str,
    compose_file: Path,
    environment: Mapping[str, Any] | None = None,
) -> HunyuanOCRRuntimeContract:
    safe_volume = validate_hunyuan_ocr_volume_name(volume_name)
    normalized_manifest_sha = str(manifest_sha256 or "").strip().lower()
    if not _SHA256.fullmatch(normalized_manifest_sha):
        raise HunyuanOCRRuntimeContractError(
            "HunyuanOCR ready manifest SHA-256이 올바르지 않습니다."
        )
    if (
        hashlib.sha256(manifest_bytes).hexdigest().lower()
        != normalized_manifest_sha
    ):
        raise HunyuanOCRRuntimeContractError(
            "HunyuanOCR ready manifest SHA-256이 마운트된 바이트와 다릅니다."
        )
    try:
        manifest = json.loads(manifest_bytes.decode("utf-8-sig"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise HunyuanOCRRuntimeContractError(
            "HunyuanOCR ready manifest를 파싱할 수 없습니다."
        ) from exc
    if not isinstance(manifest, dict):
        raise HunyuanOCRRuntimeContractError(
            "HunyuanOCR ready manifest는 객체여야 합니다."
        )

    required_header = {
        "schema_version": HUNYUAN_OCR_RUNTIME_MANIFEST_SCHEMA_VERSION,
        "runtime": "HunyuanOCR-llama.cpp",
        "preparation_version": HUNYUAN_OCR_RUNTIME_PREPARATION_VERSION,
        "volume_name": safe_volume,
        "ready": True,
        "source_image_id": llama_image_id,
    }
    for key, expected in required_header.items():
        if manifest.get(key) != expected:
            raise HunyuanOCRRuntimeContractError(
                f"HunyuanOCR manifest 항목 {key!r}이 런타임 계약과 다릅니다."
            )
    # CUDA 12 태그로 준비한 volume도 CUDA 13 기본값에서 그대로 통과해야 한다.
    # 실제 이미지 동일성은 위의 source_image_id 비교가 지킨다.
    manifest_image_ref = manifest.get("source_image_ref")
    if manifest_image_ref != llama_image_ref and not is_supported_llama_cpp_image(
        manifest_image_ref
    ):
        raise HunyuanOCRRuntimeContractError(
            "HunyuanOCR manifest 항목 'source_image_ref'가 지원되는 "
            "llama.cpp 이미지가 아닙니다."
        )
    smoke_test = manifest.get("smoke_test")
    if not isinstance(smoke_test, dict) or smoke_test.get("passed") is not True:
        raise HunyuanOCRRuntimeContractError(
            "HunyuanOCR manifest에 통과한 스모크 테스트가 없습니다."
        )
    if (
        smoke_test.get("device") != "CUDA"
        or smoke_test.get("model_alias") != HUNYUAN_OCR_MODEL_ALIAS
    ):
        raise HunyuanOCRRuntimeContractError(
            "HunyuanOCR 스모크 테스트는 CUDA 장치와 정확한 모델 alias를 검증해야 합니다."
        )

    files = manifest.get("files")
    if not isinstance(files, list) or len(files) != len(HUNYUAN_OCR_MODEL_SPECS):
        raise HunyuanOCRRuntimeContractError(
            "HunyuanOCR manifest 파일 목록이 불완전합니다."
        )
    entries = {
        str(entry.get("name", "")): entry
        for entry in files
        if isinstance(entry, dict)
    }
    if set(entries) != set(HUNYUAN_OCR_MODEL_SPECS):
        raise HunyuanOCRRuntimeContractError(
            "HunyuanOCR manifest 파일 목록이 제품 등록 정보와 다릅니다."
        )
    for name, spec in HUNYUAN_OCR_MODEL_SPECS.items():
        entry = entries[name]
        if (
            int(entry.get("bytes", -1)) != int(spec["bytes"])
            or str(entry.get("sha256", "")).lower() != str(spec["sha256"])
            or str(entry.get("role", "")) != str(spec["role"])
            or int(observed_file_bytes.get(name, -1)) != int(spec["bytes"])
        ):
            raise HunyuanOCRRuntimeContractError(
                f"HunyuanOCR 모델 계약 불일치: {name}"
            )

    runtime_options = resolve_hunyuan_ocr_runtime_options(environment)
    command = build_hunyuan_ocr_server_command(runtime_options)
    compose_sha256 = hashlib.sha256(compose_file.read_bytes()).hexdigest()
    command_sha256 = _canonical_sha256(command)
    fingerprint = _canonical_sha256(
        {
            "contract_schema_version": 1,
            "volume_name": safe_volume,
            "ready_manifest_name": DEFAULT_HUNYUAN_OCR_READY_MANIFEST,
            "ready_manifest_sha256": normalized_manifest_sha,
            "llama_image_ref": llama_image_ref,
            "llama_image_id": llama_image_id,
            "compose_file_sha256": compose_sha256,
            "command_sha256": command_sha256,
            "model_registry": HUNYUAN_OCR_MODEL_SPECS,
            "runtime_options": runtime_options,
        }
    )
    return HunyuanOCRRuntimeContract(
        volume_name=safe_volume,
        ready_manifest_name=DEFAULT_HUNYUAN_OCR_READY_MANIFEST,
        ready_manifest_sha256=normalized_manifest_sha,
        preparation_version=HUNYUAN_OCR_RUNTIME_PREPARATION_VERSION,
        llama_image_ref=llama_image_ref,
        llama_image_id=llama_image_id,
        compose_file_sha256=compose_sha256,
        command_sha256=command_sha256,
        fingerprint=fingerprint,
        command=command,
        runtime_options=runtime_options,
    )

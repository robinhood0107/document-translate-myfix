"""Pure contracts for the managed local llama.cpp Router.

The Router must never take over a user supplied endpoint.  For that reason the
candidate check intentionally compares the configured endpoint as an exact
URL, instead of using the legacy managed-runtime URL normalizers that discard
query strings and fragments.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlsplit

from modules.utils.llama_cpp_runtime import DEFAULT_LLAMA_CPP_IMAGE


ROUTER_CONTRACT_SCHEMA_VERSION = 2
ROUTER_OWNER_LABEL = "com.comictranslate.local-llama-router-owner"
ROUTER_PAIR_LABEL = "com.comictranslate.local-llama-router-pair"
ROUTER_FINGERPRINT_LABEL = "com.comictranslate.local-llama-router-fingerprint"
ROUTER_PROJECT_LABEL = "com.docker.compose.project"
ROUTER_OWNER_VALUE = "comic-translate-llamacpp-router-v2"
ROUTER_PROJECT_NAME = "comic-translate-llama-router-v2"
ROUTER_SERVICE_NAME = "llama-router"

DEFAULT_CROP_ROUTER_ENDPOINT = "http://127.0.0.1:18000/v1/chat/completions"
DEFAULT_SPOTTING_ROUTER_ENDPOINT = "http://127.0.0.1:18002/v1/chat/completions"
DEFAULT_HUNYUAN_ROUTER_ENDPOINT = "http://127.0.0.1:28080/v1"
DEFAULT_MANGALMM_ROUTER_ENDPOINT = "http://127.0.0.1:28081/v1"
DEFAULT_GEMMA_ROUTER_ENDPOINT = "http://127.0.0.1:18080/v1"
# 모든 pair가 이 하나의 Gemma 호스트 포트를 publish하므로, 어느 pair의 포트를
# 풀어도 다른 pair가 남긴 Gemma listener까지 함께 회수된다.
ROUTER_GEMMA_HOST_PORT = 18080
DEFAULT_GEMMA_ROUTER_MODEL = "gemma-4-26B-IQ4_NL.gguf"
DEFAULT_ROUTER_IMAGE = DEFAULT_LLAMA_CPP_IMAGE


class RouterPairKind(str, Enum):
    CROP = "crop"
    SPOTTING = "spotting"
    HUNYUAN = "hunyuan"
    MANGALMM = "mangalmm"


@dataclass(frozen=True)
class RouterPair:
    """One mutually exclusive Paddle OCR + Gemma Router deployment."""

    kind: RouterPairKind
    ocr_engine_key: str
    ocr_alias: str
    ocr_endpoint: str
    gemma_endpoint: str
    ocr_port: int
    gemma_port: int
    compose_file: Path
    preset_file: Path
    container_name: str

    @property
    def router_base_url(self) -> str:
        return f"http://127.0.0.1:{self.ocr_port}"

    def payload(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "ocr_engine_key": self.ocr_engine_key,
            "ocr_alias": self.ocr_alias,
            "ocr_endpoint": self.ocr_endpoint,
            "gemma_endpoint": self.gemma_endpoint,
            "ocr_port": self.ocr_port,
            "gemma_port": self.gemma_port,
            "compose_file": self.compose_file.name,
            "preset_file": self.preset_file.name,
            "container_name": self.container_name,
        }


_ROOT_DIR = Path(__file__).resolve().parents[3]
_ROUTER_PAIRS: tuple[RouterPair, ...] = (
    RouterPair(
        kind=RouterPairKind.CROP,
        ocr_engine_key="PaddleOCR VL",
        ocr_alias="PaddleOCR-VL-1.6-0.9B",
        ocr_endpoint=DEFAULT_CROP_ROUTER_ENDPOINT,
        gemma_endpoint=DEFAULT_GEMMA_ROUTER_ENDPOINT,
        ocr_port=18000,
        gemma_port=18080,
        compose_file=(
            _ROOT_DIR / "paddleocr_vl_docker_files" / "docker-compose.router.yaml"
        ),
        preset_file=_ROOT_DIR / "paddleocr_vl_docker_files" / "router-models.ini",
        container_name="comic-translate-router-crop-v2",
    ),
    RouterPair(
        kind=RouterPairKind.SPOTTING,
        ocr_engine_key="PaddleOCR VL Spotting",
        ocr_alias="PaddleOCR-VL-1.6-Spotting",
        ocr_endpoint=DEFAULT_SPOTTING_ROUTER_ENDPOINT,
        gemma_endpoint=DEFAULT_GEMMA_ROUTER_ENDPOINT,
        ocr_port=18002,
        gemma_port=18080,
        compose_file=(
            _ROOT_DIR
            / "paddleocr_vl_spotting_docker_files"
            / "docker-compose.router.yaml"
        ),
        preset_file=(
            _ROOT_DIR / "paddleocr_vl_spotting_docker_files" / "router-models.ini"
        ),
        container_name="comic-translate-router-spotting-v2",
    ),
    RouterPair(
        kind=RouterPairKind.HUNYUAN,
        ocr_engine_key="HunyuanOCR",
        ocr_alias="HunyuanOCR.Q8_0.gguf",
        ocr_endpoint=DEFAULT_HUNYUAN_ROUTER_ENDPOINT,
        gemma_endpoint=DEFAULT_GEMMA_ROUTER_ENDPOINT,
        ocr_port=28080,
        gemma_port=ROUTER_GEMMA_HOST_PORT,
        compose_file=(
            _ROOT_DIR / "hunyuanocr_docker_files" / "docker-compose.router.yaml"
        ),
        preset_file=_ROOT_DIR / "hunyuanocr_docker_files" / "router-models.ini",
        container_name="comic-translate-router-hunyuan-v2",
    ),
    RouterPair(
        kind=RouterPairKind.MANGALMM,
        ocr_engine_key="MangaLMM",
        # 라우터는 명시적 모델명을 요구하고, MangaLMM 엔진은 추론 요청에
        # 파일명을 그대로 보낸다. preset 섹션명·pair alias·엔진이 보내는 값이
        # 모두 같아야 추론이 400으로 거부되지 않는다.
        ocr_alias="MangaLMM.Q8_0.gguf",
        ocr_endpoint=DEFAULT_MANGALMM_ROUTER_ENDPOINT,
        gemma_endpoint=DEFAULT_GEMMA_ROUTER_ENDPOINT,
        ocr_port=28081,
        gemma_port=ROUTER_GEMMA_HOST_PORT,
        compose_file=(
            _ROOT_DIR / "mangalmm_docker_files" / "docker-compose.router.yaml"
        ),
        preset_file=_ROOT_DIR / "mangalmm_docker_files" / "router-models.ini",
        container_name="comic-translate-router-mangalmm-v2",
    ),
)


def canonical_sha256(value: Any) -> str:
    """Return the stable fingerprint used for Router identity and cache keys."""

    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def exact_endpoint_matches(value: Any, expected: str) -> bool:
    """Return true only for the literal managed endpoint contract.

    A direct equality check is intentional: ``?foo``, ``#fragment``, userinfo,
    a trailing slash, another host, and another path are all custom endpoint
    choices and must not be routed through the product-owned container.
    """

    configured = str(value or "").strip()
    if configured != expected:
        return False
    parsed = urlsplit(configured)
    expected_parsed = urlsplit(expected)
    return bool(
        parsed.scheme == "http"
        and parsed.username is None
        and parsed.password is None
        and parsed.query == ""
        and parsed.fragment == ""
        and parsed.hostname == "127.0.0.1"
        and parsed.port == expected_parsed.port
        and parsed.path == expected_parsed.path
    )


def router_pair_for_ocr_endpoint(
    engine_key: Any,
    endpoint: Any,
) -> RouterPair | None:
    """Return the matching Router pair for an exact Paddle endpoint only."""

    normalized_engine = str(engine_key or "").strip()
    for pair in _ROUTER_PAIRS:
        if (
            normalized_engine == pair.ocr_engine_key
            and exact_endpoint_matches(endpoint, pair.ocr_endpoint)
        ):
            return pair
    return None


def router_pair_for_engine_key(engine_key: Any) -> RouterPair | None:
    """endpoint를 보지 않고 OCR 엔진이 속한 Router pair를 반환한다.

    separate-server 경로는 이전 프로세스가 남긴 Router 컨테이너를 회수하기 위해
    pair의 호스트 포트를 알아야 하는데, 이 시점은 어떤 endpoint도 아직 Router
    경로로 분류될 수 없는 단계다.
    """

    normalized_engine = str(engine_key or "").strip()
    if not normalized_engine:
        return None
    for pair in _ROUTER_PAIRS:
        if normalized_engine == pair.ocr_engine_key:
            return pair
    return None


def classify_router_pair(
    engine_key: Any,
    ocr_endpoint: Any,
    gemma_endpoint: Any,
    gemma_model: Any,
) -> RouterPair | None:
    """Classify the complete default Paddle + Gemma combination.

    The configured Gemma model is part of the ownership boundary as well: a
    different model at the default port remains a separate user-managed path.
    """

    pair = router_pair_for_ocr_endpoint(engine_key, ocr_endpoint)
    if pair is None:
        return None
    if not exact_endpoint_matches(gemma_endpoint, pair.gemma_endpoint):
        return None
    if str(gemma_model or "").strip() != DEFAULT_GEMMA_ROUTER_MODEL:
        return None
    return pair


@dataclass(frozen=True)
class RouterModelMaterial:
    """Prepared model/volume evidence reused from the product contracts."""

    alias: str
    model_file: str
    model_sha256: str
    volume_name: str
    ready_manifest_sha256: str
    source_fingerprint: str
    runtime_options: Mapping[str, str]
    preparation_version: int
    mmproj_file: str = ""
    mmproj_sha256: str = ""

    def payload(self) -> dict[str, Any]:
        return {
            "alias": self.alias,
            "model_file": self.model_file,
            "model_sha256": self.model_sha256,
            "mmproj_file": self.mmproj_file,
            "mmproj_sha256": self.mmproj_sha256,
            "volume_name": self.volume_name,
            "ready_manifest_sha256": self.ready_manifest_sha256,
            "source_fingerprint": self.source_fingerprint,
            "runtime_options": dict(sorted(self.runtime_options.items())),
            "preparation_version": self.preparation_version,
        }


@dataclass(frozen=True)
class RouterRuntimeSpec:
    """Inputs that are known before the Router container is started."""

    pair: RouterPair
    ocr_model: RouterModelMaterial
    gemma_model: RouterModelMaterial
    image_ref: str = DEFAULT_ROUTER_IMAGE

    def payload(self) -> dict[str, Any]:
        return {
            "pair": self.pair.payload(),
            "ocr_model": self.ocr_model.payload(),
            "gemma_model": self.gemma_model.payload(),
            "image_ref": self.image_ref,
        }


@dataclass(frozen=True)
class RouterRuntimeContract:
    """Complete fingerprint of the running Router deployment."""

    pair: RouterPair
    fingerprint: str
    image_ref: str
    image_id: str
    repo_digest: str
    entrypoint: tuple[str, ...]
    binary_version: str
    resolved_compose_config: Mapping[str, Any]
    effective_environment: Mapping[str, str]
    port_mapping: Mapping[str, Any]
    volume_mapping: Mapping[str, Any]
    device_mapping: Mapping[str, Any]
    server_args: tuple[str, ...]
    command_sha256: str
    preset_sha256: str
    ocr_model: RouterModelMaterial
    gemma_model: RouterModelMaterial
    ownership_labels: Mapping[str, str]

    def payload(self) -> dict[str, Any]:
        return {
            "contract_schema_version": ROUTER_CONTRACT_SCHEMA_VERSION,
            "pair": self.pair.payload(),
            "image": {
                "ref": self.image_ref,
                "id": self.image_id,
                "repo_digest": self.repo_digest,
                "entrypoint": list(self.entrypoint),
                "binary_version": self.binary_version,
            },
            "resolved_compose_config": self.resolved_compose_config,
            "effective_environment": dict(sorted(self.effective_environment.items())),
            "port_mapping": self.port_mapping,
            "volume_mapping": self.volume_mapping,
            "device_mapping": self.device_mapping,
            "server_args": list(self.server_args),
            "command_sha256": self.command_sha256,
            "preset_sha256": self.preset_sha256,
            "ocr_model": self.ocr_model.payload(),
            "gemma_model": self.gemma_model.payload(),
            "ownership_labels": dict(sorted(self.ownership_labels.items())),
        }


def expected_router_server_args(pair: RouterPair) -> tuple[str, ...]:
    """The command that keeps the Router alive with zero autoloaded models."""

    del pair  # Pair-specific models are configured in the mounted preset.
    return (
        "--models-preset",
        "/router/models.ini",
        "--models-max",
        "1",
        "--no-models-autoload",
        "--host",
        "0.0.0.0",
        "--port",
        "8080",
        "--metrics",
    )


def router_ownership_labels(
    pair: RouterPair,
    fingerprint: str,
) -> dict[str, str]:
    return {
        ROUTER_OWNER_LABEL: ROUTER_OWNER_VALUE,
        ROUTER_PAIR_LABEL: pair.kind.value,
        ROUTER_FINGERPRINT_LABEL: fingerprint,
        ROUTER_PROJECT_LABEL: ROUTER_PROJECT_NAME,
    }


def build_router_fingerprint(
    *,
    pair: RouterPair,
    image_ref: str,
    image_id: str,
    repo_digest: str,
    entrypoint: tuple[str, ...],
    binary_version: str,
    resolved_compose_config: Mapping[str, Any],
    effective_environment: Mapping[str, str],
    port_mapping: Mapping[str, Any],
    volume_mapping: Mapping[str, Any],
    device_mapping: Mapping[str, Any],
    server_args: tuple[str, ...],
    preset_sha256: str,
    ocr_model: RouterModelMaterial,
    gemma_model: RouterModelMaterial,
) -> str:
    """Fingerprint every dependency that can change a model handoff."""

    command_sha256 = canonical_sha256(server_args)
    labels = router_ownership_labels(pair, "<computed>")
    contract = RouterRuntimeContract(
        pair=pair,
        fingerprint="<computed>",
        image_ref=image_ref,
        image_id=image_id,
        repo_digest=repo_digest,
        entrypoint=entrypoint,
        binary_version=binary_version,
        resolved_compose_config=resolved_compose_config,
        effective_environment=effective_environment,
        port_mapping=port_mapping,
        volume_mapping=volume_mapping,
        device_mapping=device_mapping,
        server_args=server_args,
        command_sha256=command_sha256,
        preset_sha256=preset_sha256,
        ocr_model=ocr_model,
        gemma_model=gemma_model,
        ownership_labels=labels,
    )
    return canonical_sha256(contract.payload())


def build_router_contract(
    *,
    spec: RouterRuntimeSpec,
    image_id: str,
    repo_digest: str,
    entrypoint: tuple[str, ...],
    binary_version: str,
    resolved_compose_config: Mapping[str, Any],
    effective_environment: Mapping[str, str],
    port_mapping: Mapping[str, Any],
    volume_mapping: Mapping[str, Any],
    device_mapping: Mapping[str, Any],
    server_args: tuple[str, ...],
    preset_sha256: str,
) -> RouterRuntimeContract:
    """Construct a Router contract after Docker metadata is independently read."""

    fingerprint = build_router_fingerprint(
        pair=spec.pair,
        image_ref=spec.image_ref,
        image_id=image_id,
        repo_digest=repo_digest,
        entrypoint=entrypoint,
        binary_version=binary_version,
        resolved_compose_config=resolved_compose_config,
        effective_environment=effective_environment,
        port_mapping=port_mapping,
        volume_mapping=volume_mapping,
        device_mapping=device_mapping,
        server_args=server_args,
        preset_sha256=preset_sha256,
        ocr_model=spec.ocr_model,
        gemma_model=spec.gemma_model,
    )
    return RouterRuntimeContract(
        pair=spec.pair,
        fingerprint=fingerprint,
        image_ref=spec.image_ref,
        image_id=image_id,
        repo_digest=repo_digest,
        entrypoint=entrypoint,
        binary_version=binary_version,
        resolved_compose_config=resolved_compose_config,
        effective_environment=effective_environment,
        port_mapping=port_mapping,
        volume_mapping=volume_mapping,
        device_mapping=device_mapping,
        server_args=server_args,
        command_sha256=canonical_sha256(server_args),
        preset_sha256=preset_sha256,
        ocr_model=spec.ocr_model,
        gemma_model=spec.gemma_model,
        ownership_labels=router_ownership_labels(spec.pair, fingerprint),
    )


def router_environment(
    contract: RouterRuntimeContract,
) -> dict[str, str]:
    """Return the complete non-secret environment used for compose resolution."""

    values = {
        "LLAMA_ROUTER_IMAGE": contract.image_ref,
        "LLAMA_ROUTER_FINGERPRINT": contract.fingerprint,
        "LLAMA_ROUTER_PRESET_SHA256": contract.preset_sha256,
        "PADDLEOCR_ROUTER_MODEL_VOLUME": contract.ocr_model.volume_name,
        "GEMMA_ROUTER_MODEL_VOLUME": contract.gemma_model.volume_name,
        "PADDLEOCR_ROUTER_READY_MANIFEST_SHA256": (
            contract.ocr_model.ready_manifest_sha256
        ),
        "GEMMA_ROUTER_READY_MANIFEST_SHA256": (
            contract.gemma_model.ready_manifest_sha256
        ),
        "PADDLEOCR_ROUTER_MODEL_SHA256": contract.ocr_model.model_sha256,
        "PADDLEOCR_ROUTER_MMPROJ_SHA256": contract.ocr_model.mmproj_sha256,
        "GEMMA_ROUTER_MODEL_SHA256": contract.gemma_model.model_sha256,
    }
    for key, value in contract.effective_environment.items():
        normalized_key = str(key)
        normalized_value = str(value)
        if (
            normalized_key not in values
            and normalized_value != "<router-dynamic>"
        ):
            values[normalized_key] = normalized_value
    return values

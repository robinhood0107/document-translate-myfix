"""Policy helpers that keep managed local inference on llama.cpp only."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any


MANAGED_LOCAL_INFERENCE_BACKEND = "llama.cpp"
MANAGED_LLAMA_CPP_MIGRATION_VERSION = 1
MANAGED_LLAMA_CPP_MIGRATION_VERSION_KEY = (
    "managed_runtime/llamacpp_only_version"
)

# These keys were used by local experiments or older runtime bundles.  They
# are never forwarded to managed Compose processes now.  Custom endpoint URLs
# are intentionally not part of this list.
LEGACY_VLLM_ENVIRONMENT_KEYS = (
    "CT_PADDLEOCR_BACKEND",
    "PADDLEOCR_BACKEND",
    "PADDLEOCR_RUNTIME_BACKEND",
    "PADDLEOCR_VL_BACKEND",
    "PADDLEOCR_VLLM_CONFIG",
    "PADDLEOCR_VLLM_IMAGE",
)

# Historical releases did not expose a backend selector in the current UI,
# but development builds could persist these keys.  Migration touches only a
# value that explicitly names vLLM and preserves every endpoint and sampler.
LEGACY_VLLM_QSETTINGS_KEYS = (
    "paddleocr_vl/backend",
    "paddleocr_vl/runtime_backend",
    "paddleocr_vl_spotting/backend",
    "mangalmm_ocr/backend",
    "hunyuan_ocr/backend",
    "gemma_local_server/backend",
)

LEGACY_VLLM_CONTAINER_NAMES = ("paddleocr-vllm",)

_RETIRED_BACKEND_VALUES = frozenset(
    {
        "vllm",
        "vllm-server",
        "paddleocr-vllm",
        "paddleocr_genai_vllm",
    }
)
_VLLM_PROCESS_PATTERN = re.compile(
    r"(?<![A-Za-z0-9])vllm(?![A-Za-z0-9])",
    re.IGNORECASE,
)


def is_retired_vllm_backend(value: Any) -> bool:
    normalized = str(value or "").strip().lower().replace(" ", "-")
    return normalized in _RETIRED_BACKEND_VALUES


def sanitize_managed_runtime_environment(
    source: Mapping[str, Any],
) -> tuple[dict[str, str], dict[str, str]]:
    """Return a Compose environment with retired vLLM controls removed."""

    environment = {
        str(key): str(value)
        for key, value in source.items()
    }
    ignored: dict[str, str] = {}
    for key in LEGACY_VLLM_ENVIRONMENT_KEYS:
        if key not in environment:
            continue
        value = environment[key]
        if key.endswith(("_CONFIG", "_IMAGE")) or is_retired_vllm_backend(
            value
        ):
            ignored[key] = value
            environment.pop(key, None)
    return environment, ignored


def find_vllm_process_commands(process_output: str) -> list[str]:
    """Return process-table lines that indicate an active vLLM runtime."""

    violations: list[str] = []
    for raw_line in str(process_output or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        normalized = " ".join(line.split())
        lowered = normalized.lower()
        if lowered in {"args", "command", "cmd"}:
            continue
        if "--backend vllm" in lowered or "--backend=vllm" in lowered:
            violations.append(normalized)
            continue
        if _VLLM_PROCESS_PATTERN.search(lowered):
            violations.append(normalized)
    return violations


__all__ = [
    "LEGACY_VLLM_CONTAINER_NAMES",
    "LEGACY_VLLM_ENVIRONMENT_KEYS",
    "LEGACY_VLLM_QSETTINGS_KEYS",
    "MANAGED_LLAMA_CPP_MIGRATION_VERSION",
    "MANAGED_LLAMA_CPP_MIGRATION_VERSION_KEY",
    "MANAGED_LOCAL_INFERENCE_BACKEND",
    "find_vllm_process_commands",
    "is_retired_vllm_backend",
    "sanitize_managed_runtime_environment",
]

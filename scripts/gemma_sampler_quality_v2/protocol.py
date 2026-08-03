"""Frozen, text-free protocol definitions for the sampler-quality v2 lab."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Iterable, Mapping, Sequence


PROTOCOL_VERSION = "gemma-sampler-quality-v2"
CORPUS_OCCURRENCE_COUNT = 758
CORPUS_CASE_COUNT = 478
TUNING_CASE_COUNT = 382
HOLDOUT_CASE_COUNT = 96
SEEDS: tuple[int, int] = (20260802, 20260803)

PINNED_LLAMA_CPP_COMMIT = "ff067f76dd8e9e05f0528056f1274adf01a54d70"
PINNED_LLAMA_CPP_BUILD = "b10133"
PINNED_LLAMA_CPP_IMAGE = (
    "ghcr.io/ggml-org/llama.cpp@sha256:"
    "22e0e3bfe967af4fd1df6a918022abbfd4e72e4d40a4769e616a4176790acbcb"
)
GEMMA_MODEL_ALIAS = "gemma-4-26B-IQ4_NL.gguf"

MAX_COMPLETION_TOKENS = 512
CONTEXT_SIZE = 4096
CHUNK_SIZE = 6
REASONING_ENABLED = False
DEFAULT_TUPLE = (0.7, 0.95, 64, 0.0)
TEMPERATURE_VALUES = tuple(round(value / 10.0, 1) for value in range(1, 11))
TOP_P_VALUES = (0.85, 0.90, 0.95, 0.98, 1.00)
TOP_K_VALUES = (0, 32, 64, 128, 256, 512)
MIN_P_VALUES = (0.00, 0.01, 0.03, 0.05, 0.10)


class ProtocolError(ValueError):
    """Raised when a run would deviate from the fixed v2 protocol."""


def canonical_sha256(value: Any) -> str:
    """Stable SHA-256 used for contract, reference, and aggregate identities."""

    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, order=True)
class SamplerTuple:
    """The only request fields that vary across v2 sampler arms."""

    temperature: float
    top_p: float
    top_k: int
    min_p: float

    def __post_init__(self) -> None:
        if not 0.0 < float(self.temperature) <= 1.0:
            raise ProtocolError("Temperature must be in (0.0, 1.0]; temperature 0 is excluded.")
        if not 0.0 <= float(self.top_p) <= 1.0:
            raise ProtocolError("top_p must be in [0.0, 1.0].")
        if int(self.top_k) < 0:
            raise ProtocolError("top_k cannot be negative.")
        if not 0.0 <= float(self.min_p) <= 1.0:
            raise ProtocolError("min_p must be in [0.0, 1.0].")

    @property
    def key(self) -> str:
        return (
            f"t{self.temperature:.2f}-p{self.top_p:.2f}"
            f"-k{int(self.top_k)}-m{self.min_p:.2f}"
        )

    def payload(self) -> dict[str, float | int]:
        return {
            "temperature": round(float(self.temperature), 4),
            "top_p": round(float(self.top_p), 4),
            "top_k": int(self.top_k),
            "min_p": round(float(self.min_p), 4),
        }


@dataclass(frozen=True)
class SamplerArm:
    """One sampler tuple in a phase; every arm runs every frozen case twice."""

    phase: str
    sampler: SamplerTuple
    reused: bool = False

    @property
    def key(self) -> str:
        return f"{self.phase}-{self.sampler.key}"

    def payload(self) -> dict[str, Any]:
        return {
            "phase": self.phase,
            "arm_key": self.key,
            "sampler": self.sampler.payload(),
            "reused": self.reused,
        }


def default_sampler_tuple() -> SamplerTuple:
    return SamplerTuple(*DEFAULT_TUPLE)


def temperature_arms() -> tuple[SamplerArm, ...]:
    return tuple(
        SamplerArm("temperature", SamplerTuple(value, 0.95, 64, 0.0))
        for value in TEMPERATURE_VALUES
    )


def _require_distinct_temperatures(values: Sequence[float], *, count: int) -> tuple[float, ...]:
    normalized = tuple(round(float(value), 4) for value in values)
    if len(normalized) != count or len(set(normalized)) != count:
        raise ProtocolError(f"Expected exactly {count} distinct selected temperatures.")
    if any(value not in TEMPERATURE_VALUES for value in normalized):
        raise ProtocolError("Selected temperatures must come from the temperature phase.")
    return normalized


def joint_top_p_top_k_arms(selected_temperatures: Sequence[float]) -> tuple[SamplerArm, ...]:
    """Return the complete 2 x 5 x 6 joint grid, marking temp-stage reuse."""

    temperatures = _require_distinct_temperatures(selected_temperatures, count=2)
    arms: list[SamplerArm] = []
    for temperature in temperatures:
        for top_p in TOP_P_VALUES:
            for top_k in TOP_K_VALUES:
                sampler = SamplerTuple(temperature, top_p, top_k, 0.0)
                reused = top_p == 0.95 and top_k == 64
                arms.append(SamplerArm("joint_top_p_top_k", sampler, reused=reused))
    return tuple(arms)


def min_p_arms(selected_tuples: Sequence[SamplerTuple]) -> tuple[SamplerArm, ...]:
    """Return the complete 3 x 5 min-p grid, marking pre-existing min_p=0."""

    tuples = tuple(selected_tuples)
    if len(tuples) != 3 or len({item.key for item in tuples}) != 3:
        raise ProtocolError("Expected exactly three distinct selected sampler tuples.")
    arms: list[SamplerArm] = []
    for selected in tuples:
        for min_p in MIN_P_VALUES:
            sampler = SamplerTuple(
                selected.temperature,
                selected.top_p,
                selected.top_k,
                min_p,
            )
            arms.append(SamplerArm("min_p", sampler, reused=min_p == 0.0))
    return tuple(arms)


def new_arms(arms: Iterable[SamplerArm]) -> tuple[SamplerArm, ...]:
    return tuple(arm for arm in arms if not arm.reused)


def expected_response_counts(
    *,
    case_count: int = CORPUS_CASE_COUNT,
    seed_count: int = len(SEEDS),
) -> dict[str, int]:
    """Return the plan's exact new-response totals without opening corpus text."""

    if case_count != CORPUS_CASE_COUNT:
        raise ProtocolError("v2 response totals require the frozen 478-case corpus.")
    if seed_count != len(SEEDS):
        raise ProtocolError("v2 response totals require the two fixed seeds.")
    per_arm = case_count * seed_count
    temperature = len(temperature_arms()) * per_arm
    joint = len(new_arms(joint_top_p_top_k_arms((0.1, 0.2)))) * per_arm
    min_p = len(new_arms(min_p_arms((
        SamplerTuple(0.1, 0.95, 64, 0.0),
        SamplerTuple(0.2, 0.95, 64, 0.0),
        SamplerTuple(0.3, 0.95, 64, 0.0),
    )))) * per_arm
    return {
        "temperature": temperature,
        "joint_top_p_top_k": joint,
        "min_p": min_p,
        "total_new": temperature + joint + min_p,
    }


def filters_disabled(*, top_k: int, top_p: float) -> bool:
    """Pinned llama.cpp contract: k<=0 and p>=1 are no-op samplers.

    This is intentionally a narrow contract predicate.  Runtime verification
    must additionally prove the exact pinned image and binary revision before
    accepting a request with either disabled filter.
    """

    return int(top_k) <= 0 and float(top_p) >= 1.0


def _matches_pinned_llama_cpp_build(binary_version: str) -> bool:
    """Accept the pinned build's legacy and current llama-server formats."""

    text = str(binary_version or "").strip()
    numeric_build = PINNED_LLAMA_CPP_BUILD.removeprefix("b")
    for line in text.splitlines():
        legacy_tokens = {
            token.strip("(),")
            for token in line.split()
            if token.strip("(),")
        }
        if PINNED_LLAMA_CPP_BUILD in legacy_tokens:
            return True
        label, separator, value = line.partition(":")
        if label.strip().casefold() != "version" or not separator:
            continue
        reported_build = value.strip().split(maxsplit=1)[0] if value.strip() else ""
        if reported_build == numeric_build:
            return True
    return False


def assert_pinned_sampler_contract(
    *,
    image_ref: str,
    binary_version: str,
    payload: Mapping[str, Any],
) -> None:
    """Fail closed if the known top-k/top-p semantics lack pinned build proof."""

    if str(image_ref) != PINNED_LLAMA_CPP_IMAGE:
        raise ProtocolError("Sampler test requires the pinned llama.cpp image digest.")
    if not _matches_pinned_llama_cpp_build(binary_version):
        raise ProtocolError("Sampler test requires the pinned llama.cpp binary revision.")
    top_k = payload.get("top_k")
    top_p = payload.get("top_p")
    if not isinstance(top_k, int) or isinstance(top_k, bool):
        raise ProtocolError("Sampler payload omitted integer top_k.")
    if not isinstance(top_p, (int, float)) or isinstance(top_p, bool):
        raise ProtocolError("Sampler payload omitted numeric top_p.")
    # Both values must survive serialization exactly; the pinned source's
    # top-k <= 0 empty sampler and top-p >= 1 return paths are the evidence.
    if int(top_k) == 0 and float(top_p) == 1.0 and not filters_disabled(
        top_k=int(top_k), top_p=float(top_p)
    ):
        raise ProtocolError("Disabled-filter sampler contract was not preserved.")


def fixed_request_contract_payload() -> dict[str, Any]:
    return {
        "protocol_version": PROTOCOL_VERSION,
        "llama_cpp": {
            "commit": PINNED_LLAMA_CPP_COMMIT,
            "build": PINNED_LLAMA_CPP_BUILD,
            "image": PINNED_LLAMA_CPP_IMAGE,
        },
        "model": GEMMA_MODEL_ALIAS,
        "max_completion_tokens": MAX_COMPLETION_TOKENS,
        "context_size": CONTEXT_SIZE,
        "chunk_size": CHUNK_SIZE,
        "reasoning_enabled": REASONING_ENABLED,
        "seeds": list(SEEDS),
        "case_identity": "language+source_text+context_after_text",
    }


def assert_fixed_request_contract(contract: Mapping[str, Any]) -> None:
    required = fixed_request_contract_payload()
    for key, expected in required.items():
        if contract.get(key) != expected:
            raise ProtocolError(f"Fixed request contract changed at {key}.")


if expected_response_counts() != {
    "temperature": 9560,
    "joint_top_p_top_k": 55448,
    "min_p": 11472,
    "total_new": 76480,
}:
    raise RuntimeError("Sampler v2 matrix totals do not match the approved plan.")

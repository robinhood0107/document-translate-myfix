"""관리형 llama.cpp 런타임 볼륨의 자가복구.

관리형 런타임 다섯 개(Gemma, HunyuanOCR, MangaLMM, PaddleOCR-VL,
PaddleOCR-VL Spotting)는 모두 준비된 Docker 볼륨 안의 ready manifest 로 자신을
증명한다. manifest 는 준비 당시의 llama.cpp image identity 를 함께 봉인한다.

기본 image 참조가 고정 digest 가 아니라 움직이는 태그(``:server-cuda``)이므로,
업스트림이 그 태그를 갱신하면 로컬 image digest 가 바뀐다. 그러면 모델 파일이
완벽히 멀쩡한데도 manifest 의 image identity 만 어긋나 런타임 계약이 깨진다.
이 모듈은 그 상태를 정확히 식별하고, 준비 스크립트의 ``Reseal`` 경로로
되돌린다. ``Reseal`` 은 볼륨 안 모델을 다시 복사하지 않고, 현재 image 로 실제
smoke 를 다시 통과시킨 뒤 manifest 만 다시 쓴다.

여기서 고치는 것은 image identity drift 하나뿐이다. 모델 SHA-256 불일치나
스키마 위반처럼 실제로 볼륨을 신뢰할 수 없는 상태는 그대로 실패시킨다.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from modules.utils.exceptions import OperationCancelledError
from modules.utils.llama_cpp_runtime import run_docker_command

logger = logging.getLogger(__name__)

# Reseal 은 실제 GPU 모델 적재 smoke 를 포함한다. 대형 모델의 첫 적재는 분 단위다.
DEFAULT_RESEAL_TIMEOUT_SEC = 1800.0
# 원본을 새로 내려받아야 하는 Prepare 경로까지 포함하면 훨씬 길어질 수 있다.
DEFAULT_PROVISION_TIMEOUT_SEC = 21600.0

MANIFEST_IMAGE_IDENTITY_KEYS = (
    "source_image_ref",
    "source_image_digest",
    "source_image_id",
)


class ManagedRuntimeRepairError(RuntimeError):
    """자가복구 자체가 실패했을 때."""


@dataclass(frozen=True)
class ManagedRuntimeRepairPlan:
    """한 런타임을 복구하는 데 필요한 것 전부."""

    runtime_label: str
    """사람이 읽을 이름. 진행 메시지와 로그에 쓴다."""

    prepare_script: Path
    """저장소의 ``scripts/prepare_*_runtime.ps1`` 경로."""

    volume_name: str
    """준비된 모델 볼륨 이름."""

    volume_argument: str = "-VolumeName"
    """볼륨 이름을 전달할 스크립트 파라미터 이름."""

    extra_arguments: tuple[str, ...] = ()
    """런타임별 추가 인자(예: PaddleOCR 의 ``-Accelerator``)."""


def _decode_manifest(manifest_bytes: bytes) -> dict[str, Any]:
    payload = json.loads(manifest_bytes.decode("utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError("manifest is not a JSON object")
    return payload


def manifest_recorded_image_identity(
    manifest_bytes: bytes,
) -> tuple[str, str] | None:
    """manifest 가 봉인한 ``(image_ref, image_id)``. 읽지 못하면 ``None``."""

    try:
        payload = _decode_manifest(manifest_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
        return None
    image_ref = str(payload.get("source_image_ref") or "").strip()
    image_id = str(payload.get("source_image_id") or "").strip()
    if not image_id:
        return None
    return image_ref, image_id


def is_image_identity_only_drift(
    manifest_bytes: bytes,
    *,
    current_image_id: str,
    revalidate: Callable[[str, str], Any],
) -> bool:
    """어긋난 것이 봉인된 image identity **뿐인지** 판정한다.

    ``revalidate`` 는 ``(image_ref, image_id)`` 를 받아 같은 manifest 를 다시
    검증하는 호출 가능 객체다. manifest 가 스스로 기록한 image identity 로
    되돌렸을 때 전체 계약이 통과하면, 깨진 것은 image identity 하나뿐이다.

    이 판정을 통과하지 못하면 복구하지 않는다. 모델 해시 불일치를 image drift 로
    오인해 다시 봉인해 버리면, 신뢰할 수 없는 볼륨에 유효 도장을 찍는 셈이 된다.
    """

    recorded = manifest_recorded_image_identity(manifest_bytes)
    if recorded is None:
        return False
    recorded_ref, recorded_id = recorded
    if not current_image_id or recorded_id == current_image_id:
        return False
    try:
        revalidate(recorded_ref or current_image_id, recorded_id)
    except Exception:
        return False
    return True


def _resolve_powershell_executable() -> str:
    for name in ("powershell.exe", "pwsh.exe", "powershell", "pwsh"):
        resolved = shutil.which(name)
        if resolved:
            return resolved
    system_root = os.environ.get("SystemRoot", r"C:\Windows")
    fallback = Path(system_root) / "System32" / "WindowsPowerShell" / "v1.0" / "powershell.exe"
    if fallback.is_file():
        return str(fallback)
    raise ManagedRuntimeRepairError(
        "PowerShell was not found, so the managed runtime volume cannot be repaired "
        "automatically. Run the matching scripts/prepare_*_runtime.ps1 by hand."
    )


def run_managed_runtime_preparation(
    plan: ManagedRuntimeRepairPlan,
    *,
    mode: str = "Auto",
    allow_download: bool = False,
    cancel_checker: Callable[[], bool] | None = None,
    progress: Callable[[str], None] | None = None,
    timeout_sec: float | None = None,
) -> dict[str, Any]:
    """준비 스크립트를 실행하고 그 JSON 결과를 돌려준다.

    ``mode`` 는 스크립트의 ``-Mode`` 로 그대로 넘어간다. ``Auto`` 는 볼륨이 이미
    검증된 모델을 담고 있으면 ``Reseal``, 아니면 ``Prepare`` 를 고른다.
    """

    if not plan.prepare_script.is_file():
        raise ManagedRuntimeRepairError(
            f"The {plan.runtime_label} preparation script is missing: {plan.prepare_script}"
        )
    if timeout_sec is None:
        timeout_sec = (
            DEFAULT_RESEAL_TIMEOUT_SEC if mode == "Reseal" else DEFAULT_PROVISION_TIMEOUT_SEC
        )

    command = [
        _resolve_powershell_executable(),
        "-NoProfile",
        "-NonInteractive",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        str(plan.prepare_script),
        "-Mode",
        mode,
        plan.volume_argument,
        plan.volume_name,
        *plan.extra_arguments,
    ]
    if allow_download:
        command.append("-AllowDownload")
        if "-DownloadDirectory" not in plan.extra_arguments:
            command.extend(
                [
                    "-DownloadDirectory",
                    str(
                        plan.prepare_script.parent.parent
                        / "models"
                        / "managed-runtime-sources"
                    ),
                ]
            )

    if progress is not None:
        progress(
            f"{plan.runtime_label} 런타임 볼륨을 복구하는 중입니다({mode}). "
            "실제 모델 적재 검사를 포함하므로 몇 분 걸릴 수 있습니다."
        )
    logger.info(
        "Repairing the %s managed runtime volume: mode=%s volume=%s",
        plan.runtime_label,
        mode,
        plan.volume_name,
    )

    try:
        completed = run_docker_command(
            command,
            cwd=plan.prepare_script.parent.parent,
            check=False,
            timeout_sec=float(timeout_sec),
            cancel_checker=cancel_checker,
        )
    except OperationCancelledError:
        raise
    except RuntimeError as exc:
        raise ManagedRuntimeRepairError(
            f"The {plan.runtime_label} runtime volume repair did not finish.\n{exc}"
        ) from exc

    output = ((completed.stdout or "") + "\n" + (completed.stderr or "")).strip()
    if completed.returncode != 0:
        raise ManagedRuntimeRepairError(
            f"The {plan.runtime_label} runtime volume repair failed "
            f"(exit={completed.returncode}).\n{output}"
        )
    logger.info("Repaired the %s managed runtime volume.", plan.runtime_label)
    return _parse_trailing_json(completed.stdout or "")


def _parse_trailing_json(stdout: str) -> dict[str, Any]:
    """진행 로그 뒤에 붙은 JSON 결과만 떼어낸다.

    준비 스크립트는 사람이 읽는 진행 줄을 먼저 찍고 마지막에 JSON 객체 하나를
    낸다. 결과를 읽지 못하는 것 자체는 실패가 아니다. 성공 여부는 종료 코드가
    이미 말했다.
    """

    text = (stdout or "").strip()
    start = text.rfind("\n{")
    candidate = text[start + 1 :] if start >= 0 else text
    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def describe_image_identity_drift(
    manifest_bytes: bytes,
    *,
    current_image_id: str,
    runtime_label: str,
) -> str:
    """사람이 읽을 drift 설명."""

    recorded = manifest_recorded_image_identity(manifest_bytes)
    recorded_id = recorded[1] if recorded else "(unreadable)"
    return (
        f"The {runtime_label} ready manifest was sealed against a different "
        f"llama.cpp image (manifest={recorded_id}, actual={current_image_id}). "
        "The model files themselves still match the product contract."
    )


def manifest_without_image_identity(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """image identity 키를 뺀 manifest 사본. 진단 비교용."""

    return {
        key: value
        for key, value in dict(manifest).items()
        if key not in MANIFEST_IMAGE_IDENTITY_KEYS
    }

from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

from benchmarking.inpaint_detector_bakeoff.holdout import (
    canonical_holdout_lock_path,
    claim_holdout_once,
    execution_argv_sha256,
    execution_binding_sha256,
)
from scripts import run_inpaint_holdout_once_v4
from scripts.run_inpaint_holdout_once_v4 import main as holdout_main


def _sha(value: str) -> str:
    import hashlib

    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _command() -> list[str]:
    return [
        sys.executable,
        "-c",
        "raise SystemExit(0)",
        "--manifest-sha256",
        _sha("manifest"),
        "--model-sha256",
        _sha("model"),
        "--provider",
        "CUDAExecutionProvider",
    ]


def _prerequisites(command: list[str] | None = None) -> dict[str, object]:
    command = list(command or _command())
    payload: dict[str, object] = {
        "schema_version": "inpaint-holdout-prerequisites-v4",
        "holdout_id": "a5-corpus-d2",
        "manifest_sha256": _sha("manifest"),
        "code_commit": _sha("commit"),
        "model_sha256": _sha("model"),
        "runtime_provider": "CUDAExecutionProvider",
        "source_only_manifest_sealed": True,
        "synthetic_gate_passed": True,
        "e1_mask_gate_passed": True,
        "e1_cuda_gate_passed": True,
        "visual_review_passed": True,
        "onnx_parity_passed": True,
        "onnx_final_binary_xor_pixels": 0,
        "product_stack_frozen": True,
        "development_source_sha256": [_sha("e1")],
        "holdout_source_sha256": [_sha("a5")],
    }
    execution_binding = {
        "argv": command,
        "argv_sha256": execution_argv_sha256(command),
        "manifest_sha256": payload["manifest_sha256"],
        "model_sha256": payload["model_sha256"],
        "runtime_provider": payload["runtime_provider"],
    }
    execution_binding["binding_sha256"] = execution_binding_sha256(
        argv=command,
        manifest_sha256=execution_binding["manifest_sha256"],
        model_sha256=execution_binding["model_sha256"],
        runtime_provider=execution_binding["runtime_provider"],
    )
    payload["execution_binding"] = execution_binding
    return payload


def test_holdout_claim_is_atomic_and_cannot_be_reused(tmp_path: Path) -> None:
    prerequisites = tmp_path / "prerequisites.json"
    prerequisites.write_text(json.dumps(_prerequisites()), encoding="utf-8")
    lock = canonical_holdout_lock_path(prerequisites, "a5-corpus-d2")

    claim_holdout_once(
        prerequisites_path=prerequisites,
        prerequisites=_prerequisites(),
    )

    with pytest.raises(FileExistsError):
        claim_holdout_once(
            prerequisites_path=prerequisites,
            prerequisites=_prerequisites(),
        )


def test_holdout_claim_binds_the_exact_prerequisite_file(tmp_path: Path) -> None:
    prerequisites = tmp_path / "prerequisites.json"
    prerequisites.write_text(json.dumps(_prerequisites()), encoding="utf-8")
    different = _prerequisites()
    different["runtime_provider"] = "different-provider"

    with pytest.raises(ValueError, match="differs from its sealed file"):
        claim_holdout_once(
            prerequisites_path=prerequisites,
            prerequisites=different,
        )


def test_holdout_rejects_development_source_collision(tmp_path: Path) -> None:
    payload = _prerequisites()
    payload["holdout_source_sha256"] = list(payload["development_source_sha256"])
    prerequisites = tmp_path / "prerequisites.json"
    prerequisites.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="overlaps development"):
        claim_holdout_once(
            prerequisites_path=prerequisites,
            prerequisites=payload,
        )


def test_holdout_gate_failure_does_not_create_lock(tmp_path: Path) -> None:
    payload = _prerequisites()
    payload["e1_cuda_gate_passed"] = False
    prerequisites = tmp_path / "prerequisites.json"
    prerequisites.write_text(json.dumps(payload), encoding="utf-8")
    lock = canonical_holdout_lock_path(prerequisites, payload["holdout_id"])

    with pytest.raises(ValueError, match="e1_cuda_gate_passed"):
        holdout_main(
            [
                "--prerequisites",
                str(prerequisites),
                "--",
                *_command(),
            ]
        )

    assert not lock.exists()


def test_holdout_rejects_missing_onnx_xor_and_duplicate_sources(
    tmp_path: Path,
) -> None:
    payload = _prerequisites()
    payload["onnx_final_binary_xor_pixels"] = None
    prerequisites = tmp_path / "prerequisites.json"
    prerequisites.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="ONNX mask XOR zero"):
        claim_holdout_once(
            prerequisites_path=prerequisites,
            prerequisites=payload,
        )

    payload = _prerequisites()
    payload["holdout_source_sha256"] = [
        payload["holdout_source_sha256"][0],
        payload["holdout_source_sha256"][0],
    ]
    prerequisites.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="must not contain duplicates"):
        claim_holdout_once(
            prerequisites_path=prerequisites,
            prerequisites=payload,
        )


def test_holdout_runner_rejects_checkout_commit_mismatch_before_claim(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _prerequisites()
    payload["code_commit"] = "a" * 40
    prerequisites = tmp_path / "prerequisites.json"
    prerequisites.write_text(json.dumps(payload), encoding="utf-8")
    lock = canonical_holdout_lock_path(prerequisites, payload["holdout_id"])

    class _Completed:
        stdout = "b" * 40 + "\n"
        returncode = 0

    monkeypatch.setattr(
        run_inpaint_holdout_once_v4.subprocess,
        "run",
        lambda *args, **kwargs: _Completed(),
    )

    with pytest.raises(ValueError, match="does not match HEAD"):
        holdout_main(
            [
                "--prerequisites",
                str(prerequisites),
                "--",
                *_command(),
            ]
        )

    assert not lock.exists()


def test_holdout_runner_rejects_alternate_lock_and_command_tampering(
    tmp_path: Path,
) -> None:
    payload = _prerequisites()
    prerequisites = tmp_path / "prerequisites.json"
    prerequisites.write_text(json.dumps(payload), encoding="utf-8")
    canonical_lock = canonical_holdout_lock_path(prerequisites, payload["holdout_id"])
    alternate_lock = tmp_path / "alternate.lock.json"

    with pytest.raises(SystemExit) as alternate_lock_error:
        holdout_main(
            [
                "--prerequisites",
                str(prerequisites),
                "--lock",
                str(alternate_lock),
                "--",
                *_command(),
            ]
        )

    assert alternate_lock_error.value.code == 2
    assert not alternate_lock.exists()
    assert not canonical_lock.exists()

    with pytest.raises(ValueError, match="differs from sealed execution argv"):
        holdout_main(
            [
                "--prerequisites",
                str(prerequisites),
                "--",
                sys.executable,
                "-c",
                "raise SystemExit(1)",
            ]
        )
    assert not canonical_lock.exists()


@pytest.mark.parametrize("field", ["runtime_provider", "model_sha256"])
def test_holdout_rejects_wrong_execution_provider_or_model_binding(
    tmp_path: Path,
    field: str,
) -> None:
    payload = _prerequisites()
    execution = payload["execution_binding"]
    assert isinstance(execution, dict)
    execution[field] = "CPUExecutionProvider" if field == "runtime_provider" else _sha(
        "wrong-model"
    )
    prerequisites = tmp_path / "prerequisites.json"
    prerequisites.write_text(json.dumps(payload), encoding="utf-8")
    lock = canonical_holdout_lock_path(prerequisites, payload["holdout_id"])

    with pytest.raises(ValueError, match=f"execution binding {field} does not match"):
        claim_holdout_once(
            prerequisites_path=prerequisites,
            prerequisites=payload,
        )
    assert not lock.exists()


@pytest.mark.parametrize(
    ("flag", "wrong_value"),
    [
        ("--provider", "CPUExecutionProvider"),
        ("--model-sha256", _sha("wrong-model")),
    ],
)
def test_holdout_runner_rejects_wrong_provider_or_model_command(
    tmp_path: Path,
    flag: str,
    wrong_value: str,
) -> None:
    payload = _prerequisites()
    prerequisites = tmp_path / "prerequisites.json"
    prerequisites.write_text(json.dumps(payload), encoding="utf-8")
    lock = canonical_holdout_lock_path(prerequisites, payload["holdout_id"])
    wrong_command = _command()
    wrong_command[wrong_command.index(flag) + 1] = wrong_value

    with pytest.raises(ValueError, match="differs from sealed execution argv"):
        holdout_main(
            ["--prerequisites", str(prerequisites), "--", *wrong_command]
        )
    assert not lock.exists()


def test_holdout_runner_uses_canonical_lock_and_runs_exactly_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _prerequisites()
    payload["code_commit"] = "a" * 40
    prerequisites = tmp_path / "prerequisites.json"
    prerequisites.write_text(json.dumps(payload), encoding="utf-8")
    lock = canonical_holdout_lock_path(prerequisites, payload["holdout_id"])
    candidate_runs: list[list[str]] = []

    class _Completed:
        stdout = "a" * 40 + "\n"
        returncode = 0

    def _run(command: list[str], **kwargs: object) -> _Completed:
        if command[:3] == ["git", "rev-parse", "HEAD"]:
            return _Completed()
        candidate_runs.append(command)
        return _Completed()

    monkeypatch.setattr(run_inpaint_holdout_once_v4.subprocess, "run", _run)
    runner_argv = ["--prerequisites", str(prerequisites), "--", *_command()]

    assert holdout_main(runner_argv) == 0
    assert lock.exists()
    assert candidate_runs == [_command()]

    with pytest.raises(FileExistsError):
        holdout_main(runner_argv)
    assert candidate_runs == [_command()]

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np
import pytest

from benchmarking.inpaint_detector_bakeoff import holdout as holdout_contract
from benchmarking.inpaint_detector_bakeoff.holdout import (
    canonical_holdout_command,
    canonical_holdout_lock_path,
    claim_holdout_once,
    execution_argv_sha256,
    execution_binding_sha256,
    holdout_manifest_page_artifact_sha256,
)
from scripts import run_inpaint_holdout_once_v4
from scripts.run_inpaint_a5_candidate_v4 import main as candidate_main
from scripts.run_inpaint_holdout_once_v4 import main as holdout_main


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_sha(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


@pytest.fixture(autouse=True)
def _private_lock_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(holdout_contract, "HOLDOUT_LOCK_ROOT", tmp_path / "locks")
    # Existing lower-level contract tests exercise the future binding logic in
    # isolation. Production remains fail-closed; the explicit unavailable test
    # below restores the real default and proves no source/lock access occurs.
    monkeypatch.setattr(
        holdout_contract,
        "A5_PRODUCT_STACK_RUNNER_AVAILABLE",
        True,
    )


def test_a5_prerequisites_and_command_are_unavailable_before_artifact_access(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        holdout_contract,
        "A5_PRODUCT_STACK_RUNNER_AVAILABLE",
        False,
    )
    missing = tmp_path / "never-read.json"

    with pytest.raises(RuntimeError, match="A5 unavailable"):
        canonical_holdout_command({"manifest_path": str(missing)})
    with pytest.raises(RuntimeError, match="A5 unavailable"):
        holdout_contract.validate_holdout_prerequisites(
            {"manifest_path": str(missing)}
        )
    with pytest.raises(RuntimeError, match="A5 unavailable"):
        claim_holdout_once(
            prerequisites_path=missing,
            prerequisites={"manifest_path": str(missing)},
        )

    assert not missing.exists()
    assert not holdout_contract.HOLDOUT_LOCK_ROOT.exists()


def test_a5_holdout_script_main_is_unavailable_before_missing_path_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        holdout_contract,
        "A5_PRODUCT_STACK_RUNNER_AVAILABLE",
        False,
    )
    missing = tmp_path / "never-read-by-script.json"

    with pytest.raises(RuntimeError, match="A5 unavailable"):
        holdout_main(["--prerequisites", str(missing)])

    assert not missing.exists()
    assert not holdout_contract.HOLDOUT_LOCK_ROOT.exists()


def _prerequisites(tmp_path: Path) -> dict[str, object]:
    source = tmp_path / "source.png"
    assert cv2.imwrite(str(source), np.full((8, 8, 3), 255, np.uint8))
    zero = tmp_path / "zero.png"
    full = tmp_path / "full.png"
    assert cv2.imwrite(str(zero), np.zeros((8, 8), np.uint8))
    assert cv2.imwrite(str(full), np.full((8, 8), 255, np.uint8))
    source_sha = _file_sha(source)
    page: dict[str, object] = {
        "page_id": "p1",
        "path": str(source),
        "source_sha256": source_sha,
        "target_text_mask": None,
        "preserve_mask": str(zero),
        "target_instances": [],
        "regions": [
            {
                "region_id": "r1",
                "bubble_route_class": "clean_flat",
                "bubble_interior_mask": str(full),
                "ownership_mask": str(full),
                "protected_structure_mask": str(zero),
                "ambiguous_structure_mask": str(zero),
                "corner_protect_mask": str(zero),
            }
        ],
        "protected_structure_mask": str(zero),
        "ambiguous_structure_mask": str(zero),
        "ownership_mask": str(full),
        "bubble_interior_mask": str(full),
        "corner_protect_mask": str(zero),
        "expected_edit": "none",
        "annotation_frozen_before_candidate": True,
        "target_extent_independent": True,
        "target_inventory_independent": True,
        "target_review_complete": True,
        "candidate_seen": False,
    }
    page["artifact_sha256"] = holdout_manifest_page_artifact_sha256(page)
    inventory = {"page_ids": ["p1"], "source_sha256": [source_sha]}
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "inpaint-factorized-source-manifest-v4",
                "split_role": "final_holdout_source_only",
                "annotation_frozen_before_candidate": True,
                "target_extent_independent": True,
                "target_inventory_independent": True,
                "target_review_complete": True,
                "candidate_seen": False,
                "page_count": 1,
                "page_inventory_sha256": _json_sha(inventory),
                "pages": [page],
            }
        ),
        encoding="utf-8",
    )
    model = tmp_path / "winner.onnx"
    model.write_text("model", encoding="utf-8")
    matrix = tmp_path / "matrix.json"
    matrix.write_text(
        json.dumps(
            {
                "schema_version": "inpaint-factorized-matrix-v3",
                "final_model_sha256": _file_sha(model),
                "runtime_provider": "CUDAExecutionProvider",
                "a5_runtime_detector": {
                    "family_id": "winner",
                    "variant": "raw",
                    "detect_size": 1280,
                    "max_batch_size": 4,
                },
                "axes": {
                    "detector": ["winner"],
                    "ownership": ["ownership"],
                    "silhouette": ["silhouette"],
                    "router": ["R0"],
                    "expansion": ["raw"],
                    "fill": ["current_lama"],
                },
                "controls": {
                    "detector": "winner",
                    "ownership": "ownership",
                    "silhouette": "silhouette",
                    "router": "R0",
                    "expansion": "raw",
                    "fill": "current_lama",
                },
                "families": {
                    "detector": {"winner": {"pages": {}}},
                    "ownership": {"ownership": {"pages": {}}},
                    "silhouette": {"silhouette": {"pages": {}}},
                    "router": {"R0": {"algorithm": "R0", "pages": {}}},
                },
            }
        ),
        encoding="utf-8",
    )
    payload: dict[str, object] = {
        "schema_version": "inpaint-holdout-prerequisites-v4",
        "holdout_id": "a5-corpus-d2",
        "manifest_sha256": _file_sha(manifest),
        "code_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=holdout_contract.REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "model_sha256": _file_sha(model),
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
        "holdout_source_sha256": [source_sha],
    }
    runner = (
        holdout_contract.REPO_ROOT / holdout_contract.HOLDOUT_RUNNER_RELATIVE_PATH
    ).resolve()
    execution_binding: dict[str, object] = {
        "runner_relative_path": holdout_contract.HOLDOUT_RUNNER_RELATIVE_PATH,
        "runner_sha256": _file_sha(runner),
        "holdout_id": payload["holdout_id"],
        "manifest_path": str(manifest.resolve()),
        "manifest_sha256": payload["manifest_sha256"],
        "matrix_path": str(matrix.resolve()),
        "matrix_sha256": _file_sha(matrix),
        "model_path": str(model.resolve()),
        "model_sha256": payload["model_sha256"],
        "runtime_provider": payload["runtime_provider"],
        "source_inventory_sha256": _json_sha(inventory),
        "output_dir": str((tmp_path / "output").resolve()),
    }
    execution_binding["canonical_argv_sha256"] = execution_argv_sha256(
        canonical_holdout_command(execution_binding)
    )
    execution_binding["binding_sha256"] = execution_binding_sha256(
        execution_binding
    )
    payload["execution_binding"] = execution_binding
    return payload


def _command(payload: dict[str, object]) -> list[str]:
    execution = payload["execution_binding"]
    assert isinstance(execution, dict)
    return canonical_holdout_command(execution)


def _refresh_execution_binding(payload: dict[str, object]) -> list[str]:
    execution = payload["execution_binding"]
    assert isinstance(execution, dict)
    execution["canonical_argv_sha256"] = execution_argv_sha256(
        canonical_holdout_command(execution)
    )
    execution["binding_sha256"] = execution_binding_sha256(execution)
    return canonical_holdout_command(execution)


def _claim_for_candidate(tmp_path: Path, payload: dict[str, object]) -> Path:
    prerequisites = tmp_path / "candidate-prerequisites.json"
    prerequisites.write_text(json.dumps(payload), encoding="utf-8")
    claim_holdout_once(prerequisites_path=prerequisites, prerequisites=payload)
    return prerequisites


def test_holdout_claim_is_atomic_across_copied_prerequisite(
    tmp_path: Path,
) -> None:
    payload = _prerequisites(tmp_path)
    prerequisites = tmp_path / "prerequisites.json"
    prerequisites.write_text(json.dumps(payload), encoding="utf-8")
    copied_dir = tmp_path / "copied"
    copied_dir.mkdir()
    copied = copied_dir / "renamed-prerequisites.json"
    copied.write_bytes(prerequisites.read_bytes())
    source_inventory_sha256 = payload["execution_binding"][
        "source_inventory_sha256"
    ]
    original_lock = canonical_holdout_lock_path(
        prerequisites,
        source_inventory_sha256,
    )
    copied_lock = canonical_holdout_lock_path(copied, source_inventory_sha256)

    assert original_lock == copied_lock
    claim_holdout_once(
        prerequisites_path=prerequisites,
        prerequisites=payload,
    )
    with pytest.raises(FileExistsError):
        claim_holdout_once(
            prerequisites_path=copied,
            prerequisites=payload,
        )


def test_holdout_lock_identity_does_not_change_with_caller_holdout_id(
    tmp_path: Path,
) -> None:
    payload = _prerequisites(tmp_path)
    source_inventory_sha256 = payload["execution_binding"][
        "source_inventory_sha256"
    ]

    first = canonical_holdout_lock_path(
        tmp_path / "a5-first.json",
        source_inventory_sha256,
    )
    payload["holdout_id"] = "a5-renamed-by-caller"
    second = canonical_holdout_lock_path(
        tmp_path / "a5-second.json",
        payload["execution_binding"]["source_inventory_sha256"],
    )

    assert first == second


def test_holdout_claim_binds_the_exact_prerequisite_file(tmp_path: Path) -> None:
    payload = _prerequisites(tmp_path)
    prerequisites = tmp_path / "prerequisites.json"
    prerequisites.write_text(json.dumps(payload), encoding="utf-8")
    different = dict(payload)
    different["runtime_provider"] = "different-provider"

    with pytest.raises(ValueError, match="differs from its sealed file"):
        claim_holdout_once(
            prerequisites_path=prerequisites,
            prerequisites=different,
        )


def test_holdout_rejects_development_source_collision(tmp_path: Path) -> None:
    payload = _prerequisites(tmp_path)
    payload["holdout_source_sha256"] = list(payload["development_source_sha256"])
    prerequisites = tmp_path / "prerequisites.json"
    prerequisites.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="overlaps development"):
        claim_holdout_once(
            prerequisites_path=prerequisites,
            prerequisites=payload,
        )


def test_holdout_gate_failure_does_not_create_lock(tmp_path: Path) -> None:
    payload = _prerequisites(tmp_path)
    payload["e1_cuda_gate_passed"] = False
    prerequisites = tmp_path / "prerequisites.json"
    prerequisites.write_text(json.dumps(payload), encoding="utf-8")
    lock = canonical_holdout_lock_path(
        prerequisites,
        payload["execution_binding"]["source_inventory_sha256"],
    )

    with pytest.raises(ValueError, match="e1_cuda_gate_passed"):
        holdout_main(
            ["--prerequisites", str(prerequisites), "--", *_command(payload)]
        )
    assert not lock.exists()


def test_holdout_rejects_missing_onnx_xor_and_duplicate_sources(
    tmp_path: Path,
) -> None:
    payload = _prerequisites(tmp_path)
    payload["onnx_final_binary_xor_pixels"] = None
    prerequisites = tmp_path / "prerequisites.json"
    prerequisites.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="ONNX mask XOR zero"):
        claim_holdout_once(
            prerequisites_path=prerequisites,
            prerequisites=payload,
        )

    payload = _prerequisites(tmp_path)
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
    payload = _prerequisites(tmp_path)
    payload["code_commit"] = "a" * 40
    prerequisites = tmp_path / "prerequisites.json"
    prerequisites.write_text(json.dumps(payload), encoding="utf-8")
    lock = canonical_holdout_lock_path(
        prerequisites,
        payload["execution_binding"]["source_inventory_sha256"],
    )

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
            ["--prerequisites", str(prerequisites), "--", *_command(payload)]
        )
    assert not lock.exists()


def test_holdout_runner_rejects_alternate_lock_and_arbitrary_script(
    tmp_path: Path,
) -> None:
    payload = _prerequisites(tmp_path)
    prerequisites = tmp_path / "prerequisites.json"
    prerequisites.write_text(json.dumps(payload), encoding="utf-8")
    lock = canonical_holdout_lock_path(
        prerequisites,
        payload["execution_binding"]["source_inventory_sha256"],
    )
    alternate_lock = tmp_path / "alternate.lock.json"

    with pytest.raises(SystemExit) as alternate_lock_error:
        holdout_main(
            [
                "--prerequisites",
                str(prerequisites),
                "--lock",
                str(alternate_lock),
                "--",
                *_command(payload),
            ]
        )
    assert alternate_lock_error.value.code == 2
    assert not alternate_lock.exists()

    with pytest.raises(ValueError, match="differs from sealed execution argv"):
        holdout_main(
            [
                "--prerequisites",
                str(prerequisites),
                "--",
                sys.executable,
                "-c",
                "raise SystemExit(0)",
                "--manifest-sha256",
                str(payload["manifest_sha256"]),
                "--model-sha256",
                str(payload["model_sha256"]),
                "--provider",
                "CUDAExecutionProvider",
            ]
        )
    assert not lock.exists()


@pytest.mark.parametrize("field", ["runtime_provider", "model_sha256"])
def test_holdout_rejects_wrong_execution_provider_or_model_binding(
    tmp_path: Path,
    field: str,
) -> None:
    payload = _prerequisites(tmp_path)
    execution = payload["execution_binding"]
    assert isinstance(execution, dict)
    execution[field] = "CPUExecutionProvider" if field == "runtime_provider" else _sha(
        "wrong-model"
    )
    prerequisites = tmp_path / "prerequisites.json"
    prerequisites.write_text(json.dumps(payload), encoding="utf-8")
    lock = canonical_holdout_lock_path(
        prerequisites,
        payload["execution_binding"]["source_inventory_sha256"],
    )

    with pytest.raises(ValueError, match=f"execution binding {field} does not match"):
        claim_holdout_once(
            prerequisites_path=prerequisites,
            prerequisites=payload,
        )
    assert not lock.exists()


@pytest.mark.parametrize(
    "mutation",
    ["missing", "duplicate", "unknown", "provider", "model"],
)
def test_holdout_runner_rejects_noncanonical_flags(
    tmp_path: Path,
    mutation: str,
) -> None:
    payload = _prerequisites(tmp_path)
    prerequisites = tmp_path / "prerequisites.json"
    prerequisites.write_text(json.dumps(payload), encoding="utf-8")
    lock = canonical_holdout_lock_path(
        prerequisites,
        payload["execution_binding"]["source_inventory_sha256"],
    )
    command = _command(payload)
    flag = "--provider" if mutation == "provider" else "--model-sha256"
    if mutation == "missing":
        index = command.index("--manifest-sha256")
        del command[index : index + 2]
    elif mutation == "duplicate":
        command.extend(["--provider", "CUDAExecutionProvider"])
    elif mutation == "unknown":
        command.extend(["--unknown", "value"])
    else:
        index = command.index(flag)
        command[index + 1] = (
            "CPUExecutionProvider" if mutation == "provider" else _sha("wrong-model")
        )

    with pytest.raises(ValueError, match="differs from sealed execution argv"):
        holdout_main(["--prerequisites", str(prerequisites), "--", *command])
    assert not lock.exists()


def test_holdout_runner_uses_canonical_lock_and_runs_exactly_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _prerequisites(tmp_path)
    payload["code_commit"] = "a" * 40
    prerequisites = tmp_path / "prerequisites.json"
    prerequisites.write_text(json.dumps(payload), encoding="utf-8")
    lock = canonical_holdout_lock_path(
        prerequisites,
        payload["execution_binding"]["source_inventory_sha256"],
    )
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
    command = _command(payload)
    runner_argv = ["--prerequisites", str(prerequisites), "--", *command]

    assert holdout_main(runner_argv) == 0
    assert lock.exists()
    assert candidate_runs == [command]

    with pytest.raises(FileExistsError):
        holdout_main(runner_argv)
    assert candidate_runs == [command]


def test_a5_candidate_rejects_precomputed_detector_artifacts(
    tmp_path: Path,
) -> None:
    payload = _prerequisites(tmp_path)
    execution = payload["execution_binding"]
    assert isinstance(execution, dict)
    matrix_path = Path(str(execution["matrix_path"]))
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    matrix["families"]["detector"]["winner"]["pages"] = {
        "p1": {"raw": "precomputed.png"}
    }
    matrix_path.write_text(json.dumps(matrix), encoding="utf-8")
    execution["matrix_sha256"] = _file_sha(matrix_path)
    command = _refresh_execution_binding(payload)
    _claim_for_candidate(tmp_path, payload)

    with pytest.raises(RuntimeError, match="A5 unavailable"):
        candidate_main(command[2:])
    assert not Path(str(execution["output_dir"])).exists()


def test_a5_candidate_cannot_run_before_canonical_lock(tmp_path: Path) -> None:
    payload = _prerequisites(tmp_path)
    execution = payload["execution_binding"]
    assert isinstance(execution, dict)

    with pytest.raises(RuntimeError, match="A5 unavailable"):
        candidate_main(_command(payload)[2:])
    assert not Path(str(execution["output_dir"])).exists()


@pytest.mark.parametrize("artifact", ["source", "mask"])
def test_a5_candidate_rejects_source_or_mask_mutation_after_lock(
    tmp_path: Path,
    artifact: str,
) -> None:
    payload = _prerequisites(tmp_path)
    execution = payload["execution_binding"]
    assert isinstance(execution, dict)
    command = _command(payload)
    _claim_for_candidate(tmp_path, payload)
    manifest = json.loads(Path(str(execution["manifest_path"])).read_text(encoding="utf-8"))
    page = manifest["pages"][0]
    changed_path = Path(page["path"] if artifact == "source" else page["preserve_mask"])
    changed = (
        np.full((8, 8, 3), 100, np.uint8)
        if artifact == "source"
        else np.full((8, 8), 255, np.uint8)
    )
    assert cv2.imwrite(str(changed_path), changed)

    with pytest.raises(RuntimeError, match="A5 unavailable"):
        candidate_main(command[2:])
    assert not Path(str(execution["output_dir"])).exists()


def test_holdout_rejects_legacy_manifest_schema_before_lock(tmp_path: Path) -> None:
    payload = _prerequisites(tmp_path)
    execution = payload["execution_binding"]
    assert isinstance(execution, dict)
    manifest_path = Path(str(execution["manifest_path"]))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["schema_version"] = "inpaint-detector-bakeoff-manifest-v3"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    payload["manifest_sha256"] = _file_sha(manifest_path)
    execution["manifest_sha256"] = payload["manifest_sha256"]
    _refresh_execution_binding(payload)
    prerequisites = tmp_path / "legacy-prerequisites.json"
    prerequisites.write_text(json.dumps(payload), encoding="utf-8")
    lock = canonical_holdout_lock_path(
        prerequisites,
        payload["execution_binding"]["source_inventory_sha256"],
    )

    with pytest.raises(ValueError, match="strict source-only manifest v4"):
        claim_holdout_once(prerequisites_path=prerequisites, prerequisites=payload)
    assert not lock.exists()


def test_holdout_rejects_prerequisite_source_inventory_mismatch(
    tmp_path: Path,
) -> None:
    payload = _prerequisites(tmp_path)
    payload["holdout_source_sha256"] = [_sha("different-holdout-source")]
    prerequisites = tmp_path / "inventory-prerequisites.json"
    prerequisites.write_text(json.dumps(payload), encoding="utf-8")
    lock = canonical_holdout_lock_path(
        prerequisites,
        payload["execution_binding"]["source_inventory_sha256"],
    )

    with pytest.raises(ValueError, match="source inventory differs from manifest"):
        claim_holdout_once(prerequisites_path=prerequisites, prerequisites=payload)
    assert not lock.exists()


@pytest.mark.parametrize(
    "field",
    [
        "code_commit",
        "execution_argv_sha256",
        "execution_binding_sha256",
    ],
)
def test_a5_candidate_rejects_tampered_lock_execution_identity(
    tmp_path: Path,
    field: str,
) -> None:
    payload = _prerequisites(tmp_path)
    execution = payload["execution_binding"]
    assert isinstance(execution, dict)
    prerequisites = _claim_for_candidate(tmp_path, payload)
    lock = canonical_holdout_lock_path(
        prerequisites,
        payload["execution_binding"]["source_inventory_sha256"],
    )
    lock_payload = json.loads(lock.read_text(encoding="utf-8"))
    lock_payload[field] = "0" * (40 if field == "code_commit" else 64)
    lock.write_text(json.dumps(lock_payload), encoding="utf-8")

    with pytest.raises(RuntimeError, match="A5 unavailable"):
        candidate_main(_command(payload)[2:])
    assert not Path(str(execution["output_dir"])).exists()


def test_a5_candidate_rejects_model_changed_after_seal(tmp_path: Path) -> None:
    payload = _prerequisites(tmp_path)
    execution = payload["execution_binding"]
    assert isinstance(execution, dict)
    command = _command(payload)
    _claim_for_candidate(tmp_path, payload)
    Path(str(execution["model_path"])).write_text("different-model", encoding="utf-8")

    with pytest.raises(RuntimeError, match="A5 unavailable"):
        candidate_main(command[2:])
    assert not Path(str(execution["output_dir"])).exists()


def test_a5_candidate_is_unavailable_before_detector_or_output(
    tmp_path: Path,
) -> None:
    payload = _prerequisites(tmp_path)
    execution = payload["execution_binding"]
    assert isinstance(execution, dict)
    _claim_for_candidate(tmp_path, payload)
    with pytest.raises(RuntimeError, match="A5 unavailable"):
        candidate_main(_command(payload)[2:])

    assert not Path(str(execution["output_dir"])).exists()

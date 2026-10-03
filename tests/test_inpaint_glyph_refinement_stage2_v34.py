from __future__ import annotations

import hashlib
from pathlib import Path
import subprocess

import cv2
import numpy as np
import pytest

from benchmarking.inpaint_detector_bakeoff.contracts import mask_sha256
from benchmarking.inpaint_detector_bakeoff.incremental_plan import (
    composite_incremental_result,
)
import scripts.benchmark_inpaint_glyph_refinement_stage2_v34 as stage2_v34
from scripts.benchmark_inpaint_glyph_refinement_stage2_v34 import (
    MaskBinding,
    OUTPUT_ARTIFACT_ROLES,
    OUTPUT_INVENTORY_SCHEMA_VERSION,
    _canonical_sha256,
    _write_png,
    execute_stage2_page,
    reconstruct_page_plan,
    reopen_bound_mask,
    validate_cuda_bf16_diagnostics,
    validate_stage2_output_inventory,
)


def _image(shape: tuple[int, int] = (9, 10), value: int = 30) -> np.ndarray:
    return np.full((*shape, 3), value, dtype=np.uint8)


def _mask(shape: tuple[int, int] = (9, 10)) -> np.ndarray:
    return np.zeros(shape, dtype=np.uint8)


def _good_diagnostic() -> dict[str, object]:
    return {
        "actual_device": "cuda:0",
        "actual_precision": "bf16",
        "cpu_fallback_used": False,
        "device_verified_from_model": True,
        "status": "completed",
    }


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=root, text=True, stderr=subprocess.STDOUT
    ).strip()


def _init_git_repo(root: Path) -> None:
    _git(root, "init")
    _git(root, "config", "user.name", "Example Developer")
    _git(root, "config", "user.email", "developer@example.invalid")
    _git(root, "config", "core.autocrlf", "false")
    _git(root, "config", "commit.gpgsign", "false")
    (root / "seed.txt").write_bytes(b"seed\n")
    _git(root, "add", "seed.txt")
    _git(root, "commit", "-m", "test: seed fixture repository")


def test_context_additive_regenerates_context_but_commits_only_addition() -> None:
    source = _image()
    baseline = source.copy()
    baseline_mask = _mask()
    baseline_mask[3:6, 3:5] = 255
    baseline[baseline_mask > 0] = (80, 80, 80)
    commit = _mask()
    commit[3:6, 5:7] = 255
    generation = cv2.bitwise_or(baseline_mask, commit)
    plan = reconstruct_page_plan(
        mode="context_additive",
        generation_mask=generation,
        commit_mask=commit,
        replacement_source_edit=_mask(),
        baseline_mask=baseline_mask,
    )
    calls: list[np.ndarray] = []

    def inpaint(value: np.ndarray, mask: np.ndarray) -> np.ndarray:
        calls.append(mask.copy())
        result = value.copy()
        result[mask > 0] = (200, 200, 200)
        return result

    execution = execute_stage2_page(
        source=source,
        baseline=baseline,
        baseline_mask=baseline_mask,
        plan=plan,
        inpaint=inpaint,
        provider="cuda:0/source_lama_large/bf16",
    )

    assert len(calls) == 1
    assert np.array_equal(calls[0], generation)
    assert np.all(execution.candidate[baseline_mask > 0] == 80)
    assert np.all(execution.candidate[commit > 0] == 200)
    assert not np.any(
        (execution.incremental_changed > 0) & (commit == 0)
    )
    assert execution.receipt.request_count == 1


def test_narrow_replacement_restores_old_pr6_pixels_outside_new_commit() -> None:
    source = _image(value=20)
    baseline = source.copy()
    baseline_mask = _mask()
    baseline_mask[2:6, 2:5] = 255
    baseline[baseline_mask > 0] = (90, 90, 90)
    commit = _mask()
    commit[3:6, 4:7] = 255
    replacement = baseline_mask.copy()
    generation = cv2.bitwise_or(commit, replacement)
    plan = reconstruct_page_plan(
        mode="narrow_replacement",
        generation_mask=generation,
        commit_mask=commit,
        replacement_source_edit=replacement,
        baseline_mask=baseline_mask,
    )

    def inpaint(value: np.ndarray, mask: np.ndarray) -> np.ndarray:
        result = value.copy()
        result[mask > 0] = (170, 170, 170)
        return result

    execution = execute_stage2_page(
        source=source,
        baseline=baseline,
        baseline_mask=baseline_mask,
        plan=plan,
        inpaint=inpaint,
        provider="cuda:0/source_lama_large/bf16",
    )
    restored = (replacement > 0) & (commit == 0)
    assert np.array_equal(execution.candidate[restored], source[restored])
    assert np.all(execution.candidate[commit > 0] == 170)
    expected_mask = np.where(commit > 0, 255, 0).astype(np.uint8)
    assert np.array_equal(execution.final_mask, expected_mask)
    assert not np.any(
        (execution.incremental_changed > 0)
        & (execution.affected_union == 0)
    )


def test_empty_plan_is_byte_identical_and_makes_no_request() -> None:
    source = _image(value=15)
    baseline = source.copy()
    baseline_mask = _mask()
    baseline_mask[1:3, 1:4] = 255
    baseline[baseline_mask > 0] = (55, 55, 55)
    plan = reconstruct_page_plan(
        mode="conditional_segmenter",
        generation_mask=_mask(),
        commit_mask=_mask(),
        replacement_source_edit=_mask(),
        baseline_mask=baseline_mask,
    )

    def forbidden(_source: np.ndarray, _mask_value: np.ndarray) -> np.ndarray:
        raise AssertionError("empty page must not call LaMa")

    execution = execute_stage2_page(
        source=source,
        baseline=baseline,
        baseline_mask=baseline_mask,
        plan=plan,
        inpaint=forbidden,
        provider="cuda:0/source_lama_large/bf16",
    )
    assert execution.generated is None
    assert execution.receipt.request_count == 0
    assert np.array_equal(execution.candidate, baseline)
    assert np.array_equal(execution.final_mask, baseline_mask)


def test_plan_reconstruction_rejects_unbound_context_and_disconnected_replacement() -> None:
    baseline = _mask()
    baseline[1:3, 1:3] = 255
    commit = _mask()
    commit[5:7, 5:7] = 255
    generation = commit.copy()
    generation[0, 9] = 255
    with pytest.raises(ValueError, match="unbound context"):
        reconstruct_page_plan(
            mode="context_additive",
            generation_mask=generation,
            commit_mask=commit,
            replacement_source_edit=_mask(),
            baseline_mask=baseline,
        )

    generation = cv2.bitwise_or(commit, baseline)
    with pytest.raises(ValueError, match="must contact commit_mask"):
        reconstruct_page_plan(
            mode="narrow_replacement",
            generation_mask=generation,
            commit_mask=commit,
            replacement_source_edit=baseline,
            baseline_mask=baseline,
        )


def test_expected_final_mask_sha_rejects_mask_only_tamper() -> None:
    source = _image()
    baseline = source.copy()
    baseline_mask = _mask()
    commit = _mask()
    commit[4:6, 4:6] = 255
    plan = reconstruct_page_plan(
        mode="context_additive",
        generation_mask=commit,
        commit_mask=commit,
        replacement_source_edit=_mask(),
        baseline_mask=baseline_mask,
    )
    with pytest.raises(ValueError, match="mask-only scoring"):
        execute_stage2_page(
            source=source,
            baseline=baseline,
            baseline_mask=baseline_mask,
            plan=plan,
            inpaint=lambda value, _mask_value: value.copy(),
            provider="cuda:0/source_lama_large/bf16",
            expected_final_mask_sha256="0" * 64,
        )


def test_bound_mask_reopen_fails_closed_after_file_tamper(tmp_path: Path) -> None:
    value = _mask((6, 7))
    value[2:4, 3:5] = 255
    path = tmp_path / "mask.png"
    metadata = _write_png(path, value)
    binding = MaskBinding(
        page_id="example-page",
        role="commit_mask",
        shape=value.shape,
        pixel_sha256=mask_sha256(value),
        pixel_count=int(np.count_nonzero(value)),
        storage="artifact",
        artifact_id="example-page:commit_mask",
        artifact_path=path,
        artifact_file_sha256=str(metadata["file_sha256"]),
    )
    assert np.array_equal(reopen_bound_mask(binding), value)
    path.write_bytes(path.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="file SHA"):
        reopen_bound_mask(binding)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("actual_device", "cpu"),
        ("actual_precision", "fp32"),
        ("cpu_fallback_used", True),
        ("device_verified_from_model", False),
        ("status", "failed"),
    ],
)
def test_runtime_contract_rejects_non_cuda_bf16_evidence(
    field: str, value: object
) -> None:
    diagnostic = _good_diagnostic()
    diagnostic[field] = value
    with pytest.raises(RuntimeError, match="CUDA bf16"):
        validate_cuda_bf16_diagnostics(
            [diagnostic], expected_request_count=1
        )


def test_runtime_contract_requires_one_diagnostic_per_request() -> None:
    with pytest.raises(RuntimeError, match="diagnostic count"):
        validate_cuda_bf16_diagnostics([], expected_request_count=1)
    result = validate_cuda_bf16_diagnostics(
        [_good_diagnostic()], expected_request_count=1
    )
    assert result["runtime_telemetry_complete"] is True
    assert result["cpu_fallback_count"] == 0


def test_official_code_inventory_rejects_untracked_dependency(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _init_git_repo(tmp_path)
    (tmp_path / "dependency.py").write_bytes(b"VALUE = 1\n")
    monkeypatch.setattr(stage2_v34, "ROOT", tmp_path)
    monkeypatch.setattr(
        stage2_v34, "OFFICIAL_CODE_DEPENDENCIES", ("dependency.py",)
    )

    with pytest.raises(RuntimeError, match="not committed"):
        stage2_v34._committed_dependency_inventory()


def test_official_code_inventory_rejects_worktree_bytes_different_from_head(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _init_git_repo(tmp_path)
    dependency = tmp_path / "dependency.py"
    dependency.write_bytes(b"VALUE = 1\n")
    _git(tmp_path, "add", "dependency.py")
    _git(tmp_path, "commit", "-m", "test: add dependency fixture")
    dependency.write_bytes(b"VALUE = 2\n")
    monkeypatch.setattr(stage2_v34, "ROOT", tmp_path)
    monkeypatch.setattr(
        stage2_v34, "OFFICIAL_CODE_DEPENDENCIES", ("dependency.py",)
    )
    # Exercise the stronger HEAD-blob comparison independently of the earlier
    # tracked-worktree guard.
    monkeypatch.setattr(stage2_v34, "_tracked_worktree_clean", lambda: True)

    with pytest.raises(RuntimeError, match="differs from HEAD"):
        stage2_v34._committed_dependency_inventory()


def test_official_code_inventory_detects_clean_head_change_during_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _init_git_repo(tmp_path)
    dependency = tmp_path / "dependency.py"
    dependency.write_bytes(b"VALUE = 1\n")
    _git(tmp_path, "add", "dependency.py")
    _git(tmp_path, "commit", "-m", "test: add dependency fixture")
    monkeypatch.setattr(stage2_v34, "ROOT", tmp_path)
    monkeypatch.setattr(
        stage2_v34, "OFFICIAL_CODE_DEPENDENCIES", ("dependency.py",)
    )
    started = stage2_v34._committed_dependency_inventory()

    dependency.write_bytes(b"VALUE = 2\n")
    _git(tmp_path, "add", "dependency.py")
    _git(tmp_path, "commit", "-m", "test: simulate mid-run dependency update")

    with pytest.raises(RuntimeError, match="changed during the official run"):
        stage2_v34._assert_dependency_inventory_unchanged(started)


def test_output_inventory_binds_required_actual_bytes_and_detects_tamper(
    tmp_path: Path,
) -> None:
    page_id = "example-page"
    records: list[dict[str, object]] = []
    for role in OUTPUT_ARTIFACT_ROLES:
        value = _image((5, 6)) if role == "candidate_image" else _mask((5, 6))
        path = tmp_path / role / f"{page_id}.png"
        row = {
            "page_id": page_id,
            "role": role,
            "relative_path": path.relative_to(tmp_path).as_posix(),
            **_write_png(path, value),
        }
        source_roles = {
            "seed": "owned_positive_seed",
            "generation_mask": "generation_mask",
            "commit_mask": "commit_mask",
            "replacement_source_edit": "replacement_source_edit",
        }
        if role in source_roles:
            row["source_mask_binding"] = {
                "role": source_roles[role],
                "storage": "inline_zero",
                "artifact_id": None,
                "pixel_sha256": row["mask_pixel_sha256"],
                "pixel_count": row["pixel_count"],
            }
        records.append(row)
    unsigned = {
        "schema_version": OUTPUT_INVENTORY_SCHEMA_VERSION,
        "candidate_id": "b1_context_additive",
        "page_ids": [page_id],
        "records": records,
    }
    inventory = {**unsigned, "inventory_sha256": _canonical_sha256(unsigned)}
    validate_stage2_output_inventory(inventory, output_root=tmp_path)

    path = tmp_path / "candidate_image" / f"{page_id}.png"
    path.write_bytes(path.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="file SHA"):
        validate_stage2_output_inventory(inventory, output_root=tmp_path)


def test_receipt_detects_generated_pixel_tamper_before_composite() -> None:
    source = _image()
    baseline = source.copy()
    commit = _mask()
    commit[4:6, 4:6] = 255
    plan = reconstruct_page_plan(
        mode="context_additive",
        generation_mask=commit,
        commit_mask=commit,
        replacement_source_edit=_mask(),
        baseline_mask=_mask(),
    )
    execution = execute_stage2_page(
        source=source,
        baseline=baseline,
        baseline_mask=_mask(),
        plan=plan,
        inpaint=lambda value, mask: np.where(
            mask[:, :, None] > 0, np.uint8(200), value
        ).astype(np.uint8),
        provider="cuda:0/source_lama_large/bf16",
    )
    assert execution.receipt.generated_sha256 is not None
    assert execution.generated is not None
    tampered = execution.generated.copy()
    tampered[4, 4] = 1
    assert hashlib.sha256(tampered.tobytes()).hexdigest() != hashlib.sha256(
        execution.generated.tobytes()
    ).hexdigest()
    with pytest.raises(ValueError, match="generated-image SHA"):
        composite_incremental_result(
            source,
            baseline,
            tampered,
            _mask(),
            plan,
            execution.receipt,
        )

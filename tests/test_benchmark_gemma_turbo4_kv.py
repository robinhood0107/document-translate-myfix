from __future__ import annotations

import importlib.util
import hashlib
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for entry in (ROOT, SCRIPTS):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


runtime = _load("turbo4_lab_runtime_test", SCRIPTS / "turbo4_lab_runtime.py")
matrix = _load("benchmark_gemma_turbo4_kv_test", SCRIPTS / "benchmark_gemma_turbo4_kv.py")


def _candidate(key: str = "turbo4"):
    catalog = matrix.candidate_catalog(
        turbo_image="comic-translate/turbo4-lab:test",
        fork_commit=matrix.FORK_COMMIT,
    )
    return catalog[key]


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _response(translation: str) -> dict[str, object]:
    return {
        "choices": [
            {
                "index": 0,
                "content": json.dumps({"translation": translation}, ensure_ascii=False),
                "finish_reason": "stop",
            }
        ]
    }


def _result(
    *,
    key: str,
    seconds: float,
    quality: str = "quality",
    ledger: str = "ledger",
    mode: str = "replay",
):
    response = _response(quality)
    request_rows = [
        {
            "logical_request_index": 0,
            "model": "gemma-test",
            "prompt_sha256": _digest("prompt"),
            "schema_sha256": _digest("schema"),
            "seed": matrix.DEFAULT_SEED,
            "payload_sha256": _digest(ledger),
        }
    ]
    replay_rows = [{"logical_request_index": 0, **response}]
    if mode == "full-auto":
        attempts = [
            {
                "attempt_index": 0,
                "model": "gemma-test",
                "prompt_sha256": _digest("prompt"),
                "schema_sha256": _digest("schema"),
                "seed": matrix.DEFAULT_SEED,
                "payload_sha256": _digest(ledger),
            }
        ]
        response_rows = [{"attempt_index": 0, "canonical_response": response}]
        ledger_payload = {
            "fixed_contract": {"model": "gemma-test"},
            "translation_start_order": [],
            "actual_http_attempts": attempts,
        }
        request_ledger = {
            **ledger_payload,
            "runtime_variant": {"cache_type_v": "f16"},
            "response_ledger": {
                "rows": response_rows,
                "sha256": matrix._canonical_sha256(response_rows),
            },
            "record_complete": True,
            "record_failures": [],
            "sha256": matrix._canonical_sha256(ledger_payload),
        }
    else:
        request_ledger = {
            "rows": request_rows,
            "sha256": matrix._canonical_sha256(request_rows),
        }
    return {
        "candidate": {"key": key},
        "status": "passed",
        "request_wall_sec": seconds,
        "pipeline_wall_sec": seconds,
        "response_ledger": {
            "rows": replay_rows,
            "sha256": matrix._canonical_sha256(replay_rows),
        },
        "snapshot_sha256": _digest(f"snapshot:{quality}"),
        "pre_translation_snapshot_sha256": _digest(f"upstream:{quality}"),
        "page_output_sha256": [_digest(f"pixels:{quality}")],
        "request_ledger": request_ledger,
        "resource_gates": {"passed": True},
    }


def _semantic_pass(approval: dict[str, object]) -> dict[str, object]:
    approval["decision"] = "PASS"
    for comparison in approval["comparisons"]:
        comparison["reviewed_count"] = len(comparison["items"])
        comparison["unresolved_count"] = 0
        comparison["semantic_reject_count"] = 0
        for item in comparison["items"]:
            item.update(
                {
                    "decision": "PASS",
                    "review_scope": matrix.semantic_review.TEXT_ONLY_SCOPE,
                    "page_checked": False,
                    "requires_user_confirmation": False,
                    "classification": "style_or_register",
                    "rejection_category": "",
                    "attestations": {
                        name: True
                        for name in matrix.semantic_review._PASS_ATTESTATIONS
                    },
                }
            )
    return approval


def test_protocol_pins_the_approved_fork_and_disables_active_r3() -> None:
    protocol = matrix.load_protocol()

    assert protocol["fork"]["commit"] == matrix.FORK_COMMIT
    assert (
        protocol["translation_quality_gate"]["semantic_review_schema_version"]
        == matrix.semantic_review.SEMANTIC_REVIEW_SCHEMA_VERSION
    )
    assert protocol["safety"]["r3_residency_threshold"] == pytest.approx(0.90)
    assert protocol["safety"]["active_r3_execution"] is False
    assert {
        "full_auto_translation_response_semantic_review",
        "shipping_f16_replay_abba",
    } <= set(protocol["translation_quality_gate"]["hard_contracts"])
    assert set(protocol["telemetry_only"]) == {
        "gpu_background",
        "windows_available_ram",
        "wsl_swap",
        "container_swap",
        "shared_gpu",
    }
    assert {"qat", "mtp", "draft", "ngram", "speculative", "new_gguf"} <= set(protocol["forbidden"])


def test_candidate_catalog_locks_the_only_allowed_kv_difference() -> None:
    catalog = matrix.candidate_catalog(
        turbo_image="comic-translate/turbo4-lab:test",
        fork_commit=matrix.FORK_COMMIT,
    )
    f16 = runtime.Turbo4LabRuntimeConfig(
        protocol_version=matrix.PROTOCOL_VERSION,
        candidate_key="fork-f16",
        image_ref=catalog["fork-f16"].image_ref,
        model_volume=matrix.MODEL_VOLUME,
        model_name=matrix.MODEL_NAME,
        model_sha256=matrix.MODEL_SHA256,
        port=18081,
        container_name="ct-gemma-turbo4-fork-f16-test",
        cache_type_v="f16",
        fork_commit=matrix.FORK_COMMIT,
    )
    turbo = runtime.Turbo4LabRuntimeConfig(
        protocol_version=matrix.PROTOCOL_VERSION,
        candidate_key="turbo4",
        image_ref=catalog["turbo4"].image_ref,
        model_volume=matrix.MODEL_VOLUME,
        model_name=matrix.MODEL_NAME,
        model_sha256=matrix.MODEL_SHA256,
        port=18081,
        container_name="ct-gemma-turbo4-turbo4-test",
        cache_type_v="turbo4",
        fork_commit=matrix.FORK_COMMIT,
    )

    changed = [
        (left, right)
        for left, right in zip(f16.command, turbo.command)
        if left != right
    ]
    assert changed == [("f16", "turbo4")]
    assert "--spec-type" in turbo.command
    assert "draft" not in " ".join(turbo.command).lower()
    assert "mtp" not in " ".join(turbo.command).lower()


def test_runtime_config_and_cleanup_reject_product_container_names() -> None:
    with pytest.raises(runtime.Turbo4LabRuntimeError):
        runtime._assert_lab_name("gemma-local-server")

    with pytest.raises(runtime.Turbo4LabRuntimeError):
        runtime.Turbo4LabRuntimeConfig(
            protocol_version=matrix.PROTOCOL_VERSION,
            candidate_key="turbo4",
            image_ref="local:test",
            model_volume=matrix.MODEL_VOLUME,
            model_name=matrix.MODEL_NAME,
            model_sha256=matrix.MODEL_SHA256,
            port=18081,
            container_name="gemma-local-server",
            cache_type_v="turbo4",
            fork_commit=matrix.FORK_COMMIT,
        ).validate()


def test_model_identity_accepts_only_the_mounted_file_spellings() -> None:
    allowed = runtime._expected_model_identifiers(matrix.MODEL_NAME)

    assert allowed == {matrix.MODEL_NAME, f"/models/{matrix.MODEL_NAME}"}
    assert "/models/other.gguf" not in allowed


@pytest.mark.parametrize("identifier", [matrix.MODEL_NAME, f"/models/{matrix.MODEL_NAME}"])
def test_runtime_identity_accepts_both_llamacpp_model_spellings(monkeypatch, identifier: str) -> None:
    config = runtime.Turbo4LabRuntimeConfig(
        protocol_version=matrix.PROTOCOL_VERSION,
        candidate_key="shipping-f16",
        image_ref="local:test",
        model_volume=matrix.MODEL_VOLUME,
        model_name=matrix.MODEL_NAME,
        model_sha256=matrix.MODEL_SHA256,
        port=18081,
        container_name="ct-gemma-turbo4-model-id-test",
        cache_type_v="f16",
        fork_commit="",
    )
    manager = runtime.Turbo4LabRuntimeManager(inner=None, config=config)
    monkeypatch.setattr(runtime, "_http_json", lambda *_args, **_kwargs: {"data": [{"id": identifier}]})
    monkeypatch.setattr(
        runtime,
        "_inspect_container",
        lambda _name: {
            "Config": {
                "Cmd": config.command,
                "Labels": {
                    runtime.LAB_PROTOCOL_LABEL: config.protocol_version,
                    runtime.LAB_ROLE_LABEL: config.candidate_key,
                    runtime.LAB_COMMIT_LABEL: "shipping",
                },
            },
            "Mounts": [
                {
                    "Destination": "/models",
                    "Type": "volume",
                    "Name": config.model_volume,
                    "RW": False,
                }
            ],
            "Image": "",
        },
    )

    manager._validate_model_identity()


def test_startup_failure_requires_confirmed_gpu_release(monkeypatch) -> None:
    config = runtime.Turbo4LabRuntimeConfig(
        protocol_version=matrix.PROTOCOL_VERSION,
        candidate_key="turbo4",
        image_ref="comic-translate/turbo4-lab:test",
        model_volume=matrix.MODEL_VOLUME,
        model_name=matrix.MODEL_NAME,
        model_sha256=matrix.MODEL_SHA256,
        port=18081,
        container_name="ct-gemma-turbo4-startup-release-test",
        cache_type_v="turbo4",
        fork_commit=matrix.FORK_COMMIT,
    )
    manager = runtime.Turbo4LabRuntimeManager(inner=None, config=config)
    removed: list[bool] = []
    waited: list[bool] = []

    monkeypatch.setattr(runtime, "_inspect_container", lambda _name: {})
    monkeypatch.setattr(runtime, "_gpu_used_mib", lambda: 100)
    monkeypatch.setattr(
        runtime,
        "_run",
        lambda *_args, **_kwargs: __import__("subprocess").CompletedProcess([], 0, "", ""),
    )
    monkeypatch.setattr(manager, "_wait_for_ready", lambda **_kwargs: None)

    def fail_identity() -> None:
        raise runtime.Turbo4LabRuntimeError("identity mismatch")

    monkeypatch.setattr(manager, "_validate_model_identity", fail_identity)
    monkeypatch.setattr(manager, "_remove_exact_container", lambda: removed.append(True))
    monkeypatch.setattr(
        manager,
        "_wait_for_gpu_release",
        lambda **_kwargs: waited.append(True)
        or {"status": "timeout", "observed": False},
    )

    with pytest.raises(runtime.Turbo4LabRuntimeError, match="GPU release was not confirmed"):
        manager.ensure_server(object())

    assert removed == [True]
    assert waited == [True]
    assert manager.evidence()["release"] == {"status": "timeout", "observed": False}
    assert manager.shutdown() == {"status": "timeout", "observed": False}


def test_fork_ref_must_match_the_immutable_protocol_commit(monkeypatch) -> None:
    monkeypatch.setattr(
        matrix,
        "_run",
        lambda *_args, **_kwargs: __import__("subprocess").CompletedProcess(
            [], 0, stdout=matrix.FORK_COMMIT + "\trefs/heads/x\n", stderr=""
        ),
    )
    assert matrix.verify_fork_ref()["commit"] == matrix.FORK_COMMIT

    with pytest.raises(matrix.Turbo4BenchmarkError):
        matrix.verify_fork_ref(fork_commit="0" * 40)


def test_model_volume_identity_accepts_a_windows_utf8_bom(monkeypatch, tmp_path: Path) -> None:
    calls = 0
    manifest = {
        "files": [
            {
                "name": matrix.MODEL_NAME,
                "sha256": matrix.MODEL_SHA256,
                "bytes": 123,
            }
        ]
    }

    def fake_run(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            return __import__("subprocess").CompletedProcess([], 0, '{"Name":"models"}', "")
        return __import__("subprocess").CompletedProcess(
            [],
            0,
            "__MANIFEST__\n\ufeff"
            + json.dumps(manifest)
            + "\n__BYTES__\n123\n__SHA256__\n"
            + matrix.MODEL_SHA256
            + "  /models/"
            + matrix.MODEL_NAME
            + "\n",
            "",
        )

    monkeypatch.setattr(matrix, "_run", fake_run)

    evidence = matrix.verify_model_volume(output_dir=tmp_path)

    assert evidence["model_sha256"] == matrix.MODEL_SHA256
    assert evidence["observed_model_sha256"] == matrix.MODEL_SHA256
    assert evidence["model_bytes"] == 123


def test_model_volume_identity_rejects_a_mounted_file_with_the_wrong_sha(
    monkeypatch,
    tmp_path: Path,
) -> None:
    manifest = {
        "files": [
            {
                "name": matrix.MODEL_NAME,
                "sha256": matrix.MODEL_SHA256,
                "bytes": 123,
            }
        ]
    }
    responses = iter(
        [
            __import__("subprocess").CompletedProcess([], 0, '{"Name":"models"}', ""),
            __import__("subprocess").CompletedProcess(
                [],
                0,
                "__MANIFEST__\n"
                + json.dumps(manifest)
                + "\n__BYTES__\n123\n__SHA256__\n"
                + "0" * 64
                + "  /models/"
                + matrix.MODEL_NAME
                + "\n",
                "",
            ),
        ]
    )
    monkeypatch.setattr(matrix, "_run", lambda *_args, **_kwargs: next(responses))

    with pytest.raises(matrix.Turbo4BenchmarkError):
        matrix.verify_model_volume(output_dir=tmp_path)


def test_prepared_runtime_manifest_revalidates_immutable_image_ids(monkeypatch, tmp_path: Path) -> None:
    image_tag = "comic-translate/turbo4-lab:test"
    manifest_path = tmp_path / "runtime-manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "fork": {"commit": matrix.FORK_COMMIT},
                "shipping_image": {"id": "sha256:shipping"},
                "turbo_image": {
                    "image": {"reference": image_tag, "id": "sha256:turbo"},
                    "fork_commit": matrix.FORK_COMMIT,
                    "help_sha256": "a" * 64,
                    "turbo4_help_present": True,
                },
                "model_volume": {
                    "model_name": matrix.MODEL_NAME,
                    "model_sha256": matrix.MODEL_SHA256,
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        matrix,
        "verify_fork_ref",
        lambda **_kwargs: {"commit": matrix.FORK_COMMIT},
    )
    monkeypatch.setattr(
        matrix,
        "_inspect_image",
        lambda image: {
            "reference": image,
            "id": "sha256:shipping" if image == matrix.SHIPPING_IMAGE else "sha256:turbo",
        },
    )

    result = matrix._load_prepared_runtime_manifest(
        manifest_path,
        image_tag=image_tag,
        fork_commit=matrix.FORK_COMMIT,
    )

    assert result["turbo_image"]["reused_prepared_validation"] is True
    assert result["build"]["reused"] is True


def test_translation_replay_requires_fixed_seed_and_rejects_draft(tmp_path: Path) -> None:
    valid = tmp_path / "valid.json"
    payload = {
        "requests": [
            {
                "model": matrix.MODEL_NAME,
                "seed": matrix.DEFAULT_SEED,
                "messages": [{"role": "user", "content": "x"}],
                "response_format": {"type": "json_object"},
            }
        ]
    }
    valid.write_text(json.dumps(payload), encoding="utf-8")
    assert matrix.load_translation_replay(valid)[0]["seed"] == matrix.DEFAULT_SEED

    payload["requests"][0]["seed"] = 7
    valid.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(matrix.Turbo4BenchmarkError):
        matrix.load_translation_replay(valid)

    payload["requests"][0]["seed"] = matrix.DEFAULT_SEED
    payload["requests"][0]["metadata"] = "draft"
    valid.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(matrix.Turbo4BenchmarkError):
        matrix.load_translation_replay(valid)


def test_page_snapshot_replay_preserves_contextual_single_request_order(tmp_path: Path) -> None:
    snapshots = tmp_path / "page_snapshots.json"
    snapshots.write_text(
        json.dumps(
            {
                "pages": [
                    {
                        "blocks": [
                            {"text": "first"},
                            {"text": ""},
                            {"text": "second"},
                        ]
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    requests = matrix.build_translation_replay_from_page_snapshots(snapshots)

    assert len(requests) == 2
    assert [request["seed"] for request in requests] == [matrix.DEFAULT_SEED] * 2
    assert all(request["model"] == matrix.MODEL_NAME for request in requests)
    user_prompts = [request["messages"][1]["content"][0]["text"] for request in requests]
    assert '"target_block": "block_0"' in user_prompts[0]
    assert '"target_block": "block_1"' in user_prompts[1]
    assert all(request["response_format"]["type"] == "json_schema" for request in requests)


def test_resource_gate_keeps_host_memory_and_swap_as_observations() -> None:
    sampler = object.__new__(matrix.ResourceSampler)
    sampler.samples = [
        {
            "gpu": {"available": True, "used_mib": 100},
            "windows_available_bytes": 6 * 1024**3,
            "wsl_swap_used_bytes": 100,
            "shared_gpu_used_mib": 0.0,
        },
        {
            "gpu": {"available": True, "used_mib": 120},
            "windows_available_bytes": 6 * 1024**3 - 1,
            "wsl_swap_used_bytes": 101,
            "shared_gpu_used_mib": 0.5,
        },
    ]

    gate = matrix.resource_gate_report(sampler=sampler, cgroup_swap_peak_bytes=1)
    assert gate["passed"] is True
    assert gate["failures"] == []
    assert set(gate["observations"]) >= {
        "windows_available_ram_below_6gib",
        "wsl_swap_growth_observed",
        "shared_gpu_growth_observed",
        "container_swap_observed",
    }


def test_resource_emergency_reason_never_stops_for_host_memory_or_swap() -> None:
    samples = [
        {
            "windows_available_bytes": 6 * 1024**3,
            "wsl_swap_used_bytes": 100,
            "shared_gpu_used_mib": 0.0,
        },
        {
            "windows_available_bytes": 1,
            "wsl_swap_used_bytes": 101,
            "shared_gpu_used_mib": 10.0,
        },
    ]

    assert matrix._resource_emergency_reason(samples) is None


def test_structural_gate_stops_before_next_candidate_on_oom(
    monkeypatch,
    tmp_path: Path,
) -> None:
    catalog = matrix.candidate_catalog(
        turbo_image="comic-translate/turbo4-lab:test",
        fork_commit=matrix.FORK_COMMIT,
    )
    calls: list[str] = []

    def fake_execute(candidate, **_kwargs):
        calls.append(candidate.key)
        return {
            "status": "rejected",
            "resource_gates": {
                "passed": False,
                "failures": ["oom_detected"],
            },
        }

    monkeypatch.setattr(matrix, "execute_replay_once", fake_execute)

    result = matrix.execute_structural_gate(
        candidates=catalog,
        payloads=[],
        output_dir=tmp_path,
        port=18081,
        image_ids={key: "sha256:test" for key in catalog},
    )

    assert calls == ["fork-f16"]
    assert result["decision"] == "REJECT"
    assert result["aborted_early"] is True


def test_preflight_keeps_low_ram_and_missing_shared_counter_as_observations(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        matrix,
        "resource_preflight",
        lambda: {
            "gpu": {"available": True, "used_mib": 100},
            "windows_available_bytes": 1,
            "wsl_swap_used_bytes": 0,
            "failures": ["windows_available_ram_below_6gib"],
            "passed": False,
        },
    )
    monkeypatch.setattr(matrix, "query_shared_gpu_used_mib", lambda: None)
    monkeypatch.setattr(matrix, "_running_container_names", lambda: [])

    result = matrix.require_preflight(output_dir=tmp_path)

    assert result["passed"] is True
    assert set(result["observations"]) == {
        "windows_available_ram_below_6gib",
        "windows_shared_gpu_metrics_unavailable",
    }
    assert result["settle"]["attempt_count"] == 1
    assert result["settle"]["host_memory_and_swap"] == "telemetry_only"


def test_preflight_keeps_gpu_background_and_swap_metric_as_observations(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        matrix,
        "resource_preflight",
        lambda: {
            "gpu": {"available": True, "used_mib": 4096},
            "windows_available_bytes": 8 * 1024**3,
            "wsl_swap_used_bytes": None,
            "failures": ["gpu_background_above_2gib", "wsl_swap_metrics_unavailable"],
            "passed": False,
        },
    )
    monkeypatch.setattr(matrix, "query_shared_gpu_used_mib", lambda: 0.0)
    monkeypatch.setattr(matrix, "_running_container_names", lambda: [])

    result = matrix.require_preflight(output_dir=tmp_path)

    assert result["passed"] is True
    assert set(result["observations"]) == {
        "gpu_background_above_2gib",
        "wsl_swap_metrics_unavailable",
    }


def test_structural_gate_does_not_abort_for_host_observations(monkeypatch, tmp_path: Path) -> None:
    catalog = matrix.candidate_catalog(
        turbo_image="comic-translate/turbo4-lab:test",
        fork_commit=matrix.FORK_COMMIT,
    )
    calls: list[str] = []

    def fake_execute(candidate, **_kwargs):
        calls.append(candidate.key)
        result = _result(key=candidate.key, seconds=1.0, quality="same", ledger="same")
        result["resource_gates"] = {
            "passed": True,
            "failures": [],
            "observations": [
                "windows_available_ram_below_6gib",
                "wsl_swap_growth_observed",
                "shared_gpu_growth_observed",
                "container_swap_observed",
            ],
        }
        return result

    monkeypatch.setattr(matrix, "execute_replay_once", fake_execute)
    result = matrix.execute_structural_gate(
        candidates=catalog,
        payloads=[],
        output_dir=tmp_path,
        port=18081,
        image_ids={key: "sha256:test" for key in catalog},
    )

    assert calls == ["fork-f16", "turbo4", "shipping-f16"]
    assert result["decision"] == "PASS"


def test_shipping_replay_abba_runs_after_the_fork_only_gate(monkeypatch, tmp_path: Path) -> None:
    catalog = matrix.candidate_catalog(
        turbo_image="comic-translate/turbo4-lab:test",
        fork_commit=matrix.FORK_COMMIT,
    )
    calls: list[str] = []

    def fake_execute(candidate, **_kwargs):
        calls.append(candidate.key)
        return _result(
            key=candidate.key,
            seconds=1.0 if candidate.key == "shipping-f16" else 0.9,
            quality="same",
            ledger="same",
        )

    monkeypatch.setattr(matrix, "execute_replay_once", fake_execute)
    result = matrix.execute_shipping_replay_abba(
        candidates=catalog,
        payloads=[],
        output_dir=tmp_path,
        port=18081,
        image_ids={key: "sha256:test" for key in catalog},
        initial_rounds=2,
        max_rounds=2,
    )

    assert calls == ["shipping-f16", "turbo4", "turbo4", "shipping-f16"]
    assert result["stage"] == "shipping-f16-vs-turbo4-replay"
    assert result["decision"] == "PASS"


def test_pair_matrix_is_abba_and_stops_at_two_for_clear_win(tmp_path: Path) -> None:
    calls: list[str] = []

    def execute(candidate, _run_dir, _round):
        calls.append(candidate.key)
        return _result(
            key=candidate.key,
            seconds=10.0 if candidate.key == "shipping-f16" else 9.0,
        )

    result = matrix.execute_pair_matrix(
        baseline=_candidate("shipping-f16"),
        candidate=_candidate("turbo4"),
        mode="replay",
        output_dir=tmp_path,
        execute=execute,
        initial_rounds=2,
        max_rounds=7,
    )

    assert calls == ["shipping-f16", "turbo4", "turbo4", "shipping-f16"]
    assert result["round_count"] == 2
    assert result["decision"] == "PASS"


def test_pair_matrix_collects_two_runs_then_requires_review_for_raw_response_mismatch(tmp_path: Path) -> None:
    calls: list[str] = []

    def execute(candidate, _run_dir, round_index):
        calls.append(candidate.key)
        quality = "baseline" if candidate.key == "shipping-f16" else "candidate"
        return _result(
            key=candidate.key,
            seconds=10.0 if candidate.key == "shipping-f16" else 9.0,
            quality=quality,
            ledger="same",
        )

    result = matrix.execute_pair_matrix(
        baseline=_candidate("shipping-f16"),
        candidate=_candidate("turbo4"),
        mode="replay",
        output_dir=tmp_path,
        execute=execute,
        initial_rounds=2,
        max_rounds=7,
    )

    assert len(calls) == 4
    assert result["round_count"] == 2
    assert result["quality_exact"] is False
    assert result["raw_response_exact"] is False
    assert result["semantic_review_status"] == "REVIEW_REQUIRED"
    assert result["decision"] == "REVIEW_REQUIRED"


def test_pair_matrix_allows_abba_speed_evidence_after_semantic_approval(tmp_path: Path) -> None:
    calls: list[str] = []

    def execute(candidate, _run_dir, _round):
        calls.append(candidate.key)
        quality = "baseline" if candidate.key == "shipping-f16" else "candidate"
        return _result(
            key=candidate.key,
            seconds=10.0 if candidate.key == "shipping-f16" else 9.0,
            quality=quality,
            ledger="same",
        )

    pending = matrix.execute_pair_matrix(
        baseline=_candidate("shipping-f16"),
        candidate=_candidate("turbo4"),
        mode="replay",
        output_dir=tmp_path / "pending",
        execute=execute,
        initial_rounds=2,
        max_rounds=7,
        semantic_stage="replay-abba",
    )
    approval = _semantic_pass(pending["semantic_review"]["template"])
    result = matrix.execute_pair_matrix(
        baseline=_candidate("shipping-f16"),
        candidate=_candidate("turbo4"),
        mode="replay",
        output_dir=tmp_path / "approved",
        execute=execute,
        initial_rounds=2,
        max_rounds=7,
        semantic_approval=approval,
        semantic_stage="replay-abba",
    )

    assert len(calls) >= 4
    assert result["semantic_review_status"] == "PASS"
    assert result["hard_contract_pass"] is True
    assert result["decision"] == "PASS"


def test_full_auto_allows_final_pixel_difference_after_upstream_parity(tmp_path: Path) -> None:
    def execute(candidate, _run_dir, _round):
        result = _result(
            key=candidate.key,
            seconds=10.0 if candidate.key == "shipping-f16" else 9.0,
            quality="response",
            ledger="same",
            mode="full-auto",
        )
        result["pre_translation_snapshot_sha256"] = _digest("upstream-exact")
        result["page_output_sha256"] = [
            _digest(
                "baseline-render" if candidate.key == "shipping-f16" else "candidate-render"
            )
        ]
        return result

    result = matrix.execute_pair_matrix(
        baseline=_candidate("shipping-f16"),
        candidate=_candidate("turbo4"),
        mode="full-auto",
        output_dir=tmp_path,
        execute=execute,
        initial_rounds=2,
        max_rounds=7,
    )

    assert result["pre_translation_exact"] is True
    assert result["final_output_exact"] is False
    assert result["final_output_mismatches"]
    assert result["hard_contract_pass"] is True
    assert result["decision"] == "PASS"


def test_full_auto_translation_delta_requires_hash_bound_semantic_review(tmp_path: Path) -> None:
    def execute(candidate, _run_dir, _round):
        result = _result(
            key=candidate.key,
            seconds=10.0 if candidate.key == "shipping-f16" else 9.0,
            quality="baseline" if candidate.key == "shipping-f16" else "candidate",
            ledger="same",
            mode="full-auto",
        )
        result["pre_translation_snapshot_sha256"] = _digest("upstream-exact")
        return result

    result = matrix.execute_pair_matrix(
        baseline=_candidate("shipping-f16"),
        candidate=_candidate("turbo4"),
        mode="full-auto",
        output_dir=tmp_path,
        execute=execute,
        initial_rounds=2,
        max_rounds=7,
        semantic_stage="s1",
    )

    assert result["semantic_review_status"] == "REVIEW_REQUIRED"
    assert result["semantic_response_mismatches"]
    assert result["decision"] == "REVIEW_REQUIRED"


def test_full_auto_final_output_diagnostic_uses_decoded_pixels_not_snapshot_text(tmp_path: Path) -> None:
    def execute(candidate, _run_dir, _round):
        result = _result(
            key=candidate.key,
            seconds=10.0 if candidate.key == "shipping-f16" else 9.0,
            quality="response",
            ledger="same",
            mode="full-auto",
        )
        result["pre_translation_snapshot_sha256"] = _digest("upstream-exact")
        result["snapshot_sha256"] = _digest(candidate.key)
        result["page_output_sha256"] = [_digest("same-decoded-pixels")]
        return result

    result = matrix.execute_pair_matrix(
        baseline=_candidate("shipping-f16"),
        candidate=_candidate("turbo4"),
        mode="full-auto",
        output_dir=tmp_path,
        execute=execute,
        initial_rounds=2,
        max_rounds=7,
    )

    assert result["final_output_exact"] is True


def test_full_auto_rejects_upstream_snapshot_difference(tmp_path: Path) -> None:
    def execute(candidate, _run_dir, _round):
        result = _result(
            key=candidate.key,
            seconds=9.0,
            quality="response",
            ledger="same",
            mode="full-auto",
        )
        result["pre_translation_snapshot_sha256"] = _digest(candidate.key)
        return result

    result = matrix.execute_pair_matrix(
        baseline=_candidate("shipping-f16"),
        candidate=_candidate("turbo4"),
        mode="full-auto",
        output_dir=tmp_path,
        execute=execute,
        initial_rounds=2,
        max_rounds=7,
    )

    assert result["pre_translation_exact"] is False
    assert result["hard_contract_pass"] is False
    assert result["decision"] == "REJECT"


def test_pair_matrix_keeps_request_ledger_difference_as_a_hard_reject(tmp_path: Path) -> None:
    def execute(candidate, _run_dir, _round):
        return _result(
            key=candidate.key,
            seconds=9.0,
            quality="same",
            ledger="baseline" if candidate.key == "shipping-f16" else "candidate",
        )

    result = matrix.execute_pair_matrix(
        baseline=_candidate("shipping-f16"),
        candidate=_candidate("turbo4"),
        mode="replay",
        output_dir=tmp_path,
        execute=execute,
        initial_rounds=2,
        max_rounds=7,
    )

    assert result["request_ledger_exact"] is False
    assert result["hard_contract_pass"] is False
    assert result["decision"] == "REJECT"


def test_pair_matrix_stops_before_the_other_arm_after_a_fatal_resource_failure(
    tmp_path: Path,
) -> None:
    calls: list[str] = []

    def execute(candidate, _run_dir, _round):
        calls.append(candidate.key)
        result = _result(key=candidate.key, seconds=1.0)
        result["status"] = "rejected"
        result["resource_gates"] = {
            "passed": False,
            "failures": ["oom_detected"],
        }
        return result

    result = matrix.execute_pair_matrix(
        baseline=_candidate("shipping-f16"),
        candidate=_candidate("turbo4"),
        mode="replay",
        output_dir=tmp_path,
        execute=execute,
        initial_rounds=2,
        max_rounds=7,
    )

    assert calls == ["shipping-f16"]
    assert result["round_count"] == 1
    assert result["aborted_early"] is True
    assert result["fatal_resource_failures"] == ["oom_detected"]
    assert result["decision"] == "REJECT"


def test_pair_matrix_caps_an_unproven_candidate_at_seven_rounds(tmp_path: Path) -> None:
    calls: list[str] = []

    def execute(candidate, _run_dir, round_index):
        calls.append(candidate.key)
        candidate_seconds = 11.0 if candidate.key == "turbo4" else 10.0
        return _result(key=candidate.key, seconds=candidate_seconds)

    result = matrix.execute_pair_matrix(
        baseline=_candidate("shipping-f16"),
        candidate=_candidate("turbo4"),
        mode="replay",
        output_dir=tmp_path,
        execute=execute,
        initial_rounds=2,
        max_rounds=100,
    )

    assert result["round_count"] == matrix.DEFAULT_MAX_ROUNDS
    assert len(calls) == matrix.DEFAULT_MAX_ROUNDS * 2
    assert result["decision"] == "REJECT"


def test_structural_gate_requires_shipping_and_fork_controls(monkeypatch, tmp_path: Path) -> None:
    def fake_execute(candidate, **_kwargs):
        return _result(key=candidate.key, seconds=1.0, quality="same", ledger="same")

    monkeypatch.setattr(matrix, "execute_replay_once", fake_execute)
    catalog = matrix.candidate_catalog(
        turbo_image="comic-translate/turbo4-lab:test",
        fork_commit=matrix.FORK_COMMIT,
    )
    result = matrix.execute_structural_gate(
        candidates=catalog,
        payloads=[
            {
                "model": matrix.MODEL_NAME,
                "seed": matrix.DEFAULT_SEED,
                "messages": [],
                "response_format": {"type": "json_object"},
            }
        ],
        output_dir=tmp_path,
        port=18081,
        image_ids={key: "sha256:test" for key in catalog},
    )

    assert result["decision"] == "PASS"
    assert list(result["runs"]) == ["fork-f16", "turbo4", "shipping-f16"]


def test_structural_gate_returns_review_required_for_content_only_differences(
    monkeypatch,
    tmp_path: Path,
) -> None:
    def fake_execute(candidate, **_kwargs):
        quality = "shipping" if candidate.key == "shipping-f16" else candidate.key
        return _result(key=candidate.key, seconds=1.0, quality=quality, ledger="same")

    monkeypatch.setattr(matrix, "execute_replay_once", fake_execute)
    catalog = matrix.candidate_catalog(
        turbo_image="comic-translate/turbo4-lab:test",
        fork_commit=matrix.FORK_COMMIT,
    )
    result = matrix.execute_structural_gate(
        candidates=catalog,
        payloads=[],
        output_dir=tmp_path,
        port=18081,
        image_ids={key: "sha256:test" for key in catalog},
    )

    assert result["decision"] == "REVIEW_REQUIRED"
    assert result["raw_reproducibility"]["status"] == "REVIEW_REQUIRED"
    assert result["semantic_review"]["status"] == "REVIEW_REQUIRED"
    assert (tmp_path / "semantic-review-template.json").is_file()


def test_structural_gate_accepts_a_hash_bound_semantic_approval(monkeypatch, tmp_path: Path) -> None:
    def fake_execute(candidate, **_kwargs):
        quality = "shipping" if candidate.key == "shipping-f16" else candidate.key
        return _result(key=candidate.key, seconds=1.0, quality=quality, ledger="same")

    monkeypatch.setattr(matrix, "execute_replay_once", fake_execute)
    catalog = matrix.candidate_catalog(
        turbo_image="comic-translate/turbo4-lab:test",
        fork_commit=matrix.FORK_COMMIT,
    )
    pending = matrix.execute_structural_gate(
        candidates=catalog,
        payloads=[],
        output_dir=tmp_path / "pending",
        port=18081,
        image_ids={key: "sha256:test" for key in catalog},
    )
    approval = json.loads(
        (tmp_path / "pending" / "semantic-review-template.json").read_text(encoding="utf-8")
    )
    approval = _semantic_pass(approval)

    approved = matrix.execute_structural_gate(
        candidates=catalog,
        payloads=[],
        output_dir=tmp_path / "approved",
        port=18081,
        image_ids={key: "sha256:test" for key in catalog},
        semantic_approval=approval,
    )

    assert pending["decision"] == "REVIEW_REQUIRED"
    assert approved["decision"] == "PASS"
    assert approved["semantic_review"]["status"] == "PASS"


def test_structural_resume_approves_stored_review_without_gpu_replay(monkeypatch, tmp_path: Path) -> None:
    def fake_execute(candidate, **_kwargs):
        quality = "shipping" if candidate.key == "shipping-f16" else candidate.key
        return _result(key=candidate.key, seconds=1.0, quality=quality, ledger="same")

    monkeypatch.setattr(matrix, "execute_replay_once", fake_execute)
    catalog = matrix.candidate_catalog(
        turbo_image="comic-translate/turbo4-lab:test",
        fork_commit=matrix.FORK_COMMIT,
    )
    image_ids = {key: "sha256:test" for key in catalog}
    matrix.execute_structural_gate(
        candidates=catalog,
        payloads=[],
        output_dir=tmp_path,
        port=18081,
        image_ids=image_ids,
    )
    approval = json.loads(
        (tmp_path / "semantic-review-template.json").read_text(encoding="utf-8")
    )
    approval = _semantic_pass(approval)

    approved = matrix._load_or_approve_structural_gate(
        tmp_path / "structural-summary.json",
        image_ids=image_ids,
        semantic_approval=approval,
    )

    assert approved["decision"] == "PASS"
    assert approved["semantic_review"]["status"] == "PASS"


def test_structural_gate_keeps_truncated_response_as_a_hard_reject(monkeypatch, tmp_path: Path) -> None:
    def fake_execute(candidate, **_kwargs):
        result = _result(key=candidate.key, seconds=1.0, quality="same", ledger="same")
        if candidate.key == "turbo4":
            result["response_ledger"]["rows"][0]["choices"][0]["finish_reason"] = "length"
            result["response_ledger"]["sha256"] = matrix._canonical_sha256(
                result["response_ledger"]["rows"]
            )
        return result

    monkeypatch.setattr(matrix, "execute_replay_once", fake_execute)
    catalog = matrix.candidate_catalog(
        turbo_image="comic-translate/turbo4-lab:test",
        fork_commit=matrix.FORK_COMMIT,
    )
    result = matrix.execute_structural_gate(
        candidates=catalog,
        payloads=[],
        output_dir=tmp_path,
        port=18081,
        image_ids={key: "sha256:test" for key in catalog},
    )

    assert result["decision"] == "REJECT"
    assert result["hard_contract_failures"]


def test_r3_estimate_uses_90_percent_only_and_never_runs_it() -> None:
    result = matrix.r3_estimate_from_results(
        [
            {
                "candidate": {"key": "turbo4"},
                "resource_gates": {"gpu_peak_mib": 10_000},
                "preflight": {"gpu": {"used_mib": 500, "total_mib": 12_282}},
            }
        ]
    )

    assert result["available"] is True
    assert result["threshold"] == pytest.approx(0.90)
    assert result["active_r3_executed"] is False
    assert result["combined_peak_mib"] == matrix.PADDLE_MEASURED_PEAK_MIB + 9_500


def test_full_auto_preset_isolates_every_candidate_in_the_lab_adapter() -> None:
    shipping = matrix._full_auto_preset(
        _candidate("shipping-f16"), port=18081, round_index=1, image_id="sha256:shipping"
    )
    turbo = matrix._full_auto_preset(
        _candidate("turbo4"), port=18081, round_index=1, image_id="sha256:turbo"
    )

    assert shipping["benchmark_turbo4_kv"]["cache_type_v"] == "f16"
    assert turbo["benchmark_turbo4_kv"]["cache_type_v"] == "turbo4"
    assert shipping["gemma"]["endpoint_url"].endswith(":18081/v1")
    assert turbo["gemma"]["endpoint_url"].endswith(":18081/v1")
    assert turbo["benchmark_http"]["gemma_seed"] == matrix.DEFAULT_SEED


def test_full_auto_request_ledger_hashes_actual_requests_not_cache_variant(tmp_path: Path) -> None:
    records = tmp_path / "gemma-http-records.jsonl"
    request = {
        "model": matrix.MODEL_NAME,
        "seed": matrix.DEFAULT_SEED,
        "messages": [{"role": "user", "content": "fixture"}],
        "response_format": {"type": "json_object"},
    }
    records.write_text(
        json.dumps(
            {
                "attempt_index": 0,
                "request": request,
                "response": {
                    "choices": [
                        {
                            "index": 0,
                            "message": {"content": '{"translation":"fixture"}'},
                            "finish_reason": "stop",
                        }
                    ]
                },
                "status_code": 200,
                "error": "",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with records.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {"record_type": "summary", "attempt_count": 1, "write_error": ""}
            )
            + "\n"
        )
    metrics = tmp_path / "metrics.jsonl"
    metrics.write_text(
        json.dumps({"tag": "translate_start", "image_index": 0, "block_count": 1}) + "\n",
        encoding="utf-8",
    )
    fixed = {
        "gemma": {
            "model": matrix.MODEL_NAME,
            "context_size": 4096,
            "n_gpu_layers": 23,
            "n_parallel": 1,
            "threads": 10,
            "batch_size": 2048,
            "ubatch_size": 512,
            "cache_type_k": "f16",
            "cache_type_v": "f16",
            "spec_type": "none",
        },
        "benchmark_http": {"gemma_seed": matrix.DEFAULT_SEED},
    }
    candidate = json.loads(json.dumps(fixed))
    candidate["gemma"]["cache_type_v"] = "turbo4"

    baseline_ledger = matrix._pipeline_request_ledger(
        preset=fixed,
        metrics_path=metrics,
        record_path=records,
    )
    candidate_ledger = matrix._pipeline_request_ledger(
        preset=candidate,
        metrics_path=metrics,
        record_path=records,
    )

    assert baseline_ledger["record_complete"] is True
    assert baseline_ledger["sha256"] == candidate_ledger["sha256"]
    assert baseline_ledger["runtime_variant"]["cache_type_v"] == "f16"
    assert candidate_ledger["runtime_variant"]["cache_type_v"] == "turbo4"


def test_full_auto_request_ledger_rejects_missing_attempt_indexes(tmp_path: Path) -> None:
    request = {
        "model": matrix.MODEL_NAME,
        "seed": matrix.DEFAULT_SEED,
        "messages": [{"role": "user", "content": "fixture"}],
        "response_format": {"type": "json_object"},
    }
    records = tmp_path / "gemma-http-records.jsonl"
    records.write_text(
        "\n".join(
            json.dumps(
                {
                    "attempt_index": index,
                    "request": request,
                    "response": {"choices": []},
                    "status_code": 200,
                    "error": "",
                }
            )
            for index in (0, 2)
        )
        + "\n"
        + json.dumps(
            {"record_type": "summary", "attempt_count": 3, "write_error": ""}
        )
        + "\n",
        encoding="utf-8",
    )

    ledger = matrix._pipeline_request_ledger(
        preset={"gemma": {}, "benchmark_http": {"gemma_seed": matrix.DEFAULT_SEED}},
        metrics_path=tmp_path / "absent-metrics.jsonl",
        record_path=records,
    )

    assert ledger["record_complete"] is False
    assert "gemma_http_summary_attempt_count_mismatch" in ledger["record_failures"]
    assert "gemma_http_attempt_index_sequence_invalid" in ledger["record_failures"]


def test_full_auto_request_ledger_rejects_nonstop_or_non_json_response(tmp_path: Path) -> None:
    request = {
        "model": matrix.MODEL_NAME,
        "seed": matrix.DEFAULT_SEED,
        "messages": [{"role": "user", "content": "fixture"}],
        "response_format": {"type": "json_object"},
    }
    records = tmp_path / "gemma-http-records.jsonl"
    records.write_text(
        json.dumps(
            {
                "attempt_index": 0,
                "request": request,
                "response": {
                    "choices": [
                        {
                            "index": 0,
                            "message": {"content": "not-json"},
                            "finish_reason": "length",
                        }
                    ]
                },
                "status_code": 200,
                "error": "",
            }
        )
        + "\n"
        + json.dumps({"record_type": "summary", "attempt_count": 1, "write_error": ""})
        + "\n",
        encoding="utf-8",
    )

    ledger = matrix._pipeline_request_ledger(
        preset={"gemma": {}, "benchmark_http": {"gemma_seed": matrix.DEFAULT_SEED}},
        metrics_path=tmp_path / "absent-metrics.jsonl",
        record_path=records,
    )

    assert ledger["record_complete"] is False
    assert "gemma_http_response_contract_invalid_line_1" in ledger["record_failures"]


def test_full_auto_quality_gate_requires_decoded_output_per_page() -> None:
    failures = matrix._full_auto_quality_failures(
        snapshot={
            "pages": [
                {
                    "page_failed": False,
                    "translated_image_exists": True,
                    "translated_image_decoded_pixel_sha256": "a" * 64,
                    "stage_status": {},
                }
            ]
        },
        summary={"page_failed_count": 0, "page_done_count": 1},
        expected_page_count=1,
    )

    assert failures == []


def test_pre_translation_snapshot_hash_excludes_translation_and_render() -> None:
    snapshot = {
        "pages": [
            {
                "source_lang": "ja",
                "target_lang": "ko",
                "ocr_quality": {"block_count": 1},
                "inpaint_decoded_pixel_sha256": "a" * 64,
                "stage_status": {
                    "detect": {"status": "completed", "updated_at": "dynamic"},
                    "ocr": {"status": "completed", "cache_status": "warm"},
                    "inpaint": {"status": "completed"},
                    "translation": {"status": "completed"},
                    "render": {"status": "completed"},
                },
                "blocks": [
                    {
                        "xyxy": [1, 2, 3, 4],
                        "text": "原文",
                        "block_final_mask_pixel_count": 12,
                        "translation": "번역 A",
                        "render_text": "번역 A",
                    }
                ],
            }
        ]
    }
    changed = json.loads(json.dumps(snapshot, ensure_ascii=False))
    changed["pages"][0]["blocks"][0]["translation"] = "번역 B"
    changed["pages"][0]["blocks"][0]["render_text"] = "번역 B"
    changed["pages"][0]["stage_status"]["translation"] = {"status": "failed"}
    changed["pages"][0]["stage_status"]["render"] = {"status": "failed"}

    assert matrix.canonical_pre_translation_snapshot_sha256(snapshot) == (
        matrix.canonical_pre_translation_snapshot_sha256(changed)
    )

    changed["pages"][0]["blocks"][0]["text"] = "別文"
    assert matrix.canonical_pre_translation_snapshot_sha256(snapshot) != (
        matrix.canonical_pre_translation_snapshot_sha256(changed)
    )


def test_pre_translation_snapshot_hash_includes_mask_decision_and_reject_reason() -> None:
    snapshot = {
        "pages": [
            {
                "source_lang": "ja",
                "target_lang": "ko",
                "ocr_quality": {},
                "inpaint_decoded_pixel_sha256": "a" * 64,
                "stage_status": {},
                "blocks": [
                    {
                        "xyxy": [1, 2, 3, 4],
                        "text": "原文",
                        "mask_decision": "accepted",
                        "mask_reject_reason": "",
                    }
                ],
            }
        ]
    }
    changed = json.loads(json.dumps(snapshot, ensure_ascii=False))
    changed["pages"][0]["blocks"][0]["mask_decision"] = "rejected"

    assert matrix.canonical_pre_translation_snapshot_sha256(snapshot) != (
        matrix.canonical_pre_translation_snapshot_sha256(changed)
    )


def test_installing_adapter_does_not_start_a_container() -> None:
    class Window:
        def __init__(self) -> None:
            self.local_translation_runtime_manager = runtime.LocalGemmaRuntimeManager()

    window = Window()
    original = window.local_translation_runtime_manager
    installed = runtime.install_turbo4_lab_runtime_adapter(
        window,
        {
            "protocol_version": matrix.PROTOCOL_VERSION,
            "candidate_key": "turbo4",
            "image_ref": "comic-translate/turbo4-lab:test",
            "model_volume": matrix.MODEL_VOLUME,
            "model_name": matrix.MODEL_NAME,
            "model_sha256": matrix.MODEL_SHA256,
            "port": 18081,
            "container_name": "ct-gemma-turbo4-adapter-test",
            "cache_type_v": "turbo4",
            "fork_commit": matrix.FORK_COMMIT,
        },
    )

    assert isinstance(window.local_translation_runtime_manager, runtime.Turbo4LabRuntimeManager)
    assert installed.adapter.evidence()["events"] == []
    installed.closed = True  # No Docker cleanup is needed because nothing started.
    window.local_translation_runtime_manager = original


def test_parent_abort_recovery_uses_the_captured_gpu_baseline(monkeypatch) -> None:
    config = runtime.Turbo4LabRuntimeConfig(
        protocol_version=matrix.PROTOCOL_VERSION,
        candidate_key="turbo4",
        image_ref="comic-translate/turbo4-lab:test",
        model_volume=matrix.MODEL_VOLUME,
        model_name=matrix.MODEL_NAME,
        model_sha256=matrix.MODEL_SHA256,
        port=18081,
        container_name="ct-gemma-turbo4-parent-abort-test",
        cache_type_v="turbo4",
        fork_commit=matrix.FORK_COMMIT,
    )
    manager = runtime.Turbo4LabRuntimeManager(inner=None, config=config)
    monkeypatch.setattr(runtime, "_inspect_container", lambda _name: {"Id": "container"})
    monkeypatch.setattr(
        manager,
        "shutdown",
        lambda: {"status": "released", "observed": True},
    )

    result = manager.recover_after_parent_abort(baseline_gpu_mib=565)

    assert result["observed"] is True
    assert manager._pre_start_gpu_mib == 565


def test_parent_abort_recovery_rejects_when_no_container_and_gpu_never_returns(
    monkeypatch,
) -> None:
    config = runtime.Turbo4LabRuntimeConfig(
        protocol_version=matrix.PROTOCOL_VERSION,
        candidate_key="turbo4",
        image_ref="comic-translate/turbo4-lab:test",
        model_volume=matrix.MODEL_VOLUME,
        model_name=matrix.MODEL_NAME,
        model_sha256=matrix.MODEL_SHA256,
        port=18081,
        container_name="ct-gemma-turbo4-parent-abort-release-timeout-test",
        cache_type_v="turbo4",
        fork_commit=matrix.FORK_COMMIT,
    )
    manager = runtime.Turbo4LabRuntimeManager(inner=None, config=config)
    monkeypatch.setattr(runtime, "_inspect_container", lambda _name: {})
    monkeypatch.setattr(
        manager,
        "_wait_for_gpu_release",
        lambda **_kwargs: {"status": "timeout", "observed": False},
    )

    with pytest.raises(runtime.Turbo4LabRuntimeError, match="GPU release was not confirmed"):
        manager.recover_after_parent_abort(baseline_gpu_mib=565)

    assert manager._pre_start_gpu_mib == 565
    assert manager.evidence()["release"] == {"status": "timeout", "observed": False}

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "benchmark_serving_scheduler_matrix.py"
SPEC = importlib.util.spec_from_file_location("benchmark_serving_scheduler_matrix", MODULE_PATH)
assert SPEC and SPEC.loader
matrix = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = matrix
SPEC.loader.exec_module(matrix)


def test_catalog_keeps_exhausted_axes_out_of_default_plan() -> None:
    phases = matrix.staged_candidate_keys()
    flattened = {key for keys in phases.values() for key in keys}
    assert "idle1" in flattened
    assert "np2-w4" in flattened
    assert "np4-w6-http4" in flattened
    assert "np4-w6-poll0" in flattened
    assert "batch1024" not in flattened
    assert "ubatch256" not in flattened
    assert not any("token" in key for key in flattened)
    assert not any("global" in key for key in flattened)
    assert "np4-w6-http4" in matrix.candidate_catalog()


def test_paddle_command_has_candidate_runtime_contract() -> None:
    candidate = matrix.candidate_catalog()["np4-w8"]
    command = matrix.build_paddle_server_command(candidate)
    joined = " ".join(command)
    assert "-np 4" in joined
    assert "--metrics" in command
    assert "--slots" in command
    assert "--sleep-idle-seconds 5" in joined
    assert "-c 16384" in joined
    assert "/app/llama-server" not in command
    assert f"/models/{matrix.PADDLE_MODEL_FILE}" in command
    assert f"/models/{matrix.PADDLE_MMPROJ_FILE}" in command


def test_http_and_poll_candidates_change_only_their_axis() -> None:
    catalog = matrix.candidate_catalog()
    baseline = catalog["baseline"]
    http = catalog["http4"]
    poll = catalog["poll0"]
    assert http.threads_http == 4
    assert http.n_parallel == baseline.n_parallel
    assert http.client_workers == baseline.client_workers
    assert poll.poll == 0
    assert poll.poll_batch == 0
    assert poll.threads_http == baseline.threads_http


def test_router_preset_uses_one_model_contract_and_exact_aliases() -> None:
    preset = matrix.build_router_preset()
    command = matrix.build_router_command()
    assert f"[{matrix.PADDLE_MODEL_ALIAS}]" in preset
    assert f"[{matrix.GEMMA_MODEL_ALIAS}]" in preset
    assert "load-on-startup = false" in preset
    assert command[command.index("--models-max") + 1] == "1"
    assert "--no-models-autoload" in command
    assert "/app/llama-server" not in command


def test_lab_container_cleanup_refuses_product_container(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(matrix, "_run", lambda *args, **kwargs: calls.append(args))
    with pytest.raises(matrix.BenchmarkContractError):
        matrix._remove_exact_lab_container("paddleocr-llamacpp")
    assert calls == []


def test_runtime_option_parser_rejects_missing_router_features() -> None:
    full = "\n".join(sorted(matrix.REQUIRED_LLAMA_OPTIONS))
    assert matrix.missing_required_options(full) == []
    missing = matrix.missing_required_options(full.replace("--models-max", ""))
    assert missing == ["--models-max"]


def test_volume_probe_validates_ready_manifest_and_fast_file_sizes() -> None:
    manifest = {
        "schema_version": 1,
        "runtime": matrix.PADDLE_RUNTIME_NAME,
        "preparation_version": matrix.PADDLE_PREPARATION_VERSION,
        "volume_name": matrix.PADDLE_MODEL_VOLUME,
        "ready": True,
        "source_image_ref": matrix.PINNED_LLAMA_IMAGE,
        "source_image_id": matrix.PINNED_LLAMA_IMAGE.rsplit("@", 1)[-1],
        "files": [
            {
                "name": matrix.PADDLE_MODEL_FILE,
                "bytes": matrix.PADDLE_MODEL_BYTES,
                "sha256": matrix.PADDLE_MODEL_SHA256,
                "role": "vlm",
            },
            {
                "name": matrix.PADDLE_MMPROJ_FILE,
                "bytes": matrix.PADDLE_MMPROJ_BYTES,
                "sha256": matrix.PADDLE_MMPROJ_SHA256,
                "role": "vision-projector",
            },
        ],
        "smoke_test": {"passed": True},
    }
    manifest_bytes = json.dumps(manifest).encode("utf-8")
    valid = matrix.validate_paddle_volume_probe(
        labels={
            "comic-translate.runtime": matrix.PADDLE_RUNTIME_NAME,
            "comic-translate.preparation-version": str(
                matrix.PADDLE_PREPARATION_VERSION
            ),
        },
        manifest_bytes=manifest_bytes,
        manifest_sha256=matrix.hashlib.sha256(manifest_bytes).hexdigest(),
        observed_file_bytes={
            matrix.PADDLE_MODEL_FILE: matrix.PADDLE_MODEL_BYTES,
            matrix.PADDLE_MMPROJ_FILE: matrix.PADDLE_MMPROJ_BYTES,
        },
    )
    wrong_size = matrix.validate_paddle_volume_probe(
        labels=valid["volume_labels"],
        manifest_bytes=manifest_bytes,
        manifest_sha256=valid["ready_manifest_sha256"],
        observed_file_bytes={
            matrix.PADDLE_MODEL_FILE: matrix.PADDLE_MODEL_BYTES - 1,
            matrix.PADDLE_MMPROJ_FILE: matrix.PADDLE_MMPROJ_BYTES,
        },
    )
    assert valid["passed"] is True
    assert wrong_size["passed"] is False
    assert any(
        failure.startswith("volume_file_contract_mismatch:")
        for failure in wrong_size["failures"]
    )


def test_residency_preflight_uses_relaxed_95_percent_gate() -> None:
    safe = matrix.residency_preflight(
        physical_mib=12_282,
        paddle_peak_mib=300,
        gemma_peak_mib=11_300,
    )
    unsafe = matrix.residency_preflight(
        physical_mib=12_282,
        paddle_peak_mib=2_407,
        gemma_peak_mib=11_634,
    )
    assert safe["may_run_dual_model"] is True
    assert unsafe["may_run_dual_model"] is False
    assert unsafe["threshold"] == pytest.approx(0.95)
    assert unsafe["combined_ratio"] > 1.0


def test_resource_preflight_fails_closed_without_swap_metrics(monkeypatch) -> None:
    monkeypatch.setattr(
        matrix,
        "_gpu_snapshot",
        lambda: {"available": True, "used_mib": 100, "total_mib": 12_282},
    )
    monkeypatch.setattr(
        matrix,
        "_windows_available_bytes",
        lambda: matrix.WINDOWS_AVAILABLE_LIMIT_BYTES,
    )
    monkeypatch.setattr(matrix, "_wsl_swap_used_bytes", lambda: None)
    result = matrix.resource_preflight()
    assert result["passed"] is False
    assert result["failures"] == ["wsl_swap_metrics_unavailable"]


def test_swap_gate_prefers_cgroup_peak_and_falls_back_to_global() -> None:
    clean = matrix.swap_gate_report(
        global_delta_bytes=0,
        cgroup_peak_bytes=[0],
    )
    cgroup_swap = matrix.swap_gate_report(
        global_delta_bytes=0,
        cgroup_peak_bytes=[4096],
    )
    fallback = matrix.swap_gate_report(
        global_delta_bytes=0,
        cgroup_peak_bytes=[None],
    )
    assert clean["passed"] is True
    assert clean["source"] == "cgroup-v2+global-wsl"
    assert cgroup_swap["passed"] is False
    assert fallback["passed"] is True
    assert fallback["source"] == "global-wsl-fallback"


def test_container_swap_peak_reads_cgroup_v2(monkeypatch) -> None:
    monkeypatch.setattr(
        matrix,
        "_run",
        lambda *_args, **_kwargs: __import__("subprocess").CompletedProcess(
            [], 0, stdout="8192\n", stderr=""
        ),
    )
    assert matrix._container_swap_peak_bytes("ct-serving-matrix-paddle-test") == 8192


def test_windows_resource_thresholds_keep_promotion_and_emergency_separate() -> None:
    assert matrix.WINDOWS_AVAILABLE_LIMIT_BYTES == 6 * 1024**3
    assert matrix.WINDOWS_EMERGENCY_LIMIT_BYTES == 1 * 1024**3
    assert (
        matrix.WINDOWS_EMERGENCY_LIMIT_BYTES
        < matrix.WINDOWS_AVAILABLE_LIMIT_BYTES
    )


def test_slot_context_report_requires_4096_per_parallel_slot() -> None:
    valid = matrix.slot_context_report(
        props={
            "total_slots": 2,
            "default_generation_settings": {"n_ctx": 8192},
        },
        slots={"value": [{"n_ctx": 4096}, {"n_ctx": 4096}]},
        expected_parallel=2,
    )
    reduced = matrix.slot_context_report(
        props={
            "total_slots": 2,
            "default_generation_settings": {"n_ctx": 4096},
        },
        slots={"value": [{"n_ctx": 2048}, {"n_ctx": 2048}]},
        expected_parallel=2,
    )
    assert valid["passed"] is True
    assert reduced["passed"] is False
    assert "slot_context_below_4096" in reduced["failures"]


def test_canonical_snapshot_ignores_paths_and_timestamps_but_not_ocr() -> None:
    base = {
        "generated_at": 1,
        "pages": [
            {
                "image_path": "private-a",
                "image_name": "a.jpg",
                "source_lang": "ja",
                "target_lang": "ko",
                "page_failed": False,
                "ocr_quality": {"block_count": 1, "non_empty": 1},
                "blocks": [
                    {
                        "xyxy": [1, 2, 3, 4],
                        "text": "原文",
                        "normalized_text": "原文",
                        "ocr_status": "ok",
                    }
                ],
            }
        ],
    }
    changed_path = json.loads(json.dumps(base, ensure_ascii=False))
    changed_path["generated_at"] = 99
    changed_path["pages"][0]["image_path"] = "private-b"
    assert matrix.canonical_snapshot_sha256(base) == matrix.canonical_snapshot_sha256(
        changed_path
    )
    changed_ocr = json.loads(json.dumps(base, ensure_ascii=False))
    changed_ocr["pages"][0]["blocks"][0]["text"] = "別文"
    assert matrix.canonical_snapshot_sha256(base) != matrix.canonical_snapshot_sha256(
        changed_ocr
    )


def test_freeze_ocr_workload_locks_detector_geometry_and_source_hash(
    tmp_path: Path,
) -> None:
    source = tmp_path / "page.png"
    source.write_bytes(b"frozen-source")
    snapshot = tmp_path / "page_snapshots.json"
    snapshot.write_text(
        json.dumps(
            {
                "pages": [
                    {
                        "image_path": str(source),
                        "image_name": "page.png",
                        "source_lang": "Japanese",
                        "target_lang": "Korean",
                        "blocks": [
                            {
                                "xyxy": [1.0, 2.0, 30.0, 40.0],
                                "bubble_xyxy": [0.0, 1.0, 31.0, 41.0],
                                "angle": 90,
                                "text_class": "text_bubble",
                                "text": "candidate-specific OCR must not freeze",
                            }
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    workload = matrix.freeze_ocr_workload(
        snapshot,
        output_path=tmp_path / "frozen-workload.json",
    )
    assert workload["page_count"] == 1
    assert workload["block_count"] == 1
    assert workload["pages"][0]["source_sha256"] == matrix._sha256_file(source)
    assert workload["pages"][0]["blocks"] == [
        {
            "block_id": "page-0000-block-0000",
            "xyxy": [1, 2, 30, 40],
            "bubble_xyxy": [0, 1, 31, 41],
            "angle": 90,
            "text_class": "text_bubble",
        }
    ]


def test_full_snapshot_includes_translation_and_final_render_hash() -> None:
    base = {
        "pages": [
            {
                "source_lang": "ja",
                "target_lang": "ko",
                "translated_image_exists": True,
                "translated_image_sha256": "a" * 64,
                "blocks": [{"text": "原文", "translation": "번역"}],
            }
        ]
    }
    changed = json.loads(json.dumps(base, ensure_ascii=False))
    changed["pages"][0]["blocks"][0]["translation"] = "다른 번역"
    assert matrix.canonical_full_snapshot_sha256(base) != (
        matrix.canonical_full_snapshot_sha256(changed)
    )


def test_pair_statistics_require_positive_proven_gain() -> None:
    clear = matrix.summarize_pairs([10.0, 10.0, 10.0], [9.0, 9.0, 9.0])
    mixed = matrix.summarize_pairs([10.0, 10.0, 10.0], [9.0, 11.0, 9.5])
    assert clear["one_sided_95_bootstrap_lower_percent"] > 0
    assert matrix.should_continue_adaptive(clear, rounds=3) is False
    assert mixed["one_sided_95_bootstrap_lower_percent"] <= 0
    assert matrix.should_continue_adaptive(mixed, rounds=3) is True


def test_snapshot_mismatch_collects_initial_ab_ba_window_then_stops() -> None:
    clear = matrix.summarize_pairs([10.0, 10.0], [9.0, 9.0])
    mismatch = [{"round": 1, "profile": "np2-w2"}]
    assert matrix.should_stop_pair_matrix(
        summary=clear,
        axis_summary=clear,
        rounds=1,
        initial_rounds=2,
        snapshot_mismatches=mismatch,
    ) is False
    assert matrix.should_stop_pair_matrix(
        summary=clear,
        axis_summary=clear,
        rounds=2,
        initial_rounds=2,
        snapshot_mismatches=mismatch,
    ) is True


def test_pair_matrix_caps_rounds_at_protocol_maximum(tmp_path: Path, monkeypatch) -> None:
    workload = {"kind": "frozen-paddle-crop-workload", "pages": []}
    calls = []

    def fake_execute(profile, *, frozen_workload=None, **_kwargs):
        calls.append((profile.key, frozen_workload))
        return {
            "execution_mode": "isolated-frozen-ocr-replay",
            "start": {"start_to_health_sec": 1.0},
            "pipeline": {
                "wall_sec": 2.0,
                "snapshot_sha256": "same",
                "ocr_request_metrics": {"request_wall_ms": 1_000.0},
            },
            "swap_gate_pass": True,
            "windows_ram_gate_pass": True,
        }

    monkeypatch.setattr(matrix, "execute_candidate_once", fake_execute)
    result = matrix.execute_pair_matrix(
        matrix.candidate_catalog()["np2-w2"],
        output_dir=tmp_path,
        sample_dir=tmp_path,
        sample_count=1,
        initial_rounds=100,
        max_rounds=100,
        python_executable="python.exe",
        shared_frozen_workload=workload,
    )
    assert result["round_count"] == matrix.DEFAULT_MAX_ROUNDS
    assert len(calls) == matrix.DEFAULT_MAX_ROUNDS * 2


def test_pair_matrix_reuses_supplied_frozen_workload(
    tmp_path: Path, monkeypatch
) -> None:
    workload = {"kind": "frozen-paddle-crop-workload", "pages": []}
    calls = []

    def fake_execute(profile, *, frozen_workload=None, **kwargs):
        calls.append((profile.key, frozen_workload, kwargs["run_dir"]))
        return {
            "execution_mode": "isolated-frozen-ocr-replay",
            "start": {"start_to_health_sec": 1.0},
            "pipeline": {
                "wall_sec": 2.0,
                "snapshot_sha256": "same",
                "ocr_request_metrics": {"request_wall_ms": 1_000.0},
            },
            "swap_gate_pass": True,
            "windows_ram_gate_pass": True,
        }

    monkeypatch.setattr(matrix, "execute_candidate_once", fake_execute)
    reference = matrix.candidate_catalog()["np4-w6"]
    result = matrix.execute_pair_matrix(
        matrix.candidate_catalog()["np4-w6-http2"],
        reference_candidate=reference,
        output_dir=tmp_path,
        sample_dir=tmp_path,
        sample_count=1,
        initial_rounds=2,
        max_rounds=2,
        python_executable="python.exe",
        shared_frozen_workload=workload,
    )
    assert result["round_count"] == 2
    assert result["reference_candidate"]["key"] == "np4-w6"
    assert len(calls) == 4
    assert all(call[1] is workload for call in calls)
    assert [call[0] for call in calls] == [
        "np4-w6",
        "np4-w6-http2",
        "np4-w6-http2",
        "np4-w6",
    ]
    assert not (tmp_path / "np4-w6-http2" / "frozen-capture").exists()


def test_handoff_axis_reads_paddle_release_metric_separately_from_e2e() -> None:
    result = {
        "pipeline": {
            "wall_sec": 100.0,
            "summary": {
                "performance_stats": {
                    "runtime": {
                        "paddleocr_vl": {"release_wall_ms": 1_250.0}
                    }
                }
            },
        }
    }
    assert matrix._pipeline_wall_value(result) == pytest.approx(100.0)
    assert matrix._axis_timing_value(result, axis="handoff") == pytest.approx(
        1.25
    )


def test_isolated_pipeline_wall_includes_server_startup() -> None:
    result = {
        "execution_mode": "isolated-ocr-ceiling",
        "start": {"start_to_health_sec": 4.0},
        "pipeline": {"wall_sec": 10.0},
    }
    assert matrix._pipeline_wall_value(result) == pytest.approx(14.0)
    frozen = {
        "execution_mode": "isolated-frozen-ocr-replay",
        "start": {"start_to_health_sec": 4.0},
        "pipeline": {"wall_sec": 10.0},
    }
    assert matrix._pipeline_wall_value(frozen) == pytest.approx(14.0)


def test_adaptive_health_polling_uses_short_then_backoff_intervals(monkeypatch) -> None:
    clock = {"value": 0.0}
    attempts = {"value": 0}

    def fake_http(_url: str):
        attempts["value"] += 1
        if attempts["value"] < 4:
            raise OSError("not ready")
        return {"status": "ok"}

    def monotonic() -> float:
        return clock["value"]

    def sleeper(value: float) -> None:
        clock["value"] += value

    monkeypatch.setattr(matrix, "_http_json", fake_http)
    result = matrix.adaptive_wait_for_health(
        "http://example.invalid/health",
        timeout_sec=5,
        monotonic=monotonic,
        sleeper=sleeper,
    )
    assert result["attempts"] == 4
    assert result["intervals"] == [0.1, 0.1, 0.1]


def test_candidate_preset_disables_all_product_result_caches() -> None:
    preset = matrix._build_candidate_preset(
        matrix.candidate_catalog()["idle1"], endpoint_port=18001
    )
    assert preset["ocr_client"]["server_url"].endswith(
        ":18001/v1/chat/completions"
    )
    assert preset["ocr_client"]["parallel_workers"] == 8
    assert preset["benchmark_http"]["gemma_seed"] == 20260801
    assert preset["ocr_runtime"] == {
        "kind": "paddleocr_vl",
        "llama_cpp_image": matrix.PINNED_LLAMA_IMAGE,
        "pull_policy": "never",
        "model_path": f"/models/{matrix.PADDLE_MODEL_FILE}",
        "mmproj_path": f"/models/{matrix.PADDLE_MMPROJ_FILE}",
        "model_alias": matrix.PADDLE_MODEL_ALIAS,
        "context_size": 4096,
        "n_parallel": 1,
        "threads": 10,
        "batch_size": 2048,
        "ubatch_size": 512,
        "n_gpu_layers": "all",
        "sleep_idle_seconds": 1,
        "model_volume": matrix.PADDLE_MODEL_VOLUME,
    }
    assert "front_device" not in preset["ocr_runtime"]
    assert "image" not in preset["ocr_runtime"]
    assert preset["benchmark_cache_policy"] == {
        "paddleocr_persistent": False,
        "translation_persistent": False,
        "exact_tm": False,
        "project_checkpoint": False,
        "translation_result_cache_limit": 50000,
        "translation_candidate_limit": 5000,
    }


def test_full_auto_pipeline_command_uses_stage_aware_product_runtime() -> None:
    command = matrix.build_product_pipeline_command(
        preset_path=Path("preset.json"),
        run_dir=Path("run"),
        sample_dir=Path("sample"),
        sample_count=1,
        python_executable="python.exe",
        stage_ceiling="render",
        runtime_mode="attach-running",
        runtime_services="full",
        product_managed_runtime=True,
    )
    assert command[command.index("--stage-ceiling") + 1] == "render"
    assert command[command.index("--runtime-mode") + 1] == "attach-running"
    assert command[command.index("--runtime-services") + 1] == "full"
    assert "--product-managed-runtime" in command
    assert "--clear-app-caches" not in command

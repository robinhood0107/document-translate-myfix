from __future__ import annotations

import importlib.util
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


runtime = _load("gemma_offload_runtime_test", SCRIPTS / "turbo4_lab_runtime.py")
lab = _load("gemma_cpu_moe_offload_test", SCRIPTS / "benchmark_gemma_cpu_moe_offload.py")


def test_protocol_pins_shipping_and_only_residency_deltas() -> None:
    protocol = lab.load_protocol()

    assert protocol["shipping"]["image"] == lab.SHIPPING_IMAGE
    assert protocol["model"]["sha256"] == lab.MODEL_SHA256
    assert set(protocol["candidate_delta"]) == {"--no-kv-offload", "--n-cpu-moe N"}
    assert "turbo4" in protocol["forbidden"]
    assert "ssd_offload" in protocol["forbidden"]


def test_offload_command_preserves_shipping_settings_except_two_residency_controls() -> None:
    baseline = runtime.ShippingOffloadLabRuntimeConfig(
        protocol_version=lab.PROTOCOL_VERSION,
        candidate_key="shipping-f16",
        image_ref=lab.SHIPPING_IMAGE,
        model_volume=lab.MODEL_VOLUME,
        model_name=lab.MODEL_NAME,
        model_sha256=lab.MODEL_SHA256,
        port=18082,
        container_name="ct-gemma-offload-baseline-test",
        kv_offload=True,
        n_cpu_moe=0,
    )
    candidate = runtime.ShippingOffloadLabRuntimeConfig(
        protocol_version=lab.PROTOCOL_VERSION,
        candidate_key="no-kv-moe2",
        image_ref=lab.SHIPPING_IMAGE,
        model_volume=lab.MODEL_VOLUME,
        model_name=lab.MODEL_NAME,
        model_sha256=lab.MODEL_SHA256,
        port=18082,
        container_name="ct-gemma-offload-candidate-test",
        kv_offload=False,
        n_cpu_moe=2,
    )

    assert "--kv-offload" in baseline.command
    assert "--no-kv-offload" not in baseline.command
    assert "--no-kv-offload" in candidate.command
    assert candidate.command[-2:] == ["--n-cpu-moe", "2"]
    assert baseline.command[: baseline.command.index("--kv-offload")] == candidate.command[: candidate.command.index("--no-kv-offload")]
    assert baseline.command[baseline.command.index("--kv-offload") + 1 :] == candidate.command[candidate.command.index("--no-kv-offload") + 1 : -2]


def test_offload_config_rejects_nonshipping_or_unfixed_values() -> None:
    with pytest.raises(runtime.Turbo4LabRuntimeError):
        runtime.ShippingOffloadLabRuntimeConfig(
            protocol_version=lab.PROTOCOL_VERSION,
            candidate_key="bad",
            image_ref=lab.SHIPPING_IMAGE,
            model_volume=lab.MODEL_VOLUME,
            model_name=lab.MODEL_NAME,
            model_sha256=lab.MODEL_SHA256,
            port=18082,
            container_name="gemma-local-server",
            kv_offload=False,
            n_cpu_moe=1,
        ).validate()

    with pytest.raises(runtime.Turbo4LabRuntimeError):
        runtime.ShippingOffloadLabRuntimeConfig(
            protocol_version=lab.PROTOCOL_VERSION,
            candidate_key="bad-layer",
            image_ref=lab.SHIPPING_IMAGE,
            model_volume=lab.MODEL_VOLUME,
            model_name=lab.MODEL_NAME,
            model_sha256=lab.MODEL_SHA256,
            port=18082,
            container_name="ct-gemma-offload-bad-layer",
            kv_offload=False,
            n_cpu_moe=1,
            n_gpu_layers=22,
        ).validate()


def test_screen_ladder_is_ordered_and_deduplicated() -> None:
    candidates = lab._screen_candidates((2, 1, 2, 4))

    assert [(item.kv_offload, item.n_cpu_moe) for item in candidates] == [
        (False, 0),
        (False, 1),
        (False, 2),
        (False, 4),
    ]


def test_residency_estimate_requires_candidate_to_fit_with_paddle() -> None:
    result = {
        "preflight": {"gpu": {"used_mib": 800, "total_mib": 12282}},
        "resource_gates": {"gpu_peak_mib": 10675},
    }

    estimate = lab._residency_estimate(result)

    assert estimate["gemma_increment_mib"] == 9875
    assert estimate["physical_fit"]["combined_peak_mib"] == 12282
    assert estimate["physical_fit"]["may_run_dual_model"] is True
    assert estimate["headroom_95"]["may_run_dual_model"] is False


def test_selected_physical_fit_preflight_rejects_an_unfit_manual_level(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        lab,
        "execute_replay_once",
        lambda *args, **kwargs: {
            "status": "passed",
            "residency": {"physical_fit": {"may_run_dual_model": False}},
        },
    )

    result = lab.execute_selected_physical_fit_preflight(
        payloads=[{"model": lab.MODEL_NAME}],
        n_cpu_moe=1,
        output_dir=tmp_path,
        port=18082,
        image_id="sha256:test",
    )

    assert result["decision"] == "REJECT"
    assert (tmp_path / "physical-fit-summary.json").is_file()


def test_offload_preflight_treats_an_active_turbo4_lab_as_a_conflict(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        lab.replay_tools,
        "_running_container_names",
        lambda: ["ct-gemma-turbo4-running"],
    )

    assert lab._running_conflicts() == ["ct-gemma-turbo4-running"]


def test_co_resident_probe_still_shuts_down_gemma_when_paddle_cleanup_fails(
    monkeypatch,
    tmp_path: Path,
) -> None:
    class FakeSampler:
        instances: list["FakeSampler"] = []

        def __init__(self) -> None:
            self.samples: list[dict[str, object]] = []
            self.started = False
            self.stopped = False
            FakeSampler.instances.append(self)

        @property
        def emergency_reason(self) -> None:
            return None

        def start(self) -> None:
            self.started = True

        def stop(self) -> None:
            self.stopped = True

    class FakeManager:
        instances: list["FakeManager"] = []

        def __init__(self, *, inner, config) -> None:
            del inner
            self.config = config
            self.shutdown_calls = 0
            FakeManager.instances.append(self)

        def ensure_server(self, *_args, **_kwargs) -> None:
            return None

        def shutdown(self) -> dict[str, object]:
            self.shutdown_calls += 1
            return {"status": "released", "observed": True, "cgroup_swap_peak_bytes": 0}

        def evidence(self) -> dict[str, object]:
            return {"release": {"status": "released", "observed": True}}

    removed: list[str] = []

    def fail_paddle_cleanup(name: str) -> None:
        removed.append(name)
        raise OSError("paddle cleanup failure")

    monkeypatch.setattr(
        lab,
        "require_preflight",
        lambda **_kwargs: {"gpu": {"total_mib": 12_282}},
    )
    monkeypatch.setattr(lab, "Turbo4LabRuntimeManager", FakeManager)
    monkeypatch.setattr(lab.replay_tools, "ResourceSampler", FakeSampler)
    monkeypatch.setattr(
        lab.serving_tools,
        "_start_paddle_container",
        lambda *_args, **_kwargs: {"started": True},
    )
    monkeypatch.setattr(
        lab.replay_tools,
        "_http_json",
        lambda *_args, **_kwargs: {"choices": [{"message": {"content": "{}"}}]},
    )
    monkeypatch.setattr(lab, "_read_logs", lambda *_args, **_kwargs: "")
    monkeypatch.setattr(lab, "_remove_paddle_lab", fail_paddle_cleanup)
    monkeypatch.setattr(
        lab.replay_tools,
        "resource_gate_report",
        lambda **_kwargs: {"passed": True, "failures": [], "gpu_peak_mib": 12_000},
    )
    monkeypatch.setattr(lab, "_running_conflicts", lambda: [])

    output_dir = tmp_path / "co-resident"
    output_dir.mkdir()
    result = lab.execute_co_resident_probe(
        payload={"model": lab.MODEL_NAME},
        n_cpu_moe=11,
        output_dir=output_dir,
        port=18082,
        paddle_port=18002,
        image_id="sha256:test",
    )

    assert result["status"] == "REJECT"
    assert "OSError: paddle cleanup failure" in result["error"]
    assert removed and removed[0].startswith(lab.serving_tools.LAB_CONTAINER_PREFIX)
    assert FakeManager.instances[0].shutdown_calls == 1
    assert FakeSampler.instances[0].started is True
    assert FakeSampler.instances[0].stopped is True


def test_redacted_output_keeps_a_manual_selected_level(capsys) -> None:
    lab._print_redacted(
        {
            "decision": "PASS",
            "selected_n_cpu_moe": 11,
            "co_resident": {"status": "PASS"},
        }
    )

    assert '"selected_n_cpu_moe": 11' in capsys.readouterr().out


def test_main_completes_the_managed_archive_for_an_expected_reject(
    monkeypatch,
    tmp_path: Path,
) -> None:
    class ManagedRun:
        def __init__(self) -> None:
            self.completed = []
            self.failed = []
            self._closed = False

        def complete(self, *, metadata) -> None:
            if not self._closed:
                self.completed.append(dict(metadata))
                self._closed = True

        def fail(self, error, *, metadata) -> None:
            self.failed.append((error, dict(metadata)))
            self._closed = True

    managed = ManagedRun()
    monkeypatch.setattr(lab.replay_tools, "load_translation_replay", lambda _path: [{"model": lab.MODEL_NAME}])
    monkeypatch.setattr(lab, "select_managed_output_directory", lambda **_kwargs: (tmp_path, managed))
    monkeypatch.setattr(lab, "verify_runtime", lambda **_kwargs: {"shipping_image": {"id": "sha256:test"}})
    monkeypatch.setattr(
        lab,
        "execute_screen",
        lambda **_kwargs: {"decision": "REJECT", "selected_n_cpu_moe": None, "runs": []},
    )

    status = lab.main(["--mode", "screen", "--translation-replay", str(tmp_path / "replay.json")])

    assert status == 2
    assert managed.failed == []
    assert managed.completed == [
        {
            "protocol_version": lab.PROTOCOL_VERSION,
            "mode": "screen",
            "decision": "REJECT",
        }
    ]

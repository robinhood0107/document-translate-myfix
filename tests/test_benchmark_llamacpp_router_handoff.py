from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "benchmark_llamacpp_router_handoff.py"
SPEC = importlib.util.spec_from_file_location("benchmark_llamacpp_router_handoff", MODULE_PATH)
assert SPEC and SPEC.loader
router_lab = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = router_lab
SPEC.loader.exec_module(router_lab)


def _router_runtime_module(name: str):
    runtime_path = ROOT / "scripts" / "router_handoff_lab_runtime.py"
    runtime_spec = importlib.util.spec_from_file_location(name, runtime_path)
    assert runtime_spec and runtime_spec.loader
    runtime = importlib.util.module_from_spec(runtime_spec)
    sys.modules[runtime_spec.name] = runtime
    runtime_spec.loader.exec_module(runtime)
    return runtime


def _private_stage_contract() -> dict[str, str]:
    return {
        "detection_sha256": "d" * 64,
        "ocr_raw_results_sha256": "r" * 64,
        "ocr_page_profile_sha256": "p" * 64,
        "inpaint_decoded_pixel_sha256": "i" * 64,
        "inpaint_diagnostics_sha256": "g" * 64,
    }


def test_abba_order_is_fixed_once() -> None:
    assert router_lab.ARMS == ("baseline", "router", "router", "baseline")
    assert len(router_lab.ARMS) == 4


def test_speed_rule_requires_both_directions_and_ignores_resource_observations() -> None:
    assert router_lab._both_directions_faster(
        baseline_ab=10.0,
        router_ab=9.9,
        router_ba=9.8,
        baseline_ba=10.1,
    )
    assert not router_lab._both_directions_faster(
        baseline_ab=10.0,
        router_ab=9.9,
        router_ba=10.2,
        baseline_ba=10.1,
    )


def test_router_compose_assets_keep_single_loaded_model_contract() -> None:
    compose_root = ROOT / "benchmarks" / "llamacpp_router_handoff" / "compose"
    for path in sorted(compose_root.glob("*.router.yaml")):
        payload = path.read_text(encoding="utf-8")
        assert "--models-max" in payload
        assert "network_mode: bridge" in payload
        assert '"1"' in payload
        assert "--no-models-autoload" in payload
        assert "/models/ocr" in payload
        assert "/models/gemma" in payload
        assert "read_only: true" in payload
        assert "com.comictranslate.benchmark-owner" in payload


def test_protocol_retires_turboquant_and_keeps_no_active_dual_residency() -> None:
    protocol = json.loads(
        (ROOT / "benchmarks" / "llamacpp_router_handoff" / "protocol-v1.json").read_text(encoding="utf-8")
    )
    assert protocol["decision"]["turboquant"] == "retired"
    assert protocol["decision"]["active_dual_residency"] is False
    assert protocol["decision"]["max_abba_cycles_per_pair"] == 1


def test_paddle_pair_presets_preserve_promoted_request_defaults() -> None:
    crop = router_lab._pair_preset(
        SimpleNamespace(key="paddle-crop", engine_key="PaddleOCR VL"),
        arm="router",
        arm_dir=ROOT,
    )
    spotting = router_lab._pair_preset(
        SimpleNamespace(key="paddle-spotting", engine_key="PaddleOCR VL Spotting"),
        arm="router",
        arm_dir=ROOT,
    )

    assert crop["ocr_client"]["parallel_workers"] == 8
    assert spotting["paddle_spotting_ocr_client"] == {
        "server_url": "http://127.0.0.1:18002/v1/chat/completions",
        "max_completion_tokens": 3000,
        "request_timeout_sec": 360,
    }


def test_mangalmm_pair_preset_preserves_full_page_request_defaults() -> None:
    manga = router_lab._pair_preset(
        SimpleNamespace(key="mangalmm", engine_key="MangaLMM"),
        arm="router",
        arm_dir=ROOT,
    )

    assert manga["mangalmm_ocr_client"] == {
        "server_url": "http://127.0.0.1:28081/v1",
        "max_completion_tokens": 4096,
        "parallel_workers": 1,
        "request_timeout_sec": 60,
        "raw_response_logging": False,
        "safe_resize": True,
        "max_pixels": 2116800,
        "max_long_side": 1728,
    }


def test_router_models_mapping_uses_router_value_state() -> None:
    runtime_path = ROOT / "scripts" / "router_handoff_lab_runtime.py"
    runtime_spec = importlib.util.spec_from_file_location(
        "router_handoff_lab_runtime_for_test", runtime_path
    )
    assert runtime_spec and runtime_spec.loader
    runtime = importlib.util.module_from_spec(runtime_spec)
    sys.modules[runtime_spec.name] = runtime
    runtime_spec.loader.exec_module(runtime)

    states = runtime._models_by_alias(
        {
            "models": {
                "PaddleOCR-VL-1.6-0.9B": {"value": "unloaded"},
                "gemma-4-26B-IQ4_NL.gguf": {"state": {"value": "loaded"}},
            }
        }
    )

    assert states == {
        "PaddleOCR-VL-1.6-0.9B": "unloaded",
        "gemma-4-26B-IQ4_NL.gguf": "loaded",
    }


def test_router_proxy_preserves_real_paddle_crop_and_spotting_payloads() -> None:
    """The router adds no fields and mutates no request made by Paddle itself."""

    runtime = _router_runtime_module("router_handoff_lab_runtime_proxy_test")
    from modules.ocr.paddle_crop.transport import build_direct_ocr_payload
    from modules.ocr.paddle_spotting.engine import PaddleOCRVLSpottingEngine

    crop_payload, _size = build_direct_ocr_payload(b"official-crop-png", max_tokens=1024)

    spotting = object.__new__(PaddleOCRVLSpottingEngine)
    spotting.max_completion_tokens = 3000
    spotting._raise_if_cancelled = lambda: None
    spotting._extract_content = lambda _response: ("", "stop")
    spotting._has_repetition = lambda _content: False
    built_spotting_payloads: list[dict[str, object]] = []

    def capture_spotting(payload):
        built_spotting_payloads.append(copy.deepcopy(payload))
        return {"choices": [{"message": {"content": ""}, "finish_reason": "stop"}]}

    spotting._send_request = capture_spotting
    spotting._run_attempt(
        "data:image/png;base64,AA==",
        repeat_penalty=1.05,
        repeat_last_n=64,
        attempt_index=0,
    )
    spotting_payload = built_spotting_payloads[0]

    class Session:
        def __init__(self, alias: str, port: int) -> None:
            self.pair = SimpleNamespace(
                ocr_alias=alias,
                ocr_port=port,
            )
            self.gemma_port = 18080
            self.captured: list[tuple[str, dict[str, object]]] = []

        def capture_ocr_request(self, url, payload) -> None:
            self.captured.append((url, copy.deepcopy(payload)))

        def begin_http_request(self, _alias: str) -> float:
            return 1.0

        def finish_http_request(self, _alias: str, *, started: float, successful: bool) -> None:
            assert started == 1.0
            assert successful is True

    class Sender:
        exceptions = Exception

        def __init__(self) -> None:
            self.requests: list[dict[str, object]] = []

        def post(self, *_args, **kwargs):
            self.requests.append(copy.deepcopy(kwargs["json"]))
            return SimpleNamespace(status_code=200)

    for payload, port in ((crop_payload, 18000), (spotting_payload, 18002)):
        original = copy.deepcopy(payload)
        sender = Sender()
        session = Session(str(payload["model"]), port)
        proxy = runtime._RequestsProxy(
            sender,
            session=session,
            alias=str(payload["model"]),
            inject_model=False,
        )
        response = proxy.post(
            f"http://127.0.0.1:{port}/v1/chat/completions",
            json=payload,
        )
        assert response.status_code == 200
        assert payload == original
        assert sender.requests == [original]
        assert session.captured == [
            (f"http://127.0.0.1:{port}/v1/chat/completions", original)
        ]


def test_router_proxy_rejects_loopback_lookalike_before_sender() -> None:
    runtime = _router_runtime_module("router_handoff_lab_runtime_url_test")

    class Sender:
        exceptions = Exception

        def __init__(self) -> None:
            self.calls = 0

        def post(self, *_args, **_kwargs):
            self.calls += 1
            return SimpleNamespace(status_code=200)

    session = SimpleNamespace(
        pair=SimpleNamespace(ocr_alias="PaddleOCR-VL-1.6-0.9B", ocr_port=18000),
        gemma_port=18080,
    )
    sender = Sender()
    proxy = runtime._RequestsProxy(
        sender,
        session=session,
        alias="PaddleOCR-VL-1.6-0.9B",
        inject_model=False,
    )
    with pytest.raises(runtime.RouterHandoffLabError):
        proxy.post(
            "http://127.0.0.1:18000@attacker.invalid/v1/chat/completions",
            json={"model": "PaddleOCR-VL-1.6-0.9B"},
        )
    assert sender.calls == 0


def test_router_window_contract_rejects_loopback_lookalike_endpoint() -> None:
    runtime = _router_runtime_module("router_handoff_lab_runtime_window_url_test")

    class Widget:
        def __init__(self, value: str) -> None:
            self.value = value

        def text(self) -> str:
            return self.value

    adapter = object.__new__(runtime.InstalledRouterHandoffLabRuntime)
    adapter.session = SimpleNamespace(
        gemma_port=18080,
        pair=SimpleNamespace(ocr_port=18000, engine_key="PaddleOCR VL"),
    )
    adapter.window = SimpleNamespace(
        settings_page=SimpleNamespace(
            ui=SimpleNamespace(
                credential_widgets={
                    "Custom Local Server(Gemma)_api_url": Widget("http://127.0.0.1:18080/v1"),
                    "Custom Local Server(Gemma)_model": Widget(runtime.GEMMA_ALIAS),
                },
                paddleocr_vl_server_url_input=Widget(
                    "http://127.0.0.1:18000@attacker.invalid/v1/chat/completions"
                ),
                paddleocr_vl_spotting_server_url_input=Widget(
                    "http://127.0.0.1:18002/v1/chat/completions"
                ),
                hunyuan_ocr_server_url_input=Widget("http://127.0.0.1:28080/v1"),
                mangalmm_ocr_server_url_input=Widget("http://127.0.0.1:28081/v1"),
            )
        )
    )

    with pytest.raises(runtime.RouterHandoffLabError):
        adapter.validate_window_contract()


def test_router_volume_identity_uses_observed_sha_not_filename(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _router_runtime_module("router_handoff_lab_runtime_identity_test")
    expected = {"model.gguf": "a" * 64, "projector.gguf": "b" * 64}

    def matching_run(*_args, **_kwargs):
        return SimpleNamespace(
            stdout=(
                f"{'a' * 64}  /models/model.gguf\n"
                f"{'b' * 64}  /models/projector.gguf\n"
            )
        )

    monkeypatch.setattr(runtime, "_run", matching_run)
    observed = runtime.RouterLabSession._verify_volume_files("safe-volume", expected)
    assert observed["files"]["model.gguf"]["sha256"] == "a" * 64

    monkeypatch.setattr(
        runtime,
        "_run",
        lambda *_args, **_kwargs: SimpleNamespace(
            stdout=(
                f"{'0' * 64}  /models/model.gguf\n"
                f"{'b' * 64}  /models/projector.gguf\n"
            )
        ),
    )
    with pytest.raises(runtime.RouterHandoffLabError):
        runtime.RouterLabSession._verify_volume_files("safe-volume", expected)


def test_round_trip_uses_arbiter_and_product_gpu_return_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from contextlib import contextmanager

    runtime = _router_runtime_module("router_handoff_lab_runtime_round_trip_test")
    events: list[str] = []

    class Session:
        pair = SimpleNamespace(engine_key="PaddleOCR VL", ocr_alias="ocr")
        _round_trip_verified = False

        def load_model(self, alias, *, cancel_checker):
            assert alias == "ocr"
            assert callable(cancel_checker)
            assert cancel_checker() is False
            events.append("load")

        def execute_captured_ocr_round_trip_request(self, *, cancel_checker):
            assert callable(cancel_checker)
            assert cancel_checker() is False
            events.append("request")

        def unload_model(self, alias, *, cancel_checker):
            assert alias == "ocr"
            assert cancel_checker is None
            events.append("unload")
            return {"runtime_state": "sleeping", "gpu_release_expected": True}

        def mark_ocr_round_trip_verified(self, *, elapsed_ms, gpu_return_gate):
            assert elapsed_ms >= 0
            assert gpu_return_gate == {"required": True, "observed": True}
            self._round_trip_verified = True
            events.append("marked")
            return {"verified": True}

    class Arbiter:
        def snapshot(self):
            events.append("snapshot")
            return SimpleNamespace(active_model=None, states={})

        def reset(self):
            events.append("reset")

        def token(self, service):
            assert service == "paddleocr_vl"
            events.append("token")
            return "token"

        @contextmanager
        def model_start(self, token, *, cancel_checker, stale_cleanup):
            assert token == "token"
            assert callable(cancel_checker)
            assert cancel_checker() is False
            assert callable(stale_cleanup)
            events.append("start")
            yield

        @contextmanager
        def model_release(self, service):
            assert service == "paddleocr_vl"
            events.append("release")
            yield SimpleNamespace(target_state=None)

    class Processor:
        def _ocr_runtime_service_name(self, engine_key):
            assert engine_key == "PaddleOCR VL"
            return "paddleocr_vl"

        def _raise_if_cancelled(self):
            events.append("cancel-check")

        def _prewarm_cancel_checker(self):
            # A completed pipeline makes the product prewarm lifecycle look
            # cancelled.  The post-run router probe must not reuse it.
            return True

        def _capture_runtime_gpu_start_baseline(self, service):
            assert service == "paddleocr_vl"
            events.append("baseline")

        def _verify_managed_runtime_gpu_release(self, service, report, *, before):
            assert service == "paddleocr_vl"
            assert report["gpu_release_expected"] is True
            assert before == {"gpu": "before"}
            events.append("gpu-gate")
            return {"required": True, "observed": True}

    adapter = object.__new__(runtime.InstalledRouterHandoffLabRuntime)
    adapter.session = Session()
    adapter.arbiter = Arbiter()
    adapter.window = SimpleNamespace(
        pipeline=SimpleNamespace(stage_batched_processor=Processor())
    )
    monkeypatch.setattr(runtime, "query_cuda_handoff_metrics", lambda: {"gpu": "before"})

    assert adapter.verify_round_trip() == {"verified": True}
    assert [event for event in events if event != "cancel-check"] == [
        "snapshot",
        "reset",
        "token",
        "start",
        "baseline",
        "load",
        "request",
        "release",
        "unload",
        "gpu-gate",
        "marked",
    ]
    assert events.count("cancel-check") == 4


def test_round_trip_refuses_reset_when_pipeline_left_active_or_failed_lease() -> None:
    runtime = _router_runtime_module("router_handoff_lab_runtime_round_trip_guard_test")

    class Session:
        pair = SimpleNamespace(engine_key="PaddleOCR VL", ocr_alias="ocr")
        _round_trip_verified = False

    class Processor:
        def _ocr_runtime_service_name(self, _engine_key):
            raise AssertionError("must not ask for a token before lease guard")

        def _raise_if_cancelled(self):
            return None

    for snapshot in (
        SimpleNamespace(active_model="paddleocr_vl", states={}),
        SimpleNamespace(active_model=None, states={"paddleocr_vl": "release_failed"}),
    ):
        calls: list[str] = []

        class Arbiter:
            def snapshot(self):
                calls.append("snapshot")
                return snapshot

            def reset(self):
                calls.append("reset")

        adapter = object.__new__(runtime.InstalledRouterHandoffLabRuntime)
        adapter.session = Session()
        adapter.arbiter = Arbiter()
        adapter.window = SimpleNamespace(
            pipeline=SimpleNamespace(stage_batched_processor=Processor())
        )

        with pytest.raises(runtime.RouterHandoffLabError):
            adapter.verify_round_trip()
        assert calls == ["snapshot"]


def test_owned_router_cleanup_never_targets_an_unowned_container(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[str]] = []

    labels = {
        router_lab.LAB_LABEL_PROTOCOL: router_lab.PROTOCOL_VERSION,
        router_lab.LAB_LABEL_PAIR: "hunyuanocr",
        router_lab.LAB_LABEL_OWNER: "a" * 16,
    }

    def fake_run(command, **_kwargs):
        calls.append(list(command))
        if command[:3] == ["docker", "inspect", "ct-router-lab-hunyuanocr-123"] and "--format" in command:
            return SimpleNamespace(returncode=0, stdout=json.dumps(labels))
        return SimpleNamespace(returncode=1)

    monkeypatch.setattr(router_lab, "_run", fake_run)
    cleanup = router_lab._cleanup_owned_router_container(
        "ct-router-lab-hunyuanocr-123",
        pair="hunyuanocr",
        owner_token="a" * 16,
    )
    assert cleanup["attempted"] is True
    assert cleanup["orphan"] is False
    assert cleanup["ownership_verified"] is True
    assert calls == [
        [
            "docker",
            "inspect",
            "ct-router-lab-hunyuanocr-123",
            "--format",
            "{{json .Config.Labels}}",
        ],
        ["docker", "stop", "--timeout", "10", "ct-router-lab-hunyuanocr-123"],
        ["docker", "rm", "ct-router-lab-hunyuanocr-123"],
        ["docker", "inspect", "ct-router-lab-hunyuanocr-123"],
    ]
    calls.clear()
    labels[router_lab.LAB_LABEL_OWNER] = "b" * 16
    cleanup = router_lab._cleanup_owned_router_container(
        "ct-router-lab-hunyuanocr-123",
        pair="hunyuanocr",
        owner_token="a" * 16,
    )
    assert cleanup == {
        "attempted": False,
        "orphan": False,
        "ownership_verified": False,
        "foreign_container_present": True,
    }
    assert calls == [
        [
            "docker",
            "inspect",
            "ct-router-lab-hunyuanocr-123",
            "--format",
            "{{json .Config.Labels}}",
        ]
    ]
    with pytest.raises(router_lab.RouterHandoffBenchmarkError):
        router_lab._cleanup_owned_router_container(
            "gemma-local-server",
            pair="hunyuanocr",
            owner_token="a" * 16,
        )


def test_router_adapter_defers_gpu_return_proof_to_product_arbiter() -> None:
    runtime_path = ROOT / "scripts" / "router_handoff_lab_runtime.py"
    runtime_spec = importlib.util.spec_from_file_location(
        "router_handoff_lab_runtime_for_release_test", runtime_path
    )
    assert runtime_spec and runtime_spec.loader
    runtime = importlib.util.module_from_spec(runtime_spec)
    sys.modules[runtime_spec.name] = runtime
    runtime_spec.loader.exec_module(runtime)

    session = object.__new__(runtime.RouterLabSession)
    session._lock = __import__("threading").RLock()
    session._prepared = True
    session._active_alias = "ocr"
    session._events = []
    session._wait_model_state = lambda alias, expected, cancel_checker: {  # type: ignore[method-assign]
        "ocr": "unloaded",
        "gemma": "unloaded",
    }
    session.model_states = lambda: {"ocr": "loaded", "gemma": "unloaded"}  # type: ignore[method-assign]

    original_json_response = runtime._json_response
    try:
        runtime._json_response = lambda *args, **kwargs: (200, {"ok": True})
        report = session.unload_model("ocr")
    finally:
        runtime._json_response = original_json_response

    assert report["runtime_state"] == "sleeping"
    assert report["gpu_release_expected"] is True
    assert session._events[-1]["gpu_return_gate"] == "stage_batched_processor"


def test_router_gpu_return_requires_product_arbiter_evidence() -> None:
    pair = SimpleNamespace(engine_key="PaddleOCR VL")
    evidence, failures = router_lab._router_stage_gpu_return(
        summary={
            "performance_stats": {
                "runtime": {
                    "paddleocr_vl": {
                        "vram_release_gate_observed_count": 1,
                        "vram_release_gate_wall_ms": 32.5,
                    },
                    "gemma": {
                        "vram_release_gate_observed_count": 1,
                        "vram_release_gate_wall_ms": 41.0,
                    },
                }
            }
        },
        pair=pair,
    )
    assert failures == []
    assert evidence["paddleocr_vl"]["vram_return_ms"] == 32.5

    _evidence, failures = router_lab._router_stage_gpu_return(
        summary={"performance_stats": {"runtime": {}}},
        pair=pair,
    )
    assert failures == [
        "gpu_return_gate_missing:paddleocr_vl",
        "gpu_return_gate_missing:gemma",
    ]


def test_router_command_queue_is_reported_against_e2e_only() -> None:
    observed = router_lab._router_command_queue_observation(
        {
            "events": [
                {"event": "arbiter_command_enter", "queue_wait_ms": 10.0},
                {"event": "arbiter_command_enter", "queue_wait_ms": 30.0},
            ]
        },
        e2e_seconds=2.0,
    )
    assert observed == {
        "status": "observed",
        "sample_count": 2,
        "median_queue_wait_ms": 20.0,
        "max_queue_wait_ms": 30.0,
        "e2e_percent": 1.0,
    }


def test_resource_sampler_records_a_synchronous_background_sample(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    def sample(phase: str) -> dict[str, object]:
        events.append(phase)
        return {"phase": phase, "gpu_used_mib": 100}

    class FakeThread:
        def __init__(self, *args, **kwargs) -> None:
            events.append("thread-created")

        def start(self) -> None:
            events.append("thread-started")

        def join(self, timeout=None) -> None:
            del timeout

    monkeypatch.setattr(router_lab.ResourceSampler, "_sample", staticmethod(sample))
    monkeypatch.setattr(router_lab.threading, "Thread", FakeThread)
    sampler = router_lab.ResourceSampler()
    sampler.start()
    assert events == ["background", "thread-created", "thread-started"]
    assert sampler.summary()["background"] == {
        "phase": "background",
        "gpu_used_mib": 100,
    }


def test_public_report_redacts_private_not_eligible_diagnostics() -> None:
    private_reason = "docker: /mnt/private/source-title/raw-response.json"
    report = router_lab._render_report(
        {
            "pairs": [
                {
                    "pair": "mangalmm",
                    "status": "NOT_ELIGIBLE",
                    "reason": private_reason,
                }
            ]
        }
    )
    assert private_reason not in report
    assert "/mnt/private" not in report
    assert "preflight or runtime contract was unavailable" in report


def test_partial_run_cannot_publish_the_tracked_aggregate_report() -> None:
    catalog = {"paddle-crop": object(), "hunyuanocr": object()}
    assert not router_lab._can_publish_report(
        mode="review",
        selected=["paddle-crop"],
        catalog=catalog,
    )
    assert not router_lab._can_publish_report(
        mode="abba",
        selected=list(catalog),
        catalog=catalog,
    )
    assert router_lab._can_publish_report(
        mode="review",
        selected=list(catalog),
        catalog=catalog,
    )


def test_zero_failed_pages_passes_the_small_full_auto_completion_check() -> None:
    snapshot = {
        "pages": [
            {
                "page_failed": False,
                "translated_image_exists": True,
                "translated_image_decoded_pixel_sha256": "a" * 64,
                "private_stage_contract": _private_stage_contract(),
            }
        ]
    }
    assert router_lab._full_auto_quality_failures(
        snapshot,
        {"page_failed_count": 0, "page_done_count": 1},
    ) == []


def test_upstream_snapshot_requires_raw_ocr_and_inpaint_contract_hashes() -> None:
    page = {
        "source_lang": "Japanese",
        "target_lang": "Korean",
        "ocr_quality": {},
        "stage_status": {},
        "blocks": [],
        "private_stage_contract": _private_stage_contract(),
    }
    first = router_lab._pre_translation_snapshot_sha256({"pages": [page]})
    changed = copy.deepcopy(page)
    changed["private_stage_contract"]["ocr_raw_results_sha256"] = "x" * 64
    second = router_lab._pre_translation_snapshot_sha256({"pages": [changed]})
    assert first != second

    incomplete = copy.deepcopy(page)
    incomplete["private_stage_contract"].pop("inpaint_decoded_pixel_sha256")
    with pytest.raises(router_lab.RouterHandoffBenchmarkError):
        router_lab._pre_translation_snapshot_sha256({"pages": [incomplete]})


def test_baseline_ocr_quality_failure_marks_fixture_not_eligible() -> None:
    results = [
        {"failures": ["gemma_http_record_empty", "summary_page_failed"]},
        {"failures": []},
        {"failures": []},
        {"failures": ["summary_page_failed", "gemma_http_record_empty"]},
    ]
    assert router_lab._baseline_fixture_not_eligible(results) == (
        "baseline_fixture_ocr_quality_gate_failed_before_translation"
    )

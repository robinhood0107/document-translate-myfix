from __future__ import annotations

import copy
import csv
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import benchmark_gemma_final_translation as benchmark  # noqa: E402


class GemmaFinalTranslationBenchmarkTests(unittest.TestCase):
    def _frozen_fixture(
        self,
        root: Path,
    ) -> tuple[Path, Path, dict[str, object]]:
        input_root = root / "example_source_chapter"
        input_root.mkdir()
        snapshot_corpus = root / "snapshot_corpus"
        snapshot_corpus.mkdir()
        pages = []
        counts = [13] * 21 + [19]
        for page_index, block_count in enumerate(counts, start=1):
            name = f"page-{page_index:03d}.png"
            image_path = input_root / name
            Image.new(
                "RGB",
                (4, 4),
                color=(page_index, page_index, page_index),
            ).save(image_path)
            snapshot_image_path = snapshot_corpus / name
            snapshot_image_path.write_bytes(image_path.read_bytes())
            pages.append(
                {
                    "image_name": name,
                    "image_path": str(snapshot_image_path),
                    "source_lang": "Japanese",
                    "target_lang": "Korean",
                    "blocks": [
                        {
                            "text": f"neutral source {page_index}-{block_index}",
                            "xyxy": [0, block_index, 100, block_index + 1],
                        }
                        for block_index in range(1, block_count + 1)
                    ],
                }
            )
        snapshot_path = root / "page_snapshots.json"
        snapshot_path.write_text(
            json.dumps(
                {
                    "page_count": benchmark.EXPECTED_PAGE_COUNT,
                    "pages": pages,
                }
            ),
            encoding="utf-8",
        )
        manifest = benchmark.build_input_manifest(input_root)
        return input_root, snapshot_path, manifest

    def test_frozen_corpus_requires_exact_order_and_292_blocks(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_root, snapshot_path, manifest = self._frozen_fixture(root)

            corpus = benchmark.load_frozen_corpus(
                input_root=input_root,
                snapshot_path=snapshot_path,
                input_manifest=manifest,
            )

        self.assertEqual(corpus["page_count"], 22)
        self.assertEqual(corpus["block_count"], 292)
        self.assertEqual(len(corpus["page_contract"]), 22)
        self.assertEqual(
            benchmark.expected_grouped_request_count(corpus["pages"], 7),
            45,
        )

    def test_frozen_corpus_rejects_changed_snapshot_image(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_root, snapshot_path, manifest = self._frozen_fixture(root)
            first_image = input_root / "page-001.png"
            Image.new("RGB", (4, 4), color=(255, 0, 0)).save(first_image)

            with self.assertRaisesRegex(
                ValueError,
                "changed after manifest",
            ):
                benchmark.load_frozen_corpus(
                    input_root=input_root,
                    snapshot_path=snapshot_path,
                    input_manifest=manifest,
                )

    def test_frozen_asset_digests_reject_a_structurally_valid_edit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_root, snapshot_path, manifest = self._frozen_fixture(root)
            corpus = benchmark.load_frozen_corpus(
                input_root=input_root,
                snapshot_path=snapshot_path,
                input_manifest=manifest,
            )
            benchmark.validate_frozen_asset_digests(
                input_manifest=manifest,
                corpus=corpus,
                expected_input_manifest_sha256=str(
                    manifest["manifest_sha256"]
                ),
                expected_ocr_snapshot_sha256=str(corpus["snapshot_sha256"]),
            )
            frozen_snapshot_sha256 = str(corpus["snapshot_sha256"])
            payload = json.loads(snapshot_path.read_text(encoding="utf-8"))
            payload["pages"][0]["blocks"][0]["text"] = "edited source"
            snapshot_path.write_text(
                json.dumps(payload),
                encoding="utf-8",
            )
            edited_corpus = benchmark.load_frozen_corpus(
                input_root=input_root,
                snapshot_path=snapshot_path,
                input_manifest=manifest,
            )

            with self.assertRaisesRegex(ValueError, "OCR snapshot SHA-256"):
                benchmark.validate_frozen_asset_digests(
                    input_manifest=manifest,
                    corpus=edited_corpus,
                    expected_input_manifest_sha256=str(
                        manifest["manifest_sha256"]
                    ),
                    expected_ocr_snapshot_sha256=frozen_snapshot_sha256,
                )

    def test_translation_behavior_contract_changes_with_group_size(self) -> None:
        group_seven = benchmark.build_translation_behavior_contract(
            group_size=7
        )
        group_eight = benchmark.build_translation_behavior_contract(
            group_size=8
        )

        self.assertNotEqual(
            group_seven["contract_sha256"],
            group_eight["contract_sha256"],
        )
        self.assertIn(
            "modules/translation/llm/custom_local_gemma.py",
            group_seven["source_sha256"],
        )
        self.assertIn(
            "modules/utils/gpu_metrics.py",
            group_seven["source_sha256"],
        )

    def test_current_checkout_refuses_retired_grouped_replay(self) -> None:
        with self.assertRaisesRegex(
            RuntimeError,
            benchmark.HISTORICAL_GROUPED_REPLAY_COMMIT,
        ):
            benchmark.require_historical_grouped_runtime()

    def test_historical_grouped_replay_guard_accepts_live_contract(self) -> None:
        with patch.object(
            benchmark.gemma_runtime,
            "GEMMA_REQUEST_MODE_CONTEXTUAL_GROUPED",
            benchmark.GEMMA_REQUEST_MODE_CONTEXTUAL_GROUPED,
            create=True,
        ):
            benchmark.require_historical_grouped_runtime()

    def test_api_base_url_is_loopback_only(self) -> None:
        self.assertEqual(
            benchmark.validate_api_base_url(
                "http://127.0.0.1:18080/v1"
            ),
            "http://127.0.0.1:18080/v1",
        )
        for invalid in (
            "http://example.com:18080/v1",
            "http://127.0.0.1:18081/v1",
            "http://user@127.0.0.1:18080/v1",
            "http://127.0.0.1:18080/v1?redirect=1",
        ):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(ValueError, "loopback"):
                    benchmark.validate_api_base_url(invalid)

    def test_candidate_orders_counterbalance_three_rounds(self) -> None:
        self.assertEqual(
            benchmark.candidate_order(1),
            (
                benchmark.CANDIDATE_BASELINE,
                benchmark.CANDIDATE_GROUPED_F16,
                benchmark.CANDIDATE_GROUPED_Q8,
            ),
        )
        self.assertEqual(
            benchmark.candidate_order(2),
            (
                benchmark.CANDIDATE_GROUPED_F16,
                benchmark.CANDIDATE_GROUPED_Q8,
                benchmark.CANDIDATE_BASELINE,
            ),
        )
        self.assertEqual(
            benchmark.candidate_order(3),
            (
                benchmark.CANDIDATE_GROUPED_Q8,
                benchmark.CANDIDATE_BASELINE,
                benchmark.CANDIDATE_GROUPED_F16,
            ),
        )
        positions = {
            candidate: [
                benchmark.candidate_order(round_index).index(candidate)
                for round_index in (1, 2, 3)
            ]
            for candidate in benchmark.CANDIDATE_KEYS
        }
        self.assertEqual(
            positions,
            {
                benchmark.CANDIDATE_BASELINE: [0, 2, 1],
                benchmark.CANDIDATE_GROUPED_F16: [1, 0, 2],
                benchmark.CANDIDATE_GROUPED_Q8: [2, 1, 0],
            },
        )

    def test_third_round_is_added_for_high_variance_or_close_finalists(self) -> None:
        required, reasons = benchmark.should_add_third_round(
            {
                benchmark.CANDIDATE_BASELINE: [100.0, 101.0],
                benchmark.CANDIDATE_GROUPED_F16: [70.0, 71.0],
                benchmark.CANDIDATE_GROUPED_Q8: [72.0, 72.5],
            }
        )
        self.assertTrue(required)
        self.assertTrue(any("finalist difference" in reason for reason in reasons))

        required, reasons = benchmark.should_add_third_round(
            {
                benchmark.CANDIDATE_BASELINE: [100.0, 108.0],
                benchmark.CANDIDATE_GROUPED_F16: [70.0, 70.5],
                benchmark.CANDIDATE_GROUPED_Q8: [80.0, 80.5],
            }
        )
        self.assertTrue(required)
        self.assertTrue(any("run variation" in reason for reason in reasons))

        required, reasons = benchmark.should_add_third_round(
            {
                benchmark.CANDIDATE_BASELINE: [100.0, 101.0],
                benchmark.CANDIDATE_GROUPED_F16: [70.0, 70.5],
                benchmark.CANDIDATE_GROUPED_Q8: [80.0, 80.5],
            }
        )
        self.assertFalse(required)
        self.assertEqual(reasons, [])

    def _runtime_inspect_payload(
        self,
        *,
        candidate: str,
    ) -> list[dict[str, object]]:
        command = benchmark.expected_candidate_command(
            candidate=candidate,
            model_name=benchmark.DEFAULT_MODEL_NAME,
        )
        return [
            {
                "Id": "container-id",
                "Created": "2026-01-01T00:00:00Z",
                "Image": benchmark.DEFAULT_IMAGE_ID,
                "Config": {
                    "Cmd": command,
                    "Labels": {
                        "comic-translate.runtime": "gemma-probe",
                        "comic-translate.profile": "neutral-profile",
                        "comic-translate.config-fingerprint": (
                            benchmark.DEFAULT_RUNTIME_CONFIG_FINGERPRINTS[
                                candidate
                            ]
                        ),
                        "org.opencontainers.image.version": "b-test",
                        "org.opencontainers.image.revision": "revision",
                    },
                },
                "State": {"Status": "exited"},
                "HostConfig": {
                    "PortBindings": {
                        "8080/tcp": [
                            {
                                "HostIp": "127.0.0.1",
                                "HostPort": "18080",
                            }
                        ],
                    },
                    "NetworkMode": "bridge",
                    "PublishAllPorts": False,
                    "Privileged": False,
                    "AutoRemove": False,
                    "RestartPolicy": {
                        "Name": "no",
                        "MaximumRetryCount": 0,
                    },
                    "DeviceRequests": [{"Driver": ""}],
                },
                "NetworkSettings": {"Ports": {}},
                "Mounts": [
                    {
                        "Destination": "/models",
                        "Type": "volume",
                        "Name": benchmark.DEFAULT_MODEL_VOLUME,
                        "RW": False,
                    }
                ],
            }
        ]

    def test_runtime_contract_rejects_mixed_candidate_settings(self) -> None:
        payload = self._runtime_inspect_payload(
            candidate=benchmark.CANDIDATE_GROUPED_Q8
        )
        with patch.object(benchmark, "docker_json", return_value=payload):
            contract = benchmark.inspect_candidate_runtime(
                docker_executable="docker.exe",
                container_name="neutral-container",
                candidate=benchmark.CANDIDATE_GROUPED_Q8,
                expected_image_id=benchmark.DEFAULT_IMAGE_ID,
                expected_model_name=benchmark.DEFAULT_MODEL_NAME,
                expected_model_volume=benchmark.DEFAULT_MODEL_VOLUME,
                expected_config_fingerprint=(
                    benchmark.DEFAULT_RUNTIME_CONFIG_FINGERPRINTS[
                        benchmark.CANDIDATE_GROUPED_Q8
                    ]
                ),
            )
        self.assertEqual(contract["model_mount_read_only"], True)

        with patch.object(benchmark, "docker_json", return_value=payload):
            with self.assertRaisesRegex(ValueError, "full llama.cpp command"):
                benchmark.inspect_candidate_runtime(
                    docker_executable="docker.exe",
                    container_name="neutral-container",
                    candidate=benchmark.CANDIDATE_GROUPED_F16,
                    expected_image_id=benchmark.DEFAULT_IMAGE_ID,
                    expected_model_name=benchmark.DEFAULT_MODEL_NAME,
                    expected_model_volume=benchmark.DEFAULT_MODEL_VOLUME,
                    expected_config_fingerprint=(
                        benchmark.DEFAULT_RUNTIME_CONFIG_FINGERPRINTS[
                            benchmark.CANDIDATE_GROUPED_F16
                        ]
                    ),
                )

    def test_runtime_contract_rejects_exposure_and_command_drift(self) -> None:
        candidate = benchmark.CANDIDATE_GROUPED_F16
        expected_fingerprint = (
            benchmark.DEFAULT_RUNTIME_CONFIG_FINGERPRINTS[candidate]
        )
        base_payload = self._runtime_inspect_payload(candidate=candidate)

        exposed = copy.deepcopy(base_payload)
        exposed[0]["HostConfig"]["PortBindings"]["8080/tcp"][0][
            "HostIp"
        ] = ""
        with patch.object(benchmark, "docker_json", return_value=exposed):
            with self.assertRaisesRegex(ValueError, "127.0.0.1:18080"):
                benchmark.inspect_candidate_runtime(
                    docker_executable="docker.exe",
                    container_name="neutral-container",
                    candidate=candidate,
                    expected_image_id=benchmark.DEFAULT_IMAGE_ID,
                    expected_model_name=benchmark.DEFAULT_MODEL_NAME,
                    expected_model_volume=benchmark.DEFAULT_MODEL_VOLUME,
                    expected_config_fingerprint=expected_fingerprint,
                )

        alternate_model = copy.deepcopy(base_payload)
        alternate_model[0]["Config"]["Cmd"][1] = (
            f"/other/{benchmark.DEFAULT_MODEL_NAME}"
        )
        with patch.object(
            benchmark,
            "docker_json",
            return_value=alternate_model,
        ):
            with self.assertRaisesRegex(ValueError, "full llama.cpp command"):
                benchmark.inspect_candidate_runtime(
                    docker_executable="docker.exe",
                    container_name="neutral-container",
                    candidate=candidate,
                    expected_image_id=benchmark.DEFAULT_IMAGE_ID,
                    expected_model_name=benchmark.DEFAULT_MODEL_NAME,
                    expected_model_volume=benchmark.DEFAULT_MODEL_VOLUME,
                    expected_config_fingerprint=expected_fingerprint,
                )

        extra_flag = copy.deepcopy(base_payload)
        extra_flag[0]["Config"]["Cmd"].extend(["--seed", "123"])
        with patch.object(benchmark, "docker_json", return_value=extra_flag):
            with self.assertRaisesRegex(ValueError, "full llama.cpp command"):
                benchmark.inspect_candidate_runtime(
                    docker_executable="docker.exe",
                    container_name="neutral-container",
                    candidate=candidate,
                    expected_image_id=benchmark.DEFAULT_IMAGE_ID,
                    expected_model_name=benchmark.DEFAULT_MODEL_NAME,
                    expected_model_volume=benchmark.DEFAULT_MODEL_VOLUME,
                    expected_config_fingerprint=expected_fingerprint,
                )

    def test_runtime_contract_rejects_nested_model_mount_shadowing(self) -> None:
        candidate = benchmark.CANDIDATE_GROUPED_F16
        payload = self._runtime_inspect_payload(candidate=candidate)
        payload[0]["Mounts"].append(
            {
                "Destination": f"/models/{benchmark.DEFAULT_MODEL_NAME}",
                "Type": "bind",
                "Source": "C:\\shadow\\alternate.gguf",
                "RW": False,
            }
        )
        with patch.object(benchmark, "docker_json", return_value=payload):
            with self.assertRaisesRegex(ValueError, "none below /models"):
                benchmark.inspect_candidate_runtime(
                    docker_executable="docker.exe",
                    container_name="neutral-container",
                    candidate=candidate,
                    expected_image_id=benchmark.DEFAULT_IMAGE_ID,
                    expected_model_name=benchmark.DEFAULT_MODEL_NAME,
                    expected_model_volume=benchmark.DEFAULT_MODEL_VOLUME,
                    expected_config_fingerprint=(
                        benchmark.DEFAULT_RUNTIME_CONFIG_FINGERPRINTS[
                            candidate
                        ]
                    ),
                )

    def test_runtime_contract_rejects_unsafe_network_and_lifecycle(
        self,
    ) -> None:
        candidate = benchmark.CANDIDATE_GROUPED_F16
        expected_fingerprint = (
            benchmark.DEFAULT_RUNTIME_CONFIG_FINGERPRINTS[candidate]
        )
        unsafe_values = (
            ("NetworkMode", "host", "network mode"),
            ("PublishAllPorts", True, "publish all"),
            ("Privileged", True, "privileged"),
            ("AutoRemove", True, "auto-remove"),
            (
                "RestartPolicy",
                {"Name": "always", "MaximumRetryCount": 0},
                "restart policy",
            ),
        )
        for field, value, expected_error in unsafe_values:
            with self.subTest(field=field):
                payload = self._runtime_inspect_payload(
                    candidate=candidate
                )
                payload[0]["HostConfig"][field] = value
                with patch.object(
                    benchmark,
                    "docker_json",
                    return_value=payload,
                ):
                    with self.assertRaisesRegex(
                        ValueError,
                        expected_error,
                    ):
                        benchmark.inspect_candidate_runtime(
                            docker_executable="docker.exe",
                            container_name="neutral-container",
                            candidate=candidate,
                            expected_image_id=benchmark.DEFAULT_IMAGE_ID,
                            expected_model_name=benchmark.DEFAULT_MODEL_NAME,
                            expected_model_volume=(
                                benchmark.DEFAULT_MODEL_VOLUME
                            ),
                            expected_config_fingerprint=(
                                expected_fingerprint
                            ),
                        )

    def test_all_runtime_contracts_are_validated_before_any_stop(self) -> None:
        first_contract = {
            "candidate": benchmark.CANDIDATE_BASELINE,
            "container_name": "baseline",
            "container_id": "baseline-id",
        }
        stop_mock = MagicMock()
        with (
            patch.object(
                benchmark,
                "inspect_candidate_runtime",
                side_effect=[
                    first_contract,
                    ValueError("invalid second runtime"),
                ],
            ),
            patch.object(benchmark, "stop_container", stop_mock),
        ):
            with self.assertRaisesRegex(ValueError, "invalid second runtime"):
                benchmark.inspect_and_stop_candidate_runtimes(
                    docker_executable="docker.exe",
                    containers={
                        benchmark.CANDIDATE_BASELINE: "baseline",
                        benchmark.CANDIDATE_GROUPED_F16: "f16",
                        benchmark.CANDIDATE_GROUPED_Q8: "q8",
                    },
                    expected_image_id=benchmark.DEFAULT_IMAGE_ID,
                    expected_model_name=benchmark.DEFAULT_MODEL_NAME,
                    expected_model_volume=benchmark.DEFAULT_MODEL_VOLUME,
                    expected_runtime_fingerprints=(
                        benchmark.DEFAULT_RUNTIME_CONFIG_FINGERPRINTS
                    ),
                )
        stop_mock.assert_not_called()

    def test_stat_aggregation_preserves_gauges_and_detects_request_drift(
        self,
    ) -> None:
        combined: dict[str, int | float] = {}
        page_stats = {
            "gemma_telemetry_schema_version": 1,
            "gemma_configured_group_size": 7,
            "gemma_max_requested_group_size": 7,
            "gemma_logical_request_count": 3,
        }
        benchmark._sum_stats(combined, page_stats)
        benchmark._sum_stats(combined, page_stats)

        self.assertEqual(combined["gemma_telemetry_schema_version"], 1)
        self.assertEqual(combined["gemma_configured_group_size"], 7)
        self.assertEqual(combined["gemma_max_requested_group_size"], 7)
        self.assertEqual(combined["gemma_logical_request_count"], 6)

        expected = {
            "gemma_logical_request_count": 6,
            "gemma_http_attempt_count": 6,
        }
        mismatches = benchmark.request_stat_mismatches(
            stats={"gemma_logical_request_count": 7},
            expected_stats=expected,
        )
        self.assertEqual(
            mismatches,
            {
                "gemma_logical_request_count": {
                    "expected": 6,
                    "actual": 7,
                },
                "gemma_http_attempt_count": {
                    "expected": 6,
                    "actual": 0,
                },
            },
        )

    def test_resume_rejects_path_escape_and_result_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_root, snapshot_path, manifest = self._frozen_fixture(root)
            corpus = benchmark.load_frozen_corpus(
                input_root=input_root,
                snapshot_path=snapshot_path,
                input_manifest=manifest,
            )
            output_dir = root / "results"
            runs_dir = output_dir / "runs"
            runs_dir.mkdir(parents=True)
            fingerprint = "f" * 64

            escaped_path = root / "escaped.json"
            benchmark.write_json(
                escaped_path,
                {
                    "candidate": benchmark.CANDIDATE_BASELINE,
                    "round": 1,
                    "status": "failed",
                    "contract_fingerprint": fingerprint,
                },
            )
            escaped_state = {
                "runs": [
                    {
                        "candidate": benchmark.CANDIDATE_BASELINE,
                        "round": 1,
                        "result_file": "../escaped.json",
                        "result_sha256": benchmark.sha256_file(escaped_path),
                    }
                ]
            }
            with self.assertRaisesRegex(ValueError, "escapes"):
                benchmark._load_completed_results(
                    output_dir=output_dir,
                    state=escaped_state,
                    contract_fingerprint=fingerprint,
                    corpus=corpus,
                    group_size=7,
                )

            result_path = runs_dir / "failed.json"
            benchmark.write_json(
                result_path,
                {
                    "candidate": benchmark.CANDIDATE_BASELINE,
                    "round": 1,
                    "status": "failed",
                    "contract_fingerprint": fingerprint,
                },
            )
            tampered_state = {
                "runs": [
                    {
                        "candidate": benchmark.CANDIDATE_BASELINE,
                        "round": 1,
                        "result_file": "runs/failed.json",
                        "result_sha256": benchmark.sha256_file(result_path),
                    }
                ]
            }
            result_path.write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "SHA-256 differs"):
                benchmark._load_completed_results(
                    output_dir=output_dir,
                    state=tampered_state,
                    contract_fingerprint=fingerprint,
                    corpus=corpus,
                    group_size=7,
                )

    def test_start_failure_still_stops_the_pinned_container(self) -> None:
        stop_mock = MagicMock()
        with (
            patch.object(
                benchmark,
                "start_container",
                side_effect=TimeoutError("docker start timed out"),
            ),
            patch.object(benchmark, "stop_container", stop_mock),
            patch.object(
                benchmark,
                "query_resource_snapshot",
                return_value={
                    "gpu_compute_processes": {
                        "available": False,
                        "rows": [],
                    }
                },
            ),
        ):
            result = benchmark.run_candidate(
                candidate=benchmark.CANDIDATE_BASELINE,
                round_index=1,
                container_name="candidate-name",
                container_id="pinned-container-id",
                docker_executable="docker.exe",
                api_base_url="http://127.0.0.1:18080/v1",
                model_name=benchmark.DEFAULT_MODEL_NAME,
                corpus={"pages": []},
                group_size=7,
                contract_fingerprint="f" * 64,
                start_timeout_sec=1,
                request_timeout_sec=1,
            )

        self.assertEqual(result["status"], "failed")
        stop_mock.assert_called_once_with(
            "docker.exe",
            "candidate-name",
            expected_container_id="pinned-container-id",
        )
        self.assertTrue(result["container_stopped"])

    def test_mid_candidate_failure_keeps_completed_output_and_telemetry(
        self,
    ) -> None:
        class FakeEngine:
            def __init__(self) -> None:
                self.calls = 0
                self.last_benchmark_stats: dict[str, int] = {}

            def translate(self, blocks, image, extra_context):
                self.calls += 1
                if self.calls == 2:
                    self.last_benchmark_stats = {
                        "gemma_parser_error_count": 1,
                    }
                    raise RuntimeError("synthetic page failure")
                for block in blocks:
                    block.translation = "번역"
                self.last_benchmark_stats = {
                    "gemma_logical_request_count": len(blocks),
                    "gemma_http_attempt_count": len(blocks),
                    "gemma_contextual_single_request_count": len(blocks),
                    "gemma_configured_group_size": 6,
                    "gemma_tm_requested_block_count": len(blocks),
                }
                return blocks

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_root, snapshot_path, manifest = self._frozen_fixture(root)
            corpus = benchmark.load_frozen_corpus(
                input_root=input_root,
                snapshot_path=snapshot_path,
                input_manifest=manifest,
            )
            stop_mock = MagicMock()
            with (
                patch.object(benchmark, "start_container"),
                patch.object(benchmark, "stop_container", stop_mock),
                patch.object(benchmark, "wait_for_runtime", return_value="model"),
                patch.object(
                    benchmark,
                    "build_engine",
                    return_value=FakeEngine(),
                ),
                patch.object(
                    benchmark,
                    "warm_runtime",
                    return_value={"elapsed_sec": 0.0},
                ),
                patch.object(
                    benchmark,
                    "query_resource_snapshot",
                    return_value={
                        "gpu_compute_processes": {
                            "available": False,
                            "rows": [],
                        }
                    },
                ),
            ):
                result = benchmark.run_candidate(
                    candidate=benchmark.CANDIDATE_BASELINE,
                    round_index=1,
                    container_name="candidate-name",
                    container_id="pinned-container-id",
                    docker_executable="docker.exe",
                    api_base_url="http://127.0.0.1:18080/v1",
                    model_name=benchmark.DEFAULT_MODEL_NAME,
                    corpus=corpus,
                    group_size=7,
                    contract_fingerprint="f" * 64,
                    start_timeout_sec=1,
                    request_timeout_sec=1,
                )

        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["page_count"], 1)
        self.assertEqual(result["block_count"], 13)
        self.assertEqual(len(result["outputs"]), 13)
        self.assertEqual(result["failed_page"]["page_index"], 2)
        self.assertEqual(result["stats"]["gemma_parser_error_count"], 1)
        self.assertTrue(result["container_stopped"])

    def test_gpu_memory_requires_stable_external_inventory_and_docker_pid(
        self,
    ) -> None:
        def snapshot(
            rows: list[dict[str, object]],
            docker_pids: list[int],
        ) -> dict[str, object]:
            return {
                "gpu_compute_processes": {
                    "available": True,
                    "rows": rows,
                },
                "docker_processes": {
                    "available": True,
                    "pids": docker_pids,
                },
            }

        external = {
            "pid": 101,
            "process_name": "external.exe",
            "gpu_uuid": "GPU-1",
            "memory_used_mb": 500.0,
        }
        candidate_before = {
            "pid": 202,
            "process_name": "llama-server",
            "gpu_uuid": "GPU-1",
            "memory_used_mb": 9_100.0,
        }
        candidate_after = {
            **candidate_before,
            "memory_used_mb": 9_200.0,
        }
        attribution = benchmark.attribute_candidate_gpu_memory(
            idle_snapshot=snapshot([external], []),
            before_translation=snapshot(
                [external, candidate_before],
                [202, 303],
            ),
            after_translation=snapshot(
                [external, candidate_after],
                [202, 303],
            ),
        )
        self.assertTrue(attribution["available"])
        self.assertEqual(attribution["memory_used_mb"], 9_200.0)

        changed_external = {
            "pid": 404,
            "process_name": "new-external.exe",
            "gpu_uuid": "GPU-1",
            "memory_used_mb": 100.0,
        }
        unavailable = benchmark.attribute_candidate_gpu_memory(
            idle_snapshot=snapshot([external], []),
            before_translation=snapshot(
                [external, candidate_before],
                [202],
            ),
            after_translation=snapshot(
                [external, changed_external, candidate_after],
                [202],
            ),
        )
        self.assertFalse(unavailable["available"])
        self.assertIsNone(unavailable["memory_used_mb"])

    def test_execution_resume_clears_preflight_terminal_fields(self) -> None:
        state = {
            "status": "preflight_passed",
            "completed_at": 123.0,
            "failure_reason": "stale",
            "runs": [],
        }
        benchmark.reset_state_for_execution(state)
        self.assertEqual(state["status"], "running")
        self.assertNotIn("completed_at", state)
        self.assertNotIn("failure_reason", state)

    def test_resume_validation_failure_preserves_terminal_state(self) -> None:
        state = {
            "status": "preflight_passed",
            "completed_at": 123.0,
            "failure_reason": "preserve-me",
            "runs": [],
        }
        original = copy.deepcopy(state)
        with patch.object(
            benchmark,
            "_completed_result_map",
            side_effect=ValueError("tampered resume"),
        ):
            with self.assertRaisesRegex(ValueError, "tampered resume"):
                benchmark.validate_resume_then_reset_state(
                    output_dir=Path("unused"),
                    state=state,
                    contract_fingerprint="f" * 64,
                    corpus={"pages": []},
                    group_size=7,
                )
        self.assertEqual(state, original)

    def test_successful_resume_reuses_passed_results_and_preserves_runs(
        self,
    ) -> None:
        run_records = [{"sentinel": "preserve"}]
        state = {
            "status": "preflight_passed",
            "completed_at": 123.0,
            "failure_reason": "stale",
            "runs": run_records,
        }
        completed = {
            (1, candidate): {
                "candidate": candidate,
                "round": 1,
                "status": "passed",
            }
            for candidate in benchmark.CANDIDATE_KEYS
        }
        with patch.object(
            benchmark,
            "_completed_result_map",
            return_value=completed,
        ):
            resumed = benchmark.validate_resume_then_reset_state(
                output_dir=Path("unused"),
                state=state,
                contract_fingerprint="f" * 64,
                corpus={"pages": []},
                group_size=7,
            )

        self.assertIs(resumed, completed)
        self.assertEqual(state["status"], "running")
        self.assertNotIn("completed_at", state)
        self.assertNotIn("failure_reason", state)
        self.assertIs(state["runs"], run_records)

        run_mock = MagicMock()
        write_mock = MagicMock()
        with (
            patch.object(benchmark, "run_candidate", run_mock),
            patch.object(benchmark, "write_json", write_mock),
        ):
            succeeded = benchmark.execute_benchmark_round(
                round_index=1,
                completed=resumed,
                state=state,
                output_dir=Path("unused"),
                containers={
                    candidate: f"{candidate}-container"
                    for candidate in benchmark.CANDIDATE_KEYS
                },
                container_ids={
                    candidate: f"{candidate}-id"
                    for candidate in benchmark.CANDIDATE_KEYS
                },
                docker_executable="docker.exe",
                api_base_url="http://127.0.0.1:18080/v1",
                model_name=benchmark.DEFAULT_MODEL_NAME,
                corpus={"pages": []},
                group_size=7,
                contract_fingerprint="f" * 64,
                start_timeout_sec=1,
                request_timeout_sec=1,
            )

        self.assertTrue(succeeded)
        run_mock.assert_not_called()
        write_mock.assert_not_called()
        self.assertIs(state["runs"], run_records)

    def test_run_candidate_success_enforces_hard_and_request_gates(
        self,
    ) -> None:
        class FakeEngine:
            def __init__(self) -> None:
                self.last_benchmark_stats: dict[str, int] = {}

            def translate(self, blocks, image, extra_context):
                for block in blocks:
                    block.translation = f"번역: {block.text}"
                count = len(blocks)
                self.last_benchmark_stats = {
                    "gemma_logical_request_count": count,
                    "gemma_http_attempt_count": count,
                    "gemma_contextual_single_request_count": count,
                    "gemma_configured_group_size": (
                        benchmark.DEFAULT_BASELINE_GROUP_SIZE
                    ),
                    "gemma_tm_requested_block_count": count,
                }
                return blocks

        resource_snapshot = {
            "gpu_compute_processes": {
                "available": False,
                "rows": [],
                "reason": "synthetic unavailable",
            },
            "docker_processes": {
                "available": False,
                "pids": [],
            },
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_root, snapshot_path, manifest = self._frozen_fixture(root)
            corpus = benchmark.load_frozen_corpus(
                input_root=input_root,
                snapshot_path=snapshot_path,
                input_manifest=manifest,
            )
            stop_mock = MagicMock()
            with (
                patch.object(benchmark, "start_container"),
                patch.object(benchmark, "stop_container", stop_mock),
                patch.object(
                    benchmark,
                    "wait_for_runtime",
                    return_value="model",
                ),
                patch.object(
                    benchmark,
                    "build_engine",
                    return_value=FakeEngine(),
                ),
                patch.object(
                    benchmark,
                    "warm_runtime",
                    return_value={"elapsed_sec": 0.0},
                ),
                patch.object(
                    benchmark,
                    "query_resource_snapshot",
                    return_value=resource_snapshot,
                ),
            ):
                result = benchmark.run_candidate(
                    candidate=benchmark.CANDIDATE_BASELINE,
                    round_index=1,
                    container_name="candidate-name",
                    container_id="pinned-container-id",
                    docker_executable="docker.exe",
                    api_base_url="http://127.0.0.1:18080/v1",
                    model_name=benchmark.DEFAULT_MODEL_NAME,
                    corpus=corpus,
                    group_size=7,
                    contract_fingerprint="f" * 64,
                    start_timeout_sec=1,
                    request_timeout_sec=1,
                )

        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["page_count"], 22)
        self.assertEqual(result["block_count"], 292)
        self.assertTrue(result["gates"]["hard_gate_passed"])
        self.assertTrue(result["gates"]["request_contract_passed"])
        self.assertEqual(result["gates"]["request_stat_mismatches"], {})
        self.assertEqual(result["gates"]["clean_run_telemetry_count"], 0)
        self.assertTrue(result["container_stopped"])
        stop_mock.assert_called_once_with(
            "docker.exe",
            "candidate-name",
            expected_container_id="pinned-container-id",
        )

    def test_measurement_environment_identity_is_fail_closed(self) -> None:
        valid = {
            "docker_server": {
                "available": True,
                "version": "29.5.2",
                "api_version": "1.54",
                "git_commit": "abc123",
                "os": "linux",
                "arch": "amd64",
                "kernel_version": "6.18",
            },
            "nvidia_driver": {
                "available": True,
                "gpus": [
                    {
                        "uuid": "GPU-1",
                        "name": "RTX",
                        "driver_version": "610.52",
                    }
                ],
            },
        }
        benchmark.validate_measurement_environment(valid)

        for field in ("docker_server", "nvidia_driver"):
            with self.subTest(field=field):
                invalid = copy.deepcopy(valid)
                invalid[field] = {"available": False}
                with self.assertRaisesRegex(
                    ValueError,
                    "Measurement environment",
                ):
                    benchmark.validate_measurement_environment(invalid)

    def _synthetic_result(
        self,
        candidate: str,
        round_index: int,
        elapsed: float,
        peak_vram: int,
    ) -> dict[str, object]:
        return {
            "candidate": candidate,
            "round": round_index,
            "status": "passed",
            "translation_elapsed_sec": elapsed,
            "gates": {
                "hard_gate_passed": True,
                "clean_run_passed": True,
            },
            "stats": {
                "gemma_logical_request_count": 10,
            },
            "candidate_gpu_memory": {
                "available": True,
                "memory_used_mb": peak_vram,
            },
            "outputs": [],
        }

    def test_suite_summary_applies_speed_and_q8_resource_gates(self) -> None:
        results = []
        for round_index in (1, 2):
            results.extend(
                [
                    self._synthetic_result(
                        benchmark.CANDIDATE_BASELINE,
                        round_index,
                        100.0 + round_index,
                        10_000,
                    ),
                    self._synthetic_result(
                        benchmark.CANDIDATE_GROUPED_F16,
                        round_index,
                        70.0 + round_index,
                        9_800,
                    ),
                    self._synthetic_result(
                        benchmark.CANDIDATE_GROUPED_Q8,
                        round_index,
                        69.0 + round_index,
                        9_000,
                    ),
                ]
            )

        summary = benchmark.build_suite_summary(
            results=results,
            third_round_required=False,
            third_round_reasons=[],
            expected_baseline_requests=292,
            expected_grouped_requests=53,
            q8_vram_materiality_mb=512,
        )

        self.assertEqual(summary["status"], "awaiting_user_quality_review")
        self.assertTrue(
            summary["gates"][
                "grouped_f16_translation_improvement_at_least_20_percent"
            ]
        )
        self.assertTrue(
            summary["gates"][
                "q8_at_least_3_percent_faster_or_material_vram_savings"
            ]
        )
        self.assertFalse(
            summary["gates"]["full_pipeline_promotion_allowed"]
        )

        for result in results:
            result["candidate_gpu_memory"] = {
                "available": False,
                "memory_used_mb": None,
            }
        no_attribution = benchmark.build_suite_summary(
            results=results,
            third_round_required=False,
            third_round_reasons=[],
            expected_baseline_requests=292,
            expected_grouped_requests=53,
            q8_vram_materiality_mb=512,
        )
        self.assertFalse(
            no_attribution["gates"][
                "q8_at_least_3_percent_faster_or_material_vram_savings"
            ]
        )

    def test_blind_review_hides_candidate_names_and_escapes_content(self) -> None:
        results = []
        neutral_outputs = {
            benchmark.CANDIDATE_BASELINE: "첫 번째 번역",
            benchmark.CANDIDATE_GROUPED_F16: "두 번째 번역",
            benchmark.CANDIDATE_GROUPED_Q8: "세 번째 번역",
        }
        for candidate in benchmark.CANDIDATE_KEYS:
            outputs = [
                {
                    "row_id": "p001-b001",
                    "page_name": "page-001.png",
                    "block_index": 1,
                    "source": "source | line",
                    "translation": f"{neutral_outputs[candidate]} | translated",
                }
            ]
            result = self._synthetic_result(
                candidate,
                1,
                10.0,
                9_000,
            )
            result["outputs"] = outputs
            results.append(result)
        summary = {
            "candidate_summaries": [
                {
                    "candidate": candidate,
                    "representative_round": 1,
                }
                for candidate in benchmark.CANDIDATE_KEYS
            ]
        }
        key = {
            "label_to_candidate": {
                "A": benchmark.CANDIDATE_GROUPED_Q8,
                "B": benchmark.CANDIDATE_BASELINE,
                "C": benchmark.CANDIDATE_GROUPED_F16,
            }
        }
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary)
            benchmark.write_blind_review(
                output_dir=output_dir,
                results=results,
                summary=summary,
                existing_key=key,
            )
            markdown = (output_dir / "blind_review.md").read_text(
                encoding="utf-8"
            )
            csv_text = (output_dir / "blind_review.csv").read_text(
                encoding="utf-8-sig"
            )

        self.assertNotIn(benchmark.CANDIDATE_BASELINE, markdown)
        self.assertNotIn(benchmark.CANDIDATE_GROUPED_F16, markdown)
        self.assertNotIn(benchmark.CANDIDATE_GROUPED_Q8, markdown)
        self.assertIn("source &#124; line", markdown)
        self.assertIn("decision", csv_text)

    def test_blind_review_rejects_invalid_key_and_output_order(self) -> None:
        results = []
        for candidate in benchmark.CANDIDATE_KEYS:
            result = self._synthetic_result(candidate, 1, 10.0, 9_000)
            result["outputs"] = [
                {
                    "row_id": row_id,
                    "page_name": "page-001.png",
                    "block_index": index,
                    "source": f"source-{index}",
                    "translation": f"{candidate}-{index}",
                }
                for index, row_id in enumerate(
                    ("p001-b001", "p001-b002"),
                    start=1,
                )
            ]
            results.append(result)
        summary = {
            "candidate_summaries": [
                {
                    "candidate": candidate,
                    "representative_round": 1,
                }
                for candidate in benchmark.CANDIDATE_KEYS
            ]
        }
        invalid_key = {
            "label_to_candidate": {
                "A": benchmark.CANDIDATE_BASELINE,
                "B": benchmark.CANDIDATE_GROUPED_F16,
                "C": benchmark.CANDIDATE_GROUPED_F16,
            }
        }
        valid_key = {
            "label_to_candidate": {
                "A": benchmark.CANDIDATE_BASELINE,
                "B": benchmark.CANDIDATE_GROUPED_F16,
                "C": benchmark.CANDIDATE_GROUPED_Q8,
            }
        }
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary)
            with self.assertRaisesRegex(ValueError, "key is invalid"):
                benchmark.write_blind_review(
                    output_dir=output_dir,
                    results=results,
                    summary=summary,
                    existing_key=invalid_key,
                )

            q8_result = next(
                result
                for result in results
                if result["candidate"] == benchmark.CANDIDATE_GROUPED_Q8
            )
            q8_result["outputs"] = list(reversed(q8_result["outputs"]))
            with self.assertRaisesRegex(ValueError, "order differs"):
                benchmark.write_blind_review(
                    output_dir=output_dir,
                    results=results,
                    summary=summary,
                    existing_key=valid_key,
                )

    def test_blind_review_csv_prefixes_formula_like_cells(self) -> None:
        translations = {
            benchmark.CANDIDATE_BASELINE: "+cmd",
            benchmark.CANDIDATE_GROUPED_F16: "-danger",
            benchmark.CANDIDATE_GROUPED_Q8: "@SUM(A1)",
        }
        results = []
        for candidate in benchmark.CANDIDATE_KEYS:
            result = self._synthetic_result(candidate, 1, 10.0, 9_000)
            result["outputs"] = [
                {
                    "row_id": "p001-b001",
                    "page_name": "+page-001.png",
                    "block_index": 1,
                    "source": "=1+1",
                    "translation": translations[candidate],
                }
            ]
            results.append(result)
        summary = {
            "candidate_summaries": [
                {
                    "candidate": candidate,
                    "representative_round": 1,
                }
                for candidate in benchmark.CANDIDATE_KEYS
            ]
        }
        key = {
            "label_to_candidate": {
                "A": benchmark.CANDIDATE_BASELINE,
                "B": benchmark.CANDIDATE_GROUPED_F16,
                "C": benchmark.CANDIDATE_GROUPED_Q8,
            }
        }
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary)
            benchmark.write_blind_review(
                output_dir=output_dir,
                results=results,
                summary=summary,
                existing_key=key,
            )
            with (output_dir / "blind_review.csv").open(
                encoding="utf-8-sig",
                newline="",
            ) as stream:
                rows = list(csv.reader(stream))

        self.assertEqual(rows[1][1], "'+page-001.png")
        self.assertEqual(rows[1][3], "'=1+1")
        self.assertEqual(rows[1][4], "'+cmd")
        self.assertEqual(rows[1][5], "'-danger")
        self.assertEqual(rows[1][6], "'@SUM(A1)")

    def test_gpu_process_parser_and_multiple_candidates_fail_closed(
        self,
    ) -> None:
        completed = MagicMock()
        completed.stdout = (
            "202, llama-server, GPU-1, N/A\n"
            "303, helper-process, GPU-1, 128\n"
        )
        with patch.object(
            benchmark,
            "_run_process",
            return_value=completed,
        ):
            parsed = benchmark.query_gpu_compute_processes()

        self.assertTrue(parsed["available"])
        self.assertIsNone(parsed["rows"][0]["memory_used_mb"])
        self.assertEqual(parsed["rows"][1]["memory_used_mb"], 128.0)

        def snapshot() -> dict[str, object]:
            return {
                "gpu_compute_processes": parsed,
                "docker_processes": {
                    "available": True,
                    "pids": [202, 303],
                },
            }

        attribution = benchmark.attribute_candidate_gpu_memory(
            idle_snapshot={
                "gpu_compute_processes": {
                    "available": True,
                    "rows": [],
                },
            },
            before_translation=snapshot(),
            after_translation=snapshot(),
        )
        self.assertFalse(attribution["available"])
        self.assertIsNone(attribution["memory_used_mb"])
        self.assertIn("exactly one new GPU process", attribution["reason"])

    def test_model_contract_verifies_host_and_volume_full_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            model_source = Path(temporary) / "model.gguf"
            model_source.write_bytes(b"small synthetic model")
            expected_sha256 = benchmark.sha256_file(model_source)
            helper_contract = {
                "reference": "helper",
                "image_id": "sha256:" + ("a" * 64),
                "repo_digests": [],
            }
            volume_contract = {
                "name": "model-volume",
                "driver": "local",
                "created_at": "2026-01-01T00:00:00Z",
                "labels": {},
            }
            with (
                patch.object(
                    benchmark,
                    "inspect_helper_image",
                    return_value=helper_contract,
                ),
                patch.object(
                    benchmark,
                    "inspect_model_volume",
                    return_value=volume_contract,
                ),
                patch.object(
                    benchmark,
                    "volume_model_size",
                    return_value=model_source.stat().st_size,
                ),
                patch.object(
                    benchmark,
                    "volume_model_sha256",
                    return_value=expected_sha256,
                ),
            ):
                contract = benchmark.prepare_model_contract(
                    docker_executable="docker.exe",
                    helper_image="helper",
                    expected_helper_image_id=helper_contract["image_id"],
                    model_source=model_source,
                    model_name=model_source.name,
                    model_volume="model-volume",
                    expected_size=model_source.stat().st_size,
                    expected_sha256=expected_sha256,
                )

            self.assertEqual(contract["source_sha256"], expected_sha256)
            self.assertEqual(contract["volume_sha256"], expected_sha256)
            self.assertTrue(contract["model_copy_verified"])
            self.assertFalse(contract["full_hash_reused"])

            with (
                patch.object(
                    benchmark,
                    "inspect_helper_image",
                    return_value=helper_contract,
                ),
                patch.object(
                    benchmark,
                    "inspect_model_volume",
                    return_value=volume_contract,
                ),
                patch.object(
                    benchmark,
                    "volume_model_size",
                    return_value=model_source.stat().st_size,
                ),
                patch.object(
                    benchmark,
                    "volume_model_sha256",
                    return_value="0" * 64,
                ),
            ):
                with self.assertRaisesRegex(
                    ValueError,
                    "Model SHA-256 contract failed",
                ):
                    benchmark.prepare_model_contract(
                        docker_executable="docker.exe",
                        helper_image="helper",
                        expected_helper_image_id=(
                            helper_contract["image_id"]
                        ),
                        model_source=model_source,
                        model_name=model_source.name,
                        model_volume="model-volume",
                        expected_size=model_source.stat().st_size,
                        expected_sha256=expected_sha256,
                    )

    def test_main_preflight_only_completes_through_exclusive_lock(self) -> None:
        runtime_contracts = [
            {
                "candidate": candidate,
                "container_name": f"{candidate}-container",
                "container_id": f"{candidate}-id",
                "config_fingerprint": (
                    benchmark.DEFAULT_RUNTIME_CONFIG_FINGERPRINTS[candidate]
                ),
            }
            for candidate in benchmark.CANDIDATE_KEYS
        ]
        measurement_environment = {
            "docker_server": {
                "available": True,
                "version": "29.5.2",
                "api_version": "1.54",
                "git_commit": "abc123",
                "os": "linux",
                "arch": "amd64",
                "kernel_version": "6.18",
            },
            "nvidia_driver": {
                "available": True,
                "gpus": [
                    {
                        "uuid": "GPU-1",
                        "name": "RTX",
                        "driver_version": "610.52",
                    }
                ],
            },
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / ".git").mkdir()
            input_root = root / "input"
            input_root.mkdir()
            snapshot_path = root / "snapshot.json"
            snapshot_path.write_text("{}\n", encoding="utf-8")
            model_source = root / benchmark.DEFAULT_MODEL_NAME
            model_source.write_bytes(b"model")
            output_dir = root / "output"
            input_manifest = {
                "manifest_sha256": "1" * 64,
                "files": [],
            }
            corpus = {
                "snapshot_sha256": "2" * 64,
                "contract_sha256": "3" * 64,
                "page_count": 22,
                "block_count": 292,
                "pages": [],
            }
            model_contract = {
                "hash_helper_image": {
                    "image_id": benchmark.DEFAULT_HASH_HELPER_IMAGE_ID,
                }
            }
            execute_mock = MagicMock()
            with (
                patch.object(benchmark, "ROOT", root),
                patch.object(
                    benchmark,
                    "_ensure_results_are_untracked",
                ),
                patch.object(
                    benchmark,
                    "build_input_manifest",
                    return_value=input_manifest,
                ),
                patch.object(
                    benchmark,
                    "load_frozen_corpus",
                    return_value=corpus,
                ),
                patch.object(
                    benchmark,
                    "validate_frozen_asset_digests",
                ),
                patch.object(
                    benchmark,
                    "resolve_docker_executable",
                    return_value="docker.exe",
                ),
                patch.object(
                    benchmark,
                    "inspect_and_stop_candidate_runtimes",
                    return_value=runtime_contracts,
                ),
                patch.object(
                    benchmark,
                    "prepare_model_contract",
                    return_value=model_contract,
                ),
                patch.object(
                    benchmark,
                    "query_docker_server_identity",
                    return_value=measurement_environment["docker_server"],
                ),
                patch.object(
                    benchmark,
                    "query_nvidia_driver_identity",
                    return_value=measurement_environment["nvidia_driver"],
                ),
                patch.object(
                    benchmark,
                    "build_translation_behavior_contract",
                    return_value={"contract_sha256": "4" * 64},
                ),
                patch.object(
                    benchmark,
                    "execute_benchmark_round",
                    execute_mock,
                ),
                patch.object(
                    benchmark,
                    "require_historical_grouped_runtime",
                ),
            ):
                exit_code = benchmark.main(
                    [
                        "--input-root",
                        str(input_root),
                        "--ocr-snapshot",
                        str(snapshot_path),
                        "--results-root",
                        str(root / "results"),
                        "--output-dir",
                        str(output_dir),
                        "--baseline-container",
                        "baseline-container",
                        "--f16-container",
                        "f16-container",
                        "--q8-container",
                        "q8-container",
                        "--model-source",
                        str(model_source),
                        "--expected-model-sha256",
                        "5" * 64,
                        "--preflight-only",
                    ]
                )

            state = benchmark.read_json(output_dir / "suite_state.json")
            self.assertEqual(exit_code, 0)
            self.assertEqual(state["status"], "preflight_passed")
            self.assertEqual(state["quality_status"], "pending_user_review")
            self.assertFalse(state["full_pipeline_executed"])
            self.assertIn("completed_at", state)
            self.assertTrue(
                (
                    root
                    / ".git"
                    / "gemma-final-translation-suite.lock"
                ).is_file()
            )
            execute_mock.assert_not_called()

    def test_resume_accepts_valid_result_and_rejects_semantic_tamper(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_root, snapshot_path, manifest = self._frozen_fixture(root)
            corpus = benchmark.load_frozen_corpus(
                input_root=input_root,
                snapshot_path=snapshot_path,
                input_manifest=manifest,
            )
            output_dir = root / "results"
            result_path = output_dir / "runs" / "baseline.json"
            fingerprint = "f" * 64
            outputs = []
            for page in corpus["pages"]:
                for block in page["blocks"]:
                    translation = f"번역 {block['row_id']}"
                    outputs.append(
                        {
                            "row_id": block["row_id"],
                            "source_sha256": block["source_sha256"],
                            "source": block["source"],
                            "translation": translation,
                            "translation_sha256": hashlib.sha256(
                                translation.encode("utf-8")
                            ).hexdigest(),
                        }
                    )
            expected_stats = benchmark.expected_request_stats(
                candidate=benchmark.CANDIDATE_BASELINE,
                corpus=corpus,
                group_size=7,
            )
            result = {
                "candidate": benchmark.CANDIDATE_BASELINE,
                "round": 1,
                "status": "passed",
                "contract_fingerprint": fingerprint,
                "page_count": 22,
                "block_count": 292,
                "outputs": outputs,
                "stats": expected_stats,
                "gates": {
                    "hard_gate_passed": True,
                    "clean_run_passed": True,
                    "request_contract_passed": True,
                },
                "container_stopped": True,
            }
            benchmark.write_json(result_path, result)
            state = {
                "runs": [
                    {
                        "candidate": benchmark.CANDIDATE_BASELINE,
                        "round": 1,
                        "result_file": "runs/baseline.json",
                        "result_sha256": benchmark.sha256_file(result_path),
                    }
                ]
            }

            loaded = benchmark._load_completed_results(
                output_dir=output_dir,
                state=state,
                contract_fingerprint=fingerprint,
                corpus=corpus,
                group_size=7,
            )
            self.assertEqual(len(loaded), 1)
            self.assertEqual(loaded[0]["block_count"], 292)

            result["outputs"][0]["translation"] = "tampered"
            benchmark.write_json(result_path, result)
            state["runs"][0]["result_sha256"] = benchmark.sha256_file(
                result_path
            )
            with self.assertRaisesRegex(
                ValueError,
                "no longer passes the final contract",
            ):
                benchmark._load_completed_results(
                    output_dir=output_dir,
                    state=state,
                    contract_fingerprint=fingerprint,
                    corpus=corpus,
                    group_size=7,
                )

    def test_helper_image_and_volume_hash_commands_are_pinned(self) -> None:
        with patch.object(
            benchmark,
            "docker_json",
            return_value=[{"Id": "sha256:unexpected"}],
        ):
            with self.assertRaisesRegex(ValueError, "image ID differs"):
                benchmark.inspect_helper_image(
                    docker_executable="docker.exe",
                    helper_image="helper@sha256:digest",
                    expected_image_id="sha256:expected",
                )

        size_process = MagicMock()
        size_process.stdout = "123\n"
        with patch.object(
            benchmark,
            "_run_process",
            return_value=size_process,
        ) as run_mock:
            size = benchmark.volume_model_size(
                docker_executable="docker.exe",
                helper_image="helper@sha256:digest",
                volume_name="model-volume",
                model_name="model.gguf",
            )
        self.assertEqual(size, 123)
        size_command = run_mock.call_args.args[0]
        self.assertIn(
            "type=volume,source=model-volume,target=/models,readonly",
            size_command,
        )
        self.assertEqual(size_command[-3:], ["-c", "%s", "/models/model.gguf"])

        digest = "a" * 64
        hash_process = MagicMock()
        hash_process.stdout = f"{digest}  /models/model.gguf\n"
        with patch.object(
            benchmark,
            "_run_process",
            return_value=hash_process,
        ) as run_mock:
            actual = benchmark.volume_model_sha256(
                docker_executable="docker.exe",
                helper_image="helper@sha256:digest",
                volume_name="model-volume",
                model_name="model.gguf",
            )
        self.assertEqual(actual, digest)
        hash_command = run_mock.call_args.args[0]
        self.assertIn(
            "type=volume,source=model-volume,target=/models,readonly",
            hash_command,
        )
        self.assertEqual(
            hash_command[-3:],
            ["sha256sum", "helper@sha256:digest", "/models/model.gguf"],
        )

        invalid_process = MagicMock()
        invalid_process.stdout = "not-a-digest\n"
        with patch.object(
            benchmark,
            "_run_process",
            return_value=invalid_process,
        ):
            with self.assertRaisesRegex(ValueError, "output is invalid"):
                benchmark.volume_model_sha256(
                    docker_executable="docker.exe",
                    helper_image="helper@sha256:digest",
                    volume_name="model-volume",
                    model_name="model.gguf",
                )

    def test_lifecycle_blocks_occupied_port_and_reports_stale_cleanup(
        self,
    ) -> None:
        with (
            patch.object(benchmark, "inspect_container_id"),
            patch.object(
                benchmark,
                "running_port_18080_containers",
                return_value=["stale-container"],
            ),
            patch.object(benchmark, "_run_process") as run_mock,
        ):
            with self.assertRaisesRegex(RuntimeError, "already in use"):
                benchmark.start_container(
                    docker_executable="docker.exe",
                    container_name="candidate",
                    expected_container_id="candidate-id",
                )
        run_mock.assert_not_called()

        runtime_contracts = {
            candidate: {
                "candidate": candidate,
                "container_name": f"{candidate}-container",
                "container_id": f"{candidate}-id",
            }
            for candidate in benchmark.CANDIDATE_KEYS
        }
        stop_mock = MagicMock()
        with (
            patch.object(
                benchmark,
                "inspect_candidate_runtime",
                side_effect=lambda **kwargs: runtime_contracts[
                    kwargs["candidate"]
                ],
            ),
            patch.object(benchmark, "stop_container", stop_mock),
            patch.object(
                benchmark,
                "running_port_18080_containers",
                return_value=["unexpected-container"],
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "remains occupied"):
                benchmark.inspect_and_stop_candidate_runtimes(
                    docker_executable="docker.exe",
                    containers={
                        candidate: f"{candidate}-container"
                        for candidate in benchmark.CANDIDATE_KEYS
                    },
                    expected_image_id=benchmark.DEFAULT_IMAGE_ID,
                    expected_model_name=benchmark.DEFAULT_MODEL_NAME,
                    expected_model_volume=benchmark.DEFAULT_MODEL_VOLUME,
                    expected_runtime_fingerprints=(
                        benchmark.DEFAULT_RUNTIME_CONFIG_FINGERPRINTS
                    ),
                )
        self.assertEqual(stop_mock.call_count, 3)
        for candidate in benchmark.CANDIDATE_KEYS:
            stop_mock.assert_any_call(
                "docker.exe",
                f"{candidate}-container",
                expected_container_id=f"{candidate}-id",
            )


if __name__ == "__main__":
    unittest.main()

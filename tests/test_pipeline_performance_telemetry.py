from __future__ import annotations

import json
import os
import sys
import tempfile
import threading
import unittest
from concurrent.futures import Future
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from modules.utils.memlog import MemLogger
from pipeline.batch_processor import BatchProcessor
from pipeline.performance_telemetry import (
    PIPELINE_PERFORMANCE_TELEMETRY_SCHEMA_VERSION,
    PipelinePerformanceTelemetry,
)
from pipeline.performance_ranges import performance_range
from pipeline.stage_batched_processor import StageBatchedProcessor


class _FakeClock:
    def __init__(self) -> None:
        self.value = 100.0

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += float(seconds)


class PipelinePerformanceTelemetryTests(unittest.TestCase):
    def test_nvtx_range_is_disabled_without_the_explicit_lab_toggle(self) -> None:
        nvtx = SimpleNamespace(
            range_push=mock.Mock(),
            range_pop=mock.Mock(),
        )
        torch_module = SimpleNamespace(cuda=SimpleNamespace(nvtx=nvtx))

        with mock.patch.dict(os.environ, {"CT_PERFORMANCE_NVTX": ""}), mock.patch.dict(
            sys.modules,
            {"torch": torch_module},
        ):
            with performance_range("ct:test:disabled"):
                completed = True

        self.assertTrue(completed)
        nvtx.range_push.assert_not_called()
        nvtx.range_pop.assert_not_called()

    def test_nvtx_range_wraps_enabled_lab_work(self) -> None:
        nvtx = SimpleNamespace(
            range_push=mock.Mock(),
            range_pop=mock.Mock(),
        )
        torch_module = SimpleNamespace(cuda=SimpleNamespace(nvtx=nvtx))

        with mock.patch.dict(os.environ, {"CT_PERFORMANCE_NVTX": "1"}), mock.patch.dict(
            sys.modules,
            {"torch": torch_module},
        ):
            with performance_range("ct:test:enabled"):
                completed = True

        self.assertTrue(completed)
        nvtx.range_push.assert_called_once_with("ct:test:enabled")
        nvtx.range_pop.assert_called_once_with()

    def test_nvtx_failure_never_changes_the_product_result(self) -> None:
        nvtx = SimpleNamespace(
            range_push=mock.Mock(side_effect=RuntimeError("nvtx unavailable")),
            range_pop=mock.Mock(),
        )
        torch_module = SimpleNamespace(cuda=SimpleNamespace(nvtx=nvtx))

        with mock.patch.dict(os.environ, {"CT_PERFORMANCE_NVTX": "true"}), mock.patch.dict(
            sys.modules,
            {"torch": torch_module},
        ):
            with performance_range("ct:test:failure"):
                completed = True

        self.assertTrue(completed)
        nvtx.range_push.assert_called_once_with("ct:test:failure")
        nvtx.range_pop.assert_not_called()

    def test_stage_runtime_request_cache_and_resource_stats_are_versioned(self) -> None:
        clock = _FakeClock()
        resource_samples = iter(
            [
                {
                    "gpu": {
                        "primary": {
                            "memory_used_mb": 2048,
                            "gpu_util_percent": 35,
                        }
                    },
                    "wsl_swap": {"available": True, "swap_used_mb": 128},
                },
                {
                    "gpu": {
                        "primary": {
                            "memory_used_mb": 4096,
                            "gpu_util_percent": 80,
                        }
                    },
                    "wsl_swap": {"available": True, "swap_used_mb": 512},
                },
                {
                    "gpu": {
                        "primary": {
                            "memory_used_mb": 3072,
                            "gpu_util_percent": 60,
                        }
                    },
                    "wsl_swap": {"available": True, "swap_used_mb": 256},
                },
            ]
        )
        telemetry = PipelinePerformanceTelemetry(
            clock=clock,
            wall_clock=lambda: 1_000.0,
            resource_sampler=lambda: next(resource_samples),
            resource_sampling_enabled=True,
        )

        telemetry.observe_event(
            "batch_run_start",
            payload={"pipeline_mode": "batch", "workflow_mode": "stage_batched_pipeline"},
        )
        telemetry.observe_event("ocr_start", image_path="page.png")
        clock.advance(0.125)
        ocr_event = telemetry.observe_event(
            "ocr_end",
            image_path="page.png",
            payload={
                "cache_status": "hit",
                "ocr_page_profile": {
                    "performance": {
                        "schema_version": 1,
                        "job_count": 2,
                        "queue_wait_ms": 3.5,
                        "encode_ms": 4.5,
                        "logical_request_count": 2,
                        "http_attempt_count": 3,
                        "http_retry_count": 1,
                        "persistent_cache_hit_count": 2,
                        "persistent_cache_miss_count": 1,
                        "persistent_cache_runtime_miss_count": 1,
                        "persistent_cache_disabled_count": 0,
                    }
                },
            },
        )
        telemetry.observe_event("translate_start", image_path="page.png")
        clock.advance(0.250)
        telemetry.observe_event(
            "translate_end",
            image_path="page.png",
            payload={
                "cache_status": "persistent-partial",
                "gemma_logical_request_count": 2,
                "gemma_http_attempt_count": 3,
                "gemma_http_retry_count": 1,
                "gemma_tm_result_cache_hit_count": 1,
                "gemma_tm_result_cache_miss_count": 1,
            },
        )
        telemetry.record_runtime_duration(
            service="gemma",
            operation="start",
            elapsed_ms=800.0,
        )
        telemetry.record_runtime_duration(
            service="gemma",
            operation="wait",
            elapsed_ms=25.0,
        )
        clock.advance(0.025)
        done = telemetry.observe_event("batch_run_done")

        self.assertEqual(
            ocr_event["performance_telemetry_schema_version"],
            PIPELINE_PERFORMANCE_TELEMETRY_SCHEMA_VERSION,
        )
        self.assertEqual(ocr_event["performance_stage_elapsed_ms"], 125.0)
        stats = done["performance_stats"]
        self.assertEqual(stats["schema_version"], 2)
        self.assertEqual(stats["status"], "completed")
        self.assertEqual(stats["stages"]["ocr"], {"wall_ms": 125.0, "count": 1})
        self.assertEqual(stats["stages"]["translate"], {"wall_ms": 250.0, "count": 1})
        self.assertEqual(stats["cache"]["ocr_hit_count"], 1)
        self.assertEqual(stats["cache"]["translate_partial_count"], 1)
        self.assertEqual(stats["paddleocr_vl"]["logical_request_count"], 2)
        self.assertEqual(stats["paddleocr_vl"]["http_attempt_count"], 3)
        self.assertEqual(
            stats["paddleocr_vl"]["persistent_cache_hit_count"],
            2,
        )
        self.assertEqual(
            stats["paddleocr_vl"]["persistent_cache_miss_count"],
            1,
        )
        self.assertEqual(stats["gemma"]["gemma_http_attempt_count"], 3)
        self.assertEqual(stats["runtime"]["gemma"]["start_wall_ms"], 800.0)
        self.assertEqual(stats["runtime"]["gemma"]["wait_wall_ms"], 25.0)
        self.assertEqual(
            stats["resources"]["summary"],
            {
                "gpu_memory_used_peak_mb": 4096.0,
                "gpu_util_peak_percent": 80.0,
                "wsl_swap_used_peak_mb": 512.0,
                "wsl_swap_used_start_mb": 128.0,
                "wsl_swap_used_end_mb": 512.0,
                "wsl_swap_used_delta_mb": 384.0,
            },
        )
        self.assertEqual(
            stats["stage_details"]["ocr"]["queue_wait"]["wall_ms"],
            3.5,
        )
        self.assertEqual(
            stats["stage_details"]["ocr"]["image_encode"]["wall_ms"],
            4.5,
        )

    def test_resource_sampling_is_disabled_by_default_when_requested(self) -> None:
        telemetry = PipelinePerformanceTelemetry(resource_sampling_enabled=False)
        telemetry.observe_event("batch_run_start")
        done = telemetry.observe_event("batch_run_done")

        resources = done["performance_stats"]["resources"]
        self.assertFalse(resources["sampling_enabled"])
        self.assertEqual(resources["samples"], [])

        telemetry.observe_event("batch_run_start")
        failed = telemetry.observe_event("batch_run_failed")
        self.assertEqual(failed["performance_stats"]["status"], "failed")

    def test_project_ocr_hit_does_not_replay_historical_request_metrics(
        self,
    ) -> None:
        telemetry = PipelinePerformanceTelemetry(
            resource_sampling_enabled=False,
        )
        telemetry.observe_event("batch_run_start")
        telemetry.observe_event("ocr_start", image_path="page.png")
        telemetry.observe_event(
            "ocr_end",
            image_path="page.png",
            payload={
                "cache_status": "hit",
                "ocr_page_profile": {
                    "performance": {
                        "logical_request_count": 30,
                        "http_attempt_count": 30,
                        "request_wall_ms": 123_456.0,
                    },
                    "project_checkpoint": {
                        "status": "hit",
                        "inference_count": 0,
                        "http_request_count": 0,
                    },
                },
            },
        )
        done = telemetry.observe_event("batch_run_done")

        stats = done["performance_stats"]
        self.assertEqual(stats["cache"]["ocr_hit_count"], 1)
        self.assertEqual(stats["paddleocr_vl"], {})

    def test_invalid_project_ocr_marker_falls_back_without_crashing(
        self,
    ) -> None:
        telemetry = PipelinePerformanceTelemetry(
            resource_sampling_enabled=False,
        )
        telemetry.observe_event("batch_run_start")
        telemetry.observe_event(
            "ocr_end",
            image_path="page.png",
            payload={
                "cache_status": "hit",
                "ocr_page_profile": {
                    "performance": {"http_attempt_count": 2},
                    "project_checkpoint": {
                        "status": "hit",
                        "inference_count": "unknown",
                        "http_request_count": "unknown",
                    },
                },
            },
        )
        done = telemetry.observe_event("batch_run_done")

        self.assertEqual(
            done["performance_stats"]["paddleocr_vl"][
                "http_attempt_count"
            ],
            2,
        )

    def test_batch_processor_enriches_existing_memlog_contract(self) -> None:
        events: list[tuple[str, dict]] = []
        processor = object.__new__(BatchProcessor)
        processor.main_page = SimpleNamespace(
            _current_batch_run_type="batch",
            settings_page=SimpleNamespace(
                get_workflow_mode=lambda: "stage_batched_pipeline",
            ),
            emit_memlog=lambda tag, **payload: events.append((tag, dict(payload))),
        )

        processor._emit_benchmark_event("batch_run_start", total_images=1)
        processor._emit_benchmark_event("batch_run_done", total_images=1)

        start_tag, start_payload = events[0]
        done_tag, done_payload = events[1]
        self.assertEqual(start_tag, "batch_run_start")
        self.assertEqual(done_tag, "batch_run_done")
        self.assertTrue(start_payload["product_pipeline_entrypoint"])
        self.assertEqual(
            start_payload["performance_telemetry_schema_version"],
            2,
        )
        self.assertEqual(
            start_payload["performance_run_id"],
            done_payload["performance_run_id"],
        )
        self.assertEqual(processor.last_performance_stats["status"], "completed")

    def test_stage_batched_runtime_start_and_wait_are_recorded_separately(self) -> None:
        processor = object.__new__(StageBatchedProcessor)
        processor.main_page = SimpleNamespace(
            is_current_task_cancelled=lambda: False,
            current_worker=None,
        )
        processor.performance_telemetry = PipelinePerformanceTelemetry(
            resource_sampling_enabled=False,
        )
        processor.last_performance_stats = {}
        processor._prewarm_jobs = {}
        fallback_calls: list[str] = []

        processor._await_prewarm_or_run(
            "ocr",
            "OCR",
            "paddleocr_vl",
            lambda: fallback_calls.append("start"),
        )
        completed_job: Future = Future()
        completed_job.set_result(None)
        processor._prewarm_jobs["gemma"] = completed_job
        processor._await_prewarm_or_run(
            "gemma",
            "Gemma",
            "gemma",
            lambda: fallback_calls.append("unexpected"),
        )

        stats = processor.performance_telemetry.snapshot()
        self.assertEqual(fallback_calls, ["start"])
        self.assertEqual(
            stats["runtime"]["paddleocr_vl"]["start_count"],
            1,
        )
        self.assertEqual(
            stats["runtime"]["gemma"]["wait_count"],
            1,
        )
        self.assertNotIn("start_count", stats["runtime"]["gemma"])

    def test_runtime_progress_callback_records_process_model_and_failure_states(
        self,
    ) -> None:
        forwarded: list[dict] = []
        clock = _FakeClock()
        processor = object.__new__(StageBatchedProcessor)
        processor.main_page = SimpleNamespace(
            report_runtime_progress=lambda payload: forwarded.append(dict(payload)),
        )
        processor.performance_telemetry = PipelinePerformanceTelemetry(
            clock=clock,
            resource_sampling_enabled=False,
        )
        processor.last_performance_stats = {}
        processor._runtime_progress_lock = threading.RLock()
        processor._runtime_progress_started = {}
        observe = processor._runtime_progress_callback()

        with mock.patch(
            "pipeline.stage_batched_processor.time.perf_counter",
            side_effect=clock,
        ):
            observe(
                {
                    "service": "paddleocr_vl",
                    "step_key": "container_start",
                    "status": "starting",
                }
            )
            clock.advance(0.25)
            observe(
                {
                    "service": "paddleocr_vl",
                    "step_key": "container_start",
                    "status": "completed",
                }
            )
            observe(
                {
                    "service": "paddleocr_vl",
                    "step_key": "health_wait",
                    "status": "waiting_health",
                }
            )
            clock.advance(0.5)
            observe(
                {
                    "service": "paddleocr_vl",
                    "step_key": "health_wait",
                    "status": "failed",
                }
            )

        stats = processor.performance_telemetry.snapshot()
        self.assertEqual(len(forwarded), 4)
        self.assertEqual(
            stats["runtime"]["paddleocr_vl"]["container_start_wall_ms"],
            250.0,
        )
        self.assertEqual(
            stats["runtime"]["paddleocr_vl"]["health_wait_wall_ms"],
            500.0,
        )
        self.assertEqual(
            stats["runtime_state"]["current"]["paddleocr_vl"],
            "stopped",
        )
        self.assertEqual(
            [item["to"] for item in stats["runtime_state"]["transitions"]],
            [
                "process_starting",
                "process_ready",
                "model_loading",
                "stopped",
            ],
        )

    def test_v2_workload_graph_runtime_state_and_private_values_are_safe(self) -> None:
        clock = _FakeClock()
        telemetry = PipelinePerformanceTelemetry(
            clock=clock,
            resource_sampling_enabled=False,
        )
        telemetry.observe_event("batch_run_start")
        telemetry.record_workload_features(
            "ocr",
            {
                "page_count": 6,
                "source_language": "ja",
                "source_path": r"C:\private\chapter01.jpg",
                "image_name": "private-page.jpg",
                "raw_text": "secret",
            },
        )
        telemetry.record_runtime_transition(
            service="paddleocr_vl",
            to_state="model_loading",
        )
        with telemetry.measure(
            stage="ocr",
            operation="stage_window",
            workload={"crop_count": 73},
            node_id="stage.ocr",
            dependencies=("stage.detect",),
            service="paddleocr_vl",
        ):
            clock.advance(0.75)
        telemetry.record_runtime_transition(
            service="paddleocr_vl",
            to_state="model_ready",
            elapsed_ms=750.0,
        )

        stats = telemetry.snapshot()
        self.assertEqual(stats["workload"]["ocr"]["page_count"], 6)
        self.assertEqual(stats["workload"]["ocr"]["source_language"], "ja")
        self.assertNotIn("source_path", stats["workload"]["ocr"])
        self.assertNotIn("image_name", stats["workload"]["ocr"])
        self.assertNotIn("raw_text", stats["workload"]["ocr"])
        self.assertEqual(
            stats["stage_details"]["ocr"]["stage_window"]["wall_ms"],
            750.0,
        )
        self.assertEqual(stats["work_graph"][0]["node_id"], "stage.ocr")
        self.assertEqual(
            stats["work_graph"][0]["dependencies"],
            ["stage.detect"],
        )
        self.assertEqual(
            stats["runtime_state"]["current"]["paddleocr_vl"],
            "model_ready",
        )

    def test_measure_propagates_operation_errors_and_records_failure(self) -> None:
        telemetry = PipelinePerformanceTelemetry(resource_sampling_enabled=False)

        with self.assertRaisesRegex(RuntimeError, "boom"):
            with telemetry.measure(stage="render", operation="commit"):
                raise RuntimeError("boom")

        detail = telemetry.snapshot()["stage_details"]["render"]["commit"]
        self.assertEqual(detail["failed_count"], 1)

    def test_measurement_recording_failure_does_not_change_product_outcome(self) -> None:
        telemetry = PipelinePerformanceTelemetry(resource_sampling_enabled=False)

        with mock.patch.object(
            telemetry,
            "record_stage_detail",
            side_effect=RuntimeError("telemetry failed"),
        ):
            with telemetry.measure(stage="render", operation="commit"):
                completed = True

        self.assertTrue(completed)

        with mock.patch.object(
            telemetry,
            "record_stage_detail",
            side_effect=RuntimeError("telemetry failed"),
        ), self.assertRaisesRegex(ValueError, "product failed"):
            with telemetry.measure(stage="render", operation="commit"):
                raise ValueError("product failed")

    def test_benchmark_memlog_tick_includes_gpu_and_wsl_swap(self) -> None:
        logger = MemLogger(SimpleNamespace())
        logger._gpu_bench_enabled = True
        with mock.patch(
            "modules.utils.gpu_metrics.query_gpu_metrics",
            return_value={"available": True},
        ), mock.patch(
            "modules.utils.gpu_metrics.query_wsl_swap_metrics",
            return_value={"available": True, "swap_used_mb": 64.0},
        ):
            snapshot = logger._snapshot("tick")

        self.assertEqual(snapshot["gpu"], {"available": True})
        self.assertEqual(snapshot["wsl_swap"]["swap_used_mb"], 64.0)

    def test_memlog_can_bind_to_debug_runtime_sidecar(self) -> None:
        logger = MemLogger(SimpleNamespace())
        logger._gpu_bench_enabled = True
        with tempfile.TemporaryDirectory() as temporary:
            runtime = Path(temporary) / "runtime"
            logger.bind_debug_run(str(runtime))
            logger._append_payload(
                {
                    "ts": 1.0,
                    "tag": "tick",
                    "gpu": {"available": True},
                    "wsl_swap": {"swap_used_mb": 32.0},
                }
            )

            memlog = [
                json.loads(line)
                for line in (runtime / "memlog.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            gpu_log = [
                json.loads(line)
                for line in (runtime / "gpu-bench.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertEqual(memlog[0]["tag"], "tick")
            self.assertEqual(gpu_log[0]["gpu"], {"available": True})
            logger.unbind_debug_run()
            self.assertIsNone(logger._debug_runtime_dir)

    def test_memlog_ignores_empty_debug_runtime_path(self) -> None:
        logger = MemLogger(SimpleNamespace())

        logger.bind_debug_run("")

        self.assertIsNone(logger._debug_runtime_dir)
        self.assertIsNone(logger._path)


if __name__ == "__main__":
    unittest.main()

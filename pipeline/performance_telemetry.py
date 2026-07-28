from __future__ import annotations

import copy
import os
import time
import uuid
from collections import defaultdict
from typing import Any, Callable, Mapping

from modules.utils.gpu_metrics import (
    query_gpu_metrics_cached,
    query_wsl_swap_metrics,
)


PIPELINE_PERFORMANCE_TELEMETRY_SCHEMA_VERSION = 1
_STAGE_NAMES = frozenset({"detect", "ocr", "inpaint", "translate", "render"})
_TERMINAL_RUN_TAGS = frozenset(
    {
        "batch_run_done",
        "batch_run_cancelled",
        "batch_run_failed",
        "webtoon_run_done",
        "webtoon_run_cancelled",
        "webtoon_run_failed",
    }
)
_ADDITIVE_GEMMA_METRICS = frozenset(
    {
        "gemma_logical_request_count",
        "gemma_http_attempt_count",
        "gemma_request_retry_count",
        "gemma_http_retry_count",
        "gemma_request_wall_ms",
        "gemma_prompt_tokens",
        "gemma_completion_tokens",
        "gemma_total_tokens",
        "gemma_cached_prompt_tokens",
        "gemma_prompt_eval_ms",
        "gemma_decode_ms",
        "gemma_tm_result_cache_hit_count",
        "gemma_tm_result_cache_miss_count",
        "gemma_tm_exact_hit_count",
        "gemma_tm_stale_reject_count",
        "gemma_tm_cache_disabled_count",
        "gemma_tm_runtime_skipped_count",
    }
)
_ADDITIVE_PADDLE_METRICS = frozenset(
    {
        "job_count",
        "queue_wait_ms",
        "crop_ms",
        "text_guard_ms",
        "encode_ms",
        "base64_ms",
        "payload_build_ms",
        "logical_request_count",
        "http_attempt_count",
        "http_retry_count",
        "compatibility_retry_count",
        "retry_backoff_ms",
        "request_wall_ms",
        "response_decode_ms",
        "parse_sanitize_ms",
        "request_bytes",
        "base64_chars",
        "persistent_cache_hit_count",
        "persistent_cache_miss_count",
        "persistent_cache_runtime_miss_count",
        "persistent_cache_disabled_count",
    }
)


def _env_enabled(name: str) -> bool:
    value = str(os.environ.get(name, "") or "").strip().lower()
    return value in {"1", "true", "yes", "on"}


def _default_resource_sampler() -> dict[str, Any]:
    return {
        "sampled_at": time.time(),
        "gpu": query_gpu_metrics_cached(ttl_sec=0.0),
        "wsl_swap": query_wsl_swap_metrics(),
    }


class PipelinePerformanceTelemetry:
    """Versioned, report-agnostic performance counters for product pipelines.

    The class only records generic measurements. Candidate ranking, benchmark
    gates, and report generation remain in the benchmarking layer.
    """

    def __init__(
        self,
        *,
        clock: Callable[[], float] = time.perf_counter,
        wall_clock: Callable[[], float] = time.time,
        resource_sampler: Callable[[], Mapping[str, Any]] | None = None,
        resource_sampling_enabled: bool | None = None,
    ) -> None:
        self._clock = clock
        self._wall_clock = wall_clock
        self._resource_sampler = resource_sampler or _default_resource_sampler
        if resource_sampling_enabled is None:
            resource_sampling_enabled = (
                _env_enabled("CT_ENABLE_GPU_BENCH")
                or _env_enabled("CT_PERFORMANCE_RESOURCE_TELEMETRY")
            )
        self.resource_sampling_enabled = bool(resource_sampling_enabled)
        self.reset(sample_resources=False)

    def reset(
        self,
        metadata: Mapping[str, Any] | None = None,
        *,
        sample_resources: bool = True,
    ) -> None:
        self.run_id = uuid.uuid4().hex
        self._run_started_at = float(self._clock())
        self._run_started_wall = float(self._wall_clock())
        self._metadata = dict(metadata or {})
        self._active_stages: dict[tuple[str, str], float] = {}
        self._stage_wall_ms: defaultdict[str, float] = defaultdict(float)
        self._stage_count: defaultdict[str, int] = defaultdict(int)
        self._runtime: dict[str, defaultdict[str, float | int]] = {}
        self._cache_counts: defaultdict[str, int] = defaultdict(int)
        self._gemma_totals: defaultdict[str, float] = defaultdict(float)
        self._paddle_totals: defaultdict[str, float] = defaultdict(float)
        self._resource_samples: list[dict[str, Any]] = []
        self._terminal_status = "running"
        self._last_snapshot: dict[str, Any] = {}
        if self.resource_sampling_enabled and sample_resources:
            self.sample_resources("run_start")

    def observe_event(
        self,
        tag: str,
        *,
        image_path: str | None = None,
        payload: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        event_payload = dict(payload or {})
        normalized_tag = str(tag or "").strip().lower()
        if normalized_tag in {"batch_run_start", "webtoon_run_start"}:
            self.reset(
                {
                    "pipeline_mode": event_payload.get("pipeline_mode", ""),
                    "run_type": event_payload.get("run_type", ""),
                    "workflow_mode": event_payload.get("workflow_mode", ""),
                }
            )

        now = float(self._clock())
        stage_name, action = self._stage_action(normalized_tag)
        scope = str(image_path or event_payload.get("image_path") or "__run__")
        stage_elapsed_ms: float | None = None
        if normalized_tag == "page_failed":
            self._finish_stage("page", scope, now)
            failed_stage = str(event_payload.get("failed_stage") or "").strip().lower()
            if failed_stage in _STAGE_NAMES:
                stage_elapsed_ms = self._finish_stage(failed_stage, scope, now)
                stage_name = failed_stage
        elif stage_name and action == "start":
            self._active_stages[(stage_name, scope)] = now
        elif stage_name and action == "end":
            stage_elapsed_ms = self._finish_stage(stage_name, scope, now)

        self._ingest_cache_status(stage_name, event_payload)
        self._ingest_gemma_metrics(event_payload)
        self._ingest_paddle_metrics(event_payload)

        event: dict[str, Any] = {
            "performance_telemetry_schema_version": (
                PIPELINE_PERFORMANCE_TELEMETRY_SCHEMA_VERSION
            ),
            "performance_run_id": self.run_id,
            "performance_run_elapsed_ms": round(
                max(0.0, now - self._run_started_at) * 1000.0,
                3,
            ),
        }
        if stage_name:
            event["performance_stage_name"] = stage_name
        if stage_elapsed_ms is not None:
            event["performance_stage_elapsed_ms"] = round(stage_elapsed_ms, 3)
            event["performance_stage_total_ms"] = round(
                self._stage_wall_ms[stage_name],
                3,
            )
            event["performance_stage_count"] = int(self._stage_count[stage_name])

        if normalized_tag in _TERMINAL_RUN_TAGS:
            if normalized_tag.endswith("_cancelled"):
                self._terminal_status = "cancelled"
            elif normalized_tag.endswith("_failed"):
                self._terminal_status = "failed"
            else:
                self._terminal_status = "completed"
            if self.resource_sampling_enabled:
                self.sample_resources("run_end")
            self._last_snapshot = self.snapshot(now=now)
            event["performance_stats"] = copy.deepcopy(self._last_snapshot)
        return event

    def record_runtime_duration(
        self,
        *,
        service: str,
        operation: str,
        elapsed_ms: float,
        outcome: str = "completed",
    ) -> None:
        normalized_service = str(service or "unknown").strip().lower() or "unknown"
        normalized_operation = str(operation or "unknown").strip().lower() or "unknown"
        normalized_outcome = str(outcome or "completed").strip().lower() or "completed"
        metrics = self._runtime.setdefault(
            normalized_service,
            defaultdict(float),
        )
        metrics[f"{normalized_operation}_count"] = int(
            metrics.get(f"{normalized_operation}_count", 0)
        ) + 1
        metrics[f"{normalized_operation}_wall_ms"] = float(
            metrics.get(f"{normalized_operation}_wall_ms", 0.0)
        ) + max(0.0, float(elapsed_ms))
        metrics[f"{normalized_operation}_{normalized_outcome}_count"] = int(
            metrics.get(f"{normalized_operation}_{normalized_outcome}_count", 0)
        ) + 1

    def increment_counter(self, name: str, amount: int = 1) -> None:
        normalized = str(name or "").strip()
        if not normalized:
            return
        self._cache_counts[normalized] += int(amount)

    def sample_resources(self, label: str) -> dict[str, Any]:
        sample: dict[str, Any] = {
            "label": str(label or "sample"),
            "run_elapsed_ms": round(
                max(0.0, float(self._clock()) - self._run_started_at) * 1000.0,
                3,
            ),
        }
        try:
            sampled = self._resource_sampler()
            if isinstance(sampled, Mapping):
                sample.update(copy.deepcopy(dict(sampled)))
        except Exception as exc:
            sample["available"] = False
            sample["reason"] = f"{type(exc).__name__}: {exc}"
        self._resource_samples.append(sample)
        return copy.deepcopy(sample)

    def snapshot(self, *, now: float | None = None) -> dict[str, Any]:
        current = float(self._clock()) if now is None else float(now)
        stages = {
            name: {
                "wall_ms": round(float(self._stage_wall_ms[name]), 3),
                "count": int(self._stage_count[name]),
            }
            for name in sorted(self._stage_wall_ms)
        }
        runtime = {
            service: {
                key: (
                    int(value)
                    if str(key).endswith("_count")
                    else round(float(value), 3)
                )
                for key, value in sorted(metrics.items())
            }
            for service, metrics in sorted(self._runtime.items())
        }
        return {
            "schema_version": PIPELINE_PERFORMANCE_TELEMETRY_SCHEMA_VERSION,
            "run_id": self.run_id,
            "status": self._terminal_status,
            "started_at": self._run_started_wall,
            "run_wall_ms": round(
                max(0.0, current - self._run_started_at) * 1000.0,
                3,
            ),
            "metadata": copy.deepcopy(self._metadata),
            "stages": stages,
            "runtime": runtime,
            "cache": {
                key: int(value)
                for key, value in sorted(self._cache_counts.items())
            },
            "gemma": self._rounded_numeric_map(self._gemma_totals),
            "paddleocr_vl": self._rounded_numeric_map(self._paddle_totals),
            "resources": {
                "sampling_enabled": self.resource_sampling_enabled,
                "samples": copy.deepcopy(self._resource_samples),
                "summary": self._resource_summary(),
            },
        }

    @property
    def last_snapshot(self) -> dict[str, Any]:
        return copy.deepcopy(self._last_snapshot)

    @staticmethod
    def _stage_action(tag: str) -> tuple[str, str]:
        if tag in {"batch_run_start", "webtoon_run_start"}:
            return "pipeline", "start"
        if tag in _TERMINAL_RUN_TAGS:
            return "pipeline", "end"
        if tag == "page_start":
            return "page", "start"
        if tag in {"page_done", "page_failed"}:
            return "page", "end"
        for stage in _STAGE_NAMES:
            if tag == f"{stage}_start":
                return stage, "start"
            if tag == f"{stage}_end":
                return stage, "end"
        return "", ""

    def _finish_stage(self, stage: str, scope: str, now: float) -> float | None:
        started = self._active_stages.pop((stage, scope), None)
        if started is None and stage == "pipeline":
            started = self._run_started_at
        if started is None:
            return None
        elapsed_ms = max(0.0, now - started) * 1000.0
        self._stage_wall_ms[stage] += elapsed_ms
        self._stage_count[stage] += 1
        return elapsed_ms

    def _ingest_cache_status(
        self,
        stage_name: str,
        payload: Mapping[str, Any],
    ) -> None:
        status = str(payload.get("cache_status") or "").strip().lower()
        if not status:
            return
        prefix = stage_name or "pipeline"
        if status in {"hit", "persistent-hit", "all-hit"}:
            self._cache_counts[f"{prefix}_hit_count"] += 1
        elif status in {
            "miss",
            "refreshed",
            "persistent-refreshed",
            "persistent-miss",
        }:
            self._cache_counts[f"{prefix}_miss_count"] += 1
        elif status in {"partial", "persistent-partial"}:
            self._cache_counts[f"{prefix}_partial_count"] += 1
        elif status in {"disabled", "persistent-disabled"}:
            self._cache_counts[f"{prefix}_disabled_count"] += 1

    def _ingest_gemma_metrics(self, payload: Mapping[str, Any]) -> None:
        for key in _ADDITIVE_GEMMA_METRICS:
            value = payload.get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            self._gemma_totals[key] += float(value)

    def _ingest_paddle_metrics(self, payload: Mapping[str, Any]) -> None:
        profile = payload.get("ocr_page_profile")
        if not isinstance(profile, Mapping):
            return
        performance = profile.get("performance")
        if not isinstance(performance, Mapping):
            return
        for key in _ADDITIVE_PADDLE_METRICS:
            value = performance.get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            self._paddle_totals[key] += float(value)

    @staticmethod
    def _rounded_numeric_map(
        values: Mapping[str, float],
    ) -> dict[str, int | float]:
        rounded: dict[str, int | float] = {}
        for key, value in sorted(values.items()):
            if str(key).endswith(("_count", "_tokens", "_bytes", "_chars")):
                rounded[key] = int(round(float(value)))
            else:
                rounded[key] = round(float(value), 3)
        return rounded

    def _resource_summary(self) -> dict[str, Any]:
        gpu_used: list[float] = []
        gpu_util: list[float] = []
        swap_used: list[float] = []
        for sample in self._resource_samples:
            gpu = sample.get("gpu")
            primary = gpu.get("primary") if isinstance(gpu, Mapping) else None
            if isinstance(primary, Mapping):
                used = primary.get("memory_used_mb")
                util = primary.get("gpu_util_percent")
                if isinstance(used, (int, float)) and not isinstance(used, bool):
                    gpu_used.append(float(used))
                if isinstance(util, (int, float)) and not isinstance(util, bool):
                    gpu_util.append(float(util))
            wsl_swap = sample.get("wsl_swap")
            if isinstance(wsl_swap, Mapping):
                used = wsl_swap.get("swap_used_mb")
                if isinstance(used, (int, float)) and not isinstance(used, bool):
                    swap_used.append(float(used))
        return {
            "gpu_memory_used_peak_mb": max(gpu_used) if gpu_used else None,
            "gpu_util_peak_percent": max(gpu_util) if gpu_util else None,
            "wsl_swap_used_peak_mb": max(swap_used) if swap_used else None,
        }

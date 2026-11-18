from __future__ import annotations

from contextlib import nullcontext
import os
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import mock

import numpy as np

from app.projects.checkpoint_store import (
    ProjectCheckpointStore,
    checkpoint_reference_for_save,
)
from app.projects.stage_checkpoints import (
    DetectionCheckpointResult,
    InpaintCheckpointResult,
    RenderCheckpointResult,
    TranslationCheckpointResult,
    build_project_ocr_fingerprint,
    build_project_ocr_identity,
    record_ocr_checkpoint,
)
from app.projects.stage_checkpoints import decoded_image_sha256
from modules.ocr.factory import OCRFactory
from modules.ocr.local_runtime import LocalOCRRuntimeManager
from modules.ocr.ocr_paddle_VL import PaddleOCRVLEngine
from modules.ocr.persistent_cache import (
    OCRPersistentResultCache,
    snapshot_raw_ocr_result,
)
from modules.translation.local_runtime import LocalGemmaRuntimeManager
from modules.utils.ocr_debug import all_empty_blocks_are_rejected
from modules.utils.correction_dictionary import apply_ocr_result_dictionary
from modules.utils.ocr_quality import summarize_ocr_quality
from modules.utils.textblock import TextBlock, ensure_text_block_id
from pipeline import render_worker
from pipeline.runtime_resource_arbiter import RuntimeLeaseConflictError
from pipeline.stage_batched_processor import StageBatchedProcessor, StagePageContext
from modules.utils.exceptions import OperationCancelledError


class StageBatchedCancellationTests(unittest.TestCase):
    def _processor(self, *, cancelled: bool) -> StageBatchedProcessor:
        processor = object.__new__(StageBatchedProcessor)
        processor._released_page_buffer_bytes = 0
        processor.main_page = SimpleNamespace(
            is_current_task_cancelled=lambda: cancelled,
            settings_page=object(),
        )
        processor.block_detection = SimpleNamespace(block_detector_cache=object())
        return processor

    def test_cancel_check_raises_operation_cancelled_error(self) -> None:
        processor = self._processor(cancelled=True)

        with self.assertRaises(OperationCancelledError):
            processor._raise_if_cancelled()

    def test_internal_source_failure_keeps_existing_skip_message(self) -> None:
        processor = self._processor(cancelled=False)
        processor.main_page.image_ctrl = SimpleNamespace(
            update_processing_summary=mock.Mock(),
            mark_processing_stage=mock.Mock(),
        )
        processor.main_page.image_skipped = mock.Mock()
        processor._emit_benchmark_event = mock.Mock()
        processor._release_page_buffers = mock.Mock()
        page = StagePageContext(
            image_path="example_source.png",
            image_name="example_source.png",
            source_lang="Japanese",
            target_lang="Korean",
        )

        processor._mark_page_failed(
            page,
            index=0,
            total_images=1,
            stage="inpaint",
            reason="source_changed_after_detect",
            detail="",
        )

        self.assertEqual(page.failed_reason, "source_changed_after_detect")
        processor.main_page.image_skipped.emit.assert_called_once_with(
            "example_source.png",
            "inpaint",
            "",
        )

    def test_detect_stage_stops_before_processing_page_when_cancelled(self) -> None:
        processor = self._processor(cancelled=True)
        page = StagePageContext(
            image_path="page.png",
            image_name="page.png",
            source_lang="Japanese",
            target_lang="Korean",
        )

        with self.assertRaises(OperationCancelledError):
            processor._detect_all([page])

    def test_prewarm_fallback_is_not_called_after_cancel(self) -> None:
        processor = self._processor(cancelled=True)
        processor._prewarm_jobs = {}
        called = False

        def fallback() -> None:
            nonlocal called
            called = True

        with self.assertRaises(OperationCancelledError):
            processor._await_prewarm_or_run("ocr", "OCR", "hunyuanocr", fallback)

        self.assertFalse(called)

    def test_main_body_failure_or_cancel_survives_terminal_telemetry_failure(self) -> None:
        from collections import deque
        from contextlib import nullcontext

        for error, requested, expected_status in (
            (RuntimeError("worker sentinel"), False, "failed"),
            (OperationCancelledError("cancel sentinel"), True, "cancelled"),
        ):
            with self.subTest(expected_status=expected_status):
                processor = StageBatchedProcessor.__new__(StageBatchedProcessor)
                processor.main_page = SimpleNamespace(
                    file_handler=SimpleNamespace(),
                    is_current_task_cancelled=lambda: requested,
                    _batch_cancel_requested=requested,
                    _batch_pause_requested=False,
                )
                processor._recent_page_durations = deque(maxlen=5)
                processor._emit_benchmark_event = mock.Mock(
                    side_effect=lambda tag, **_kwargs: (
                        (_ for _ in ()).throw(RuntimeError("telemetry sentinel"))
                        if tag in {"batch_run_failed", "batch_run_cancelled"}
                        else None
                    )
                )
                processor._record_performance_workload = mock.Mock()
                processor._benchmark_stage_ceiling = mock.Mock(return_value="")
                processor._reset_prewarm_lifecycle = mock.Mock()
                processor._load_page_contexts = mock.Mock(return_value=[])
                processor._ensure_stage_policy = mock.Mock(return_value={})
                processor._measure_performance = mock.Mock(return_value=nullcontext())
                processor._raise_if_cancelled = mock.Mock(side_effect=error)
                processor._finalize_batch_run = mock.Mock()

                with mock.patch(
                    "pipeline.stage_batched_processor.open_project_stage_checkpoint_store",
                    return_value=None,
                ):
                    if requested:
                        processor.batch_process([])
                    else:
                        with self.assertRaisesRegex(RuntimeError, "worker sentinel"):
                            processor.batch_process([])

                processor._finalize_batch_run.assert_called_once()
                self.assertEqual(
                    processor._finalize_batch_run.call_args.kwargs["processing_status"],
                    expected_status,
                )
                if not requested:
                    self.assertIs(
                        processor._finalize_batch_run.call_args.kwargs[
                            "processing_error"
                        ],
                        error,
                    )

    def test_paddle_persistent_cache_defers_ocr_prewarm_until_lookup(self) -> None:
        processor = self._processor(cancelled=False)
        processor.main_page.local_ocr_runtime_manager = LocalOCRRuntimeManager()
        processor.main_page.settings_page = SimpleNamespace(
            get_paddleocr_vl_settings=lambda: {
                "persistent_cache_enabled": True,
            }
        )
        processor._get_paddleocr_cache_store = mock.Mock(
            return_value=SimpleNamespace(
                stats=lambda: {
                    "enabled": True,
                    "item_count": 1,
                }
            )
        )
        processor._start_prewarm = mock.Mock()

        processor._start_ocr_prewarm(
            {"primary_ocr_engine": "PaddleOCR VL"}
        )

        processor._start_prewarm.assert_not_called()

    def test_empty_paddle_cache_defers_runtime_prewarm_until_plan_miss(self) -> None:
        processor = self._processor(cancelled=False)
        processor.main_page.local_ocr_runtime_manager = LocalOCRRuntimeManager()
        processor.main_page.settings_page = SimpleNamespace(
            get_paddleocr_vl_settings=lambda: {
                "persistent_cache_enabled": True,
            }
        )
        processor._get_paddleocr_cache_store = mock.Mock(
            return_value=SimpleNamespace(
                stats=lambda: {
                    "enabled": True,
                    "item_count": 0,
                }
            )
        )
        processor._start_prewarm = mock.Mock()

        processor._start_ocr_prewarm(
            {"primary_ocr_engine": "PaddleOCR VL"}
        )

        processor._start_prewarm.assert_not_called()

    def test_ocr_prewarm_remains_for_disabled_paddle_cache_and_other_engines(
        self,
    ) -> None:
        for engine_key, enabled in (
            ("PaddleOCR VL", False),
            ("HunyuanOCR", True),
        ):
            with self.subTest(engine_key=engine_key, enabled=enabled):
                processor = self._processor(cancelled=False)
                processor.main_page.local_ocr_runtime_manager = (
                    LocalOCRRuntimeManager()
                )
                processor.main_page.settings_page = SimpleNamespace(
                    get_paddleocr_vl_settings=lambda: {
                        "persistent_cache_enabled": enabled,
                    }
                )
                processor._start_prewarm = mock.Mock()

                processor._start_ocr_prewarm(
                    {"primary_ocr_engine": engine_key}
                )

                processor._start_prewarm.assert_called_once()

    def test_confirmed_paddle_cache_miss_bypasses_unrelated_rows(self) -> None:
        processor = self._processor(cancelled=False)
        processor.main_page.local_ocr_runtime_manager = LocalOCRRuntimeManager()
        processor.main_page.settings_page = SimpleNamespace(
            get_paddleocr_vl_settings=lambda: {
                "persistent_cache_enabled": True,
            }
        )
        processor._project_checkpoint_store = SimpleNamespace(
            has_stage_record=lambda _page_key, _stage: True,
        )
        processor._project_checkpoint_page_keys = ["page:00000000"]
        processor._get_paddleocr_cache_store = mock.Mock(
            return_value=SimpleNamespace(
                stats=lambda: {
                    "enabled": True,
                    "item_count": 42,
                }
            )
        )
        processor._start_prewarm = mock.Mock()

        processor._start_ocr_prewarm(
            {"primary_ocr_engine": "PaddleOCR VL"},
            cache_miss_confirmed=True,
        )

        processor._start_prewarm.assert_called_once()

    def test_detected_page_global_cache_hit_keeps_runtime_stopped(self) -> None:
        processor = self._processor(cancelled=False)
        runtime_manager = LocalOCRRuntimeManager()
        runtime_manager.get_ocr_cache_identity = mock.Mock(
            return_value={"runtime_fingerprint": "runtime"}
        )
        processor.main_page.local_ocr_runtime_manager = runtime_manager
        processor.main_page.settings_page = SimpleNamespace(
            get_paddleocr_vl_settings=lambda: {
                "persistent_cache_enabled": True,
            }
        )
        processor._project_checkpoint_store = None
        processor._prewarm_jobs = {}
        processor._paddleocr_cache_identity = None
        processor._canonicalize_ocr_inputs = mock.Mock()
        processor._prepare_paddleocr_cache_plans = mock.Mock(
            return_value=False
        )
        processor._start_ocr_prewarm = mock.Mock()
        processor._emit_benchmark_event = mock.Mock()
        page = StagePageContext(
            image_path="page.png",
            image_name="page.png",
            source_lang="Japanese",
            target_lang="Korean",
            image=np.zeros((20, 20, 3), dtype=np.uint8),
            blk_list=[
                TextBlock(
                    text_bbox=np.array([1, 1, 10, 10], dtype=np.int32)
                )
            ],
        )

        processor._plan_detected_page_ocr_prewarm(
            page,
            {
                "primary_ocr_engine": "PaddleOCR VL",
                "normalized_ocr_mode": "best_local",
            },
            index=0,
            total_images=1,
        )

        processor._prepare_paddleocr_cache_plans.assert_called_once()
        processor._start_ocr_prewarm.assert_not_called()
        processor._emit_benchmark_event.assert_called_once_with(
            "ocr_prewarm_decision",
            image_path="page.png",
            image_index=0,
            total_images=1,
            decision="defer",
            reason="persistent_cache_hit",
        )

    def test_detected_page_cache_miss_starts_overlap_despite_other_rows(
        self,
    ) -> None:
        processor = self._processor(cancelled=False)
        runtime_manager = LocalOCRRuntimeManager()
        runtime_manager.get_ocr_cache_identity = mock.Mock(
            return_value={"runtime_fingerprint": "runtime"}
        )
        processor.main_page.local_ocr_runtime_manager = runtime_manager
        processor.main_page.settings_page = SimpleNamespace(
            get_paddleocr_vl_settings=lambda: {
                "persistent_cache_enabled": True,
            }
        )
        processor._project_checkpoint_store = None
        processor._prewarm_jobs = {}
        processor._paddleocr_cache_identity = None
        processor._canonicalize_ocr_inputs = mock.Mock()
        processor._prepare_paddleocr_cache_plans = mock.Mock(
            return_value=True
        )
        processor._start_ocr_prewarm = mock.Mock()
        processor._emit_benchmark_event = mock.Mock()
        page = StagePageContext(
            image_path="new-page.png",
            image_name="new-page.png",
            source_lang="Japanese",
            target_lang="Korean",
            image=np.zeros((20, 20, 3), dtype=np.uint8),
            blk_list=[
                TextBlock(
                    text_bbox=np.array([1, 1, 10, 10], dtype=np.int32)
                )
            ],
        )

        processor._plan_detected_page_ocr_prewarm(
            page,
            {
                "primary_ocr_engine": "PaddleOCR VL",
                "normalized_ocr_mode": "best_local",
            },
            index=0,
            total_images=6,
        )

        processor._start_ocr_prewarm.assert_called_once_with(
            {
                "primary_ocr_engine": "PaddleOCR VL",
                "normalized_ocr_mode": "best_local",
            },
            cache_miss_confirmed=True,
        )
        processor._emit_benchmark_event.assert_called_once_with(
            "ocr_prewarm_decision",
            image_path="new-page.png",
            image_index=0,
            total_images=6,
            decision="start",
            reason="persistent_cache_miss",
        )

    def test_detected_page_project_hit_skips_global_cache_and_runtime(
        self,
    ) -> None:
        processor = self._processor(cancelled=False)
        runtime_manager = LocalOCRRuntimeManager()
        runtime_manager.get_ocr_cache_identity = mock.Mock(
            return_value={"runtime_fingerprint": "runtime"}
        )
        processor.main_page.local_ocr_runtime_manager = runtime_manager
        processor.main_page.settings_page = SimpleNamespace(
            get_paddleocr_vl_settings=lambda: {
                "persistent_cache_enabled": True,
            }
        )
        processor._project_checkpoint_store = object()
        processor._prewarm_jobs = {}
        processor._paddleocr_cache_identity = None
        processor._canonicalize_ocr_inputs = mock.Mock()
        checkpoint_hit = object()

        def prepare_project(pages, _policy, _identity) -> None:
            pages[0].project_ocr_hit = checkpoint_hit

        processor._prepare_project_ocr_hits = mock.Mock(
            side_effect=prepare_project
        )
        processor._prepare_paddleocr_cache_plans = mock.Mock()
        processor._start_ocr_prewarm = mock.Mock()
        processor._emit_benchmark_event = mock.Mock()
        page = StagePageContext(
            image_path="page.png",
            image_name="page.png",
            source_lang="Japanese",
            target_lang="Korean",
            image=np.zeros((20, 20, 3), dtype=np.uint8),
            blk_list=[
                TextBlock(
                    text_bbox=np.array([1, 1, 10, 10], dtype=np.int32)
                )
            ],
        )

        processor._plan_detected_page_ocr_prewarm(
            page,
            {
                "primary_ocr_engine": "PaddleOCR VL",
                "normalized_ocr_mode": "best_local",
            },
            index=0,
            total_images=1,
        )

        processor._prepare_paddleocr_cache_plans.assert_not_called()
        processor._start_ocr_prewarm.assert_not_called()
        processor._emit_benchmark_event.assert_called_once_with(
            "ocr_prewarm_decision",
            image_path="page.png",
            image_index=0,
            total_images=1,
            decision="defer",
            reason="project_checkpoint_hit",
        )

    def test_non_empty_project_cache_defers_ocr_prewarm(self) -> None:
        processor = self._processor(cancelled=False)
        processor.main_page.local_ocr_runtime_manager = LocalOCRRuntimeManager()
        processor.main_page.settings_page = SimpleNamespace(
            get_paddleocr_vl_settings=lambda: {
                "persistent_cache_enabled": False,
            }
        )
        processor._project_checkpoint_page_keys = ["page:00000000"]
        processor._project_checkpoint_store = SimpleNamespace(
            has_stage_record=lambda page_key, stage: (
                page_key == "page:00000000" and stage == "ocr"
            ),
        )
        processor._start_prewarm = mock.Mock()

        processor._start_ocr_prewarm(
            {"primary_ocr_engine": "PaddleOCR VL"}
        )

        processor._start_prewarm.assert_not_called()

    def test_detection_checkpoint_defers_ocr_prewarm_for_no_text_hit(
        self,
    ) -> None:
        processor = self._processor(cancelled=False)
        processor.main_page.local_ocr_runtime_manager = LocalOCRRuntimeManager()
        processor.main_page.settings_page = SimpleNamespace(
            get_paddleocr_vl_settings=lambda: {
                "persistent_cache_enabled": False,
            }
        )
        processor._project_checkpoint_page_keys = ["page:00000000"]
        processor._project_checkpoint_store = SimpleNamespace(
            has_stage_record=lambda page_key, stage: (
                page_key == "page:00000000" and stage == "detection"
            ),
        )
        processor._start_prewarm = mock.Mock()

        processor._start_ocr_prewarm(
            {"primary_ocr_engine": "PaddleOCR VL"}
        )

        processor._start_prewarm.assert_not_called()

    def test_empty_project_cache_preserves_cold_runtime_prewarm(self) -> None:
        processor = self._processor(cancelled=False)
        processor.main_page.local_ocr_runtime_manager = LocalOCRRuntimeManager()
        processor.main_page.settings_page = SimpleNamespace(
            get_paddleocr_vl_settings=lambda: {
                "persistent_cache_enabled": False,
            }
        )
        processor._project_checkpoint_page_keys = ["page:00000000"]
        processor._project_checkpoint_store = SimpleNamespace(
            has_stage_record=lambda _page_key, _stage: False,
        )
        processor._start_prewarm = mock.Mock()

        processor._start_ocr_prewarm(
            {"primary_ocr_engine": "PaddleOCR VL"}
        )

        processor._start_prewarm.assert_called_once()

    def test_paddle_all_hit_lookup_skips_runtime_start(self) -> None:
        processor = self._processor(cancelled=False)
        runtime_manager = LocalOCRRuntimeManager()
        processor.main_page.local_ocr_runtime_manager = runtime_manager
        processor.main_page.settings_page = SimpleNamespace(
            get_paddleocr_vl_settings=lambda: {
                "persistent_cache_enabled": True,
            },
            is_gpu_enabled=lambda: False,
            get_tool_selection=lambda _tool: "PaddleOCR VL",
        )
        processor.main_page.image_ctrl = SimpleNamespace(
            mark_processing_stage=mock.Mock(),
            update_processing_summary=mock.Mock(),
        )
        processor._paddleocr_cache_store = None
        processor._paddleocr_cache_identity = None
        processor._await_ocr_runtime = mock.Mock()
        processor._prepare_paddleocr_cache_plans = mock.Mock(
            return_value=False
        )
        processor._set_current_image = mock.Mock()
        processor.emit_progress = mock.Mock()
        processor._emit_benchmark_event = mock.Mock()
        processor._log_ocr_quality = mock.Mock()
        processor._persist_ocr_state = mock.Mock()
        processor._shutdown_runtime_with_retry = mock.Mock()
        processor._raise_if_cancelled = mock.Mock()
        processor._run_primary_ocr = mock.Mock(
            return_value={
                "quality": {"non_empty": 1, "low_quality": False},
                "metrics": {"ocr_non_empty_block_count": 1},
                "cache_status": "hit",
                "attempt_count": 1,
                "page_profile": {
                    "persistent_cache": {
                        "hit_count": 1,
                        "miss_count": 0,
                    }
                },
                "engine_name": "PaddleOCRVLEngine",
            }
        )
        page = StagePageContext(
            image_path="page.png",
            image_name="page.png",
            source_lang="Japanese",
            target_lang="Korean",
            image=object(),
            blk_list=[object()],
        )
        identity = {"runtime_fingerprint": "runtime"}

        with mock.patch.object(
            runtime_manager,
            "get_ocr_cache_identity",
            return_value=identity,
        ):
            processor._ocr_all(
                [page],
                {
                    "primary_ocr_engine": "PaddleOCR VL",
                    "normalized_ocr_mode": "best_local",
                },
            )

        processor._prepare_paddleocr_cache_plans.assert_called_once_with(
            [page],
            {
                "primary_ocr_engine": "PaddleOCR VL",
                "normalized_ocr_mode": "best_local",
            },
            identity,
        )
        processor._await_ocr_runtime.assert_not_called()
        processor._run_primary_ocr.assert_called_once()

    def test_project_ocr_all_hit_short_circuits_global_cache_and_runtime(self) -> None:
        processor = self._processor(cancelled=False)
        runtime_manager = LocalOCRRuntimeManager()
        processor.main_page.local_ocr_runtime_manager = runtime_manager
        processor.main_page.lang_mapping = {"Japanese": "Japanese"}
        processor.main_page.settings_page = SimpleNamespace(
            get_paddleocr_vl_settings=lambda: {
                "persistent_cache_enabled": True,
            },
            is_gpu_enabled=lambda: False,
            get_tool_selection=lambda _tool: "PaddleOCR VL",
        )
        processor.main_page.image_ctrl = SimpleNamespace(
            mark_processing_stage=mock.Mock(),
        )
        processor._project_checkpoint_store = object()
        processor._paddleocr_cache_store = None
        processor._paddleocr_cache_identity = None
        processor._await_ocr_runtime = mock.Mock()
        call_order: list[str] = []
        checkpoint_hit = SimpleNamespace()

        def prepare_project(pages, _policy, _identity) -> None:
            call_order.append("project")
            pages[0].project_ocr_hit = checkpoint_hit
            pages[0].project_ocr_checkpoint_status = "hit"

        processor._prepare_project_ocr_hits = mock.Mock(
            side_effect=prepare_project
        )
        processor._prepare_paddleocr_cache_plans = mock.Mock(
            return_value=False
        )
        processor._set_current_image = mock.Mock()
        processor.emit_progress = mock.Mock()
        processor._emit_benchmark_event = mock.Mock()
        processor._log_ocr_quality = mock.Mock()
        processor._persist_ocr_state = mock.Mock()
        processor._record_project_ocr_result = mock.Mock()
        processor._shutdown_runtime_with_retry = mock.Mock()
        processor._raise_if_cancelled = mock.Mock()
        processor._run_primary_ocr = mock.Mock(
            return_value={
                "quality": {"non_empty": 1, "low_quality": False},
                "metrics": {"ocr_non_empty_block_count": 1},
                "cache_status": "hit",
                "attempt_count": 1,
                "page_profile": {
                    "project_checkpoint": {
                        "status": "hit",
                        "inference_count": 0,
                        "http_request_count": 0,
                    }
                },
                "engine_name": "PaddleOCRVLEngine",
                "raw_results": {},
            }
        )
        page = StagePageContext(
            image_path="page.png",
            image_name="page.png",
            source_lang="Japanese",
            target_lang="Korean",
            image=np.zeros((20, 20, 3), dtype=np.uint8),
            blk_list=[
                TextBlock(
                    text_bbox=np.array([1, 1, 10, 10], dtype=np.int32),
                    text="raw",
                )
            ],
            detection_fingerprint="a" * 64,
        )
        policy = {
            "primary_ocr_engine": "PaddleOCR VL",
            "normalized_ocr_mode": "best_local",
        }

        with mock.patch.object(
            runtime_manager,
            "get_ocr_cache_identity",
            return_value={"runtime_fingerprint": "runtime"},
        ):
            processor._ocr_all([page], policy)

        self.assertEqual(call_order, ["project"])
        processor._prepare_paddleocr_cache_plans.assert_not_called()
        processor._await_ocr_runtime.assert_not_called()
        processor._run_primary_ocr.assert_called_once()

    def test_failed_project_ocr_result_is_not_checkpointed(self) -> None:
        processor = object.__new__(StageBatchedProcessor)
        page = StagePageContext(
            image_path="example_source_page.png",
            image_name="example_source_page.png",
            source_lang="Japanese",
            target_lang="Korean",
            project_ocr_checkpoint_status="miss",
            project_ocr_fingerprint="a" * 64,
            project_ocr_identity={"engine": "PaddleOCR VL Spotting"},
            failed_stage="ocr",
        )

        with mock.patch(
            "pipeline.stage_batched_processor.record_ocr_checkpoint"
        ) as record:
            processor._record_project_ocr_result(
                page,
                {"page_profile": {"failure_reason": "response_incomplete"}},
            )

        record.assert_not_called()

    def test_disabled_persistent_caches_skip_runtime_identity_probe(self) -> None:
        processor = self._processor(cancelled=False)
        runtime_manager = LocalOCRRuntimeManager()
        processor.main_page.local_ocr_runtime_manager = runtime_manager
        processor.main_page.settings_page = SimpleNamespace(
            get_paddleocr_vl_settings=lambda: {
                "persistent_cache_enabled": False,
            },
            is_gpu_enabled=lambda: False,
            get_tool_selection=lambda _tool: "PaddleOCR VL",
        )
        processor.main_page.image_ctrl = SimpleNamespace(
            mark_processing_stage=mock.Mock(),
        )
        processor._project_checkpoint_store = None
        processor._paddleocr_cache_store = None
        processor._paddleocr_cache_identity = None
        processor._await_ocr_runtime = mock.Mock()
        processor._set_current_image = mock.Mock()
        processor.emit_progress = mock.Mock()
        processor._emit_benchmark_event = mock.Mock()
        processor._log_ocr_quality = mock.Mock()
        processor._persist_ocr_state = mock.Mock()
        processor._record_project_ocr_result = mock.Mock()
        processor._shutdown_runtime_with_retry = mock.Mock()
        processor._raise_if_cancelled = mock.Mock()
        processor._run_primary_ocr = mock.Mock(
            return_value={
                "quality": {"non_empty": 1, "low_quality": False},
                "metrics": {"ocr_non_empty_block_count": 1},
                "cache_status": "refreshed",
                "attempt_count": 1,
                "page_profile": {},
                "engine_name": "PaddleOCRVLEngine",
                "raw_results": {},
            }
        )
        page = StagePageContext(
            image_path="page.png",
            image_name="page.png",
            source_lang="Japanese",
            target_lang="Korean",
            image=np.zeros((20, 20, 3), dtype=np.uint8),
            blk_list=[
                TextBlock(
                    text_bbox=np.array([1, 1, 10, 10], dtype=np.int32),
                    text="raw",
                )
            ],
        )
        policy = {
            "primary_ocr_engine": "PaddleOCR VL",
            "normalized_ocr_mode": "best_local",
        }

        with mock.patch.object(
            runtime_manager,
            "get_ocr_cache_identity",
        ) as identity_probe:
            processor._ocr_all([page], policy)

        identity_probe.assert_not_called()
        processor._await_ocr_runtime.assert_called_once_with(policy)
        processor._run_primary_ocr.assert_called_once()
        processor._shutdown_runtime_with_retry.assert_called_once_with(
            "OCR",
            runtime_manager,
            context="OCR-to-translate handoff",
            raise_on_failure=True,
            release_for_handoff=True,
            service="paddleocr_vl",
        )

    def test_detection_checkpoint_hit_skips_detector_construction_and_inference(self) -> None:
        processor = self._processor(cancelled=False)
        processor._project_checkpoint_store = object()
        processor.block_detection = SimpleNamespace(
            block_detector_cache=None,
        )
        processor.main_page.image_files = ["page.png"]
        processor.main_page.lang_mapping = {"Japanese": "Japanese"}
        processor.main_page.image_ctrl = SimpleNamespace(
            load_image=mock.Mock(
                return_value=np.zeros((20, 20, 3), dtype=np.uint8)
            ),
            update_processing_summary=mock.Mock(),
        )
        processor.main_page.settings_page = SimpleNamespace(
            get_tool_selection=lambda tool: {
                "detector": "RT-DETR-v2",
                "inpainter": "LaMa",
            }.get(tool, ""),
            is_gpu_enabled=lambda: False,
        )
        processor._raise_if_cancelled = mock.Mock()
        processor._set_current_image = mock.Mock()
        processor.emit_progress = mock.Mock()
        processor._start_page_summary = mock.Mock()
        processor._log_page_start = mock.Mock()
        processor._emit_benchmark_event = mock.Mock()
        processor._persist_detect_state = mock.Mock()
        processor._effective_export_settings = mock.Mock(return_value={})
        processor._write_detector_overlay_debug_image = mock.Mock(
            return_value=""
        )
        processor._maybe_emit_preview_image = mock.Mock()
        block = TextBlock(
            text_bbox=np.array([1, 1, 10, 10], dtype=np.int32),
            text_class="text_bubble",
            block_id="stable-block",
        )
        hit = DetectionCheckpointResult(
            blocks=[block],
            precomputed_mask_details={"mask": "stable"},
        )
        page = StagePageContext(
            image_path="page.png",
            image_name="page.png",
            source_lang="Japanese",
            target_lang="Korean",
        )
        identity = {
            "detector": "RT-DETR-v2",
            "engine": "RTDetrV2ONNXDetection",
            "device": "cpu",
        }

        with mock.patch(
            "pipeline.stage_batched_processor.build_detection_identity",
            return_value=identity,
        ), mock.patch(
            "pipeline.stage_batched_processor.lookup_detection_checkpoint",
            return_value=hit,
        ), mock.patch(
            "pipeline.stage_batched_processor.TextBlockDetector",
        ) as detector_type:
            processor._detect_all([page])

        detector_type.assert_not_called()
        self.assertEqual(page.detection_checkpoint_status, "hit")
        self.assertEqual(
            [item.block_id for item in page.blk_list],
            ["stable-block"],
        )
        processor._persist_detect_state.assert_called_once()

    def test_detect_records_source_hash_without_project_checkpoint(self) -> None:
        processor = self._processor(cancelled=False)
        detector = SimpleNamespace(
            detect=mock.Mock(
                return_value=[
                    TextBlock(
                        text_bbox=np.array(
                            [1, 1, 10, 10],
                            dtype=np.int32,
                        ),
                        text_class="text_free",
                    )
                ]
            ),
            last_mask_details=None,
            detector="RT-DETR-v2",
            last_engine_name="RTDetrV2ONNXDetection",
            last_device="cpu",
        )
        processor._project_checkpoint_store = None
        processor.block_detection = SimpleNamespace(
            block_detector_cache=detector,
        )
        processor.main_page.image_files = ["page.png"]
        processor.main_page.lang_mapping = {"Japanese": "Japanese"}
        processor.main_page.image_ctrl = SimpleNamespace(
            load_image=mock.Mock(
                return_value=np.zeros((20, 20, 3), dtype=np.uint8)
            ),
            update_processing_summary=mock.Mock(),
        )
        processor.main_page.settings_page = SimpleNamespace(
            get_tool_selection=lambda tool: {
                "detector": "RT-DETR-v2",
                "inpainter": "LaMa",
            }.get(tool, ""),
            is_gpu_enabled=lambda: False,
        )
        processor._raise_if_cancelled = mock.Mock()
        processor._set_current_image = mock.Mock()
        processor.emit_progress = mock.Mock()
        processor._start_page_summary = mock.Mock()
        processor._log_page_start = mock.Mock()
        processor._emit_benchmark_event = mock.Mock()
        processor._persist_detect_state = mock.Mock()
        processor._effective_export_settings = mock.Mock(return_value={})
        processor._write_detector_overlay_debug_image = mock.Mock(
            return_value=""
        )
        processor._maybe_emit_preview_image = mock.Mock()
        page = StagePageContext(
            image_path="page.png",
            image_name="page.png",
            source_lang="Japanese",
            target_lang="Korean",
        )

        with mock.patch(
            "pipeline.stage_batched_processor.decoded_image_sha256",
            return_value="source-identity",
        ) as decoded_hash, mock.patch(
            "pipeline.stage_batched_processor.build_detection_identity",
        ) as build_identity:
            processor._detect_all([page])

        decoded_hash.assert_called_once()
        build_identity.assert_not_called()
        detector.detect.assert_called_once()
        self.assertEqual(page.detection_checkpoint_status, "disabled")
        self.assertEqual(page.source_decoded_sha256, "source-identity")
        self.assertEqual(page.source_pixel_count, 20 * 20)
        self.assertIsNone(page.image)

    def test_paddle_cache_hit_applies_current_dictionary_once(self) -> None:
        processor = self._processor(cancelled=False)
        processor.main_page.lang_mapping = {"Japanese": "Japanese"}
        processor.main_page.settings_page = SimpleNamespace(
            is_gpu_enabled=lambda: False,
            get_ocr_result_dictionary_rules=lambda: [
                {"source": "raw", "target": "dict"}
            ],
        )
        processor.cache_manager = mock.Mock()
        processor._paddleocr_cache_store = None
        processor._paddleocr_cache_identity = {"runtime": "identity"}
        block = TextBlock(
            text_bbox=np.array([10, 10, 100, 100], dtype=np.int32),
            text_class="text_bubble",
            source_lang="ja",
        )
        engine = PaddleOCRVLEngine()
        plan = SimpleNamespace(
            lookup_disabled=False,
            all_hit=True,
            hit_count=1,
            requires_runtime=False,
        )

        def restore_raw(_plan) -> None:
            block.text = "raw"
            block.ocr_status = "ok"

        engine.process_persistent_cache_plan = mock.Mock(
            side_effect=restore_raw
        )
        engine.build_persistent_cache_records = mock.Mock(return_value=[])
        engine.last_page_profile = {}
        page = StagePageContext(
            image_path="page.png",
            image_name="page.png",
            source_lang="Japanese",
            target_lang="Korean",
            image=np.zeros((120, 120, 3), dtype=np.uint8),
            blk_list=[block],
            paddleocr_cache_plan=plan,
            paddleocr_cache_engine=engine,
        )

        def apply_dictionary(blocks, _rules) -> None:
            for item in blocks:
                item.text = item.text.replace("raw", "dict")

        with mock.patch(
            "pipeline.stage_batched_processor.apply_ocr_result_dictionary",
            side_effect=apply_dictionary,
        ) as dictionary:
            result = processor._run_primary_ocr(
                page,
                {
                    "primary_ocr_engine": "PaddleOCR VL",
                    "normalized_ocr_mode": "best_local",
                },
            )

        dictionary.assert_called_once()
        self.assertEqual(block.text, "dict")
        self.assertEqual(result["cache_status"], "hit")

    def test_project_ocr_hit_applies_current_dictionary_once_without_retry(self) -> None:
        processor = self._processor(cancelled=False)
        processor.main_page.lang_mapping = {"Japanese": "Japanese"}
        processor.main_page.settings_page = SimpleNamespace(
            is_gpu_enabled=lambda: False,
            get_ocr_result_dictionary_rules=lambda: [
                {"source": "raw", "target": "dict"}
            ],
        )
        processor.cache_manager = mock.Mock()
        processor._paddleocr_cache_store = None
        processor._paddleocr_cache_identity = {"runtime": "identity"}
        processor._await_ocr_runtime = mock.Mock()
        block = TextBlock(
            text_bbox=np.array([10, 10, 100, 100], dtype=np.int32),
            text_class="text_bubble",
            source_lang="ja",
            block_id="block-a",
            text="raw",
        )
        raw_result = {
            "text": "raw",
            "texts": ["raw"],
            "confidence": 1.0,
            "status": "ok",
            "empty_reason": "",
            "attempt_count": 1,
            "raw_text": "raw",
            "sanitized_text": "raw",
            "reject_reason": "",
            "ocr_regions": [],
            "ocr_crop_bbox": None,
            "ocr_resize_scale": 1.0,
            "ocr_effective_crop_xyxy": None,
            "ocr_retry_crop_xyxy": None,
            "ocr_crop_source": "text",
        }
        checkpoint_hit = SimpleNamespace(
            raw_results={"block-a": raw_result},
            page_profile={},
            attempt_count=1,
            engine_name="PaddleOCRVLEngine",
        )
        page = StagePageContext(
            image_path="page.png",
            image_name="page.png",
            source_lang="Japanese",
            target_lang="Korean",
            image=np.zeros((120, 120, 3), dtype=np.uint8),
            blk_list=[block],
            project_ocr_hit=checkpoint_hit,
        )

        def apply_dictionary(blocks, _rules) -> None:
            for item in blocks:
                item.text = item.text.replace("raw", "dict")

        with mock.patch(
            "pipeline.stage_batched_processor.apply_ocr_result_dictionary",
            side_effect=apply_dictionary,
        ) as dictionary, mock.patch.object(
            OCRFactory,
            "create_engine",
        ) as create_engine:
            result = processor._run_primary_ocr(
                page,
                {
                    "primary_ocr_engine": "PaddleOCR VL",
                    "normalized_ocr_mode": "best_local",
                },
            )

        dictionary.assert_called_once()
        create_engine.assert_not_called()
        processor._await_ocr_runtime.assert_not_called()
        self.assertEqual(block.text, "dict")
        self.assertEqual(result["cache_status"], "hit")
        self.assertEqual(
            result["page_profile"]["project_checkpoint"]["inference_count"],
            0,
        )

    def test_paddle_folder_path_never_uses_sampled_fuzzy_memory_cache(self) -> None:
        processor = self._processor(cancelled=False)
        processor.main_page.lang_mapping = {"Japanese": "Japanese"}
        processor.main_page.settings_page = SimpleNamespace(
            is_gpu_enabled=lambda: False,
            get_ocr_result_dictionary_rules=lambda: [],
        )
        processor.cache_manager = mock.Mock()
        processor._paddleocr_cache_store = None
        processor._paddleocr_cache_identity = None
        block = TextBlock(
            text_bbox=np.array([10, 10, 100, 100], dtype=np.int32),
            text_class="text_bubble",
            source_lang="ja",
        )
        engine = PaddleOCRVLEngine()

        def process(_image, blocks) -> None:
            blocks[0].text = "raw"
            blocks[0].ocr_status = "ok"

        engine.process_image = mock.Mock(side_effect=process)
        engine.last_page_profile = {}
        page = StagePageContext(
            image_path="page.png",
            image_name="page.png",
            source_lang="Japanese",
            target_lang="Korean",
            image=np.zeros((120, 120, 3), dtype=np.uint8),
            blk_list=[block],
        )

        with mock.patch.object(OCRFactory, "create_engine", return_value=engine):
            processor._run_primary_ocr(
                page,
                {
                    "primary_ocr_engine": "PaddleOCR VL",
                    "normalized_ocr_mode": "best_local",
                },
            )

        processor.cache_manager._get_ocr_cache_key.assert_not_called()
        processor.cache_manager._can_serve_all_blocks_from_ocr_cache.assert_not_called()
        processor.cache_manager._cache_ocr_results.assert_not_called()

    def test_paddle_cache_plan_failure_is_page_scoped(self) -> None:
        processor = self._processor(cancelled=False)
        processor.main_page.lang_mapping = {"Japanese": "Japanese"}
        processor.main_page.settings_page = object()
        processor._get_paddleocr_cache_store = mock.Mock(return_value=object())
        processor._paddleocr_cache_store = None
        processor._paddleocr_cache_identity = None
        processor._mark_page_failed = mock.Mock(
            side_effect=lambda page, **_kwargs: setattr(page, "failed_stage", "ocr")
        )
        first_engine = PaddleOCRVLEngine()
        second_engine = PaddleOCRVLEngine()
        first_engine.prepare_persistent_cache = mock.Mock(
            side_effect=RuntimeError("cache-plan-failure")
        )
        second_plan = SimpleNamespace(requires_runtime=False)
        second_engine.prepare_persistent_cache = mock.Mock(
            return_value=second_plan
        )
        pages = [
            StagePageContext(
                image_path="page-1.png",
                image_name="page-1.png",
                source_lang="Japanese",
                target_lang="Korean",
                image=np.zeros((120, 120, 3), dtype=np.uint8),
                blk_list=[
                    TextBlock(
                        text_bbox=np.array([10, 10, 100, 100], dtype=np.int32)
                    )
                ],
            ),
            StagePageContext(
                image_path="page-2.png",
                image_name="page-2.png",
                source_lang="Japanese",
                target_lang="Korean",
                image=np.zeros((120, 120, 3), dtype=np.uint8),
                blk_list=[
                    TextBlock(
                        text_bbox=np.array([10, 10, 100, 100], dtype=np.int32)
                    )
                ],
            ),
        ]

        with mock.patch.object(
            OCRFactory,
            "create_engine",
            side_effect=[first_engine, second_engine],
        ):
            requires_runtime = processor._prepare_paddleocr_cache_plans(
                pages,
                {
                    "primary_ocr_engine": "PaddleOCR VL",
                    "normalized_ocr_mode": "best_local",
                },
                {"runtime_fingerprint": "runtime"},
            )

        self.assertFalse(requires_runtime)
        self.assertEqual(pages[0].failed_stage, "ocr")
        self.assertIsNone(pages[0].paddleocr_cache_plan)
        self.assertIs(pages[1].paddleocr_cache_engine, second_engine)
        self.assertIs(pages[1].paddleocr_cache_plan, second_plan)
        processor._mark_page_failed.assert_called_once()

    def test_cancelled_paddle_cache_page_drops_plan_and_engine(self) -> None:
        processor = self._processor(cancelled=False)
        runtime_manager = LocalOCRRuntimeManager()
        runtime_manager.get_ocr_cache_identity = mock.Mock(
            return_value={"runtime_fingerprint": "synthetic"}
        )
        processor.main_page.local_ocr_runtime_manager = runtime_manager
        processor.main_page.settings_page = SimpleNamespace(
            get_paddleocr_vl_settings=lambda: {
                "persistent_cache_enabled": True,
            }
        )
        processor._paddleocr_cache_identity = None
        processor._canonicalize_ocr_inputs = mock.Mock()
        processor._prepare_project_ocr_hits = mock.Mock()
        plan = SimpleNamespace(requires_runtime=True)
        engine = PaddleOCRVLEngine()

        def prepare_page(pages, *_args, **_kwargs):
            pages[0].paddleocr_cache_plan = plan
            pages[0].paddleocr_cache_engine = engine
            return True

        processor._prepare_paddleocr_cache_plans = mock.Mock(
            side_effect=prepare_page
        )
        processor._await_ocr_runtime = mock.Mock()
        processor._set_current_image = mock.Mock()
        processor.emit_progress = mock.Mock()
        processor._emit_benchmark_event = mock.Mock()
        processor._run_primary_ocr = mock.Mock(
            side_effect=OperationCancelledError("synthetic cancellation")
        )
        page = StagePageContext(
            image_path="synthetic.png",
            image_name="synthetic.png",
            source_lang="Japanese",
            target_lang="Korean",
            blk_list=[
                TextBlock(
                    text_bbox=np.array([1, 1, 20, 20], dtype=np.int32),
                    text_class="text_bubble",
                )
            ],
        )

        with self.assertRaises(OperationCancelledError):
            processor._ocr_all(
                [page],
                {
                    "primary_ocr_engine": "PaddleOCR VL",
                    "normalized_ocr_mode": "best_local",
                },
            )

        self.assertIsNone(page.paddleocr_cache_plan)
        self.assertIsNone(page.paddleocr_cache_engine)
        processor._prepare_paddleocr_cache_plans.assert_called_once()
        processor._await_ocr_runtime.assert_called_once()
        self.assertEqual(runtime_manager.get_ocr_cache_identity.call_count, 2)

    def test_project_identity_drift_after_consumed_hit_fails_closed(self) -> None:
        with TemporaryDirectory() as temporary:
            processor = self._processor(cancelled=False)
            identity_before = {"runtime_fingerprint": "before-start"}
            identity_after = {"runtime_fingerprint": "after-start"}
            runtime_manager = LocalOCRRuntimeManager()
            runtime_manager.get_ocr_cache_identity = mock.Mock(
                side_effect=[identity_before, identity_after]
            )
            processor.main_page.local_ocr_runtime_manager = runtime_manager
            settings = SimpleNamespace(
                get_paddleocr_vl_settings=lambda: {
                    "persistent_cache_enabled": True,
                    "max_new_tokens": 256,
                    "prettify_markdown": False,
                    "visualize": False,
                },
                get_ocr_result_dictionary_rules=lambda: [],
                get_tool_selection=lambda _tool: "PaddleOCR VL",
                is_gpu_enabled=lambda: False,
            )
            processor.main_page.settings_page = settings
            processor.main_page.lang_mapping = {"Japanese": "Japanese"}
            processor.main_page.image_ctrl = SimpleNamespace(
                update_processing_summary=mock.Mock(),
                mark_processing_stage=mock.Mock(),
            )
            processor.main_page.image_skipped = mock.Mock()
            project_file = Path(temporary) / "synthetic-project.ctpr"
            project_file.write_bytes(b"synthetic project")
            project_reference = checkpoint_reference_for_save(
                None,
                project_file,
            )
            project_store = ProjectCheckpointStore(
                project_file,
                project_reference,
                enabled=True,
            )
            self.assertTrue(project_store.ensure_initialized())
            processor._project_checkpoint_store = project_store
            processor._paddleocr_cache_identity = None
            processor._canonicalize_ocr_inputs = mock.Mock()
            processor._prepare_project_ocr_hits = StageBatchedProcessor._prepare_project_ocr_hits.__get__(
                processor
            )
            processor._await_ocr_runtime = mock.Mock()
            processor._set_current_image = mock.Mock()
            processor.emit_progress = mock.Mock()
            processor._emit_benchmark_event = mock.Mock()
            processor._persist_ocr_state = mock.Mock()
            processor._shutdown_runtime_with_retry = mock.Mock()
            plan = SimpleNamespace(requires_runtime=True)
            engine = PaddleOCRVLEngine()

            def prepare_page(pages, *_args, **_kwargs):
                pages[0].paddleocr_cache_plan = plan
                pages[0].paddleocr_cache_engine = engine
                return True

            processor._prepare_paddleocr_cache_plans = mock.Mock(
                side_effect=prepare_page
            )
            run_calls: list[tuple[str, bool]] = []

            def run_primary(ctx, _policy):
                run_calls.append((ctx.image_name, ctx.project_ocr_hit is not None))
                if ctx.project_ocr_hit is None:
                    raise AssertionError("OCR inference started after identity drift")
                return {
                    "quality": {"non_empty": 1, "low_quality": False},
                    "metrics": {},
                    "cache_status": "hit",
                    "attempt_count": 1,
                    "page_profile": {},
                    "engine_name": "PaddleOCRVLEngine",
                }

            processor._run_primary_ocr = run_primary
            policy = {
                "primary_ocr_engine": "PaddleOCR VL",
                "normalized_ocr_mode": "best_local",
            }
            pages = [
                StagePageContext(
                    image_path=f"synthetic-{index}.png",
                    image_name=f"synthetic-{index}.png",
                    source_lang="Japanese",
                    target_lang="Korean",
                    project_checkpoint_page_key=f"page:{index:08d}",
                    detection_fingerprint=f"detection-{index}",
                    blk_list=[
                        TextBlock(
                            text_bbox=np.array([2, 2, 24, 24], dtype=np.int32),
                            text_class="text_bubble",
                            source_lang="ja",
                        )
                    ],
                )
                for index in range(2)
            ]
            seed_block = pages[0].blk_list[0]
            seed_block_id = ensure_text_block_id(seed_block)
            seed_block.text = "cached under old identity"
            seed_block.ocr_status = "ok"
            seed_identity = build_project_ocr_identity(
                detection_fingerprint=pages[0].detection_fingerprint,
                runtime_identity=identity_before,
                policy=policy,
                paddle_settings=settings.get_paddleocr_vl_settings(),
                source_lang_english="Japanese",
            )
            self.assertTrue(
                record_ocr_checkpoint(
                    project_store,
                    page_key=pages[0].project_checkpoint_page_key,
                    fingerprint=build_project_ocr_fingerprint(seed_identity),
                    identity=seed_identity,
                    blocks=[seed_block],
                    raw_results={seed_block_id: snapshot_raw_ocr_result(seed_block)},
                    attempt_count=1,
                    engine_name="PaddleOCRVLEngine",
                    page_profile={},
                )
            )

            with mock.patch(
                "pipeline.stage_batched_processor.record_ocr_checkpoint"
            ) as record_checkpoint:
                processor._ocr_all(pages, policy)

            self.assertEqual(run_calls, [("synthetic-0.png", True)])
            self.assertTrue(all(page.failed_stage == "ocr" for page in pages))
            self.assertTrue(
                all(
                    page.failed_reason
                    == "ocr_runtime_identity_changed_after_page"
                    for page in pages
                )
            )
            self.assertEqual(record_checkpoint.call_count, 0)
            processor._await_ocr_runtime.assert_called_once()
            processor._shutdown_runtime_with_retry.assert_called_once()
            processor.main_page.image_skipped.emit.assert_has_calls(
                [
                    mock.call("synthetic-0.png", "ocr", ""),
                    mock.call("synthetic-1.png", "ocr", ""),
                ]
            )
            failed_events = [
                call
                for call in processor._emit_benchmark_event.call_args_list
                if call.args[0] == "page_failed"
            ]
            self.assertEqual(len(failed_events), 2)
            self.assertTrue(
                all(
                    call.kwargs["reason"]
                    == "ocr_runtime_identity_changed_after_page"
                    for call in failed_events
                )
            )
            self.assertTrue(
                all(page.paddleocr_cache_plan is None for page in pages)
            )
            self.assertTrue(
                all(page.paddleocr_cache_engine is None for page in pages)
            )

    def test_low_quality_all_hit_retry_rejects_runtime_identity_drift(self) -> None:
        with TemporaryDirectory() as temporary:
            identity_before = {"runtime_fingerprint": "before-start"}
            identity_after = {"runtime_fingerprint": "after-start"}
            settings = SimpleNamespace(
                get_paddleocr_vl_settings=lambda: {
                    "persistent_cache_enabled": True,
                    "parallel_workers": 1,
                    "max_new_tokens": 256,
                    "prettify_markdown": False,
                    "visualize": False,
                },
                get_ocr_generic_settings=lambda: {
                    "paddleocr_vl_scheduler_mode": "fixed",
                    "manga_expansion_percentage": 7,
                    "crop_padding_ratio": 0.05,
                },
                get_ocr_result_dictionary_rules=lambda: [],
                get_tool_selection=lambda _tool: "PaddleOCR VL",
                is_gpu_enabled=lambda: False,
            )
            image_before = np.full((120, 120, 3), 220, dtype=np.uint8)
            image_after = np.full((120, 120, 3), 128, dtype=np.uint8)
            cache_store = OCRPersistentResultCache(
                Path(temporary) / "ocr-results.sqlite3"
            )

            def make_block(block_id: str) -> TextBlock:
                block = TextBlock(
                    text_bbox=np.array([20, 20, 80, 60], dtype=np.int32),
                    text_class="text_bubble",
                    source_lang="ja",
                    direction="horizontal",
                )
                block.block_id = block_id
                return block

            seed_engine = PaddleOCRVLEngine()
            seed_engine.initialize(settings)
            seed_engine._resolve_gpu_metrics_for_scheduler = lambda: {}
            seed_block = make_block("retry-block")
            seed_plan = seed_engine.prepare_persistent_cache(
                image_before,
                [seed_block],
                cache_store,
                identity_before,
            )
            self.assertTrue(seed_plan.requires_runtime)
            self.assertEqual(len(seed_plan.runtime_jobs), 1)
            seed_engine._mark_empty(
                seed_plan.runtime_jobs[0]["block"],
                "PaddleOCR VL returned no usable text.",
            )
            seed_plan.runtime_jobs[0]["request_record"]["status"] = "empty"
            seed_plan.page_profile["page_status"] = "ok"
            seed_records = seed_engine.build_persistent_cache_records(seed_plan)
            self.assertEqual(len(seed_records), 1)
            self.assertTrue(cache_store.store_records(seed_records))

            probe_engine = PaddleOCRVLEngine()
            probe_engine.initialize(settings)
            probe_engine._resolve_gpu_metrics_for_scheduler = lambda: {}
            hit_plan = probe_engine.prepare_persistent_cache(
                image_before,
                [make_block("retry-block")],
                cache_store,
                identity_before,
            )
            self.assertTrue(hit_plan.all_hit)
            self.assertFalse(hit_plan.requires_runtime)
            probe_engine.process_persistent_cache_plan(hit_plan)
            self.assertTrue(summarize_ocr_quality(hit_plan.blocks)["low_quality"])
            self.assertFalse(all_empty_blocks_are_rejected(hit_plan.blocks))

            cache_store.store_records = mock.Mock(wraps=cache_store.store_records)
            project_file = Path(temporary) / "synthetic-project.ctpr"
            project_file.write_bytes(b"synthetic project")
            project_reference = checkpoint_reference_for_save(
                None,
                project_file,
            )
            project_store = ProjectCheckpointStore(
                project_file,
                project_reference,
                enabled=True,
            )
            self.assertTrue(project_store.ensure_initialized())
            processor = self._processor(cancelled=False)
            processor.main_page.settings_page = settings
            processor.main_page.lang_mapping = {"Japanese": "Japanese"}
            processor.main_page.image_ctrl = SimpleNamespace(
                load_image=lambda path: (
                    image_before if path == "page-before.png" else image_after
                ).copy(),
                update_processing_summary=mock.Mock(),
                mark_processing_stage=mock.Mock(),
            )
            processor.main_page.image_skipped = mock.Mock()
            processor._project_checkpoint_store = project_store
            processor._paddleocr_cache_store = cache_store
            processor._paddleocr_cache_identity = None
            processor._get_paddleocr_cache_store = lambda: cache_store
            runtime_manager = LocalOCRRuntimeManager()
            runtime_manager.get_ocr_cache_identity = mock.Mock(
                side_effect=[identity_before, identity_after, identity_after]
            )
            runtime_manager.ensure_engine = mock.Mock()
            processor.main_page.local_ocr_runtime_manager = runtime_manager
            processor._set_current_image = mock.Mock()
            processor.emit_progress = mock.Mock()
            processor._emit_benchmark_event = mock.Mock()
            processor._persist_ocr_state = mock.Mock()
            processor._shutdown_runtime_with_retry = mock.Mock()
            processor._capture_runtime_gpu_start_baseline = mock.Mock()
            processor._record_runtime_performance = mock.Mock()
            processor._sample_performance_resources = mock.Mock()
            fake_arbiter = SimpleNamespace(
                token=lambda service: service,
                model_start=lambda *_args, **_kwargs: nullcontext(),
            )
            processor._runtime_resource_arbiter = lambda: fake_arbiter
            processor._prewarm_jobs = {}
            await_calls: list[str] = []
            actual_await = StageBatchedProcessor._await_ocr_runtime.__get__(
                processor
            )

            def await_runtime(current_policy):
                await_calls.append("ocr")
                return actual_await(current_policy)

            processor._await_ocr_runtime = await_runtime
            pages = [
                StagePageContext(
                    image_path="page-before.png",
                    image_name="page-before.png",
                    source_lang="Japanese",
                    target_lang="Korean",
                    image=image_before.copy(),
                    source_decoded_sha256=decoded_image_sha256(image_before),
                    source_pixel_count=120 * 120,
                    project_checkpoint_page_key="page:00000000",
                    detection_fingerprint="detection-before",
                    blk_list=[make_block("retry-block")],
                ),
                StagePageContext(
                    image_path="page-after.png",
                    image_name="page-after.png",
                    source_lang="Japanese",
                    target_lang="Korean",
                    image=image_after.copy(),
                    source_decoded_sha256=decoded_image_sha256(image_after),
                    source_pixel_count=120 * 120,
                    project_checkpoint_page_key="page:00000001",
                    detection_fingerprint="detection-after",
                    blk_list=[make_block("later-block")],
                ),
            ]
            prepare_identities: list[tuple[str, bool]] = []
            process_plans: list[tuple[bool, bool]] = []
            request_calls: list[mock.Mock] = []

            def create_engine(*_args, **_kwargs):
                engine = PaddleOCRVLEngine()
                engine.initialize(settings)
                engine._resolve_gpu_metrics_for_scheduler = lambda: {}
                prepare_page = engine.prepare_persistent_cache

                def prepare(image, blocks, store, runtime_identity, **kwargs):
                    prepare_identities.append(
                        (
                            str(runtime_identity["runtime_fingerprint"]),
                            bool(kwargs.get("lookup", True)),
                        )
                    )
                    return prepare_page(
                        image,
                        blocks,
                        store,
                        runtime_identity,
                        **kwargs,
                    )

                engine.prepare_persistent_cache = prepare
                process_plan = engine.process_persistent_cache_plan

                def process(plan):
                    process_plans.append((plan.all_hit, plan.requires_runtime))
                    return process_plan(plan)

                engine.process_persistent_cache_plan = process
                request = mock.Mock(return_value="synthetic retry OCR")
                engine._request_ocr_text_from_encoded = request
                request_calls.append(request)
                return engine

            real_record = record_ocr_checkpoint
            project_records: list[str] = []

            def record_project(store, **kwargs):
                project_records.append(
                    str(kwargs["identity"]["runtime"]["runtime_fingerprint"])
                )
                return real_record(store, **kwargs)

            policy = {
                "primary_ocr_engine": "PaddleOCR VL",
                "normalized_ocr_mode": "best_local",
            }
            try:
                with mock.patch.object(
                    OCRFactory,
                    "create_engine",
                    side_effect=create_engine,
                ), mock.patch(
                    "pipeline.stage_batched_processor.record_ocr_checkpoint",
                    side_effect=record_project,
                ):
                    processor._ocr_all(pages, policy)
            finally:
                cache_stats = cache_store.stats()
                cache_store.close()

            self.assertEqual(process_plans, [(True, False)])
            self.assertEqual(prepare_identities, [("before-start", True)])
            self.assertEqual(await_calls, ["ocr"])
            self.assertEqual(runtime_manager.ensure_engine.call_count, 1)
            self.assertEqual(runtime_manager.get_ocr_cache_identity.call_count, 2)
            self.assertTrue(all(not request.called for request in request_calls))
            cache_store.store_records.assert_not_called()
            self.assertEqual(project_records, [])
            self.assertEqual(cache_stats["item_count"], 1)
            self.assertTrue(all(page.failed_stage == "ocr" for page in pages))
            self.assertEqual(
                [page.failed_reason for page in pages],
                [
                    "ocr_runtime_identity_changed_after_page",
                    "ocr_runtime_identity_changed_after_page",
                ],
            )
            processor._shutdown_runtime_with_retry.assert_called_once()

    def test_low_quality_all_hit_retry_preserves_same_runtime_identity(self) -> None:
        with TemporaryDirectory() as temporary:
            identity = {"runtime_fingerprint": "same-runtime"}
            settings = SimpleNamespace(
                get_paddleocr_vl_settings=lambda: {
                    "persistent_cache_enabled": True,
                    "parallel_workers": 1,
                    "max_new_tokens": 256,
                    "prettify_markdown": False,
                    "visualize": False,
                },
                get_ocr_generic_settings=lambda: {
                    "paddleocr_vl_scheduler_mode": "fixed",
                    "manga_expansion_percentage": 7,
                    "crop_padding_ratio": 0.05,
                },
                get_ocr_result_dictionary_rules=lambda: [
                    {"keyword": "synthetic", "sub": "fresh"}
                ],
                get_tool_selection=lambda _tool: "PaddleOCR VL",
                is_gpu_enabled=lambda: False,
            )
            image = np.full((120, 120, 3), 220, dtype=np.uint8)
            cache_store = OCRPersistentResultCache(
                Path(temporary) / "ocr-results.sqlite3"
            )

            def make_block() -> TextBlock:
                block = TextBlock(
                    text_bbox=np.array([20, 20, 80, 60], dtype=np.int32),
                    text_class="text_bubble",
                    source_lang="ja",
                    direction="horizontal",
                )
                block.block_id = "retry-block"
                return block

            seed_engine = PaddleOCRVLEngine()
            seed_engine.initialize(settings)
            seed_engine._resolve_gpu_metrics_for_scheduler = lambda: {}
            seed_plan = seed_engine.prepare_persistent_cache(
                image,
                [make_block()],
                cache_store,
                identity,
            )
            self.assertTrue(seed_plan.requires_runtime)
            seed_engine._mark_empty(
                seed_plan.runtime_jobs[0]["block"],
                "PaddleOCR VL returned no usable text.",
            )
            seed_plan.runtime_jobs[0]["request_record"]["status"] = "empty"
            seed_plan.page_profile["page_status"] = "ok"
            self.assertTrue(
                cache_store.store_records(
                    seed_engine.build_persistent_cache_records(seed_plan)
                )
            )
            cache_store.store_records = mock.Mock(wraps=cache_store.store_records)

            project_file = Path(temporary) / "synthetic-project.ctpr"
            project_file.write_bytes(b"synthetic project")
            project_reference = checkpoint_reference_for_save(None, project_file)
            project_store = ProjectCheckpointStore(
                project_file,
                project_reference,
                enabled=True,
            )
            self.assertTrue(project_store.ensure_initialized())

            processor = self._processor(cancelled=False)
            processor.main_page.settings_page = settings
            processor.main_page.lang_mapping = {"Japanese": "Japanese"}
            processor.main_page.image_ctrl = SimpleNamespace(
                load_image=lambda _path: image.copy(),
                update_processing_summary=mock.Mock(),
                mark_processing_stage=mock.Mock(),
            )
            processor.main_page.image_skipped = mock.Mock()
            processor._project_checkpoint_store = project_store
            processor._get_paddleocr_cache_store = lambda: cache_store
            runtime_manager = LocalOCRRuntimeManager()
            runtime_manager.get_ocr_cache_identity = mock.Mock(
                side_effect=[identity, identity]
            )
            runtime_manager.ensure_engine = mock.Mock()
            processor.main_page.local_ocr_runtime_manager = runtime_manager
            processor._canonicalize_ocr_inputs = mock.Mock()
            processor._set_current_image = mock.Mock()
            processor.emit_progress = mock.Mock()
            processor._emit_benchmark_event = mock.Mock()
            processor._log_ocr_quality = mock.Mock()
            processor._persist_ocr_state = mock.Mock()
            processor._shutdown_runtime_with_retry = mock.Mock()
            processor._capture_runtime_gpu_start_baseline = mock.Mock()
            processor._record_runtime_performance = mock.Mock()
            processor._sample_performance_resources = mock.Mock()
            processor._prewarm_jobs = {}
            fake_arbiter = SimpleNamespace(
                token=lambda service: service,
                model_start=lambda *_args, **_kwargs: nullcontext(),
            )
            processor._runtime_resource_arbiter = lambda: fake_arbiter
            await_calls: list[str] = []
            actual_await = StageBatchedProcessor._await_ocr_runtime.__get__(
                processor
            )

            def await_runtime(current_policy):
                await_calls.append("ocr")
                return actual_await(current_policy)

            processor._await_ocr_runtime = await_runtime
            page = StagePageContext(
                image_path="synthetic.png",
                image_name="synthetic.png",
                source_lang="Japanese",
                target_lang="Korean",
                image=image.copy(),
                source_decoded_sha256=decoded_image_sha256(image),
                source_pixel_count=120 * 120,
                project_checkpoint_page_key="page:00000001",
                detection_fingerprint="synthetic-detection",
                blk_list=[make_block()],
            )
            engines: list[PaddleOCRVLEngine] = []
            requests: list[mock.Mock] = []

            def create_engine(*_args, **_kwargs):
                engine = PaddleOCRVLEngine()
                engine.initialize(settings)
                engine._resolve_gpu_metrics_for_scheduler = lambda: {}
                request = mock.Mock(return_value="synthetic retry OCR")
                engine._request_ocr_text_from_encoded = request
                engines.append(engine)
                requests.append(request)
                return engine

            policy = {
                "primary_ocr_engine": "PaddleOCR VL",
                "normalized_ocr_mode": "best_local",
            }
            try:
                with mock.patch.object(
                    OCRFactory,
                    "create_engine",
                    side_effect=create_engine,
                ), mock.patch(
                    "pipeline.stage_batched_processor.record_ocr_checkpoint",
                    wraps=record_ocr_checkpoint,
                ) as record_checkpoint, mock.patch(
                    "pipeline.stage_batched_processor.apply_ocr_result_dictionary",
                    wraps=apply_ocr_result_dictionary,
                ) as apply_dictionary:
                    processor._ocr_all([page], policy)
            finally:
                cache_store.close()

            self.assertFalse(page.failed_stage)
            self.assertEqual(page.blk_list[0].text, "fresh retry OCR")
            self.assertEqual(ensure_text_block_id(page.blk_list[0]), "retry-block")
            self.assertEqual(await_calls, ["ocr"])
            runtime_manager.ensure_engine.assert_called_once()
            self.assertEqual(runtime_manager.get_ocr_cache_identity.call_count, 2)
            self.assertEqual(len(engines), 1)
            requests[0].assert_called_once()
            cache_store.store_records.assert_called_once()
            self.assertEqual(apply_dictionary.call_count, 2)
            self.assertEqual(processor._persist_ocr_state.call_args.args[-1], 2)
            self.assertEqual(record_checkpoint.call_count, 1)
            self.assertEqual(
                record_checkpoint.call_args.kwargs["identity"]["runtime"],
                identity,
            )
            self.assertTrue(project_store.has_stage_record("page:00000001", "ocr"))

    def test_paddle_cache_hit_and_guard_only_miss_skip_runtime_start(self) -> None:
        for plan in (
            SimpleNamespace(
                requires_runtime=False,
                all_hit=True,
                hit_count=1,
                miss_count=0,
                lookup_disabled=False,
            ),
            SimpleNamespace(
                requires_runtime=False,
                all_hit=False,
                hit_count=0,
                miss_count=1,
                lookup_disabled=False,
            ),
        ):
            with self.subTest(plan=plan):
                processor = self._processor(cancelled=False)
                runtime_manager = LocalOCRRuntimeManager()
                runtime_manager.get_ocr_cache_identity = mock.Mock(
                    return_value={"runtime_fingerprint": "synthetic"}
                )
                processor.main_page.local_ocr_runtime_manager = runtime_manager
                processor.main_page.settings_page = SimpleNamespace(
                    get_paddleocr_vl_settings=lambda: {
                        "persistent_cache_enabled": True,
                    },
                    is_gpu_enabled=lambda: False,
                    get_tool_selection=lambda _tool: "PaddleOCR VL",
                )
                processor.main_page.lang_mapping = {"Japanese": "Japanese"}
                processor.main_page.image_ctrl = SimpleNamespace(
                    load_image=lambda _path: np.zeros(
                        (32, 32, 3), dtype=np.uint8
                    )
                )
                processor._paddleocr_cache_identity = None
                processor._paddleocr_cache_store = None
                processor._canonicalize_ocr_inputs = mock.Mock()
                processor._prepare_project_ocr_hits = mock.Mock()
                processor._get_paddleocr_cache_store = mock.Mock(
                    return_value=object()
                )
                processor._await_ocr_runtime = mock.Mock()
                processor._set_current_image = mock.Mock()
                processor.emit_progress = mock.Mock()
                processor._emit_benchmark_event = mock.Mock()
                processor._log_ocr_quality = mock.Mock()
                processor._persist_ocr_state = mock.Mock()
                processor._shutdown_runtime_with_retry = mock.Mock()
                processor._run_primary_ocr = mock.Mock(
                    return_value={
                        "quality": {"non_empty": 1, "low_quality": False},
                        "metrics": {},
                        "cache_status": "hit",
                        "attempt_count": 1,
                        "page_profile": {},
                        "engine_name": "PaddleOCRVLEngine",
                    }
                )
                engine = PaddleOCRVLEngine()
                engine.prepare_persistent_cache = mock.Mock(return_value=plan)
                page = StagePageContext(
                    image_path="synthetic.png",
                    image_name="synthetic.png",
                    source_lang="Japanese",
                    target_lang="Korean",
                    image=np.zeros((32, 32, 3), dtype=np.uint8),
                    blk_list=[
                        TextBlock(
                            text_bbox=np.array([2, 2, 24, 24], dtype=np.int32)
                        )
                    ],
                )

                with mock.patch.object(
                    OCRFactory,
                    "create_engine",
                    return_value=engine,
                ):
                    processor._ocr_all(
                        [page],
                        {
                            "primary_ocr_engine": "PaddleOCR VL",
                            "normalized_ocr_mode": "best_local",
                        },
                    )

                processor._await_ocr_runtime.assert_not_called()
                processor._run_primary_ocr.assert_called_once()
                self.assertIsNone(page.paddleocr_cache_plan)
                self.assertIsNone(page.paddleocr_cache_engine)

    def test_batch_cleanup_stops_both_managed_runtimes_after_cancel(self) -> None:
        processor = self._processor(cancelled=True)
        ocr_manager = LocalOCRRuntimeManager()
        gemma_manager = LocalGemmaRuntimeManager()
        processor.main_page.local_ocr_runtime_manager = ocr_manager
        processor.main_page.local_translation_runtime_manager = gemma_manager

        with mock.patch.object(ocr_manager, "shutdown") as shutdown_ocr, \
             mock.patch.object(gemma_manager, "shutdown") as shutdown_gemma:
            processor._shutdown_managed_runtimes()

        shutdown_ocr.assert_called_once_with()
        shutdown_gemma.assert_called_once_with()

    def test_normal_cleanup_preserves_only_sleeping_paddle_runtime(self) -> None:
        processor = self._processor(cancelled=False)
        ocr_manager = LocalOCRRuntimeManager()
        gemma_manager = LocalGemmaRuntimeManager()
        processor.main_page.local_ocr_runtime_manager = ocr_manager
        processor.main_page.local_translation_runtime_manager = gemma_manager

        with mock.patch.object(
            ocr_manager,
            "release_for_handoff",
        ) as release_ocr, mock.patch.object(
            ocr_manager,
            "shutdown",
        ) as shutdown_ocr, mock.patch.object(
            gemma_manager,
            "shutdown",
        ) as shutdown_gemma:
            processor._shutdown_managed_runtimes(
                preserve_sleeping_paddle=True,
            )

        release_ocr.assert_called_once_with()
        shutdown_ocr.assert_not_called()
        shutdown_gemma.assert_called_once_with()

    def test_batch_cleanup_still_stops_gemma_when_ocr_shutdown_fails(self) -> None:
        processor = self._processor(cancelled=True)
        ocr_manager = LocalOCRRuntimeManager()
        gemma_manager = LocalGemmaRuntimeManager()
        processor.main_page.local_ocr_runtime_manager = ocr_manager
        processor.main_page.local_translation_runtime_manager = gemma_manager

        with mock.patch.object(
            ocr_manager,
            "shutdown",
            side_effect=RuntimeError("stop failed"),
        ) as shutdown_ocr, \
             mock.patch.object(gemma_manager, "shutdown") as shutdown_gemma, \
             self.assertLogs("pipeline.stage_batched_processor", level="WARNING"):
            processor._shutdown_managed_runtimes()

        self.assertEqual(shutdown_ocr.call_count, 2)
        shutdown_gemma.assert_called_once_with()

    def test_batch_startup_preflight_stops_both_runtimes_then_fails_closed(self) -> None:
        processor = self._processor(cancelled=False)
        ocr_manager = LocalOCRRuntimeManager()
        gemma_manager = LocalGemmaRuntimeManager()
        processor.main_page.local_ocr_runtime_manager = ocr_manager
        processor.main_page.local_translation_runtime_manager = gemma_manager

        with mock.patch.object(
            ocr_manager,
            "shutdown",
            side_effect=RuntimeError("retained OCR runtime"),
        ) as shutdown_ocr, mock.patch.object(
            gemma_manager,
            "shutdown",
        ) as shutdown_gemma, self.assertLogs(
            "pipeline.stage_batched_processor",
            level="WARNING",
        ), self.assertRaisesRegex(RuntimeError, "retained OCR runtime"):
            processor._shutdown_managed_runtimes(
                context="batch startup preflight",
                raise_on_failure=True,
            )

        self.assertEqual(shutdown_ocr.call_count, 2)
        shutdown_gemma.assert_called_once_with()

    def test_cleanup_uses_actual_paddle_service_key(self) -> None:
        processor = self._processor(cancelled=False)
        ocr_manager = LocalOCRRuntimeManager()
        processor.main_page.local_ocr_runtime_manager = ocr_manager

        arbiter = processor._runtime_resource_arbiter()
        with arbiter.model_start(arbiter.token("paddleocr_vl")):
            pass

        with mock.patch.object(ocr_manager, "shutdown") as shutdown_ocr:
            processor._shutdown_managed_runtimes(
                ocr_service="paddleocr_vl",
            )

        shutdown_ocr.assert_called_once_with()
        self.assertIsNone(arbiter.snapshot().active_model)

    def test_batch_ocr_stage_ceiling_skips_later_stages(self) -> None:
        processor = self._processor(cancelled=False)
        processor.main_page.image_files = ["page.png"]
        processor.main_page.file_handler = SimpleNamespace(
            should_pre_materialize=lambda _paths: False,
        )
        processor.main_page.reset_automatic_output_reservations = mock.Mock()
        processor._recent_page_durations = []
        pages = [
            StagePageContext(
                image_path="page.png",
                image_name="page.png",
                source_lang="Japanese",
                target_lang="Korean",
            )
        ]
        processor._emit_benchmark_event = mock.Mock()
        processor._reset_prewarm_lifecycle = mock.Mock()
        processor._load_page_contexts = mock.Mock(return_value=pages)
        processor._ensure_stage_policy = mock.Mock(return_value={"primary_ocr_engine": "PaddleOCR VL"})
        processor._raise_if_cancelled = mock.Mock()
        shutdown_calls = []

        def shutdown_managed_runtimes(
            *,
            context: str,
            raise_on_failure: bool = False,
            preserve_sleeping_paddle: bool = False,
            ocr_service: str = "ocr",
        ) -> None:
            shutdown_calls.append(context)

        processor._shutdown_managed_runtimes = shutdown_managed_runtimes
        processor._persist_stage_rates = lambda: None
        processor._shutdown_page_cache_executor = lambda: None
        processor._shutdown_render_executor = lambda *, raise_on_failure=False: None
        processor._release_page_buffers = lambda _ctx: 0
        processor._write_run_report = mock.Mock()
        stage_order: list[str] = []
        processor._start_ocr_prewarm = mock.Mock(
            side_effect=lambda _policy: stage_order.append("ocr_prewarm")
        )
        processor._detect_all = mock.Mock(
            side_effect=lambda _pages, _policy: stage_order.append("detect")
        )
        processor._ocr_all = mock.Mock(
            side_effect=lambda _pages, _policy: stage_order.append("ocr")
        )
        processor._complete_ocr_stage_ceiling = mock.Mock()
        processor._inpaint_all = mock.Mock()
        processor._translate_all = mock.Mock()
        processor._render_all = mock.Mock()
        processor._shutdown_prewarm_executor = mock.Mock()

        with mock.patch.dict(os.environ, {"CT_BENCH_STAGE_CEILING": "ocr"}):
            processor.batch_process()

        processor._detect_all.assert_called_once_with(
            pages,
            {"primary_ocr_engine": "PaddleOCR VL"},
        )
        processor._ocr_all.assert_called_once_with(
            pages,
            {"primary_ocr_engine": "PaddleOCR VL"},
        )
        processor._complete_ocr_stage_ceiling.assert_called_once_with(pages)
        processor._write_run_report.assert_called_once()
        self.assertEqual(
            processor._write_run_report.call_args.kwargs["processing_status"],
            "completed",
        )
        processor._inpaint_all.assert_not_called()
        processor._translate_all.assert_not_called()
        processor._render_all.assert_not_called()
        processor._emit_benchmark_event.assert_any_call(
            "batch_run_done",
            total_images=1,
            stage_ceiling="ocr",
        )
        self.assertEqual(shutdown_calls, ["batch startup preflight", "terminal cleanup"])
        self.assertEqual(stage_order, ["detect", "ocr_prewarm", "ocr"])

    def test_complete_ocr_stage_ceiling_marks_only_successful_pages_done(self) -> None:
        processor = self._processor(cancelled=False)
        processor.main_page.image_ctrl = SimpleNamespace(
            update_processing_summary=mock.Mock(),
        )
        processor._ensure_page_state = mock.Mock(return_value={})
        processor._emit_benchmark_event = mock.Mock()
        processor._log_page_done = mock.Mock()
        processor.emit_progress = mock.Mock()
        successful = StagePageContext(
            image_path="success.png",
            image_name="success.png",
            source_lang="Japanese",
            target_lang="Korean",
            blk_list=[object()],
            page_ocr_metrics={"ocr_non_empty_block_count": 1},
        )
        failed = StagePageContext(
            image_path="failed.png",
            image_name="failed.png",
            source_lang="Japanese",
            target_lang="Korean",
            failed_stage="ocr",
        )

        processor._complete_ocr_stage_ceiling([successful, failed])

        processor.main_page.image_ctrl.update_processing_summary.assert_called_once_with(
            "success.png",
            {"benchmark_stage_ceiling": "ocr"},
        )
        processor._emit_benchmark_event.assert_called_once_with(
            "page_done",
            image_path="success.png",
            image_index=0,
            total_images=2,
            block_count=1,
            patch_count=0,
            stage_ceiling="ocr",
            ocr_non_empty_block_count=1,
        )
        processor._log_page_done.assert_called_once_with(0, 2, "success.png")
        processor.emit_progress.assert_called_once_with(
            0, 2, 10, 10, False, stage_name='save-and-finish'
        )

    def test_stage_transition_continues_after_verified_stop_retry(self) -> None:
        processor = self._processor(cancelled=False)
        for label, manager, next_stage in (
            ("OCR", LocalOCRRuntimeManager(), "inpaint"),
            ("Gemma", LocalGemmaRuntimeManager(), "render"),
        ):
            reached: list[str] = []
            release_method = (
                "release_for_handoff" if label == "OCR" else "shutdown"
            )
            with self.subTest(label=label), mock.patch.object(
                manager,
                release_method,
                side_effect=[RuntimeError("first stop failed"), None],
            ) as release, self.assertLogs(
                "pipeline.stage_batched_processor",
                level="WARNING",
            ):
                processor._shutdown_runtime_with_retry(
                    label,
                    manager,
                    context=f"before {next_stage}",
                    raise_on_failure=True,
                    release_for_handoff=(label == "OCR"),
                )
                reached.append(next_stage)

            self.assertEqual(release.call_count, 2)
            self.assertEqual(reached, [next_stage])

    def test_stage_transition_fails_closed_after_two_stop_failures(self) -> None:
        processor = self._processor(cancelled=False)
        manager = LocalGemmaRuntimeManager()

        with mock.patch.object(
            manager,
            "shutdown",
            side_effect=RuntimeError("stop failed"),
        ) as shutdown, self.assertLogs(
            "pipeline.stage_batched_processor",
            level="WARNING",
        ), self.assertRaisesRegex(RuntimeError, "stop failed"):
            processor._shutdown_runtime_with_retry(
                "Gemma",
                manager,
                context="translation-to-render handoff",
                raise_on_failure=True,
            )

        self.assertEqual(shutdown.call_count, 2)

    def test_handoff_without_release_api_records_stopped_state(self) -> None:
        processor = self._processor(cancelled=False)
        processor._record_runtime_transition = mock.Mock()
        processor._record_runtime_performance = mock.Mock()
        processor._sample_performance_resources = mock.Mock()
        manager = SimpleNamespace(shutdown=mock.Mock())

        processor._shutdown_runtime_with_retry(
            "OCR",
            manager,
            context="test handoff",
            raise_on_failure=True,
            release_for_handoff=True,
            service="paddleocr_vl",
        )

        manager.shutdown.assert_called_once_with()
        self.assertEqual(
            processor._record_runtime_transition.call_args_list[-1].kwargs[
                "to_state"
            ],
            "stopped",
        )

    def test_handoff_stop_report_records_stopped_state(self) -> None:
        processor = self._processor(cancelled=False)
        processor._record_runtime_transition = mock.Mock()
        processor._record_runtime_performance = mock.Mock()
        processor._sample_performance_resources = mock.Mock()
        manager = SimpleNamespace(
            release_for_handoff=mock.Mock(
                return_value={"runtime_state": "stopped"}
            )
        )

        processor._shutdown_runtime_with_retry(
            "OCR",
            manager,
            context="test handoff",
            raise_on_failure=True,
            release_for_handoff=True,
            service="paddleocr_vl",
        )

        self.assertEqual(
            processor._record_runtime_transition.call_args_list[-1].kwargs[
                "to_state"
            ],
            "stopped",
        )

    def test_unobserved_managed_runtime_release_keeps_gpu_lease(self) -> None:
        # 미확인 해제를 실패로 처리하는 계약이 대상이므로 진단용 강제를 켠다.
        patcher = mock.patch(
            "pipeline.stage_batched_processor.gpu_release_enforcement_enabled",
            return_value=True,
        )
        patcher.start()
        self.addCleanup(patcher.stop)
        processor = self._processor(cancelled=False)
        processor._record_runtime_transition = mock.Mock()
        processor._record_runtime_performance = mock.Mock()
        processor._sample_performance_resources = mock.Mock()
        manager = SimpleNamespace(
            shutdown=mock.Mock(
                return_value={
                    "runtime_state": "stopped",
                    "gpu_release_expected": True,
                }
            )
        )
        metrics = {"driver": {"available": True}}
        failed_gate = {
            "required": True,
            "observed": False,
            "status": "timeout",
            "elapsed_sec": 0.0,
        }

        with mock.patch(
            "pipeline.stage_batched_processor.query_cuda_handoff_metrics",
            return_value=metrics,
        ), mock.patch(
            "pipeline.stage_batched_processor.wait_for_global_vram_release",
            return_value=failed_gate,
        ), self.assertLogs(
            "pipeline.stage_batched_processor",
            level="WARNING",
        ), self.assertRaisesRegex(
            RuntimeError,
            "GPU release was not confirmed",
        ):
            processor._shutdown_runtime_with_retry(
                "Gemma",
                manager,
                context="test release gate",
                raise_on_failure=True,
                service="gemma",
            )

        self.assertEqual(manager.shutdown.call_count, 2)
        snapshot = processor._runtime_resource_arbiter().snapshot()
        self.assertEqual(snapshot.active_model, "gemma")
        self.assertEqual(snapshot.states["gemma"], "release_failed")
        with self.assertRaises(RuntimeLeaseConflictError):
            with processor._runtime_resource_arbiter().model_start(
                processor._runtime_resource_arbiter().token("ocr")
            ):
                pass

    def test_observed_managed_runtime_release_returns_gpu_lease(self) -> None:
        processor = self._processor(cancelled=False)
        processor._record_runtime_transition = mock.Mock()
        processor._record_runtime_performance = mock.Mock()
        processor._sample_performance_resources = mock.Mock()
        manager = SimpleNamespace(
            shutdown=mock.Mock(
                return_value={
                    "runtime_state": "stopped",
                    "gpu_release_expected": True,
                }
            )
        )
        observed_gate = {
            "required": True,
            "observed": True,
            "status": "observed",
            "elapsed_sec": 0.01,
        }

        with mock.patch(
            "pipeline.stage_batched_processor.query_cuda_handoff_metrics",
            return_value={"driver": {"available": True}},
        ), mock.patch(
            "pipeline.stage_batched_processor.wait_for_global_vram_release",
            return_value=observed_gate,
        ):
            processor._shutdown_runtime_with_retry(
                "Gemma",
                manager,
                context="test release gate",
                raise_on_failure=True,
                service="gemma",
            )

        snapshot = processor._runtime_resource_arbiter().snapshot()
        self.assertIsNone(snapshot.active_model)
        self.assertEqual(snapshot.states["gemma"], "stopped")

    def test_cancelled_start_with_unobserved_cleanup_keeps_gpu_lease(self) -> None:
        # 이 테스트들은 미확인 해제를 실패로 처리하는 계약 자체가 대상이다. 그
        # 처리는 진단용 강제를 켰을 때만 일어나므로 여기서 명시적으로 켠다.
        patcher = mock.patch(
            "pipeline.stage_batched_processor.gpu_release_enforcement_enabled",
            return_value=True,
        )
        patcher.start()
        self.addCleanup(patcher.stop)
        processor = self._processor(cancelled=False)
        processor._record_runtime_transition = mock.Mock()
        processor._record_runtime_performance = mock.Mock()
        processor._sample_performance_resources = mock.Mock()
        manager = SimpleNamespace(
            shutdown=mock.Mock(
                return_value={
                    "runtime_state": "stopped",
                    "gpu_release_expected": True,
                }
            )
        )
        arbiter = processor._runtime_resource_arbiter()
        token = arbiter.token("gemma")
        failed_gate = {
            "required": True,
            "observed": False,
            "status": "timeout",
            "elapsed_sec": 0.0,
        }

        with mock.patch(
            "pipeline.stage_batched_processor.query_cuda_handoff_metrics",
            return_value={"driver": {"available": True}},
        ), mock.patch(
            "pipeline.stage_batched_processor.wait_for_global_vram_release",
            return_value=failed_gate,
        ), self.assertRaisesRegex(RuntimeError, "GPU release was not confirmed"):
            with arbiter.model_start(
                token,
                stale_cleanup=lambda: processor._managed_runtime_stale_cleanup(
                    "gemma",
                    manager,
                ),
            ):
                arbiter.cancel_generation()

        snapshot = arbiter.snapshot()
        self.assertEqual(snapshot.active_model, "gemma")
        self.assertEqual(snapshot.states["gemma"], "release_failed")

    def test_prewarm_shutdown_waits_and_cancels_queued_late_start(self) -> None:
        processor = self._processor(cancelled=False)
        processor._prewarm_cancel_event = threading.Event()
        processor._prewarm_executor = ThreadPoolExecutor(max_workers=1)
        processor._prewarm_jobs = {}
        blocker_started = threading.Event()
        release_blocker = threading.Event()
        late_start_called = threading.Event()

        def blocker() -> None:
            blocker_started.set()
            release_blocker.wait(timeout=5.0)

        def late_start() -> None:
            late_start_called.set()

        blocker_job = processor._prewarm_executor.submit(blocker)
        late_job = processor._prewarm_executor.submit(late_start)
        processor._prewarm_jobs = {
            "ocr": blocker_job,
            "gemma": late_job,
        }
        self.assertTrue(blocker_started.wait(timeout=2.0))

        shutdown_done = threading.Event()

        def shutdown() -> None:
            processor._shutdown_prewarm_executor()
            shutdown_done.set()

        shutdown_thread = threading.Thread(target=shutdown)
        shutdown_thread.start()
        deadline = time.monotonic() + 2.0
        while not processor._prewarm_cancel_event.is_set() and time.monotonic() < deadline:
            time.sleep(0.01)

        self.assertTrue(processor._prewarm_cancel_event.is_set())
        self.assertFalse(shutdown_done.is_set())
        release_blocker.set()
        shutdown_thread.join(timeout=2.0)

        self.assertTrue(shutdown_done.is_set())
        self.assertFalse(late_start_called.is_set())
        self.assertTrue(late_job.cancelled())
        self.assertEqual(processor._prewarm_jobs, {})

    def test_running_prewarm_cannot_report_ready_after_shutdown(self) -> None:
        processor = self._processor(cancelled=False)
        processor._prewarm_cancel_event = threading.Event()
        processor._prewarm_executor = None
        processor._prewarm_jobs = {}
        progress_statuses: list[str] = []
        processor._prewarm_progress = lambda **payload: progress_statuses.append(
            str(payload.get("status") or "")
        )
        started = threading.Event()
        release = threading.Event()
        cleanup_calls: list[str] = []

        def prewarm() -> None:
            started.set()
            release.wait(timeout=5.0)

        processor._start_prewarm(
            "gemma",
            "Gemma",
            "gemma",
            prewarm,
            stale_cleanup=lambda: cleanup_calls.append("cleanup"),
        )
        self.assertTrue(started.wait(timeout=2.0))
        shutdown_thread = threading.Thread(target=processor._shutdown_prewarm_executor)
        shutdown_thread.start()
        deadline = time.monotonic() + 2.0
        while not processor._prewarm_cancel_event.is_set() and time.monotonic() < deadline:
            time.sleep(0.01)
        release.set()
        shutdown_thread.join(timeout=2.0)

        self.assertFalse(shutdown_thread.is_alive())
        self.assertEqual(progress_statuses, ["starting"])
        self.assertEqual(cleanup_calls, ["cleanup"])
        self.assertIsNone(
            processor._runtime_resource_arbiter().snapshot().active_model
        )

    def test_all_hit_folder_skips_gemma_prewarm_and_readiness_wait(self) -> None:
        processor = self._processor(cancelled=False)
        processor.main_page.settings_page = SimpleNamespace(
            get_llm_settings=lambda: {"extra_context": "context"},
            get_tool_selection=lambda _tool: "Custom Local Server(Gemma)",
            get_translation_result_dictionary_rules=lambda: [],
        )
        processor._inpainter_release_gate = {
            "required": True,
            "observed": False,
        }
        processor._start_gemma_prewarm = mock.Mock()
        processor._await_gemma_runtime = mock.Mock()
        processor._set_current_image = mock.Mock()
        processor.emit_progress = mock.Mock()
        processor._report_runtime_progress = mock.Mock()
        processor._emit_benchmark_event = mock.Mock()
        processor._persist_translation_state = mock.Mock()
        pages = [
            StagePageContext(
                image_path=f"page-{index}.png",
                image_name=f"page-{index}.png",
                source_lang="Japanese",
                target_lang="Korean",
                image=object(),
                blk_list=[SimpleNamespace(text=f"source-{index}", translation="")],
            )
            for index in range(2)
        ]

        class _AllHitTranslator:
            def __init__(self, *_args, **_kwargs) -> None:
                self.engine = SimpleNamespace(last_benchmark_stats={})
                self.translation_cache_status = "persistent-hit"
                self.uses_persistent_translation_memory = True

            @staticmethod
            def prepare_translation(_blocks, _extra_context) -> bool:
                return False

            @staticmethod
            def translate_with_cache_manager(
                blocks,
                _image,
                _extra_context,
                _cache_manager,
            ):
                for block in blocks:
                    block.translation = "cached"
                return blocks, "persistent-hit"

        processor.cache_manager = object()
        with mock.patch(
            "pipeline.stage_batched_processor.Translator",
            _AllHitTranslator,
        ), mock.patch(
            "pipeline.stage_batched_processor.apply_translation_result_dictionary",
        ):
            processor._translate_all(pages)

        processor._start_gemma_prewarm.assert_not_called()
        processor._await_gemma_runtime.assert_not_called()
        self.assertEqual(
            [page.blk_list[0].translation for page in pages],
            ["cached", "cached"],
        )

    def test_project_inpaint_all_hit_never_loads_or_releases_model(
        self,
    ) -> None:
        processor = self._processor(cancelled=False)
        processor._project_checkpoint_store = object()
        processor.main_page.settings_page = SimpleNamespace(
            get_hd_strategy_settings=lambda: {"strategy": "Original"},
            get_tool_selection=lambda _tool: "AOT",
            get_mask_refiner_settings=lambda: {"mask_refiner": "ctd"},
            get_inpainter_runtime_settings=lambda _key=None: {
                "backend": "torch",
                "device": "cpu",
                "inpaint_size": 2048,
                "precision": "fp32",
            },
            is_gpu_enabled=lambda: False,
            ui=SimpleNamespace(
                value_mappings={},
                tr=lambda value: value,
            ),
        )
        processor.main_page.image_ctrl = SimpleNamespace(
            mark_processing_stage=mock.Mock(),
        )
        processor._effective_export_settings = mock.Mock(return_value={})
        processor._ensure_page_state = mock.Mock(
            return_value={"brush_strokes": []}
        )
        processor._set_current_image = mock.Mock()
        processor.emit_progress = mock.Mock()
        processor._emit_benchmark_event = mock.Mock()
        processor._finish_inpaint_page = mock.Mock()
        processor._submit_or_inline_render = mock.Mock()
        processor._ensure_inpainter = mock.Mock()
        processor.inpainting = SimpleNamespace(
            inpainter_cache=None,
            release_inpainter_resources=mock.Mock(),
        )
        block = TextBlock(
            text_bbox=np.array([1, 1, 8, 8], dtype=np.int32),
            text="OCR",
            block_id="stable",
        )
        image = np.zeros((10, 10, 3), dtype=np.uint8)
        mask = np.zeros((10, 10), dtype=np.uint8)
        mask[1:4, 1:4] = 255
        page = StagePageContext(
            image_path="page.png",
            image_name="page.png",
            source_lang="Japanese",
            target_lang="Korean",
            image=image,
            blk_list=[block],
            source_decoded_sha256="1" * 64,
            project_checkpoint_page_key="page:00000000",
            detection_fingerprint="2" * 64,
            project_ocr_fingerprint="3" * 64,
        )
        hit = InpaintCheckpointResult(
            cleaned_image=image.copy(),
            raw_mask=mask.copy(),
            final_mask=mask.copy(),
            cleanup_stats={
                "applied": False,
                "component_count": 0,
                "block_count": 0,
            },
            cleaned_object_sha256="4" * 64,
            cleaned_decoded_sha256="5" * 64,
            block_states=[
                {
                    "block_id": "stable",
                    "attributes": {
                        "block_final_mask_pixel_count": 9,
                        "block_mask_bbox": [1, 1, 4, 4],
                        "block_mask_source": "ctd-refined",
                        "block_mask_decision": "accepted",
                    },
                }
            ],
        )

        with mock.patch(
            "pipeline.stage_batched_processor.generate_mask",
            side_effect=AssertionError(
                "mask generation must be skipped for a checkpoint hit"
            ),
        ), mock.patch(
            "pipeline.stage_batched_processor.lookup_inpaint_checkpoint",
            return_value=hit,
        ):
            processor._inpaint_all([page])

        processor._ensure_inpainter.assert_not_called()
        processor.inpainting.release_inpainter_resources.assert_not_called()
        self.assertEqual(page.project_inpaint_checkpoint_status, "hit")
        self.assertEqual(
            processor._inpainter_release_gate["status"],
            "not-loaded",
        )
        self.assertEqual(
            page.project_inpaint_artifact_sha256,
            "5" * 64,
        )
        self.assertEqual(block.block_final_mask_pixel_count, 9)
        self.assertEqual(block.block_mask_decision, "accepted")

    def test_preserve_only_page_skips_inpainter_and_keeps_pixels_exact(
        self,
    ) -> None:
        processor = self._processor(cancelled=False)
        processor._project_checkpoint_store = object()
        processor.main_page.settings_page = SimpleNamespace(
            get_hd_strategy_settings=lambda: {"strategy": "Original"},
            get_tool_selection=lambda _tool: "AOT",
            get_mask_refiner_settings=lambda: {"mask_refiner": "ctd"},
            get_inpainter_runtime_settings=lambda _key=None: {
                "backend": "torch",
                "device": "cuda",
                "inpaint_size": 2048,
                "precision": "fp32",
            },
            is_gpu_enabled=lambda: True,
            ui=SimpleNamespace(
                value_mappings={},
                tr=lambda value: value,
            ),
        )
        processor.main_page.image_ctrl = SimpleNamespace(
            mark_processing_stage=mock.Mock(),
        )
        processor._effective_export_settings = mock.Mock(return_value={})
        processor._ensure_page_state = mock.Mock(
            return_value={"brush_strokes": []}
        )
        processor._set_current_image = mock.Mock()
        processor.emit_progress = mock.Mock()
        processor._emit_benchmark_event = mock.Mock()
        processor._finish_inpaint_page = mock.Mock()
        processor._submit_or_inline_render = mock.Mock()
        processor._ensure_inpainter = mock.Mock()
        processor.inpainting = SimpleNamespace(
            inpainter_cache=None,
            release_inpainter_resources=mock.Mock(),
        )
        block = TextBlock(
            text_bbox=np.array([1, 1, 8, 8], dtype=np.int32),
            text="ドン",
            text_class="sfx",
            block_id="sfx-stable",
        )
        image = np.arange(10 * 10 * 3, dtype=np.uint8).reshape(
            10,
            10,
            3,
        )
        page = StagePageContext(
            image_path="page.png",
            image_name="page.png",
            source_lang="Japanese",
            target_lang="Korean",
            image=image,
            blk_list=[block],
            source_decoded_sha256="1" * 64,
            project_checkpoint_page_key="page:00000000",
            detection_fingerprint="2" * 64,
            project_ocr_fingerprint="3" * 64,
        )

        with mock.patch(
            "pipeline.stage_batched_processor.generate_mask",
            side_effect=AssertionError(
                "mask generation must be skipped for preserve-only pages"
            ),
        ), mock.patch(
            "pipeline.stage_batched_processor.lookup_inpaint_checkpoint",
            side_effect=AssertionError(
                "checkpoint lookup must be skipped for preserve-only pages"
            ),
        ):
            processor._inpaint_all([page])

        processor._ensure_inpainter.assert_not_called()
        processor.inpainting.release_inpainter_resources.assert_not_called()
        processor._finish_inpaint_page.assert_called_once()
        np.testing.assert_array_equal(page.inpaint_input_img, image)
        self.assertEqual(int(np.count_nonzero(page.raw_mask)), 0)
        self.assertEqual(int(np.count_nonzero(page.mask)), 0)
        self.assertEqual(
            page.project_inpaint_checkpoint_status,
            "skipped",
        )
        self.assertEqual(
            page.inpaint_diagnostics["status"],
            "processing_action_skipped",
        )
        self.assertEqual(
            page.inpaint_diagnostics["inference_call_count"],
            0,
        )
        self.assertEqual(
            processor._inpainter_release_gate["status"],
            "not-loaded",
        )

    def test_no_text_page_gets_renderable_skipped_stage_fingerprints(
        self,
    ) -> None:
        processor = self._processor(cancelled=False)
        processor._project_checkpoint_store = object()
        processor.main_page.settings_page = SimpleNamespace(
            get_hd_strategy_settings=lambda: {"strategy": "Original"},
            get_tool_selection=lambda _tool: "AOT",
            get_mask_refiner_settings=lambda: {"mask_refiner": "ctd"},
            get_inpainter_runtime_settings=lambda _key=None: {
                "backend": "torch",
                "device": "cpu",
                "inpaint_size": 2048,
                "precision": "fp32",
            },
            is_gpu_enabled=lambda: False,
            ui=SimpleNamespace(
                value_mappings={},
                tr=lambda value: value,
            ),
        )
        processor.main_page.image_ctrl = SimpleNamespace(
            mark_processing_stage=mock.Mock(),
        )
        processor._effective_export_settings = mock.Mock(return_value={})
        processor._set_current_image = mock.Mock()
        processor.emit_progress = mock.Mock()
        processor._emit_benchmark_event = mock.Mock()
        processor._ensure_inpainter = mock.Mock()
        # 이 테스트는 인페인트 단계의 skipped-stage 핑거프린트만 검증한다 — 페이지
        # N 인페인팅 완료 즉시 제출되는 렌더(Phase 3a 융합)는 별도로 검증되므로
        # 여기서는 무동작으로 둔다.
        processor._submit_or_inline_render = mock.Mock()
        processor.inpainting = SimpleNamespace(
            inpainter_cache=None,
            release_inpainter_resources=mock.Mock(),
        )
        image = np.zeros((10, 10, 3), dtype=np.uint8)
        page = StagePageContext(
            image_path="page.png",
            image_name="page.png",
            source_lang="Japanese",
            target_lang="Korean",
            image=image,
            blk_list=[],
            source_decoded_sha256="1" * 64,
            project_checkpoint_page_key="page:00000000",
            detection_fingerprint="2" * 64,
            no_text_detected=True,
        )

        processor._inpaint_all([page])

        processor._ensure_inpainter.assert_not_called()
        self.assertEqual(page.project_inpaint_checkpoint_status, "skipped")
        self.assertEqual(
            page.project_translation_checkpoint_status,
            "skipped",
        )
        self.assertEqual(len(page.project_translation_fingerprint), 64)
        self.assertEqual(len(page.project_inpaint_fingerprint), 64)
        self.assertEqual(len(page.project_inpaint_artifact_sha256), 64)

    def test_project_translation_all_hit_skips_gemma_and_http(self) -> None:
        processor = self._processor(cancelled=False)
        runtime_manager = LocalGemmaRuntimeManager()
        processor.main_page.local_translation_runtime_manager = (
            runtime_manager
        )
        processor.main_page.settings_page = SimpleNamespace(
            get_llm_settings=lambda: {"extra_context": "context"},
            get_tool_selection=lambda _tool: (
                "Custom Local Server(Gemma)"
            ),
            get_translation_result_dictionary_rules=lambda: [],
        )
        processor._project_checkpoint_store = object()
        processor._inpainter_release_gate = {
            "required": False,
            "observed": True,
        }
        processor._start_gemma_prewarm = mock.Mock()
        processor._await_gemma_runtime = mock.Mock()
        processor._shutdown_runtime_with_retry = mock.Mock()
        processor._set_current_image = mock.Mock()
        processor.emit_progress = mock.Mock()
        processor._report_runtime_progress = mock.Mock()
        processor._emit_benchmark_event = mock.Mock()
        processor._persist_translation_state = mock.Mock()
        processor.cache_manager = object()
        block = TextBlock(
            text_bbox=np.array([1, 1, 8, 8], dtype=np.int32),
            text="source",
            translation="",
            block_id="stable",
        )
        page = StagePageContext(
            image_path="page.png",
            image_name="page.png",
            source_lang="Japanese",
            target_lang="Korean",
            image=np.zeros((10, 10, 3), dtype=np.uint8),
            blk_list=[block],
            project_checkpoint_page_key="page:00000000",
            project_ocr_fingerprint="3" * 64,
            project_translation_snapshot=[
                {
                    "block_id": "stable",
                    "source_text": "source",
                    "translation": "캐시 번역",
                    "rich_text": "",
                    "source_lang": "Japanese",
                    "target_lang": "Korean",
                    "repetition_guard": None,
                }
            ],
        )
        translate_call = mock.Mock(
            side_effect=AssertionError("HTTP translation must be skipped")
        )

        class _ProjectHitTranslator:
            uses_persistent_translation_memory = True
            translator_key = "Custom Local Server(Gemma)"

            def __init__(self, *_args, **_kwargs) -> None:
                self.engine = SimpleNamespace(last_benchmark_stats={})
                self.translate_with_cache_manager = translate_call

            @staticmethod
            def prepare_translation(*_args, **_kwargs) -> bool:
                raise AssertionError(
                    "global translation cache lookup must be skipped"
                )

        result = TranslationCheckpointResult(
            translations={
                "stable": {
                    "translation": "캐시 번역",
                    "rich_text": "",
                    "source_lang": "Japanese",
                    "target_lang": "Korean",
                    "repetition_guard": None,
                }
            }
        )
        with mock.patch(
            "pipeline.stage_batched_processor.Translator",
            _ProjectHitTranslator,
        ), mock.patch.object(
            processor,
            "_build_project_translation_identity",
            return_value={"identity": "stable"},
        ), mock.patch(
            "pipeline.stage_batched_processor.lookup_translation_checkpoint",
            return_value=result,
        ):
            processor._translate_all([page])

        processor._start_gemma_prewarm.assert_not_called()
        processor._await_gemma_runtime.assert_not_called()
        processor._shutdown_runtime_with_retry.assert_not_called()
        translate_call.assert_not_called()
        self.assertEqual(block.translation, "캐시 번역")
        self.assertEqual(
            page.project_translation_checkpoint_status,
            "hit",
        )
        outcome = processor._page_outcomes([page])[0]
        self.assertEqual(
            outcome["source_units"][0]["status"],
            "translated",
        )
        self.assertEqual(
            outcome["source_units"][0]["semantic_quality_status"],
            "unjudged",
        )

    def test_preserve_only_page_skips_translator_and_gemma(self) -> None:
        processor = self._processor(cancelled=False)
        processor.main_page.settings_page = SimpleNamespace(
            get_llm_settings=lambda: {"extra_context": "context"},
            get_tool_selection=lambda _tool: (
                "Custom Local Server(Gemma)"
            ),
            get_translation_result_dictionary_rules=lambda: [],
        )
        processor.main_page.image_ctrl = SimpleNamespace(
            mark_processing_stage=mock.Mock(),
        )
        processor._project_checkpoint_store = object()
        processor._inpainter_release_gate = {
            "required": False,
            "observed": True,
        }
        processor._start_gemma_prewarm = mock.Mock()
        processor._await_gemma_runtime = mock.Mock()
        processor._shutdown_runtime_with_retry = mock.Mock()
        processor._set_current_image = mock.Mock()
        processor.emit_progress = mock.Mock()
        processor._report_runtime_progress = mock.Mock()
        processor._emit_benchmark_event = mock.Mock()
        processor._persist_translation_state = mock.Mock()
        processor.cache_manager = object()
        block = TextBlock(
            text_bbox=np.array([1, 1, 8, 8], dtype=np.int32),
            text="ドン",
            text_class="sfx",
            block_id="sfx-stable",
        )
        page = StagePageContext(
            image_path="page.png",
            image_name="page.png",
            source_lang="Japanese",
            target_lang="Korean",
            image=np.zeros((10, 10, 3), dtype=np.uint8),
            blk_list=[block],
            source_decoded_sha256="1" * 64,
            detection_fingerprint="2" * 64,
            project_ocr_fingerprint="3" * 64,
        )

        with mock.patch(
            "pipeline.stage_batched_processor.Translator",
            side_effect=AssertionError(
                "translator must not be constructed for preserve-only pages"
            ),
        ):
            processor._translate_all([page])

        processor._start_gemma_prewarm.assert_not_called()
        processor._await_gemma_runtime.assert_not_called()
        processor._shutdown_runtime_with_retry.assert_not_called()
        self.assertEqual(
            page.project_translation_checkpoint_status,
            "skipped",
        )
        processor.main_page.image_ctrl.mark_processing_stage.assert_called_once_with(
            "page.png",
            "translation",
            "skipped",
            reason="no_translate_inpaint_blocks",
        )
        outcome = processor._page_outcomes([page])[0]
        self.assertEqual(outcome["source_units"][0]["status"], "preserved")
        self.assertEqual(
            outcome["source_units"][0]["semantic_quality_status"],
            "unjudged",
        )

    def test_preserve_block_never_creates_render_item(self) -> None:
        # `_render_page_text_items`의 순수 계산은 Phase 3a 로 `render_worker.
        # _compute_render_text_items`(전용 렌더 워커에서 실행)로 옮겨졌다.
        # 이 테스트는 그 순수 함수만 직접 검증한다 — 시그널 발신·page_state
        # 기록은 `_finish_render_page_bookkeeping`(파이프라인 스레드) 쪽이며
        # `tests/test_render_worker.py`에서 별도로 다룬다.
        block = TextBlock(
            text_bbox=np.array([1, 1, 8, 8], dtype=np.int32),
            text="ドン",
            translation="쾅",
            text_class="sfx",
            block_id="sfx-stable",
        )
        render_settings = SimpleNamespace(
            font_family="Arial",
            color="#000000",
            alignment_id=1,
        )

        with mock.patch(
            "pipeline.render_worker.pyside_word_wrap",
            side_effect=AssertionError(
                "preserved text must not reach render layout"
            ),
        ):
            text_items_state, blk_rendered_events = render_worker._compute_render_text_items(
                [block],
                image_path="page.png",
                render_settings=render_settings,
                trg_lng_cd="Korean",
                strict_render_symbols=False,
                alignment=1,
                vertical_alignment=1,
            )

        self.assertEqual(text_items_state, [])
        self.assertEqual(blk_rendered_events, [])
        self.assertEqual(
            block._render_skip_reason,
            "processing_action_preserve",
        )

    def test_project_render_all_hit_skips_renderer_and_output_encode(
        self,
    ) -> None:
        processor = self._processor(cancelled=False)
        processor.main_page.lang_mapping = {"Korean": "Korean"}
        processor.main_page.render_settings = mock.Mock(
            return_value=SimpleNamespace(upper_case=False)
        )
        processor.main_page.image_ctrl = SimpleNamespace(
            update_processing_summary=mock.Mock(),
        )
        processor._ensure_page_state = mock.Mock(return_value={})
        processor._effective_export_settings = mock.Mock(return_value={})
        processor._set_current_image = mock.Mock()
        processor.emit_progress = mock.Mock()
        processor._write_json_exports = mock.Mock()
        processor._emit_benchmark_event = mock.Mock()
        processor._restore_render_project_state = mock.Mock()
        processor._reserve_render_output_path = mock.Mock()
        processor._warm_render_font_caches = mock.Mock()
        processor._log_page_done = mock.Mock()
        processor._pending_render_jobs = []
        page = StagePageContext(
            image_path="page.png",
            image_name="page.png",
            source_lang="Japanese",
            target_lang="Korean",
            directory=".",
            image=np.zeros((10, 10, 3), dtype=np.uint8),
            blk_list=[
                TextBlock(
                    text_bbox=np.array([1, 1, 8, 8], dtype=np.int32),
                    text="source",
                    translation="번역",
                    block_id="stable",
                )
            ],
            inpaint_input_img=np.zeros((10, 10, 3), dtype=np.uint8),
            mask=np.zeros((10, 10), dtype=np.uint8),
        )
        result = RenderCheckpointResult(
            output_path="cached.png",
            output_root=".",
            output_sha256="5" * 64,
            output_bytes=None,
            output_exists=True,
        )
        with mock.patch.object(
            processor,
            "_prepare_render_checkpoint",
            return_value=(result, "."),
        ), mock.patch(
            "pipeline.stage_batched_processor.materialize_render_checkpoint_output",
            return_value="cached.png",
        ) as materialize:
            processor._submit_or_inline_render(
                page, index=0, total_images=1, export_settings={}
            )

        materialize.assert_called_once_with(result)
        processor._restore_render_project_state.assert_called_once_with(page)
        # 체크포인트가 적중했으므로 전용 렌더 워커에 제출할 필요가 없다.
        processor._reserve_render_output_path.assert_not_called()
        self.assertEqual(processor._pending_render_jobs, [])

    def test_render_materialization_error_falls_back_to_renderer(self) -> None:
        # 체크포인트 materialize 가 실패하면 "miss" 취급으로 떨어져 정상 렌더
        # 경로로 넘어간다. Phase 3a 이후 정상 경로는 즉시 렌더하지 않고 전용
        # 단일 워커에 작업을 제출한다 — 여기서는 그 제출이 일어났는지만 본다.
        # 워커가 실제로 하는 순수 계산은 tests/test_render_worker.py 가 다룬다.
        processor = self._processor(cancelled=False)
        processor.main_page.lang_mapping = {"Korean": "Korean"}
        processor.main_page.render_settings = mock.Mock(
            return_value=SimpleNamespace(upper_case=False, alignment_id=1)
        )
        processor.main_page.image_ctrl = SimpleNamespace(
            update_processing_summary=mock.Mock(),
        )
        processor.main_page.curr_img_idx = -1
        processor.main_page.image_files = []
        processor.main_page.button_to_alignment = {1: 1}
        processor.main_page.button_to_vertical_alignment = {1: 1}
        page_state = {"viewer_state": {}}
        processor._ensure_page_state = mock.Mock(return_value=page_state)
        processor._effective_export_settings = mock.Mock(return_value={})
        processor._set_current_image = mock.Mock()
        processor.emit_progress = mock.Mock()
        processor._write_json_exports = mock.Mock()
        processor._emit_benchmark_event = mock.Mock()
        processor._restore_render_project_state = mock.Mock()
        processor._reserve_render_output_path = mock.Mock(
            return_value=("fresh.png", ".", "png")
        )
        processor._warm_render_font_caches = mock.Mock()
        processor._log_page_done = mock.Mock()
        processor._pending_render_jobs = []
        processor._render_cancel_event = threading.Event()
        fake_executor = SimpleNamespace(submit=mock.Mock(return_value=mock.Mock()))
        processor._ensure_render_executor = mock.Mock(return_value=fake_executor)
        page = StagePageContext(
            image_path="page.png",
            image_name="page.png",
            source_lang="Japanese",
            target_lang="Korean",
            directory=".",
            image=np.zeros((10, 10, 3), dtype=np.uint8),
            blk_list=[],
            inpaint_input_img=np.zeros((10, 10, 3), dtype=np.uint8),
            mask=np.zeros((10, 10), dtype=np.uint8),
            no_text_detected=True,
        )
        result = RenderCheckpointResult(
            output_path="cached.png",
            output_root=".",
            output_sha256="5" * 64,
            output_bytes=b"invalid",
            output_exists=False,
        )

        with mock.patch.object(
            processor,
            "_prepare_render_checkpoint",
            return_value=(result, "."),
        ), mock.patch(
            "pipeline.stage_batched_processor.materialize_render_checkpoint_output",
            side_effect=ValueError("invalid cached output"),
        ):
            processor._submit_or_inline_render(
                page, index=0, total_images=1, export_settings={}
            )

        processor._restore_render_project_state.assert_not_called()
        self.assertEqual(page.project_render_checkpoint_status, "miss")
        fake_executor.submit.assert_called_once()
        self.assertEqual(len(processor._pending_render_jobs), 1)
        submitted_job = fake_executor.submit.call_args.args[1]
        self.assertEqual(submitted_job.output_path, "fresh.png")

    def test_cache_state_change_cannot_bypass_unobserved_vram_release_gate(self) -> None:
        # 이 테스트들은 미확인 해제를 실패로 처리하는 계약 자체가 대상이다. 그
        # 처리는 진단용 강제를 켰을 때만 일어나므로 여기서 명시적으로 켠다.
        patcher = mock.patch(
            "pipeline.stage_batched_processor.gpu_release_enforcement_enabled",
            return_value=True,
        )
        patcher.start()
        self.addCleanup(patcher.stop)
        processor = self._processor(cancelled=False)
        processor.main_page.settings_page = SimpleNamespace(
            get_llm_settings=lambda: {"extra_context": "context"},
            get_tool_selection=lambda _tool: "Custom Local Server(Gemma)",
            get_translation_result_dictionary_rules=lambda: [],
        )
        processor._inpainter_release_gate = {
            "required": True,
            "observed": False,
        }
        processor.cache_manager = object()
        processor._start_gemma_prewarm = mock.Mock()
        processor._await_gemma_runtime = mock.Mock()
        processor._set_current_image = mock.Mock()
        processor.emit_progress = mock.Mock()
        processor._report_runtime_progress = mock.Mock()
        processor._emit_benchmark_event = mock.Mock()
        processor._persist_translation_state = mock.Mock()
        page = StagePageContext(
            image_path="page.png",
            image_name="page.png",
            source_lang="Japanese",
            target_lang="Korean",
            image=object(),
            blk_list=[SimpleNamespace(text="source", translation="")],
        )

        class _ChangingCacheTranslator:
            uses_persistent_translation_memory = True

            def __init__(self, *_args, **_kwargs) -> None:
                self.engine = SimpleNamespace(last_benchmark_stats={})
                self._prepare_results = iter((False, True))

            def prepare_translation(self, _blocks, _extra_context) -> bool:
                return next(self._prepare_results)

            @staticmethod
            def translate_with_cache_manager(*_args, **_kwargs):
                raise AssertionError("translation must not start before the VRAM gate")

        with mock.patch(
            "pipeline.stage_batched_processor.Translator",
            _ChangingCacheTranslator,
        ), self.assertRaisesRegex(RuntimeError, "VRAM release"):
            processor._translate_all([page])

        processor._start_gemma_prewarm.assert_not_called()
        processor._await_gemma_runtime.assert_not_called()

class StageSourceUnitReportingTests(unittest.TestCase):
    def _run_translation(self, fake_translator):
        processor = object.__new__(StageBatchedProcessor)
        processor.main_page = SimpleNamespace(
            is_current_task_cancelled=lambda: False,
            settings_page=SimpleNamespace(
                get_llm_settings=lambda: {"extra_context": "context"},
                get_tool_selection=lambda _tool: "example-translator",
                get_translation_result_dictionary_rules=lambda: [],
            ),
        )
        processor.cache_manager = object()
        processor._project_checkpoint_store = None
        processor._inpainter_release_gate = {
            "required": False,
            "observed": True,
        }
        processor._start_gemma_prewarm = mock.Mock()
        processor._await_gemma_runtime = mock.Mock()
        processor._shutdown_runtime_with_retry = mock.Mock()
        processor._set_current_image = mock.Mock()
        processor._report_runtime_progress = mock.Mock()
        processor._emit_benchmark_event = mock.Mock()
        processor._persist_translation_state = mock.Mock()
        processor._mark_page_failed = mock.Mock()
        processor._release_source_image = mock.Mock()
        processor._measure_performance = lambda **_kwargs: nullcontext()
        processor.emit_progress = mock.Mock()
        block = TextBlock(
            text_bbox=np.asarray([1, 1, 10, 10]),
            text="example source",
            text_class="text_bubble",
            block_id="example-source-unit",
            processing_action="translate_inpaint",
            processing_decision_source="example_detection",
            semantic_role="dialogue_bubble",
        )
        page = StagePageContext(
            image_path="example_source_page.png",
            image_name="example_source_page.png",
            source_lang="Japanese",
            target_lang="Korean",
            image=np.zeros((12, 12, 3), dtype=np.uint8),
            blk_list=[block],
        )
        with mock.patch(
            "pipeline.stage_batched_processor.Translator",
            return_value=fake_translator,
        ):
            processor._translate_all([page])
        return processor, page

    def test_canonical_alias_and_seam_lineage_stays_separate_until_output(self) -> None:
        processor = object.__new__(StageBatchedProcessor)
        canonical = TextBlock(
            text_bbox=np.asarray([1, 1, 10, 10]),
            block_id="source-canonical",
        )
        alias = TextBlock(
            text_bbox=np.asarray([1, 1, 10, 10]),
            block_id="source-alias",
        )
        merged = TextBlock(
            text_bbox=np.asarray([20, 1, 30, 10]),
            block_id="seam-top",
            compound_group_id="compound-example",
            merge_split_diagnostics={
                "source_units": [
                    {"block_id": "seam-top", "image_path": "example_top.png"},
                    {"block_id": "seam-bottom", "image_path": "example_bottom.png"},
                ]
            },
        )
        page = StagePageContext(
            image_path="example_top.png",
            image_name="example_top.png",
            source_lang="Japanese",
            target_lang="Korean",
            source_decoded_sha256="example-source",
            blk_list=[canonical, alias, merged],
        )

        StageBatchedProcessor._canonicalize_ocr_inputs([page])

        pending = processor._page_outcomes([page])[0]
        pending_units = {
            item["source_unit_id"]: item for item in pending["source_units"]
        }
        self.assertEqual(pending["source_unit_mapping_status"], "observed")
        for source_id in ("source-canonical", "source-alias", "seam-top", "seam-bottom"):
            self.assertEqual(pending_units[source_id]["output_unit_ids"], [])
            self.assertEqual(
                pending_units[source_id]["output_relation_status"],
                "unavailable",
            )

        canonical.translation = "example translated sentence"
        merged.translation = "example translated sentence"
        StageBatchedProcessor._record_source_unit_translation_results(
            page,
            page.blk_list,
            page.blk_list,
        )
        page.output_path = "example-output.png"
        outcome = processor._page_outcomes([page])[0]
        units = {item["source_unit_id"]: item for item in outcome["source_units"]}
        self.assertEqual(
            units["source-alias"]["output_unit_ids"], ["source-canonical"]
        )
        self.assertEqual(
            units["seam-top"]["output_unit_ids"], ["seam-top"]
        )
        self.assertEqual(
            units["seam-bottom"]["output_unit_ids"], ["seam-top"]
        )
        self.assertEqual(
            units["seam-top"]["output_group_ids"], ["compound-example"]
        )
        self.assertEqual(units["source-alias"]["status"], "translated")
        self.assertEqual(units["seam-bottom"]["status"], "translated")
        self.assertEqual(
            units["seam-bottom"]["source_image_path"], "example_bottom.png"
        )
        self.assertEqual(units["seam-bottom"]["semantic_quality_status"], "unjudged")

    def test_ocr_preserve_skip_and_failure_statuses_reach_page_outcome(self) -> None:
        from modules.ocr.persistent_cache import snapshot_raw_ocr_result
        from modules.ocr.paddle_crop.engine import PaddleOCRVLEngine
        from modules.utils.ocr_debug import (
            OCR_EMPTY_REASON_EMBEDDED_UI_CLUSTER,
            OCR_EMPTY_REASON_HUNYUAN_TRUNCATED,
            OCR_EMPTY_REASON_NON_TEXT_RESPONSE,
        )
        from modules.utils.run_report import build_run_report

        processor = object.__new__(StageBatchedProcessor)
        processor.main_page = SimpleNamespace(
            lang_mapping={"Japanese": "Japanese"},
            settings_page=SimpleNamespace(
                is_gpu_enabled=lambda: False,
                get_ocr_result_dictionary_rules=lambda: [],
            ),
        )
        processor.cache_manager = mock.Mock()
        processor._paddleocr_cache_store = None
        processor._paddleocr_cache_identity = None
        processor._await_ocr_runtime = mock.Mock()
        translated = TextBlock(
            text_bbox=np.asarray([1, 1, 10, 10]),
            text="example text",
            text_class="text_bubble",
            block_id="example-translated",
        )
        preserved = TextBlock(
            text_bbox=np.asarray([12, 1, 20, 10]),
            text_class="text_free",
            block_id="example-preserved",
        )
        skipped = TextBlock(
            text_bbox=np.asarray([22, 1, 30, 10]),
            text_class="text_free",
            block_id="example-skipped",
        )
        failed = TextBlock(
            text_bbox=np.asarray([32, 1, 40, 10]),
            text_class="text_bubble",
            block_id="example-failed",
        )
        hunyuan_failed = TextBlock(
            text_bbox=np.asarray([42, 1, 50, 10]),
            text_class="text_bubble",
            block_id="example-hunyuan-failed",
        )
        preserved.ocr_empty_reason = OCR_EMPTY_REASON_NON_TEXT_RESPONSE
        skipped.ocr_empty_reason = OCR_EMPTY_REASON_EMBEDDED_UI_CLUSTER
        failed.ocr_empty_reason = OCR_EMPTY_REASON_NON_TEXT_RESPONSE
        failed.ocr_reject_reason = PaddleOCRVLEngine.TRUNCATED_OCR_REASON
        hunyuan_failed.ocr_empty_reason = OCR_EMPTY_REASON_HUNYUAN_TRUNCATED
        blocks = [translated, preserved, skipped, failed, hunyuan_failed]
        for block in blocks:
            block.ocr_status = "ok" if block is translated else "empty_initial"
            block.text = "example text" if block is translated else ""
        page = StagePageContext(
            image_path="example_source_page.png",
            image_name="example_source_page.png",
            source_lang="Japanese",
            target_lang="Korean",
            image=np.zeros((48, 48, 3), dtype=np.uint8),
            blk_list=blocks,
            project_ocr_hit=SimpleNamespace(
                raw_results={
                    block.block_id: snapshot_raw_ocr_result(block)
                    for block in blocks
                },
                page_profile={},
                attempt_count=1,
                engine_name="PaddleOCRVLEngine",
            ),
        )

        processor._run_primary_ocr(
            page,
            {
                "primary_ocr_engine": "PaddleOCR VL",
                "normalized_ocr_mode": "best_local",
            },
        )
        page.output_path = "example-output.png"

        units = {
            row["source_unit_id"]: row
            for row in processor._page_outcomes([page])[0]["source_units"]
        }
        self.assertEqual(units["example-preserved"]["status"], "preserved")
        self.assertEqual(units["example-skipped"]["status"], "intentionally_skipped")
        self.assertEqual(units["example-failed"]["status"], "failed")
        self.assertEqual(units["example-failed"]["reason_code"], "ocr_truncated")
        self.assertEqual(units["example-failed"]["output_unit_ids"], [])
        self.assertEqual(units["example-hunyuan-failed"]["status"], "failed")
        self.assertEqual(
            units["example-hunyuan-failed"]["reason_code"], "ocr_truncated"
        )
        self.assertEqual(
            units["example-hunyuan-failed"]["semantic_quality_status"],
            "unjudged",
        )
        self.assertEqual(units["example-hunyuan-failed"]["output_unit_ids"], [])
        self.assertEqual(
            units["example-failed"]["output_relation_status"], "unavailable"
        )
        self.assertEqual(
            units["example-translated"]["output_relation_status"], "observed"
        )
        self.assertEqual(units["example-translated"]["status"], "unjudged")
        report = build_run_report(
            telemetry=None,
            total_wall_sec=1.0,
            page_outcomes=[processor._page_outcomes([page])[0]],
            output_summary={
                "input_count": 1,
                "output_count": 1,
                "fallback_count": 0,
                "fallbacks": [],
                "missing": [],
            },
            terminal_status="completed",
            processing_status="completed",
            report_status="completed",
            cleanup_status="completed",
        )
        self.assertEqual(report["quality"]["status"], "failed")
        self.assertEqual(report["terminal_status"], "failed")
        self.assertEqual(report["pages"][0]["quality_status"], "failed")

    def test_translation_failure_is_attributed_only_by_block_id(self) -> None:
        error = RuntimeError("example truncated response")
        error.block_failure_kind = "translation_truncated"
        error.block_ids = ["example-source-unit"]

        class _FailingTranslator:
            uses_persistent_translation_memory = False
            translator_key = "example-translator"
            engine = SimpleNamespace(last_benchmark_stats={})

            @staticmethod
            def translate_with_cache_manager(*_args, **_kwargs):
                raise error

        processor, page = self._run_translation(_FailingTranslator())

        outcome = processor._page_outcomes([page])[0]
        unit = outcome["source_units"][0]
        self.assertEqual(unit["source_unit_id"], "example-source-unit")
        self.assertEqual(unit["status"], "failed")
        self.assertEqual(unit["reason_code"], "translation_truncated")
        self.assertEqual(unit["semantic_quality_status"], "unjudged")

    def test_cancelled_translation_stays_unjudged_with_reason(self) -> None:
        class _CancelledTranslator:
            uses_persistent_translation_memory = False
            translator_key = "example-translator"
            engine = SimpleNamespace(last_benchmark_stats={})

            @staticmethod
            def translate_with_cache_manager(*_args, **_kwargs):
                raise OperationCancelledError("example cancellation")

        processor = object.__new__(StageBatchedProcessor)
        processor.main_page = SimpleNamespace(
            is_current_task_cancelled=lambda: False,
            settings_page=SimpleNamespace(
                get_llm_settings=lambda: {"extra_context": "context"},
                get_tool_selection=lambda _tool: "example-translator",
                get_translation_result_dictionary_rules=lambda: [],
            ),
        )
        processor.cache_manager = object()
        processor._project_checkpoint_store = None
        processor._inpainter_release_gate = {
            "required": False,
            "observed": True,
        }
        processor._start_gemma_prewarm = mock.Mock()
        processor._await_gemma_runtime = mock.Mock()
        processor._shutdown_runtime_with_retry = mock.Mock()
        processor._set_current_image = mock.Mock()
        processor._report_runtime_progress = mock.Mock()
        processor._emit_benchmark_event = mock.Mock()
        processor._persist_translation_state = mock.Mock()
        processor._mark_page_failed = mock.Mock()
        processor._release_source_image = mock.Mock()
        processor._measure_performance = lambda **_kwargs: nullcontext()
        processor.emit_progress = mock.Mock()
        block = TextBlock(
            text_bbox=np.asarray([1, 1, 10, 10]),
            text="example source",
            text_class="text_bubble",
            block_id="example-source-unit",
            processing_action="translate_inpaint",
            processing_decision_source="example_detection",
            semantic_role="dialogue_bubble",
        )
        page = StagePageContext(
            image_path="example_source_page.png",
            image_name="example_source_page.png",
            source_lang="Japanese",
            target_lang="Korean",
            image=np.zeros((12, 12, 3), dtype=np.uint8),
            blk_list=[block],
        )
        with mock.patch(
            "pipeline.stage_batched_processor.Translator",
            return_value=_CancelledTranslator(),
        ), self.assertRaises(OperationCancelledError):
            processor._translate_all([page])

        outcome = processor._page_outcomes(
            [page],
            processing_status="cancelled",
        )[0]
        unit = outcome["source_units"][0]
        self.assertEqual(unit["status"], "unjudged")
        self.assertEqual(
            unit["reason_code"],
            "run_cancelled_before_unit_outcome",
        )


if __name__ == "__main__":
    unittest.main()

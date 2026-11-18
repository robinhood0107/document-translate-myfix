from __future__ import annotations

import threading
import unittest
from contextlib import nullcontext
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import mock

import numpy as np
from PIL import Image
from PySide6 import QtCore

from app.controllers.series import SeriesController
from app.projects.checkpoint_store import (
    ProjectCheckpointStore,
    checkpoint_reference_for_save,
)
from app.projects.project_types import PROJECT_KIND_SINGLE
from app.projects.stage_checkpoints import (
    build_project_ocr_fingerprint,
    build_project_ocr_identity,
    lookup_ocr_checkpoint,
    record_ocr_checkpoint,
)
from app.projects.series_state_v1 import (
    create_series_project,
    load_series_project,
    load_series_project_blob,
)
from modules.ocr.local_runtime import LocalOCRRuntimeManager
from modules.ocr.ocr_paddle_VL import PaddleOCRVLEngine
from modules.ocr.persistent_cache import snapshot_raw_ocr_result
from modules.utils.exceptions import OperationCancelledError
from modules.utils.textblock import TextBlock, ensure_text_block_id
from pipeline.stage_batched_processor import (
    StageBatchedProcessor,
    StagePageContext,
    decoded_image_sha256,
)


class ConsecutiveStageBatchedOutputTests(unittest.TestCase):
    def _processor(self, output_root: Path) -> StageBatchedProcessor:
        processor = object.__new__(StageBatchedProcessor)
        cancelled = {"value": False}
        processor.main_page = SimpleNamespace(
            image_files=[],
            file_handler=SimpleNamespace(should_pre_materialize=lambda _paths: False),
            settings_page=object(),
            project_ctrl=None,
            is_current_task_cancelled=lambda: cancelled["value"],
        )
        processor._test_cancelled = cancelled
        processor._render_cancel_event = threading.Event()
        processor._render_executor = None
        processor._pending_render_jobs = []
        processor._recent_page_durations = []
        processor._benchmark_stage_ceiling = lambda: None
        processor._measure_performance = lambda **_kwargs: nullcontext()
        processor._load_page_contexts = lambda paths: [
            StagePageContext(
                image_path=path,
                image_name=Path(path).name,
                source_lang="Japanese",
                target_lang="Korean",
                directory=str(output_root),
                image=np.zeros((8, 8, 3), dtype=np.uint8),
                blk_list=[],
                mask=np.zeros((8, 8), dtype=np.uint8),
                no_text_detected=True,
            )
            for path in paths
        ]
        processor._ensure_stage_policy = lambda _pages: {"primary_ocr_engine": "Default"}
        processor._ocr_runtime_service_name = lambda _key: "ocr"
        processor._effective_export_settings = lambda _settings: {}
        for method in (
            "_emit_benchmark_event",
            "_record_performance_workload",
            "_reset_prewarm_lifecycle",
            "_shutdown_managed_runtimes",
            "_start_ocr_container_prewarm",
            "_detect_all",
            "_await_ocr_container_prewarm",
            "_start_ocr_prewarm",
            "_start_gemma_page_cache_prefetch",
            "_ocr_all",
            "_translate_all",
            "_release_gemma_before_inpainter",
            "_inpaint_all",
            "_sample_performance_resources",
            "_record_stage_memory_telemetry",
            "_persist_stage_rates",
            "_shutdown_page_cache_executor",
            "_shutdown_prewarm_executor",
            "_write_run_report",
        ):
            setattr(processor, method, mock.Mock())
        return processor

    def test_batch_entrypoint_prepares_and_releases_one_paddle_plan_per_page(
        self,
    ) -> None:
        with TemporaryDirectory() as temporary:
            processor = self._processor(Path(temporary))
            image = np.full((120, 120, 3), 220, dtype=np.uint8)
            source_sha = decoded_image_sha256(image)
            pages = [
                StagePageContext(
                    image_path=f"synthetic-{index}.png",
                    image_name=f"synthetic-{index}.png",
                    source_lang="Japanese",
                    target_lang="Korean",
                    image=image.copy(),
                    source_decoded_sha256=source_sha,
                    source_pixel_count=120 * 120,
                    project_checkpoint_page_key=f"page:{index:08d}",
                    detection_fingerprint=f"synthetic-detection-{index}",
                    blk_list=[
                        TextBlock(
                            text_bbox=np.array([30, 30, 90, 60], dtype=np.int32),
                            text_class="text_bubble",
                            source_lang="ja",
                            direction="horizontal",
                        )
                    ],
                )
                for index in range(2)
            ]
            settings = SimpleNamespace(
                get_paddleocr_vl_settings=lambda: {
                    "persistent_cache_enabled": True,
                    "parallel_workers": 1,
                    "max_new_tokens": 256,
                    "prettify_markdown": False,
                    "visualize": False,
                },
                get_project_checkpoint_settings=lambda: {"enabled": True},
                get_ocr_generic_settings=lambda: {
                    "paddleocr_vl_scheduler_mode": "fixed",
                    "manga_expansion_percentage": 7,
                    "crop_padding_ratio": 0.05,
                },
                get_ocr_result_dictionary_rules=lambda: [],
                get_tool_selection=lambda _tool: "PaddleOCR VL",
                is_gpu_enabled=lambda: False,
            )
            processor.main_page.settings_page = settings
            processor.main_page.lang_mapping = {"Japanese": "Japanese"}
            processor.main_page.image_files = [page.image_path for page in pages]
            project_file = Path(temporary) / "synthetic-project.ctpr"
            project_file.write_bytes(b"synthetic project")
            project_reference = checkpoint_reference_for_save(
                None,
                project_file,
            )
            processor.main_page.project_file = str(project_file)
            processor.main_page.project_kind = PROJECT_KIND_SINGLE
            processor.main_page.project_checkpoint_reference = (
                project_reference.to_dict()
            )
            processor.main_page.project_checkpoint_reference_persisted = True
            project_store = ProjectCheckpointStore(
                project_file,
                project_reference,
                enabled=True,
            )
            self.assertTrue(project_store.ensure_initialized())
            runtime_manager = LocalOCRRuntimeManager()
            identity_before = {"runtime_fingerprint": "before-start"}
            identity_after = {"runtime_fingerprint": "after-start"}
            policy = {
                "primary_ocr_engine": "PaddleOCR VL",
                "normalized_ocr_mode": "best_local",
            }
            runtime_manager.get_ocr_cache_identity = mock.Mock(
                side_effect=[identity_before, identity_after]
            )
            processor.main_page.local_ocr_runtime_manager = runtime_manager
            processor.main_page.image_ctrl = SimpleNamespace(
                load_image=lambda _path: image.copy(),
                mark_processing_stage=mock.Mock(),
                update_processing_summary=mock.Mock(),
            )
            processor._load_page_contexts = lambda _paths: pages
            processor._ensure_stage_policy = lambda _pages: policy
            processor._detect_all = mock.Mock()
            processor._ocr_all = StageBatchedProcessor._ocr_all.__get__(processor)
            processor._get_paddleocr_cache_store = lambda: SimpleNamespace(
                lookup_many=lambda _keys: {},
                store_records=lambda _records: True,
            )
            processor._await_ocr_runtime = mock.Mock()
            processor._shutdown_runtime_with_retry = mock.Mock()
            processor._persist_ocr_state = mock.Mock()
            processor._set_current_image = mock.Mock()
            processor.emit_progress = mock.Mock()
            processor._complete_ocr_stage_ceiling = mock.Mock()
            processor._benchmark_stage_ceiling = lambda: "ocr"
            live_plan_counts: list[int] = []
            plans_at_runtime_start: list[list[str]] = []
            prepared_bytes_at_runtime_start: list[tuple[int, int]] = []
            prepared_identities: list[dict[str, str]] = []
            process_calls = 0
            project_lookups: list[tuple[str, str]] = []
            project_records: list[tuple[str, str]] = []
            project_hits_consumed: list[tuple[str, bool]] = []
            result_texts: dict[str, str] = {}

            seed_block = pages[1].blk_list[0]
            seed_block_id = ensure_text_block_id(seed_block)
            seed_block.text = "cached under old identity"
            seed_block.ocr_status = "ok"
            seed_identity = build_project_ocr_identity(
                detection_fingerprint=pages[1].detection_fingerprint,
                runtime_identity=identity_before,
                policy=policy,
                paddle_settings=settings.get_paddleocr_vl_settings(),
                source_lang_english="Japanese",
            )
            self.assertTrue(
                record_ocr_checkpoint(
                    project_store,
                    page_key=pages[1].project_checkpoint_page_key,
                    fingerprint=build_project_ocr_fingerprint(seed_identity),
                    identity=seed_identity,
                    blocks=[seed_block],
                    raw_results={seed_block_id: snapshot_raw_ocr_result(seed_block)},
                    attempt_count=1,
                    engine_name="PaddleOCRVLEngine",
                    page_profile={},
                )
            )

            def lookup_checkpoint(store, **kwargs):
                project_lookups.append(
                    (
                        kwargs["page_key"],
                        str(kwargs["identity"]["runtime"]["runtime_fingerprint"]),
                    )
                )
                return lookup_ocr_checkpoint(store, **kwargs)

            def record_checkpoint(store, **kwargs):
                project_records.append(
                    (
                        kwargs["page_key"],
                        str(kwargs["identity"]["runtime"]["runtime_fingerprint"]),
                    )
                )
                return record_ocr_checkpoint(store, **kwargs)

            run_primary_ocr = StageBatchedProcessor._run_primary_ocr.__get__(
                processor
            )

            def run_primary_with_observation(ctx, current_policy):
                result = run_primary_ocr(ctx, current_policy)
                project_hits_consumed.append(
                    (ctx.image_name, ctx.project_ocr_hit is not None)
                )
                result_texts[ctx.image_name] = str(
                    getattr(ctx.blk_list[0], "text", "") or ""
                )
                return result

            processor._run_primary_ocr = run_primary_with_observation

            def create_engine(*_args, **_kwargs):
                live_plan_counts.append(
                    sum(page.paddleocr_cache_plan is not None for page in pages)
                )
                engine = PaddleOCRVLEngine()
                engine.initialize(settings)
                engine._resolve_gpu_metrics_for_scheduler = lambda: {}
                prepare_plan = engine.prepare_persistent_cache

                def prepare(*args, **kwargs):
                    prepared_identities.append(dict(args[3]))
                    return prepare_plan(*args, **kwargs)

                engine.prepare_persistent_cache = prepare

                def process(plan):
                    nonlocal process_calls
                    process_calls += 1
                    text = "" if process_calls == 1 else "synthetic OCR"
                    status = "empty" if process_calls == 1 else "ok"
                    for job in plan.runtime_jobs:
                        job["block"].text = text
                        job["block"].ocr_status = status
                        job["request_record"]["status"] = status
                    plan.page_profile["page_status"] = "ok"
                    engine.last_page_profile = plan.page_profile
                    return plan.blocks

                engine.process_persistent_cache_plan = process
                return engine

            def snapshot_runtime_start(_policy) -> None:
                live_plans = [
                    page.paddleocr_cache_plan
                    for page in pages
                    if page.paddleocr_cache_plan is not None
                ]
                plans_at_runtime_start.append(
                    [
                        page.image_name
                        for page in pages
                        if page.paddleocr_cache_plan is not None
                    ]
                )
                prepared_bytes_at_runtime_start.append(
                    (
                        sum(
                            int(job["prepared_crop"].nbytes)
                            for plan in live_plans
                            for job in plan.jobs
                            if job.get("prepared_crop") is not None
                        ),
                        sum(
                            len(job["prepared_image_bytes"])
                            for plan in live_plans
                            for job in plan.jobs
                            if job.get("prepared_image_bytes") is not None
                        ),
                    )
                )

            processor._await_ocr_runtime.side_effect = snapshot_runtime_start
            with mock.patch(
                "pipeline.stage_batched_processor.OCRFactory.create_engine",
                side_effect=create_engine,
            ), mock.patch(
                "pipeline.stage_batched_processor.lookup_ocr_checkpoint",
                side_effect=lookup_checkpoint,
            ), mock.patch(
                "pipeline.stage_batched_processor.record_ocr_checkpoint",
                side_effect=record_checkpoint,
            ):
                processor.batch_process([page.image_path for page in pages])

            self.assertEqual(
                project_hits_consumed,
                [("synthetic-0.png", False), ("synthetic-1.png", False)],
            )
            self.assertEqual(
                result_texts,
                {
                    "synthetic-0.png": "synthetic OCR",
                    "synthetic-1.png": "synthetic OCR",
                },
            )
            self.assertEqual(
                project_lookups,
                [
                    ("page:00000000", "before-start"),
                    ("page:00000000", "after-start"),
                    ("page:00000001", "after-start"),
                ],
            )
            self.assertEqual(
                project_records,
                [
                    ("page:00000000", "after-start"),
                    ("page:00000001", "after-start"),
                ],
            )
            self.assertEqual(live_plan_counts, [0, 0, 0])
            self.assertEqual(plans_at_runtime_start, [["synthetic-0.png"]])
            self.assertEqual(len(prepared_bytes_at_runtime_start), 1)
            self.assertGreater(prepared_bytes_at_runtime_start[0][0], 0)
            self.assertGreater(prepared_bytes_at_runtime_start[0][1], 0)
            self.assertEqual(processor._await_ocr_runtime.call_count, 1)
            self.assertEqual(
                prepared_identities,
                [identity_before, identity_after, identity_after, identity_after],
            )
            self.assertEqual(runtime_manager.get_ocr_cache_identity.call_count, 2)
            self.assertEqual(process_calls, 3)
            identity_after_page_one = build_project_ocr_identity(
                detection_fingerprint=pages[1].detection_fingerprint,
                runtime_identity=identity_after,
                policy=policy,
                paddle_settings=settings.get_paddleocr_vl_settings(),
                source_lang_english="Japanese",
            )
            page_one_hit = lookup_ocr_checkpoint(
                project_store,
                page_key=pages[1].project_checkpoint_page_key,
                fingerprint=build_project_ocr_fingerprint(identity_after_page_one),
                identity=identity_after_page_one,
                detection_blocks=pages[1].blk_list,
            )
            self.assertIsNotNone(page_one_hit)
            self.assertEqual(page_one_hit.blocks[0].text, "synthetic OCR")
            self.assertTrue(
                all(page.paddleocr_cache_plan is None for page in pages)
            )

    def test_batch_empty_paddle_cache_defers_model_start_until_plan_miss(
        self,
    ) -> None:
        for guard_only in (True, False):
            with self.subTest(guard_only=guard_only), TemporaryDirectory() as temporary:
                processor = self._processor(Path(temporary))
                image = np.full((240, 320, 3), 128, dtype=np.uint8)
                block = TextBlock(
                    text_bbox=np.array([40, 50, 220, 95], dtype=np.int32),
                    bubble_bbox=None if guard_only else np.array(
                        [30, 40, 230, 105], dtype=np.int32
                    ),
                    text_class="text_free" if guard_only else "text_bubble",
                    source_lang="en" if guard_only else "ja",
                    direction="horizontal",
                )
                page = StagePageContext(
                    image_path="synthetic.png",
                    image_name="synthetic.png",
                    source_lang="Japanese",
                    target_lang="Korean",
                    image=image.copy(),
                    source_decoded_sha256=decoded_image_sha256(image),
                    source_pixel_count=240 * 320,
                    blk_list=[block],
                )
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
                policy = {
                    "primary_ocr_engine": "PaddleOCR VL",
                    "normalized_ocr_mode": "best_local",
                }
                processor.main_page.settings_page = settings
                processor.main_page.lang_mapping = {"Japanese": "Japanese"}
                processor.main_page.image_files = [page.image_path]
                runtime_manager = LocalOCRRuntimeManager()
                runtime_manager.get_ocr_cache_identity = mock.Mock(
                    return_value={"runtime_fingerprint": "synthetic"}
                )
                runtime_manager.ensure_engine = mock.Mock()
                processor.main_page.local_ocr_runtime_manager = runtime_manager
                processor.main_page.image_ctrl = SimpleNamespace(
                    load_image=lambda _path: image.copy(),
                    mark_processing_stage=mock.Mock(),
                    update_processing_summary=mock.Mock(),
                )
                processor._load_page_contexts = lambda _paths: [page]
                processor._ensure_stage_policy = lambda _pages: policy
                processor._detect_all = mock.Mock()
                processor._ocr_all = StageBatchedProcessor._ocr_all.__get__(
                    processor
                )
                processor._start_ocr_prewarm = (
                    StageBatchedProcessor._start_ocr_prewarm.__get__(processor)
                )
                processor._start_prewarm = mock.Mock()
                processor._prewarm_jobs = {}
                processor._get_paddleocr_cache_store = lambda: SimpleNamespace(
                    stats=lambda: {"enabled": True, "item_count": 0},
                    lookup_many=lambda _keys: {},
                    store_records=lambda _records: True,
                )
                processor._await_ocr_runtime = (
                    StageBatchedProcessor._await_ocr_runtime.__get__(processor)
                )
                processor._capture_runtime_gpu_start_baseline = mock.Mock()
                processor._record_runtime_performance = mock.Mock()
                processor._shutdown_runtime_with_retry = mock.Mock()
                processor._persist_ocr_state = mock.Mock()
                processor._ensure_page_state = mock.Mock(return_value={})
                processor._set_current_image = mock.Mock()
                processor.emit_progress = mock.Mock()
                processor._log_ocr_quality = mock.Mock()
                processor._complete_ocr_stage_ceiling = mock.Mock()
                processor._benchmark_stage_ceiling = lambda: "ocr"
                await_calls: list[str] = []
                await_runtime = processor._await_ocr_runtime

                def observe_await(current_policy):
                    await_calls.append("ocr")
                    return await_runtime(current_policy)

                processor._await_ocr_runtime = observe_await
                created_engines: list[PaddleOCRVLEngine] = []

                def create_engine(*_args, **_kwargs):
                    engine = PaddleOCRVLEngine()
                    engine.initialize(settings)
                    engine._resolve_gpu_metrics_for_scheduler = lambda: {}
                    engine._request_ocr_text_from_encoded = mock.Mock(
                        side_effect=AssertionError("real OCR request is forbidden")
                    )
                    if not guard_only:
                        def process(plan):
                            for job in plan.runtime_jobs:
                                job["block"].text = "synthetic OCR"
                                job["block"].ocr_status = "ok"
                                job["request_record"]["status"] = "ok"
                            plan.page_profile["page_status"] = "ok"
                            engine.last_page_profile = plan.page_profile
                            return plan.blocks

                        engine.process_persistent_cache_plan = process
                    created_engines.append(engine)
                    return engine

                with mock.patch(
                    "pipeline.stage_batched_processor.OCRFactory.create_engine",
                    side_effect=create_engine,
                ), mock.patch(
                    "pipeline.stage_batched_processor.open_project_stage_checkpoint_store",
                    return_value=None,
                ):
                    processor.batch_process([page.image_path])

                processor._start_prewarm.assert_not_called()
                self.assertEqual(await_calls, [] if guard_only else ["ocr"])
                if guard_only:
                    runtime_manager.ensure_engine.assert_not_called()
                else:
                    runtime_manager.ensure_engine.assert_called_once()
                self.assertTrue(created_engines)
                for engine in created_engines:
                    engine._request_ocr_text_from_encoded.assert_not_called()

    def test_same_processor_writes_distinct_pixels_on_two_runs(self) -> None:
        with TemporaryDirectory() as temporary:
            output_root = Path(temporary)
            processor = self._processor(output_root)
            values = iter((31, 219))

            def render(pages: list[StagePageContext]) -> None:
                if processor._render_cancel_event.is_set():
                    raise OperationCancelledError("A retired render token was reused")
                value = next(values)
                for page in pages:
                    target = output_root / f"{Path(page.image_path).stem}-translated.png"
                    Image.new("RGB", (8, 8), (value, 0, 0)).save(target)
                    page.output_path = str(target)

            processor._render_all = render
            with mock.patch(
                "pipeline.stage_batched_processor.open_project_stage_checkpoint_store",
                return_value=None,
            ):
                processor.batch_process(["chapter-a.png"])
                retired_event = processor._render_cancel_event
                processor.batch_process(["chapter-b.png"])

            self.assertTrue(retired_event.is_set())
            self.assertIsNot(retired_event, processor._render_cancel_event)
            for name, value in (("chapter-a", 31), ("chapter-b", 219)):
                target = output_root / f"{name}-translated.png"
                self.assertTrue(target.is_file(), name)
                with Image.open(target) as image:
                    self.assertEqual(image.getpixel((0, 0)), (value, 0, 0))
            self.assertEqual(processor._write_run_report.call_count, 2)

    def test_missing_output_is_reported_as_failure_not_success(self) -> None:
        with TemporaryDirectory() as temporary:
            processor = self._processor(Path(temporary))
            processor._render_all = lambda _pages: None
            processor._write_fallback_export = mock.Mock(return_value=False)
            processor._release_page_buffers = mock.Mock()
            with mock.patch(
                "pipeline.stage_batched_processor.open_project_stage_checkpoint_store",
                return_value=None,
            ), self.assertRaisesRegex(RuntimeError, "0/1"):
                processor.batch_process(["chapter-a.png"])

            self.assertEqual(
                processor._write_run_report.call_args.kwargs["output_summary"]["missing"],
                ["chapter-a.png"],
            )
            self.assertEqual(
                processor._write_run_report.call_args.kwargs["output_summary"]["missing_source_unit_indices"],
                [0],
            )
            tags = [call.args[0] for call in processor._emit_benchmark_event.call_args_list]
            self.assertIn("batch_run_failed", tags)
            self.assertNotIn("batch_run_done", tags)

    def test_report_failure_runs_after_cleanup_and_keeps_output(self) -> None:
        with TemporaryDirectory() as temporary:
            output_root = Path(temporary)
            processor = self._processor(output_root)
            events: list[str] = []

            def shutdown(
                *,
                context: str,
                raise_on_failure: bool = False,
                preserve_sleeping_paddle: bool = False,
                ocr_service: str = "ocr",
            ) -> None:
                if context == "terminal cleanup":
                    events.append("cleanup")

            def render(pages: list[StagePageContext]) -> None:
                target = output_root / "chapter-a-translated.png"
                Image.new("RGB", (8, 8), (31, 0, 0)).save(target)
                pages[0].output_path = str(target)

            def write_report(*_args, **_kwargs) -> None:
                events.append("report")
                raise OSError("synthetic report failure")

            processor._shutdown_managed_runtimes.side_effect = shutdown
            processor._render_all = render
            processor._write_run_report.side_effect = write_report
            with mock.patch(
                "pipeline.stage_batched_processor.open_project_stage_checkpoint_store",
                return_value=None,
            ):
                processor.batch_process(["chapter-a.png"])

            self.assertEqual(events, ["cleanup", "report"])
            self.assertTrue((output_root / "chapter-a-translated.png").is_file())
            self.assertEqual(
                processor._write_run_report.call_args.kwargs["processing_status"],
                "completed",
            )
            self.assertEqual(
                processor._write_run_report.call_args.kwargs["cleanup_status"],
                "completed",
            )

    def test_cleanup_failure_does_not_replace_processing_exception(self) -> None:
        with TemporaryDirectory() as temporary:
            processor = self._processor(Path(temporary))

            def shutdown(
                *,
                context: str,
                raise_on_failure: bool = False,
                preserve_sleeping_paddle: bool = False,
                ocr_service: str = "ocr",
            ) -> None:
                if context == "terminal cleanup":
                    raise RuntimeError("synthetic cleanup failure")

            processor._shutdown_managed_runtimes.side_effect = shutdown
            processor._render_all = mock.Mock(
                side_effect=RuntimeError("synthetic processing failure")
            )
            processor._render_executor = SimpleNamespace(
                shutdown=mock.Mock(side_effect=RuntimeError("synthetic render cleanup failure"))
            )
            with mock.patch(
                "pipeline.stage_batched_processor.open_project_stage_checkpoint_store",
                return_value=None,
            ), self.assertRaisesRegex(RuntimeError, "synthetic processing failure"):
                processor.batch_process(["chapter-a.png"])

            self.assertEqual(
                processor._write_run_report.call_args.kwargs["processing_status"],
                "failed",
            )
            self.assertEqual(
                processor._write_run_report.call_args.kwargs["cleanup_status"],
                "failed",
            )
            self.assertEqual(
                processor._write_run_report.call_args.kwargs["processing_error"].args,
                ("synthetic processing failure",),
            )
            self.assertIn(
                "shutdown_render_executor",
                [
                    item["step"]
                    for item in processor._write_run_report.call_args.kwargs["cleanup_errors"]
                ],
            )
            self.assertTrue(
                processor._shutdown_managed_runtimes.call_args.kwargs["raise_on_failure"]
            )

    def test_preparation_failures_write_terminal_report(self) -> None:
        for failure_point in (
            "pdf_preflight",
            "load_page_contexts",
            "ensure_stage_policy",
            "checkpoint_store",
        ):
            with self.subTest(failure_point=failure_point), TemporaryDirectory() as temporary:
                processor = self._processor(Path(temporary))
                error = RuntimeError(f"synthetic {failure_point} failure")
                if failure_point == "pdf_preflight":
                    processor.main_page.file_handler.preflight_for_processing = mock.Mock(
                        side_effect=error
                    )
                elif failure_point == "load_page_contexts":
                    processor._load_page_contexts = mock.Mock(side_effect=error)
                elif failure_point == "ensure_stage_policy":
                    processor._ensure_stage_policy = mock.Mock(side_effect=error)

                checkpoint_error = (
                    mock.patch(
                        "pipeline.stage_batched_processor.open_project_stage_checkpoint_store",
                        side_effect=error,
                    )
                    if failure_point == "checkpoint_store"
                    else mock.patch(
                        "pipeline.stage_batched_processor.open_project_stage_checkpoint_store",
                        return_value=None,
                    )
                )
                with checkpoint_error, self.assertRaisesRegex(RuntimeError, failure_point):
                    processor.batch_process(["chapter-a.png"])

                self.assertEqual(processor._write_run_report.call_count, 1)
                self.assertEqual(
                    processor._write_run_report.call_args.kwargs["processing_status"],
                    "failed",
                )
                self.assertIsNone(
                    processor._write_run_report.call_args.kwargs["output_summary"]["output_count"]
                )

    def test_series_queue_two_children_save_pixels_and_done_states(self) -> None:
        class SeriesMain(QtCore.QObject):
            def __init__(self) -> None:
                super().__init__()
                self.pipeline_status_panel = SimpleNamespace(
                    set_series_queue_pause_visible=mock.Mock()
                )
                self.project_ctrl = SimpleNamespace(save_current_state=mock.Mock())

        with TemporaryDirectory() as temporary:
            output_root = Path(temporary)
            series_file = output_root / "example.seriesctpr"
            items = [
                {
                    "series_item_id": f"item-{index}",
                    "queue_index": index,
                    "display_name": f"chapter-{index}.ctpr",
                    "source_kind": "ctpr_import",
                    "source_origin_path": str(output_root / f"chapter-{index}.ctpr"),
                    "source_origin_relpath": f"chapter-{index}.ctpr",
                    "imported_at": "2026-01-01T00:00:00",
                    "updated_at": "2026-01-01T00:00:00",
                    "status": "pending",
                    "embedded_project_blob_hash": f"hash-{index}",
                    "child_page_count": 1,
                }
                for index in (1, 2)
            ]
            create_series_project(
                str(series_file),
                root_dir=str(output_root),
                items=items,
                embedded_projects=[
                    {
                        "project_hash": f"hash-{index}",
                        "display_name": f"chapter-{index}.ctpr",
                        "project_size": 7,
                        "project_blob": b"project",
                    }
                    for index in (1, 2)
                ],
            )
            controller = SeriesController(SeriesMain())
            state = load_series_project(str(series_file))
            controller.series_file = str(series_file)
            controller.series_manifest = dict(state["manifest"])
            controller.series_items = list(state["items"])
            controller._queue_active = True
            controller._queue_pending_ids = ["item-1", "item-2"]
            controller._queue_completed_ids = []
            controller._queue_failed_ids = []
            controller._queue_skipped_ids = []
            controller._queue_retry_remaining = {}
            controller._apply_workspace_state = mock.Mock()
            controller._show_board = mock.Mock()
            controller._set_series_window_title = mock.Mock()
            processor = self._processor(output_root)

            def render(pages: list[StagePageContext]) -> None:
                if processor._render_cancel_event.is_set():
                    raise OperationCancelledError("A retired render token was reused")
                for page in pages:
                    target = output_root / f"{Path(page.image_path).stem}-translated.png"
                    value = 35 if "chapter-1" in page.image_path else 225
                    Image.new("RGB", (8, 8), (value, 0, 0)).save(target)
                    page.output_path = str(target)

            processor._render_all = render

            def open_child(item_id: str, *, push_history: bool, after_loaded) -> None:
                self.assertFalse(push_history)
                controller.active_child_item_id = item_id
                controller.active_child_project_path = str(output_root / f"{item_id}.ctpr")
                after_loaded()

            def process_child() -> None:
                item_id = str(controller.active_child_item_id)
                processor.batch_process([f"chapter-{item_id[-1]}.png"])
                controller.on_batch_process_finished(was_cancelled=False, failed=False)

            controller._open_item = open_child
            controller._start_batch_for_active_child = process_child

            def save_child(_main, child_path: str, **_kwargs) -> None:
                index = int(Path(child_path).stem[-1])
                output = output_root / f"chapter-{index}-translated.png"
                Path(child_path).write_bytes(output.read_bytes())

            with (
                mock.patch(
                    "pipeline.stage_batched_processor.open_project_stage_checkpoint_store",
                    return_value=None,
                ),
                mock.patch(
                    "app.controllers.series.QtCore.QTimer.singleShot",
                    side_effect=lambda _delay, _receiver, callback: callback(),
                ),
                mock.patch("app.controllers.series.save_state_to_proj_file", save_child),
            ):
                controller._run_next_queue_item()

            reloaded = load_series_project(str(series_file))
            self.assertEqual([item["status"] for item in reloaded["items"]], ["done", "done"])
            self.assertEqual(reloaded["manifest"]["series_queue_runtime"]["queue_state"], "idle")
            self.assertEqual(processor._write_run_report.call_count, 2)
            for index, value in ((1, 35), (2, 225)):
                with Image.open(output_root / f"chapter-{index}-translated.png") as image:
                    self.assertEqual(image.getpixel((0, 0)), (value, 0, 0))
                item = reloaded["items"][index - 1]
                blob = load_series_project_blob(
                    str(series_file), item["embedded_project_blob_hash"]
                )
                with Image.open(BytesIO(blob)) as embedded:
                    self.assertEqual(embedded.getpixel((0, 0)), (value, 0, 0))

    def test_cancelled_run_does_not_poison_the_next_run(self) -> None:
        with TemporaryDirectory() as temporary:
            output_root = Path(temporary)
            processor = self._processor(output_root)
            processor._test_cancelled["value"] = True

            def render(pages: list[StagePageContext]) -> None:
                self.assertFalse(processor._render_cancel_event.is_set())
                target = output_root / "retry-translated.png"
                Image.new("RGB", (8, 8), (83, 0, 0)).save(target)
                pages[0].output_path = str(target)

            processor._render_all = render
            with mock.patch(
                "pipeline.stage_batched_processor.open_project_stage_checkpoint_store",
                return_value=None,
            ):
                processor.batch_process(["cancelled.png"])
                processor._test_cancelled["value"] = False
                processor.batch_process(["retry.png"])

            with Image.open(output_root / "retry-translated.png") as image:
                self.assertEqual(image.getpixel((0, 0)), (83, 0, 0))
            self.assertEqual(processor._write_run_report.call_count, 2)

    def test_failed_run_can_retry_with_a_new_render_token(self) -> None:
        with TemporaryDirectory() as temporary:
            output_root = Path(temporary)
            processor = self._processor(output_root)
            attempts = 0

            def render(pages: list[StagePageContext]) -> None:
                nonlocal attempts
                attempts += 1
                self.assertFalse(processor._render_cancel_event.is_set())
                if attempts == 1:
                    raise RuntimeError("synthetic render failure")
                target = output_root / "retry-translated.png"
                Image.new("RGB", (8, 8), (97, 0, 0)).save(target)
                pages[0].output_path = str(target)

            processor._render_all = render
            with mock.patch(
                "pipeline.stage_batched_processor.open_project_stage_checkpoint_store",
                return_value=None,
            ):
                with self.assertRaisesRegex(RuntimeError, "synthetic render failure"):
                    processor.batch_process(["first.png"])
                processor.batch_process(["retry.png"])

            self.assertEqual(attempts, 2)
            with Image.open(output_root / "retry-translated.png") as image:
                self.assertEqual(image.getpixel((0, 0)), (97, 0, 0))

    def test_unrequested_render_cancellation_is_a_failure(self) -> None:
        with TemporaryDirectory() as temporary:
            processor = self._processor(Path(temporary))
            processor._render_all = mock.Mock(
                side_effect=OperationCancelledError("orphaned render token")
            )
            with mock.patch(
                "pipeline.stage_batched_processor.open_project_stage_checkpoint_store",
                return_value=None,
            ), self.assertRaisesRegex(RuntimeError, "orphaned render token"):
                processor.batch_process(["chapter-a.png"])

            tags = [call.args[0] for call in processor._emit_benchmark_event.call_args_list]
            self.assertIn("batch_run_failed", tags)
            self.assertNotIn("batch_run_cancelled", tags)

    def test_checkpoint_only_run_followed_by_render_miss_uses_fresh_token(self) -> None:
        with TemporaryDirectory() as temporary:
            output_root = Path(temporary)
            processor = self._processor(output_root)
            cached = output_root / "cached-translated.png"
            Image.new("RGB", (8, 8), (17, 0, 0)).save(cached)

            def render(pages: list[StagePageContext]) -> None:
                page = pages[0]
                if page.image_path == "cached.png":
                    # No render job was submitted: the checkpoint materialized
                    # a pre-existing, valid output on the first run.
                    page.output_path = str(cached)
                    return
                if processor._render_cancel_event.is_set():
                    raise OperationCancelledError("retired token after checkpoint hit")
                target = output_root / "fresh-translated.png"
                Image.new("RGB", (8, 8), (117, 0, 0)).save(target)
                page.output_path = str(target)

            processor._render_all = render
            with mock.patch(
                "pipeline.stage_batched_processor.open_project_stage_checkpoint_store",
                return_value=None,
            ):
                processor.batch_process(["cached.png"])
                processor.batch_process(["fresh.png"])

            with Image.open(output_root / "fresh-translated.png") as image:
                self.assertEqual(image.getpixel((0, 0)), (117, 0, 0))
            self.assertEqual(processor._write_run_report.call_count, 2)

    def test_checkpoint_hit_then_real_qt_pool_job_keeps_new_token(self) -> None:
        with TemporaryDirectory() as temporary:
            output_root = Path(temporary)
            processor = self._processor(output_root)
            cached = output_root / "cached-translated.png"
            Image.new("RGB", (8, 8), (27, 0, 0)).save(cached)
            processor.main_page.curr_img_idx = -1
            processor.main_page.button_to_alignment = {1: QtCore.Qt.AlignmentFlag.AlignCenter}
            processor.main_page.button_to_vertical_alignment = {1: "center"}
            processor._lazy_render_context = lambda _ctx: (SimpleNamespace(alignment_id=1), "ko")
            processor._set_current_image = mock.Mock()
            processor._write_json_exports = mock.Mock()
            processor._ensure_page_state = lambda _path: {"viewer_state": {}}
            processor._restore_render_project_state = mock.Mock()
            processor._finish_render_checkpoint_hit = lambda ctx, **_kwargs: setattr(
                ctx, "output_path", str(cached)
            )
            processor._reserve_render_output_path = lambda ctx, **_kwargs: (
                str(output_root / f"{Path(ctx.image_path).stem}-translated.png"),
                str(output_root),
                "png",
            )
            processor._render_worker_count = lambda: 1
            processor._record_performance_detail = mock.Mock()

            def finish_render(pending, *, result, exc) -> None:
                if exc is not None:
                    raise exc
                pending.ctx.output_path = result.final_output_path

            def prepare_checkpoint(ctx, **_kwargs):
                if ctx.image_path == "cached.png":
                    return SimpleNamespace(
                        output_root=str(output_root), output_exists=True
                    ), str(output_root)
                return None, str(output_root)

            def inpaint_and_submit(pages: list[StagePageContext]) -> None:
                for index, page in enumerate(pages):
                    processor._submit_or_inline_render(
                        page, index=index, total_images=len(pages), export_settings={}
                    )

            processor._finish_render_page_bookkeeping = finish_render
            processor._prepare_render_checkpoint = prepare_checkpoint
            processor._inpaint_all = inpaint_and_submit
            processor._render_all = lambda pages: processor._drain_render_futures(block=True)
            seen_tokens: list[bool] = []

            def render_job(job):
                seen_tokens.append(bool(job.is_cancelled()))
                if job.is_cancelled():
                    raise OperationCancelledError("retired token reached Qt pool")
                Image.new("RGB", (8, 8), (127, 0, 0)).save(job.output_path)
                return SimpleNamespace(final_output_path=job.output_path)

            with (
                mock.patch(
                    "pipeline.stage_batched_processor.open_project_stage_checkpoint_store",
                    return_value=None,
                ),
                mock.patch(
                    "pipeline.stage_batched_processor.materialize_render_checkpoint_output",
                    return_value=str(cached),
                ),
                mock.patch("pipeline.stage_batched_processor.run_render_job", render_job),
            ):
                processor.batch_process(["cached.png"])
                retired = processor._render_cancel_event
                processor.batch_process(["fresh.png"])

            self.assertTrue(retired.is_set())
            self.assertIsNot(retired, processor._render_cancel_event)
            self.assertEqual(seen_tokens, [False])
            with Image.open(output_root / "fresh-translated.png") as image:
                self.assertEqual(image.getpixel((0, 0)), (127, 0, 0))


if __name__ == "__main__":
    unittest.main()

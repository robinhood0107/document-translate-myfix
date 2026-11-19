from __future__ import annotations

import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest import mock

import numpy as np

from app.projects.stage_checkpoints import decoded_image_sha256
from modules.utils.textblock import TextBlock
from pipeline.stage_batched_processor import (
    StagePageContext,
    StageBatchedProcessor,
    _approximate_buffer_bytes,
)


PAGE_SHAPE = (2160, 3840, 3)
PAGE_BYTES = PAGE_SHAPE[0] * PAGE_SHAPE[1] * PAGE_SHAPE[2]
MIB = 1024 * 1024


def _loaded_page() -> StagePageContext:
    """스테이지 배치 파이프라인이 렌더 직전에 들고 있는 상태 그대로."""

    ctx = StagePageContext(
        image_path="page.png",
        image_name="page.png",
        source_lang="Japanese",
        target_lang="Korean",
    )
    ctx.image = np.zeros(PAGE_SHAPE, dtype=np.uint8)
    ctx.inpaint_input_img = np.zeros(PAGE_SHAPE, dtype=np.uint8)
    ctx.raw_mask = np.zeros(PAGE_SHAPE[:2], dtype=np.uint8)
    ctx.mask = np.zeros(PAGE_SHAPE[:2], dtype=np.uint8)
    ctx.mask[:100, :100] = 255
    ctx.mask_details = {"raw_mask": ctx.raw_mask, "final_mask": ctx.mask}
    ctx.patches = [{"bbox": (0, 0, 64, 64), "cleaned": np.zeros((64, 64, 3), np.uint8)}]
    return ctx


class PageBufferReleaseTests(unittest.TestCase):
    def test_release_source_image_preserves_other_stage_data(self) -> None:
        ctx = _loaded_page()
        ctx.source_decoded_sha256 = "a" * 64
        ctx.project_checkpoint_page_key = "page:00000001"
        ctx.blk_list = [object()]
        ctx.precomputed_mask_details = {"final_mask": ctx.mask.copy()}
        expected = ctx.image.nbytes
        blocks = ctx.blk_list
        mask_details = ctx.precomputed_mask_details

        released = ctx.release_source_image()

        self.assertEqual(released, expected)
        self.assertIsNone(ctx.image)
        self.assertIsNotNone(ctx.inpaint_input_img)
        self.assertIsNotNone(ctx.mask)
        self.assertIs(ctx.blk_list, blocks)
        self.assertIs(ctx.precomputed_mask_details, mask_details)
        self.assertEqual(ctx.source_decoded_sha256, "a" * 64)
        self.assertEqual(ctx.project_checkpoint_page_key, "page:00000001")
        self.assertEqual(ctx.released_buffer_bytes, expected)
        self.assertEqual(ctx.release_source_image(), 0)

    def test_release_frees_every_full_resolution_array(self) -> None:
        ctx = _loaded_page()
        ctx.paddleocr_cache_plan = SimpleNamespace(
            jobs=[{"prepared_crop": np.zeros((16, 16, 3), dtype=np.uint8)}]
        )
        ctx.paddleocr_cache_engine = object()

        ctx.release_page_buffers()

        self.assertIsNone(ctx.image)
        self.assertIsNone(ctx.inpaint_input_img)
        self.assertIsNone(ctx.raw_mask)
        self.assertIsNone(ctx.mask)
        self.assertEqual(ctx.patches, [])
        self.assertIsNone(ctx.paddleocr_cache_plan)
        self.assertIsNone(ctx.paddleocr_cache_engine)
        # mask_details 는 같은 마스크를 다시 참조한다. 비우지 않으면 위에서
        # 놓아준 배열이 그대로 살아남는다.
        self.assertEqual(ctx.mask_details, {})

    def test_release_reports_the_bytes_it_freed(self) -> None:
        ctx = _loaded_page()

        released = ctx.release_page_buffers()

        # 원본 + 인페인팅 결과(각 23.7 MiB)에 마스크 두 장(각 7.9 MiB)만 해도
        # 페이지당 60 MiB 를 넘는다. 이게 배치 내내 쌓이던 값이다.
        self.assertGreater(released, 60 * MIB)
        self.assertEqual(ctx.released_buffer_bytes, released)

    def test_mask_pixel_count_survives_the_release(self) -> None:
        # 스윕이 끝난 뒤의 집계 때문에 이미지를 붙들고 있을 이유는 없다.
        ctx = _loaded_page()
        expected = int(np.count_nonzero(ctx.mask))

        ctx.release_page_buffers()

        self.assertEqual(expected, 100 * 100)
        self.assertEqual(ctx.mask_pixel_count, expected)

    def test_releasing_twice_is_harmless(self) -> None:
        ctx = _loaded_page()

        first = ctx.release_page_buffers()
        second = ctx.release_page_buffers()

        self.assertGreater(first, 0)
        self.assertEqual(second, 0)
        self.assertEqual(ctx.released_buffer_bytes, first)

    def test_release_on_a_page_that_never_loaded_is_harmless(self) -> None:
        ctx = StagePageContext(
            image_path="page.png",
            image_name="page.png",
            source_lang="Japanese",
            target_lang="Korean",
        )

        self.assertEqual(ctx.release_page_buffers(), 0)
        self.assertEqual(ctx.mask_pixel_count, 0)


class BufferSizeEstimateTests(unittest.TestCase):
    def test_arrays_nested_in_containers_are_counted(self) -> None:
        array = np.zeros((100, 100), dtype=np.uint8)
        self.assertEqual(
            _approximate_buffer_bytes({"a": [array], "b": (array,)}),
            2 * array.nbytes,
        )

    def test_a_cycle_terminates(self) -> None:
        payload: dict[str, object] = {"array": np.zeros((10, 10), np.uint8)}
        payload["self"] = payload

        self.assertEqual(_approximate_buffer_bytes(payload), 100)

    def test_objects_holding_arrays_as_attributes_are_counted(self) -> None:
        class CheckpointHit:
            def __init__(self) -> None:
                self.cleaned_image = np.zeros((50, 50, 3), np.uint8)

        hit = CheckpointHit()
        self.assertEqual(_approximate_buffer_bytes(hit), hit.cleaned_image.nbytes)

    def test_plain_scalars_contribute_nothing(self) -> None:
        self.assertEqual(_approximate_buffer_bytes({"n": 5, "s": "x" * 1000}), 0)

    def test_stage_memory_telemetry_is_observational_only(self) -> None:
        processor = StageBatchedProcessor.__new__(StageBatchedProcessor)
        processor._released_page_buffer_bytes = 123
        processor._record_performance_workload = mock.Mock()
        page = StagePageContext(
            image_path="page.png",
            image_name="page.png",
            source_lang="Japanese",
            target_lang="Korean",
            image=np.zeros((10, 20, 3), dtype=np.uint8),
        )
        memory = mock.Mock(available=4_000_000_000)
        process = mock.Mock()
        process.memory_info.return_value = mock.Mock(rss=1_000_000_000)

        with mock.patch(
            "pipeline.stage_batched_processor.psutil.Process",
            return_value=process,
        ), mock.patch(
            "pipeline.stage_batched_processor.psutil.virtual_memory",
            return_value=memory,
        ):
            processor._record_stage_memory_telemetry("detect", [page])

        processor._record_performance_workload.assert_called_once_with(
            "detect",
            process_rss_bytes=1_000_000_000,
            process_rss_bytes_source="psutil.Process.memory_info.rss",
            process_rss_bytes_unit="bytes",
            process_rss_bytes_status="observed",
            process_rss_bytes_reason_code=None,
            available_ram_bytes=4_000_000_000,
            available_ram_bytes_source="psutil.virtual_memory.available",
            available_ram_bytes_unit="bytes",
            available_ram_bytes_status="observed",
            available_ram_bytes_reason_code=None,
            full_resolution_page_buffer_count=1,
            estimated_page_buffer_bytes=page.image.nbytes,
            released_page_buffer_bytes=123,
        )
        self.assertIsNotNone(page.image)

    def test_stage_memory_probe_results_reach_the_run_report_independently(
        self,
    ) -> None:
        from app.controllers.batch_report import BatchReportController
        from pipeline.performance_telemetry import PipelinePerformanceTelemetry

        observations = (
            ("process_rss_bytes", OSError("private detail"), 0, None),
            ("available_ram_bytes", None, OSError("private detail"), 1_000_000_000),
            (None, None, 0, 1_000_000_000),
        )
        for failed_metric, process_error, memory_error, expected_rss in observations:
            with self.subTest(failed_metric=failed_metric):
                processor = StageBatchedProcessor.__new__(StageBatchedProcessor)
                processor._released_page_buffer_bytes = 123
                processor.performance_telemetry = PipelinePerformanceTelemetry(
                    resource_sampling_enabled=False,
                )
                report_ctrl = BatchReportController.__new__(BatchReportController)
                report_ctrl._current_batch_report = {"run_report_id": "example-run"}
                processor.main_page = SimpleNamespace(batch_report_ctrl=report_ctrl)
                processor._click_to_now_seconds = mock.Mock(return_value=1.0)
                processor._page_outcomes = mock.Mock(return_value=[])
                processor._run_started_wall_text = mock.Mock(return_value="")
                page = StagePageContext(
                    image_path="page.png",
                    image_name="page.png",
                    source_lang="Japanese",
                    target_lang="Korean",
                    image=np.zeros((10, 20, 3), dtype=np.uint8),
                )
                process = mock.Mock()
                process.memory_info.return_value = mock.Mock(rss=1_000_000_000)
                process_probe = (
                    mock.Mock(side_effect=process_error)
                    if process_error
                    else mock.Mock(return_value=process)
                )
                memory_probe = (
                    mock.Mock(side_effect=memory_error)
                    if isinstance(memory_error, Exception)
                    else mock.Mock(return_value=mock.Mock(available=memory_error))
                )

                with mock.patch(
                    "pipeline.stage_batched_processor.psutil.Process",
                    process_probe,
                ), mock.patch(
                    "pipeline.stage_batched_processor.psutil.virtual_memory",
                    memory_probe,
                ):
                    processor._record_stage_memory_telemetry("detect", [page])

                self.assertEqual(
                    processor._write_run_report(
                        [page],
                        output_summary={"input_count": 1, "output_count": 1},
                        processing_status="completed",
                    ),
                    "",
                )
                report = report_ctrl._current_batch_report["pipeline_run_report_data"]
                workload = report["telemetry"]["value"]["workload"]["detect"]

                if failed_metric == "process_rss_bytes":
                    self.assertIsNone(workload["process_rss_bytes"])
                    self.assertEqual(
                        workload["process_rss_bytes_status"], "unavailable"
                    )
                    self.assertEqual(
                        workload["available_ram_bytes"], 0
                    )
                    self.assertEqual(
                        workload["available_ram_bytes_status"], "observed"
                    )
                elif failed_metric == "available_ram_bytes":
                    self.assertEqual(workload["process_rss_bytes"], expected_rss)
                    self.assertEqual(
                        workload["process_rss_bytes_status"], "observed"
                    )
                    self.assertIsNone(workload["available_ram_bytes"])
                    self.assertEqual(
                        workload["available_ram_bytes_status"], "unavailable"
                    )
                else:
                    self.assertEqual(workload["process_rss_bytes"], expected_rss)
                    self.assertEqual(workload["available_ram_bytes"], memory_error)
                    self.assertEqual(
                        workload["process_rss_bytes_status"], "observed"
                    )
                    self.assertEqual(
                        workload["available_ram_bytes_status"], "observed"
                    )

                for metric in ("process_rss_bytes", "available_ram_bytes"):
                    self.assertEqual(workload[f"{metric}_unit"], "bytes")
                    self.assertTrue(workload[f"{metric}_source"].startswith("psutil."))
                    if workload[f"{metric}_status"] == "unavailable":
                        self.assertIsNone(workload[metric])
                        self.assertEqual(
                            workload[f"{metric}_reason_code"], "probe_failed"
                        )
                    else:
                        self.assertIsNone(workload[f"{metric}_reason_code"])
                self.assertNotIn("private detail", str(workload))
                self.assertEqual(workload["full_resolution_page_buffer_count"], 1)
                self.assertEqual(workload["estimated_page_buffer_bytes"], page.image.nbytes)
                self.assertEqual(workload["released_page_buffer_bytes"], 123)


class SourceReloadTests(unittest.TestCase):
    def _processor(self, loaded_image):
        processor = StageBatchedProcessor.__new__(StageBatchedProcessor)
        processor.main_page = SimpleNamespace(
            image_ctrl=SimpleNamespace(
                load_image=mock.Mock(return_value=loaded_image),
            ),
        )
        processor._released_page_buffer_bytes = 0
        return processor

    def test_reload_accepts_only_the_original_decoded_image(self) -> None:
        source = np.arange(12 * 16 * 3, dtype=np.uint8).reshape((12, 16, 3))
        original = source.copy()
        reloaded = source.copy()
        block = TextBlock(
            text_bbox=np.array([1, 2, 8, 9], dtype=np.int32),
            text_class="text_bubble",
            block_id="example-block",
        )
        ctx = StagePageContext(
            image_path="example_source.png",
            image_name="example_source.png",
            source_lang="Japanese",
            target_lang="Korean",
            source_decoded_sha256=decoded_image_sha256(source),
            blk_list=[block],
        )
        processor = self._processor(reloaded)

        image = processor._ensure_source_image(ctx)

        self.assertIs(image, reloaded)
        np.testing.assert_array_equal(image, original)
        np.testing.assert_array_equal(source, original)
        self.assertEqual(tuple(image.shape), (12, 16, 3))
        self.assertEqual(ctx.blk_list[0].xyxy.tolist(), [1, 2, 8, 9])

    def test_reload_fails_closed_when_source_is_missing_or_changed(self) -> None:
        source = np.arange(12 * 16 * 3, dtype=np.uint8).reshape((12, 16, 3))
        changed = source.copy()
        changed[0, 0, 0] ^= 1
        changed_ctx = StagePageContext(
            image_path="example_source.png",
            image_name="example_source.png",
            source_lang="Japanese",
            target_lang="Korean",
            source_decoded_sha256=decoded_image_sha256(source),
        )

        with self.assertRaisesRegex(RuntimeError, "source_changed_after_detect"):
            self._processor(changed)._ensure_source_image(changed_ctx)
        self.assertIsNone(changed_ctx.image)

        processor = self._processor(None)
        missing_ctx = StagePageContext(
            image_path="example_source.png",
            image_name="example_source.png",
            source_lang="Japanese",
            target_lang="Korean",
            source_decoded_sha256=decoded_image_sha256(source),
        )
        with mock.patch(
            "pipeline.stage_batched_processor.ensure_path_materialized"
        ) as materialize, mock.patch(
            "pipeline.stage_batched_processor.imk.read_image",
            return_value=None,
        ):
            with self.assertRaisesRegex(RuntimeError, "source_reload_failed"):
                processor._ensure_source_image(missing_ctx)
        materialize.assert_called_once_with("example_source.png")
        self.assertIsNone(missing_ctx.image)

    def test_reload_requires_a_detection_identity(self) -> None:
        processor = self._processor(np.zeros((12, 16, 3), dtype=np.uint8))
        ctx = StagePageContext(
            image_path="example_source.png",
            image_name="example_source.png",
            source_lang="Japanese",
            target_lang="Korean",
        )

        with self.assertRaisesRegex(
            RuntimeError,
            "source_identity_missing_after_detect",
        ):
            processor._ensure_source_image(ctx)


class SyntheticDetectLifetimeTests(unittest.TestCase):
    def test_detect_releases_each_source_without_clearing_geometry(self) -> None:
        paths = [f"synthetic-{index}.png" for index in range(3)]
        images = {
            path: np.full((16, 24, 3), index, dtype=np.uint8)
            for index, path in enumerate(paths)
        }
        pages = [
            StagePageContext(
                image_path=path,
                image_name=path,
                source_lang="Japanese",
                target_lang="Korean",
            )
            for path in paths
        ]

        class Detector:
            detector = "RT-DETR-v2"
            last_engine_name = "synthetic"
            last_device = "cpu"

            def __init__(self) -> None:
                self.maximum_context_images = 0
                self.last_mask_details = {
                    "final_mask": np.zeros((16, 24), dtype=np.uint8)
                }

            def detect(self, _image):
                resident = sum(page.image is not None for page in pages)
                self.maximum_context_images = max(
                    self.maximum_context_images,
                    resident,
                )
                return [
                    TextBlock(
                        text_bbox=np.array([2, 2, 12, 12], dtype=np.int32),
                        text_class="text_bubble",
                        block_id="example-block",
                    )
                ]

        detector = Detector()
        processor = StageBatchedProcessor.__new__(StageBatchedProcessor)
        processor.main_page = SimpleNamespace(
            image_files=paths,
            lang_mapping={"Japanese": "Japanese"},
            settings_page=SimpleNamespace(
                get_tool_selection=lambda tool: {
                    "detector": "RT-DETR-v2",
                    "inpainter": "LaMa",
                }.get(tool, ""),
                is_gpu_enabled=lambda: False,
            ),
            image_ctrl=SimpleNamespace(
                load_image=lambda path: images[path],
                update_processing_summary=mock.Mock(),
            ),
        )
        processor.block_detection = SimpleNamespace(block_detector_cache=detector)
        processor._project_checkpoint_store = None
        processor._released_page_buffer_bytes = 0
        processor._record_performance_workload = mock.Mock()
        processor._measure_performance = lambda **_kwargs: nullcontext()
        processor._raise_if_cancelled = mock.Mock()
        processor._set_current_image = mock.Mock()
        processor.emit_progress = mock.Mock()
        processor._start_page_summary = mock.Mock()
        processor._log_page_start = mock.Mock()
        processor._emit_benchmark_event = mock.Mock()
        processor._persist_detect_state = mock.Mock()
        processor._effective_export_settings = mock.Mock(return_value={})
        processor._write_detector_overlay_debug_image = mock.Mock(return_value="")
        processor._maybe_emit_preview_image = mock.Mock()

        with mock.patch(
            "pipeline.stage_batched_processor.get_best_render_area",
            return_value=None,
        ), mock.patch(
            "pipeline.stage_batched_processor.sort_blk_list",
            side_effect=lambda blocks, _rtl: list(blocks),
        ):
            processor._detect_all(pages)

        self.assertEqual(detector.maximum_context_images, 1)
        self.assertTrue(all(page.image is None for page in pages))
        self.assertEqual(
            [page.source_decoded_sha256 for page in pages],
            [decoded_image_sha256(images[path]) for path in paths],
        )
        self.assertEqual(
            processor._released_page_buffer_bytes,
            sum(image.nbytes for image in images.values()),
        )
        for page in pages:
            self.assertEqual(page.blk_list[0].xyxy.tolist(), [2, 2, 12, 12])
            self.assertIs(page.precomputed_mask_details, detector.last_mask_details)


class RenderCheckpointBufferLifetimeTests(unittest.TestCase):
    def test_submit_checkpoint_hit_releases_buffers_after_bookkeeping(self) -> None:
        processor = StageBatchedProcessor.__new__(StageBatchedProcessor)
        processor._released_page_buffer_bytes = 0
        processor.main_page = SimpleNamespace(
            image_ctrl=SimpleNamespace(
                update_processing_summary=mock.Mock(),
                mark_processing_stage=mock.Mock(),
            ),
            render_state_ready=mock.Mock(),
        )
        page_state = {"viewer_state": {}}
        processor._ensure_page_state = lambda _path: page_state
        processor._lazy_render_context = lambda _ctx: (
            SimpleNamespace(alignment_id=1),
            "ko",
        )
        processor._set_current_image = mock.Mock()
        processor._write_json_exports = mock.Mock()
        processor._emit_benchmark_event = mock.Mock()
        processor._log_page_done = mock.Mock()
        processor.emit_progress = mock.Mock()
        processor._raise_if_cancelled = mock.Mock()
        block = TextBlock(
            text_bbox=np.array([1, 2, 8, 9], dtype=np.int32),
            text_class="text_bubble",
            block_id="example-block",
        )
        hit = SimpleNamespace(
            output_root="example-output-root",
            output_exists=True,
        )
        ctx = StagePageContext(
            image_path="example_source_page.png",
            image_name="example_source_page.png",
            source_lang="Japanese",
            target_lang="Korean",
            image=np.zeros((12, 16, 3), dtype=np.uint8),
            inpaint_input_img=np.ones((12, 16, 3), dtype=np.uint8),
            raw_mask=np.zeros((12, 16), dtype=np.uint8),
            mask=np.full((12, 16), 255, dtype=np.uint8),
            mask_details={"final_mask": np.full((12, 16), 255, dtype=np.uint8)},
            patches=[{"cleaned": np.zeros((2, 2, 3), dtype=np.uint8)}],
            project_render_blocks=[block],
            project_viewer_state={"text_items_state": [{"text": "example translated"}]},
            project_render_identity={"runtime": "synthetic"},
            project_render_fingerprint="synthetic-fingerprint",
        )

        def prepare_checkpoint(page, **_kwargs):
            page.project_render_checkpoint_status = "hit"
            return hit, hit.output_root

        processor._prepare_render_checkpoint = prepare_checkpoint

        with mock.patch(
            "pipeline.stage_batched_processor.materialize_render_checkpoint_output",
            return_value="example_rendered_output.png",
        ):
            processor._submit_or_inline_render(
                ctx,
                index=0,
                total_images=1,
                export_settings={},
            )

        self.assertEqual(ctx.output_path, "example_rendered_output.png")
        self.assertEqual(ctx.project_render_checkpoint_status, "hit")
        self.assertEqual(ctx.project_render_fingerprint, "synthetic-fingerprint")
        self.assertEqual(ctx.blk_list[0].xyxy.tolist(), [1, 2, 8, 9])
        self.assertEqual(page_state["viewer_state"]["text_items_state"], [{"text": "example translated"}])
        summary = processor.main_page.image_ctrl.update_processing_summary.call_args.args[1]
        self.assertEqual(summary["render_project_checkpoint_status"], "hit")
        self.assertIsNone(ctx.image)
        self.assertIsNone(ctx.inpaint_input_img)
        self.assertEqual(ctx.mask_pixel_count, 12 * 16)
        self.assertIsNone(ctx.raw_mask)
        self.assertIsNone(ctx.mask)
        self.assertEqual(ctx.mask_details, {})
        self.assertEqual(ctx.patches, [])
        self.assertGreater(processor._released_page_buffer_bytes, 0)

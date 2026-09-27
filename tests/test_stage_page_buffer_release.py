from __future__ import annotations

import unittest
from concurrent.futures import Future
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
        expected = ctx.image.nbytes

        released = ctx.release_source_image()

        self.assertEqual(released, expected)
        self.assertIsNone(ctx.image)
        self.assertIsNotNone(ctx.inpaint_input_img)
        self.assertIsNotNone(ctx.mask)
        self.assertEqual(ctx.source_decoded_sha256, "a" * 64)
        self.assertEqual(ctx.released_buffer_bytes, expected)

    def test_release_frees_every_full_resolution_array(self) -> None:
        ctx = _loaded_page()

        ctx.release_page_buffers()

        self.assertIsNone(ctx.image)
        self.assertIsNone(ctx.inpaint_input_img)
        self.assertIsNone(ctx.raw_mask)
        self.assertIsNone(ctx.mask)
        self.assertEqual(ctx.patches, [])
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
            available_ram_bytes=4_000_000_000,
            full_resolution_page_buffer_count=1,
            estimated_page_buffer_bytes=page.image.nbytes,
            released_page_buffer_bytes=123,
        )
        self.assertIsNotNone(page.image)


class SourceReloadTests(unittest.TestCase):
    def test_reload_requires_the_same_decoded_source_sha(self) -> None:
        source = np.zeros((12, 16, 3), dtype=np.uint8)
        changed = source.copy()
        changed[0, 0, 0] = 1
        processor = StageBatchedProcessor.__new__(StageBatchedProcessor)
        processor.main_page = mock.Mock()
        processor.main_page.image_ctrl.load_image = mock.Mock(
            side_effect=[source, changed]
        )
        ctx = StagePageContext(
            image_path="example.png",
            image_name="example.png",
            source_lang="Japanese",
            target_lang="Korean",
            source_decoded_sha256=decoded_image_sha256(source),
        )

        self.assertIs(processor._ensure_source_image(ctx), source)
        processor._release_source_image(ctx)
        with self.assertRaisesRegex(RuntimeError, "source_changed_after_detect"):
            processor._ensure_source_image(ctx)
        self.assertIsNone(ctx.image)


class SyntheticDetectLifetimeTests(unittest.TestCase):
    def test_one_and_thirty_two_page_sweeps_release_each_source_rgb(self) -> None:
        for page_count in (1, 32):
            with self.subTest(page_count=page_count):
                processor = StageBatchedProcessor.__new__(StageBatchedProcessor)
                paths = [f"synthetic-{index}.png" for index in range(page_count)]
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
                    last_mask_details = None

                    def __init__(self) -> None:
                        self.maximum_context_images = 0

                    def detect(self, _image):
                        resident = sum(page.image is not None for page in pages)
                        self.maximum_context_images = max(
                            self.maximum_context_images,
                            resident,
                        )
                        return [
                            TextBlock(
                                text_bbox=np.array(
                                    [2, 2, 12, 12],
                                    dtype=np.int32,
                                ),
                                text_class="text_bubble",
                                block_id=f"block-{resident}",
                            )
                        ]

                detector = Detector()
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
                        load_image=lambda _path: np.zeros(
                            (16, 24, 3),
                            dtype=np.uint8,
                        ),
                        update_processing_summary=mock.Mock(),
                    ),
                )
                processor.block_detection = SimpleNamespace(
                    block_detector_cache=detector
                )
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
                processor._effective_export_settings = mock.Mock(
                    return_value={}
                )
                processor._write_detector_overlay_debug_image = mock.Mock(
                    return_value=""
                )
                processor._maybe_emit_preview_image = mock.Mock()

                processor._detect_all(pages)

                self.assertEqual(detector.maximum_context_images, 1)
                self.assertTrue(all(page.image is None for page in pages))
                self.assertEqual(
                    sum(page.source_pixel_count for page in pages),
                    page_count * 16 * 24,
                )
                self.assertEqual(
                    processor._released_page_buffer_bytes,
                    page_count * 16 * 24 * 3,
                )
                process = mock.Mock()
                process.memory_info.return_value = SimpleNamespace(rss=4096)
                with mock.patch(
                    "pipeline.stage_batched_processor.psutil.Process",
                    return_value=process,
                ), mock.patch(
                    "pipeline.stage_batched_processor.psutil.virtual_memory",
                    return_value=SimpleNamespace(available=8192),
                ):
                    processor._record_stage_memory_telemetry("detect", pages)
                telemetry = processor._record_performance_workload.call_args
                self.assertEqual(
                    telemetry.kwargs["full_resolution_page_buffer_count"],
                    0,
                )
                self.assertEqual(
                    telemetry.kwargs["estimated_page_buffer_bytes"],
                    0,
                )
                self.assertEqual(
                    telemetry.kwargs["released_page_buffer_bytes"],
                    page_count * 16 * 24 * 3,
                )


class RenderBackpressureTests(unittest.TestCase):
    def _pending(
        self,
        buffer_bytes: int,
        *,
        completed: bool = True,
    ) -> SimpleNamespace:
        future = Future()
        if completed:
            future.set_result(None)
        return SimpleNamespace(future=future, buffer_bytes=buffer_bytes, ctx=None)

    def _processor(
        self,
        pending: list[SimpleNamespace],
    ) -> StageBatchedProcessor:
        processor = StageBatchedProcessor.__new__(StageBatchedProcessor)
        processor._pending_render_jobs = pending
        processor._raise_if_cancelled = mock.Mock()
        processor._resolve_render_future = mock.Mock()
        processor._drain_render_futures = mock.Mock()
        processor._record_render_pending_telemetry = mock.Mock()
        return processor

    def test_render_pending_never_grows_past_two_jobs(self) -> None:
        pending = [
            self._pending(100),
            self._pending(100, completed=False),
        ]
        processor = self._processor(pending)
        with mock.patch(
            "pipeline.stage_batched_processor.psutil.virtual_memory",
            return_value=mock.Mock(available=2_000_000_000),
        ):
            processor._wait_for_render_capacity(100)

        self.assertEqual(len(pending), 1)
        processor._resolve_render_future.assert_called_once()

    def test_render_pending_bytes_use_current_available_ram(self) -> None:
        pending = [self._pending(300)]
        processor = self._processor(pending)
        with mock.patch(
            "pipeline.stage_batched_processor.psutil.virtual_memory",
            return_value=mock.Mock(available=600),
        ):
            processor._wait_for_render_capacity(1)

        self.assertEqual(pending, [])
        processor._resolve_render_future.assert_called_once()

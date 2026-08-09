"""Phase 3a 렌더 워커(`pipeline/render_worker.py`)의 순수성·취소 검증.

렌더 워커는 순수 함수여야 한다: 여기 있는 모든 테스트는 `main_page`,
`StageBatchedProcessor`, Qt 시그널을 전혀 참조하지 않고 `run_render_job`을
직접 호출한다. 그 자체가 순수성 계약의 증거다.
"""

from __future__ import annotations

import os
import tempfile
import unittest
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PySide6 import QtCore, QtWidgets

import imkit as imk

from modules.rendering.render import TextRenderingSettings
from modules.utils.exceptions import OperationCancelledError
from modules.utils.textblock import TextBlock
from pipeline.render_pool import QtRenderPool
from pipeline.render_worker import RenderJobInput, run_render_job


def _nonuniform_image(height: int = 48, width: int = 72) -> np.ndarray:
    """검은 이미지로는 잡을 수 없는 회귀가 있다.

    렌더 워커가 아무것도 그리지 않고 조용히 성공하면 결과는 통째로 검정이 된다.
    입력이 `np.zeros` 면 그 실패와 정상 동작이 구분되지 않으므로, 픽셀마다
    값이 다른 이미지를 쓴다.
    """
    ys, xs = np.mgrid[0:height, 0:width]
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:, :, 0] = (ys * 5 + 40) % 256
    image[:, :, 1] = (xs * 3 + 90) % 256
    image[:, :, 2] = ((ys + xs) * 7 + 20) % 256
    return np.ascontiguousarray(image)


def _render_settings(**overrides) -> TextRenderingSettings:
    defaults = dict(
        alignment_id=1,
        vertical_alignment_id=1,
        font_family="Arial",
        min_font_size=5,
        max_font_size=40,
        color="#000000",
        force_font_color=False,
        smart_global_apply_all=False,
        upper_case=False,
        outline=False,
        outline_color="#FFFFFF",
        outline_width="0",
        bold=False,
        italic=False,
        underline=False,
        line_spacing="1.0",
        direction=QtCore.Qt.LayoutDirection.LeftToRight,
    )
    defaults.update(overrides)
    return TextRenderingSettings(**defaults)


def _job(tmp_path: str, **overrides) -> RenderJobInput:
    defaults = dict(
        image_path="page.png",
        image=np.zeros((40, 60, 3), dtype=np.uint8),
        inpaint_input_img=np.zeros((40, 60, 3), dtype=np.uint8),
        mask=np.zeros((40, 60), dtype=np.uint8),
        patches=[],
        blk_list=[],
        translation_blocks=[],
        no_text_detected=False,
        trg_lng_cd="ko",
        render_settings=_render_settings(),
        strict_render_symbols=False,
        alignment=QtCore.Qt.AlignmentFlag.AlignCenter,
        vertical_alignment="center",
        viewer_state={},
        output_path=os.path.join(tmp_path, "out.png"),
        output_format="png",
        is_cancelled=lambda: False,
    )
    defaults.update(overrides)
    return RenderJobInput(**defaults)


class RenderWorkerPurityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        # ImageSaveRenderer 가 QGraphicsScene/QPainter 를 쓰므로 QApplication 은
        # 필요하다 — 하지만 main_page 나 그 어떤 GUI 컨트롤러도 필요 없다는 것이
        # 이 파일의 검증 대상이다.
        cls._app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def test_cancelled_before_start_raises_immediately(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            job = _job(tmp_dir, is_cancelled=lambda: True)
            with self.assertRaises(OperationCancelledError):
                run_render_job(job)
            self.assertFalse(os.path.exists(job.output_path))

    def test_empty_block_list_writes_output_with_no_main_page(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            job = _job(tmp_dir)

            result = run_render_job(job)

            self.assertTrue(os.path.exists(job.output_path))
            self.assertEqual(result.final_output_path, job.output_path)
            self.assertEqual(result.viewer_state["text_items_state"], [])
            self.assertEqual(result.blk_rendered_events, [])
            self.assertFalse(result.restore_applied)

    def test_no_text_detected_skips_format_and_area_resolution(self) -> None:
        called = {"format": False, "area": False}

        def fake_format_translations(*_args, **_kwargs):
            called["format"] = True

        def fake_get_best_render_area(*_args, **_kwargs):
            called["area"] = True

        with tempfile.TemporaryDirectory() as tmp_dir, mock.patch(
            "pipeline.render_worker.format_translations",
            side_effect=fake_format_translations,
        ), mock.patch(
            "pipeline.render_worker.get_best_render_area",
            side_effect=fake_get_best_render_area,
        ):
            job = _job(tmp_dir, no_text_detected=True)
            run_render_job(job)

        self.assertFalse(called["format"])
        self.assertFalse(called["area"])

    def test_translate_inpaint_block_produces_text_item_and_render_event(self) -> None:
        block = TextBlock(
            text_bbox=np.array([5, 5, 40, 20], dtype=np.int32),
            text_class="text_bubble",
            text="hello",
            translation="hello",
            block_id="stable",
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            job = _job(
                tmp_dir,
                blk_list=[block],
                translation_blocks=[block],
                trg_lng_cd="ko",
            )

            result = run_render_job(job)

            self.assertTrue(os.path.exists(job.output_path))
            self.assertEqual(len(result.viewer_state["text_items_state"]), 1)
            self.assertEqual(len(result.blk_rendered_events), 1)
            translation, _font_size, rendered_block = result.blk_rendered_events[0]
            self.assertEqual(translation.strip().lower(), "hello")
            self.assertIs(rendered_block, block)

    def test_worker_thread_output_matches_main_thread_pixel_for_pixel(self) -> None:
        """렌더 워커는 Qt 스레드에서 돌아야 한다 (회귀 가드).

        `QGraphicsScene` 은 Qt 이벤트 디스패처가 있는 스레드를 요구한다.
        `concurrent.futures.ThreadPoolExecutor` 가 만드는 plain Python 스레드
        에서는 `addItem()` 의 지연 처리가 끝나지 않아 `scene.render()` 가
        아무것도 그리지 않고 **조용히 성공**하고, 산출물이 통째로 검정이 된다
        (예외도 Qt 경고도 없다). 그래서 결과를 파일 존재 여부가 아니라
        메인 스레드 렌더와의 픽셀 일치로 확인한다.
        """
        image = _nonuniform_image()

        with tempfile.TemporaryDirectory() as tmp_dir:
            inline_job = _job(
                tmp_dir,
                image=image,
                inpaint_input_img=image.copy(),
                mask=np.zeros(image.shape[:2], dtype=np.uint8),
                output_path=os.path.join(tmp_dir, "inline.png"),
            )
            inline_result = run_render_job(inline_job)

            worker_job = _job(
                tmp_dir,
                image=image,
                inpaint_input_img=image.copy(),
                mask=np.zeros(image.shape[:2], dtype=np.uint8),
                output_path=os.path.join(tmp_dir, "worker.png"),
            )
            pool = QtRenderPool(max_workers=1)
            try:
                worker_result = pool.submit(run_render_job, worker_job).result(timeout=120)
            finally:
                pool.shutdown(wait=True)

            inline_pixels = imk.read_image(inline_result.final_output_path)
            worker_pixels = imk.read_image(worker_result.final_output_path)

            # 렌더가 통째로 비어 있으면(검정) 잡는다.
            self.assertGreater(int(inline_pixels.astype(np.int64).sum()), 0)
            self.assertGreater(int(worker_pixels.astype(np.int64).sum()), 0)
            self.assertTrue(
                np.array_equal(inline_pixels, worker_pixels),
                "렌더 워커 산출물이 메인 스레드 산출물과 다르다",
            )

    def test_render_pool_propagates_worker_exception_to_future(self) -> None:
        pool = QtRenderPool(max_workers=1)

        def boom(_job):
            raise RuntimeError("render exploded")

        try:
            future = pool.submit(boom, None)
            with self.assertRaisesRegex(RuntimeError, "render exploded"):
                future.result(timeout=120)
        finally:
            pool.shutdown(wait=True)

    def test_cancellation_mid_run_aborts_before_rasterize(self) -> None:
        calls = {"n": 0}

        def is_cancelled() -> bool:
            # 첫 두 번(시작, format_translations 뒤)은 통과시키고 rasterize
            # 직전에 취소되게 한다.
            calls["n"] += 1
            return calls["n"] >= 3

        with tempfile.TemporaryDirectory() as tmp_dir:
            job = _job(tmp_dir, is_cancelled=is_cancelled)
            with self.assertRaises(OperationCancelledError):
                run_render_job(job)
            self.assertFalse(os.path.exists(job.output_path))


if __name__ == "__main__":
    unittest.main()

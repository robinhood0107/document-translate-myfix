from __future__ import annotations

import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.stage_batched_processor import (  # noqa: E402
    StageBatchedProcessor,
    StagePageContext,
)


def _page(name: str, *, failed: str = "", with_image: bool = True) -> StagePageContext:
    ctx = StagePageContext(
        image_path=f"C:/pages/{name}",
        image_name=name,
        source_lang="Japanese",
        target_lang="Korean",
    )
    if with_image:
        ctx.image = np.zeros((8, 8, 3), dtype=np.uint8)
    if failed:
        ctx.failed_stage = failed
        ctx.failed_reason = f"{failed} blew up"
    return ctx


class _Recorder:
    """폴백이 실제로 파일을 남겼는지 본다."""

    def __init__(self, out_dir: Path) -> None:
        self.out_dir = out_dir
        self.exports: list[tuple[str, int]] = []
        self.summaries: dict[str, dict] = {}
        self.events: list[tuple[str, dict]] = []
        self.preflight_errors: list[tuple[str, str]] = []

    def existing_output(self, name: str) -> str:
        """정상 렌더가 이미 남긴 파일을 흉내낸다."""

        path = self.out_dir / name
        path.write_bytes(b"png")
        return str(path)


def _processor(recorder: _Recorder, *, export_fails: set[str] | None = None):
    processor = object.__new__(StageBatchedProcessor)
    processor._released_page_buffer_bytes = 0
    failing = export_fails or set()

    def write_export(directory, token, image_path, image, patches, viewer, settings, **kw):
        name = Path(image_path).name
        if name in failing:
            raise OSError("disk full")
        recorder.exports.append((name, int(kw["page_index"])))
        # 정합성 검사는 디스크의 파일을 본다. 폴백도 진짜 파일을 남겨야 한다.
        written = recorder.out_dir / name
        written.write_bytes(b"png")
        return (str(written), str(recorder.out_dir))

    processor._write_final_render_export = write_export
    processor._emit_benchmark_event = lambda tag, **kw: recorder.events.append((tag, kw))
    processor._stage_tr = lambda text: text

    image_ctrl = types.SimpleNamespace(
        update_processing_summary=lambda path, payload: recorder.summaries.setdefault(
            Path(path).name, {}
        ).update(payload),
        load_image=lambda path: np.zeros((8, 8, 3), dtype=np.uint8),
    )
    batch_report = types.SimpleNamespace(
        register_preflight_error=lambda title, details="": recorder.preflight_errors.append(
            (title, details)
        )
    )
    processor.main_page = types.SimpleNamespace(
        image_ctrl=image_ctrl,
        batch_report_ctrl=batch_report,
    )
    return processor


class OutputReconciliationTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.out_dir = Path(self._tmp.name)

    def _recorder(self) -> _Recorder:
        return _Recorder(self.out_dir)

    def test_a_failed_page_still_produces_a_file(self) -> None:
        # 366장 입력에서 347장만 나오던 동작이 정확히 이 지점이었다.
        recorder = self._recorder()
        processor = _processor(recorder)
        pages = [_page("001.png"), _page("002.png", failed="OCR"), _page("003.png")]
        pages[0].output_path = recorder.existing_output("001.png")
        pages[2].output_path = recorder.existing_output("003.png")

        summary = processor._reconcile_page_outputs(pages, export_settings={})

        self.assertEqual(summary["input_count"], 3)
        self.assertEqual(summary["output_count"], 3)
        self.assertEqual(summary["fallback_count"], 1)
        self.assertEqual(summary["missing"], [])
        self.assertEqual(recorder.exports, [("002.png", 1)])

    def test_the_fallback_keeps_the_original_page_index(self) -> None:
        # 아카이브 모드는 페이지 번호로 파일명을 만든다. 순서가 어긋나면 안 된다.
        recorder = self._recorder()
        processor = _processor(recorder)
        pages = [_page(f"{i:03d}.png", failed="OCR") for i in range(1, 5)]
        pages[1].output_path = recorder.existing_output("002.png")

        processor._reconcile_page_outputs(pages, export_settings={})

        self.assertEqual(recorder.exports, [("001.png", 0), ("003.png", 2), ("004.png", 3)])

    def test_an_inpainted_page_falls_back_to_the_cleaned_image(self) -> None:
        recorder = self._recorder()
        processor = _processor(recorder)
        ctx = _page("010.png", failed="Translator")
        ctx.inpaint_input_img = np.full((8, 8, 3), 7, dtype=np.uint8)

        processor._reconcile_page_outputs([ctx], export_settings={})

        self.assertEqual(ctx.output_fallback_kind, "inpainted")

    def test_a_released_page_is_reloaded_from_disk(self) -> None:
        # 메모리 회수 뒤라도 폴백은 원본을 다시 읽어 반드시 파일을 남긴다.
        recorder = self._recorder()
        processor = _processor(recorder)
        ctx = _page("011.png", failed="OCR", with_image=False)

        processor._reconcile_page_outputs([ctx], export_settings={})

        self.assertEqual(ctx.output_fallback_kind, "source-reloaded")
        self.assertTrue(ctx.output_path.endswith("011.png"))

    def test_the_fallback_reason_reaches_the_processing_summary(self) -> None:
        # 조용한 폴백은 조용한 누락만큼 나쁘다.
        recorder = self._recorder()
        processor = _processor(recorder)
        ctx = _page("012.png", failed="OCR")

        processor._reconcile_page_outputs([ctx], export_settings={})

        summary = recorder.summaries["012.png"]
        self.assertEqual(summary["output_fallback_kind"], "source")
        self.assertEqual(summary["output_fallback_reason"], "OCR blew up")
        tags = [tag for tag, _payload in recorder.events]
        self.assertIn("page_output_fallback", tags)

    def test_an_unwritable_page_is_reported_not_swallowed(self) -> None:
        recorder = self._recorder()
        processor = _processor(recorder, export_fails={"020.png"})
        pages = [_page("020.png", failed="OCR"), _page("021.png")]
        pages[1].output_path = recorder.existing_output("021.png")

        summary = processor._reconcile_page_outputs(pages, export_settings={})

        self.assertEqual(summary["output_count"], 1)
        self.assertEqual(summary["missing"], ["020.png"])
        self.assertEqual(len(recorder.preflight_errors), 1)
        title, details = recorder.preflight_errors[0]
        self.assertIn("출력 페이지 수", title)
        self.assertIn("020.png", details)

    def test_a_fully_successful_batch_writes_nothing_extra(self) -> None:
        recorder = self._recorder()
        processor = _processor(recorder)
        pages = [_page("030.png"), _page("031.png")]
        for ctx in pages:
            ctx.output_path = recorder.existing_output(ctx.image_name)

        summary = processor._reconcile_page_outputs(pages, export_settings={})

        self.assertEqual(recorder.exports, [])
        self.assertEqual(summary["fallback_count"], 0)
        self.assertEqual(summary["output_count"], 2)
        self.assertEqual(recorder.preflight_errors, [])

    def test_a_recorded_output_that_is_not_on_disk_is_rewritten(self) -> None:
        # 내부 기록만 믿으면 안 된다. 실측으로 366장 중 15장이 인페인팅까지
        # 끝나고도 렌더 결과가 기록되지 않은 채 조용히 빠졌다. 디스크에 있는
        # 파일이 유일한 진실이다.
        recorder = self._recorder()
        processor = _processor(recorder)
        ctx = _page("050.png")
        ctx.output_path = str(self.out_dir / "never-written.png")

        summary = processor._reconcile_page_outputs([ctx], export_settings={})

        self.assertEqual(recorder.exports, [("050.png", 0)])
        self.assertEqual(summary["output_count"], 1)
        self.assertTrue(ctx.output_path.endswith("050.png"))

    def test_a_page_with_no_recorded_failure_states_the_real_cause(self) -> None:
        # 없는 실패를 지어내면 안 된다. 실패가 기록되지 않았는데 출력이 없으면
        # 원인은 다른 곳이다.
        recorder = self._recorder()
        processor = _processor(recorder)
        ctx = _page("051.png")

        summary = processor._reconcile_page_outputs([ctx], export_settings={})

        cause = summary["fallbacks"][0]["cause"]
        self.assertEqual(cause, "no rendered output was recorded for it")
        self.assertNotIn("unknown", cause)

    def test_reconciliation_releases_the_pages_it_touched(self) -> None:
        recorder = self._recorder()
        processor = _processor(recorder)
        ctx = _page("040.png", failed="OCR")
        ctx.inpaint_input_img = np.zeros((8, 8, 3), dtype=np.uint8)

        processor._reconcile_page_outputs([ctx], export_settings={})

        self.assertIsNone(ctx.image)
        self.assertIsNone(ctx.inpaint_input_img)


class BatchProcessWiringTests(unittest.TestCase):
    def test_batch_process_reconciles_before_reporting_completion(self) -> None:
        # 정합성 검사는 배치를 완료로 표시하기 전에 끝나야 한다. 그래야 누락이
        # 성공으로 보고되지 않는다.
        import inspect

        source = inspect.getsource(StageBatchedProcessor.batch_process)
        reconcile = source.index("_reconcile_page_outputs")
        # 앞쪽의 `batch_completed = True` 는 벤치마크 전용 stage-ceiling 분기다.
        # 그 경로는 렌더 자체를 돌지 않으므로 맞출 출력이 없다.
        completed = source.rindex("batch_completed = True")
        done = source.rindex('"batch_run_done"')
        self.assertLess(reconcile, done)
        self.assertLess(done, completed)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import numpy as np

from pipeline.batch_processor import BatchProcessor


class _Signal:
    def __init__(self) -> None:
        self.events: list[tuple] = []

    def emit(self, *args) -> None:
        self.events.append(args)


class _ImageController:
    def __init__(self) -> None:
        self.summary_updates: list[tuple[str, dict]] = []
        self.stage_updates: list[tuple[str, str, str, dict]] = []

    def update_processing_summary(self, image_path: str, payload: dict) -> None:
        self.summary_updates.append((image_path, payload))

    def mark_processing_stage(self, image_path: str, stage: str, status: str, **extra) -> None:
        self.stage_updates.append((image_path, stage, status, extra))


class _MainPage:
    def __init__(self) -> None:
        self.image_ctrl = _ImageController()
        self.image_skipped = _Signal()
        self.memlog_events: list[tuple[str, dict]] = []
        self._current_batch_run_type = "batch"

    def emit_memlog(self, tag: str, **payload) -> None:
        self.memlog_events.append((tag, payload))


def test_legacy_inpaint_failure_records_stage_and_traceback() -> None:
    processor = object.__new__(BatchProcessor)
    processor.main_page = _MainPage()
    skipped_saves: list[tuple] = []
    skipped_logs: list[tuple] = []
    processor.skip_save = lambda *args: skipped_saves.append(args)
    processor.log_skipped_image = lambda *args: skipped_logs.append(args)

    try:
        raise IndexError("index 2160 is out of bounds for axis 0 with size 2160")
    except Exception as error:
        processor._handle_legacy_inpaint_failure(
            index=0,
            total_images=2,
            image_path="/tmp/page-031.png",
            directory="/tmp",
            export_token="May-29-2026_03-48-56AM",
            base_name="page-031",
            extension=".png",
            archive_bname="False_Honour_8_Part_3_English",
            image=np.zeros((8, 8, 3), dtype=np.uint8),
            error=error,
            page_ocr_metrics={"ocr_total_block_count": 1},
        )

    assert processor.main_page.image_skipped.events
    image_path, stage, detail = processor.main_page.image_skipped.events[0]
    assert image_path == "/tmp/page-031.png"
    assert stage == "inpaint"
    assert "IndexError: index 2160" in detail
    assert "Traceback" in detail

    assert processor.main_page.image_ctrl.stage_updates == [
        (
            "/tmp/page-031.png",
            "inpaint",
            "failed",
            {"reason": "index 2160 is out of bounds for axis 0 with size 2160"},
        )
    ]
    assert processor.main_page.memlog_events[0][0] == "page_failed"
    assert processor.main_page.memlog_events[0][1]["failed_stage"] == "inpaint"
    assert processor.main_page.memlog_events[0][1]["ocr_total_block_count"] == 1
    assert skipped_saves
    assert skipped_logs[0][3].startswith("Inpaint: index 2160")

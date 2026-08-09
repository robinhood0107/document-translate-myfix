from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules.utils.run_report import (  # noqa: E402
    build_run_report,
    format_hms,
    format_minutes,
    render_run_report_text,
    write_run_report,
)


TELEMETRY = {
    "stages": {
        "detect": {"wall_ms": 60_000.0, "count": 366},
        "ocr": {"wall_ms": 71_000.0, "count": 366},
        "translate": {"wall_ms": 900_000.0, "count": 366},
        "inpaint": {"wall_ms": 467_000.0, "count": 366},
        "render": {"wall_ms": 110_000.0, "count": 366},
    }
}

OUTPUT_SUMMARY = {
    "input_count": 366,
    "output_count": 366,
    "fallback_count": 1,
    "fallbacks": [
        {
            "image_name": "233.png",
            "kind": "source",
            "failed_stage": "OCR",
            "reason": "Direct Paddle llama.cpp OCR response was truncated.",
        }
    ],
    "missing": [],
}


def _report(**overrides):
    payload = {
        "telemetry": TELEMETRY,
        "total_wall_sec": 1608.0,
        "page_outcomes": [
            {"image_name": "001.png", "output_path": "C:/out/001.png"},
            {"image_name": "233.png", "output_path": "C:/out/233.png"},
        ],
        "output_summary": OUTPUT_SUMMARY,
        "started_at_local": "2026-08-07 03:18:25",
    }
    payload.update(overrides)
    return build_run_report(**payload)


class DurationFormatTests(unittest.TestCase):
    def test_hms_covers_hours(self) -> None:
        self.assertEqual(format_hms(3661), "01:01:01")

    def test_hms_clamps_negative_values(self) -> None:
        self.assertEqual(format_hms(-5), "00:00:00")

    def test_minutes_are_reported_to_one_decimal(self) -> None:
        self.assertEqual(format_minutes(90), "1.5분")


class BuildRunReportTests(unittest.TestCase):
    def test_the_total_is_measured_not_summed_from_stages(self) -> None:
        # 단계 합(1608초)과 우연히 같아도, 총 시간은 클릭부터 잰 실측값이어야 한다.
        report = _report(total_wall_sec=1750.0)
        self.assertEqual(report["total_wall_sec"], 1750.0)
        self.assertEqual(report["total_wall_text"], "00:29:10")

    def test_stages_are_ordered_by_measured_cost(self) -> None:
        report = _report()
        self.assertEqual(
            [row["stage"] for row in report["stages"]],
            ["translate", "inpaint", "render", "ocr", "detect"],
        )
        self.assertEqual(report["stages"][0]["seconds"], 900.0)

    def test_per_page_seconds_come_from_the_measured_total(self) -> None:
        report = _report(total_wall_sec=732.0)
        self.assertAlmostEqual(report["seconds_per_page"], 2.0)

    def test_a_zero_page_run_reports_no_per_page_value(self) -> None:
        report = _report(output_summary={"input_count": 0, "output_count": 0})
        self.assertIsNone(report["seconds_per_page"])

    def test_missing_telemetry_does_not_break_the_report(self) -> None:
        report = _report(telemetry={})
        self.assertEqual(report["stages"], [])
        self.assertEqual(report["total_wall_text"], "00:26:48")


class RenderRunReportTests(unittest.TestCase):
    def test_the_text_names_the_fallback_page_and_its_reason(self) -> None:
        # 조용한 폴백은 조용한 누락만큼 나쁘다. 리포트에 반드시 보여야 한다.
        text = render_run_report_text(_report())
        self.assertIn("233.png", text)
        self.assertIn("truncated", text)
        self.assertIn("출력: 366장 / 입력 366장", text)

    def test_missing_pages_are_called_out(self) -> None:
        summary = dict(OUTPUT_SUMMARY)
        summary["output_count"] = 365
        summary["missing"] = ["343.png"]
        text = render_run_report_text(_report(output_summary=summary))
        self.assertIn("누락(저장 실패): 1장 -> 343.png", text)

    def test_every_stage_gets_a_minutes_column(self) -> None:
        text = render_run_report_text(_report())
        self.assertIn("15.0분", text)  # translate 900s
        self.assertIn("7.8분", text)  # inpaint 467s


class WriteRunReportTests(unittest.TestCase):
    def test_both_files_are_written_and_readable(self) -> None:
        report = _report()
        with tempfile.TemporaryDirectory() as tmp:
            path = write_run_report(report, log_dir=tmp)

            self.assertTrue(path.endswith(".txt"))
            written = sorted(p.name for p in Path(tmp).iterdir())
            self.assertEqual(len(written), 2)
            json_path = Path(tmp) / written[0]
            restored = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertEqual(restored["page_count"], 366)

    def test_an_unwritable_directory_does_not_raise(self) -> None:
        # 리포트는 진단 자료다. 못 써도 배치를 실패시키면 안 된다.
        self.assertEqual(write_run_report(_report(), log_dir="\0invalid"), "")


if __name__ == "__main__":
    unittest.main()

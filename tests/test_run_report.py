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
    start_run_report,
    write_run_report,
)


# `stages` 는 중첩·동시 측정까지 전부 더한 값이라 실행 시간을 넘길 수 있다.
# 단계가 실제로 점유한 구간은 `stage_details[...]["stage_window"]` 하나뿐이다.
def _window(seconds: float, count: int = 1) -> dict:
    return {"stage_window": {"wall_ms": seconds * 1000.0, "count": count}}


TELEMETRY = {
    "stage_details": {
        "detect": _window(60.0),
        "ocr": _window(71.0),
        "translate": _window(900.0),
        "inpaint": {
            **_window(467.0),
            # 세부 연산의 합이 작업량이다. 창(467s)보다 작을 수도 클 수도 있다.
            "mask_generation": {"wall_ms": 150_000.0, "count": 347},
            "model_forward": {"wall_ms": 100_000.0, "count": 347},
            "cleanup_and_composite": {"wall_ms": 50_000.0, "count": 347},
        },
        "render": {
            **_window(0.2),
            # 창은 0.2초인데 작업은 110초다. 그 차이가 융합으로 숨긴 양이다.
            "worker": {"wall_ms": 110_000.0, "count": 347},
        },
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
    def test_source_unit_processing_status_gates_page_quality(self) -> None:
        shared = {
            "source_unit_index": 0,
            "image_path": "example_source_chapter/page-001.png",
            "image_name": "page-001.png",
            "output_path": "example_output/page-001.png",
            "output_status": "measured",
            "quality_status": "completed",
            "source_unit_mapping_status": "observed",
        }
        output_summary = {
            "input_count": 1,
            "output_count": 1,
            "fallback_count": 0,
            "fallbacks": [],
            "missing": [],
        }
        terminal = {
            "terminal_status": "completed",
            "processing_status": "completed",
            "report_status": "completed",
            "cleanup_status": "completed",
        }
        failed = build_run_report(
            telemetry=None,
            total_wall_sec=1.0,
            page_outcomes=[
                {
                    **shared,
                    "source_units": [
                        {
                            "source_unit_id": "example-truncated-unit",
                            "status": "failed",
                            "reason_code": "ocr_truncated",
                            "semantic_quality_status": "unjudged",
                        },
                        {
                            "source_unit_id": "example-translated-unit",
                            "status": "translated",
                            "semantic_quality_status": "unjudged",
                        },
                    ],
                }
            ],
            output_summary=output_summary,
            **terminal,
        )
        self.assertEqual(failed["quality"]["status"], "failed")
        self.assertEqual(failed["terminal_status"], "failed")
        self.assertEqual(failed["quality"]["required_units"]["value"], 1)
        self.assertEqual(failed["quality"]["required_units"]["unit"], "source_pages")
        self.assertEqual(failed["quality"]["failed_units"]["value"], 1)
        self.assertEqual(failed["quality"]["failed_units"]["unit"], "source_pages")
        self.assertEqual(failed["pages"][0]["quality_status"], "failed")

        review = build_run_report(
            telemetry=None,
            total_wall_sec=1.0,
            page_outcomes=[
                {
                    **shared,
                    "source_units": [
                        {
                            "source_unit_id": "example-review-unit",
                            "status": "unjudged",
                            "reason_code": "ocr_review_required",
                            "semantic_quality_status": "unjudged",
                        }
                    ],
                }
            ],
            output_summary=output_summary,
            **terminal,
        )
        self.assertEqual(review["quality"]["status"], "review_required")
        self.assertEqual(review["quality"]["review_units"]["value"], 1)
        self.assertEqual(review["terminal_status"], "incomplete")
        self.assertEqual(review["pages"][0]["quality_status"], "review_required")

        unobserved = build_run_report(
            telemetry=None,
            total_wall_sec=1.0,
            page_outcomes=[
                {
                    **shared,
                    "source_units": [
                        {
                            "source_unit_id": "example-unobserved-unit",
                            "status": "unjudged",
                            "reason_code": "translation_not_observed",
                            "semantic_quality_status": "unjudged",
                        }
                    ],
                }
            ],
            output_summary=output_summary,
            **terminal,
        )
        self.assertEqual(unobserved["quality"]["status"], "incomplete")
        self.assertEqual(unobserved["quality"]["unjudged_units"]["value"], 1)
        self.assertEqual(unobserved["terminal_status"], "incomplete")

        preserved_and_skipped = build_run_report(
            telemetry=None,
            total_wall_sec=1.0,
            page_outcomes=[
                {
                    **shared,
                    "source_units": [
                        {
                            "source_unit_id": "example-translated-unit",
                            "status": "translated",
                            "semantic_quality_status": "unjudged",
                        },
                        {
                            "source_unit_id": "example-preserved-unit",
                            "status": "preserved",
                            "reason_code": "preserve_policy",
                            "semantic_quality_status": "unjudged",
                        },
                        {
                            "source_unit_id": "example-skipped-unit",
                            "status": "intentionally_skipped",
                            "reason_code": "embedded_ui_cluster",
                            "semantic_quality_status": "unjudged",
                        },
                    ],
                }
            ],
            output_summary=output_summary,
            **terminal,
        )
        self.assertEqual(preserved_and_skipped["quality"]["status"], "completed")
        self.assertEqual(preserved_and_skipped["quality"]["failed_units"]["value"], 0)
        self.assertEqual(preserved_and_skipped["terminal_status"], "completed")

    def test_source_unit_mapping_keeps_relations_and_operation_statuses_separate(self) -> None:
        source_units = [
            {
                "source_unit_id": "source-a",
                "status": "translated",
                "reason_code": None,
                "output_unit_ids": ["output-a", "output-b"],
                "semantic_quality_status": "unjudged",
            },
            {
                "source_unit_id": "source-b",
                "status": "preserved",
                "reason_code": "preserve_policy",
                "output_unit_ids": ["output-b"],
                "semantic_quality_status": "unjudged",
            },
            {
                "source_unit_id": "source-c",
                "status": "intentionally_skipped",
                "reason_code": "embedded_ui_cluster",
                "output_unit_ids": [],
                "semantic_quality_status": "unjudged",
            },
            {
                "source_unit_id": "source-d",
                "status": "failed",
                "reason_code": "ocr_truncated",
                "output_unit_ids": [],
                "semantic_quality_status": "unjudged",
            },
            {
                "source_unit_id": "source-e",
                "status": "unjudged",
                "reason_code": None,
                "output_unit_ids": [],
                "semantic_quality_status": "unjudged",
            },
        ]
        report = build_run_report(
            telemetry=None,
            total_wall_sec=None,
            page_outcomes=[
                {
                    "source_unit_index": 0,
                    "image_path": "C:/input/example.png",
                    "image_name": "example.png",
                    "output_path": "C:/output/example.png",
                    "output_status": "measured",
                    "quality_status": "unjudged",
                    "source_unit_mapping_status": "observed",
                    "source_units": source_units,
                }
            ],
            output_summary={
                "input_count": 1,
                "output_count": 1,
                "fallback_count": 0,
                "fallbacks": [],
                "missing": [],
            },
        )

        mapping = report["source_unit_mapping"]
        self.assertEqual(mapping["status"], "observed")
        self.assertEqual(
            [item["source_unit_id"] for item in mapping["entries"]],
            ["source-a", "source-b", "source-c", "source-d", "source-e"],
        )
        self.assertEqual(mapping["entries"][0]["output_unit_ids"], ["output-a", "output-b"])
        self.assertEqual(mapping["entries"][3]["status"], "failed")
        self.assertEqual(mapping["entries"][3]["reason_code"], "ocr_truncated")
        self.assertIsNone(mapping["entries"][4]["reason_code"])
        self.assertEqual(report["quality"]["required_units"]["value"], 1)
        self.assertEqual(report["quality"]["required_units"]["unit"], "source_pages")

    def test_terminal_report_preserves_telemetry_and_source_unit_outcomes(self) -> None:
        telemetry = {
            "status": "failed",
            "gemma": {"gemma_logical_request_count": 2, "gemma_http_attempt_count": 3},
            "runtime": {"gemma": {"start_wall_ms": 12.0}},
            "resources": {"sampling_enabled": False, "samples": [], "summary": {}},
        }
        report = build_run_report(
            telemetry=telemetry,
            total_wall_sec=None,
            page_outcomes=[
                {"source_unit_index": 0, "image_name": "a.png", "output_path": "out/a.png", "quality_status": "completed"},
                {"source_unit_index": 1, "image_name": "b.png", "output_path": None, "failed_stage": "ocr", "quality_status": "failed"},
                {"source_unit_index": 2, "image_name": "c.png", "output_path": None, "quality_status": "unjudged"},
            ],
            output_summary={
                "input_count": 3,
                "output_count": None,
                "fallback_count": 1,
                "fallbacks": [{"image_name": "b.png", "kind": "source"}],
                "missing": None,
            },
            terminal_status="failed",
            processing_status="failed",
            report_status="completed",
            cleanup_status="failed",
        )

        self.assertEqual(report["telemetry"]["value"], telemetry)
        self.assertEqual(report["quality"]["required_units"]["value"], 3)
        self.assertEqual(report["quality"]["failed_units"]["value"], 1)
        self.assertEqual(report["quality"]["unjudged_units"]["value"], 1)
        self.assertEqual(report["quality"]["unmeasured_units"]["value"], 2)
        self.assertEqual(report["quality"]["source_fallback_units"]["value"], 1)
        self.assertEqual(report["output"]["measurement"]["status"], "unavailable")
        self.assertEqual(report["output"]["measurement_reason_code"], "output_reconciliation_not_observed")
        self.assertEqual(report["total_wall_measurement"]["status"], "unavailable")
        self.assertEqual(report["resource_collection"]["reason_code"], "resource_sampling_disabled")

    def test_unjudged_or_unmeasured_input_prevents_completed_status(self) -> None:
        report = _report(
            page_outcomes=[
                {"source_unit_index": 0, "image_name": "a.png", "output_path": "out/a.png", "quality_status": "completed"}
            ],
            output_summary={
                "input_count": 2,
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

        self.assertEqual(report["quality"]["unjudged_units"]["value"], 0)
        self.assertEqual(report["quality"]["unmeasured_units"]["value"], 1)
        self.assertEqual(report["quality"]["status"], "incomplete")
        self.assertEqual(report["terminal_status"], "incomplete")

        unjudged = _report(
            page_outcomes=[{"quality_status": "unjudged", "output_path": None}],
            output_summary={"input_count": 1, "output_count": 1},
            terminal_status="completed",
            processing_status="completed",
            report_status="completed",
            cleanup_status="completed",
        )
        self.assertEqual(unjudged["quality"]["unjudged_units"]["value"], 1)
        self.assertEqual(unjudged["quality"]["status"], "incomplete")
        self.assertEqual(unjudged["terminal_status"], "incomplete")

    def test_missing_input_denominator_is_not_inferred_from_observed_pages(self) -> None:
        report = _report(
            page_outcomes=[
                {"source_unit_index": 0, "image_name": "a.png", "output_path": "out/a.png", "quality_status": "completed"}
            ],
            output_summary={"output_count": 1},
        )

        self.assertIsNone(report["quality"]["required_units"]["value"])
        self.assertEqual(report["quality"]["required_units"]["status"], "unavailable")
        self.assertEqual(report["quality"]["required_units"]["reason_code"], "input_count_not_observed")
        self.assertIsNone(report["page_count"])
        self.assertIsNone(report["quality"]["unjudged_units"]["value"])

    def test_start_run_report_leaves_a_durable_running_record(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = start_run_report(
                log_dir=tmp,
                run_id="test-run-001",
                started_at_local="2026-11-17 03:00:00",
                input_count=2,
            )

            record = json.loads(Path(path).read_text(encoding="utf-8"))

        self.assertEqual(record["terminal_status"], "running")
        self.assertEqual(record["processing_status"], "running")
        self.assertEqual(record["report_status"], "running")
        self.assertEqual(record["cleanup_status"], "not_started")
        self.assertIsNone(record["telemetry"]["value"])
        self.assertEqual(record["telemetry"]["reason_code"], "run_in_progress")

    def test_preserved_review_source_is_not_reported_as_completed_quality(self) -> None:
        report = _report(page_outcomes=[{
            "image_name": "example.png", "output_path": "C:/out/example.png",
            "failed_stage": "", "block_failure_ledger": [{
                "block_id": "example-block", "stage": "inpaint",
                "cause": "source_glyph_extension_unresolved",
                "page_blocking": False, "review_required": True,
            }],
        }])
        self.assertEqual(report["quality"]["status"], "review_required")
        self.assertEqual(report["quality"]["review_page_count"], 1)
        self.assertIn("검토 필요 1장", render_run_report_text(report))

    def test_completed_terminal_is_lowered_for_failed_or_unjudged_quality(self) -> None:
        failed = _report(
            page_outcomes=[{
                "source_unit_index": 0,
                "image_name": "example_source.png",
                "output_path": "example_output.png",
                "quality_status": "failed",
            }],
            output_summary={
                "input_count": 1,
                "output_count": 0,
                "fallback_count": 0,
                "fallbacks": [],
                "missing": ["example_source.png"],
            },
            terminal_status="completed",
            processing_status="completed",
        )
        review = _report(
            page_outcomes=[{
                "source_unit_index": 0,
                "image_name": "example_source.png",
                "output_path": "example_output.png",
                "quality_status": "review_required",
            }],
            output_summary={
                "input_count": 1,
                "output_count": 1,
                "fallback_count": 0,
                "fallbacks": [],
                "missing": [],
            },
            terminal_status="completed",
            processing_status="completed",
        )

        self.assertEqual(failed["quality"]["status"], "failed")
        self.assertEqual(failed["terminal_status"], "failed")
        self.assertEqual(review["quality"]["status"], "review_required")
        self.assertEqual(review["terminal_status"], "incomplete")

    def test_normal_output_and_stage_failure_have_distinct_quality(self) -> None:
        normal = _report(
            page_outcomes=[{"quality_status": "completed", "output_path": "out/a.png"}],
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
        self.assertEqual(normal["quality"]["status"], "completed")
        report = _report(
            page_outcomes=[{"failed_stage": "render"}],
            processing_status="failed",
        )
        self.assertEqual(report["quality"]["status"], "failed")

    def test_reloaded_source_fallback_and_disjoint_failures_use_source_indexes(self) -> None:
        report = _report(
            page_outcomes=[
                {"source_unit_index": 0, "image_name": "same.png", "output_path": "out/first.png", "quality_status": "completed"},
                {"source_unit_index": 1, "image_name": "same.png", "output_path": "out/second.png", "quality_status": "completed"},
                {"source_unit_index": 2, "image_name": "failed.png", "output_path": None, "failed_stage": "ocr", "quality_status": "failed"},
                {"source_unit_index": 3, "image_name": "missing.png", "output_path": "out/missing.png", "quality_status": "completed"},
            ],
            output_summary={
                "input_count": 4,
                "output_count": 3,
                "fallback_count": 1,
                "fallbacks": [{"source_unit_index": 1, "image_name": "same.png", "kind": "source-reloaded"}],
                "missing": ["missing.png"],
                "missing_source_unit_indices": [3],
            },
        )

        self.assertEqual(report["quality"]["failed_units"]["value"], 2)
        self.assertEqual(report["quality"]["source_fallback_units"]["value"], 1)
        self.assertEqual(report["quality"]["completed_units"]["value"], 1)

    def test_the_total_is_measured_not_summed_from_stages(self) -> None:
        # 단계 합(1608초)과 우연히 같아도, 총 시간은 클릭부터 잰 실측값이어야 한다.
        report = _report(total_wall_sec=1750.0)
        self.assertEqual(report["total_wall_sec"], 1750.0)
        self.assertEqual(report["total_wall_text"], "00:29:10")

    def test_stages_are_ordered_by_measured_cost(self) -> None:
        # 렌더는 인페인팅 뒤에 숨으므로 창이 거의 0이다. 작업이 없다는 뜻이
        # 아니라 겹쳐서 숨겼다는 뜻이며, 그건 stage_work 가 보여준다.
        report = _report()
        self.assertEqual(
            [row["stage"] for row in report["stages"]],
            ["translate", "inpaint", "ocr", "detect", "render"],
        )
        self.assertEqual(report["stages"][0]["seconds"], 900.0)

    def test_a_hidden_stage_shows_work_far_above_its_window(self) -> None:
        report = _report()
        window = next(r for r in report["stages"] if r["stage"] == "render")
        work = next(r for r in report["stage_work"] if r["stage"] == "render")

        self.assertEqual(window["seconds"], 0.2)
        self.assertEqual(work["work_seconds"], 110.0)

    def test_per_page_seconds_come_from_the_measured_total(self) -> None:
        report = _report(total_wall_sec=732.0)
        self.assertAlmostEqual(report["seconds_per_page"], 2.0)

    def test_a_zero_page_run_reports_no_per_page_value(self) -> None:
        report = _report(output_summary={"input_count": 0, "output_count": 0})
        self.assertIsNone(report["seconds_per_page"])

    def test_missing_telemetry_does_not_break_the_report(self) -> None:
        report = _report(telemetry={})
        self.assertEqual(report["stages"], [])
        self.assertEqual(report["stage_work"], [])
        self.assertEqual(report["total_wall_text"], "00:26:48")

    def test_stage_time_uses_the_window_not_the_sum_of_nested_measurements(self) -> None:
        # 실측 사고: 41분 실행에서 inpaint 가 64시간으로 찍혔다. 페이지별 측정을
        # 전부 더했기 때문이다. 단계 시간은 벽시계 구간이어야 한다.
        report = _report()
        inpaint = next(r for r in report["stages"] if r["stage"] == "inpaint")

        self.assertEqual(inpaint["seconds"], 467.0)
        self.assertLess(inpaint["seconds"], report["total_wall_sec"])
        for row in report["stages"]:
            self.assertLessEqual(row["seconds"], report["total_wall_sec"])

    def test_work_totals_come_from_the_operations_not_the_mixed_bucket(self) -> None:
        # `telemetry["stages"]` 는 여러 출처를 한 통에 담는다. 실측으로 42분
        # 실행에서 inpaint 가 4030분으로 찍혔지만 세부 연산을 다 더하면 21분이다.
        # 작업 합계는 항상 설명 가능한 값이어야 한다.
        report = _report()
        inpaint = next(r for r in report["stage_work"] if r["stage"] == "inpaint")

        # 세부 연산 합계이며, 부풀려진 232411초가 아니다.
        self.assertAlmostEqual(inpaint["work_seconds"], 300.0)
        self.assertEqual(inpaint["count"], 347)
        self.assertLessEqual(
            inpaint["work_seconds"],
            report["total_wall_sec"],
        )

    def test_a_stage_with_only_a_window_reports_no_work(self) -> None:
        # 세부 연산이 없으면 더할 작업도 없다. 창 시간을 작업량으로 둔갑시키지 않는다.
        report = _report(
            telemetry={"stage_details": {"detect": _window(60.0)}},
        )
        self.assertEqual(report["stage_work"], [])


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

    def test_unknown_page_and_input_counts_stay_unmeasured(self) -> None:
        report = _report(output_summary={"input_count": None, "output_count": None})
        text = render_run_report_text(report)

        self.assertIn("페이지: 미측정", text)
        self.assertIn("출력: 미측정 / 입력 미측정", text)
        self.assertNotIn("None장", text)


class WriteRunReportTests(unittest.TestCase):
    def test_stage_standalone_report_write_preserves_failed_terminal_quality(self) -> None:
        from types import SimpleNamespace
        from unittest import mock

        from pipeline.stage_batched_processor import StageBatchedProcessor

        processor = StageBatchedProcessor.__new__(StageBatchedProcessor)
        processor.main_page = SimpleNamespace(batch_report_ctrl=None)
        processor._performance_telemetry = lambda: SimpleNamespace(
            snapshot=lambda: {}
        )
        processor._click_to_now_seconds = lambda: 1.0
        processor._run_started_wall_text = lambda: "2026-11-17 00:00:00"
        processor._page_outcomes = lambda _pages, **_kwargs: [{
            "source_unit_index": 0,
            "image_name": "example_source.png",
            "output_path": None,
            "output_status": "unmeasured",
            "quality_status": "failed",
        }]

        with tempfile.TemporaryDirectory() as temporary, mock.patch(
            "modules.utils.paths.get_log_dir", return_value=temporary
        ):
            path = processor._write_run_report(
                [object()],
                output_summary={
                    "input_count": 1,
                    "output_count": 1,
                    "fallback_count": 0,
                    "fallbacks": [],
                    "missing": [],
                },
                processing_status="completed",
                cleanup_status="completed",
            )
            record = json.loads(Path(path).with_suffix(".json").read_text(encoding="utf-8"))

        self.assertEqual(record["quality"]["status"], "failed")
        self.assertEqual(record["terminal_status"], "failed")

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

    def test_second_file_write_failure_rolls_back_partial_text_report(self) -> None:
        from unittest import mock
        from modules.utils import run_report

        with tempfile.TemporaryDirectory() as tmp:
            report = _report(run_id="partial-write")
            start_path = start_run_report(
                log_dir=tmp,
                run_id="partial-write",
                started_at_local=report["started_at_local"],
                input_count=366,
            )
            original_json = Path(start_path).read_bytes()
            atomic_write = run_report._atomic_write
            calls = 0

            def fail_second_write(path: str, content: str) -> None:
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise OSError("synthetic JSON replacement failure")
                atomic_write(path, content)

            with mock.patch.object(run_report, "_atomic_write", side_effect=fail_second_write):
                self.assertEqual(write_run_report(report, log_dir=tmp), "")

            self.assertEqual(Path(start_path).read_bytes(), original_json)
            self.assertEqual(list(Path(tmp).glob("*.txt")), [])

    def test_keyboard_interrupt_on_second_write_rolls_back_then_propagates(self) -> None:
        from unittest import mock
        from modules.utils import run_report

        with tempfile.TemporaryDirectory() as tmp:
            report = _report(run_id="interrupt-write")
            text_path = Path(write_run_report(report, log_dir=tmp))
            json_path = text_path.with_suffix(".json")
            original_text = text_path.read_bytes()
            original_json = json_path.read_bytes()
            updated_report = _report(run_id="interrupt-write", total_wall_sec=1700.0)
            atomic_write = run_report._atomic_write
            calls = 0

            def interrupt_second_write(path: str, content: str) -> None:
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise KeyboardInterrupt("synthetic interrupt")
                atomic_write(path, content)

            with mock.patch.object(
                run_report, "_atomic_write", side_effect=interrupt_second_write
            ):
                with self.assertRaisesRegex(KeyboardInterrupt, "synthetic interrupt"):
                    write_run_report(updated_report, log_dir=tmp)

            self.assertEqual(text_path.read_bytes(), original_text)
            self.assertEqual(json_path.read_bytes(), original_json)
            self.assertEqual(list(Path(tmp).glob(".run-report-*")), [])

    def test_keyboard_interrupt_remains_primary_if_rollback_fails(self) -> None:
        from unittest import mock
        from modules.utils import run_report

        with tempfile.TemporaryDirectory() as tmp:
            report = _report(run_id="interrupt-rollback")
            write_run_report(report, log_dir=tmp)
            updated_report = _report(run_id="interrupt-rollback", total_wall_sec=1700.0)
            atomic_write = run_report._atomic_write
            calls = 0

            def interrupt_then_fail_rollback(path: str, content: str) -> None:
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise KeyboardInterrupt("primary interrupt")
                if calls >= 3:
                    raise OSError("rollback failed")
                atomic_write(path, content)

            with mock.patch.object(
                run_report, "_atomic_write", side_effect=interrupt_then_fail_rollback
            ):
                with self.assertRaisesRegex(KeyboardInterrupt, "primary interrupt"):
                    write_run_report(updated_report, log_dir=tmp)

            self.assertEqual(calls, 4)

    def test_an_unwritable_directory_does_not_raise(self) -> None:
        # 리포트는 진단 자료다. 못 써도 배치를 실패시키면 안 된다.
        self.assertEqual(write_run_report(_report(), log_dir="\0invalid"), "")


if __name__ == "__main__":
    unittest.main()

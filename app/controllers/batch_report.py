from __future__ import annotations

import logging
import os
import time
import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any, Mapping

from PySide6 import QtCore, QtWidgets

from app.ui.dayu_widgets.drawer import MDrawer
from app.ui.messages import Messages
from modules.utils.run_report import build_run_report, format_hms, start_run_report, write_run_report

logger = logging.getLogger(__name__)


def _error_summary(error: object) -> dict[str, str] | None:
    if not isinstance(error, Mapping):
        return None
    summary = {
        key: str(error[key])
        for key in ("step", "error_type", "phase", "reason_code")
        if error.get(key)
    }
    return summary or None


def _cleanup_summaries(errors: object) -> list[dict[str, str]]:
    if not isinstance(errors, list):
        return []
    return [summary for error in errors if (summary := _error_summary(error)) is not None]


def _page_outcome_summaries(outcomes: object) -> list[dict[str, str]]:
    if not isinstance(outcomes, list):
        return []
    return [
        {"image_path": item["image_path"], "status": item["status"]}
        for item in outcomes
        if isinstance(item, Mapping)
        and isinstance(item.get("image_path"), str)
        and item.get("status") in {"completed", "failed", "review_required", "unjudged"}
    ]

if TYPE_CHECKING:
    from controller import ComicTranslate


class BatchReportController:
    def __init__(self, main: ComicTranslate):
        self.main = main
        self._current_batch_report = None
        self._latest_batch_report = None
        self._batch_report_drawer: MDrawer | None = None

    def _tr(self, text: str) -> str:
        return QtCore.QCoreApplication.translate("BatchReportController", text)

    def clear_latest_report(self) -> None:
        self._current_batch_report = None
        self._latest_batch_report = None
        if self._batch_report_drawer is not None:
            try:
                self._batch_report_drawer.close()
            except Exception:
                pass
            self._batch_report_drawer = None
        self.refresh_action_buttons()

    def _current_page_available(self) -> bool:
        return 0 <= getattr(self.main, "curr_img_idx", -1) < len(getattr(self.main, "image_files", []))

    def _normalize_datetime(self, value):
        if isinstance(value, datetime):
            return value
        if isinstance(value, str):
            try:
                return datetime.fromisoformat(value)
            except ValueError:
                return None
        return None

    def export_latest_report_for_project(self) -> dict | None:
        report = self._latest_batch_report
        if report is None:
            return None
        return {
            "run_type": report.get("run_type", "batch"),
            "started_at": (
                report.get("started_at").isoformat()
                if isinstance(report.get("started_at"), datetime)
                else report.get("started_at")
            ),
            "finished_at": (
                report.get("finished_at").isoformat()
                if isinstance(report.get("finished_at"), datetime)
                else report.get("finished_at")
            ),
            "was_cancelled": bool(report.get("was_cancelled", False)),
            "total_images": int(report.get("total_images", 0) or 0),
            "skipped_count": int(report.get("skipped_count", 0) or 0),
            "completed_count": int(report.get("completed_count", 0) or 0),
            "failed_count": int(report.get("failed_count", 0) or 0),
            "paths": list(report.get("paths", [])),
            "retry_paths": list(report.get("retry_paths", [])),
            "page_outcomes": [dict(item) for item in report.get("page_outcomes", [])],
            "run_page_outcomes": [dict(item) for item in report.get("run_page_outcomes", [])],
            "run_total_images": int(report.get("run_total_images", 0) or 0),
            "run_completed_count": int(report.get("run_completed_count", 0) or 0),
            "run_failed_count": int(report.get("run_failed_count", 0) or 0),
            "run_unjudged_count": int(report.get("run_unjudged_count", 0) or 0),
            "skipped_entries": list(report.get("skipped_entries", [])),
            "preflight_errors": list(report.get("preflight_errors", [])),
            "preflight_warnings": list(report.get("preflight_warnings", [])),
            "processing_status": report.get("processing_status", "unknown"),
            "report_status": report.get("report_status", "unknown"),
            "cleanup_status": report.get("cleanup_status", "unknown"),
            "pipeline_cleanup_status": report.get("pipeline_cleanup_status", "unknown"),
            "application_cleanup_status": report.get("application_cleanup_status", "unknown"),
            "archive_status": report.get("archive_status", "unknown"),
            "auto_export_status": report.get("auto_export_status", "unknown"),
            "auto_export_targets": list(report.get("auto_export_targets", [])),
            "auto_export_completed_targets": list(
                report.get("auto_export_completed_targets", [])
            ),
            "auto_export_failed_target": report.get("auto_export_failed_target"),
            "auto_export_reason_code": report.get("auto_export_reason_code"),
            "auto_export_error": _error_summary(report.get("auto_export_error")),
            "terminal_status": report.get("terminal_status", "unknown"),
            "worker_status": report.get("worker_status", "unknown"),
            "start_record_status": report.get("start_record_status", "unknown"),
            "start_record_reason_code": report.get("start_record_reason_code"),
            "report_reason_code": report.get("report_reason_code"),
            "processing_error": _error_summary(report.get("processing_error")),
            "cleanup_errors": _cleanup_summaries(report.get("cleanup_errors", [])),
            "unjudged_count": int(report.get("unjudged_count", 0) or 0),
        }

    def import_latest_report_from_project(self, payload: dict | None, refresh: bool = True) -> None:
        if not payload:
            self._latest_batch_report = None
            if refresh:
                self.refresh_action_buttons()
            return

        was_cancelled = bool(payload.get("was_cancelled", False))
        terminal_status = payload.get("terminal_status")
        if not terminal_status and was_cancelled:
            terminal_status = "cancelled"
        self._latest_batch_report = {
            "run_type": payload.get("run_type", "batch"),
            "started_at": self._normalize_datetime(payload.get("started_at")),
            "finished_at": self._normalize_datetime(payload.get("finished_at")),
            "was_cancelled": was_cancelled,
            "total_images": int(payload.get("total_images", 0) or 0),
            "skipped_count": int(payload.get("skipped_count", 0) or 0),
            "completed_count": int(payload.get("completed_count", 0) or 0),
            "failed_count": int(payload.get("failed_count", 0) or 0),
            "paths": list(payload.get("paths", [])),
            "retry_paths": list(payload.get("retry_paths", [])),
            "page_outcomes": _page_outcome_summaries(payload.get("page_outcomes")),
            "run_page_outcomes": _page_outcome_summaries(payload.get("run_page_outcomes")),
            "run_total_images": int(payload.get("run_total_images", 0) or 0),
            "run_completed_count": int(payload.get("run_completed_count", 0) or 0),
            "run_failed_count": int(payload.get("run_failed_count", 0) or 0),
            "run_unjudged_count": int(payload.get("run_unjudged_count", 0) or 0),
            "skipped_entries": list(payload.get("skipped_entries", [])),
            "preflight_errors": list(payload.get("preflight_errors", [])),
            "preflight_warnings": list(payload.get("preflight_warnings", [])),
            "processing_status": str(payload.get("processing_status", "unknown")),
            "report_status": str(payload.get("report_status", "unknown")),
            "cleanup_status": str(payload.get("cleanup_status", "unknown")),
            "pipeline_cleanup_status": str(payload.get("pipeline_cleanup_status", "unknown")),
            "application_cleanup_status": str(payload.get("application_cleanup_status", "unknown")),
            "archive_status": str(payload.get("archive_status", "unknown")),
            "auto_export_status": str(payload.get("auto_export_status", "unknown")),
            "auto_export_targets": list(payload.get("auto_export_targets", [])),
            "auto_export_completed_targets": list(
                payload.get("auto_export_completed_targets", [])
            ),
            "auto_export_failed_target": payload.get("auto_export_failed_target"),
            "auto_export_reason_code": payload.get("auto_export_reason_code"),
            "auto_export_error": _error_summary(payload.get("auto_export_error")),
            "terminal_status": str(terminal_status or "unknown"),
            "worker_status": str(payload.get("worker_status", "unknown")),
            "start_record_status": str(payload.get("start_record_status", "unknown")),
            "start_record_reason_code": payload.get("start_record_reason_code"),
            "report_reason_code": payload.get("report_reason_code"),
            "processing_error": _error_summary(payload.get("processing_error")),
            "cleanup_errors": _cleanup_summaries(payload.get("cleanup_errors", [])),
            "unjudged_count": int(payload.get("unjudged_count", 0) or 0),
        }
        if refresh:
            self.refresh_action_buttons()

    def _get_filtered_retry_paths(self, report: dict | None) -> list[str]:
        if report is None:
            return []

        current_paths = set(self.main.image_files)
        retry_paths = report.get("retry_paths")
        if retry_paths is None:
            retry_paths = [entry["image_path"] for entry in report.get("skipped_entries", [])]

        filtered: list[str] = []
        seen: set[str] = set()
        for image_path in retry_paths:
            if image_path not in current_paths or image_path in seen:
                continue
            seen.add(image_path)
            filtered.append(image_path)
        return filtered

    def refresh_action_buttons(self) -> None:
        has_report = self._latest_batch_report is not None
        self.main.batch_report_button.setEnabled(has_report)

        retry_enabled = bool(self._get_filtered_retry_paths(self._latest_batch_report))
        if getattr(self.main, "_batch_active", False):
            retry_enabled = False
        self.main.retry_failed_button.setEnabled(retry_enabled)

        one_page_enabled = (
            self._current_page_available()
            and bool(self.main.automatic_radio.isChecked())
            and not getattr(self.main, "_batch_active", False)
        )
        if hasattr(self.main, "one_page_auto_button"):
            self.main.one_page_auto_button.setEnabled(one_page_enabled)

    def get_latest_retry_paths(self) -> list[str]:
        return list(self._get_filtered_retry_paths(self._latest_batch_report))

    def start_batch_report(
        self,
        batch_paths: list[str],
        run_type: str = "batch",
        *,
        run_report_log_dir: str | None = None,
    ):
        tracked_paths = [
            path
            for path in batch_paths
            if not self.main.image_states.get(path, {}).get("skip", False)
        ]
        started_at = datetime.now()
        started_monotonic = time.monotonic()
        started_at_local = started_at.strftime("%Y-%m-%d %H:%M:%S")
        run_report_id = uuid.uuid4().hex
        run_report_path = (
            start_run_report(
                log_dir=run_report_log_dir,
                run_id=run_report_id,
                started_at_local=started_at_local,
                input_count=len(tracked_paths),
            )
            if run_report_log_dir
            else ""
        )
        self._current_batch_report = {
            "run_type": run_type,
            "started_at": started_at,
            "started_at_local": started_at_local,
            "started_monotonic": started_monotonic,
            "paths": tracked_paths,
            "path_set": set(tracked_paths),
            "skipped": {},
            "completed_paths": set(),
            "batch_output_paths": {},
            "previous_report": self.export_latest_report_for_project(),
            "preflight_errors": [],
            "preflight_warnings": [],
            "preflight_seen": set(),
            "run_report_id": run_report_id,
            "run_report_log_dir": run_report_log_dir or "",
            "run_report_status": "running" if run_report_path else "not_configured" if not run_report_log_dir else "failed",
            "start_record_status": "running" if run_report_path else "not_configured" if not run_report_log_dir else "failed",
            "start_record_reason_code": None if run_report_path else "start_record_write_failed" if run_report_log_dir else "run_report_not_configured",
            "pipeline_processing_status": "",
            "pipeline_cleanup_status": "",
            "pipeline_processing_error": None,
            "pipeline_cleanup_errors": [],
            "pipeline_telemetry_provider": None,
            "pipeline_telemetry_run_id": None,
        }

    @staticmethod
    def _store_pipeline_run_report(
        target: dict[str, Any], report: Mapping[str, object]
    ) -> None:
        target["pipeline_processing_status"] = str(
            report.get("processing_status", "unknown")
        )
        target["pipeline_cleanup_status"] = str(
            report.get("cleanup_status", "unknown")
        )
        target["pipeline_processing_error"] = report.get("processing_error")
        target["pipeline_cleanup_errors"] = list(
            report.get("cleanup_errors", []) or []
        )
        target["pipeline_run_report_data"] = dict(report)

    def register_pipeline_run_report(self, report: Mapping[str, object]) -> bool:
        current = self._current_batch_report
        if not current:
            return False
        self._store_pipeline_run_report(current, report)
        return True

    def register_batch_telemetry(self, telemetry: object) -> bool:
        current = self._current_batch_report
        run_id = getattr(telemetry, "run_id", None)
        if (
            not current
            or not isinstance(run_id, str)
            or not run_id
            or not callable(getattr(telemetry, "snapshot", None))
        ):
            return False
        current["pipeline_telemetry_provider"] = telemetry
        current["pipeline_telemetry_run_id"] = run_id
        return True

    def _pipeline_telemetry_snapshot(
        self, report: Mapping[str, Any]
    ) -> dict[str, Any] | None:
        telemetry = report.get("pipeline_telemetry_provider")
        expected_run_id = report.get("pipeline_telemetry_run_id")
        if (
            not isinstance(expected_run_id, str)
            or not expected_run_id
            or getattr(telemetry, "run_id", None) != expected_run_id
        ):
            return None
        latest = getattr(telemetry, "last_snapshot", None)
        if (
            isinstance(latest, Mapping)
            and latest.get("run_id") == expected_run_id
        ):
            return dict(latest)
        try:
            snapshot = telemetry.snapshot()
        except Exception:
            logger.debug("Could not snapshot pipeline telemetry.", exc_info=True)
            return None
        return (
            dict(snapshot)
            if isinstance(snapshot, Mapping)
            and snapshot.get("run_id") == expected_run_id
            else None
        )

    def mark_pipeline_run_report_failed(self, error: BaseException) -> None:
        current = self._current_batch_report
        if not current:
            return
        current["pipeline_report_reason_code"] = "terminal_report_build_failed"
        current["run_report_error"] = {
            "error_type": type(error).__name__,
            "message": str(error)[:1024],
        }

    def register_batch_success(self, image_path: str) -> None:
        report = self._current_batch_report
        if report and image_path in report["path_set"]:
            report["completed_paths"].add(image_path)

    def register_batch_output(self, image_path: str, output_path: str) -> bool:
        report = self._current_batch_report
        if (
            not report
            or image_path not in report["path_set"]
            or not isinstance(output_path, str)
            or not os.path.isfile(output_path)
        ):
            return False
        report["batch_output_paths"][image_path] = output_path
        report["completed_paths"].add(image_path)
        return True

    def _legacy_output_run_report(
        self,
        report: Mapping[str, Any],
        *,
        application_cleanup_status: str,
        archive_status: str,
    ) -> dict[str, Any] | None:
        if report.get("pipeline_report_reason_code"):
            return None
        paths = list(report.get("paths", []))
        output_paths = report.get("batch_output_paths")
        if not paths or not isinstance(output_paths, Mapping):
            return None
        if set(output_paths) != set(paths) or any(
            not isinstance(output_paths[path], str)
            or not os.path.isfile(output_paths[path])
            for path in paths
        ):
            return None
        telemetry = self._pipeline_telemetry_snapshot(report)
        return build_run_report(
            telemetry=telemetry,
            total_wall_sec=None,
            page_outcomes=[
                {
                    "source_unit_index": index,
                    "image_path": path,
                    "image_name": os.path.basename(path),
                    "output_path": output_paths[path],
                    "output_status": "measured",
                    "quality_status": "completed",
                }
                for index, path in enumerate(paths)
            ],
            output_summary={
                "input_count": len(paths),
                "output_count": len(paths),
                "fallback_count": 0,
                "fallbacks": [],
                "missing": [],
            },
            started_at_local=str(report.get("started_at_local") or ""),
            run_id=str(report.get("run_report_id") or ""),
            terminal_status="completed",
            processing_status="completed",
            report_status="completed",
            cleanup_status="not_applicable",
            pipeline_cleanup_status="not_applicable",
            application_cleanup_status=application_cleanup_status,
            archive_status=archive_status,
            telemetry_reason_code="pipeline_telemetry_not_observed",
        )

    def _previous_cancelled_unresolved_paths(
        self,
        previous: Mapping[str, Any],
        *,
        run_paths: set[str],
        current_paths: set[str],
    ) -> list[str]:
        if not previous.get("was_cancelled"):
            return []
        explicit_outcomes = {
            item["image_path"]: item["status"]
            for item in _page_outcome_summaries(previous.get("page_outcomes"))
        }
        return [
            path
            for path in previous.get("paths", [])
            if path in current_paths
            and path not in run_paths
            and explicit_outcomes.get(path) != "completed"
        ]

    def auto_export_ineligibility_reason(
        self,
        *,
        processing_status: str,
        application_cleanup_status: str,
        archive_status: str,
    ) -> str | None:
        report = self._current_batch_report
        if not report:
            return "batch_report_not_available"
        if processing_status != "completed":
            return "processing_not_completed"
        pipeline_data = report.get("pipeline_run_report_data")
        if not isinstance(pipeline_data, Mapping):
            pipeline_data = self._legacy_output_run_report(
                report,
                application_cleanup_status=application_cleanup_status,
                archive_status=archive_status,
            )
            if pipeline_data is not None:
                self.register_pipeline_run_report(pipeline_data)
        if not isinstance(pipeline_data, Mapping):
            return "pipeline_outcome_not_observed"
        if report.get("pipeline_processing_status") != "completed":
            return "processing_not_completed"
        if report.get("pipeline_cleanup_status") not in {"completed", "not_applicable"} or application_cleanup_status not in {
            "completed",
            "not_applicable",
        }:
            return "cleanup_not_completed"
        if archive_status not in {"completed", "not_applicable"}:
            return "archive_not_completed"
        if report.get("skipped") or report.get("preflight_errors"):
            return "run_has_unresolved_pages"

        run_paths = set(report.get("paths", []))
        previous = report.get("previous_report") or {}
        for item in _page_outcome_summaries(previous.get("page_outcomes")):
            if item["image_path"] not in run_paths and item["image_path"] in self.main.image_files:
                if item["status"] != "completed":
                    return "previous_run_has_unresolved_pages"
        for image_path in previous.get("retry_paths", []):
            if image_path in self.main.image_files and image_path not in run_paths:
                return "previous_run_has_unresolved_pages"
        for entry in previous.get("skipped_entries", []):
            image_path = entry.get("image_path")
            if image_path in self.main.image_files and image_path not in run_paths:
                return "previous_run_has_unresolved_pages"
        if self._previous_cancelled_unresolved_paths(
            previous,
            run_paths=run_paths,
            current_paths=set(self.main.image_files),
        ):
            return "previous_run_has_unresolved_pages"

        quality = pipeline_data.get("quality")
        if not isinstance(quality, Mapping) or quality.get("status") != "completed":
            return "page_quality_not_completed"
        output = pipeline_data.get("output")
        run_path_count = len(report.get("paths", []))
        if (
            not isinstance(output, Mapping)
            or output.get("input_count") != run_path_count
            or output.get("output_count") != run_path_count
            or output.get("missing")
        ):
            return "page_quality_not_completed"
        for metric in (
            "failed_units",
            "unjudged_units",
            "unmeasured_units",
            "source_fallback_units",
        ):
            value = (quality.get(metric) or {}).get("value")
            if value != 0:
                return "page_quality_not_completed"

        pages = pipeline_data.get("pages")
        if not isinstance(pages, list):
            return "pipeline_outcome_not_observed"
        pages_by_path = {
            item["image_path"]: item
            for item in pages
            if isinstance(item, Mapping) and isinstance(item.get("image_path"), str)
        }
        pages_by_index = {
            item.get("source_unit_index", index): item
            for index, item in enumerate(pages)
            if isinstance(item, Mapping)
            and isinstance(item.get("source_unit_index", index), int)
            and not isinstance(item.get("source_unit_index", index), bool)
        }
        for index, image_path in enumerate(report.get("paths", [])):
            page = pages_by_path.get(image_path)
            if page is None:
                legacy_page = pages_by_index.get(index)
                if legacy_page is not None and "image_path" not in legacy_page:
                    page = legacy_page
            if (
                not isinstance(page, Mapping)
                or page.get("quality_status") != "completed"
                or page.get("output_status", "measured" if page.get("output_path") else "unmeasured")
                != "measured"
                or page.get("fallback_kind") in {"source", "source-reloaded"}
            ):
                return "page_quality_not_completed"
        return None

    def register_preflight_error(self, title: str, details: str = "") -> None:
        report = self._current_batch_report
        if not report:
            return
        normalized_title = str(title or "").strip() or self._tr("Setup Error")
        normalized_details = self._sanitize_batch_skip_error(details)
        key = (normalized_title, normalized_details)
        if key in report["preflight_seen"]:
            return
        report["preflight_seen"].add(key)
        report["preflight_errors"].append(
            {
                "title": normalized_title,
                "details": normalized_details,
            }
        )

    def register_preflight_warning(self, title: str, details: str = "") -> None:
        report = self._current_batch_report
        if not report:
            return
        normalized_title = str(title or "").strip() or self._tr("Setup Warning")
        normalized_details = self._sanitize_batch_skip_error(details)
        key = (normalized_title, normalized_details)
        if key in report["preflight_seen"]:
            return
        report["preflight_seen"].add(key)
        report["preflight_warnings"].append(
            {"title": normalized_title, "details": normalized_details}
        )

    def _sanitize_batch_skip_error(self, error: str) -> str:
        if not error:
            return ""
        for raw_line in str(error).splitlines():
            line = raw_line.strip()
            if not line:
                continue
            if line.lower().startswith("traceback"):
                break
            if line.startswith('File "') or line.startswith("File '"):
                continue
            if line.startswith("During handling of the above exception"):
                continue
            if len(line) > 180:
                return f"{line[:177]}..."
            return line
        return ""

    def _is_content_flagged_error_text(self, error: str) -> bool:
        lowered = (error or "").lower()
        return (
            "flagged as unsafe" in lowered
            or "content was flagged" in lowered
            or "safety filters" in lowered
        )

    def _is_user_skip_reason(self, skip_reason: str, error: str = "") -> bool:
        text = f"{skip_reason} {error}".lower()
        return ("user-skipped" in text) or ("user skipped" in text)

    def _is_no_text_detection_skip(self, skip_reason: str) -> bool:
        return skip_reason == "Text Blocks"

    def _localize_batch_skip_detail(self, error: str) -> str:
        summary = self._sanitize_batch_skip_error(error)
        if not summary:
            return ""

        lowered = summary.lower()
        if "out of bounds" in lowered and "index" in lowered:
            return QtCore.QCoreApplication.translate(
                "BatchReportController",
                "Mask boundary exceeded image bounds",
            )
        if self._is_content_flagged_error_text(summary):
            return self._tr("The AI provider flagged this content")
        if "timed out" in lowered or "timeout" in lowered:
            return self._tr("Request timed out")
        if (
            "too many requests" in lowered
            or "rate limit" in lowered
            or "429" in lowered
        ):
            return self._tr("Rate limited by provider")
        if (
            "unauthorized" in lowered
            or "invalid api key" in lowered
            or "401" in lowered
            or "403" in lowered
        ):
            return self._tr("Authentication failed")
        if (
            "connection" in lowered
            or "network" in lowered
            or "name or service not known" in lowered
            or "failed to establish" in lowered
        ):
            return self._tr("Network or connection error")
        if (
            "bad gateway" in lowered
            or "service unavailable" in lowered
            or "gateway timeout" in lowered
            or "502" in lowered
            or "503" in lowered
            or "504" in lowered
        ):
            return self._tr("Provider unavailable")
        if (
            "jsondecodeerror" in lowered
            or "empty json" in lowered
            or "expecting value" in lowered
        ):
            return self._tr("Invalid translation response")
        return self._tr("Unexpected tool error")

    def _localize_batch_skip_action(self, skip_reason: str, error: str) -> str:
        summary = self._sanitize_batch_skip_error(error)
        lowered = summary.lower()

        if self._is_content_flagged_error_text(summary):
            if "ocr" in (skip_reason or "").lower():
                return self._tr("Try another text recognition tool")
            if "translator" in (skip_reason or "").lower() or "translation" in (
                skip_reason or ""
            ).lower():
                return self._tr("Try another translator")
            return self._tr("Try another tool")
        if "timed out" in lowered or "timeout" in lowered:
            return self._tr("Try again")
        if (
            "too many requests" in lowered
            or "rate limit" in lowered
            or "429" in lowered
        ):
            return self._tr("Wait and try again")
        if (
            "unauthorized" in lowered
            or "invalid api key" in lowered
            or "401" in lowered
            or "403" in lowered
        ):
            return self._tr("Check API settings")
        if (
            "connection" in lowered
            or "network" in lowered
            or "name or service not known" in lowered
            or "failed to establish" in lowered
        ):
            return self._tr("Check your connection")
        if (
            "bad gateway" in lowered
            or "service unavailable" in lowered
            or "gateway timeout" in lowered
            or "502" in lowered
            or "503" in lowered
            or "504" in lowered
        ):
            return self._tr("Try again later")
        if (
            "jsondecodeerror" in lowered
            or "empty json" in lowered
            or "expecting value" in lowered
        ):
            return self._tr("Try again")

        if skip_reason == "Text Blocks":
            return ""
        if "inpaint" in (skip_reason or "").lower():
            return QtCore.QCoreApplication.translate(
                "BatchReportController",
                "Try rerunning inpainting",
            )
        if "ocr" in (skip_reason or "").lower():
            return self._tr("Try another text recognition tool")
        if "translator" in (skip_reason or "").lower() or "translation" in (
            skip_reason or ""
        ).lower():
            return self._tr("Try another translator")
        return self._tr("Try again")

    def _format_batch_skip_reason(self, skip_reason: str, error: str) -> str:
        detail = self._localize_batch_skip_detail(error)
        action = self._localize_batch_skip_action(skip_reason, error)
        reason_map = {
            "Text Blocks": self._tr("No text blocks detected"),
            "OCR": self._tr("Text recognition failed"),
            "inpaint": QtCore.QCoreApplication.translate(
                "BatchReportController",
                "Inpainting failed",
            ),
            "Inpaint": QtCore.QCoreApplication.translate(
                "BatchReportController",
                "Inpainting failed",
            ),
            "Translator": self._tr("Translation failed"),
            "OCR Chunk Failed": self._tr("Webtoon text recognition chunk failed"),
            "Translation": self._tr("Webtoon translation chunk failed"),
            "Translation Chunk Failed": self._tr(
                "Webtoon translation chunk failed"
            ),
        }
        base_reason = reason_map.get(skip_reason, self._tr("Page processing failed"))
        reason = base_reason
        if detail:
            reason = f"{base_reason}: {detail}"
        if action:
            reason = f"{reason}. {action}."
        return reason

    def register_batch_skip(self, image_path: str, skip_reason: str, error: str):
        report = self._current_batch_report
        if not report:
            return
        if image_path not in report["path_set"]:
            return
        if self._is_user_skip_reason(skip_reason, error):
            return
        if self._is_no_text_detection_skip(skip_reason):
            return

        reason_text = self._format_batch_skip_reason(skip_reason, error)
        existing = report["skipped"].get(image_path)
        if existing is None:
            report["skipped"][image_path] = {
                "image_path": image_path,
                "image_name": os.path.basename(image_path),
                "stages": [skip_reason] if skip_reason else [],
                "errors": [self._sanitize_batch_skip_error(error)] if error else [],
                "reasons": [reason_text] if reason_text else [],
            }
            return

        if skip_reason and skip_reason not in existing.setdefault("stages", []):
            existing["stages"].append(skip_reason)
        sanitized_error = self._sanitize_batch_skip_error(error)
        if sanitized_error and sanitized_error not in existing.setdefault("errors", []):
            existing["errors"].append(sanitized_error)
        if reason_text and reason_text not in existing["reasons"]:
            existing["reasons"].append(reason_text)

    def finalize_batch_report(
        self,
        was_cancelled: bool,
        *,
        processing_status: str | None = None,
        processing_error: Mapping[str, Any] | None = None,
        application_cleanup_status: str = "not_observed",
        application_cleanup_errors: list[Mapping[str, Any]] | None = None,
        archive_status: str = "not_applicable",
        auto_export_result: Mapping[str, Any] | None = None,
    ):
        report = self._current_batch_report
        self._current_batch_report = None
        if not report:
            return None

        worker_status = processing_status or (
            "cancelled" if was_cancelled else "completed"
        )
        pipeline_data = report.get("pipeline_run_report_data")
        pipeline_data = dict(pipeline_data) if isinstance(pipeline_data, Mapping) else None
        pipeline_status = str(report.get("pipeline_processing_status") or "")
        processing_status = (
            "interrupted"
            if worker_status == "interrupted"
            else pipeline_status or worker_status
        )
        if not isinstance(pipeline_data, Mapping) and processing_status == "completed":
            pipeline_data = self._legacy_output_run_report(
                report,
                application_cleanup_status=application_cleanup_status,
                archive_status=archive_status,
            )
            if pipeline_data is not None:
                self._store_pipeline_run_report(report, pipeline_data)
                pipeline_status = str(report.get("pipeline_processing_status") or "")
        processing_error = report.get("pipeline_processing_error") or processing_error
        if processing_error is None and report.get("preflight_errors"):
            first_preflight_error = report["preflight_errors"][0]
            processing_error = {
                "error_type": "PreflightError",
                "message": ": ".join(
                    value
                    for value in (
                        str(first_preflight_error.get("title") or ""),
                        str(first_preflight_error.get("details") or ""),
                    )
                    if value
                )[:1024],
            }
        pipeline_cleanup_status = str(report.get("pipeline_cleanup_status") or "not_observed")
        cleanup_status = (
            "failed"
            if "failed" in {pipeline_cleanup_status, application_cleanup_status}
            else "completed"
            if pipeline_cleanup_status in {"completed", "not_applicable"}
            and application_cleanup_status in {"completed", "not_applicable"}
            else "not_observed"
        )
        cleanup_errors = list(report.get("pipeline_cleanup_errors", []))
        cleanup_errors.extend(dict(item) for item in application_cleanup_errors or [])
        auto_export_result = dict(auto_export_result or {})
        auto_export_status = str(auto_export_result.get("status") or "not_applicable")
        auto_export_targets = [
            str(target)
            for target in auto_export_result.get("selected_targets", [])
            if isinstance(target, str)
        ]
        auto_export_completed_targets = [
            str(target)
            for target in auto_export_result.get("completed_targets", [])
            if isinstance(target, str)
        ]
        auto_export_failed_target = auto_export_result.get("failed_target")
        if not isinstance(auto_export_failed_target, str):
            auto_export_failed_target = None
        auto_export_reason_code = auto_export_result.get("reason_code")
        auto_export_error = auto_export_result.get("error")
        if not isinstance(auto_export_error, Mapping):
            auto_export_error = None

        previous = report.get("previous_report") or {}
        current_paths = set(self.main.image_files)
        run_paths = list(dict.fromkeys(report["paths"]))
        all_paths = list(dict.fromkeys(
            [path for path in previous.get("paths", []) if path in current_paths]
            + run_paths
        ))
        previous_outcomes = {
            item["image_path"]: item["status"]
            for item in _page_outcome_summaries(previous.get("page_outcomes"))
            if item.get("image_path") in current_paths
        }
        prior_cancelled_unresolved_paths = self._previous_cancelled_unresolved_paths(
            previous,
            run_paths=set(run_paths),
            current_paths=current_paths,
        )
        pipeline_pages = {}
        pipeline_pages_by_path = {}
        if isinstance(pipeline_data, Mapping):
            for index, item in enumerate(pipeline_data.get("pages", []) or []):
                source_index = item.get("source_unit_index", index)
                if isinstance(source_index, int) and not isinstance(source_index, bool):
                    pipeline_pages[source_index] = item
                image_path = item.get("image_path")
                if isinstance(image_path, str):
                    pipeline_pages_by_path[image_path] = item
        run_page_outcomes = []
        for index, image_path in enumerate(run_paths):
            skip = report["skipped"].get(image_path)
            page = pipeline_pages_by_path.get(image_path)
            indexed_page = pipeline_pages.get(index)
            identity_conflict = bool(
                indexed_page is not None
                and isinstance(indexed_page.get("image_path"), str)
                and indexed_page["image_path"] != image_path
            )
            if page is None and indexed_page is not None and not identity_conflict:
                page = indexed_page
            if skip:
                status = (
                    "review_required"
                    if any("review_required" in reason for reason in skip["reasons"])
                    else "failed"
                )
            elif identity_conflict:
                status = "unjudged"
            elif page is not None:
                quality_status = page.get("quality_status")
                status = (
                    "failed"
                    if quality_status == "failed"
                    else "review_required"
                    if quality_status == "review_required"
                    else "unjudged"
                    if quality_status != "completed"
                    or page.get("output_status", "measured" if page.get("output_path") else "unmeasured") != "measured"
                    or page.get("fallback_kind") in {"source", "source-reloaded"}
                    else "completed"
                )
            else:
                status = (
                    "completed"
                    if processing_status in {"completed", "failed"}
                    and image_path in report["completed_paths"]
                    and not identity_conflict
                    else "unjudged"
                )
            run_page_outcomes.append({"image_path": image_path, "status": status})

        page_statuses = {path: previous_outcomes.get(path, "unjudged") for path in all_paths}
        for item in run_page_outcomes:
            if item["status"] != "unjudged" or page_statuses[item["image_path"]] == "unjudged":
                page_statuses[item["image_path"]] = item["status"]
        page_outcomes = [
            {"image_path": path, "status": page_statuses[path]}
            for path in all_paths
        ]
        run_total_images = len(run_page_outcomes)
        run_completed_count = sum(item["status"] == "completed" for item in run_page_outcomes)
        run_failed_count = sum(item["status"] == "failed" for item in run_page_outcomes)
        run_review_count = sum(item["status"] == "review_required" for item in run_page_outcomes)
        run_unjudged_count = sum(item["status"] == "unjudged" for item in run_page_outcomes)
        total_images = len(all_paths)
        completed_count = sum(status == "completed" for status in page_statuses.values())
        failed_count = sum(status in {"failed", "review_required"} for status in page_statuses.values())
        unjudged_count = sum(status == "unjudged" for status in page_statuses.values())
        pending = {
            entry["image_path"]: entry
            for entry in previous.get("skipped_entries", [])
            if entry.get("image_path") in current_paths
        }
        for item in run_page_outcomes:
            if item["status"] == "completed":
                pending.pop(item["image_path"], None)
        pending.update(report["skipped"])
        skipped_entries = sorted(
            pending.values(),
            key=lambda entry: entry["image_name"].lower(),
        )
        skipped_count = len(skipped_entries)
        retry_paths = [
            image_path
            for image_path in all_paths
            if page_statuses[image_path] != "completed"
        ]
        pipeline_quality = (
            pipeline_data.get("quality", {})
            if isinstance(pipeline_data, Mapping)
            else {}
        )
        pipeline_quality_failed = (
            pipeline_quality.get("status") == "failed"
            or bool((pipeline_quality.get("failed_units") or {}).get("value"))
        )
        pipeline_quality_incomplete = (
            pipeline_data is None
            or pipeline_quality.get("status") not in {"completed", "failed"}
            or bool((pipeline_quality.get("unjudged_units") or {}).get("value"))
            or bool((pipeline_quality.get("unmeasured_units") or {}).get("value"))
            or bool((pipeline_quality.get("source_fallback_units") or {}).get("value"))
        )
        started_monotonic = report.get("started_monotonic")
        total_wall_sec = (
            round(max(time.monotonic() - started_monotonic, 0.0), 3)
            if isinstance(started_monotonic, (int, float))
            else None
        )
        terminal_status = (
            processing_status
            if processing_status in {"failed", "cancelled", "interrupted", "paused"}
            else "cleanup_failed"
            if cleanup_status == "failed"
            else "failed"
            if archive_status == "failed"
            or auto_export_status == "failed"
            or pipeline_quality_failed
            or run_failed_count
            else "incomplete"
            if processing_status != "completed"
            or cleanup_status != "completed"
            or archive_status not in {"completed", "not_applicable"}
            or auto_export_status in {"not_attempted", "not_observed"}
            or run_review_count
            or run_unjudged_count
            or prior_cancelled_unresolved_paths
            or pipeline_quality_incomplete
            else "completed"
        )

        run_report_log_dir = str(report.get("run_report_log_dir") or "")
        run_report_status = "not_configured" if not run_report_log_dir else "running"
        report_reason_code = report.get("pipeline_report_reason_code")
        if run_report_log_dir:
            try:
                if pipeline_data is None:
                    report_page_outcomes = []
                    for index, item in enumerate(run_page_outcomes):
                        quality_status = item["status"]
                        report_page_outcomes.append(
                            {
                                "source_unit_index": index,
                                "image_name": os.path.basename(item["image_path"]),
                                "output_path": None,
                                "quality_status": quality_status,
                            }
                        )
                    run_report = build_run_report(
                        telemetry=self._pipeline_telemetry_snapshot(report),
                        total_wall_sec=total_wall_sec,
                        page_outcomes=report_page_outcomes,
                        output_summary={
                            "input_count": run_total_images,
                            "output_count": None,
                            "fallback_count": None,
                            "fallbacks": None,
                            "missing": None,
                        },
                        started_at_local=report["started_at_local"],
                        run_id=str(report["run_report_id"]),
                        terminal_status="running",
                        processing_status=processing_status,
                        report_status="completed",
                        cleanup_status=cleanup_status,
                        pipeline_cleanup_status=pipeline_cleanup_status,
                        application_cleanup_status=application_cleanup_status,
                        archive_status=archive_status,
                        processing_error=processing_error,
                        cleanup_errors=cleanup_errors,
                        telemetry_reason_code="pipeline_telemetry_not_observed",
                    )
                else:
                    run_report = dict(pipeline_data)
                pipeline_wall_sec = run_report.get("total_wall_sec")
                run_report["pipeline_wall_sec"] = pipeline_wall_sec
                run_report["total_wall_sec"] = total_wall_sec
                run_report["total_wall_measurement"] = {
                    "value": total_wall_sec,
                    "unit": "seconds",
                    "status": "measured" if total_wall_sec is not None else "unavailable",
                    "reason_code": None if total_wall_sec is not None else "wall_time_not_observed",
                }
                run_report["total_wall_text"] = format_hms(total_wall_sec) if total_wall_sec is not None else "미측정"
                run_report["seconds_per_page"] = (
                    round(total_wall_sec / run_report["page_count"], 3)
                    if total_wall_sec is not None and run_report.get("page_count")
                    else None
                )
                run_report.update(
                    {
                        "run_id": str(report["run_report_id"]),
                        "started_at_local": report["started_at_local"],
                        "terminal_status": terminal_status,
                        "processing_status": processing_status,
                        "report_status": "completed",
                        "cleanup_status": cleanup_status,
                        "pipeline_cleanup_status": pipeline_cleanup_status,
                        "application_cleanup_status": application_cleanup_status,
                        "archive_status": archive_status,
                        "auto_export_status": auto_export_status,
                        "auto_export_targets": auto_export_targets,
                        "auto_export_completed_targets": auto_export_completed_targets,
                        "auto_export_failed_target": auto_export_failed_target,
                        "auto_export_reason_code": auto_export_reason_code,
                        "auto_export_error": (
                            dict(auto_export_error) if auto_export_error else None
                        ),
                        "processing_error": processing_error,
                        "cleanup_errors": cleanup_errors,
                        "start_record_status": report.get("start_record_status", "unknown"),
                        "start_record_reason_code": report.get("start_record_reason_code"),
                    }
                )
                run_report["terminal_status"] = terminal_status
                path = write_run_report(
                    run_report,
                    log_dir=run_report_log_dir,
                    run_id=str(report["run_report_id"]),
                )
                run_report_status = "completed" if path else "failed"
                if not path:
                    report_reason_code = "terminal_report_write_failed"
            except Exception:
                logger.warning("Could not build the terminal run report.", exc_info=True)
                run_report_status = "failed"
                report_reason_code = "terminal_report_build_failed"
        if run_report_status == "failed" and terminal_status not in {
            "failed", "cleanup_failed", "cancelled", "interrupted", "paused"
        }:
            terminal_status = "report_failed"
        elif run_report_status == "not_configured" and terminal_status == "completed":
            terminal_status = "incomplete"

        finalized = {
            "run_type": report.get("run_type", "batch"),
            "started_at": report["started_at"],
            "finished_at": datetime.now(),
            "was_cancelled": processing_status == "cancelled",
            "worker_status": worker_status,
            "processing_status": processing_status,
            "report_status": run_report_status,
            "cleanup_status": cleanup_status,
            "pipeline_cleanup_status": pipeline_cleanup_status,
            "application_cleanup_status": application_cleanup_status,
            "archive_status": archive_status,
            "auto_export_status": auto_export_status,
            "auto_export_targets": auto_export_targets,
            "auto_export_completed_targets": auto_export_completed_targets,
            "auto_export_failed_target": auto_export_failed_target,
            "auto_export_reason_code": auto_export_reason_code,
            "auto_export_error": _error_summary(auto_export_error),
            "terminal_status": terminal_status,
            "start_record_status": (
                "persisted"
                if report.get("start_record_status") == "running"
                else report.get("start_record_status", "unknown")
            ),
            "start_record_reason_code": report.get("start_record_reason_code"),
            "report_reason_code": report_reason_code,
            "processing_error": _error_summary(processing_error),
            "cleanup_errors": _cleanup_summaries(cleanup_errors),
            "total_images": total_images,
            "skipped_count": skipped_count,
            "completed_count": completed_count,
            "failed_count": failed_count,
            "unjudged_count": unjudged_count,
            "run_total_images": run_total_images,
            "run_completed_count": run_completed_count,
            "run_failed_count": run_failed_count,
            "run_unjudged_count": run_unjudged_count,
            "paths": all_paths,
            "page_outcomes": page_outcomes,
            "run_page_outcomes": run_page_outcomes,
            "skipped_entries": skipped_entries,
            "retry_paths": retry_paths,
            "preflight_errors": list(report.get("preflight_errors", [])),
            "preflight_warnings": list(report.get("preflight_warnings", [])),
        }
        self._latest_batch_report = finalized
        self.refresh_action_buttons()
        mark_dirty = getattr(self.main, "mark_project_dirty", None)
        if callable(mark_dirty):
            mark_dirty(render_dirty=False)
        return finalized

    def _open_image_from_batch_report(self, image_path: str):
        if image_path not in self.main.image_files:
            Messages.show_warning(
                self.main,
                self._tr("This image is not in the current project."),
                duration=5,
                closable=True,
                source="batch_report",
            )
            return

        index = self.main.image_files.index(image_path)
        self.main.show_main_page()
        self.main.page_list.setCurrentRow(index)

    def _open_report_row_image(self, table: QtWidgets.QTableWidget, row: int):
        item = table.item(row, 0)
        if item is None:
            return
        image_path = item.data(QtCore.Qt.ItemDataRole.UserRole)
        if not image_path:
            return
        self._open_image_from_batch_report(image_path)

    def _build_batch_report_widget(self, report: dict) -> QtWidgets.QWidget:
        container = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(container)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(10)

        terminal_status = report.get("terminal_status")
        if terminal_status == "completed":
            status_text = QtCore.QCoreApplication.translate(
                "BatchReportController", "Completed"
            )
        elif terminal_status == "cancelled":
            status_text = QtCore.QCoreApplication.translate(
                "BatchReportController", "Cancelled"
            )
        elif terminal_status == "failed":
            status_text = QtCore.QCoreApplication.translate(
                "BatchReportController", "Failed"
            )
        elif terminal_status == "interrupted":
            status_text = QtCore.QCoreApplication.translate(
                "BatchReportController", "Interrupted"
            )
        elif terminal_status == "paused":
            status_text = QtCore.QCoreApplication.translate(
                "BatchReportController", "Paused"
            )
        elif terminal_status == "report_failed":
            status_text = QtCore.QCoreApplication.translate(
                "BatchReportController", "Report failed"
            )
        elif terminal_status == "cleanup_failed":
            status_text = QtCore.QCoreApplication.translate(
                "BatchReportController", "Cleanup failed"
            )
        else:
            status_text = QtCore.QCoreApplication.translate(
                "BatchReportController", "Incomplete"
            )
        finished_at = self._normalize_datetime(report.get("finished_at"))
        finished = (
            finished_at.strftime("%Y-%m-%d %H:%M")
            if finished_at is not None
            else self._tr("Unknown")
        )
        run_type_labels = {
            "batch": self._tr("Batch"),
            "retry_failed": self._tr("Retry Failed"),
            "one_page_auto": self._tr("One-Page Auto"),
        }
        run_type_text = run_type_labels.get(report.get("run_type", "batch"), self._tr("Batch"))
        meta_label = QtWidgets.QLabel(
            self._tr("{0}  |  {1}  |  Updated {2}").format(run_type_text, status_text, finished)
        )
        meta_label.setStyleSheet("color: rgba(130,130,130,0.95);")
        layout.addWidget(meta_label)

        stats_layout = QtWidgets.QHBoxLayout()
        stats_layout.setSpacing(8)

        def make_stat_card(label: str, value: str) -> QtWidgets.QFrame:
            card = QtWidgets.QFrame()
            card.setObjectName("batchStatCard")
            card_layout = QtWidgets.QVBoxLayout(card)
            card_layout.setContentsMargins(10, 8, 10, 8)
            card_layout.setSpacing(2)

            value_label = QtWidgets.QLabel(value)
            value_label.setObjectName("batchStatValue")
            value_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)

            label_widget = QtWidgets.QLabel(label)
            label_widget.setObjectName("batchStatLabel")
            label_widget.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)

            card_layout.addWidget(value_label)
            card_layout.addWidget(label_widget)
            return card

        stats_layout.addWidget(make_stat_card(self._tr("Total"), str(report["total_images"])))
        stats_layout.addWidget(make_stat_card(self._tr("Skipped"), str(report["skipped_count"])))
        stats_layout.addWidget(
            make_stat_card(
                self._tr("Setup Errors"),
                str(len(report.get("preflight_errors", []))),
            )
        )
        stats_layout.addWidget(
            make_stat_card(
                self._tr("Setup Warnings"),
                str(len(report.get("preflight_warnings", []))),
            )
        )
        layout.addLayout(stats_layout)

        preflight_warnings = list(report.get("preflight_warnings", []))
        if preflight_warnings:
            warning_header = QtWidgets.QLabel(
                self._tr("Setup Warnings ({0})").format(len(preflight_warnings))
            )
            warning_header.setStyleSheet("font-weight: 600;")
            layout.addWidget(warning_header)
            warning_list = QtWidgets.QListWidget()
            warning_list.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.NoSelection)
            visible_rows = min(max(len(preflight_warnings), 1), 6)
            warning_list.setMinimumHeight((visible_rows * 42) + 12)
            warning_list.setMaximumHeight((visible_rows * 42) + 12)
            for entry in preflight_warnings:
                title = str(entry.get("title", "") or self._tr("Setup Warning"))
                details = str(entry.get("details", "") or "")
                item_text = title if not details else f"{title}\n{details}"
                item = QtWidgets.QListWidgetItem(item_text)
                item.setToolTip(item_text)
                warning_list.addItem(item)
            layout.addWidget(warning_list)

        preflight_errors = list(report.get("preflight_errors", []))
        if preflight_errors:
            preflight_header = QtWidgets.QLabel(
                self._tr("Setup / Runtime Errors ({0})").format(len(preflight_errors))
            )
            preflight_header.setStyleSheet("font-weight: 600;")
            layout.addWidget(preflight_header)
            preflight_list = QtWidgets.QListWidget()
            preflight_list.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.NoSelection)
            visible_rows = min(max(len(preflight_errors), 1), 6)
            preflight_list.setMinimumHeight((visible_rows * 42) + 12)
            preflight_list.setMaximumHeight((visible_rows * 42) + 12)
            for entry in preflight_errors:
                title = str(entry.get("title", "") or self._tr("Setup Error"))
                details = str(entry.get("details", "") or "")
                item_text = title if not details else f"{title}\n{details}"
                item = QtWidgets.QListWidgetItem(item_text)
                item.setToolTip(item_text)
                preflight_list.addItem(item)
            layout.addWidget(preflight_list)

        container.setStyleSheet(
            "QFrame#batchStatCard { border: 1px solid rgba(128,128,128,0.35); border-radius: 8px; }"
            "QLabel#batchStatValue { font-size: 18px; font-weight: 600; }"
            "QLabel#batchStatLabel { color: rgba(140,140,140,0.95); font-size: 11px; }"
        )

        header_label = QtWidgets.QLabel(
            self._tr("Skipped Images ({0})").format(report["skipped_count"])
        )
        header_label.setStyleSheet("font-weight: 600;")
        layout.addWidget(header_label)

        skipped_entries = report["skipped_entries"]
        if skipped_entries:
            hint = QtWidgets.QLabel(self._tr("Double-click a row to open that page."))
            hint.setWordWrap(True)
            layout.addWidget(hint)

            table = QtWidgets.QTableWidget(len(skipped_entries), 2)
            table.setHorizontalHeaderLabels([self._tr("Image"), self._tr("Reason")])
            table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
            table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
            table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
            table.verticalHeader().setVisible(False)
            table.horizontalHeader().setStretchLastSection(True)
            table.horizontalHeader().setSectionResizeMode(
                0, QtWidgets.QHeaderView.ResizeMode.ResizeToContents
            )
            table.horizontalHeader().setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeMode.Stretch)
            table.setWordWrap(False)
            table.setVerticalScrollMode(QtWidgets.QAbstractItemView.ScrollMode.ScrollPerPixel)
            table.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded)
            visible_rows = min(max(len(skipped_entries), 4), 12)
            table_height = (visible_rows * 28) + 36
            table.setMinimumHeight(table_height)
            table.setMaximumHeight(table_height)

            for row, entry in enumerate(skipped_entries):
                image_item = QtWidgets.QTableWidgetItem(entry["image_name"])
                image_item.setData(QtCore.Qt.ItemDataRole.UserRole, entry["image_path"])
                image_item.setToolTip(entry["image_path"])
                table.setItem(row, 0, image_item)
                reason_item = QtWidgets.QTableWidgetItem(
                    "; ".join(entry["reasons"]) or self._tr("Skipped")
                )
                reason_item.setToolTip(reason_item.text())
                table.setItem(row, 1, reason_item)

            table.itemDoubleClicked.connect(
                lambda item, t=table: self._open_report_row_image(t, item.row())
            )
            layout.addWidget(table)
        else:
            empty_label = QtWidgets.QLabel(self._tr("No skipped images in this batch."))
            empty_label.setWordWrap(True)
            layout.addWidget(empty_label)

        layout.addStretch()
        return container

    def show_latest_batch_report(self):
        report = self._latest_batch_report
        if report is None:
            Messages.show_info(
                self.main,
                self._tr("No batch report is available yet."),
                duration=5,
                closable=True,
                source="batch_report",
            )
            return

        if self._batch_report_drawer is not None:
            try:
                self._batch_report_drawer.close()
            except Exception:
                pass
            self._batch_report_drawer = None

        drawer = MDrawer(self._tr("Batch Report"), parent=self.main).right()
        drawer.setFixedWidth(max(460, int(self.main.width() * 0.42)))
        drawer.set_widget(self._build_batch_report_widget(report))
        drawer.sig_closed.connect(lambda: setattr(self, "_batch_report_drawer", None))
        self._batch_report_drawer = drawer
        drawer.show()

    def shutdown(self):
        try:
            if self._batch_report_drawer is not None:
                self._batch_report_drawer.close()
                self._batch_report_drawer = None
        except Exception:
            pass

from __future__ import annotations

import os
from datetime import datetime
from typing import TYPE_CHECKING

from PySide6 import QtCore, QtWidgets

from app.ui.dayu_widgets.drawer import MDrawer
from app.ui.dayu_widgets.message import MMessage

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
            "paths": list(report.get("paths", [])),
            "retry_paths": list(report.get("retry_paths", [])),
            "skipped_entries": list(report.get("skipped_entries", [])),
        }

    def import_latest_report_from_project(self, payload: dict | None, refresh: bool = True) -> None:
        if not payload:
            self._latest_batch_report = None
            if refresh:
                self.refresh_action_buttons()
            return

        self._latest_batch_report = {
            "run_type": payload.get("run_type", "batch"),
            "started_at": self._normalize_datetime(payload.get("started_at")),
            "finished_at": self._normalize_datetime(payload.get("finished_at")),
            "was_cancelled": bool(payload.get("was_cancelled", False)),
            "total_images": int(payload.get("total_images", 0) or 0),
            "skipped_count": int(payload.get("skipped_count", 0) or 0),
            "completed_count": int(payload.get("completed_count", 0) or 0),
            "paths": list(payload.get("paths", [])),
            "retry_paths": list(payload.get("retry_paths", [])),
            "skipped_entries": list(payload.get("skipped_entries", [])),
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

    def start_batch_report(self, batch_paths: list[str], run_type: str = "batch"):
        tracked_paths = [
            path
            for path in batch_paths
            if not self.main.image_states.get(path, {}).get("skip", False)
        ]
        self._current_batch_report = {
            "run_type": run_type,
            "started_at": datetime.now(),
            "paths": tracked_paths,
            "path_set": set(tracked_paths),
            "skipped": {},
        }

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
            "Translator": self._tr("Translation failed"),
            "OCR Chunk Failed": self._tr("Webtoon text recognition chunk failed"),
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
                "reasons": [reason_text] if reason_text else [],
            }
            return

        if reason_text and reason_text not in existing["reasons"]:
            existing["reasons"].append(reason_text)

    def finalize_batch_report(self, was_cancelled: bool):
        report = self._current_batch_report
        self._current_batch_report = None
        if not report:
            return None

        total_images = len(report["paths"])
        skipped_entries = sorted(
            report["skipped"].values(),
            key=lambda entry: entry["image_name"].lower(),
        )
        skipped_count = len(skipped_entries)
        retry_paths = [
            image_path
            for image_path in report["paths"]
            if image_path in report["skipped"]
        ]

        finalized = {
            "run_type": report.get("run_type", "batch"),
            "started_at": report["started_at"],
            "finished_at": datetime.now(),
            "was_cancelled": bool(was_cancelled),
            "total_images": total_images,
            "skipped_count": skipped_count,
            "completed_count": max(0, total_images - skipped_count),
            "paths": list(report["paths"]),
            "skipped_entries": skipped_entries,
            "retry_paths": retry_paths,
        }
        self._latest_batch_report = finalized
        self.refresh_action_buttons()
        return finalized

    def _open_image_from_batch_report(self, image_path: str):
        if image_path not in self.main.image_files:
            MMessage.warning(
                text=self._tr("This image is not in the current project."),
                parent=self.main,
                duration=5,
                closable=True,
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

        status_text = self._tr("Cancelled") if report["was_cancelled"] else self._tr("Completed")
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
        layout.addLayout(stats_layout)

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
            MMessage.info(
                text=self._tr("No batch report is available yet."),
                parent=self.main,
                duration=5,
                closable=True,
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

from __future__ import annotations

from PySide6 import QtCore, QtWidgets

from app.ui.dayu_widgets.label import MLabel
from app.ui.dayu_widgets.progress_bar import MProgressBar


class AutomaticProgressDialog(QtWidgets.QDialog):
    cancel_requested = QtCore.Signal()
    retry_requested = QtCore.Signal()
    open_settings_requested = QtCore.Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(self.tr("Automatic Translation Progress"))
        self.setModal(True)
        self.setWindowModality(QtCore.Qt.ApplicationModal)
        self.resize(640, 520)
        self._build_ui()
        self.set_running_state()

    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(12)

        self.title_label = MLabel(self.tr("자동번역 준비 중")).h3()
        self.subtitle_label = MLabel(self.tr("초기화 중...")).secondary()
        self.subtitle_label.setWordWrap(True)
        layout.addWidget(self.title_label)
        layout.addWidget(self.subtitle_label)

        self.progress_bar = MProgressBar().normal()
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)

        metrics_grid = QtWidgets.QGridLayout()
        metrics_grid.setHorizontalSpacing(16)
        metrics_grid.setVerticalSpacing(8)
        self.elapsed_value = self._add_metric(metrics_grid, 0, self.tr("경과 시간"))
        self.remaining_value = self._add_metric(metrics_grid, 1, self.tr("남은 시간"))
        self.finish_value = self._add_metric(metrics_grid, 2, self.tr("예상 완료 시각"))
        self.confidence_value = self._add_metric(metrics_grid, 3, self.tr("ETA 신뢰도"))
        layout.addLayout(metrics_grid)

        work_group = QtWidgets.QGroupBox(self.tr("현재 작업"))
        work_layout = QtWidgets.QFormLayout(work_group)
        work_layout.setContentsMargins(12, 12, 12, 12)
        self.service_value = MLabel("-")
        self.page_value = MLabel("-")
        self.stage_value = MLabel("-")
        self.image_value = MLabel("-")
        work_layout.addRow(self.tr("서비스"), self.service_value)
        work_layout.addRow(self.tr("페이지"), self.page_value)
        work_layout.addRow(self.tr("단계"), self.stage_value)
        work_layout.addRow(self.tr("파일명"), self.image_value)
        layout.addWidget(work_group)

        self.log_group = QtWidgets.QGroupBox(self.tr("세부 로그"))
        log_layout = QtWidgets.QVBoxLayout(self.log_group)
        log_layout.setContentsMargins(12, 12, 12, 12)
        self.log_view = QtWidgets.QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(200)
        log_layout.addWidget(self.log_view)
        layout.addWidget(self.log_group, 1)

        self.error_label = MLabel("").secondary()
        self.error_label.setWordWrap(True)
        self.error_label.hide()
        layout.addWidget(self.error_label)

        button_layout = QtWidgets.QHBoxLayout()
        button_layout.addStretch(1)
        self.retry_button = QtWidgets.QPushButton(self.tr("재시도"))
        self.settings_button = QtWidgets.QPushButton(self.tr("Settings 열기"))
        self.cancel_button = QtWidgets.QPushButton(self.tr("Cancel"))
        self.close_button = QtWidgets.QPushButton(self.tr("닫기"))
        self.retry_button.clicked.connect(self.retry_requested.emit)
        self.settings_button.clicked.connect(self.open_settings_requested.emit)
        self.cancel_button.clicked.connect(self.cancel_requested.emit)
        self.close_button.clicked.connect(self.hide)
        button_layout.addWidget(self.retry_button)
        button_layout.addWidget(self.settings_button)
        button_layout.addWidget(self.cancel_button)
        button_layout.addWidget(self.close_button)
        layout.addLayout(button_layout)

    def _add_metric(self, grid: QtWidgets.QGridLayout, column: int, label_text: str) -> MLabel:
        label = MLabel(label_text).secondary()
        value = MLabel(self.tr("Calculating"))
        grid.addWidget(label, 0, column)
        grid.addWidget(value, 1, column)
        return value

    def set_running_state(self) -> None:
        self.error_label.hide()
        self.retry_button.hide()
        self.settings_button.hide()
        self.cancel_button.show()
        self.close_button.hide()
        self.progress_bar.normal()

    def set_error_state(self, detail: str) -> None:
        self.error_label.setText(detail)
        self.error_label.show()
        self.retry_button.show()
        self.settings_button.show()
        self.cancel_button.hide()
        self.close_button.show()
        self.progress_bar.error()

    def set_done_state(self, success_message: str | None = None) -> None:
        if success_message:
            self.error_label.setText(success_message)
            self.error_label.show()
        self.retry_button.hide()
        self.settings_button.hide()
        self.cancel_button.hide()
        self.close_button.show()
        self.progress_bar.success()

    def set_cancelled_state(self) -> None:
        self.error_label.setText(self.tr("작업이 취소되었습니다."))
        self.error_label.show()
        self.retry_button.hide()
        self.settings_button.hide()
        self.cancel_button.hide()
        self.close_button.show()

    def append_log(self, message: str) -> None:
        if not message:
            return
        self.log_view.appendPlainText(message)
        bar = self.log_view.verticalScrollBar()
        bar.setValue(bar.maximum())

    def update_event(self, event: dict) -> None:
        phase = str(event.get("phase") or "")
        status = str(event.get("status") or "")
        message = str(event.get("message") or "")
        service = str(event.get("service") or "-")
        stage_name = str(event.get("stage_name") or event.get("step_key") or "-")
        page_total = int(event.get("page_total") or 0)
        page_index = event.get("page_index")
        image_name = str(event.get("image_name") or "-")
        progress_percent = float(event.get("overall_progress_percent") or 0.0)

        if phase in {"gemma_startup", "ocr_startup"}:
            self.title_label.setText(self.tr("자동번역 준비 중"))
            self.progress_bar.setRange(0, 0)
            if phase == "gemma_startup":
                self.service_value.setText("Gemma")
            elif phase == "ocr_startup":
                self.service_value.setText("PaddleOCR VL")
        elif phase == "pipeline":
            self.title_label.setText(self.tr("자동번역 진행 중"))
            self.progress_bar.setRange(0, 1000)
            self.progress_bar.setValue(int(progress_percent * 10))
            self.service_value.setText(service or "batch")
        elif phase == "done":
            self.progress_bar.setRange(0, 1000)
            self.progress_bar.setValue(1000)

        self.subtitle_label.setText(message or self.subtitle_label.text())
        self.elapsed_value.setText(str(event.get("elapsed_text") or self.tr("Calculating")))
        self.remaining_value.setText(str(event.get("eta_text") or self.tr("Calculating")))
        self.finish_value.setText(str(event.get("eta_finish_at_local") or self.tr("Calculating")))
        self.confidence_value.setText(str(event.get("eta_confidence") or self.tr("Calculating")))
        if page_total > 0 and page_index is not None:
            self.page_value.setText(f"{int(page_index) + 1}/{page_total}")
        else:
            self.page_value.setText("-")
        self.stage_value.setText(stage_name or "-")
        self.image_value.setText(image_name or "-")

        detail = str(event.get("detail") or "")
        log_line = message or detail
        if log_line:
            self.append_log(log_line)

        if status == "failed":
            combined = detail or message or self.tr("작업이 실패했습니다.")
            self.set_error_state(combined)
        elif phase == "done":
            self.set_done_state(message or self.tr("자동번역이 완료되었습니다."))
        else:
            self.set_running_state()

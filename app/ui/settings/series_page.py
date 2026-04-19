from PySide6 import QtCore, QtWidgets

from ..dayu_widgets.check_box import MCheckBox
from ..dayu_widgets.combo_box import MComboBox
from ..dayu_widgets.label import MLabel
from ..dayu_widgets.spin_box import MSpinBox


class SeriesPage(QtWidgets.QWidget):
    changed = QtCore.Signal()

    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QtWidgets.QVBoxLayout(self)

        title_label = MLabel(self.tr("Series Queue")).h4()
        note_label = MLabel(
            self.tr(
                "Configure the default behavior for `.seriesctpr` queue execution.\n"
                "These values are copied into new series projects and can be adjusted per series."
            )
        ).secondary()
        note_label.setWordWrap(True)

        self.failure_policy_combo = MComboBox().medium()
        self.failure_policy_combo.addItem(self.tr("Stop"), "stop")
        self.failure_policy_combo.addItem(self.tr("Skip"), "skip")
        self.failure_policy_combo.addItem(self.tr("Retry"), "retry")

        self.retry_count_spinbox = MSpinBox().small()
        self.retry_count_spinbox.setMinimum(0)
        self.retry_count_spinbox.setMaximum(10)
        self.retry_delay_spinbox = MSpinBox().small()
        self.retry_delay_spinbox.setMinimum(0)
        self.retry_delay_spinbox.setMaximum(600)

        self.auto_open_failed_checkbox = MCheckBox(self.tr("Open failed child project automatically"))
        self.resume_first_incomplete_checkbox = MCheckBox(
            self.tr("Resume from the first incomplete queue item")
        )
        self.return_to_series_checkbox = MCheckBox(
            self.tr("Return to the series board after a child project finishes")
        )

        self.auto_open_failed_checkbox.setChecked(True)
        self.resume_first_incomplete_checkbox.setChecked(True)
        self.return_to_series_checkbox.setChecked(True)

        form = QtWidgets.QFormLayout()
        form.setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignLeft)
        form.addRow(self.tr("Failure policy:"), self.failure_policy_combo)
        form.addRow(self.tr("Retry count:"), self.retry_count_spinbox)
        form.addRow(self.tr("Retry delay (sec):"), self.retry_delay_spinbox)

        layout.addWidget(title_label)
        layout.addWidget(note_label)
        layout.addLayout(form)
        layout.addWidget(self.auto_open_failed_checkbox)
        layout.addWidget(self.resume_first_incomplete_checkbox)
        layout.addWidget(self.return_to_series_checkbox)
        layout.addStretch(1)

        self.failure_policy_combo.currentIndexChanged.connect(self.changed)
        self.retry_count_spinbox.valueChanged.connect(self.changed)
        self.retry_delay_spinbox.valueChanged.connect(self.changed)
        self.auto_open_failed_checkbox.stateChanged.connect(self.changed)
        self.resume_first_incomplete_checkbox.stateChanged.connect(self.changed)
        self.return_to_series_checkbox.stateChanged.connect(self.changed)

    def get_settings(self) -> dict[str, object]:
        return {
            "queue_failure_policy": self.failure_policy_combo.currentData(),
            "retry_count": int(self.retry_count_spinbox.value()),
            "retry_delay_sec": int(self.retry_delay_spinbox.value()),
            "auto_open_failed_child": self.auto_open_failed_checkbox.isChecked(),
            "resume_from_first_incomplete": self.resume_first_incomplete_checkbox.isChecked(),
            "return_to_series_after_completion": self.return_to_series_checkbox.isChecked(),
        }

    def set_settings(self, data: dict[str, object]) -> None:
        policy = str(data.get("queue_failure_policy") or "stop")
        index = self.failure_policy_combo.findData(policy)
        self.failure_policy_combo.setCurrentIndex(index if index >= 0 else 0)
        self.retry_count_spinbox.setValue(int(data.get("retry_count", 0) or 0))
        self.retry_delay_spinbox.setValue(int(data.get("retry_delay_sec", 0) or 0))
        self.auto_open_failed_checkbox.setChecked(
            bool(data.get("auto_open_failed_child", True))
        )
        self.resume_first_incomplete_checkbox.setChecked(
            bool(data.get("resume_from_first_incomplete", True))
        )
        self.return_to_series_checkbox.setChecked(
            bool(data.get("return_to_series_after_completion", True))
        )

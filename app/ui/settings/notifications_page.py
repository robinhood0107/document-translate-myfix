from __future__ import annotations

from PySide6 import QtCore, QtWidgets

from modules.utils.notification_sound import (
    DEFAULT_NTFY_SERVER_URL,
    DEFAULT_NTFY_TIMEOUT_SEC,
    FILE_SOUND_MODE,
    SYSTEM_SOUND_MODE,
    get_music_dir,
    list_music_wav_files,
    normalize_ntfy_settings,
)

from ..dayu_widgets.check_box import MCheckBox
from ..dayu_widgets.combo_box import MComboBox
from ..dayu_widgets.label import MLabel
from ..dayu_widgets.line_edit import MLineEdit
from ..dayu_widgets.push_button import MPushButton
from ..dayu_widgets.spin_box import MSpinBox


class NotificationsPage(QtWidgets.QWidget):
    changed = QtCore.Signal()
    test_requested = QtCore.Signal()
    test_ntfy_requested = QtCore.Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._loading = False

        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(12)

        title = MLabel(self.tr("Notifications")).h4()
        intro = MLabel(
            self.tr(
                "Configure completion sounds and optional ntfy push notifications for automatic runs."
            )
        ).secondary()
        intro.setWordWrap(True)
        layout.addWidget(title)
        layout.addWidget(intro)

        sound_group = QtWidgets.QGroupBox(self.tr("Automatic Completion Sound"))
        sound_layout = QtWidgets.QVBoxLayout(sound_group)
        sound_note = MLabel(
            self.tr(
                "Play a sound when automatic processing finishes successfully. "
                "Custom files must be placed in the repository music folder."
            )
        ).secondary()
        sound_note.setWordWrap(True)
        self.enable_completion_sound_checkbox = MCheckBox(self.tr("Enable completion sound"))
        self.completion_sound_combo = MComboBox().medium()
        self.test_sound_button = MPushButton(self.tr("Test sound"))
        self.test_sound_button.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        music_dir_label = MLabel(
            self.tr("Music folder: {path}").format(path=str(get_music_dir()))
        ).secondary()
        music_dir_label.setWordWrap(True)
        sound_row = QtWidgets.QHBoxLayout()
        sound_row.addWidget(self.completion_sound_combo, 1)
        sound_row.addWidget(self.test_sound_button, 0)
        sound_layout.addWidget(sound_note)
        sound_layout.addWidget(self.enable_completion_sound_checkbox)
        sound_layout.addLayout(sound_row)
        sound_layout.addWidget(music_dir_label)
        layout.addWidget(sound_group)

        ntfy_group = QtWidgets.QGroupBox(self.tr("ntfy Push Notifications"))
        ntfy_layout = QtWidgets.QGridLayout(ntfy_group)
        ntfy_layout.setHorizontalSpacing(12)
        ntfy_layout.setVerticalSpacing(8)

        ntfy_note = MLabel(
            self.tr(
                "Send text-only notifications through ntfy when automatic processing finishes, fails, or is cancelled.\n"
                "The app keeps messages below ntfy's default 4 KiB text limit and never sends attachments."
            )
        ).secondary()
        ntfy_note.setWordWrap(True)
        ntfy_layout.addWidget(ntfy_note, 0, 0, 1, 3)

        self.enable_ntfy_checkbox = MCheckBox(self.tr("Enable ntfy notifications"))
        ntfy_layout.addWidget(self.enable_ntfy_checkbox, 1, 0, 1, 3)

        ntfy_layout.addWidget(MLabel(self.tr("Server URL")), 2, 0)
        self.ntfy_server_url_input = MLineEdit().medium()
        self.ntfy_server_url_input.setPlaceholderText(DEFAULT_NTFY_SERVER_URL)
        ntfy_layout.addWidget(self.ntfy_server_url_input, 2, 1, 1, 2)

        ntfy_layout.addWidget(MLabel(self.tr("Topic")), 3, 0)
        self.ntfy_topic_input = MLineEdit().medium()
        self.ntfy_topic_input.setPlaceholderText(self.tr("comic-translate"))
        ntfy_layout.addWidget(self.ntfy_topic_input, 3, 1, 1, 2)

        ntfy_layout.addWidget(MLabel(self.tr("Access token (optional)")), 4, 0)
        self.ntfy_access_token_input = MLineEdit().medium().password()
        self.ntfy_access_token_input.setPlaceholderText(self.tr("Bearer token"))
        ntfy_layout.addWidget(self.ntfy_access_token_input, 4, 1, 1, 2)

        ntfy_layout.addWidget(MLabel(self.tr("Timeout (sec)")), 5, 0)
        self.ntfy_timeout_spinbox = MSpinBox().small()
        self.ntfy_timeout_spinbox.setRange(3, 60)
        self.ntfy_timeout_spinbox.setValue(DEFAULT_NTFY_TIMEOUT_SEC)
        ntfy_layout.addWidget(self.ntfy_timeout_spinbox, 5, 1)

        self.test_ntfy_button = MPushButton(self.tr("Send test notification"))
        self.test_ntfy_button.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        ntfy_layout.addWidget(self.test_ntfy_button, 5, 2)

        self.ntfy_success_checkbox = MCheckBox(self.tr("Notify on completion"))
        self.ntfy_failure_checkbox = MCheckBox(self.tr("Notify on failure"))
        self.ntfy_cancelled_checkbox = MCheckBox(self.tr("Notify on cancellation"))
        ntfy_layout.addWidget(self.ntfy_success_checkbox, 6, 0)
        ntfy_layout.addWidget(self.ntfy_failure_checkbox, 6, 1)
        ntfy_layout.addWidget(self.ntfy_cancelled_checkbox, 6, 2)

        layout.addWidget(ntfy_group)
        layout.addStretch(1)

        self.refresh_sound_options()

        self.enable_completion_sound_checkbox.stateChanged.connect(self._emit_changed_if_ready)
        self.completion_sound_combo.currentIndexChanged.connect(self._emit_changed_if_ready)
        self.test_sound_button.clicked.connect(self.test_requested.emit)

        for widget in (
            self.enable_ntfy_checkbox,
            self.ntfy_success_checkbox,
            self.ntfy_failure_checkbox,
            self.ntfy_cancelled_checkbox,
        ):
            widget.stateChanged.connect(self._emit_changed_if_ready)

        for widget in (
            self.ntfy_server_url_input,
            self.ntfy_topic_input,
            self.ntfy_access_token_input,
        ):
            widget.textChanged.connect(self._emit_changed_if_ready)

        self.ntfy_timeout_spinbox.valueChanged.connect(self._emit_changed_if_ready)
        self.test_ntfy_button.clicked.connect(self.test_ntfy_requested.emit)
        self.enable_ntfy_checkbox.stateChanged.connect(self._sync_ntfy_enabled_state)
        self._sync_ntfy_enabled_state()

    def _emit_changed_if_ready(self, *_args) -> None:
        if self._loading:
            return
        self.changed.emit()

    def _sync_ntfy_enabled_state(self, *_args) -> None:
        enabled = self.enable_ntfy_checkbox.isChecked()
        for widget in (
            self.ntfy_server_url_input,
            self.ntfy_topic_input,
            self.ntfy_access_token_input,
            self.ntfy_timeout_spinbox,
            self.test_ntfy_button,
            self.ntfy_success_checkbox,
            self.ntfy_failure_checkbox,
            self.ntfy_cancelled_checkbox,
        ):
            widget.setEnabled(enabled)

    def refresh_sound_options(self) -> None:
        current_mode, current_file = self.get_sound_selection()
        self.completion_sound_combo.blockSignals(True)
        self.completion_sound_combo.clear()
        self.completion_sound_combo.addItem(self.tr("System sound"), (SYSTEM_SOUND_MODE, ""))
        for file_name in list_music_wav_files():
            self.completion_sound_combo.addItem(file_name, (FILE_SOUND_MODE, file_name))

        restored_index = 0
        for index in range(self.completion_sound_combo.count()):
            data = self.completion_sound_combo.itemData(index)
            if data == (current_mode, current_file):
                restored_index = index
                break
        self.completion_sound_combo.setCurrentIndex(restored_index)
        self.completion_sound_combo.blockSignals(False)

    def get_sound_selection(self) -> tuple[str, str]:
        data = self.completion_sound_combo.currentData()
        if isinstance(data, tuple) and len(data) == 2:
            mode, file_name = data
            return str(mode or SYSTEM_SOUND_MODE), str(file_name or "")
        return SYSTEM_SOUND_MODE, ""

    def get_notification_settings(self) -> dict[str, object]:
        mode, file_name = self.get_sound_selection()
        ntfy_settings = normalize_ntfy_settings(
            {
                "enable_ntfy_notifications": self.enable_ntfy_checkbox.isChecked(),
                "ntfy_server_url": self.ntfy_server_url_input.text(),
                "ntfy_topic": self.ntfy_topic_input.text(),
                "ntfy_access_token": self.ntfy_access_token_input.text(),
                "ntfy_send_success": self.ntfy_success_checkbox.isChecked(),
                "ntfy_send_failure": self.ntfy_failure_checkbox.isChecked(),
                "ntfy_send_cancelled": self.ntfy_cancelled_checkbox.isChecked(),
                "ntfy_timeout_sec": int(self.ntfy_timeout_spinbox.value()),
            }
        )
        return {
            "enable_completion_sound": self.enable_completion_sound_checkbox.isChecked(),
            "completion_sound_mode": mode,
            "completion_sound_file": file_name,
            **ntfy_settings,
        }

    def load_settings(
        self,
        *,
        enable_completion_sound: bool,
        completion_sound_mode: str,
        completion_sound_file: str,
        enable_ntfy_notifications: bool,
        ntfy_server_url: str,
        ntfy_topic: str,
        ntfy_access_token: str,
        ntfy_send_success: bool,
        ntfy_send_failure: bool,
        ntfy_send_cancelled: bool,
        ntfy_timeout_sec: int,
    ) -> None:
        self._loading = True
        try:
            self.enable_completion_sound_checkbox.setChecked(bool(enable_completion_sound))
            self.refresh_sound_options()
            target = (str(completion_sound_mode or SYSTEM_SOUND_MODE), str(completion_sound_file or ""))
            target_index = 0
            for index in range(self.completion_sound_combo.count()):
                if self.completion_sound_combo.itemData(index) == target:
                    target_index = index
                    break
            self.completion_sound_combo.setCurrentIndex(target_index)

            ntfy_settings = normalize_ntfy_settings(
                {
                    "enable_ntfy_notifications": enable_ntfy_notifications,
                    "ntfy_server_url": ntfy_server_url,
                    "ntfy_topic": ntfy_topic,
                    "ntfy_access_token": ntfy_access_token,
                    "ntfy_send_success": ntfy_send_success,
                    "ntfy_send_failure": ntfy_send_failure,
                    "ntfy_send_cancelled": ntfy_send_cancelled,
                    "ntfy_timeout_sec": ntfy_timeout_sec,
                }
            )
            self.enable_ntfy_checkbox.setChecked(bool(ntfy_settings["enable_ntfy_notifications"]))
            self.ntfy_server_url_input.setText(str(ntfy_settings["ntfy_server_url"]))
            self.ntfy_topic_input.setText(str(ntfy_settings["ntfy_topic"]))
            self.ntfy_access_token_input.setText(str(ntfy_settings["ntfy_access_token"]))
            self.ntfy_success_checkbox.setChecked(bool(ntfy_settings["ntfy_send_success"]))
            self.ntfy_failure_checkbox.setChecked(bool(ntfy_settings["ntfy_send_failure"]))
            self.ntfy_cancelled_checkbox.setChecked(bool(ntfy_settings["ntfy_send_cancelled"]))
            self.ntfy_timeout_spinbox.setValue(int(ntfy_settings["ntfy_timeout_sec"]))
        finally:
            self._loading = False

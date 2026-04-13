from __future__ import annotations

from PySide6 import QtCore, QtWidgets

from modules.utils.notification_sound import (
    FILE_SOUND_MODE,
    SYSTEM_SOUND_MODE,
    get_music_dir,
    list_music_wav_files,
)

from ..dayu_widgets.check_box import MCheckBox
from ..dayu_widgets.combo_box import MComboBox
from ..dayu_widgets.label import MLabel
from ..dayu_widgets.push_button import MPushButton


class NotificationsPage(QtWidgets.QWidget):
    changed = QtCore.Signal()
    test_requested = QtCore.Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._loading = False

        layout = QtWidgets.QVBoxLayout(self)

        title = MLabel(self.tr("Automatic Completion")).h4()
        note = MLabel(
            self.tr(
                "Play a sound when automatic processing finishes successfully. "
                "Custom files must be placed in the project's music folder."
            )
        ).secondary()
        note.setWordWrap(True)

        self.enable_completion_sound_checkbox = MCheckBox(self.tr("Enable Completion Sound"))
        self.completion_sound_combo = MComboBox().medium()
        self.test_sound_button = MPushButton(self.tr("Test Sound"))
        self.test_sound_button.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)

        music_dir_label = MLabel(
            self.tr("Music Folder: {path}").format(path=str(get_music_dir()))
        ).secondary()
        music_dir_label.setWordWrap(True)

        sound_row = QtWidgets.QHBoxLayout()
        sound_row.addWidget(self.completion_sound_combo, 1)
        sound_row.addWidget(self.test_sound_button, 0)

        layout.addWidget(title)
        layout.addWidget(note)
        layout.addWidget(self.enable_completion_sound_checkbox)
        layout.addLayout(sound_row)
        layout.addWidget(music_dir_label)
        layout.addStretch(1)

        self.refresh_sound_options()

        self.enable_completion_sound_checkbox.stateChanged.connect(self._emit_changed_if_ready)
        self.completion_sound_combo.currentIndexChanged.connect(self._emit_changed_if_ready)
        self.test_sound_button.clicked.connect(self.test_requested.emit)

    def _emit_changed_if_ready(self, *_args) -> None:
        if self._loading:
            return
        self.changed.emit()

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
        return {
            "enable_completion_sound": self.enable_completion_sound_checkbox.isChecked(),
            "completion_sound_mode": mode,
            "completion_sound_file": file_name,
        }

    def load_settings(
        self,
        *,
        enable_completion_sound: bool,
        completion_sound_mode: str,
        completion_sound_file: str,
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
        finally:
            self._loading = False

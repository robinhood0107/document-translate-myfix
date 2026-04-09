from PySide6 import QtWidgets

from ..dayu_widgets.check_box import MCheckBox
from ..dayu_widgets.label import MLabel


class ExportPage(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QtWidgets.QVBoxLayout(self)

        batch_label = MLabel(self.tr("Automatic Mode")).h4()
        batch_note = MLabel(
            self.tr(
                "Selected exports are saved to comic_translate_<timestamp> in the same directory as the input file/archive."
            )
        ).secondary()
        batch_note.setWordWrap(True)

        debug_note = MLabel(
            self.tr(
                "Debug exports apply to both Translate All and One-Page Auto so detector, mask, and cleanup issues can be reviewed afterwards."
            )
        ).secondary()
        debug_note.setWordWrap(True)

        self.raw_text_checkbox = MCheckBox(self.tr("Export Raw Text"))
        self.translated_text_checkbox = MCheckBox(self.tr("Export Translated text"))
        self.inpainted_image_checkbox = MCheckBox(self.tr("Export Inpainted Image"))
        self.detector_overlay_checkbox = MCheckBox(self.tr("Export Detector Overlay"))
        self.raw_mask_checkbox = MCheckBox(self.tr("Export Raw Inpaint Mask"))
        self.mask_overlay_checkbox = MCheckBox(self.tr("Export Mask Overlay"))
        self.cleanup_mask_delta_checkbox = MCheckBox(self.tr("Export Cleanup Mask Delta"))
        self.debug_metadata_checkbox = MCheckBox(self.tr("Export Debug Metadata"))

        layout.addWidget(batch_label)
        layout.addWidget(batch_note)
        layout.addWidget(self.raw_text_checkbox)
        layout.addWidget(self.translated_text_checkbox)
        layout.addWidget(self.inpainted_image_checkbox)
        layout.addSpacing(12)
        layout.addWidget(debug_note)
        layout.addWidget(self.detector_overlay_checkbox)
        layout.addWidget(self.raw_mask_checkbox)
        layout.addWidget(self.mask_overlay_checkbox)
        layout.addWidget(self.cleanup_mask_delta_checkbox)
        layout.addWidget(self.debug_metadata_checkbox)
        layout.addStretch(1)

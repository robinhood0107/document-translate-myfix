from PySide6 import QtWidgets

from ..dayu_widgets.combo_box import MComboBox
from ..dayu_widgets.check_box import MCheckBox
from ..dayu_widgets.label import MLabel
from ..dayu_widgets.spin_box import MSpinBox
from modules.utils.automatic_output import (
    OUTPUT_PRESET_BALANCED,
    OUTPUT_PRESET_FAST,
    OUTPUT_PRESET_SMALL,
    OUTPUT_FORMAT_JPG,
    OUTPUT_FORMAT_PNG,
    OUTPUT_FORMAT_SAME,
    OUTPUT_FORMAT_WEBP,
)


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

        self.raw_text_checkbox = MCheckBox(self.tr("Export Raw Text"))
        self.translated_text_checkbox = MCheckBox(self.tr("Export Translated text"))
        self.inpainted_image_checkbox = MCheckBox(self.tr("Export Inpainted Image"))
        self.detector_overlay_checkbox = MCheckBox(self.tr("Export Detector Overlay (Debug)"))
        self.raw_mask_checkbox = MCheckBox(self.tr("Export Raw Inpaint Mask (Debug)"))
        self.mask_overlay_checkbox = MCheckBox(self.tr("Export Mask Overlay (Debug)"))
        self.cleanup_mask_delta_checkbox = MCheckBox(self.tr("Export Cleanup Mask Delta (Debug)"))
        self.debug_metadata_checkbox = MCheckBox(self.tr("Export Debug Metadata (Debug)"))

        layout.addWidget(batch_label)
        layout.addWidget(batch_note)
        layout.addWidget(self.raw_text_checkbox)
        layout.addWidget(self.translated_text_checkbox)
        layout.addWidget(self.inpainted_image_checkbox)
        layout.addWidget(self.detector_overlay_checkbox)
        layout.addWidget(self.raw_mask_checkbox)
        layout.addWidget(self.mask_overlay_checkbox)
        layout.addWidget(self.cleanup_mask_delta_checkbox)
        layout.addWidget(self.debug_metadata_checkbox)
        layout.addSpacing(16)

        automatic_output_label = MLabel(self.tr("Automatic Output Image Format")).h4()
        automatic_output_note = MLabel(
            self.tr(
                "These defaults control how automatic translated and cleaned images are written.\n"
                "Project-specific quick settings can override the format and preset for the current project."
            )
        ).secondary()
        automatic_output_note.setWordWrap(True)

        format_row = QtWidgets.QHBoxLayout()
        format_label = MLabel(self.tr("Default format:"))
        self.automatic_output_format_combo = MComboBox().medium()
        self.automatic_output_format_combo.addItem(self.tr("Same as source"), OUTPUT_FORMAT_SAME)
        self.automatic_output_format_combo.addItem("PNG", OUTPUT_FORMAT_PNG)
        self.automatic_output_format_combo.addItem("JPG", OUTPUT_FORMAT_JPG)
        self.automatic_output_format_combo.addItem("WEBP", OUTPUT_FORMAT_WEBP)
        format_row.addWidget(format_label)
        format_row.addWidget(self.automatic_output_format_combo, 1)

        preset_row = QtWidgets.QHBoxLayout()
        preset_label = MLabel(self.tr("Default preset:"))
        self.automatic_output_preset_combo = MComboBox().medium()
        self.automatic_output_preset_combo.addItem(self.tr("Fast"), OUTPUT_PRESET_FAST)
        self.automatic_output_preset_combo.addItem(self.tr("Balanced"), OUTPUT_PRESET_BALANCED)
        self.automatic_output_preset_combo.addItem(self.tr("Small"), OUTPUT_PRESET_SMALL)
        preset_row.addWidget(preset_label)
        preset_row.addWidget(self.automatic_output_preset_combo, 1)

        png_row = QtWidgets.QHBoxLayout()
        png_label = MLabel(self.tr("PNG compression level:"))
        self.automatic_output_png_spinbox = MSpinBox().small()
        self.automatic_output_png_spinbox.setMinimum(0)
        self.automatic_output_png_spinbox.setMaximum(9)
        self.automatic_output_png_spinbox.setValue(6)
        png_row.addWidget(png_label)
        png_row.addWidget(self.automatic_output_png_spinbox)
        png_row.addStretch(1)

        jpg_row = QtWidgets.QHBoxLayout()
        jpg_label = MLabel(self.tr("JPG quality:"))
        self.automatic_output_jpg_spinbox = MSpinBox().small()
        self.automatic_output_jpg_spinbox.setMinimum(1)
        self.automatic_output_jpg_spinbox.setMaximum(100)
        self.automatic_output_jpg_spinbox.setValue(90)
        jpg_row.addWidget(jpg_label)
        jpg_row.addWidget(self.automatic_output_jpg_spinbox)
        jpg_row.addStretch(1)

        webp_row = QtWidgets.QHBoxLayout()
        webp_label = MLabel(self.tr("WEBP quality:"))
        self.automatic_output_webp_spinbox = MSpinBox().small()
        self.automatic_output_webp_spinbox.setMinimum(1)
        self.automatic_output_webp_spinbox.setMaximum(100)
        self.automatic_output_webp_spinbox.setValue(90)
        webp_row.addWidget(webp_label)
        webp_row.addWidget(self.automatic_output_webp_spinbox)
        webp_row.addStretch(1)

        self.automatic_output_estimate_summary_label = MLabel(
            self.tr("Current project estimate: Calculating...")
        ).secondary()
        self.automatic_output_estimate_summary_label.setWordWrap(True)

        layout.addWidget(automatic_output_label)
        layout.addWidget(automatic_output_note)
        layout.addLayout(format_row)
        layout.addLayout(preset_row)
        layout.addLayout(png_row)
        layout.addLayout(jpg_row)
        layout.addLayout(webp_row)
        layout.addWidget(self.automatic_output_estimate_summary_label)
        layout.addStretch(1)

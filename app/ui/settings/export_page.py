from PySide6 import QtWidgets

from ..dayu_widgets.combo_box import MComboBox
from ..dayu_widgets.check_box import MCheckBox
from ..dayu_widgets.label import MLabel
from ..dayu_widgets.spin_box import MSpinBox
from modules.utils.automatic_output import (
    OUTPUT_ARCHIVE_FORMAT_CBZ,
    OUTPUT_ARCHIVE_FORMAT_ZIP,
    OUTPUT_IMAGE_FORMAT_JPG,
    OUTPUT_IMAGE_FORMAT_PNG,
    OUTPUT_IMAGE_FORMAT_SAME,
    OUTPUT_IMAGE_FORMAT_WEBP,
    OUTPUT_TARGET_ARCHIVE,
    OUTPUT_TARGET_IMAGES,
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

        automatic_output_label = MLabel(self.tr("Automatic Output")).h4()
        automatic_output_note = MLabel(
            self.tr(
                "These defaults control automatic output after batch translation.\n"
                "Project-specific quick settings can override them for the current project."
            )
        ).secondary()
        automatic_output_note.setWordWrap(True)

        target_row = QtWidgets.QHBoxLayout()
        target_label = MLabel(self.tr("Default output target:"))
        self.automatic_output_target_combo = MComboBox().medium()
        self.automatic_output_target_combo.addItem(self.tr("Individual images"), OUTPUT_TARGET_IMAGES)
        self.automatic_output_target_combo.addItem(self.tr("Single archive"), OUTPUT_TARGET_ARCHIVE)
        target_row.addWidget(target_label)
        target_row.addWidget(self.automatic_output_target_combo, 1)

        format_row = QtWidgets.QHBoxLayout()
        self.individual_format_widget = QtWidgets.QWidget()
        individual_format_layout = QtWidgets.QHBoxLayout(self.individual_format_widget)
        individual_format_layout.setContentsMargins(0, 0, 0, 0)
        format_label = MLabel(self.tr("Default image format:"))
        self.automatic_output_image_format_combo = MComboBox().medium()
        self.automatic_output_image_format_combo.addItem(self.tr("Same as source"), OUTPUT_IMAGE_FORMAT_SAME)
        self.automatic_output_image_format_combo.addItem("PNG", OUTPUT_IMAGE_FORMAT_PNG)
        self.automatic_output_image_format_combo.addItem("JPG", OUTPUT_IMAGE_FORMAT_JPG)
        self.automatic_output_image_format_combo.addItem("WEBP", OUTPUT_IMAGE_FORMAT_WEBP)
        individual_format_layout.addWidget(format_label)
        individual_format_layout.addWidget(self.automatic_output_image_format_combo, 1)

        self.archive_format_widget = QtWidgets.QWidget()
        archive_format_layout = QtWidgets.QHBoxLayout(self.archive_format_widget)
        archive_format_layout.setContentsMargins(0, 0, 0, 0)
        archive_format_label = MLabel(self.tr("Default archive format:"))
        self.automatic_output_archive_format_combo = MComboBox().medium()
        self.automatic_output_archive_format_combo.addItem("ZIP", OUTPUT_ARCHIVE_FORMAT_ZIP)
        self.automatic_output_archive_format_combo.addItem("CBZ", OUTPUT_ARCHIVE_FORMAT_CBZ)
        archive_format_layout.addWidget(archive_format_label)
        archive_format_layout.addWidget(self.automatic_output_archive_format_combo, 1)

        self.archive_image_format_widget = QtWidgets.QWidget()
        archive_image_layout = QtWidgets.QHBoxLayout(self.archive_image_format_widget)
        archive_image_layout.setContentsMargins(0, 0, 0, 0)
        archive_image_label = MLabel(self.tr("Default archive image format:"))
        self.automatic_output_archive_image_format_combo = MComboBox().medium()
        self.automatic_output_archive_image_format_combo.addItem("PNG", OUTPUT_IMAGE_FORMAT_PNG)
        self.automatic_output_archive_image_format_combo.addItem("JPG", OUTPUT_IMAGE_FORMAT_JPG)
        self.automatic_output_archive_image_format_combo.addItem("WEBP", OUTPUT_IMAGE_FORMAT_WEBP)
        archive_image_layout.addWidget(archive_image_label)
        archive_image_layout.addWidget(self.automatic_output_archive_image_format_combo, 1)

        self.archive_level_widget = QtWidgets.QWidget()
        archive_level_layout = QtWidgets.QHBoxLayout(self.archive_level_widget)
        archive_level_layout.setContentsMargins(0, 0, 0, 0)
        archive_level_label = MLabel(self.tr("Default archive compression level:"))
        self.automatic_output_archive_level_spinbox = MSpinBox().small()
        self.automatic_output_archive_level_spinbox.setMinimum(0)
        self.automatic_output_archive_level_spinbox.setMaximum(9)
        self.automatic_output_archive_level_spinbox.setValue(6)
        archive_level_layout.addWidget(archive_level_label)
        archive_level_layout.addWidget(self.automatic_output_archive_level_spinbox)
        archive_level_layout.addStretch(1)

        self.automatic_output_quality_note_label = MLabel(
            self.tr("PNG/JPG/WEBP images are saved at maximum quality.")
        ).secondary()
        self.automatic_output_quality_note_label.setWordWrap(True)

        self.automatic_output_archive_note_label = MLabel(
            self.tr("Archive compression only affects the ZIP/CBZ container, not image quality.")
        ).secondary()
        self.automatic_output_archive_note_label.setWordWrap(True)

        self.automatic_output_estimate_summary_label = MLabel(
            self.tr("Load pages to see automatic output estimates.")
        ).secondary()
        self.automatic_output_estimate_summary_label.setWordWrap(True)

        layout.addWidget(automatic_output_label)
        layout.addWidget(automatic_output_note)
        layout.addLayout(target_row)
        layout.addWidget(self.individual_format_widget)
        layout.addWidget(self.archive_format_widget)
        layout.addWidget(self.archive_image_format_widget)
        layout.addWidget(self.archive_level_widget)
        layout.addWidget(self.automatic_output_quality_note_label)
        layout.addWidget(self.automatic_output_archive_note_label)
        layout.addWidget(self.automatic_output_estimate_summary_label)
        layout.addStretch(1)

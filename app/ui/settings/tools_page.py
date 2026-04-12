from PySide6 import QtCore, QtWidgets

from ..dayu_widgets.check_box import MCheckBox
from ..dayu_widgets.label import MLabel
from ..dayu_widgets.spin_box import MSpinBox
from .utils import create_title_and_combo, set_combo_box_width
from modules.utils.device import is_gpu_available


class ToolsPage(QtWidgets.QWidget):
    def __init__(
        self,
        translators: list[str],
        ocr_engines: list[str],
        detectors: list[str],
        inpainters: list[str],
        inpaint_strategy: list[str],
        parent=None,
    ):
        super().__init__(parent)
        self.translators = translators
        self.ocr_engines = ocr_engines
        self.detectors = detectors
        self.inpainters = inpainters
        self.inpaint_strategy = inpaint_strategy

        layout = QtWidgets.QVBoxLayout(self)

        translator_widget, self.translator_combo = create_title_and_combo(self.tr("Translator"), self.translators, h4=True)
        set_combo_box_width(self.translator_combo, self.translators)
        self.translator_combo.setCurrentText(self.tr("Custom Local Server(Gemma)"))

        ocr_widget, self.ocr_combo = create_title_and_combo(self.tr("Text Recognition"), self.ocr_engines, h4=True)
        set_combo_box_width(self.ocr_combo, self.ocr_engines)
        self.ocr_combo.setCurrentText(self.tr("PaddleOCR VL"))

        detector_widget, self.detector_combo = create_title_and_combo(self.tr("Text Detector"), self.detectors, h4=True)
        set_combo_box_width(self.detector_combo, self.detectors)

        self.mask_inpaint_mode_options = [
            self.tr("RT-DETR-v2 + Legacy BBox + Source LaMa"),
            self.tr("Source Parity CTD/LaMa"),
        ]
        mask_inpaint_mode_widget, self.mask_inpaint_mode_combo = create_title_and_combo(
            self.tr("Mask/Inpaint Mode"),
            self.mask_inpaint_mode_options,
            h4=False,
        )
        set_combo_box_width(self.mask_inpaint_mode_combo, self.mask_inpaint_mode_options)
        self.mask_inpaint_mode_combo.setCurrentText(self.mask_inpaint_mode_options[0])
        self.mask_inpaint_mode_hint = MLabel("")
        self.mask_inpaint_mode_hint.setWordWrap(True)

        mask_label = MLabel(self.tr("Precise Masking")).h4()
        mask_refiner_widget, self.mask_refiner_combo = create_title_and_combo(
            self.tr("Mask Refiner"),
            [self.tr("legacy_bbox"), self.tr("ctd")],
            h4=False,
        )
        set_combo_box_width(self.mask_refiner_combo, [self.tr("legacy_bbox"), self.tr("ctd")])
        self.mask_refiner_combo.setCurrentText(self.tr("legacy_bbox"))
        self.keep_existing_lines_checkbox = MCheckBox(self.tr("Keep Existing Lines"))
        self.keep_existing_lines_checkbox.setChecked(False)

        self.ctd_settings_widget = QtWidgets.QWidget()
        ctd_form = QtWidgets.QFormLayout(self.ctd_settings_widget)
        ctd_form.setContentsMargins(10, 0, 0, 0)
        ctd_form.setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignLeft)

        self.ctd_detect_size_combo = QtWidgets.QComboBox()
        self.ctd_detect_size_combo.addItems(["640", "960", "1280", "1536", "2048"])
        self.ctd_detect_size_combo.setCurrentText("1280")

        self.ctd_det_rearrange_max_batches_combo = QtWidgets.QComboBox()
        self.ctd_det_rearrange_max_batches_combo.addItems(["1", "2", "4", "8"])
        self.ctd_det_rearrange_max_batches_combo.setCurrentText("4")

        self.ctd_device_combo = QtWidgets.QComboBox()
        self.ctd_device_combo.addItems(["cuda", "cpu"])
        self.ctd_device_combo.setCurrentText("cuda")

        self.ctd_font_size_multiplier_spinbox = QtWidgets.QDoubleSpinBox()
        self.ctd_font_size_multiplier_spinbox.setDecimals(2)
        self.ctd_font_size_multiplier_spinbox.setSingleStep(0.05)
        self.ctd_font_size_multiplier_spinbox.setRange(0.1, 4.0)
        self.ctd_font_size_multiplier_spinbox.setValue(1.0)

        self.ctd_font_size_max_spinbox = MSpinBox().small()
        self.ctd_font_size_max_spinbox.setRange(-1, 4096)
        self.ctd_font_size_max_spinbox.setValue(-1)

        self.ctd_font_size_min_spinbox = MSpinBox().small()
        self.ctd_font_size_min_spinbox.setRange(-1, 4096)
        self.ctd_font_size_min_spinbox.setValue(-1)

        self.ctd_mask_dilate_size_spinbox = MSpinBox().small()
        self.ctd_mask_dilate_size_spinbox.setRange(0, 32)
        self.ctd_mask_dilate_size_spinbox.setValue(2)

        ctd_form.addRow(self.tr("detect_size"), self.ctd_detect_size_combo)
        ctd_form.addRow(self.tr("det_rearrange_max_batches"), self.ctd_det_rearrange_max_batches_combo)
        ctd_form.addRow(self.tr("device"), self.ctd_device_combo)
        ctd_form.addRow(self.tr("font size multiplier"), self.ctd_font_size_multiplier_spinbox)
        ctd_form.addRow(self.tr("font size max"), self.ctd_font_size_max_spinbox)
        ctd_form.addRow(self.tr("font size min"), self.ctd_font_size_min_spinbox)
        ctd_form.addRow(self.tr("mask dilate size"), self.ctd_mask_dilate_size_spinbox)

        inpainting_label = MLabel(self.tr("Image Cleaning")).h4()
        inpainter_widget, self.inpainter_combo = create_title_and_combo(self.tr("Inpainter"), self.inpainters, h4=False)
        set_combo_box_width(self.inpainter_combo, self.inpainters)
        self.inpainter_combo.setCurrentText(self.tr("lama_large_512px"))

        self.inpainter_runtime_widget = QtWidgets.QWidget()
        inpainter_form = QtWidgets.QFormLayout(self.inpainter_runtime_widget)
        inpainter_form.setContentsMargins(10, 0, 0, 0)
        self.inpainter_size_combo = QtWidgets.QComboBox()
        self.inpainter_device_combo = QtWidgets.QComboBox()
        self.inpainter_device_combo.addItems(["cuda", "cpu"])
        self.inpainter_device_combo.setCurrentText("cuda")
        self.inpainter_precision_combo = QtWidgets.QComboBox()
        self.inpainter_precision_combo.addItems(["fp32", "bf16"])
        self.inpainter_precision_combo.setCurrentText("bf16")
        inpainter_form.addRow(self.tr("inpaint_size"), self.inpainter_size_combo)
        inpainter_form.addRow(self.tr("device"), self.inpainter_device_combo)
        inpainter_form.addRow(self.tr("precision"), self.inpainter_precision_combo)

        inpaint_strategy_widget, self.inpaint_strategy_combo = create_title_and_combo(self.tr("HD Strategy"), self.inpaint_strategy, h4=False)
        set_combo_box_width(self.inpaint_strategy_combo, self.inpaint_strategy)
        self.inpaint_strategy_combo.setCurrentText(self.tr("Resize"))

        self.hd_strategy_widgets = QtWidgets.QWidget()
        self.hd_strategy_layout = QtWidgets.QVBoxLayout(self.hd_strategy_widgets)

        self.resize_widget = QtWidgets.QWidget()
        about_resize_layout = QtWidgets.QVBoxLayout(self.resize_widget)
        resize_layout = QtWidgets.QHBoxLayout()
        resize_label = MLabel(self.tr("Resize Limit:"))
        about_resize_label = MLabel(self.tr("Resize the longer side of the image to a specific size,\nthen do inpainting on the resized image."))
        self.resize_spinbox = MSpinBox().small()
        self.resize_spinbox.setFixedWidth(70)
        self.resize_spinbox.setMaximum(3000)
        self.resize_spinbox.setValue(960)
        resize_layout.addWidget(resize_label)
        resize_layout.addWidget(self.resize_spinbox)
        resize_layout.addStretch()
        about_resize_layout.addWidget(about_resize_label)
        about_resize_layout.addLayout(resize_layout)
        about_resize_layout.setContentsMargins(5, 5, 5, 5)
        about_resize_layout.addStretch()

        self.crop_widget = QtWidgets.QWidget()
        crop_layout = QtWidgets.QVBoxLayout(self.crop_widget)
        about_crop_label = MLabel(self.tr("Crop masking area from the original image to do inpainting."))
        crop_margin_layout = QtWidgets.QHBoxLayout()
        crop_margin_label = MLabel(self.tr("Crop Margin:"))
        self.crop_margin_spinbox = MSpinBox().small()
        self.crop_margin_spinbox.setFixedWidth(70)
        self.crop_margin_spinbox.setMaximum(3000)
        self.crop_margin_spinbox.setValue(512)
        crop_margin_layout.addWidget(crop_margin_label)
        crop_margin_layout.addWidget(self.crop_margin_spinbox)
        crop_margin_layout.addStretch()

        crop_trigger_layout = QtWidgets.QHBoxLayout()
        crop_trigger_label = MLabel(self.tr("Crop Trigger Size:"))
        self.crop_trigger_spinbox = MSpinBox().small()
        self.crop_trigger_spinbox.setFixedWidth(70)
        self.crop_trigger_spinbox.setMaximum(3000)
        self.crop_trigger_spinbox.setValue(512)
        crop_trigger_layout.addWidget(crop_trigger_label)
        crop_trigger_layout.addWidget(self.crop_trigger_spinbox)
        crop_trigger_layout.addStretch()

        crop_layout.addWidget(about_crop_label)
        crop_layout.addLayout(crop_margin_layout)
        crop_layout.addLayout(crop_trigger_layout)
        crop_layout.setContentsMargins(5, 5, 5, 5)

        self.hd_strategy_layout.addWidget(self.resize_widget)
        self.hd_strategy_layout.addWidget(self.crop_widget)
        self.resize_widget.show()
        self.crop_widget.hide()
        self.inpaint_strategy_combo.currentIndexChanged.connect(self._update_hd_strategy_widgets)
        self.inpainter_combo.currentIndexChanged.connect(self._update_inpainter_runtime_widgets)
        self.inpainter_combo.currentTextChanged.connect(
            lambda _text: self._update_inpainter_runtime_widgets(self.inpainter_combo.currentIndex())
        )
        self.mask_refiner_combo.currentIndexChanged.connect(self._update_mask_refiner_widgets)
        self.mask_refiner_combo.currentTextChanged.connect(
            lambda _text: self._update_mask_refiner_widgets(self.mask_refiner_combo.currentIndex())
        )
        self.mask_inpaint_mode_combo.currentIndexChanged.connect(self._update_mask_inpaint_mode_widgets)
        self.mask_inpaint_mode_combo.currentTextChanged.connect(
            lambda _text: self._update_mask_inpaint_mode_widgets(self.mask_inpaint_mode_combo.currentIndex())
        )

        self.use_gpu_checkbox = MCheckBox(self.tr("Use GPU"))
        self.use_gpu_checkbox.setChecked(True)
        if not is_gpu_available():
            self.use_gpu_checkbox.setVisible(False)

        layout.addWidget(translator_widget)
        layout.addSpacing(10)
        layout.addWidget(detector_widget)
        layout.addWidget(mask_inpaint_mode_widget)
        layout.addWidget(self.mask_inpaint_mode_hint)
        layout.addSpacing(10)
        layout.addWidget(mask_label)
        layout.addWidget(mask_refiner_widget)
        layout.addWidget(self.keep_existing_lines_checkbox)
        layout.addWidget(self.ctd_settings_widget)
        layout.addSpacing(10)
        layout.addWidget(ocr_widget)
        layout.addSpacing(10)
        layout.addWidget(inpainting_label)
        layout.addWidget(inpainter_widget)
        layout.addWidget(self.inpainter_runtime_widget)
        layout.addWidget(inpaint_strategy_widget)
        layout.addWidget(self.hd_strategy_widgets)
        layout.addSpacing(10)
        layout.addWidget(self.use_gpu_checkbox)
        layout.addStretch(1)

        self._update_hd_strategy_widgets(self.inpaint_strategy_combo.currentIndex())
        self._update_inpainter_runtime_widgets(self.inpainter_combo.currentIndex())
        self._update_mask_refiner_widgets(self.mask_refiner_combo.currentIndex())
        self._update_mask_inpaint_mode_widgets(self.mask_inpaint_mode_combo.currentIndex())

    def _update_hd_strategy_widgets(self, index: int):
        strategy = self.inpaint_strategy_combo.itemText(index)
        self.resize_widget.setVisible(strategy == self.tr("Resize"))
        self.crop_widget.setVisible(strategy == self.tr("Crop"))
        if strategy == self.tr("Original"):
            self.hd_strategy_widgets.setFixedHeight(0)
        else:
            self.hd_strategy_widgets.setFixedHeight(self.hd_strategy_widgets.sizeHint().height())

    def _update_mask_refiner_widgets(self, index: int):
        use_ctd = self.mask_refiner_combo.itemText(index) == self.tr("ctd")
        self.keep_existing_lines_checkbox.setVisible(use_ctd)
        self.ctd_settings_widget.setVisible(use_ctd)

    def _update_mask_inpaint_mode_widgets(self, index: int):
        mode_text = self.mask_inpaint_mode_combo.itemText(index)
        legacy_mode = mode_text == self.tr("RT-DETR-v2 + Legacy BBox + Source LaMa")
        source_parity = mode_text == self.tr("Source Parity CTD/LaMa")
        source_mode = legacy_mode or source_parity

        if source_mode:
            target_mask_refiner = self.tr("ctd") if source_parity else self.tr("legacy_bbox")
            mask_refiner_index = self.mask_refiner_combo.findText(target_mask_refiner)
            if mask_refiner_index != -1 and self.mask_refiner_combo.currentIndex() != mask_refiner_index:
                self.mask_refiner_combo.setCurrentIndex(mask_refiner_index)
            self.mask_refiner_combo.setEnabled(False)
            self.keep_existing_lines_checkbox.setChecked(False)
            self.keep_existing_lines_checkbox.setVisible(False)
            self.ctd_settings_widget.setVisible(source_parity)

            lama_index = self.inpainter_combo.findText(self.tr("lama_large_512px"))
            if lama_index != -1 and self.inpainter_combo.currentIndex() != lama_index:
                self.inpainter_combo.setCurrentIndex(lama_index)
            self.inpainter_combo.setEnabled(False)

            detector_index = self.detector_combo.findText("RT-DETR-v2")
            if legacy_mode and detector_index != -1 and self.detector_combo.currentIndex() != detector_index:
                self.detector_combo.setCurrentIndex(detector_index)
        else:
            self.mask_refiner_combo.setEnabled(True)
            self.keep_existing_lines_checkbox.setVisible(self.mask_refiner_combo.currentText() == self.tr("ctd"))
            self.ctd_settings_widget.setVisible(self.mask_refiner_combo.currentText() == self.tr("ctd"))
            self.inpainter_combo.setEnabled(True)

        self.detector_combo.setEnabled(not source_mode)
        if source_parity:
            self.mask_inpaint_mode_hint.setText(self.tr("Source Parity mode runs the source CTD detector, source grouping, source mask refinement, and source block-wise LaMa exactly as the reference flow."))
        elif legacy_mode:
            self.mask_inpaint_mode_hint.setText(self.tr("RT-DETR-v2 + Legacy BBox + Source LaMa keeps RT-DETR-v2 as the detector, restores the exact 1aec275 bbox mask flow, and then runs the exact source block-wise LaMa inpainting flow."))
        else:
            self.mask_inpaint_mode_hint.setText("")

    def _update_inpainter_runtime_widgets(self, index: int):
        key = self.inpainter_combo.itemText(index)
        if key == self.tr("lama_large_512px"):
            sizes = ["512", "768", "1024", "1536", "2048"]
            default_size = "1536"
            show_precision = True
            default_precision = "bf16"
        elif key == self.tr("lama_mpe"):
            sizes = ["1024", "2048"]
            default_size = "2048"
            show_precision = False
            default_precision = "fp32"
        else:
            sizes = ["1024", "2048"]
            default_size = "2048"
            show_precision = False
            default_precision = "fp32"

        current_size = self.inpainter_size_combo.currentText()
        self.inpainter_size_combo.blockSignals(True)
        self.inpainter_size_combo.clear()
        self.inpainter_size_combo.addItems(sizes)
        self.inpainter_size_combo.setCurrentText(current_size if current_size in sizes else default_size)
        self.inpainter_size_combo.blockSignals(False)
        self.inpainter_precision_combo.setCurrentText(default_precision)
        self.inpainter_precision_combo.setVisible(show_precision)
        form = self.inpainter_runtime_widget.layout()
        form.labelForField(self.inpainter_precision_combo).setVisible(show_precision)

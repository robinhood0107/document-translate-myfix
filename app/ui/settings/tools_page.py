from PySide6 import QtWidgets

from ..dayu_widgets.check_box import MCheckBox
from ..dayu_widgets.label import MLabel
from ..dayu_widgets.spin_box import MSpinBox
from .utils import create_title_and_combo, set_combo_box_width
from modules.utils.device import is_gpu_available
from modules.ocr.selection import WORKFLOW_MODE_OPTIONS


DEFAULT_AUTOMATIC_RUNTIME_LABEL = "RT-DETR-v2 + CTD Line Protect + Source LaMa"


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

        workflow_modes = [self.tr(label) for _key, label in WORKFLOW_MODE_OPTIONS]
        workflow_widget, self.workflow_mode_combo = create_title_and_combo(self.tr("Workflow Mode"), workflow_modes, h4=True)
        set_combo_box_width(self.workflow_mode_combo, workflow_modes)

        automatic_runtime_label = MLabel(self.tr("Automatic Runtime")).h4()
        self.automatic_runtime_value_label = MLabel(self.tr(DEFAULT_AUTOMATIC_RUNTIME_LABEL))
        self.automatic_runtime_value_label.setWordWrap(True)
        self.automatic_runtime_hint = MLabel(
            self.tr(
                "Automatic runtime is fixed to RT-DETR-v2 detection, CTD line-protect masking, and source block-wise LaMa cleaning."
            )
        )
        self.automatic_runtime_hint.setWordWrap(True)
        self.workflow_mode_hint = MLabel(
            self.tr(
                "Stage-Batched runs detect, OCR, inpaint, translate, and render in stage order. Legacy keeps the original page-by-page loop."
            )
        )
        self.workflow_mode_hint.setWordWrap(True)

        detector_widget, self.detector_combo = create_title_and_combo(self.tr("Text Detector"), self.detectors, h4=False)
        set_combo_box_width(self.detector_combo, self.detectors)
        self.detector_combo.setCurrentText("RT-DETR-v2")
        self.detector_combo.setEnabled(False)

        inpainting_label = MLabel(self.tr("Image Cleaning")).h4()
        inpainter_widget, self.inpainter_combo = create_title_and_combo(self.tr("Inpainter"), self.inpainters, h4=False)
        set_combo_box_width(self.inpainter_combo, self.inpainters)
        self.inpainter_combo.setCurrentText(self.tr("lama_large_512px"))
        self.inpainter_combo.setEnabled(False)

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

        self.use_gpu_checkbox = MCheckBox(self.tr("Use GPU"))
        self.use_gpu_checkbox.setChecked(True)
        if not is_gpu_available():
            self.use_gpu_checkbox.setVisible(False)

        layout.addWidget(translator_widget)
        layout.addSpacing(10)
        layout.addWidget(ocr_widget)
        layout.addSpacing(10)
        layout.addWidget(workflow_widget)
        layout.addWidget(self.workflow_mode_hint)
        layout.addSpacing(10)
        layout.addWidget(automatic_runtime_label)
        layout.addWidget(self.automatic_runtime_value_label)
        layout.addWidget(self.automatic_runtime_hint)
        layout.addSpacing(10)
        layout.addWidget(detector_widget)
        layout.addSpacing(10)
        layout.addWidget(inpainting_label)
        layout.addWidget(inpainter_widget)
        layout.addWidget(self.inpainter_runtime_widget)
        layout.addWidget(inpaint_strategy_widget)
        layout.addWidget(self.hd_strategy_widgets)
        layout.addSpacing(10)
        layout.addWidget(self.use_gpu_checkbox)
        layout.addStretch(1)

        self._update_inpainter_runtime_widgets(self.inpainter_combo.currentIndex())
        self._update_hd_strategy_widgets(self.inpaint_strategy_combo.currentIndex())

    def _update_hd_strategy_widgets(self, index: int):
        strategy = self.inpaint_strategy_combo.itemText(index)
        self.resize_widget.setVisible(strategy == self.tr("Resize"))
        self.crop_widget.setVisible(strategy == self.tr("Crop"))
        if strategy == self.tr("Original"):
            self.hd_strategy_widgets.setFixedHeight(0)
        else:
            self.hd_strategy_widgets.setFixedHeight(self.hd_strategy_widgets.sizeHint().height())

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

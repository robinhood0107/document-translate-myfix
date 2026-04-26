import os
from PySide6 import QtWidgets
from PySide6 import QtCore

from modules.ocr.selection import OCR_MODE_OPTIONS, WORKFLOW_MODE_OPTIONS

from ..dayu_widgets.clickable_card import ClickMeta
from ..dayu_widgets.divider import MDivider
from ..dayu_widgets.qt import MPixmap

# New imports for refactored pages
from .personalization_page import PersonalizationPage
from .tools_page import ToolsPage
from .paddleocr_vl_page import PaddleOCRVLPage
from .hunyuan_ocr_page import HunyuanOCRPage
from .mangalmm_ocr_page import MangaLMMOCRPage
from .gemma_local_server_page import GemmaLocalServerPage
from .credentials_page import CredentialsPage
from .llms_page import LlmsPage
from .text_rendering_page import TextRenderingPage
from .notifications_page import NotificationsPage
from .project_page import ProjectPage
from .series_page import SeriesPage
from .export_page import ExportPage
from .shortcuts_page import ShortcutsPage
from .about_page import AboutPage
from .user_dictionaries_page import UserDictionariesPage


class CurrentPageStack(QtWidgets.QStackedWidget):
    """A QStackedWidget that reports size based on the current page only.
    This ensures the scroll area uses only the active page's size and
    avoids empty scroll space from larger sibling pages.
    """
    def sizeHint(self):
        w = self.currentWidget()
        if w is not None:
            # Use the current page's hint without forcing a resize,
            # to avoid constraining horizontal expansion.
            return w.sizeHint()
        return super().sizeHint()

    def minimumSizeHint(self):
        w = self.currentWidget()
        if w is not None:
            return w.minimumSizeHint()
        return super().minimumSizeHint()

    def hasHeightForWidth(self):
        w = self.currentWidget()
        return w.hasHeightForWidth() if w is not None else False

    def heightForWidth(self, width):
        w = self.currentWidget()
        return w.heightForWidth(width) if w is not None else -1


class SettingsPageUI(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super(SettingsPageUI, self).__init__(parent)

        self.credential_widgets = {}

        self.inpainters = ['AOT', 'lama_large_512px', 'lama_mpe']
        self.detectors = ['RT-DETR-v2']
        self.ocr_engine_keys = [key for key, _label in OCR_MODE_OPTIONS]
        self.ocr_engines = [self.tr(label) for _key, label in OCR_MODE_OPTIONS]
        self.inpaint_strategy = [self.tr('Resize'), self.tr('Original'), self.tr('Crop')]
        self.themes = [self.tr('Dark'), self.tr('Light')]
        self.alignment = [self.tr("Left"), self.tr("Center"), self.tr("Right")]

        self.credential_services = [
            self.tr("Custom Service"),
            self.tr("Custom Local Server(Gemma)"),
            self.tr("Open AI GPT"),
            self.tr("Anthropic Claude"),
            self.tr("Google Gemini"),
            self.tr("Deepseek"),
            self.tr("Microsoft Azure"),
            self.tr("Google Cloud"),
        ]
        
        self.translator_keys = [
            "Gemini-2.5-Pro",
            "Gemini-3.0-Flash",
            "GPT-4.1",
            "GPT-4.1-mini",
            "Claude-4.6-Sonnet",
            "Claude-4.5-Haiku",
            "Deepseek-v3",
            "Custom Service",
            "Custom Local Server(Gemma)",
        ]
        self.supported_translators = [self.tr(key) for key in self.translator_keys]
        
        self.languages = [
            'English', 
            '한국어', 
            'Français', 
            '简体中文', 
            'русский', 
            '日本語', 
            'Deutsch', 
            'Español', 
            'Italiano', 
        ]
        
        self.nav_cards = []  
        self.current_highlighted_nav = None

        self.value_mappings = {
            # Language mappings
            "English": "English",
            "한국어": "한국어",
            "Français": "Français",
            "简体中文": "简体中文",
            "русский": "русский",
            "日本語": "日本語",
            "Deutsch": "Deutsch",
            "Español": "Español",
            "Italiano": "Italiano",

            # Theme mappings
            self.tr("Dark"): "Dark",
            self.tr("Light"): "Light",

            # Translator mappings
            self.tr("Custom Service"): "Custom Service",
            self.tr("Custom Local Server(Gemma)"): "Custom Local Server(Gemma)",
            self.tr("Deepseek-v3"): "Deepseek-v3",
            self.tr("GPT-4.1"): "GPT-4.1",
            self.tr("GPT-4.1-mini"): "GPT-4.1-mini",
            self.tr("DeepL"): "DeepL",
            self.tr("Claude-4.6-Sonnet"): "Claude-4.6-Sonnet",
            self.tr("Claude-4.5-Haiku"): "Claude-4.5-Haiku",
            self.tr("Gemini-3.0-Flash"): "Gemini-3.0-Flash",
            self.tr("Gemini-2.5-Pro"): "Gemini-2.5-Pro",
            self.tr("Yandex"): "Yandex",
            self.tr("Microsoft Translator"): "Microsoft Translator",

            # OCR mappings
            self.tr("Default (existing auto: MangaOCR / PPOCR / Pororo...)"): "default",
            self.tr("Optimal (HunyuanOCR / PaddleOCR VL)"): "best_local",
            self.tr("Microsoft OCR"): "microsoft_ocr",
            self.tr("Google Cloud Vision"): "google_cloud_vision",
            self.tr("Gemini-2.0-Flash"): "gemini_2_0_flash",
            self.tr("PaddleOCR VL"): "paddleocr_vl",
            self.tr("HunyuanOCR"): "hunyuanocr",
            self.tr("MangaLMM"): "mangalmm",

            # Workflow mode mappings
            self.tr("Stage-Batched Pipeline (Recommended)"): "stage_batched_pipeline",
            self.tr("Legacy Page Pipeline (Legacy)"): "legacy_page_pipeline",

            # Inpainter mappings
            "AOT": "AOT",
            "lama_large_512px": "lama_large_512px",
            "lama_mpe": "lama_mpe",
            "LaMa": "lama_large_512px",

            # Detector mappings
            "RT-DETR-v2": "RT-DETR-v2",

            # Fixed automatic runtime mapping
            self.tr("RT-DETR-v2 + CTD Line Protect + Source LaMa"): "rtdetr_legacy_bbox_source_lama",
            self.tr("RT-DETR-v2 + Legacy BBox Rescue + Source LaMa"): "rtdetr_legacy_bbox_source_lama",

            # HD Strategy mappings
            self.tr("Resize"): "Resize",
            self.tr("Original"): "Original",
            self.tr("Crop"): "Crop",

            # Alignment mappings
            self.tr("Left"): "Left",
            self.tr("Center"): "Center",
            self.tr("Right"): "Right",

            # Credential services mappings
            self.tr("Custom Service"): "Custom Service",
            self.tr("Custom Local Server(Gemma)"): "Custom Local Server(Gemma)",
            self.tr("Deepseek"): "Deepseek",
            self.tr("Open AI GPT"): "Open AI GPT",
            self.tr("Microsoft Azure"): "Microsoft Azure",
            self.tr("Google Cloud"): "Google Cloud",
            self.tr("Google Gemini"): "Google Gemini",
            self.tr("DeepL"): "DeepL",
            self.tr("Anthropic Claude"): "Anthropic Claude",
            self.tr("Yandex"): "Yandex",
        }

        # Create reverse mappings for loading.
        # Deprecated aliases like "LaMa" should not override canonical UI labels.
        self.reverse_mappings = {v: k for k, v in self.value_mappings.items()}
        self.reverse_mappings["AOT"] = "AOT"
        self.reverse_mappings["lama_large_512px"] = "lama_large_512px"
        self.reverse_mappings["lama_mpe"] = "lama_mpe"

        self._init_ui()

    def _init_ui(self):
        self.stacked_widget = CurrentPageStack()
        # Ensure the right content can expand horizontally
        self.stacked_widget.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Preferred,
        )

        # Instantiate each page widget and keep references as attributes
        self.personalization_page = PersonalizationPage(
            languages=self.languages,
            themes=self.themes,
            parent=self,
        )
        self.tools_page = ToolsPage(
            translators=self.supported_translators,
            ocr_engines=self.ocr_engines,
            detectors=self.detectors,
            inpainters=self.inpainters,
            inpaint_strategy=self.inpaint_strategy,
            parent=self,
        )
        for index, key in enumerate(self.translator_keys):
            self.tools_page.translator_combo.setItemData(index, key)
        for index, (key, _label) in enumerate(WORKFLOW_MODE_OPTIONS):
            self.tools_page.workflow_mode_combo.setItemData(index, key)
        for index, key in enumerate(self.ocr_engine_keys):
            self.tools_page.ocr_combo.setItemData(index, key)
        self.paddleocr_vl_page = PaddleOCRVLPage(parent=self)
        self.hunyuan_ocr_page = HunyuanOCRPage(parent=self)
        self.mangalmm_ocr_page = MangaLMMOCRPage(parent=self)
        self.gemma_local_server_page = GemmaLocalServerPage(parent=self)
        self.credentials_page = CredentialsPage(
            services=self.credential_services,
            value_mappings=self.value_mappings,
            parent=self,
        )
        self.llms_page = LlmsPage(parent=self)
        self.text_rendering_page = TextRenderingPage(parent=self)
        self.project_page = ProjectPage(parent=self)
        self.series_page = SeriesPage(parent=self)
        self.export_page = ExportPage(parent=self)
        self.user_dictionaries_page = UserDictionariesPage(parent=self)
        self.notifications_page = NotificationsPage(parent=self)
        self.shortcuts_page = ShortcutsPage(parent=self)
        self.about_page = AboutPage(parent=self)

        # Backward-compatible attribute proxies for existing SettingsPage references
        # Personalization
        self.lang_combo = self.personalization_page.lang_combo
        self.theme_combo = self.personalization_page.theme_combo

        # Tools
        self.translator_combo = self.tools_page.translator_combo
        self.ocr_combo = self.tools_page.ocr_combo
        self.workflow_mode_combo = self.tools_page.workflow_mode_combo
        self.detector_combo = self.tools_page.detector_combo
        self.inpainter_combo = self.tools_page.inpainter_combo
        self.inpainter_size_combo = self.tools_page.inpainter_size_combo
        self.inpainter_device_combo = self.tools_page.inpainter_device_combo
        self.inpainter_precision_combo = self.tools_page.inpainter_precision_combo
        self.inpaint_strategy_combo = self.tools_page.inpaint_strategy_combo
        self.resize_spinbox = self.tools_page.resize_spinbox
        self.crop_margin_spinbox = self.tools_page.crop_margin_spinbox
        self.crop_trigger_spinbox = self.tools_page.crop_trigger_spinbox
        self.use_gpu_checkbox = self.tools_page.use_gpu_checkbox
        self.paddleocr_vl_server_url_input = self.paddleocr_vl_page.server_url_input
        self.paddleocr_vl_prettify_checkbox = self.paddleocr_vl_page.prettify_markdown_checkbox
        self.paddleocr_vl_visualize_checkbox = self.paddleocr_vl_page.visualize_checkbox
        self.paddleocr_vl_max_new_tokens_spinbox = self.paddleocr_vl_page.max_new_tokens_spinbox
        self.paddleocr_vl_parallel_workers_spinbox = self.paddleocr_vl_page.parallel_workers_spinbox
        self.hunyuan_ocr_server_url_input = self.hunyuan_ocr_page.server_url_input
        self.hunyuan_ocr_max_completion_tokens_spinbox = self.hunyuan_ocr_page.max_completion_tokens_spinbox
        self.hunyuan_ocr_parallel_workers_spinbox = self.hunyuan_ocr_page.parallel_workers_spinbox
        self.hunyuan_ocr_request_timeout_spinbox = self.hunyuan_ocr_page.request_timeout_spinbox
        self.hunyuan_ocr_raw_response_logging_checkbox = self.hunyuan_ocr_page.raw_response_logging_checkbox
        self.mangalmm_ocr_server_url_input = self.mangalmm_ocr_page.server_url_input
        self.mangalmm_ocr_max_completion_tokens_spinbox = self.mangalmm_ocr_page.max_completion_tokens_spinbox
        self.mangalmm_ocr_parallel_workers_spinbox = self.mangalmm_ocr_page.parallel_workers_spinbox
        self.mangalmm_ocr_request_timeout_spinbox = self.mangalmm_ocr_page.request_timeout_spinbox
        self.mangalmm_ocr_raw_response_logging_checkbox = self.mangalmm_ocr_page.raw_response_logging_checkbox
        self.mangalmm_ocr_safe_resize_checkbox = self.mangalmm_ocr_page.safe_resize_checkbox
        self.mangalmm_ocr_max_pixels_spinbox = self.mangalmm_ocr_page.max_pixels_spinbox
        self.mangalmm_ocr_max_long_side_spinbox = self.mangalmm_ocr_page.max_long_side_spinbox
        self.gemma_chunk_size_spinbox = self.gemma_local_server_page.chunk_size_spinbox
        self.gemma_max_completion_tokens_spinbox = self.gemma_local_server_page.max_completion_tokens_spinbox
        self.gemma_request_timeout_spinbox = self.gemma_local_server_page.request_timeout_spinbox
        self.gemma_temperature_spinbox = self.gemma_local_server_page.temperature_spinbox
        self.gemma_top_k_spinbox = self.gemma_local_server_page.top_k_spinbox
        self.gemma_top_p_spinbox = self.gemma_local_server_page.top_p_spinbox
        self.gemma_min_p_spinbox = self.gemma_local_server_page.min_p_spinbox
        self.gemma_raw_response_logging_checkbox = self.gemma_local_server_page.raw_response_logging_checkbox

        # Credentials
        self.save_keys_checkbox = self.credentials_page.save_keys_checkbox
        self.credential_widgets = self.credentials_page.credential_widgets

        # LLMs
        self.image_checkbox = self.llms_page.image_checkbox
        self.extra_context = self.llms_page.extra_context

        # Text rendering
        self.min_font_spinbox = self.text_rendering_page.min_font_spinbox
        self.max_font_spinbox = self.text_rendering_page.max_font_spinbox
        self.font_browser = self.text_rendering_page.font_browser
        self.uppercase_checkbox = self.text_rendering_page.uppercase_checkbox
        self.ocr_dictionary_table = self.user_dictionaries_page.ocr_dictionary_table
        self.translation_dictionary_table = self.user_dictionaries_page.translation_dictionary_table
        self.enable_completion_sound_checkbox = self.notifications_page.enable_completion_sound_checkbox
        self.completion_sound_combo = self.notifications_page.completion_sound_combo
        self.test_sound_button = self.notifications_page.test_sound_button

        # Export
        self.raw_text_checkbox = self.export_page.raw_text_checkbox
        self.translated_text_checkbox = self.export_page.translated_text_checkbox
        self.inpainted_image_checkbox = self.export_page.inpainted_image_checkbox
        self.detector_overlay_checkbox = self.export_page.detector_overlay_checkbox
        self.raw_mask_checkbox = self.export_page.raw_mask_checkbox
        self.mask_overlay_checkbox = self.export_page.mask_overlay_checkbox
        self.cleanup_mask_delta_checkbox = self.export_page.cleanup_mask_delta_checkbox
        self.debug_metadata_checkbox = self.export_page.debug_metadata_checkbox
        self.individual_format_widget = self.export_page.individual_format_widget
        self.archive_format_widget = self.export_page.archive_format_widget
        self.archive_image_format_widget = self.export_page.archive_image_format_widget
        self.archive_level_widget = self.export_page.archive_level_widget
        self.automatic_output_target_combo = self.export_page.automatic_output_target_combo
        self.automatic_output_image_format_combo = self.export_page.automatic_output_image_format_combo
        self.automatic_output_archive_format_combo = self.export_page.automatic_output_archive_format_combo
        self.automatic_output_archive_image_format_combo = self.export_page.automatic_output_archive_image_format_combo
        self.automatic_output_archive_level_spinbox = self.export_page.automatic_output_archive_level_spinbox
        self.automatic_output_quality_note_label = self.export_page.automatic_output_quality_note_label
        self.automatic_output_archive_note_label = self.export_page.automatic_output_archive_note_label
        self.automatic_output_estimate_summary_label = self.export_page.automatic_output_estimate_summary_label
        self.project_autosave_interval_spinbox = self.project_page.project_autosave_interval_spinbox
        self.project_autosave_folder_input = self.project_page.project_autosave_folder_input
        self.series_failure_policy_combo = self.series_page.failure_policy_combo
        self.series_retry_count_spinbox = self.series_page.retry_count_spinbox
        self.series_retry_delay_spinbox = self.series_page.retry_delay_spinbox
        self.series_auto_open_failed_checkbox = self.series_page.auto_open_failed_checkbox
        self.series_resume_first_incomplete_checkbox = self.series_page.resume_first_incomplete_checkbox
        self.series_return_to_series_checkbox = self.series_page.return_to_series_checkbox

        # System
        self.check_update_button = self.about_page.check_update_button


        # Add pages to stacked widget (order must match navbar order)
        self.stacked_widget.addWidget(self.personalization_page)
        self.stacked_widget.addWidget(self.tools_page)
        self.stacked_widget.addWidget(self.paddleocr_vl_page)
        self.stacked_widget.addWidget(self.hunyuan_ocr_page)
        self.stacked_widget.addWidget(self.mangalmm_ocr_page)
        self.stacked_widget.addWidget(self.gemma_local_server_page)
        self.stacked_widget.addWidget(self.llms_page)
        self.stacked_widget.addWidget(self.text_rendering_page)
        self.stacked_widget.addWidget(self.user_dictionaries_page)
        self.stacked_widget.addWidget(self.notifications_page)
        self.stacked_widget.addWidget(self.project_page)
        self.stacked_widget.addWidget(self.series_page)
        self.stacked_widget.addWidget(self.export_page)
        self.stacked_widget.addWidget(self.shortcuts_page)
        self.stacked_widget.addWidget(self.credentials_page)
        self.stacked_widget.addWidget(self.about_page)

        settings_layout = QtWidgets.QHBoxLayout()
        
        # Create a separate scroll area for the left navbar
        navbar_scroll = QtWidgets.QScrollArea()
        navbar_scroll.setWidget(self._create_navbar_widget())
        navbar_scroll.setWidgetResizable(True)
        navbar_scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        navbar_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        navbar_scroll.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        # Keep navbar at a reasonable width without over-constraining layout
        navbar_scroll.setMinimumWidth(200)
        navbar_scroll.setMaximumWidth(260)
        
        settings_layout.addWidget(navbar_scroll)
        settings_layout.addWidget(MDivider(orientation=QtCore.Qt.Orientation.Vertical))

        # Make only the right-side content scrollable so the left navbar
        # remains fixed and doesn't scroll when the content is scrolled.
        self.content_scroll = QtWidgets.QScrollArea()
        self.content_scroll.setWidget(self.stacked_widget)
        self.content_scroll.setWidgetResizable(True)
        self.content_scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.content_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.content_scroll.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        # Allow the scroll area to take available space
        self.content_scroll.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Expanding,
        )
        settings_layout.addWidget(self.content_scroll, 1)
        settings_layout.setContentsMargins(3, 3, 3, 3)

        # Connect to stacked widget page changes to ensure scroll area recalculates
        self.stacked_widget.currentChanged.connect(self._on_page_changed)

        self.setLayout(settings_layout)

    def _create_navbar_widget(self):
        """Create the navbar as a widget that can be scrolled."""
        navbar_widget = QtWidgets.QWidget()
        navbar_layout = QtWidgets.QVBoxLayout(navbar_widget)
        navbar_layout.setContentsMargins(5, 5, 5, 5)

        for index, setting in enumerate([
            {"title": self.tr("Personalization"), "avatar": MPixmap(".svg")},
            {"title": self.tr("Tools"), "avatar": MPixmap(".svg")},
            {"title": self.tr("PaddleOCR VL Settings"), "avatar": MPixmap(".svg")},
            {"title": self.tr("HunyuanOCR Settings"), "avatar": MPixmap(".svg")},
            {"title": self.tr("MangaLMM Settings"), "avatar": MPixmap(".svg")},
            {"title": self.tr("Gemma Local Server Settings"), "avatar": MPixmap(".svg")},
            {"title": self.tr("LLMs"), "avatar": MPixmap(".svg")},
            {"title": self.tr("Text Rendering"), "avatar": MPixmap(".svg")},
            {"title": self.tr("User Dictionaries"), "avatar": MPixmap(".svg")},
            {"title": self.tr("Notifications"), "avatar": MPixmap(".svg")},
            {"title": self.tr("Project"), "avatar": MPixmap(".svg")},
            {"title": self.tr("Series"), "avatar": MPixmap(".svg")},
            {"title": self.tr("Export"), "avatar": MPixmap(".svg")},
            {"title": self.tr("Shortcuts"), "avatar": MPixmap(".svg")},
            {"title": self.tr("Advanced"), "avatar": MPixmap(".svg")},
            {"title": self.tr("About"), "avatar": MPixmap(".svg")},
        ]):
            nav_card = ClickMeta(extra=False)
            nav_card.setup_data(setting)
            nav_card.clicked.connect(lambda i=index, c=nav_card: self.on_nav_clicked(i, c))
            navbar_layout.addWidget(nav_card)
            self.nav_cards.append(nav_card)

        navbar_layout.addStretch(1)
        return navbar_widget

    def on_nav_clicked(self, index: int, clicked_nav: ClickMeta):
        # Remove highlight from the previously highlighted nav item
        if self.current_highlighted_nav:
            self.current_highlighted_nav.set_highlight(False)

        # Highlight the clicked nav item
        clicked_nav.set_highlight(True)
        self.current_highlighted_nav = clicked_nav

        # Set the current index of the stacked widget
        self.stacked_widget.setCurrentIndex(index)
        # Update geometry so scroll range recalculates for the new page
        self.stacked_widget.updateGeometry()
        # Reset scroll position to top for a better UX
        self.content_scroll.verticalScrollBar().setValue(0)

    def _on_page_changed(self, index):
        """Handle page changes to ensure scroll area recalculates properly."""
        # Force the stacked widget to update its size hint
        self.stacked_widget.updateGeometry()
        # Force the scroll area to recalculate
        self.content_scroll.widget().updateGeometry()
        self.content_scroll.updateGeometry()
        # Reset scroll position
        self.content_scroll.verticalScrollBar().setValue(0)

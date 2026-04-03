from PySide6 import QtCore, QtWidgets
from PySide6.QtCore import QSettings
from PySide6.QtGui import QIntValidator

from app.ui.dayu_widgets import dayu_theme
from app.ui.dayu_widgets.browser import MDragFileButton
from app.ui.dayu_widgets.button_group import MPushButtonGroup, MToolButtonGroup
from app.ui.dayu_widgets.check_box import MCheckBox
from app.ui.dayu_widgets.combo_box import MComboBox, MFontComboBox
from app.ui.dayu_widgets.divider import MDivider
from app.ui.dayu_widgets.line_edit import MLineEdit
from app.ui.dayu_widgets.loading import MLoading
from app.ui.dayu_widgets.progress_bar import MProgressBar
from app.ui.dayu_widgets.push_button import MPushButton
from app.ui.dayu_widgets.radio_button import MRadioButton
from app.ui.dayu_widgets.slider import MSlider
from app.ui.dayu_widgets.text_edit import MTextEdit
from app.ui.dayu_widgets.tool_button import MToolButton
from app.ui.search_replace_panel import SearchReplacePanel
from app.ui.main_window.constants import supported_source_languages, supported_target_languages


class WorkspaceMixin:
    def _create_scope_badge(self, text: str, tooltip: str = ""):
        badge = QtWidgets.QLabel(text)
        badge.setStyleSheet(
            "QLabel {"
            "background-color: #4a4a4a;"
            "color: #f0f0f0;"
            "border-radius: 8px;"
            "padding: 1px 6px;"
            "font-size: 10px;"
            "font-weight: 600;"
            "}"
        )
        if tooltip:
            badge.setToolTip(tooltip)
        return badge

    def _create_render_field_label(self, text: str):
        label = QtWidgets.QLabel(text)
        label.setProperty("render_field_label", "true")
        label.setMinimumWidth(60)
        label.setAlignment(QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter)
        return label

    def _create_render_section_widget(self):
        widget = QtWidgets.QWidget()
        widget.setProperty("render_section", "true")
        return widget

    def _style_render_choice_group(self, button_group: MToolButtonGroup):
        for button in button_group.get_button_group().buttons():
            button.setProperty("render_choice", "true")
            button.setMinimumHeight(30)
            button.setMinimumWidth(46)

    def _create_main_content(self):
        content_widget = QtWidgets.QWidget()

        header_layout = QtWidgets.QHBoxLayout()

        self.undo_tool_group = MToolButtonGroup(orientation=QtCore.Qt.Horizontal, exclusive=True)
        undo_tools = [
            {"svg": "undo.svg", "checkable": False, "tooltip": self.tr("Undo")},
            {"svg": "redo.svg", "checkable": False, "tooltip": self.tr("Redo")},
        ]
        self.undo_tool_group.set_button_list(undo_tools)

        button_config_list = [
            {"text": self.tr("Detect"), "dayu_type": MPushButton.DefaultType, "enabled": False},
            {"text": self.tr("Recognize"), "dayu_type": MPushButton.DefaultType, "enabled": False},
            {"text": self.tr("Translate"), "dayu_type": MPushButton.DefaultType, "enabled": False},
            {"text": self.tr("Segment"), "dayu_type": MPushButton.DefaultType, "enabled": False},
            {"text": self.tr("Clean"), "dayu_type": MPushButton.DefaultType, "enabled": False},
            {"text": self.tr("Render"), "dayu_type": MPushButton.DefaultType, "enabled": False},
        ]

        self.hbutton_group = MPushButtonGroup()
        self.hbutton_group.set_dayu_size(dayu_theme.small)
        self.hbutton_group.set_button_list(button_config_list)
        self.hbutton_group.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        for button in self.hbutton_group.get_button_group().buttons():
            button.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)

        self.progress_bar = MProgressBar().auto_color()
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)

        self.loading = MLoading().small()
        self.loading.setVisible(False)

        self.manual_radio = MRadioButton(self.tr("Manual"))
        self.manual_radio.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)

        self.automatic_radio = MRadioButton(self.tr("Automatic"))
        self.automatic_radio.setChecked(True)
        self.automatic_radio.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)

        self.webtoon_toggle = MToolButton()
        self.webtoon_toggle.set_dayu_svg("webtoon-toggle.svg")
        self.webtoon_toggle.huge()
        self.webtoon_toggle.setCheckable(True)
        self.webtoon_toggle.setToolTip(
            self.tr("Toggle Webtoon Mode. " "For comics that are read in long vertical strips")
        )
        self.webtoon_toggle.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)

        self.translate_button = MPushButton(self.tr("Translate All"))
        self.translate_button.setEnabled(True)
        self.translate_button.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        self.cancel_button = MPushButton(self.tr("Cancel"))
        self.cancel_button.setEnabled(True)
        self.cancel_button.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        self.batch_report_button = MPushButton(self.tr("Report"))
        self.batch_report_button.setEnabled(False)
        self.batch_report_button.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)

        header_layout.addWidget(self.hbutton_group)
        header_layout.addWidget(self.loading)
        header_layout.addStretch()
        header_layout.addWidget(self.webtoon_toggle)
        header_layout.addWidget(self.manual_radio)
        header_layout.addWidget(self.automatic_radio)
        header_layout.addWidget(self.translate_button)
        header_layout.addWidget(self.cancel_button)
        header_layout.addWidget(self.batch_report_button)

        self.search_panel = SearchReplacePanel(self)
        self.search_panel.setVisible(False)

        left_layout = QtWidgets.QVBoxLayout()
        left_layout.addWidget(MDivider())

        self.image_card_layout = QtWidgets.QVBoxLayout()
        self.image_card_layout.addStretch(1)

        self.page_list.setLayout(self.image_card_layout)
        left_layout.addWidget(self.page_list)
        left_layout.addWidget(self.search_panel)
        left_widget = QtWidgets.QWidget()
        left_widget.setLayout(left_layout)

        self.central_stack = QtWidgets.QStackedWidget()

        self.drag_browser = MDragFileButton(text=self.tr("Click or drag files here"), multiple=True)
        self.drag_browser.set_dayu_svg("attachment_line.svg")
        self.drag_browser.set_dayu_filters(
            [
                ".png",
                ".jpg",
                ".jpeg",
                ".webp",
                ".bmp",
                ".zip",
                ".cbz",
                ".cbr",
                ".cb7",
                ".cbt",
                ".pdf",
                ".epub",
                ".ctpr",
            ]
        )
        self.drag_browser.setToolTip(
            self.tr("Import Images, PDFs, Epubs or Comic Book Archive Files(cbr, cbz, etc)")
        )
        self.central_stack.addWidget(self.drag_browser)
        self.central_stack.addWidget(self.image_viewer)

        central_widget = QtWidgets.QWidget()
        central_layout = QtWidgets.QVBoxLayout(central_widget)
        central_layout.addWidget(self.central_stack)
        central_layout.setContentsMargins(10, 10, 10, 10)

        right_layout = QtWidgets.QVBoxLayout()
        right_layout.addWidget(MDivider())

        input_layout = QtWidgets.QHBoxLayout()

        s_combo_text_layout = QtWidgets.QVBoxLayout()
        self.s_combo = MComboBox().medium()
        self.s_combo.addItems([self.tr(lang) for lang in supported_source_languages])
        self.s_combo.setToolTip(self.tr("Source Language"))
        s_combo_text_layout.addWidget(self.s_combo)
        self.s_text_edit = MTextEdit()
        self.s_text_edit.setFixedHeight(120)
        s_combo_text_layout.addWidget(self.s_text_edit)
        input_layout.addLayout(s_combo_text_layout)

        t_combo_text_layout = QtWidgets.QVBoxLayout()
        self.t_combo = MComboBox().medium()
        self.t_combo.addItems([self.tr(lang) for lang in supported_target_languages])
        self.t_combo.setToolTip(self.tr("Target Language"))
        t_combo_text_layout.addWidget(self.t_combo)
        self.t_text_edit = MTextEdit()
        self.t_text_edit.setFixedHeight(120)
        t_combo_text_layout.addWidget(self.t_text_edit)
        input_layout.addLayout(t_combo_text_layout)

        text_render_layout = QtWidgets.QVBoxLayout()
        render_policy_hint = QtWidgets.QLabel(
            self.tr(
                "New Render items and Translate All use the controls below. "
                "Font size edits only the currently selected text item."
            )
        )
        render_policy_hint.setWordWrap(True)

        color_policy_hint = QtWidgets.QLabel(
            self.tr(
                "Text color follows the detected source text by default. "
                "Enable 'Use Selected Color' to override it."
            )
        )
        color_policy_hint.setWordWrap(True)

        self.smart_global_apply_all_checkbox = MCheckBox(self.tr("Apply All SMART Globally"))
        self.smart_global_apply_all_checkbox.hide()

        font_settings_layout = QtWidgets.QHBoxLayout()

        self.font_dropdown = MFontComboBox().small()
        self.font_dropdown.setToolTip(
            self.tr("Font family used for new Render items and Translate All output.")
        )
        self.font_size_dropdown = MComboBox().small()
        self.font_size_dropdown.setToolTip(
            self.tr(
                "Edits only the selected text item. New renders still auto-fit "
                "using the min/max font size settings."
            )
        )
        self.font_size_dropdown.addItems(
            ["4", "6", "8", "9", "10", "11", "12", "14", "16", "18", "20", "22", "24", "28", "32", "36", "48", "72"]
        )
        self.font_size_dropdown.setCurrentText("12")
        self.font_size_dropdown.setFixedWidth(60)
        self.font_size_dropdown.set_editable(True)

        self.line_spacing_dropdown = MComboBox().small()
        self.line_spacing_dropdown.setToolTip(
            self.tr("Line spacing used for new Render items and Translate All output.")
        )
        self.line_spacing_dropdown.addItems(["1.0", "1.1", "1.2", "1.3", "1.4", "1.5"])
        self.line_spacing_dropdown.setFixedWidth(60)
        self.line_spacing_dropdown.set_editable(True)

        font_settings_layout.addWidget(self.font_dropdown)
        font_settings_layout.addWidget(self.font_size_dropdown)
        font_settings_layout.addWidget(self.line_spacing_dropdown)
        font_settings_layout.addStretch()

        settings = QSettings("ComicLabs", "ComicTranslate")
        settings.beginGroup("text_rendering")
        dflt_clr = settings.value("color", "#000000")
        dflt_outline_check = settings.value("outline", True, type=bool)
        settings.endGroup()

        self.block_font_color_button = QtWidgets.QPushButton()
        self.block_font_color_button.setToolTip(
            self.tr(
                "Choose the fallback text color. By default the app keeps the "
                "detected source text color unless you enable 'Use Selected Color'."
            )
        )
        self.block_font_color_button.setFixedSize(30, 30)
        self.block_font_color_button.setStyleSheet(f"background-color: {dflt_clr}; border: none; border-radius: 5px;")
        self.block_font_color_button.setProperty("selected_color", dflt_clr)
        self.force_font_color_checkbox = MCheckBox(self.tr("Use Selected Color"))
        self.force_font_color_checkbox.setToolTip(
            self.tr(
                "Ignore detected source text color and use the selected color "
                "for all new Render items and Translate All output."
            )
        )
        self.force_font_color_checkbox.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Maximum,
            QtWidgets.QSizePolicy.Policy.Fixed,
        )

        self.alignment_tool_group = MToolButtonGroup(orientation=QtCore.Qt.Horizontal, exclusive=True)
        alignment_tools = [
            {"svg": "tabler--align-left.svg", "checkable": True, "tooltip": "Align Left"},
            {"svg": "tabler--align-center.svg", "checkable": True, "tooltip": "Align Center"},
            {"svg": "tabler--align-right.svg", "checkable": True, "tooltip": "Align Right"},
        ]
        self.alignment_tool_group.set_button_list(alignment_tools)
        self.alignment_tool_group.set_dayu_checked(1)
        self.alignment_tool_group.setToolTip(
            self.tr("Horizontal alignment for new Render items and Translate All output.")
        )

        self.vertical_alignment_tool_group = MToolButtonGroup(
            orientation=QtCore.Qt.Horizontal,
            exclusive=True,
        )
        self.vertical_alignment_tool_group.set_button_list(
            [
                {"text": self.tr("Top"), "checkable": True, "tooltip": self.tr("Place text at the top of the source text box.")},
                {"text": self.tr("Center"), "checkable": True, "tooltip": self.tr("Place text in the vertical center of the source text box.")},
                {"text": self.tr("Bottom"), "checkable": True, "tooltip": self.tr("Place text at the bottom of the source text box.")},
            ]
        )
        self.vertical_alignment_tool_group.set_dayu_checked(0)
        self._style_render_choice_group(self.vertical_alignment_tool_group)

        for button in self.alignment_tool_group.get_button_group().buttons():
            button.setMinimumSize(30, 30)

        self.bold_button = self.create_tool_button(svg="bold.svg", checkable=True)
        self.bold_button.setToolTip(
            self.tr("Bold style for new Render items and Translate All output.")
        )
        self.italic_button = self.create_tool_button(svg="italic.svg", checkable=True)
        self.italic_button.setToolTip(
            self.tr("Italic style for new Render items and Translate All output.")
        )
        self.underline_button = self.create_tool_button(svg="underline.svg", checkable=True)
        self.underline_button.setToolTip(
            self.tr("Underline style for new Render items and Translate All output.")
        )
        for button in (self.bold_button, self.italic_button, self.underline_button):
            button.setMinimumSize(30, 30)

        outline_settings_layout = QtWidgets.QHBoxLayout()

        self.outline_checkbox = MCheckBox(self.tr("Outline"))
        self.outline_checkbox.setChecked(dflt_outline_check)
        self.outline_checkbox.hide()

        self.outline_mode_group = MToolButtonGroup(
            orientation=QtCore.Qt.Horizontal,
            exclusive=True,
        )
        self.outline_mode_group.set_button_list(
            [
                {"text": self.tr("OFF"), "checkable": True, "tooltip": self.tr("Disable outline globally.")},
                {"text": self.tr("ON"), "checkable": True, "tooltip": self.tr("Enable outline globally.")},
            ]
        )
        self.outline_mode_group.set_dayu_checked(1 if dflt_outline_check else 0)
        self._style_render_choice_group(self.outline_mode_group)

        self.outline_font_color_button = QtWidgets.QPushButton()
        self.outline_font_color_button.setToolTip(
            self.tr("Outline color for new Render items and Translate All output.")
        )
        self.outline_font_color_button.setFixedSize(30, 30)
        self.outline_font_color_button.setStyleSheet("background-color: white; border: none; border-radius: 5px;")
        self.outline_font_color_button.setProperty("selected_color", "#ffffff")

        self.outline_width_dropdown = MComboBox().small()
        self.outline_width_dropdown.setFixedWidth(60)
        self.outline_width_dropdown.setToolTip(
            self.tr("Outline width for new Render items and Translate All output.")
        )
        self.outline_width_dropdown.addItems(["1.0", "1.15", "1.3", "1.4", "1.5"])
        self.outline_width_dropdown.set_editable(True)

        render_controls_widget = QtWidgets.QWidget()
        render_controls_widget.setObjectName("renderControlsContainer")
        render_controls_widget.setStyleSheet(
            f"""
            QWidget#renderControlsContainer QWidget[render_section="true"] {{
                border: 1px solid #474747;
                border-radius: 6px;
                background-color: #353535;
            }}
            QWidget#renderControlsContainer QLabel[render_field_label="true"] {{
                color: #f0f0f0;
                font-weight: 600;
                border: none;
                background: transparent;
                padding-right: 8px;
            }}
            QWidget#renderControlsContainer MToolButton[render_choice="true"] {{
                padding: 0 10px;
                min-height: 30px;
                border: 1px solid #5a5a5a;
                border-radius: 5px;
                background-color: #3b3b3b;
            }}
            QWidget#renderControlsContainer MToolButton[render_choice="true"]:hover {{
                color: {dayu_theme.primary_color};
                border-color: {dayu_theme.primary_color};
            }}
            QWidget#renderControlsContainer MToolButton[render_choice="true"]:checked {{
                color: {dayu_theme.primary_color};
                border: 1px solid {dayu_theme.primary_color};
                background-color: #4b3a12;
            }}
            """
        )
        render_controls_layout = QtWidgets.QVBoxLayout(render_controls_widget)
        render_controls_layout.setContentsMargins(0, 0, 0, 0)
        render_controls_layout.setSpacing(8)

        color_group_widget = self._create_render_section_widget()
        color_group_layout = QtWidgets.QGridLayout(color_group_widget)
        color_group_layout.setContentsMargins(10, 8, 10, 8)
        color_group_layout.setHorizontalSpacing(10)
        color_group_layout.setVerticalSpacing(6)
        color_group_layout.setColumnStretch(1, 1)
        color_group_layout.addWidget(self._create_render_field_label(self.tr("Text Color")), 0, 0)
        color_group_layout.addWidget(
            self.block_font_color_button,
            0,
            1,
            alignment=QtCore.Qt.AlignmentFlag.AlignLeft,
        )
        color_group_layout.addWidget(self.force_font_color_checkbox, 1, 0, 1, 2)

        alignment_group_widget = self._create_render_section_widget()
        alignment_group_layout = QtWidgets.QGridLayout(alignment_group_widget)
        alignment_group_layout.setContentsMargins(10, 8, 10, 8)
        alignment_group_layout.setHorizontalSpacing(10)
        alignment_group_layout.setVerticalSpacing(6)
        alignment_group_layout.setColumnStretch(1, 1)
        alignment_group_layout.addWidget(self._create_render_field_label(self.tr("Horizontal")), 0, 0)
        alignment_group_layout.addWidget(self.alignment_tool_group, 0, 1)
        alignment_group_layout.addWidget(self._create_render_field_label(self.tr("Vertical")), 1, 0)
        alignment_group_layout.addWidget(self.vertical_alignment_tool_group, 1, 1)

        style_controls_widget = QtWidgets.QWidget()
        style_controls_layout = QtWidgets.QHBoxLayout(style_controls_widget)
        style_controls_layout.setContentsMargins(0, 0, 0, 0)
        style_controls_layout.setSpacing(4)
        style_controls_layout.addWidget(self.bold_button)
        style_controls_layout.addWidget(self.italic_button)
        style_controls_layout.addWidget(self.underline_button)
        style_controls_layout.addStretch()

        outline_settings_layout.setContentsMargins(0, 0, 0, 0)
        outline_settings_layout.setSpacing(6)
        outline_settings_layout.addWidget(self.outline_mode_group)
        outline_settings_layout.addWidget(self.outline_font_color_button)
        outline_settings_layout.addWidget(self.outline_width_dropdown)
        outline_settings_layout.addStretch()

        style_outline_group_widget = self._create_render_section_widget()
        style_outline_group_layout = QtWidgets.QGridLayout(style_outline_group_widget)
        style_outline_group_layout.setContentsMargins(10, 8, 10, 8)
        style_outline_group_layout.setHorizontalSpacing(10)
        style_outline_group_layout.setVerticalSpacing(6)
        style_outline_group_layout.setColumnStretch(1, 1)
        style_outline_group_layout.addWidget(self._create_render_field_label(self.tr("Style")), 0, 0)
        style_outline_group_layout.addWidget(style_controls_widget, 0, 1)
        style_outline_group_layout.addWidget(self._create_render_field_label(self.tr("Outline")), 1, 0)
        style_outline_group_layout.addLayout(outline_settings_layout, 1, 1)

        render_controls_layout.addWidget(color_group_widget)
        render_controls_layout.addWidget(alignment_group_widget)
        render_controls_layout.addWidget(style_outline_group_widget)

        rendering_divider_top = MDivider()
        rendering_divider_bottom = MDivider()
        text_render_layout.addWidget(rendering_divider_top)
        text_render_layout.addWidget(render_policy_hint)
        text_render_layout.addWidget(color_policy_hint)
        text_render_layout.addLayout(font_settings_layout)
        text_render_layout.addWidget(render_controls_widget)
        text_render_layout.addWidget(rendering_divider_bottom)

        tools_widget = QtWidgets.QWidget()
        tools_layout = QtWidgets.QVBoxLayout()

        misc_lay = QtWidgets.QHBoxLayout()

        self.pan_button = self.create_tool_button(svg="pan_tool.svg", checkable=True)
        self.pan_button.setToolTip(self.tr("Pan Image"))
        self.pan_button.clicked.connect(self.toggle_pan_tool)
        self.tool_buttons["pan"] = self.pan_button

        self.set_all_button = MPushButton(self.tr("Set for all"))
        self.set_all_button.setToolTip(
            self.tr("Sets the Source and Target Language on the current page for all pages")
        )

        misc_lay.addWidget(self.pan_button)
        misc_lay.addWidget(self.set_all_button)
        misc_lay.addStretch()

        box_tools_lay = QtWidgets.QHBoxLayout()

        self.box_button = self.create_tool_button(svg="select.svg", checkable=True)
        self.box_button.setToolTip(self.tr("Draw or Select Text Boxes"))
        self.box_button.clicked.connect(self.toggle_box_tool)
        self.tool_buttons["box"] = self.box_button

        self.delete_button = self.create_tool_button(svg="trash_line.svg", checkable=False)
        self.delete_button.setToolTip(self.tr("Delete Selected Box"))

        self.clear_rectangles_button = self.create_tool_button(svg="clear-outlined.svg")
        self.clear_rectangles_button.setToolTip(self.tr("Remove all the Boxes on the Image"))

        self.draw_blklist_blks = self.create_tool_button(svg="gridicons--create.svg")
        self.draw_blklist_blks.setToolTip(
            self.tr(
                "Draws all the Text Blocks in the existing Text Block List\n"
                "back on the Image (for further editing)"
            )
        )

        box_tools_lay.addWidget(self.box_button)
        box_tools_lay.addWidget(self.delete_button)
        box_tools_lay.addWidget(self.clear_rectangles_button)
        box_tools_lay.addWidget(self.draw_blklist_blks)

        self.change_all_blocks_size_dec = self.create_tool_button(svg="minus_line.svg")
        self.change_all_blocks_size_dec.setToolTip(self.tr("Reduce the size of all blocks"))

        self.change_all_blocks_size_diff = MLineEdit()
        self.change_all_blocks_size_diff.setFixedWidth(30)
        self.change_all_blocks_size_diff.setText("3")

        int_validator = QIntValidator()
        self.change_all_blocks_size_diff.setValidator(int_validator)
        self.change_all_blocks_size_diff.setAlignment(QtCore.Qt.AlignCenter)

        self.change_all_blocks_size_inc = self.create_tool_button(svg="add_line.svg")
        self.change_all_blocks_size_inc.setToolTip(self.tr("Increase the size of all blocks"))

        box_tools_lay.addStretch()
        box_tools_lay.addWidget(self.change_all_blocks_size_dec)
        box_tools_lay.addWidget(self.change_all_blocks_size_diff)
        box_tools_lay.addWidget(self.change_all_blocks_size_inc)
        box_tools_lay.addStretch()

        inp_tools_lay = QtWidgets.QHBoxLayout()

        self.brush_button = self.create_tool_button(svg="brush-fill.svg", checkable=True)
        self.brush_button.setToolTip(self.tr("Draw Brush Strokes for Cleaning Image"))
        self.brush_button.clicked.connect(self.toggle_brush_tool)
        self.tool_buttons["brush"] = self.brush_button

        self.eraser_button = self.create_tool_button(svg="eraser_fill.svg", checkable=True)
        self.eraser_button.setToolTip(self.tr("Erase Brush Strokes"))
        self.eraser_button.clicked.connect(self.toggle_eraser_tool)
        self.tool_buttons["eraser"] = self.eraser_button

        self.clear_brush_strokes_button = self.create_tool_button(svg="clear-outlined.svg")
        self.clear_brush_strokes_button.setToolTip(self.tr("Remove all the brush strokes on the Image"))

        inp_tools_lay.addWidget(self.brush_button)
        inp_tools_lay.addWidget(self.eraser_button)
        inp_tools_lay.addWidget(self.clear_brush_strokes_button)
        inp_tools_lay.addStretch()

        self.brush_eraser_slider = MSlider()
        self.brush_eraser_slider.setMinimum(1)
        self.brush_eraser_slider.setMaximum(100)
        self.brush_eraser_slider.setValue(10)
        self.brush_eraser_slider.setToolTip(self.tr("Brush/Eraser Size Slider"))
        self.brush_eraser_slider.valueChanged.connect(self.set_brush_eraser_size)

        tools_layout.addLayout(misc_lay)
        box_div = MDivider(self.tr("Box Drawing"))
        tools_layout.addWidget(box_div)
        tools_layout.addLayout(box_tools_lay)

        inp_div = MDivider(self.tr("Inpainting"))
        tools_layout.addWidget(inp_div)
        tools_layout.addLayout(inp_tools_lay)
        tools_layout.addWidget(self.brush_eraser_slider)
        tools_layout.addStretch()
        tools_widget.setLayout(tools_layout)

        tools_scroll = QtWidgets.QScrollArea()
        tools_scroll.setWidgetResizable(True)
        tools_scroll.setWidget(tools_widget)
        tools_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        tools_scroll.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        tools_scroll.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)

        right_layout.addLayout(input_layout)
        right_layout.addLayout(text_render_layout)
        right_layout.addWidget(tools_scroll, 1)

        right_widget = QtWidgets.QWidget()
        right_widget.setLayout(right_layout)

        splitter = QtWidgets.QSplitter()
        splitter.addWidget(left_widget)
        splitter.addWidget(central_widget)
        splitter.addWidget(right_widget)

        right_widget.setMinimumWidth(320)

        splitter.setStretchFactor(0, 40)
        splitter.setStretchFactor(1, 80)
        splitter.setStretchFactor(2, 10)

        content_layout = QtWidgets.QVBoxLayout()
        content_layout.addLayout(header_layout)
        content_layout.addWidget(self.progress_bar)
        content_layout.addWidget(splitter)

        content_layout.setStretchFactor(header_layout, 0)
        content_layout.setStretchFactor(splitter, 1)

        content_widget.setLayout(content_layout)

        return content_widget

    def create_tool_button(self, text: str = "", svg: str = "", checkable: bool = False):
        if text:
            button = MToolButton().svg(svg).text_beside_icon()
            button.setText(text)
        else:
            button = MToolButton().svg(svg)

        button.setCheckable(True) if checkable else button.setCheckable(False)

        return button

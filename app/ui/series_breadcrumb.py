from __future__ import annotations

from PySide6 import QtCore, QtWidgets

from .dayu_widgets import dayu_theme


class SeriesBreadcrumbBar(QtWidgets.QWidget):
    """자식 프로젝트를 편집하는 동안 상시 노출되는 시리즈 컨텍스트 표시줄.

    예전에는 자식으로 들어가면 문서 화면으로 스택이 넘어가면서 보드로 돌아갈
    진입점이 사라졌다. 보드의 `←` 버튼은 보드 위젯 안에 있어 보이지 않고,
    파이프라인 상태 패널의 `Series Board` 버튼은 큐 실행 중에만 뜬다. 그래서
    수동으로 항목을 열어 편집하는 동안에는 복귀 경로가 하나도 없었다.

    같은 이유로 "지금 단일 프로젝트인가, 시리즈의 한 화인가"도 구분되지
    않았다. 유일한 단서인 창 제목은 커스텀 타이틀바가 폭에 따라 숨긴다.
    이 표시줄은 자식이 활성일 때만 나타나므로, 존재 자체가 그 구분이 된다.
    """

    board_requested = QtCore.Signal()
    back_requested = QtCore.Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("seriesBreadcrumbBar")

        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(6, 2, 8, 2)
        layout.setSpacing(4)

        self.back_button = QtWidgets.QToolButton(self)
        self.back_button.setObjectName("seriesBreadcrumbBack")
        self.back_button.setText("←")
        self.back_button.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)

        self.series_button = QtWidgets.QToolButton(self)
        self.series_button.setObjectName("seriesBreadcrumbSeries")
        self.series_button.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)

        self.separator_label = QtWidgets.QLabel("›", self)
        self.separator_label.setObjectName("seriesBreadcrumbSeparator")

        self.child_label = QtWidgets.QLabel("", self)
        self.child_label.setObjectName("seriesBreadcrumbChild")

        self.unsynced_badge = QtWidgets.QLabel(self.tr("Not Synced"), self)
        self.unsynced_badge.setObjectName("seriesBreadcrumbUnsynced")
        self.unsynced_badge.setToolTip(
            self.tr("This chapter has changes that are not written back to the series project yet.")
        )
        self.unsynced_badge.hide()

        layout.addWidget(self.back_button)
        layout.addWidget(self.series_button)
        layout.addWidget(self.separator_label)
        layout.addWidget(self.child_label)
        layout.addWidget(self.unsynced_badge)

        self.back_button.clicked.connect(self.back_requested)
        self.series_button.clicked.connect(self.board_requested)

        self._apply_theme_styles()
        self._retranslate_static_tooltips()

    def _apply_theme_styles(self) -> None:
        accent = dayu_theme.primary_color or dayu_theme.yellow or "#fadb14"
        text = dayu_theme.primary_text_color or "#d9d9d9"
        sub_text = dayu_theme.secondary_text_color or "#a6a6a6"
        self.setStyleSheet(
            f"""
            QWidget#seriesBreadcrumbBar {{
                background-color: rgba(255, 255, 255, 18);
                border: 1px solid rgba(255, 255, 255, 36);
                border-radius: 4px;
            }}
            QToolButton#seriesBreadcrumbBack {{
                color: {text};
                border: none;
                padding: 1px 4px;
                font-size: 12px;
            }}
            QToolButton#seriesBreadcrumbSeries {{
                color: {accent};
                border: none;
                padding: 1px 2px;
                font-size: 11px;
                font-weight: 600;
            }}
            QToolButton#seriesBreadcrumbBack:disabled,
            QToolButton#seriesBreadcrumbSeries:disabled {{
                color: {sub_text};
            }}
            QLabel#seriesBreadcrumbSeparator {{
                color: {sub_text};
                font-size: 11px;
            }}
            QLabel#seriesBreadcrumbChild {{
                color: {text};
                font-size: 11px;
            }}
            QLabel#seriesBreadcrumbUnsynced {{
                background-color: #8a6d1f;
                color: #f5f5f5;
                border-radius: 8px;
                padding: 1px 6px;
                font-size: 10px;
                font-weight: 600;
            }}
            """
        )

    def _retranslate_static_tooltips(self) -> None:
        self.series_button.setToolTip(self.tr("Back to the series board"))

    def _elide(self, text: str, width: int) -> str:
        metrics = self.fontMetrics()
        return metrics.elidedText(text, QtCore.Qt.TextElideMode.ElideMiddle, width)

    def set_context(
        self,
        *,
        series_name: str,
        child_name: str,
        unsynced: bool = False,
        can_back: bool = False,
        locked_reason: str = "",
    ) -> None:
        """자식 컨텍스트를 표시줄에 반영한다.

        `locked_reason` 이 비어 있지 않으면 히스토리 이동만 잠근다. 보드 링크는
        큐 실행 중에도 살아 있어야 한다 (`show_board_during_queue` 가 자식
        materialization 을 유지한 채 보드 화면만 보여준다).
        """
        self.series_button.setText(self._elide(series_name, 240))
        self.series_button.setToolTip(
            self.tr("Back to the series board") + (f"\n{series_name}" if series_name else "")
        )
        self.child_label.setText(self._elide(child_name, 240))
        self.child_label.setToolTip(child_name)
        self.unsynced_badge.setVisible(bool(unsynced))

        self.back_button.setEnabled(bool(can_back) and not locked_reason)
        self.back_button.setToolTip(locked_reason or self.tr("Back"))

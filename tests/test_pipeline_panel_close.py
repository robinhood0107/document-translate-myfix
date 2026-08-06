"""닫기를 누르면 닫혀 있어야 한다.

패널은 진행 이벤트가 도착할 때 `if not self.isVisible(): self.show()` 로 스스로를
다시 띄웠다. 실패가 반복되는 동안에는 닫는 즉시 다시 열려서, 사용자에게는 닫기
버튼이 아예 듣지 않는 것으로 보였다.
"""

from __future__ import annotations

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6 import QtWidgets  # noqa: E402

from app.ui.pipeline_status_panel import PipelineStatusPanel  # noqa: E402


class PipelinePanelCloseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def _panel(self) -> PipelineStatusPanel:
        host = QtWidgets.QWidget()
        host.resize(1200, 800)
        # 부모가 숨겨져 있으면 자식은 show() 뒤에도 isVisible() 이 False 다. 그것은
        # Qt 의 부모 가시성 규칙이고 제품 동작이 아니므로, 부모를 띄워서 판정이
        # 의미를 갖게 한다.
        host.show()
        panel = PipelineStatusPanel(host)
        self.addCleanup(panel.deleteLater)
        self.addCleanup(host.deleteLater)
        self.addCleanup(host.hide)
        return panel

    def _running_event(self, **extra) -> dict:
        payload = {
            "phase": "pipeline",
            "service": "batch",
            "status": "running",
            "step_key": "inpaint-all",
            "stage_name": "inpaint-all",
            "message": "원본 텍스트 제거(인페인팅): 92/366 페이지",
            "page_index": 91,
            "page_total": 366,
            "image_name": "092.png",
        }
        payload.update(extra)
        return payload

    def test_a_later_event_does_not_reopen_a_closed_panel(self) -> None:
        panel = self._panel()
        panel.prepare_for_new_run()
        panel.update_event(self._running_event())
        self.assertTrue(panel.isVisible())

        panel.close_panel()
        self.assertFalse(panel.isVisible())

        # 실패가 반복되는 동안 계속 들어오는 이벤트들.
        for _ in range(3):
            panel.update_event(self._running_event(status="failed", phase="error"))
        self.assertFalse(
            panel.isVisible(),
            "닫은 패널이 이후 이벤트로 다시 열렸습니다.",
        )

    def test_a_new_run_shows_the_panel_again(self) -> None:
        panel = self._panel()
        panel.prepare_for_new_run()
        panel.close_panel()
        self.assertFalse(panel.isVisible())

        panel.prepare_for_new_run()
        self.assertTrue(
            panel.isVisible(),
            "닫힘 의도가 다음 실행까지 남았습니다.",
        )

    def test_closing_while_running_in_window_mode_minimises_instead(self) -> None:
        """실행 중 창 모드에서는 최소화가 기존 동작이다. 그 의도는 유지한다."""

        panel = self._panel()
        panel.prepare_for_new_run()
        panel.set_display_mode(panel.WINDOW_MODE)
        panel.update_event(self._running_event())
        self.assertTrue(panel._pipeline_active)

        panel.close_panel()
        self.assertTrue(panel._closed_by_user)


if __name__ == "__main__":
    unittest.main()

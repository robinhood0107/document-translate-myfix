"""폰트를 고르지 않아도 렌더가 실패해서는 안 된다.

배치 시작에서 사전 검사 게이트를 없앴다. 그 전에는 폰트가 비면 `font_selected` 가
실행을 아예 거부했다. 이제는 실행이 시작되므로, 빈 폰트가 렌더 단계에서 366장을
연달아 실패시키지 않는다는 것이 계약이 된다.
"""

from __future__ import annotations

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6 import QtWidgets  # noqa: E402
from PySide6.QtGui import QFont  # noqa: E402


class RenderFontFallbackTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def test_an_empty_family_does_not_raise(self) -> None:
        from modules.rendering.render import describe_render_text_sanitization

        # 폰트를 고르지 않은 상태 그대로 넘긴다.
        result = describe_render_text_sanitization("안녕하세요", "")
        self.assertEqual(result.text, "안녕하세요")

        # 공백만 있는 값도 같은 경로를 타야 한다.
        result = describe_render_text_sanitization("안녕하세요", "   ")
        self.assertEqual(result.text, "안녕하세요")

    def test_the_application_always_has_a_usable_family(self) -> None:
        """대체 대상이 비어 있으면 대체의 의미가 없다."""

        family = QtWidgets.QApplication.font().family()
        self.assertTrue(str(family).strip())
        self.assertTrue(QFont(family, 12).family())

    def test_the_fallback_is_applied_in_the_render_source(self) -> None:
        """빈 폰트를 그대로 QFont 에 넘기면 지표 계산이 흔들린다."""

        import inspect

        from modules.rendering import render

        source = inspect.getsource(render)
        self.assertIn("QApplication.font().family()", source)


if __name__ == "__main__":
    unittest.main()

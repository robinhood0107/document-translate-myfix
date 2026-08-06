"""고른 글꼴은 저장되어야 하고, 표시와 실제 값이 어긋나서는 안 된다.

복원 코드는 `text_rendering/font_family` 를 읽고 있었는데 쓰는 곳이 없었다. 그래서
값은 언제나 빈 문자열이었고, 복원 분기는 매번 콤보를 비웠다. 결과적으로 화면에는
글꼴 이름이 보이는데 `currentText()` 는 빈 문자열이어서, 사용자에게는 "분명히 골랐는데
적용되지 않는" 것으로 보였고 번역 결과도 그 글꼴로 렌더되지 않았다.
"""

from __future__ import annotations

import os
import unittest
from types import SimpleNamespace
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6 import QtCore, QtWidgets  # noqa: E402

from app.controllers.text import TextController  # noqa: E402


GROUP = "text_rendering"
KEY = "font_family"


class RenderFontPersistenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def setUp(self) -> None:
        # 사용자의 실제 설정을 건드리지 않도록 별도 조직명을 쓴다.
        self._settings = QtCore.QSettings("ComicLabs", "ComicTranslate")
        self._original = self._settings.value(f"{GROUP}/{KEY}", "")
        self.addCleanup(self._restore)

    def _restore(self) -> None:
        self._settings.beginGroup(GROUP)
        if self._original:
            self._settings.setValue(KEY, self._original)
        else:
            self._settings.remove(KEY)
        self._settings.endGroup()
        self._settings.sync()

    def _stored(self) -> str:
        settings = QtCore.QSettings("ComicLabs", "ComicTranslate")
        return str(settings.value(f"{GROUP}/{KEY}", "") or "")

    def test_persisting_writes_the_family(self) -> None:
        TextController.persist_render_font_family("Ownglyph gumama3 Regular")
        self.assertEqual(self._stored(), "Ownglyph gumama3 Regular")

    def test_persisting_trims_surrounding_space(self) -> None:
        TextController.persist_render_font_family("  Ownglyph gumama3 Regular  ")
        self.assertEqual(self._stored(), "Ownglyph gumama3 Regular")

    def test_an_empty_choice_never_overwrites_a_real_one(self) -> None:
        """콤보가 일시적으로 빈 값을 내보내도 저장된 선택을 지우면 안 된다."""

        TextController.persist_render_font_family("Ownglyph gumama3 Regular")
        for blank in ("", "   ", None):
            TextController.persist_render_font_family(blank)
            self.assertEqual(self._stored(), "Ownglyph gumama3 Regular")

    def test_changing_the_dropdown_persists_without_a_selected_item(self) -> None:
        """텍스트 항목을 고르지 않은 상태에서도 저장돼야 한다.

        예전 핸들러는 선택된 항목이 있을 때만 동작했다. 배치 렌더의 기본 글꼴은
        선택과 무관하므로 저장은 항상 일어나야 한다.
        """

        controller = object.__new__(TextController)
        controller.main = SimpleNamespace(
            font_dropdown=SimpleNamespace(currentText=lambda: "Meiryo"),
            font_size_dropdown=SimpleNamespace(currentText=lambda: "48"),
        )
        with mock.patch.object(
            TextController, "_selected_text_items", return_value=[]
        ):
            TextController.on_font_dropdown_change(controller, "Meiryo")
        self.assertEqual(self._stored(), "Meiryo")


class RestoreKeepsDisplayAndValueAlignedTests(unittest.TestCase):
    """저장된 값이 없을 때 콤보를 비우면 표시와 값이 어긋난다."""

    def test_the_restore_branch_no_longer_blanks_the_dropdown(self) -> None:
        import inspect

        from app.controllers import projects

        source = inspect.getsource(projects)
        self.assertNotIn("font_dropdown.setCurrentText('')", source)
        self.assertIn("persist_render_font_family", source)


if __name__ == "__main__":
    unittest.main()


class ImportedFontIsTheDefaultTests(unittest.TestCase):
    """저장된 선택이 없으면 가져온 글꼴이 기본값이어야 한다.

    예전에는 이 자리에서 Qt 응용 프로그램 기본 글꼴(`맑은 고딕` 등)로 떨어졌다.
    저장된 값이 한 번 비면 시스템 UI 글꼴이 확정됐고, 만화 번역 결과가 그 글꼴로
    렌더됐다. 글꼴을 가져왔다는 것 자체가 그 글꼴로 렌더하겠다는 뜻이다.
    """

    def test_the_fallback_prefers_an_imported_font(self) -> None:
        import inspect

        from app.controllers import projects

        source = inspect.getsource(projects.ProjectController.load_main_page_settings)
        # 콤보/Qt 기본값보다 가져온 글꼴을 먼저 본다.
        self.assertLess(
            source.index("user_font_path"),
            source.index("QtWidgets.QApplication.font().family()"),
        )

    def test_the_choice_is_deterministic(self) -> None:
        """실행마다 다른 글꼴이 뽑히면 안 된다."""

        import inspect

        from app.controllers import projects

        source = inspect.getsource(projects.ProjectController.load_main_page_settings)
        self.assertIn("sorted(os.listdir(user_font_path))", source)

    def test_the_block_is_marked_for_easy_removal(self) -> None:
        """나중에 이 기본값 규칙을 걷어낼 때 어디를 지울지 분명해야 한다."""

        import inspect

        from app.controllers import projects

        source = inspect.getsource(projects.ProjectController.load_main_page_settings)
        self.assertIn("이 블록만 지우면 원래 동작", source)
        self.assertIn("--- 여기까지 ---", source)

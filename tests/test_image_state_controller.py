from __future__ import annotations

import unittest

from app.controllers.image import ImageStateController


class _DummyCombo:
    def __init__(self, value: str) -> None:
        self._value = value

    def currentText(self) -> str:
        return self._value


class _DummyMain:
    def __init__(self) -> None:
        self.image_states: dict[str, dict] = {}
        self.s_combo = _DummyCombo("Japanese")
        self.t_combo = _DummyCombo("English")


class ImageStateControllerTests(unittest.TestCase):
    def test_apply_languages_to_paths_updates_only_selected_pages(self) -> None:
        main = _DummyMain()
        controller = ImageStateController.__new__(ImageStateController)
        controller.main = main

        controller.ensure_page_state("page-a.png")
        controller.ensure_page_state("page-b.png")
        controller.ensure_page_state("page-c.png")

        controller.apply_languages_to_paths(
            ["page-a.png", "page-c.png"],
            "Chinese",
            "Korean",
        )

        self.assertEqual(main.image_states["page-a.png"]["source_lang"], "Chinese")
        self.assertEqual(main.image_states["page-a.png"]["target_lang"], "Korean")
        self.assertEqual(main.image_states["page-c.png"]["source_lang"], "Chinese")
        self.assertEqual(main.image_states["page-c.png"]["target_lang"], "Korean")
        self.assertEqual(main.image_states["page-b.png"]["source_lang"], "Japanese")
        self.assertEqual(main.image_states["page-b.png"]["target_lang"], "English")


if __name__ == "__main__":
    unittest.main()

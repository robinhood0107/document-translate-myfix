from __future__ import annotations

import unittest
from types import SimpleNamespace

from modules.utils.correction_dictionary import (
    apply_ocr_result_dictionary,
    apply_substitution_rules,
    apply_translation_result_dictionary,
)


class CorrectionDictionaryTests(unittest.TestCase):
    def test_apply_substitution_rules_supports_literal_regex_and_case_modes(self) -> None:
        rules = [
            {"keyword": "foo", "sub": "bar", "use_reg": False, "case_sens": False},
            {"keyword": r"\d{2}", "sub": "NN", "use_reg": True, "case_sens": True},
        ]

        result = apply_substitution_rules("Foo 12 fOO 34 baz", rules)

        self.assertEqual(result, "bar NN bar NN baz")

    def test_apply_substitution_rules_ignores_invalid_regex(self) -> None:
        rules = [
            {"keyword": "(", "sub": "broken", "use_reg": True, "case_sens": True},
            {"keyword": "safe", "sub": "done", "use_reg": False, "case_sens": True},
        ]

        result = apply_substitution_rules("safe", rules)

        self.assertEqual(result, "done")

    def test_block_level_dictionary_helpers_mutate_saved_fields(self) -> None:
        blocks = [
            SimpleNamespace(text="ocr foo", translation="trans foo"),
            SimpleNamespace(text="OCR FOO", translation="TRANS FOO"),
        ]
        rules = [{"keyword": "foo", "sub": "bar", "use_reg": False, "case_sens": False}]

        apply_ocr_result_dictionary(blocks, rules)
        apply_translation_result_dictionary(blocks, rules)

        self.assertEqual([blk.text for blk in blocks], ["ocr bar", "OCR bar"])
        self.assertEqual([blk.translation for blk in blocks], ["trans bar", "TRANS bar"])


if __name__ == "__main__":
    unittest.main()

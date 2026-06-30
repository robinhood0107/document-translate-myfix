from __future__ import annotations

import unittest

from modules.utils.inpainting_runtime import is_lama_family_inpainter


class InpaintingRuntimeFamilyTests(unittest.TestCase):
    def test_lama_family_accepts_current_product_variants_and_legacy_alias(self) -> None:
        self.assertTrue(is_lama_family_inpainter("lama_large_512px"))
        self.assertTrue(is_lama_family_inpainter("lama_mpe"))
        self.assertTrue(is_lama_family_inpainter("LaMa"))

    def test_lama_family_rejects_non_lama_and_empty_values(self) -> None:
        for key in ("AOT", "MI-GAN", "", None, "unknown"):
            with self.subTest(key=key):
                self.assertFalse(is_lama_family_inpainter(key))


if __name__ == "__main__":
    unittest.main()

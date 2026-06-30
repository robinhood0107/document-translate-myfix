from __future__ import annotations

import unittest
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from benchmark_common import (
    NonLaMaInpainterError,
    product_benchmark_failure_contract,
    resolve_product_benchmark_contract,
)
import benchmark_no_gemma_replay_pipeline as no_gemma_replay  # noqa: E402


class BenchmarkLaMaFamilyContractTests(unittest.TestCase):
    def test_contract_accepts_lama_family_and_records_resolved_key(self) -> None:
        contract = resolve_product_benchmark_contract(
            {
                "app": {"inpainter": "lama_mpe"},
                "mask_refiner_settings": {"mask_refiner": "ctd"},
            }
        )

        self.assertTrue(contract["product_pipeline_entrypoint"])
        self.assertEqual(contract["runner_render_mode"], "product")
        self.assertEqual(contract["inpainter_family"], "lama")
        self.assertEqual(contract["inpainter"], "lama_mpe")
        self.assertEqual(contract["mask_refiner"], "ctd")

    def test_contract_normalizes_legacy_lama_alias(self) -> None:
        contract = resolve_product_benchmark_contract(
            {
                "app": {"inpainter": "LaMa"},
                "mask_refiner_settings": {"mask_refiner": "ctd"},
            }
        )

        self.assertEqual(contract["inpainter_family"], "lama")
        self.assertEqual(contract["inpainter"], "lama_large_512px")

    def test_contract_rejects_non_lama_inpainters(self) -> None:
        for key in ("AOT", "MI-GAN", "", None, "unknown"):
            with self.subTest(key=key):
                with self.assertRaises(NonLaMaInpainterError) as raised:
                    resolve_product_benchmark_contract({"app": {"inpainter": key}})
                self.assertEqual(raised.exception.reason, "non_lama_inpainter")

    def test_failure_contract_records_requested_and_resolved_inpainter(self) -> None:
        with self.assertRaises(NonLaMaInpainterError) as raised:
            resolve_product_benchmark_contract({"app": {"inpainter": "AOT"}})

        failure = product_benchmark_failure_contract(raised.exception)

        self.assertEqual(failure["reason_code"], "non_lama_inpainter")
        self.assertEqual(failure["requested_inpainter"], "AOT")
        self.assertEqual(failure["inpainter"], "AOT")
        self.assertEqual(failure["inpainter_family"], "")

    def test_no_gemma_replay_build_preset_preserves_lama_family_for_contract(self) -> None:
        preset = no_gemma_replay._build_preset(
            {
                "app": {"inpainter": "lama_mpe"},
                "mask_refiner_settings": {"mask_refiner": "ctd"},
            },
            use_gpu=True,
        )

        contract = resolve_product_benchmark_contract(preset)

        self.assertEqual(contract["inpainter_family"], "lama")
        self.assertEqual(contract["inpainter"], "lama_mpe")

    def test_no_gemma_replay_build_preset_does_not_silently_replace_aot(self) -> None:
        preset = no_gemma_replay._build_preset({"app": {"inpainter": "AOT"}}, use_gpu=True)

        with self.assertRaises(NonLaMaInpainterError):
            resolve_product_benchmark_contract(preset)


if __name__ == "__main__":
    unittest.main()

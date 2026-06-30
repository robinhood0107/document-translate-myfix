from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import benchmark_stage_batched_pipeline as stage_pipeline  # noqa: E402
from benchmark_common import NonLaMaInpainterError  # noqa: E402


class StageBatchedBenchmarkEntrypointTests(unittest.TestCase):
    def test_default_runner_uses_product_pipeline_entrypoint(self) -> None:
        runner = object.__new__(stage_pipeline.StageBatchedRunner)
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("CT_BENCH_STAGE_BATCHED_LEGACY_CUSTOM_RUNNER", None)
            with mock.patch.object(
                stage_pipeline.StageBatchedRunner,
                "_run_product_pipeline_entrypoint",
                return_value={"product_pipeline_entrypoint": True},
            ) as product_entrypoint, mock.patch.object(
                stage_pipeline.StageBatchedRunner,
                "_close_window",
            ) as close_window, mock.patch.object(
                stage_pipeline.StageBatchedRunner,
                "_load_window",
                side_effect=AssertionError("custom runner path must stay disabled by default"),
            ):
                summary = runner.run()

        self.assertTrue(summary["product_pipeline_entrypoint"])
        product_entrypoint.assert_called_once_with()
        close_window.assert_called_once_with()

    def test_runner_product_contract_accepts_lama_family(self) -> None:
        runner = object.__new__(stage_pipeline.StageBatchedRunner)
        runner.preset = {
            "app": {"inpainter": "lama_mpe"},
            "mask_refiner_settings": {"mask_refiner": "ctd"},
        }

        contract = runner._product_benchmark_contract()

        self.assertEqual(contract["inpainter_family"], "lama")
        self.assertEqual(contract["inpainter"], "lama_mpe")
        self.assertEqual(contract["mask_refiner"], "ctd")

    def test_runner_product_contract_rejects_non_lama_family(self) -> None:
        runner = object.__new__(stage_pipeline.StageBatchedRunner)
        runner.preset = {"app": {"inpainter": "AOT"}}

        with self.assertRaises(NonLaMaInpainterError) as raised:
            runner._product_benchmark_contract()

        self.assertEqual(raised.exception.reason, "non_lama_inpainter")


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import os
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from modules.utils.local_llama_router import RouterPairKind  # noqa: E402
from modules.utils.local_llama_router.contracts import (  # noqa: E402
    DEFAULT_MANGALMM_ROUTER_ENDPOINT,
    ROUTER_GEMMA_HOST_PORT,
    classify_router_pair,
    router_pair_for_engine_key,
    router_pair_for_ocr_endpoint,
)

GEMMA_ENDPOINT = "http://127.0.0.1:18080/v1"
GEMMA_MODEL = "gemma-4-26B-IQ4_NL.gguf"


class MangaLMMRouterPairContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.pair = router_pair_for_engine_key("MangaLMM")

    def test_mangalmm_is_a_router_pair_on_its_own_ocr_port(self) -> None:
        self.assertIsNotNone(self.pair)
        self.assertEqual(self.pair.kind, RouterPairKind.MANGALMM)
        self.assertEqual(self.pair.ocr_port, 28081)
        self.assertEqual(self.pair.ocr_alias, "MangaLMM.Q8_0.gguf")
        self.assertEqual(self.pair.container_name, "comic-translate-router-mangalmm-v2")

    def test_every_pair_shares_the_one_gemma_host_port(self) -> None:
        """The shared port is why releasing any pair frees another's leftover."""

        for engine in ("PaddleOCR VL", "PaddleOCR VL Spotting", "MangaLMM"):
            pair = router_pair_for_engine_key(engine)
            self.assertEqual(pair.gemma_port, ROUTER_GEMMA_HOST_PORT, engine)

    def test_router_ocr_ports_are_unique_per_pair(self) -> None:
        ports = [
            router_pair_for_engine_key(engine).ocr_port
            for engine in ("PaddleOCR VL", "PaddleOCR VL Spotting", "MangaLMM")
        ]
        self.assertEqual(len(ports), len(set(ports)))

    def test_compose_and_preset_files_exist(self) -> None:
        self.assertTrue(Path(self.pair.compose_file).is_file())
        self.assertTrue(Path(self.pair.preset_file).is_file())

    def test_preset_declares_both_models_and_no_autoload(self) -> None:
        text = Path(self.pair.preset_file).read_text(encoding="utf-8")
        self.assertIn(f"[{self.pair.ocr_alias}]", text)
        self.assertIn(f"[{GEMMA_MODEL}]", text)
        self.assertEqual(text.count("load-on-startup = false"), 3)

    def test_preset_freezes_the_product_default_runtime_options(self) -> None:
        """A tuned separate-server option must not silently differ here."""

        from modules.ocr.mangalmm_full_page.runtime import (
            DEFAULT_MANGALMM_RUNTIME_OPTIONS,
        )

        text = Path(self.pair.preset_file).read_text(encoding="utf-8")
        expected = {
            "ctx-size": DEFAULT_MANGALMM_RUNTIME_OPTIONS["MANGALMM_LLAMA_CTX_SIZE"],
            "parallel": DEFAULT_MANGALMM_RUNTIME_OPTIONS["MANGALMM_LLAMA_PARALLEL"],
            "threads": DEFAULT_MANGALMM_RUNTIME_OPTIONS["MANGALMM_LLAMA_THREADS"],
            "batch-size": DEFAULT_MANGALMM_RUNTIME_OPTIONS["MANGALMM_LLAMA_BATCH_SIZE"],
            "ubatch-size": DEFAULT_MANGALMM_RUNTIME_OPTIONS[
                "MANGALMM_LLAMA_UBATCH_SIZE"
            ],
            "n-gpu-layers": DEFAULT_MANGALMM_RUNTIME_OPTIONS[
                "MANGALMM_LLAMA_GPU_LAYERS"
            ],
        }
        section = text.split(f"[{self.pair.ocr_alias}]", 1)[1].split("[gemma", 1)[0]
        for key, value in expected.items():
            self.assertIn(f"{key} = {value}", section)

    def test_only_the_exact_managed_endpoint_classifies(self) -> None:
        self.assertIsNotNone(
            classify_router_pair(
                "MangaLMM",
                DEFAULT_MANGALMM_ROUTER_ENDPOINT,
                GEMMA_ENDPOINT,
                GEMMA_MODEL,
            )
        )
        for custom in (
            "http://127.0.0.1:28081/v1/",
            "http://127.0.0.1:28081/v1?x=1",
            "http://127.0.0.1:28081/v1#frag",
            "http://localhost:28081/v1",
            "http://127.0.0.1:29999/v1",
        ):
            self.assertIsNone(
                router_pair_for_ocr_endpoint("MangaLMM", custom), custom
            )

    def test_a_non_default_gemma_model_stays_user_managed(self) -> None:
        self.assertIsNone(
            classify_router_pair(
                "MangaLMM",
                DEFAULT_MANGALMM_ROUTER_ENDPOINT,
                GEMMA_ENDPOINT,
                "some-other-model.gguf",
            )
        )


class RouterServiceNameTests(unittest.TestCase):
    """A Router load and its release must resolve one Arbiter lease name."""

    def _manager(self):
        from modules.ocr.local_runtime import LocalOCRRuntimeManager
        from modules.utils.local_llama_router import LocalLlamaRouterCoordinator

        return LocalOCRRuntimeManager(
            router_coordinator=LocalLlamaRouterCoordinator()
        )

    def test_every_router_engine_has_a_lease_name(self) -> None:
        manager = self._manager()
        expected = {
            "PaddleOCR VL": "paddleocr_vl",
            "PaddleOCR VL Spotting": "paddleocr_vl_spotting",
            "MangaLMM": "mangalmm",
            "HunyuanOCR": "hunyuanocr",
        }
        for engine, service in expected.items():
            self.assertEqual(manager._router_service_name(engine), service)

    def test_release_default_matches_the_owned_pair(self) -> None:
        manager = self._manager()
        self.assertEqual(manager._router_owned_service_default(), "ocr")
        manager._router_pair = router_pair_for_engine_key("MangaLMM")
        self.assertEqual(manager._router_owned_service_default(), "mangalmm")
        manager._router_pair = router_pair_for_engine_key("PaddleOCR VL Spotting")
        self.assertEqual(
            manager._router_owned_service_default(), "paddleocr_vl_spotting"
        )


class MangaLMMRouterOptionGuardTests(unittest.TestCase):
    """A tuned runtime option must fall back to the separate-server route."""

    def _manager(self):
        from modules.ocr.local_runtime import LocalOCRRuntimeManager
        from modules.utils.local_llama_router import LocalLlamaRouterCoordinator

        return LocalOCRRuntimeManager(
            router_coordinator=LocalLlamaRouterCoordinator()
        )

    def test_default_options_allow_the_router(self) -> None:
        manager = self._manager()
        pair = router_pair_for_engine_key("MangaLMM")
        self.assertTrue(manager._router_preset_matches_runtime_options(pair))

    def test_a_tuned_option_rejects_the_router(self) -> None:
        manager = self._manager()
        pair = router_pair_for_engine_key("MangaLMM")
        previous = os.environ.get("MANGALMM_LLAMA_CTX_SIZE")
        os.environ["MANGALMM_LLAMA_CTX_SIZE"] = "16384"
        try:
            self.assertFalse(manager._router_preset_matches_runtime_options(pair))
        finally:
            if previous is None:
                os.environ.pop("MANGALMM_LLAMA_CTX_SIZE", None)
            else:
                os.environ["MANGALMM_LLAMA_CTX_SIZE"] = previous

    def test_an_invalid_option_rejects_the_router_instead_of_raising(self) -> None:
        manager = self._manager()
        pair = router_pair_for_engine_key("MangaLMM")
        previous = os.environ.get("MANGALMM_LLAMA_THREADS")
        os.environ["MANGALMM_LLAMA_THREADS"] = "not-a-number"
        try:
            self.assertFalse(manager._router_preset_matches_runtime_options(pair))
        finally:
            if previous is None:
                os.environ.pop("MANGALMM_LLAMA_THREADS", None)
            else:
                os.environ["MANGALMM_LLAMA_THREADS"] = previous

    def test_the_paddle_pairs_are_unaffected_by_the_guard(self) -> None:
        manager = self._manager()
        for engine in ("PaddleOCR VL", "PaddleOCR VL Spotting"):
            pair = router_pair_for_engine_key(engine)
            self.assertTrue(manager._router_preset_matches_runtime_options(pair))


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

from modules.translation.llm.custom_local_gemma import CustomLocalGemmaTranslation
from modules.translation.local_runtime import LocalGemmaRuntimeManager
from modules.translation.processor import Translator
from modules.translation.translation_memory import TranslationMemoryStore
from modules.utils.correction_dictionary import apply_translation_result_dictionary
from modules.utils.textblock import TextBlock
from pipeline.cache_manager import CacheManager


class _Settings:
    class _UI:
        @staticmethod
        def tr(value: str) -> str:
            return value

    ui = _UI()

    def __init__(self, memory_settings: dict) -> None:
        self._memory_settings = dict(memory_settings)

    @staticmethod
    def get_tool_selection(tool: str) -> str:
        if tool == "translator":
            return "Custom Local Server(Gemma)"
        if tool == "ocr":
            return "default"
        return ""

    @staticmethod
    def get_llm_settings() -> dict:
        return {"image_input_enabled": False}

    @staticmethod
    def is_gpu_enabled() -> bool:
        return False

    def get_translation_memory_settings(self) -> dict:
        return dict(self._memory_settings)

    @staticmethod
    def get_translation_result_dictionary_rules() -> list[dict]:
        return []


def _response(translation: str) -> dict:
    return {
        "choices": [
            {
                "finish_reason": "stop",
                "message": {
                    "content": json.dumps(
                        {"translation": translation},
                        ensure_ascii=False,
                    )
                },
            }
        ],
        "usage": {},
    }


class TranslationProcessorMemoryTests(unittest.TestCase):
    @staticmethod
    def _engine(settings: _Settings) -> CustomLocalGemmaTranslation:
        engine = CustomLocalGemmaTranslation()
        engine.settings = settings
        engine.source_lang = "Japanese"
        engine.target_lang = "Korean"
        engine.chunk_size = 6
        engine.max_tokens = 512
        engine.timeout = 1
        return engine

    @staticmethod
    def _block() -> TextBlock:
        return TextBlock(
            text_bbox=np.array([0, 0, 100, 100]),
            text="source",
        )

    def test_translator_initialization_is_lazy_but_cache_miss_starts_runtime(self) -> None:
        settings = _Settings(
            {
                "persistent_cache_enabled": False,
                "exact_tm_enabled": False,
                "result_cache_limit": 1000,
                "candidate_limit": 100,
            }
        )
        manager = LocalGemmaRuntimeManager()
        main_page = SimpleNamespace(
            settings_page=settings,
            lang_mapping={"Japanese": "Japanese", "Korean": "Korean"},
            local_translation_runtime_manager=manager,
            is_current_task_cancelled=lambda: False,
            report_runtime_progress=lambda _payload: None,
        )
        with (
            tempfile.TemporaryDirectory() as temp_dir,
            TranslationMemoryStore(Path(temp_dir) / "tm.sqlite3") as store,
        ):
            main_page.translation_memory_store = store
            engine = self._engine(settings)
            with mock.patch(
                "modules.translation.processor.TranslationFactory.create_engine",
                return_value=engine,
            ), mock.patch.object(
                manager,
                "ensure_server",
            ) as ensure_server, mock.patch.object(
                engine,
                "_request_translation",
                return_value=_response("translated"),
            ):
                translator = Translator(main_page, "Japanese", "Korean")
                ensure_server.assert_not_called()
                blocks = [self._block()]
                translator.translate(
                    blocks,
                    np.zeros((1, 1, 3), dtype=np.uint8),
                    "",
                )

            ensure_server.assert_called_once()
            self.assertEqual(blocks[0].translation, "translated")

    def test_disabled_database_reports_one_clear_fail_open_warning(self) -> None:
        settings = _Settings(
            {
                "persistent_cache_enabled": True,
                "exact_tm_enabled": True,
                "result_cache_limit": 1000,
                "candidate_limit": 100,
            }
        )
        manager = LocalGemmaRuntimeManager()
        events: list[dict] = []
        main_page = SimpleNamespace(
            settings_page=settings,
            lang_mapping={"Japanese": "Japanese", "Korean": "Korean"},
            local_translation_runtime_manager=manager,
            is_current_task_cancelled=lambda: False,
            report_runtime_progress=lambda payload: events.append(dict(payload)),
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "tm.sqlite3"
            db_path.write_bytes(b"not sqlite")
            with TranslationMemoryStore(db_path) as store:
                main_page.translation_memory_store = store
                engine = self._engine(settings)
                identity = {
                    "model_sha256": "a" * 64,
                    "runtime_fingerprint": "runtime-a",
                }
                with mock.patch(
                    "modules.translation.processor.TranslationFactory.create_engine",
                    return_value=engine,
                ), mock.patch.object(
                    manager,
                    "ensure_server",
                ), mock.patch.object(
                    manager,
                    "get_translation_cache_identity",
                    return_value=identity,
                ), mock.patch.object(
                    engine,
                    "_request_translation",
                    return_value=_response("translated"),
                ):
                    translator = Translator(main_page, "Japanese", "Korean")
                    translator.translate(
                        [self._block()],
                        np.zeros((1, 1, 3), dtype=np.uint8),
                        "",
                    )
                    translator._report_translation_memory_warning()

                warnings = [
                    event
                    for event in events
                    if event.get("translation_memory_disabled")
                ]
                self.assertEqual(len(warnings), 1)
                self.assertEqual(warnings[0]["status"], "warning")
                self.assertIn("DatabaseError", warnings[0]["detail"])

    def test_non_gemma_legacy_cache_keeps_raw_translation_and_avoids_second_call(
        self,
    ) -> None:
        translator = object.__new__(Translator)
        translator._local_gemma_runtime_manager = None
        translator.source_lang = "Japanese"
        translator.target_lang = "Korean"
        translator.translator_key = "GPT-4.1"
        translator.engine = SimpleNamespace()
        translate_call_count = 0

        def translate(blocks, _image, _extra_context):
            nonlocal translate_call_count
            translate_call_count += 1
            for block in blocks:
                block.translation = "cat"
            return blocks

        translator.translate = translate
        cache_manager = CacheManager()
        image = np.zeros((8, 8, 3), dtype=np.uint8)
        rules = [
            {
                "keyword": "cat",
                "sub": "cat!",
                "use_reg": False,
                "case_sens": True,
            }
        ]

        first_blocks = [self._block()]
        _, first_status = translator.translate_with_cache_manager(
            first_blocks,
            image,
            "context",
            cache_manager,
        )
        apply_translation_result_dictionary(first_blocks, rules)

        second_blocks = [self._block()]
        _, second_status = translator.translate_with_cache_manager(
            second_blocks,
            image,
            "context",
            cache_manager,
        )
        apply_translation_result_dictionary(second_blocks, rules)

        self.assertEqual(translate_call_count, 1)
        self.assertEqual(first_status, "refreshed")
        self.assertEqual(second_status, "hit")
        self.assertEqual(first_blocks[0].translation, "cat!")
        self.assertEqual(second_blocks[0].translation, "cat!")


if __name__ == "__main__":
    unittest.main()

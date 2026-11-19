from __future__ import annotations

import json
import os
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import requests
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6 import QtCore, QtWidgets

from app.controllers.batch_report import BatchReportController
from app.controllers.image import ImageStateController

from modules.translation.llm.custom_local_gemma import CustomLocalGemmaTranslation
from modules.translation.local_runtime import LocalGemmaRuntimeManager
from modules.translation.factory import TranslationFactory
from modules.translation.llm.gpt import GPTTranslation
from modules.translation.microsoft import MicrosoftTranslation
from modules.translation.processor import Translator
from modules.translation.translation_memory import TranslationMemoryStore
from modules.utils.correction_dictionary import apply_translation_result_dictionary
from modules.utils.textblock import TextBlock
from pipeline.cache_manager import CacheManager
from pipeline.stage_batched_processor import StageBatchedProcessor, StagePageContext
from pipeline.translation_handler import TranslationHandler


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


class _CustomServiceSettings:
    class _UI:
        def __init__(self) -> None:
            self.uppercase_checkbox = SimpleNamespace(isChecked=lambda: False)

        @staticmethod
        def tr(value: str) -> str:
            return value

    def __init__(self, model: str, endpoint: str) -> None:
        self.ui = self._UI()
        self.credentials = {
            "api_key": "",
            "api_url": endpoint,
            "model": model,
        }

    def get_tool_selection(self, tool: str) -> str:
        return "Custom Service" if tool == "translator" else "default"

    def get_credentials(self, _service: str) -> dict[str, str]:
        return dict(self.credentials)

    @staticmethod
    def get_llm_settings() -> dict[str, object]:
        return {"image_input_enabled": False, "extra_context": ""}

    @staticmethod
    def is_gpu_enabled() -> bool:
        return False

    @staticmethod
    def get_translation_result_dictionary_rules() -> list[dict]:
        return [
            {
                "keyword": "cat",
                "sub": "cat!",
                "use_reg": False,
                "case_sens": True,
            }
        ]


class _MicrosoftSettings:
    class _UI:
        @staticmethod
        def tr(value: str) -> str:
            return value

    def __init__(self) -> None:
        self.ui = self._UI()

    @staticmethod
    def get_tool_selection(tool: str) -> str:
        return "Microsoft Translator" if tool == "translator" else "default"

    @staticmethod
    def get_credentials(_service: str) -> dict[str, str]:
        return {
            "api_key_translator": "example-key",
            "region_translator": "example-region",
        }

    @staticmethod
    def get_llm_settings() -> dict[str, object]:
        return {"image_input_enabled": False, "extra_context": ""}

    @staticmethod
    def is_gpu_enabled() -> bool:
        return False

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
        translator.source_lang_en = "Japanese"
        translator.target_lang_en = "Korean"
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

    def test_persistent_memory_bypasses_legacy_cache_manager(self) -> None:
        translator = object.__new__(Translator)
        translator._local_gemma_runtime_manager = object()
        translator.engine = SimpleNamespace(translation_cache_status="persistent-hit")
        blocks = [self._block()]
        image = np.zeros((8, 8, 3), dtype=np.uint8)
        translator.translate = mock.Mock(return_value=blocks)
        cache_manager = mock.Mock()

        translated, status = translator.translate_with_cache_manager(
            blocks,
            image,
            "context",
            cache_manager,
        )

        self.assertIs(translated, blocks)
        self.assertEqual(status, "persistent-hit")
        translator.translate.assert_called_once_with(blocks, image, "context")
        cache_manager._get_translation_cache_key.assert_not_called()

    def test_non_gemma_cache_uses_frozen_engine_identity_and_current_rules(self) -> None:
        settings = _CustomServiceSettings("model-a", "https://first.example.invalid/v1")
        main_page = SimpleNamespace(
            settings_page=settings,
            lang_mapping={"Japanese": "Japanese", "Korean": "Korean"},
        )
        cache_manager = CacheManager()
        image = np.zeros((8, 8, 3), dtype=np.uint8)
        dispatched: list[tuple[str, str, str, str]] = []

        def mock_provider(engine, *_args):
            dispatched.append(
                (
                    engine.model,
                    engine.api_base_url,
                    engine.source_lang,
                    engine.target_lang,
                )
            )
            return json.dumps({"block_0": "cat"})

        with (
            mock.patch.object(TranslationFactory, "_engines", {}),
            mock.patch.object(GPTTranslation, "_perform_translation", mock_provider),
        ):
            first_translator = Translator(main_page, "Japanese", "Korean")
            settings.credentials.update(
                model="model-b",
                api_url="https://second.example.invalid/v1",
            )

            first_blocks = [self._block()]
            _, first_status = first_translator.translate_with_cache_manager(
                first_blocks,
                image,
                "context",
                cache_manager,
            )
            apply_translation_result_dictionary(
                first_blocks,
                settings.get_translation_result_dictionary_rules(),
            )

            second_translator = Translator(main_page, "Japanese", "Korean")
            second_blocks = [self._block()]
            _, second_status = second_translator.translate_with_cache_manager(
                second_blocks,
                image,
                "context",
                cache_manager,
            )
            apply_translation_result_dictionary(
                second_blocks,
                settings.get_translation_result_dictionary_rules(),
            )

            repeated_blocks = [self._block()]
            _, repeated_status = second_translator.translate_with_cache_manager(
                repeated_blocks,
                image,
                "context",
                cache_manager,
            )
            apply_translation_result_dictionary(
                repeated_blocks,
                settings.get_translation_result_dictionary_rules(),
            )

        self.assertEqual(first_status, "refreshed")
        self.assertEqual(second_status, "refreshed")
        self.assertEqual(repeated_status, "hit")
        self.assertEqual(
            dispatched,
            [
                ("model-a", "https://first.example.invalid/v1", "Japanese", "Korean"),
                ("model-b", "https://second.example.invalid/v1", "Japanese", "Korean"),
            ],
        )
        self.assertEqual(first_blocks[0].translation, "cat!")
        self.assertEqual(second_blocks[0].translation, "cat!")
        self.assertEqual(repeated_blocks[0].translation, "cat!")
        self.assertNotIn("model-a", repr(cache_manager.translation_cache))
        self.assertNotIn("model-b", repr(cache_manager.translation_cache))
        self.assertNotIn("example.invalid", repr(cache_manager.translation_cache))

    def test_selected_block_cache_uses_engine_identity(self) -> None:
        settings = _CustomServiceSettings("model-a", "https://first.example.invalid/v1")
        image = np.zeros((8, 8, 3), dtype=np.uint8)
        block = self._block()
        source_combo = SimpleNamespace(value="日本語")
        source_combo.currentText = lambda: source_combo.value
        target_combo = SimpleNamespace(value="한국어")
        target_combo.currentText = lambda: target_combo.value
        main_page = SimpleNamespace(
            settings_page=settings,
            lang_mapping={
                "日本語": "Japanese",
                "한국어": "Korean",
                "Japanese": "Japanese",
                "Korean": "Korean",
            },
            s_combo=source_combo,
            t_combo=target_combo,
            image_viewer=SimpleNamespace(
                hasPhoto=lambda: True,
                get_image_array=lambda: image,
            ),
            blk_list=[block],
            curr_img_idx=0,
            image_files=[],
        )
        cache_manager = CacheManager()
        pipeline = SimpleNamespace(get_selected_block=lambda: block)
        handler = TranslationHandler(main_page, cache_manager, pipeline)
        dispatched: list[tuple[str, str, str, str]] = []

        def mock_provider(engine, *_args):
            dispatched.append(
                (
                    engine.model,
                    engine.api_base_url,
                    engine.source_lang,
                    engine.target_lang,
                )
            )
            return json.dumps({"block_0": "cat"})

        with (
            mock.patch.object(TranslationFactory, "_engines", {}),
            mock.patch.object(GPTTranslation, "_perform_translation", mock_provider),
        ):
            handler.translate_image(single_block=True)
            self.assertEqual(block.translation, "cat!")

            block.translation = ""
            settings.credentials.update(
                model="model-b",
                api_url="https://second.example.invalid/v1",
            )
            handler.translate_image(single_block=True)
            self.assertEqual(block.translation, "cat!")

            block.translation = ""
            source_combo.value = "Japanese"
            target_combo.value = "Korean"
            handler.translate_image(single_block=True)
            self.assertEqual(block.translation, "cat!")

        self.assertEqual(
            dispatched,
            [
                ("model-a", "https://first.example.invalid/v1", "Japanese", "Korean"),
                ("model-b", "https://second.example.invalid/v1", "Japanese", "Korean"),
            ],
        )


class MicrosoftTranslationResponseContractTests(unittest.TestCase):
    @staticmethod
    def _block(
        text: str,
        translation: str,
        index: int,
        *,
        action: str = "",
        semantic_role: str = "",
    ) -> TextBlock:
        block = TextBlock(
            text_bbox=np.asarray([index * 10, 0, index * 10 + 8, 8]),
            text=text,
            block_id=f"example-block-{index}",
            text_class="text_bubble",
        )
        block.translation = translation
        if action:
            block.processing_action = action
            block.semantic_role = semantic_role
        return block

    @staticmethod
    def _response(payload):
        return SimpleNamespace(
            json=lambda: payload,
            raise_for_status=mock.Mock(),
        )

    @staticmethod
    def _translator(settings: _MicrosoftSettings) -> Translator:
        main_page = SimpleNamespace(
            settings_page=settings,
            lang_mapping={"Japanese": "Japanese", "Korean": "Korean"},
        )
        return Translator(main_page, "Japanese", "Korean")

    @staticmethod
    def _rows(*translations: str) -> list[dict]:
        return [
            {"translations": [{"text": value}]}
            for value in translations
        ]

    def _run_batch(self, blocks: list[TextBlock], payload):
        settings = _MicrosoftSettings()
        image_ctrl = SimpleNamespace(
            update_processing_summary=mock.Mock(),
            mark_processing_stage=mock.Mock(),
        )
        image_skipped = SimpleNamespace(emit=mock.Mock())
        main_page = SimpleNamespace(
            settings_page=settings,
            lang_mapping={"Japanese": "Japanese", "Korean": "Korean"},
            local_translation_runtime_manager=None,
            image_ctrl=image_ctrl,
            image_skipped=image_skipped,
            is_current_task_cancelled=lambda: False,
        )
        processor = object.__new__(StageBatchedProcessor)
        processor.main_page = main_page
        processor.cache_manager = CacheManager()
        processor._project_checkpoint_store = None
        processor._inpainter_release_gate = {"required": False, "observed": True}
        processor._record_performance_detail = mock.Mock()
        processor._set_current_image = mock.Mock()
        processor._report_runtime_progress = mock.Mock()
        processor._emit_benchmark_event = mock.Mock()
        processor._persist_translation_state = mock.Mock()
        processor._released_page_buffer_bytes = 0
        processor._sample_performance_resources = mock.Mock()
        processor._translation_benchmark_metrics = mock.Mock(return_value={})
        processor._measure_performance = lambda **_kwargs: nullcontext()
        processor.emit_progress = mock.Mock()
        page = StagePageContext(
            image_path="example_source_page.png",
            image_name="example_source_page.png",
            source_lang="Japanese",
            target_lang="Korean",
            image=np.zeros((4, 4, 3), dtype=np.uint8),
            blk_list=blocks,
        )
        with (
            mock.patch.object(TranslationFactory, "_engines", {}),
            mock.patch(
                "modules.translation.microsoft.requests.post",
                return_value=self._response(payload),
            ) as post,
        ):
            processor._translate_all([page])
        return processor, page, image_ctrl, post, main_page

    def test_incomplete_responses_do_not_mutate_blocks_or_cache(self) -> None:
        responses = (
            [],
            self._rows("fresh-0"),
            [*self._rows("fresh-0"), {}],
            [*self._rows("fresh-0"), {"translations": []}],
            [*self._rows("fresh-0"), {"translations": [{}]}],
            [*self._rows("fresh-0"), {"translations": [{"text": "  "}]}],
        )
        for payload in responses:
            with self.subTest(payload=payload):
                blocks = [
                    self._block("source-0", "old-0", 0),
                    self._block("source-1", "old-1", 1),
                ]
                cache_manager = CacheManager()
                with (
                    mock.patch.object(TranslationFactory, "_engines", {}),
                    mock.patch(
                        "modules.translation.microsoft.requests.post",
                        return_value=self._response(payload),
                    ),
                ):
                    translator = self._translator(_MicrosoftSettings())
                    self.assertIsInstance(translator.engine, MicrosoftTranslation)
                    with self.assertRaisesRegex(
                        ValueError,
                        "^Invalid translation response$",
                    ):
                        translator.translate_with_cache_manager(
                            blocks,
                            np.zeros((4, 4, 3), dtype=np.uint8),
                            "",
                            cache_manager,
                        )

                self.assertEqual(
                    [block.translation for block in blocks],
                    ["old-0", "old-1"],
                )
                self.assertEqual(cache_manager.translation_cache, {})

    def test_second_batch_failure_does_not_publish_first_batch(self) -> None:
        blocks = [
            self._block(f"source-{index}", f"old-{index}", index)
            for index in range(26)
        ]
        cache_manager = CacheManager()
        with (
            mock.patch.object(TranslationFactory, "_engines", {}),
            mock.patch(
                "modules.translation.microsoft.requests.post",
                side_effect=(
                    self._response(self._rows(*(f"fresh-{index}" for index in range(25)))),
                    self._response([]),
                ),
            ) as post,
        ):
            translator = self._translator(_MicrosoftSettings())
            with self.assertRaisesRegex(
                ValueError,
                "^Invalid translation response$",
            ):
                translator.translate_with_cache_manager(
                    blocks,
                    np.zeros((4, 4, 3), dtype=np.uint8),
                    "",
                    cache_manager,
                )

        self.assertEqual(post.call_count, 2)
        self.assertEqual(
            [block.translation for block in blocks],
            [f"old-{index}" for index in range(26)],
        )
        self.assertEqual(cache_manager.translation_cache, {})

    def test_blank_source_target_is_preserved_on_success(self) -> None:
        blocks = [
            self._block("source", "old", 0),
            self._block(" \n ", "protected target", 1),
        ]
        cache_manager = CacheManager()
        with (
            mock.patch.object(TranslationFactory, "_engines", {}),
            mock.patch(
                "modules.translation.microsoft.requests.post",
                return_value=self._response(self._rows("fresh")),
            ),
        ):
            translator = self._translator(_MicrosoftSettings())
            translator.translate_with_cache_manager(
                blocks,
                np.zeros((4, 4, 3), dtype=np.uint8),
                "",
                cache_manager,
            )

        self.assertEqual(
            [block.translation for block in blocks],
            ["fresh", "protected target"],
        )

    def test_http_error_is_propagated_unchanged(self) -> None:
        error = requests.HTTPError("example transport failure")
        blocks = [self._block("source", "old", 0)]
        cache_manager = CacheManager()
        response = self._response([])
        response.raise_for_status.side_effect = error
        with (
            mock.patch.object(TranslationFactory, "_engines", {}),
            mock.patch(
                "modules.translation.microsoft.requests.post",
                return_value=response,
            ),
        ):
            translator = self._translator(_MicrosoftSettings())
            with self.assertRaises(requests.HTTPError) as caught:
                translator.translate_with_cache_manager(
                    blocks,
                    np.zeros((4, 4, 3), dtype=np.uint8),
                    "",
                    cache_manager,
                )

        self.assertIs(caught.exception, error)
        self.assertEqual(blocks[0].translation, "old")
        self.assertEqual(cache_manager.translation_cache, {})

    def test_batch_failure_does_not_persist_or_mark_page_complete(self) -> None:
        required = self._block(
            "required source",
            "previous target",
            0,
            action="translate_inpaint",
            semantic_role="dialogue_bubble",
        )
        protected = self._block(
            "protected source",
            "keep target",
            1,
            action="preserve",
            semantic_role="decorative",
        )
        processor, page, image_ctrl, _, main_page = self._run_batch(
            [required, protected],
            [],
        )

        self.assertEqual(page.failed_stage, "translation")
        self.assertEqual(page.failed_reason, "Invalid translation response")
        self.assertEqual(required.translation, "previous target")
        self.assertEqual(protected.translation, "keep target")
        self.assertEqual(processor.cache_manager.translation_cache, {})
        processor._persist_translation_state.assert_not_called()
        image_ctrl.mark_processing_stage.assert_called_once()
        self.assertEqual(
            image_ctrl.mark_processing_stage.call_args.args[:3],
            ("example_source_page.png", "translation", "failed"),
        )
        self.assertIsNone(page.image)
        self.assertGreater(processor._released_page_buffer_bytes, 0)
        emitted = main_page.image_skipped.emit.call_args.args
        self.assertEqual(
            emitted,
            (
                "example_source_page.png",
                "translation",
                "Invalid translation response",
            ),
        )

        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        translator = QtCore.QTranslator()
        qm_path = (
            Path(__file__).resolve().parents[1]
            / "resources"
            / "translations"
            / "compiled"
            / "ct_ko.qm"
        )
        self.assertTrue(translator.load(str(qm_path)))
        app.installTranslator(translator)
        try:
            page_notice = ImageStateController.__new__(ImageStateController)._build_skip_message(
                *emitted
            )
            report_reason = BatchReportController(SimpleNamespace())._format_batch_skip_reason(
                emitted[1],
                emitted[2],
            )
        finally:
            app.removeTranslator(translator)

        for message in (page_notice, report_reason):
            self.assertIn("번역 실패", message)
            self.assertIn("번역 응답이 올바르지 않습니다", message)
            self.assertIn("다른 번역기를 사용해 보세요", message)
            self.assertNotIn("Invalid translation response", message)

    def test_batch_success_translates_only_required_blocks(self) -> None:
        required = self._block(
            "required source",
            "",
            0,
            action="translate_inpaint",
            semantic_role="dialogue_bubble",
        )
        protected = self._block(
            "protected source",
            "keep target",
            1,
            action="preserve",
            semantic_role="decorative",
        )
        processor, page, _, post, _ = self._run_batch(
            [required, protected],
            self._rows("fresh"),
        )

        self.assertEqual(post.call_args.kwargs["json"], [{"text": "requiredsource"}])
        self.assertEqual(required.translation, "fresh")
        self.assertEqual(protected.translation, "keep target")
        self.assertEqual(page.source_unit_outcomes["example-block-1"]["status"], "preserved")
        self.assertEqual(page.failed_stage, "")
        processor._persist_translation_state.assert_called_once()


if __name__ == "__main__":
    unittest.main()

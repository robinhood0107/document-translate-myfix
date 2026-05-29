from __future__ import annotations

from pathlib import Path
from unittest import mock

from modules.ocr.local_runtime import LocalOCRRuntimeManager
from modules.ocr.processor import OCRProcessor
from modules.translation.local_runtime import LocalGemmaRuntimeManager
from modules.translation.processor import Translator


ROOT = Path(__file__).resolve().parents[1]
SAMPLE_JAPAN_PAGES = [
    ROOT / "Sample" / "japan" / "094.png",
    ROOT / "Sample" / "japan" / "095.png",
    ROOT / "Sample" / "japan" / "096.png",
]


class _DummyUI:
    @staticmethod
    def tr(text: str) -> str:
        return text


class _DummySettingsPage:
    ui = _DummyUI()

    def get_tool_selection(self, tool_type: str) -> str:
        if tool_type == "translator":
            return "Custom Local Server(Gemma)"
        if tool_type == "ocr":
            return "best_local"
        return ""

    def get_credentials(self, provider: str) -> dict:
        assert provider == "Custom Local Server(Gemma)"
        return {
            "api_url": "http://127.0.0.1:38080/v1",
            "model": "gemma-test.gguf",
        }

    def get_paddleocr_vl_settings(self) -> dict:
        return {"server_url": "http://127.0.0.1:28118/layout-parsing"}

    def get_hunyuan_ocr_settings(self) -> dict:
        return {"server_url": "http://127.0.0.1:28080/v1"}

    def get_mangalmm_ocr_settings(self) -> dict:
        return {"server_url": "http://127.0.0.1:28081/v1"}

    @staticmethod
    def is_gpu_enabled() -> bool:
        return True


class _DummyMainPage:
    def __init__(self) -> None:
        self.settings_page = _DummySettingsPage()
        self.lang_mapping = {"Japanese": "Japanese", "Korean": "Korean"}
        self.local_translation_runtime_manager = LocalGemmaRuntimeManager()
        self.local_ocr_runtime_manager = LocalOCRRuntimeManager()
        self.runtime_events: list[dict] = []

    def report_runtime_progress(self, payload: dict) -> None:
        self.runtime_events.append(dict(payload))

    @staticmethod
    def is_current_task_cancelled() -> bool:
        return False


def test_sample_japan_three_translator_initializations_probe_gemma_once() -> None:
    for path in SAMPLE_JAPAN_PAGES:
        assert path.is_file()

    main_page = _DummyMainPage()

    with mock.patch.object(
        main_page.local_translation_runtime_manager,
        "_wait_for_any_probe",
        return_value=True,
    ) as wait_for_probe, \
         mock.patch.object(main_page.local_translation_runtime_manager, "_validate_model_with_progress"), \
         mock.patch("modules.translation.processor.TranslationFactory.create_engine", return_value=object()):
        for _path in SAMPLE_JAPAN_PAGES:
            Translator(main_page, "Japanese", "Korean")

    assert wait_for_probe.call_count == 1
    assert sum(1 for event in main_page.runtime_events if event.get("readiness_cache_hit")) == 2


def test_sample_japan_three_ocr_initializations_probe_ocr_once() -> None:
    for path in SAMPLE_JAPAN_PAGES:
        assert path.is_file()

    main_page = _DummyMainPage()

    with mock.patch.object(main_page.local_ocr_runtime_manager, "validate_engine", return_value=None), \
         mock.patch.object(main_page.local_ocr_runtime_manager, "_probe_health_state", return_value="healthy") as probe_health, \
         mock.patch("modules.ocr.processor.OCRFactory.create_engine", return_value=object()):
        for _path in SAMPLE_JAPAN_PAGES:
            processor = OCRProcessor()
            processor.initialize(main_page, "Japanese")

    assert probe_health.call_count == 1
    assert sum(1 for event in main_page.runtime_events if event.get("readiness_cache_hit")) == 2

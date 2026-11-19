from unittest import TestCase, mock

from modules.ocr import factory
from modules.ocr.factory import OCRFactory


class _Settings:
    @staticmethod
    def get_credentials(_service):
        return {}

    @staticmethod
    def is_gpu_enabled():
        return False


class OCRFactoryRoutingTests(TestCase):
    def test_mit48_modes_create_the_matching_engine_variant(self):
        settings = _Settings()
        for engine_name, variant in (
            ("MIT 48px AR", "ar"),
            ("MIT 48px CTC", "ctc"),
        ):
            with self.subTest(engine=engine_name):
                engine = object()
                with (
                    mock.patch.dict(OCRFactory._engines, {}, clear=True),
                    mock.patch.object(factory, "resolve_device", return_value="cpu"),
                    mock.patch.object(
                        OCRFactory,
                        "_create_mit48_ocr",
                        return_value=engine,
                    ) as create_engine,
                ):
                    result = OCRFactory.create_engine(
                        settings,
                        "English",
                        engine_name,
                    )

                self.assertIs(result, engine)
                create_engine.assert_called_once_with(
                    settings,
                    variant,
                    "English",
                )

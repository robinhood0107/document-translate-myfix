"""라우터는 명시적 모델명을 요구한다. 세 이름이 어긋나면 추론이 400으로 죽는다.

`--models-max 1 --no-models-autoload` 라우터는 모델이 이미 적재돼 있어도 요청의
`model` 필드로만 대상을 고른다. 따라서

1. OCR 엔진이 추론 요청에 넣는 모델명
2. 라우터 pair가 load/unload에 쓰는 alias
3. preset 파일의 섹션명

이 셋이 완전히 같아야 한다. 컨테이너 기동만 확인하면 이 불일치를 놓치므로, 세
이름을 계약으로 고정한다.
"""

from __future__ import annotations

import os
import re
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from modules.utils.local_llama_router.contracts import (  # noqa: E402
    DEFAULT_GEMMA_ROUTER_MODEL,
    router_pair_for_engine_key,
)


def _client_model_name(engine_key: str) -> str:
    """각 OCR 엔진이 실제 추론 요청의 ``model`` 필드에 넣는 값."""

    if engine_key == "PaddleOCR VL":
        from modules.ocr.paddle_crop.transport import PADDLE_DIRECT_MODEL_ALIAS

        return PADDLE_DIRECT_MODEL_ALIAS
    if engine_key == "PaddleOCR VL Spotting":
        from modules.ocr.paddle_spotting.engine import PaddleOCRVLSpottingEngine

        return PaddleOCRVLSpottingEngine.MODEL_IDENTITY
    if engine_key == "HunyuanOCR":
        from modules.ocr.hunyuan_llamacpp_runtime_contract import (
            HUNYUAN_OCR_MODEL_ALIAS,
        )

        return HUNYUAN_OCR_MODEL_ALIAS
    if engine_key == "MangaLMM":
        from modules.ocr.mangalmm_full_page.runtime import MANGALMM_MODEL_NAME

        return MANGALMM_MODEL_NAME
    raise AssertionError(f"라우터 편입 엔진이 아닙니다: {engine_key}")


ROUTER_ENGINE_KEYS = (
    "PaddleOCR VL",
    "PaddleOCR VL Spotting",
    "HunyuanOCR",
    "MangaLMM",
)


class RouterModelIdentityAlignmentTests(unittest.TestCase):
    def test_client_model_name_matches_the_pair_alias(self) -> None:
        for engine_key in ROUTER_ENGINE_KEYS:
            with self.subTest(engine=engine_key):
                pair = router_pair_for_engine_key(engine_key)
                self.assertIsNotNone(pair)
                self.assertEqual(_client_model_name(engine_key), pair.ocr_alias)

    def test_pair_alias_has_a_preset_section(self) -> None:
        for engine_key in ROUTER_ENGINE_KEYS:
            with self.subTest(engine=engine_key):
                pair = router_pair_for_engine_key(engine_key)
                text = Path(pair.preset_file).read_text(encoding="utf-8")
                self.assertIn(f"[{pair.ocr_alias}]", text)

    def test_every_preset_declares_the_default_gemma_model(self) -> None:
        for engine_key in ROUTER_ENGINE_KEYS:
            with self.subTest(engine=engine_key):
                pair = router_pair_for_engine_key(engine_key)
                text = Path(pair.preset_file).read_text(encoding="utf-8")
                self.assertIn(f"[{DEFAULT_GEMMA_ROUTER_MODEL}]", text)

    def test_preset_sections_are_exactly_the_two_routed_models(self) -> None:
        """preset에 다른 모델이 섞이면 ``--models-max 1`` 대상이 흐려진다."""

        for engine_key in ROUTER_ENGINE_KEYS:
            with self.subTest(engine=engine_key):
                pair = router_pair_for_engine_key(engine_key)
                text = Path(pair.preset_file).read_text(encoding="utf-8")
                sections = re.findall(r"^\[(.+)\]\s*$", text, flags=re.MULTILINE)
                self.assertEqual(
                    sections,
                    ["*", pair.ocr_alias, DEFAULT_GEMMA_ROUTER_MODEL],
                )

    def test_router_spec_material_alias_comes_from_the_pair(self) -> None:
        """worker의 ``--alias``는 contract의 alias와 대조되므로 어긋나면 안 된다.

        어긋나면 컨테이너와 모델은 정상 적재되지만 GPU 귀속 증거를 찾지 못해
        기동이 실패한다. 소스에서 alias 출처를 계약으로 못박는다.
        """

        source = Path("modules/ocr/local_runtime.py").read_text(encoding="utf-8")
        spec_body = source.split("def _router_runtime_spec", 1)[1].split(
            "def validate_engine", 1
        )[0]
        alias_assignments = re.findall(r"^\s+alias=(.+),$", spec_body, flags=re.MULTILINE)
        self.assertEqual(len(alias_assignments), len(ROUTER_ENGINE_KEYS))
        for assignment in alias_assignments:
            self.assertEqual(assignment, "pair.ocr_alias")

    def test_preset_model_paths_point_at_the_mounted_volumes(self) -> None:
        for engine_key in ROUTER_ENGINE_KEYS:
            with self.subTest(engine=engine_key):
                pair = router_pair_for_engine_key(engine_key)
                text = Path(pair.preset_file).read_text(encoding="utf-8")
                ocr_section = text.split(f"[{pair.ocr_alias}]", 1)[1].split("[", 1)[0]
                gemma_section = text.split(
                    f"[{DEFAULT_GEMMA_ROUTER_MODEL}]", 1
                )[1]
                self.assertIn("model = /models/ocr/", ocr_section)
                self.assertIn(
                    f"model = /models/gemma/{DEFAULT_GEMMA_ROUTER_MODEL}",
                    gemma_section,
                )


if __name__ == "__main__":
    unittest.main()

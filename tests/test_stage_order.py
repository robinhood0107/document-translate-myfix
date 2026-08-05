"""stage-batched 단계 순서와 그 근거를 계약으로 고정한다.

인페인팅과 번역은 서로 독립이다. 둘 다 OCR 산출물(`blk_list`)만 필요하고, 렌더만
양쪽을 필요로 한다. 그래서 순서는 자유 선택이었고, 다음 이유로 번역을 앞에 둔다.

* OCR 과 Gemma 가 같은 Router 컨테이너에서 연속 스왑된다. Router v2 를 만든 목적이
  바로 이것이다. 예전 순서는 그 사이에 LaMa 를 끼워 넣어, 아무 모델도 없는 컨테이너가
  CUDA 컨텍스트로 약 278 MiB 를 붙든 채 인페인팅 sweep 전체(실측 467초)를 지났다.
* 그래서 LaMa 가 GPU 를 온전히 쓴다.
* 인페인팅과 렌더가 인접해진다. 인페인팅 결과는 `ctx.patches` 로 메모리에 남아 렌더가
  소비하므로, 인접하면 366장 분량을 동시에 들고 있을 필요가 없다.
"""

from __future__ import annotations

import inspect
import os
import re
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pipeline.stage_batched_processor import StageBatchedProcessor  # noqa: E402


def _sweep_order() -> list[str]:
    """`batch_process` 안에서 sweep 함수가 호출되는 순서."""

    source = inspect.getsource(StageBatchedProcessor.batch_process)
    names = re.findall(
        r"self\._(detect_all|ocr_all|translate_all|inpaint_all|render_all)\(",
        source,
    )
    seen: list[str] = []
    for name in names:
        if name not in seen:
            seen.append(name)
    return seen


class StageOrderTests(unittest.TestCase):
    def test_translation_runs_before_inpainting(self) -> None:
        order = _sweep_order()
        self.assertEqual(
            order,
            ["detect_all", "ocr_all", "translate_all", "inpaint_all", "render_all"],
        )

    def test_inpainting_and_render_are_adjacent(self) -> None:
        """인접해야 patches 를 페이지당 즉시 소비할 수 있다."""

        order = _sweep_order()
        self.assertEqual(
            order.index("render_all") - order.index("inpaint_all"),
            1,
        )

    def test_the_router_is_stopped_before_inpainting(self) -> None:
        """열려 있는 Router 컨테이너는 모델이 없어도 VRAM 을 붙든다."""

        source = inspect.getsource(StageBatchedProcessor.batch_process)
        gemma_release = source.index("_release_gemma_before_inpainter")
        inpaint_call = source.index("self._inpaint_all(")
        self.assertLess(gemma_release, inpaint_call)

    def test_the_ocr_handoff_keeps_the_container_for_translation(self) -> None:
        """OCR 다음이 번역이므로 컨테이너를 살려두는 것이 이제 맞다."""

        source = inspect.getsource(StageBatchedProcessor)
        self.assertIn('context="OCR-to-translate handoff"', source)
        self.assertNotIn('context="OCR-to-inpaint handoff"', source)

    def test_the_inpainter_release_is_named_for_what_follows_it(self) -> None:
        source = inspect.getsource(StageBatchedProcessor)
        self.assertIn("_release_inpainter_before_render", source)
        self.assertNotIn("_release_inpainter_before_gemma", source)


    def test_the_declared_order_matches_the_calls(self) -> None:
        """추정기는 선언된 순서를 쓴다. 실제 호출과 어긋나면 남은 시간이 틀린다."""

        from modules.utils.stage_sweep_eta import DEFAULT_STAGE_ORDER

        declared = StageBatchedProcessor.STAGE_SWEEP_ORDER
        self.assertEqual(DEFAULT_STAGE_ORDER, declared)
        # 선언 순서에서 sweep 함수 이름으로 바꿔 실제 호출 순서와 맞춘다.
        as_functions = [
            name.replace("-", "_").replace("_all", "_all")
            for name in declared
            if name != "save-and-finish"
        ]
        self.assertEqual(_sweep_order(), as_functions)

    def test_the_numeric_step_order_is_not_the_run_order(self) -> None:
        """번역(7)이 인페인팅(3)보다 먼저 돈다. 숫자 정렬을 순서로 쓰면 안 된다."""

        numeric = tuple(
            StageBatchedProcessor.STAGE_NAMES_BY_STEP[key]
            for key in sorted(StageBatchedProcessor.STAGE_NAMES_BY_STEP)
        )
        self.assertNotEqual(numeric, StageBatchedProcessor.STAGE_SWEEP_ORDER)


class StageLabelTests(unittest.TestCase):
    """숫자 step 에서 이름을 되찾는 방식이 오해를 만들었다."""

    def test_every_sweep_names_itself(self) -> None:
        source = inspect.getsource(StageBatchedProcessor)
        calls = re.findall(r"self\.emit_progress\(([^)]*)\)", source, flags=re.S)
        self.assertTrue(calls)
        for call in calls:
            with self.subTest(call=" ".join(call.split())):
                self.assertIn("stage_name=", call)

    def test_the_inpaint_sweep_is_not_called_a_setup_step(self) -> None:
        labels = StageBatchedProcessor.STAGE_LABELS
        self.assertIn("인페인팅", labels["inpaint-all"])
        self.assertNotIn("pre-inpaint-setup", set(labels.values()))

    def test_the_numeric_map_is_only_a_fallback(self) -> None:
        """레거시 표가 라벨의 근거로 쓰이면 다시 어긋난다."""

        from pipeline.batch_processor import BatchProcessor

        source = inspect.getsource(BatchProcessor.emit_progress)
        # 넘겨받은 이름이 우선이고, 표는 그것이 없을 때만 쓰인다.
        self.assertIn("stage_name or", source)
        self.assertIn("STAGE_NAMES_BY_STEP.get", source)


if __name__ == "__main__":
    unittest.main()

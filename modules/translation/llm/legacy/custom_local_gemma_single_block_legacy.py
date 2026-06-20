"""
Retired Gemma contextual single-block translation reference.

한국어:
이 파일은 이전 Gemma contextual single-block 번역 경로를 기록으로만 보존합니다.
현재 제품 런타임에서는 사용하면 안 됩니다. Gemma 기본 번역은 fast-multi 병합
입력을 사용하며, 실패 시에도 이 legacy 경로로 돌아가지 않습니다.

English:
This file preserves the retired Gemma contextual single-block translation path
as documentation only. Product runtime must not call it. The active Gemma
translator uses fast-multi merged input by default and never falls back to this
legacy path.
"""

from __future__ import annotations

from typing import NoReturn


def disabled_legacy_contextual_single_block_translation(*_args, **_kwargs) -> NoReturn:
    """
    한국어:
    이전 구현은 chunk 전체를 merged_context로 넣고 block마다 target_block 하나씩
    Gemma에 요청했습니다. 이 방식은 요청 수가 많아 느렸기 때문에 보존 전용으로
    분리됐습니다. 이 함수는 실수로 호출되는 즉시 실패해야 합니다.

    English:
    The retired implementation sent the whole chunk as merged_context but asked
    Gemma for one target_block per request. It was slower because it multiplied
    request count, so it is archived here only. Accidental calls must fail
    immediately.
    """
    raise RuntimeError(
        "Legacy Gemma contextual single-block translation is disabled. "
        "Use the active fast-multi Gemma path instead."
    )


# 한국어:
# 아래는 폐기 전 핵심 흐름의 읽기 전용 요약입니다. 실행 가능한 코드로 복원하지 마세요.
#
# English:
# The block below is a read-only summary of the retired flow. Do not restore it
# as executable production code.
#
# def _translate_contextual_single_blocks(self, blk_list, extra_context, *, prompt_profile):
#     system_prompt = self._build_system_prompt(extra_context, prompt_profile=prompt_profile)
#     updated_count = 0
#     for index, blk in enumerate(blk_list):
#         if self._should_preserve_existing_translation(blk):
#             self._current_benchmark_stats["gemma_preserved_existing_translation_count"] += 1
#             updated_count += 1
#             continue
#         user_prompt = self._build_contextual_single_block_user_prompt(blk_list, index)
#         response_data = self._request_translation(
#             system_prompt,
#             user_prompt,
#             expected_keys=["translation"],
#         )
#         translation_dict = self._extract_translation_dict(
#             response_data,
#             expected_keys=["translation"],
#             block_count=1,
#             prompt_profile=prompt_profile,
#         )
#         self._store_exact_prompt_cache(system_prompt, user_prompt, ["translation"], response_data)
#         self._apply_translation_value(blk, index, translation_dict.get("translation"))
#         updated_count += 1
#     return updated_count

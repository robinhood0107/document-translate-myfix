# 인페인팅 디버그 내보내기

Automatic Mode 디버그 내보내기는 원인을 아래 3가지로 나눠서 볼 수 있게 합니다.

- detector 문제: `detector_overlays`에서 박스가 너무 짧거나 빠져 있음
- mask 문제: 박스는 맞지만 `raw_masks`, `mask_overlays`, `cleanup_mask_delta`가 글자 획을 끝까지 못 덮음
- inpainter 문제: 마스크는 맞지만 `cleaned_images`에 글자 잔사나 얼룩이 남음

`Translate All`과 `One-Page Auto`는 같은 export 설정을 공유하고, 같은 `comic_translate_<timestamp>` 트리에 저장됩니다.

대량 검수는 `scripts/export_inpaint_debug.py`를 사용하면 되고, `Sample/japan`, `Sample/China`를 현재 detector/mask/inpaint/cleanup 흐름으로 돌려 `banchmark_result_log/inpaint_debug/...` 아래에 결과를 남깁니다.

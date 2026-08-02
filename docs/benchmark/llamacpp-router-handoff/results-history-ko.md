# Single llama.cpp Router handoff results history

| Date | Stage | Status | Notes |
|---|---|---|---|
| 2026-08-02 | Router preflight | PASS | pair별 pinned image, read-only model mount, `models-max=1`, no-autoload, unloaded-model rejection을 확인했다. HunyuanOCR은 canonical volume과 legacy baseline bind 모두 실제 SHA-256 identity를 통과했다. |
| 2026-08-02 | Paddle crop + Gemma ABBA | REJECT | 두 방향 속도는 빨랐으나 text-first 검수에서 삭제·검열 회귀가 있어 승격하지 않는다. |
| 2026-08-02 | Paddle Spotting + Gemma ABBA | REJECT | 두 방향 속도는 빨랐으나 text-first 검수에서 고유명사 훼손이 있어 승격하지 않는다. |
| 2026-08-02 | HunyuanOCR + Gemma 초기 관측 | CONTRACT-INCOMPLETE | 실제 model-byte와 upstream hash 증거가 부족해 PASS나 제품 근거로 사용하지 않는다. |
| 2026-08-02 | HunyuanOCR + Gemma 보강 ABBA | REJECT | 완결된 A→B→B→A에서 B가 두 방향 모두 느렸다. GPU 반환·orphan gate는 통과했지만 median 개선은 -66.495%였다. |
| 2026-08-02 | MangaLMM + Gemma ABBA | NOT ELIGIBLE | 제품의 현재 full-page 설정을 보존해도 baseline OCR quality gate가 translation 전에 실패했다. 사전조건 실패 시도는 성능 결과에 포함하지 않는다. |

각 pair는 독립적으로 PASS, REJECT 또는 NOT ELIGIBLE이다. 현재 PASS pair가 없으므로 product PR을 만들지 않으며, 사용자 승인 전 다음 성능 단계도 시작하지 않는다.

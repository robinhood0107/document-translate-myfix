# Single llama.cpp Router 빠른 Lab — 최종 보고

- protocol: `llamacpp-router-handoff-v1`
- 실행: eligible pair마다 `A → B → B → A` 한 묶음만 사용했다. 추가 반복, R3 동시 상주, Turbo4 재시험은 하지 않았다.
- 해석: 표본은 한 묶음이므로 기술 통계만 제시하며 유의성을 주장하지 않는다.
- 상태: **승격 가능한 pair 없음**. 제품 반영과 다음 성능 작업은 사용자 승인 전까지 중지한다.

| Pair | 판정 | E2E median A / B (초) | paired delta A→B / B→A (초) | 최종 사유 |
|---|---|---:|---:|---|
| Paddle crop + Gemma | REJECT | 59.992 / 47.852 | +11.015 / +13.265 | text-first 검수에서 삭제·검열 회귀 1건 |
| Paddle Spotting + Gemma | REJECT | 54.235 / 50.469 | +5.766 / +1.766 | text-first 검수에서 고유명사 훼손 1건 |
| HunyuanOCR + Gemma | REJECT | 57.859 / 96.297 | -37.672 / -39.204 | 두 방향 감속; upstream snapshot은 품질 PASS 근거가 아님 |
| MangaLMM + Gemma | NOT ELIGIBLE | - | - | 두 baseline 모두 OCR quality gate에서 번역 전 실패 |

Paddle crop/Spotting은 router가 더 빨랐어도, 기존 Paddle 요청 형식·`OCR:` prompt·alias·image 처리·재시도·worker/scheduling을 보존한 상태의 품질 gate 실패이므로 승격하지 않는다.

## HunyuanOCR 완결 ABBA (승격 근거 아님)

이전 Hunyuan 관측은 실제 model-byte 및 upstream-artifact 증거가 부족해 승격 근거로 사용하지 않았다. 보강된 harness로 한 완결 ABBA를 새로 실행했고, 네 arm 모두 파이프라인·render를 완료했으며 OOM과 router/container orphan은 없었다. 그러나 router arm은 96.797초와 95.797초, baseline arm은 59.125초와 56.593초여서 두 paired 방향 모두 감속했다. 속도만으로도 terminal `REJECT`다.

- router first request: OCR 0.300–0.336초 / Gemma 1.901–2.191초.
- GPU 반환: OCR 0.140–0.142초, Gemma 0.124–0.156초로 두 router arm 모두 Arbiter gate에서 관측됐다.
- Arbiter command queue median: 두 arm 모두 0.014ms, E2E의 0.001% 미만이다.
- model identity, loaded model 최대 1, unloaded-model implicit autoload 거부는 통과했다.
- detection·OCR page-profile·inpaint decoded-pixel/diagnostic 값은 arm 간 동일했다. 다만 이 run의 OCR raw-result map은 실행별 block UUID가 달라 pre-translation snapshot exact 비교가 성립하지 않았다. harness는 이후 block 순서 기반으로 수정했지만, 해당 candidate는 이미 명확히 감속했으므로 GPU 재실행이나 품질 PASS 주장을 하지 않는다.

## 자원 관측

RAM, shared GPU memory, WSL swap은 이 protocol에서 단독 탈락 사유가 아닌 관측값이다. 아래 Hunyuan은 이번 유효 ABBA의 A/B 범위이며, WSL swap은 이 Windows runner에서 사용할 수 없었다.

| Pair | arm | peak VRAM (MiB) | peak shared GPU (MiB) | 최소 Windows available RAM (GiB) | WSL swap delta |
|---|---|---:|---:|---:|---:|
| Paddle crop | B | 11,731–11,764 | 390.051–392.348 | 1.100–1.388 | +12.520–+14.184 MiB |
| Paddle Spotting | B | 11,819–11,821 | 387.645–388.297 | 1.330–1.330 | +2.950–+8.527 MiB |
| HunyuanOCR | A | 11,797–11,803 | 387.648–425.285 | 1.641–2.599 | unavailable |
| HunyuanOCR | B | 11,760–11,771 | 365.164–389.039 | 1.241–1.632 | unavailable |

MangaLMM은 live Windows full-page 설정(`4096`, worker `1`, timeout `60`, safe resize, `2,116,800` pixels, long side `1728`)을 그대로 사용했다. 이전의 사전조건 불충족 시도는 비교에 포함하지 않았고, 최종 baseline-gated 결과만 `NOT ELIGIBLE`로 기록했다.

Raw request/response, text context, source image, command, model identity, resource sample은 managed private archive에만 보관한다. 이 lab은 product Compose/runtime을 변경하지 않았다.

# Gemma 프로필 이력

이 문서는 Gemma translation-only 경로에서 실제로 사용한 주요 설정 이력을 간단히 남기는 용도입니다.

## 현재 활성 설정

| 항목 | 값 |
| --- | --- |
| temperature | `0.6` |
| top_k | `64` |
| top_p | `0.95` |
| min_p | `0.0` |
| prompt profile | `gemma4_balanced` |
| chunk size | `4` |
| max completion tokens | `512` |
| timeout sec | `180` |
| model path | `/models/gemma-4-26b-a4b-it-heretic.q3_k_m.gguf` |
| image tag | `ghcr.io/ggml-org/llama.cpp:server-cuda` |
| ctx-size | `4096` |
| threads | `12` |
| n_gpu_layers | `23` |
| `--swa-full` | `enabled` |
| reasoning | `off` |
| OCR front device | `cpu` |

## 이력

### 초기 local Gemma compose 도입

- 모델: `gemma-4-26B-A4B-it-UD-Q2_K_XL.gguf`
- `n_gpu_layers=8`
- reasoning 관련 플래그 없음

### Gemma 전용 번역 경로 도입

- `Custom Local Server(Gemma)` 분기
- JSON 강제 응답
- image input 비활성
- Gemma 4 전용 `gemma4_balanced` prompt profile 추가
- JSON/empty-content 실패 시 strict prompt 1회 재시도 후 chunk split

### translation-only benchmark 체계 정리

- creative sampler 제거
- `top_k=64`, `top_p=0.95`, `min_p=0.0` 고정
- `temperature`, `n_gpu_layers`만 탐색

### 현재 채택

- `temperature=0.6`
- `n_gpu_layers=23`
- `threads=12`
- `ctx=4096`

## 관련 문서

- [translation-optimization-ko.md](./translation-optimization-ko.md)
- [optimization-journey-ko.md](../benchmark/optimization-journey-ko.md)
- [report-ko.md](../banchmark_report/report-ko.md)

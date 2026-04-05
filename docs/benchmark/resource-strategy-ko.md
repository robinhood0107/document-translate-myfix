# 자동번역 GPU 자원 전략

## 현재 전략

현재 자동번역 파이프라인은 같은 페이지에서 OCR과 Gemma가 동시에 추론하지 않습니다.

순서는 대략 아래입니다.

1. detect
2. OCR
3. inpaint
4. translate
5. render/save

즉 병목은 “동시 추론”보다 “동시 상주 VRAM”에 더 가깝습니다.

## 현재 권장 자원 배분

- `paddleocr-vllm`: GPU
- `paddleocr-server` front: CPU
- `gemma-local-server`: GPU 레이어 최대 활용

이 전략의 이유는 아래와 같습니다.

- OCR 본 추론은 `paddleocr-vllm`이 담당
- front service는 CPU로 내려도 품질 손실 없이 전체 latency가 줄 수 있음
- 확보한 여유를 Gemma `n_gpu_layers` 확대에 쓰는 편이 translate 병목 해소에 더 유리함

## benchmark 코드와 business 코드의 경계

현재 benchmark는 완전히 분리된 구조는 아닙니다.

core/business code에 남아 있는 것:

- stage event emission
- retry/truncated/quality 통계 surface
- metrics logging hook

별도 benchmark 레이어에 있어야 하는 것:

- preset
- 실험 순서
- winner 판단
- 차트 생성
- 문서 생성

## 유지보수 의견

이 구조는 괜찮습니다. 다만 앞으로는 business code에 benchmark-specific 정책을 더 넣지 않는 것이 중요합니다.

가장 안전한 방향은:

- business code에는 “얇은 계측 훅”만 유지
- 실험과 문서화는 `scripts/` + `benchmarks/` + `docs/benchmark/`에서 해결

이렇게 해야 나중에 `develop` 비즈니스 코드가 바뀌어도 benchmark 실험 레이어가 덜 깨집니다.

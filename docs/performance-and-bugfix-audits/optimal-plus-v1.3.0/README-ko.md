# Optimal++ v1.3.0 최종 감사

이 묶음은 제품 후보를 빠른 수치만으로 승격하지 않고, 구조·의미·그림 보존과
반복 측정을 함께 통과시킨 최종 상태를 기록한다. 공개 문서는 집계 수치와 결정만
담고, 원문·이미지·검수표·raw 응답은 비추적 검증 보관소에 남긴다.

## 현재 결론

- 관리형 Gemma·Paddle crop·Paddle Spotting·MangaLMM은 모두 llama.cpp를 사용한다.
- 기본 OCR은 Paddle detector + crop OCR이다. 두 full-page 전략은 결과와 한계를
  보존한 Experimental 선택지이며, 자동 fallback이나 동시 상주는 하지 않는다.
- 번역 기본값은 IQ4_NL, contextual-single, chunk 6, F16 KV, no-spec이다.
- 정확 일치 캐시와 stage checkpoint는 재실행을 크게 줄이지만, 새 입력의 cold OCR을
  대체하지 않는다.
- 그림을 안전하게 복원할 수 없는 반투명·free-text 영역은 원본 보존과 review를
  우선한다.

## 문서 순서

| 문서 | 내용 |
|---|---|
| [00. 진실명세](00-truth-specification-ko.md) | 현재 확정값, 불변 조건, 금지선 |
| [01. 전체 실험 등록부](01-complete-experiment-register-ko.md) | 지금까지의 채택·퇴역·보류 실험 |
| [02. OCR 품질·라우팅](02-ocr-quality-and-routing-decision-ko.md) | 세 OCR 전략, 직접 llama.cpp crop, COO |
| [03. Gemma 번역·모델](03-gemma-translation-and-model-decision-ko.md) | 모델·요청·KV·batch·문맥 결론 |
| [04. 캐시·cold path](04-cache-checkpoint-and-cold-path-decision-ko.md) | result/OCR/checkpoint/page cache와 재실행 이득 |
| [05. 인페인트 품질](05-inpaint-quality-and-safe-preservation-ko.md) | 그림 보존 우선 품질 게이트 |
| [06. 런타임·자산 퇴역](06-runtime-assets-and-llamacpp-retirement-ko.md) | llama.cpp 전용, volume, WSL, 정리 원칙 |
| [07. 릴리스 검증](07-release-verification-and-branch-closure-ko.md) | CUDA·패키지·브랜치 종료 증적 |
| [08. 익명화 근거 색인](08-sanitized-evidence-index-ko.md) | 공개·lab·비추적 증적의 경계 |

## 해석 원칙

속도 개선율에는 최소 문턱을 두지 않는다. 품질이 같고 반복 측정으로 실제 이득이
확인되면 작은 개선도 채택한다. 반대로 구조 오류, candidate-only 의미 회귀, 또는
그림 훼손이 하나라도 확인되면 속도와 무관하게 제품 기본값으로 승격하지 않는다.

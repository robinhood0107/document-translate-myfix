# Optimal++ v1.3.0 진실 명세

이 문서는 v1.3.0까지의 OCR·번역·캐시·런타임 최적화에서 **현재 제품에
확정된 사실**과, 속도가 빨라도 제품 승격을 막은 품질 금지선을 한 곳에
고정한다. 숫자의 원시 근거와 사람 검수 자료는 Git 밖의
`<validation-log-root>`에만 보관한다.

## 현재 확정 상태

| 영역 | 제품 결정 | 근거 |
|---|---|---|
| 관리형 추론 | llama.cpp 전용 | 관리형 vLLM 실행 경로는 퇴역했고, 비관리형 사용자가 직접 지정한 endpoint만 별도 호환 경로로 남긴다. |
| 기본 OCR | detector + PaddleOCR-VL crop OCR | 세 OCR 전략 중 의미 텍스트 recall과 안정적인 파괴적 편집 안전성이 가장 높았다. |
| 실험 OCR | Paddle full-page Spotting, MangaLMM full-page | 좌표와 일부 인식은 유효하지만, 현재 기본 경로보다 merge/split과 의미 누락이 많아 기본값으로 올리지 않는다. |
| 번역 | IQ4_NL + contextual-single | `chunk=6`, no-spec, F16 KV, batch/ubatch `2048/512`, NGL 23을 유지한다. |
| 단계 순서 | detect → OCR → inpaint → translation → render | GPU 모델은 동시에 상주시키지 않고, 단계가 끝나면 안전하게 반환한 뒤 다음 단계로 넘긴다. |
| 재실행 캐시 | exact OCR cache, 번역 결과 cache/Exact TM, 프로젝트 checkpoint | 동일 입력의 재처리를 줄이되, 새 이미지의 첫 OCR·번역 결과를 추정하지 않는다. |
| 인페인트 | 검증된 일반 불투명 영역 경로 유지 | 반투명·구조 배경에서 안전한 복원이 보장되지 않으면 원본 보존 및 검토 대상으로 남긴다. |

## 불변 품질 조건

다음 중 하나라도 후보에서만 발생하면, 속도 수치와 관계없이 후보를 제품
기본값으로 승격하지 않는다.

- OCR/번역 결과의 누락, 빈 값, 잘림, 구조 오류, 순서 오류, 해결되지 않은 fallback
- 화자·관계·부정·행동·주체/대상·숫자·고유명사·명시적 의미의 회귀
- SFX·UI·장식 문자에 대한 번역/인페인트가 원본 그림을 훼손한 사례
- detector 권한을 벗어난 불안전한 merge/split 또는 mask 밖 픽셀 변경
- GPU OOM, 새 swap 증가, 비정상 shared GPU memory 증가, 단계 간 동시 GPU 상주

공백·줄바꿈·문장부호만의 차이는 별도 표기하지만, 의미 회귀로 계산하지
않는다. 품질이 같은 후보끼리는 반복 측정에서 실제 이득이 확인된 작은
속도 개선도 채택한다.

## 핵심 측정 결과

| 항목 | 결과 | 현재 의미 |
|---|---:|---|
| 직접 Paddle crop transport | 일본어 22페이지 311/311 정규화 OCR 동일, 직접 요청 23.970초 | 기존 중계 계층을 제거한 관리형 직접 llama.cpp 경로를 채택했다. |
| source-aware OCR prewarm | 평균/중앙 3.482% 단축, 신뢰구간 하한 2.961% | 현재 폴더에 miss가 있을 때만 prewarm한다. |
| 번역 결과 cache all-hit | 60.301초 → 1.978초, 96.720% 단축 | Gemma 시작과 HTTP 요청을 모두 생략한다. |
| global OCR cache all-hit | 19.763초 → 0.110초, 99.445% 단축 | OCR runtime과 HTTP 요청을 모두 생략한다. |
| 프로젝트 checkpoint all-hit | 139.961초 → 8.236초, 94.115% 단축 | 동일 프로젝트 재실행의 detector·OCR·번역·인페인트를 생략할 수 있다. |
| GGUF 자연 page cache | healthy 66.811초 → 즉시 재시작 16.865초, 74.757% 단축 | 모델을 VRAM에 상주시킨 것이 아니라 OS page cache 재사용 효과다. |

새로운 입력의 cold 실행에는 all-hit 수치가 적용되지 않는다. 이 경우에는
직접 crop transport, 현재 폴더 기준 prewarm, 단계별 모델 handoff처럼 결과를
바꾸지 않는 최적화만 적용한다.

## 문서 읽는 순서

1. [완전 실험 등록부](01-complete-experiment-register-ko.md): 모든 후보의 채택·보류·탈락 사유
2. [OCR 품질 및 라우팅 판정](02-ocr-quality-and-routing-decision-ko.md): 세 OCR 전략의 사람 기준 비교
3. [Gemma 번역 및 모델 판정](03-gemma-translation-and-model-decision-ko.md): 번역 모델·runtime 선택 근거
4. [캐시·checkpoint·cold 경로 판정](04-cache-checkpoint-and-cold-path-decision-ko.md): 새 입력과 재실행을 구분한 성능 계약
5. [인페인트 품질 및 안전 보존 판정](05-inpaint-quality-and-safe-preservation-ko.md): 파괴적 편집 금지선
6. [런타임 자산 및 llama.cpp 전용 전환](06-runtime-assets-and-llamacpp-retirement-ko.md): volume·WSL·퇴역 절차
7. [릴리스 검증 및 브랜치 마감](07-release-verification-and-branch-closure-ko.md): v1.3.0 검증 기록
8. [정제된 근거 색인](08-sanitized-evidence-index-ko.md): 공개 문서와 비공개 증거의 경계

## 보관 원칙

이 문서 묶음은 Git에 올릴 수 있는 정제된 결론만 담는다. 원본 페이지,
확대 crop, OCR/번역 원문, mask, inpaint/render 비교물, blind 검수표, 하드웨어
원시 telemetry, 모델 파일 식별 정보는 저장소 루트의 ignore된 검증 보관소와
`<validation-log-root>`에만 둔다. 공개 문서의 수치는 그 비공개 증거를
대체하지 않는다.

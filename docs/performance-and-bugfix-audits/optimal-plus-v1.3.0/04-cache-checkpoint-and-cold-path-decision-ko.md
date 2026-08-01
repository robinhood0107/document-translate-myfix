# 캐시·checkpoint·cold 경로 판정

캐시는 새 이미지의 첫 추론을 추측해서 빠르게 만드는 기능이 아니다. 정확히 같은
입력과 runtime identity에서만 이미 계산한 결과를 재사용한다. 따라서 v1.3.0은
**cold 실행 성능**과 **재실행 성능**을 같은 숫자로 섞지 않는다.

## 캐시 계층과 책임

| 계층 | 재사용 단위 | 저장 내용 | 전체 hit일 때 생략되는 작업 |
|---|---|---|---|
| 번역 result cache | 전체 문맥·설정이 같은 block 요청 | 사용자 사전 적용 전 sanitized 번역 | Gemma 시작·HTTP·decode |
| Exact TM | 사용자가 승인한 정확 일치 원문→번역 | 승인된 번역만 | 해당 block의 Gemma 요청 |
| global OCR cache | crop pixel과 실제 request image가 같은 OCR 요청 | OCR·진단·runtime identity | Paddle runtime·OCR HTTP |
| 프로젝트 checkpoint | source·stage fingerprint가 같은 프로젝트 단계 | detection/OCR 상태, mask, inpaint artifact, render 유효성 | 해당 stage와 downstream 외의 inference |

각 계층은 hit와 miss에서 사용자 사전이나 최종 처리 규칙을 정확히 한 번 적용한다.
OCR 결과 key에는 사전 fingerprint를 넣지 않으며, 사전이 바뀌면 재인식 대신
적용 결과와 downstream만 갱신한다. 번역 result cache와 Exact TM도 서로 다른
목적과 신뢰 경계를 가진다.

## 측정된 재실행 효과

| 시나리오 | 기준 → cache 경로 | 단축 또는 순이득 | 추가 확인 |
|---|---:|---:|---|
| 번역 54 block all-hit | 60.301초 → 1.978초 | 96.720% 단축 | Gemma 시작 0, HTTP 0 |
| global OCR all-hit | 19.763초 → 0.110초 | 99.445% 단축 | OCR runtime 0, HTTP 0 |
| 프로젝트 all-hit | 139.961초 → 8.236초 | 94.115% 단축 | detector·OCR·번역·inpaint inference 0 |
| cold 첫 실행 + all-hit 재실행 | cache 없는 동일 작업 두 번 대비 | 44.261% 순이득 | 전체 결과 동일 |
| cold 첫 실행 + 한 페이지 수정 재실행 | cache 없는 동일 작업 두 번 대비 | 0.931% 순이득 | 변경 page의 변경 stage와 downstream만 재계산 |

all-hit 수치는 새 페이지를 처음 처리할 때 적용되지 않는다. 새 입력에서는 direct
llama.cpp crop transport, source-aware prewarm, 단계별 runtime handoff가 실제
속도 후보다. 반대로 같은 프로젝트를 다시 열거나 일부만 수정하는 작업에는
checkpoint가 가장 큰 이득을 만든다.

## global OCR cache 계약

global OCR cache는 관리형 PaddleOCR-VL 경로에만 사용한다. key에는 다음 영향을
포함한다.

- raw crop pixel 및 실제 요청 image hash, shape·dtype
- text class, bubble/text-free guard 입력, source language, token·표현 옵션
- crop·encoder·parser·sanitizer·guard schema version
- Paddle model, image digest, command·llama.cpp·pipeline config identity

custom/unmanaged endpoint에는 검증 가능한 runtime identity가 없으므로 영구 OCR
cache를 적용하지 않고 정상 OCR을 수행한다. DB lock, 손상, schema mismatch에는
DB를 자동 삭제·덮어쓰기 하지 않으며, 그 실행에서 cache만 끄고 OCR을 계속하는
fail-open을 사용한다.

## 프로젝트 checkpoint 계약

프로젝트 파일 옆 sidecar에는 작은 참조와 manifest만 두고, 큰 mask·inpaint·render
artifact는 content-addressed immutable object로 보관한다. stage fingerprint에는
다음과 같은 입력 영향값을 넣는다.

- decoded source, detector/runtime/preprocess/sort, OCR 및 semantic routing
- ordered block, raw/final mask, brush edit, inpaint model/backend/precision/size
- translation identity, font file hash, style·layout·sanitizer·export format

stage 하나가 바뀌면 그 stage와 downstream만 무효화한다. sidecar가 없거나
손상되어도 프로젝트를 열 수 있어야 하며, 해당 stage를 정상 재계산한다. 강제
재계산, cache folder 열기, 미사용 cache 정리, JSONL export처럼 사용자가
통제하는 관리 기능을 제공한다.

검증된 one-time migration은 project checkpoint를 한 번 활성화하고, 이후 사용자가
설정을 바꾸면 다시 덮어쓰지 않는다. cache를 끄더라도 이미 저장된 sidecar는
자동으로 삭제하지 않는다.

디버그 산출물은 재사용 cache와 분리한다. 캐시를 꺼도 사용자가 디버그 토글을
켠 경우에만 별도 debug run을 보존하며, raw response와 하드웨어 telemetry는
민감 자료이므로 기본 off다. 디버그 자료는 cache hit 판정에 관여하지 않는다.

## cold 실행 보호선

cache miss의 단일 실행 overhead에 임의의 3% 상한을 두지 않는다. 대신 아래
실사용 시나리오의 누적 시간이 반복 측정에서 실제로 이득인지 확인한다.

1. 최초 실행 후 동일 프로젝트 재실행
2. 최초 실행 후 한 페이지 수정 재실행
3. all-hit 재실행
4. hit/miss 혼합 폴더

품질과 결과가 완전히 같고, 이 시나리오의 누적 총시간이 조금이라도 확실히
빠르면 cache를 채택한다. 반대로 새 입력 cold 경로가 느려지거나 swap·GPU
안정성 문제가 생기면 해당 scheduling 최적화만 되돌린다.

OCR 전략별 cold 경로와 prewarm 수치는
[OCR 품질 및 라우팅 판정](02-ocr-quality-and-routing-decision-ko.md), 모델 load와
OS page cache는 [Gemma 번역 및 모델 판정](03-gemma-translation-and-model-decision-ko.md),
전체 후보 이력은 [완전 실험 등록부](01-complete-experiment-register-ko.md)에
정리되어 있다.

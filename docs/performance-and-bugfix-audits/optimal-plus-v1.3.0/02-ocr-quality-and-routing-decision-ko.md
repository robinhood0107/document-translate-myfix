# OCR 품질 및 라우팅 판정

v1.3.0은 세 OCR 전략을 같은 detector block 기준으로 비교했다. 이 문서의
정확도는 한 OCR 출력을 다른 OCR 출력에 맞춘 비율이 아니라, 후보 결과를 보기
전에 원본을 직접 판독해 고정한 사람 기준 manifest를 사용한다. 공백·줄바꿈·
문장부호만의 차이는 분리하고, 의미가 맞는지와 글자 단위 전사가 맞는지를 따로
계산했다.

## 세 전략의 현재 비교

| 전략 | 구조 성공 | 페이지 완전 성공 | 전사 exact | 문자 정확도 | 의미 텍스트 recall | merge/split | 파괴적 편집 | 요청 시간 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Paddle detector + crop OCR | 22/22 | 9.09% | 59.31% | 76.10% | 177/187 (94.65%) | 2 | 0 | 196.589초 |
| Paddle full-page Spotting | 22/22 | 4.55% | 44.16% | 62.59% | 156/187 (83.42%) | 10 | 0 | 41.868초* |
| MangaLMM full-page | 22/22 | 0% | 39.12% | 57.69% | 151/187 (80.75%) | 30 | 0 | 804.550초 |

`*` Spotting의 직접 요청 시간은 startup과 후속 전체 파이프라인 비용을 포함하지
않으므로, 기본 crop 경로의 전체 시간과 직접 우열을 뜻하지 않는다.

현재 기준에서 기본값은 **Paddle detector + crop OCR**이다. 의미 텍스트
recall이 가장 높고, line/block 단위 불일치가 적으며, 기존 mask 권한과 안전하게
결합된다. 두 full-page 전략은 삭제하지 않는다. 사용자가 선택할 수 있는
Experimental 경로로 남기고, 사람 검수와 추가 개선의 비교 대상으로 유지한다.

## 공통 평가 계약

각 전략은 다음 공통 결과 계약으로 비교한다.

- 원본 크기와 실제 추론 입력 크기, 비율 유지 역변환 좌표
- detector geometry와 OCR region의 출처, canonical block 및 compound 관계
- semantic role과 처리 동작: `translate_inpaint`, `preserve`, `review`
- parser·length·retry·coverage·merge/split 진단
- final mask와 인페인트의 그림 훼손 여부

파괴적 편집의 좌표 권한은 계속 detector와 기존 mask가 가진다. full-page OCR이
detector에 없는 region을 찾아도, 안전한 의미 역할·mask·충돌 없는 매칭이
확인되기 전에는 자동 편집하지 않는다. 여러 OCR line이 하나의 detector block
안에 완전히 포함되고 읽기 순서가 명확할 때만 compound로 번역을 한 번 수행한다.
하나의 region이 여러 block을 덮거나 관계가 모호하면 텍스트를 복제하지 않고
`review`로 남긴다.

## 기본 Paddle crop 경로

관리형 crop OCR은 detector가 만든 crop을 llama.cpp의 공식 `OCR:` 계약으로
직접 전송한다. crop용 projector는 공식 pixel budget 1,003,520을 유지하며,
초과 crop만 비율을 유지해 줄인다. 기존 중계와 직접 경로의 품질·속도 검증은
다음과 같다.

| 검증 범위 | 품질 결과 | 중계 → 직접 경로 시간 | 판정 |
|---|---|---:|---|
| 일본어 6페이지, 73 block | OCR·진단·순서 동일 | startup 10.697초 → 1.490초, OCR 47.883초 → 5.804초, wall 60.216초 → 8.333초 | 채택 |
| 영어 10페이지(배경 오탐지 대조 포함) | 10/10 동일 | 22.433초 → 3.602초 | 채택 |
| 중국어 6페이지 | 41/41 동일 | 41.921초 → 7.360초 | 채택 |
| 일본어 22페이지 | 정규화 OCR 311/311 동일 | 직접 요청 23.970초 | 채택 |

이 직접 경로는 관리형 llama.cpp 정책을 만족한다. 과거 중계 endpoint를 사용자가
명시적으로 선택한 경우만 비관리형 호환 경로로 남는다.

### crop 정책의 안전선

- 기본은 native crop이며, text-first 확장과 bubble clamp는 shadow 비교에서
  기존 crop보다 품질이 좋아질 때만 채택한다.
- bbox가 무효하거나 글자가 잘렸다는 진단이 있을 때만 whole-bubble 재시도를
  한 번 허용한다.
- 같은 bubble이라는 이유만으로 서로 다른 문장을 합치지 않는다.
- crop 정책이 바뀌면 OCR global cache와 프로젝트 checkpoint identity도 함께
  바꿔 과거 결과를 재사용하지 않는다.

## full-page Spotting의 가능성과 한계

Paddle full-page Spotting은 전용 projector의 공식 pixel budget 1,605,632,
`Spotting:` prompt, `--special` 계약으로 호출한다. 반환 좌표는 0–1000 정규화
4점 좌표이며, 글자줄을 촘촘하게 감싸는 유효한 보조 geometry다. 그러나 line
region 469개 중 detector block과 바로 안전 매칭된 것은 231/311이었고, detector에
없는 추가 region도 41개였다.

이는 실패한 좌표가 아니라 **단위의 차이**다. Spotting은 한 detector block을
여러 줄로 나누고, UI·SFX·장식 문자를 더 많이 찾아낸다. 따라서 현재 제품에서는
Spotting geometry를 glyph mask 보조 신호와 Experimental OCR 결과로 유지하고,
detector block을 즉시 대체하지 않는다.

MangaLMM도 공식 full-page 입력을 사용하며, image pixel 한도 2,116,800을 넘는
경우에만 비율 유지 축소한다. crop·tile·overlap fallback, hidden Paddle fallback,
dense 페이지 용량 축소는 과거 품질 문제 때문에 사용하지 않는다. completion 4096,
반복 창 4096, penalty 1.15, 한 번의 bounded 재시도가 현재 구조 안정화 계약이다.

## SFX·UI와 COO

UI 또는 SFX를 많이 검출했다는 것만으로 전략을 탈락시키지 않는다. 대신 해당
영역이 번역·인페인트되어 그림을 훼손하면 candidate-only 실패로 계산한다.

COO는 독립 OCR이나 숨은 fallback이 아니라 `preserve/review` shadow signal로
시험했다. threshold 0.5는 11개 위험 SFX 중 5개를 보호했지만 의미 대사를
false review로 보냈고, threshold 0.65는 false positive는 피했지만 보호한 SFX가
없었다. 안전하고 유용한 threshold가 없으므로 세 OCR 경로 모두 COO를 사용하지
않는다.

## 다음 품질 개선의 기준

full-page 경로의 개선은 crop OCR 텍스트를 정답처럼 복사하지 않고, 원본 판독
manifest와 아래 지표로만 판단한다.

1. 의미 텍스트 recall과 semantic exact accuracy
2. 안전하지 않은 merge/split·coverage gap 감소
3. SFX/UI의 파괴적 오처리 0
4. parser/length 구조 실패 0
5. 품질이 같은 경우에만 cold OCR·전체 파이프라인 속도

인페인트 안전선은 [인페인트 품질 및 안전 보존 판정](05-inpaint-quality-and-safe-preservation-ko.md),
속도와 cache 동작은 [캐시·checkpoint·cold 경로 판정](04-cache-checkpoint-and-cold-path-decision-ko.md),
모든 후보 이력은 [완전 실험 등록부](01-complete-experiment-register-ko.md)에 정리되어 있다.

# MangaLMM v2 과거 실패 감사

## 결론

과거 결과는 `MangaLMM 모델 전체가 실패했다`는 증거가 아니다. 실패 원인은 서로
다른 세 설계가 섞인 데 있다.

1. 공식 full-page spotting과 다른 block crop·page tile 요청
2. 어려운 dense 페이지에서 오히려 입력 해상도와 completion 용량을 줄인 계약
3. Paddle과 Manga를 함께 GPU에 상주시킨 뒤 자동 선택하려 한 운영 구조

새 v2 경로는 공식 full-page 요청만 사용하고 두 OCR 엔진을 사용자 선택지로
분리한다. 숨은 Paddle fallback과 dual residency는 다시 도입하지 않는다.

## 커밋별 판정

| commit | 과거 접근 | 판정 |
|---|---|---|
| `cdc9254` | detector block별 crop 요청 | 공식 full-page 계약과 불일치. 재사용 금지 |
| `6f96d1b`, `673cece` | 겹치는 tile과 rescue macro | 좌표 재매핑·중복·인접 문장 merge/split 복잡도 증가. 재사용 금지 |
| `2127214` | full-page 단일 요청과 전역 block 매칭 | 방향이 맞다. 재사용 |
| `0f1c1bd` | PNG 입력, strict payload 진단, 축별 좌표 환산 | 재사용 |
| `8a6ba64` | adaptive Optimal+ profile | retry telemetry만 재사용. dense 용량 축소는 금지 |
| `e31cc4e` | accepted request의 read timeout 무제한 | 긴 요청의 증거만 보존. v2는 bounded timeout·취소 사용 |
| `d65b3b24` | 넓은 장식 glyph 문자열 삭제 | raw OCR 정리와 의미 역할 판정을 섞었다. 역할 분류에 재사용 금지 |
| `ae2a90d`, `0724b98` | Paddle+Manga dual residency와 selector | 시간·누락·merge/split 품질 게이트 실패. 재사용 금지 |
| `dade2d9` | direct one-shot, 기본 256 completion tokens | 용량이 부족한 현행 계약. 이 결과만으로 모델을 폐기하지 않음 |

`scripts/benchmark_mangalmm_v2.py audit-history`는 위 커밋의 전체 SHA와
근거 파일의 필수 코드 표식을 확인한다. 문서의 주장과 Git 이력이 달라지면
즉시 실패한다.

## 익명화한 과거 측정 근거

실제 파일명·원문·좌표는 외부 검증 manifest에만 둔다.

| neutral case | detector block | 복원 | 과거 profile | 판정 |
|---|---:|---:|---|---|
| dense-translucent-dev | 30 | 5 block | `900×1270`, 1024 tokens | 입력과 출력 용량 부족 |
| standard-dialogue-dev | 15 | 15 block | `1224×1728`, 2048 tokens | full-page 방향 유효 |
| dense-dialogue-dev | 20 | 15 block | `900×1270`, 1024 tokens | recall gap과 인접 문장 병합 |

좌표 scaling 자체는 유효했다. 반복된 실패는 region recall, 출력 잘림,
N:1·1:N 관계의 자동 병합이었다.

## 재사용 금지 목록

- block crop, page tile, overlap, rescue macro
- dense 페이지의 해상도·completion 용량 축소
- Paddle/Manga 동시 상주
- 자동 selector와 숨은 Paddle fallback
- one-shot 256-token 제품 계약
- raw 문자열 삭제를 semantic role 판정으로 사용하는 규칙
- 과거 detector snapshot을 현재 정답으로 사용하는 방식
- 개발 사례를 보고 고친 threshold를 같은 사례에서 성공으로 판정하는 방식

## v2에서 재사용하는 요소

- full-page PNG 입력
- `bbox_2d + text_content` strict parsing
- request 좌표에서 원본 좌표로의 `scale_x`·`scale_y` 환산
- detector block과 Manga region의 전역 매칭 진단
- bounded retry·parser·timeout telemetry
- Stage-Batched의 한 OCR 런타임만 상주시키는 수명주기

# 세 OCR 결과 이력과 현재 해석

이 문서는 사람 정답 벤치마크 이전의 측정값을 보존합니다. 아래 일치율은 사람 기준
정답률이 아니며 제품 승격 근거로 단독 사용하지 않습니다.

| 경로·세트 | 구조 결과 | detector block | 안전 매칭 | 시간/비고 |
|---|---:|---:|---:|---|
| MangaLMM corpus-a 22 | 22/22 | 311 | 231, 74.3% | 839.841초 |
| MangaLMM corpus-b 24 | 23/24 | 174 | 132, 75.9% | 372.565초 |
| Paddle Spotting corpus-a 22 | 22/22 | 311 | 231, 74.3% | direct 요청 66.886초, startup 제외 |

Paddle Spotting corpus-a는 469개 region을 냈고 detector 미매칭 block은 80개,
detector에 매칭되지 않은 추가 region은 41개였습니다. 기존 crop OCR을 임시 기준으로
비교한 값은 다음과 같습니다.

| 비교 | 수 |
|---|---:|
| 정규화 exact | 157/311, 50.5% |
| similarity ≥ 0.9 | 177/311, 56.9% |
| similarity ≥ 0.7 | 191/311, 61.4% |

이는 crop이 정답이라는 뜻이 아닙니다. Spotting이 맞고 crop이 틀린 사례도 있으므로
새 벤치마크에서는 Codex가 원본·확대 crop을 직접 읽어 별도 정답을 잠급니다.
과거 Paddle Spotting 원시 파일은 최종 선택 응답만 보존해 4페이지의 adaptive retry
횟수를 파일 자체에서 복원할 수 없습니다. 새 실행부터 attempt telemetry를 원시 계약에
직접 기록하며, 과거 수치를 추측해 채우지 않습니다. 과거 import에서는 attempt를 0으로
표시하고 `attempt_telemetry_complete=false`로 별도 집계합니다.

확인된 스트레스 사례:

- MangaLMM `stress-a`: 4/30 detector block만 안전 매칭
- MangaLMM `stress-b`: 0/2, N:1/coverage 문제
- Paddle Spotting `stress-a`: 23/30
- Paddle Spotting `stress-c`: 13/22
- Paddle Spotting `stress-d`: 5/10

UI 추가 검출은 자동 탈락시키지 않습니다. 최종 우선순위는 의미 텍스트 recall,
원문 정확도, merge/split 의미 손실, SFX/UI의 파괴적 편집, parser/length 실패,
마지막으로 속도입니다.

## 2026-08-01 source-first 전수 검수 완료

Japan 22페이지의 원본과 확대 crop을 후보 출력보다 먼저 판독해 truth를 잠그고, 세
경로의 A/B/C 결과를 전수 검수했습니다.

| 경로 | transcription exact | normalized character accuracy | semantic recall | merge/split |
|---|---:|---:|---:|---:|
| Paddle crop | 188/317, 59.31% | 76.10% | 177/187, 94.65% | 2 |
| Paddle Spotting | 140/317, 44.16% | 62.59% | 156/187, 83.42% | 10 |
| MangaLMM | 124/317, 39.12% | 57.69% | 151/187, 80.75% | 30 |

공백·줄바꿈·문장부호만 다른 Paddle crop 결과까지 포함한 reviewed transcription은
211/317입니다. MangaLMM 반복 방지 경로의 CUDA Japan 실행은 22/22페이지,
27 attempts, length 실패 0, 804.550초였고 ELVEN도 24/24페이지와 과거 실패 사례
2/2를 복원했습니다. 구조 안정화가 사람 기준 의미 정확도 우위를 뜻하지는 않았습니다.

세 경로 모두 파괴적 편집을 자동 허용하지 않았지만, full-page 두 경로는 detector
block과 OCR line의 N:1·1:N 관계 때문에 의미 recall과 merge/split에서 기준선보다
낮았습니다. Paddle crop을 추천·기본값으로 유지하고 다른 두 경로는 Experimental로
보존합니다. COO shadow는 안전한 SFX/의미 대사 분리점이 없어 어느 경로에도
적용하지 않습니다.

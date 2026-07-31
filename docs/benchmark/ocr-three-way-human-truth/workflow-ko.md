# 세 OCR 사람 정답·블라인드 검수 워크플로

이 벤치마크는 다음 세 경로를 동일한 원본·detector geometry·사람 정답으로
비교합니다.

1. `paddle_crop`: detector block별 PaddleOCR-VL crop OCR
2. `paddle_spotting_full_page`: PaddleOCR-VL full-page `Spotting:`
3. `mangalmm_full_page`: MangaLMM 공식 full-page spotting

이 도구는 Docker를 시작하거나 모델 요청을 보내지 않습니다. 이미 생성된 원시
결과를 표준 형식으로 가져오고 A/B/C 검수 패키지를 만드는 report-only 도구입니다.
실제 원본·정답·raw 응답·검수 결과는 전부 Git 밖에 둡니다.

## 순서

```text
원본 + detector snapshot
  → corpus manifest 생성·해시 고정
  → 후보를 보지 않고 사람/Codex 정답 CSV 작성·가져오기
  → truth lock
  → route별 source binding 생성
  → 세 route 원시 결과와 llama.cpp runtime contract 가져오기
  → A/B/C 검수 패키지 생성
  → 모든 판정 칸 작성
  → 완전성 검사 후 unblind·통계 생성
```

`truth lock` 전에는 후보 결과를 같은 폴더에 넣지 않습니다. lock에는 원본 복사본,
확대 crop, 페이지 정답 JSON의 SHA-256이 포함됩니다. lock 뒤 한 바이트라도 바뀌면
검수 패키지를 만들 수 없습니다. 초기 detector block은 하나도 삭제·병합하거나 bbox를
바꿀 수 없고, detector 밖 의미 텍스트만 `human_extra`로 추가합니다.

## 사람 정답 단위

기본 단위는 detector block입니다. detector가 놓친 의미 텍스트는
`region_source=human_extra`로 추가합니다. 각 region에는 다음을 모두 기록합니다.

- 원문 transcription
- `semantic_role`
- `processing_action`
- confidence
- detector block ID 또는 사람 추가 bbox
- 의미상 중요한 화자·행동·대상·고유명사 메모

공백·줄바꿈·문장부호를 무시한 글자 정확도와 의미 정확도를 분리합니다. 문자열
유사도는 진단값일 뿐이며, 의미 정답은 완료된 블라인드 검수에서만 계산합니다.

## full-page 추가 영역

Paddle Spotting이나 MangaLMM이 detector 밖에서 찾은 영역은 버리지 않습니다.
정답 region과 겹치면 해당 region에 연결하고, 연결되지 않으면 candidate-extra로
별도 검수합니다. UI 추가 검출 자체는 실패가 아니지만 다음은 실패로 집계합니다.
한 full-page region이 여러 detector block에 걸쳐 안전하게 귀속되지 않는 경우에도
텍스트를 복제하지 않고 `ambiguous/review` candidate-extra로 반드시 노출합니다.
마찬가지로 여러 사람 정답 영역을 가로지르는 compound unit은 각 정답 행에 복제하지
않고 한 candidate-extra 행에서 merge/split 여부를 판정합니다.

- 의미 대사를 놓침
- SFX·UI를 대사로 처리해 번역 또는 인페인트함
- 안전하지 않은 merge/split
- 원본 그림을 훼손하는 편집

## 종료 조건

세 route의 parser/length 오류, 사람 기준 의미 recall, 글자 정확도, extra false
positive, merge/split, 파괴 편집을 같은 표에 냅니다. 후보명과 속도·runtime 통계는
검수 완료 전까지 숨깁니다. 검수 CSV의 후보 원문·좌표·자산은 생성 시 잠그며 판정
칸만 수정할 수 있습니다. 이 보고서만으로 제품 기본값을 바꾸지 않으며 사용자 승인 후
별도 `develop` PR에서만 승격합니다.

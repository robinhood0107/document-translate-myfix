# Workflow Split Runtime Workflow

## execution_scope

- family: `workflow-split-runtime`
- corpus: `Sample/japan`
- pages: `13`
- smoke: `094.png`, `p_016.jpg`
- requirement_gate:
  - Requirement 1 먼저
  - Requirement 2는 Requirement 1 성공 후

## Requirement 1 실행 순서

1. 기존 레거시 페이지 단위 파이프라인 baseline 확보
2. 현재 파이프라인의 detect/OCR/translation/inpaint runtime 진입점 정리
3. 단계형 워크플로우 후보 정의
   - `baseline_legacy`
   - `candidate_stage_batched_single_ocr`
   - `candidate_stage_batched_dual_resident`
4. 체크포인트별 timing과 runtime snapshot 수집
5. Docker compose / health / reuse / timeout / retry 분해
6. 품질 동등성 검토
7. 성공 시 제품 승격 대상 runtime surface 확정

## Requirement 1 체크포인트

1. batch run start/end
2. detect stage start/end
3. OCR compose up start/end
4. OCR health wait start/end
5. OCR runtime reuse hit
6. OCR actual start/end
7. translation compose up start/end
8. translation health wait start/end
9. Gemma runtime reuse hit
10. translation actual start/end
11. inpaint stage start/end
12. render/export stage start/end
13. page done / page failed
14. timeout / retry / restart / reuse hit 모든 이벤트

## 현재 구현 상태

1. family runner / preset / BAT / report generator는 구현되었다.
2. `baseline_legacy`는 실제 offscreen app pipeline으로 바로 실행 가능하다.
3. `candidate_stage_batched_single_ocr`와 `candidate_stage_batched_dual_resident`는 결과 파일 구조까지는 잠겼지만, 아직 experimental runner가 없어 blocked 계약 run으로 기록된다.

## Requirement 2 실행 순서

1. Requirement 1 성공 판정 확인
2. 13장 기준 detect box count 수집
3. MangaLMM 결과 수와 `bbox_2d` 성공 여부 수집
4. PaddleOCR VL 결과와의 차이 수집
5. 사용자 검수용 page review pack 생성
6. 사용자 승인/비승인 결과 저장
7. selector rule 후보 도출
8. dual-resident runtime 정책과 자동 전환 정책 구현 후보 정리

## 문서화 흐름

1. 마스터 체크리스트 갱신
2. 프로젝트 명세/결정 로그 갱신
3. problem solving spec 생성 또는 갱신
4. results history 업데이트
5. generated report 업데이트

## develop 승격 흐름

1. benchmark/lab에서 full evidence 확보
2. `develop`용 별도 브랜치에서 benchmark 자산 없이 제품 코드만 이관
3. `docs/portfolio/...`에 요약형 포트폴리오 문서 저장
4. commit / push / PR

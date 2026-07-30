# GPU 인페인트 품질 게이트 워크플로

1. 개발 페이지와 잠금 holdout을 외부 manifest에서 분리한다.
2. source SHA-256, 현재 page snapshot, semantic role/action을 먼저 고정한다.
3. `capture`를 한 번만 실행해 detection/OCR geometry와 CTD mask를 동결한다.
4. 역사적 mask screen은 dilation 1·2·4와 Hough 구조 보호 경로를
   재현하는 용도로만 유지한다.
5. `mask-residual` screen에서는 product FP32 기준선, 굵은 외곽선
   dilation 2·4·6, 제한된 2차 GPU residual 후보를 비교한다.
6. 1차 mask, 2차 residual mask, 최종 합집합 mask를 각각 SHA-256
   산출물로 고정한다.
7. `preserve` 영역을 훼손하지 않은 mask만 model screen에 들어갈 수
   있다. 이 상태는 최종 품질 통과나 제품 승격을 의미하지 않는다.
8. model screen은 선택된 mask의 추출법·dilation·residual 계약 전체를
   잠그고 FP32 모델만 바꾼다.
9. 원문 잔상, 선화·망점 연속성, 외부 보존, 새 왜곡, render 순서로
   Codex가 전수 검수한다.
10. 자동 hard gate와 blind 검수를 모두 통과한 후보만 사용자에게 제시한다.
11. 사용자 승인 후보만 실제 offscreen 전체 파이프라인의 중립 개발
    케이스 두 종류에 반영한다.
12. 실제 후보 render를 새 result 계약에 첨부한 뒤 render 필수 blind
    검수를 다시 통과시킨다.
13. 규칙을 고정한 뒤 잠금 holdout은 정확히 한 번 실행한다.

동일 geometry/source duplicate는 capture 전에 canonical block 하나로
축약한다. 같은 bubble의 서로 다른 geometry 문장은 합치지 않는다.
`preserve`와 `review` block은 allowed edit window에 포함되지 않는다.
자동 hard gate에 실패한 후보는 blind 검수에 넣을 수 없다. blind bundle의
복사 산출물과 비공개 key도 SHA-256 계약으로 검증한다.

마스크 단계의 `model_screen_eligible_candidates`는 외부 보존만 통과한
중간 상태다. `coverage_eligible_candidates`는 잔상과 외부 보존을 함께
통과한 상태이며, `screen_eligible_candidates`는 모든 시각 필드를
통과한 상태다. render가 첨부된 최종 검수 전에는
`promotion_eligible_candidates`가 비어 있어야 한다.

ZITS/ZITS++는 adapter, 모델 SHA, 라이선스, CUDA FP32, 12GB VRAM 계약을
확인하기 전까지 `feasibility_not_implemented`로 기록되며 승격 후보가
아니다.

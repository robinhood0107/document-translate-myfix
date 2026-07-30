# GPU 인페인트 품질 게이트 워크플로

1. 개발 페이지와 잠금 holdout을 외부 manifest에서 분리한다.
2. source SHA-256, 현재 page snapshot, semantic role/action을 먼저 고정한다.
3. `capture`를 한 번만 실행해 detection/OCR geometry와 CTD mask를 동결한다.
4. mask screen에서 dilation 1·2·4를 후보명·속도 비공개로 검수한다.
5. 잔상 0을 만족하는 가장 작은 dilation을 고정한다.
6. model screen에서 같은 mask로 FP32 모델만 비교한다.
7. 원문 잔상, 선화·망점 연속성, 외부 보존, 새 왜곡, render 순서로
   Codex가 전수 검수한다.
8. 자동 hard gate와 blind 검수를 모두 통과한 후보만 사용자에게 제시한다.
9. 사용자 승인 후보만 실제 offscreen 전체 파이프라인에 반영해
   `p_016`, `i_102`를 다시 실행한다.
10. 실제 후보 render를 새 result 계약에 첨부한 뒤 render 필수 blind
    검수를 다시 통과시킨다.
11. 규칙을 고정한 뒤 잠금 holdout은 정확히 한 번 실행한다.

동일 geometry/source duplicate는 capture 전에 canonical block 하나로
축약한다. 같은 bubble의 서로 다른 geometry 문장은 합치지 않는다.
`preserve`와 `review` block은 allowed edit window에 포함되지 않는다.
자동 hard gate에 실패한 후보는 blind 검수에 넣을 수 없다. blind bundle의
복사 산출물과 비공개 key도 SHA-256 계약으로 검증한다.

ZITS/ZITS++는 adapter, 모델 SHA, 라이선스, CUDA FP32, 12GB VRAM 계약을
확인하기 전까지 `feasibility_not_implemented`로 기록되며 승격 후보가
아니다.

# GPU 인페인트 품질 게이트 워크플로

1. 개발 페이지와 잠금 holdout을 외부 manifest에서 분리한다.
2. source SHA-256, 현재 page snapshot, semantic role/action을 먼저 고정한다.
3. `capture`를 한 번만 실행해 detection/OCR geometry와 CTD mask를 동결한다.
4. 역사적 mask screen은 dilation 1·2·4와 Hough 구조 보호 경로를
   재현하는 용도로만 유지한다.
5. `mask-residual` screen에서는 product FP32 기준선, 굵은 외곽선
   dilation 2·4·6, 제한된 2차 GPU residual 후보와 서로 연결되지 않은
   전경 글자 성분을 별도 GPU pass로 처리하는 후보를 비교한다.
6. 1차 mask, 2차 residual mask, 최종 합집합 mask를 각각 SHA-256
   산출물로 고정한다.
7. `preserve` 영역을 훼손하지 않은 mask만 model screen에 들어갈 수
   있다. 이 상태는 최종 품질 통과나 제품 승격을 의미하지 않는다.
8. model screen은 선택된 mask의 추출법·dilation·residual 계약 전체를
   잠그고 FP32 모델만 바꾼다.
9. `structured-repair` screen은 완전 역할 주석의 불투명 말풍선을
   CUDA FP32 LaMa에 유지하고 구조 배경의 전경 글자만 결정론적 국소
   복원으로 분리한다. 이 경로도 학습형 인페인터의 CPU fallback을
   허용하지 않는다.
10. 원문 잔상, 선화·망점 연속성, 외부 보존, 새 왜곡, render 순서로
   Codex가 전수 검수한다.
11. 자동 hard gate와 blind 검수를 모두 통과한 후보만 사용자에게 제시한다.
12. 사용자 승인 후보만 실제 offscreen 전체 파이프라인의 중립 개발
    케이스 두 종류에 반영한다.
13. 실제 후보 render를 새 result 계약에 첨부한 뒤 render 필수 blind
    검수를 다시 통과시킨다.
14. 규칙을 고정한 뒤 잠금 holdout은 정확히 한 번 실행한다.

동일 geometry/source duplicate는 capture 전에 canonical block 하나로
축약한다. 같은 bubble의 서로 다른 geometry 문장은 합치지 않는다.
`preserve`와 `review` block은 allowed edit window에 포함되지 않는다.
자동 hard gate에 실패한 후보는 blind 검수에 넣을 수 없다. blind bundle의
복사 산출물과 비공개 key도 SHA-256 계약으로 검증한다.

현재 detector/OCR snapshot을 새 품질 계약으로 사용할 때는 먼저 Git
밖에서 완전 주석 template을 만든다.

```powershell
python scripts/benchmark_inpaint_quality_gates.py annotation-template `
  --manifest C:\external\cases.json `
  --output C:\external\cases-complete.json
```

사람이 원본을 보고 만든 별도 decisions JSON은
`apply-annotations --template ... --decisions ... --output ...`으로
결합한다. template과 판단 파일 SHA는 완료 manifest에 함께 고정된다.
모든 block의 역할·동작·mask 전략이 채워진 뒤에만 `capture`한다.
`complete-role-action-mask-v1` 계약은 빠진 block, 중복
index, 잘못된 action/mask 조합, 원문·좌표 SHA drift를 모두 거부한다.
`preserve`와 `review`는 항상 `preserve_original`이며,
`translate_inpaint`만 `bubble_safe`, `glyph_only`,
`glyph_only_structure_protect` 중 하나를 사용할 수 있다.

완전 주석 capture는 기존 product mask와 별도로 일반 불투명 말풍선의
`strategy_bubble_safe_mask`와 반투명·구조 배경용
`strategy_foreground_glyph_base_mask`를 저장한다. 후자는 배경 UI glyph가
아니라 굵은 전경 문자와 그 밝은 외곽선만 잡고 dilation 1·2·4 및
구조 보호 ON/OFF를 replay에서 비교한다. 두 mask 모두 각 block ROI
내부에서만 생성되며 보존 block은 0픽셀이어야 한다. 역사적 불완전
frozen contract는 재현용으로 계속 읽을 수 있지만 strategy-routed
후보에는 들어가지 않는다.

마스크 단계의 `model_screen_eligible_candidates`는 외부 보존만 통과한
중간 상태다. `coverage_eligible_candidates`는 잔상과 외부 보존을 함께
통과한 상태이며, `screen_eligible_candidates`는 모든 시각 필드를
통과한 상태다. render가 첨부된 최종 검수 전에는
`promotion_eligible_candidates`가 비어 있어야 한다.

ZITS++는 제품 인페인터가 아니라 격리된 lab feasibility adapter로만
실행한다. 공식 소스 커밋, model_512·LSM-HAWP SHA-256, digest-pinned
CUDA image를 모두 확인해야 한다. 어댑터는 한 프로필 동안 모델을 한
번만 로드하고 JSONL로 frozen case를 순차 처리한다. 실제 device가
CUDA가 아니거나 FP32가 아니거나 CPU fallback이 발생하면 실행을
실패시킨다. feasibility 후보는 자동·blind 품질 게이트를 통과하더라도
별도 제품화 승인 전에는 승격 후보가 아니다.

연결 성분 후보는 edit mask를 넓히지 않는다. 동일한 최종 mask를
8-connectivity 성분으로만 나누고 각 성분 주변의 고해상도 context를
순차 처리한다. `union_then_components`는 먼저 전체 mask를 한 번
정리해 인접 원문 글자가 문맥에 남지 않게 한 뒤 같은 성분 pass를
수행한다. 모든 pass가 끝난 뒤 원래 mask 합집합으로 다시 lossless
합성하므로 성분 밖 픽셀 변경은 허용되지 않는다. 이 후보들은 반투명
대사의 여러 세로 열을 하나의 큰 구멍으로 생성할 때 생기는 회색
얼룩과 글자형 재생성을 줄이는지 확인하기 위한 개발 실험이며, 실제
전체 파이프라인 검수 전에는 제품 승격 근거가 아니다.

필수 외부 계약은 `run`의 명시 인자 또는 같은 이름의 환경변수로
전달한다.

```text
--zits-source-root          / CT_ZITSPP_SOURCE_ROOT
--zits-model-checkpoint     / CT_ZITSPP_MODEL_CHECKPOINT
--zits-lsm-checkpoint       / CT_ZITSPP_LSM_CHECKPOINT
--zits-docker-image         / CT_ZITSPP_DOCKER_IMAGE
```

공식 원본·가중치·실행 결과는 Git 밖에 보존한다.

구조 배경 국소 복원은 `foreground_repair_mask`를 CUDA FP32 말풍선
mask와 분리해 기록한다. Telea/Navier–Stokes는 학습형 인페인터의 CPU
fallback이 아니라 mask 내부의 결정론적 보정 단계다. 최종 합성은 두
mask의 합집합 밖 픽셀을 변경할 수 없다. 구조선 재연결도 원본 mask
안에서만 허용되지만, blind 검수에서 대각선 오생성이 확인됐으므로
제품 경로로 승격하지 않는다.

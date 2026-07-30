# GPU 인페인트 품질 게이트 구조

`benchmark_inpaint_quality_gates.py`는 OCR·번역과 인페인트 후보를 분리하는
외부 frozen replay 프로토콜이다. 실제 페이지명, 좌표, 원문, annotation,
마스크, 후보 출력, 속도와 blind key는 Git 밖에만 둔다.

실행 구조는 다음과 같다.

1. 외부 case manifest와 현재 `page_snapshots.json`을 검증한다.
2. exact duplicate를 canonical block 하나로 축약한다.
3. `translate_inpaint` block만 사용해 CTD를 정확히 한 번 실행한다.
4. CTD glyph base, 기존 product mask, bubble/structure protect mask와
   allowed window를 SHA-256 계약으로 고정한다.
5. mask 단계는 같은 LaMa Large FP32/2048에서 dilation 1·2·4만 바꾼다.
6. model 단계는 선택된 dilation을 고정하고 LaMa Large FP32
   1536/2048, LaMa MPE FP32/2048, AOT FP32/2048을 비교한다.
7. BF16/1536은 비승격 baseline이며 ZITS는 lab feasibility 항목이다.
8. 후보 하나의 모델 load/OOM이 실패해도 해당 후보만 실패로 기록하고
   나머지 프로필은 계속 실행한다.
9. 후보 산출물을 무작위 A…N으로 바꿔 원본·mask·cleaned·diff·render를
   검수한다.
10. 모든 필드와 케이스별 고유 순위가 작성되기 전에는 unblind할 수 없다.

후보 inference는 전체 페이지 축소가 아니라 실제 edit mask 주변의 자동
고해상도 ROI에서 수행한다. `review_roi`는 검수 화면 범위일 뿐 모델 ROI를
넓히지 않는다. 첫 CUDA OOM에서만 더 작은 bounded ROI로 한 번 재시도하고,
다시 실패하면 CPU나 BF16으로 후퇴하지 않는다. 출력은 최종 mask로 다시
합성되며 mask 밖 변경 픽셀이 한 개라도 있으면 자동 실패다.

제품 승격 후보는 CUDA에서 실제 모델 parameter/provider가 확인되고,
FP32이며, 자동 hard gate와 직접 blind 검수를 모두 통과해야 한다.
frozen replay는 후보 선별 도구이며 제품 승격 증거를 단독으로 만들지
않는다. 최종 후보는 실제 offscreen 전체 파이프라인 render를 별도 SHA-256
manifest로 붙인 뒤 `--require-render` blind 검수를 통과해야 한다.
render 없는 중간 검수는 `screen_eligible_candidates`만 만들고,
`promotion_eligible_candidates`는 비워 둔다. render 필수 검수에서는
`NA`가 허용되지 않는다.

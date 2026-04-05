# benchmark 아키텍처와 코드 경계

## 구조

현재 benchmark 체계는 세 층으로 나뉩니다.

1. **core pipeline**
   - detect / OCR / inpaint / translate / render 단계
   - stage event emission
   - translator quality/retry 통계 surface
2. **benchmark orchestration**
   - preset 적용
   - managed / attach-running 실행
   - 결과 요약
   - translated export audit
3. **docs/report layer**
   - markdown 요약
   - pandas 집계
   - chart PNG 생성
   - 최종 문서 출력

## 왜 이렇게 나누는가

실험/문서/추천 로직이 core pipeline 안으로 많이 들어가면, 이후 business code를 업데이트할 때 benchmark 코드가 같이 깨질 가능성이 커집니다.

반대로 benchmark가 완전히 core에서 분리되어 있으면 실제 단계별 시간을 신뢰성 있게 잡기 어렵습니다.

그래서 현재 권장 경계는 다음입니다.

- core에는 “관측 지점”만 둔다
- 실험 정책은 core 밖에 둔다

## 현재 결론

지금 이 저장소는 이미 하이브리드 구조입니다. 이건 나쁜 상태가 아닙니다.

다만 앞으로는 아래를 규칙으로 삼는 것이 좋습니다.

- stage/tag schema는 안정적으로 유지
- winner 판정은 `scripts/`에서만 수행
- docs 생성은 `scripts/generate_benchmark_report.py`가 담당
- 결과 원본은 `./banchmark_result_log`에 유지

이 방향이 추후 `develop` 비즈니스 코드 업데이트와 benchmark 코드 업데이트를 가장 덜 충돌하게 만듭니다.

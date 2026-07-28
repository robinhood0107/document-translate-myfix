# Gemma 최종 A/B blind 품질 검수 latest

## 현재 상태

- protocol: v4 report-only
- source contract revalidation: PASS
- imported candidate families: 2
- imported rounds per family: 2
- verified pages: 22/22
- verified rows: 292/292
- Q8 candidate inclusion: 0
- Docker/model request count: 0
- blind review artifact: 생성 완료
- user-reviewed rows: 0/292
- unblind: 차단
- product promotion: 차단

## 다음 gate

사용자가 A1·A2·B1·B2 전체를 검수해 유효한 292행 CSV를 제출해야 한다.
검수 완료 전에는 mapping을 공개하지 않으며 제품 기본값 변경과 전체
파이프라인 비교를 시작하지 않는다.

grouped 후보에서 화자·관계·부정·행동·대상·숫자·고유명사·명시적 의미
회귀가 하나라도 확인되면 품질 탈락으로 종료한다.

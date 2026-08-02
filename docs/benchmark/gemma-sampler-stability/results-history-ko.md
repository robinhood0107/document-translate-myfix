# Gemma sampler stability results history

## 2026-08-02

- 고정 corpus: 22개 text case
  - 일본어→한국어 18개: Router mismatch 14개, meaning control 4개
  - 영어→한국어 4개: 관계·행동, explicit 표현, 금액·숫자, 정체성 변화
- manifest SHA-256: `db0e8f4de5d4afbc5e27b59119ed97a870a8079ecaf26e7115441150f80171ce`
- 전체 unique sampler plan: 990 response
- 실제 완료: 990 response
- 세 seed 순서: forward, reverse, center-out
- 모든 완료 arm: 구조 오류 0
- 모든 최종 후보: hard semantic fail 0, seed stable 20/22
- 각 최종 arm의 66 response에서 protocol channel token은 parser와 동일하게 제거되어 기록됨

최종 후보 비교:

| tuple | clean | stable | 평균 latency | 평균 completion token |
| --- | ---: | ---: | ---: | ---: |
| `temperature=0.0, top_p=0.90, top_k=32, min_p=0` | yes | 20/22 | 0.800214s | 25.379 |
| `temperature=0.0, top_p=0.90, top_k=64, min_p=0` | yes | 20/22 | 0.824385s | 25.379 |
| `temperature=0.0, top_p=0.90, top_k=128, min_p=0` | yes | 20/22 | 0.794207s | 25.379 |

latency와 completion token을 동률 해소 기준으로 적용한 lab winner는 `temperature=0.0, top_p=0.90, top_k=128, min_p=0`이다. 이 문서와 runner는 benchmark 결과를 기록할 뿐이며, 제품 기본값 변경은 별도 `develop` 기반 PR에서 수행한다.

실행 종료 증거는 private report에서 `loaded_count=0`, `container_running=false`, `release_failed=false`로 확인했다. raw request/response, canonical 원문, private judgment와 source locator는 Git에 보관하지 않는다. 기존 Router lab REJECT 기록은 수정하지 않았다.

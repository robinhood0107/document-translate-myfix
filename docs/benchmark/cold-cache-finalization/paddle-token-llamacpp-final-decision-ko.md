# PaddleOCR-VL llama.cpp completion token 최종 판정

## 결론

`max_new_tokens=768`은 제품에 승격하지 않고 1024를 유지한다. 일본어·중국어·영어를 실제 CUDA Stage-Batched 경로에서 3회씩 비교한 결과, 출력은 9/9 완전히 같았으나 반복 측정상 속도 우위를 입증하지 못했다.

## 결과

| 지표 | 평균 개선 | 중앙값 개선 | 단측 95% bootstrap 하한 |
|---|---:|---:|---:|
| 전체 elapsed | +0.057% | -0.010% | -1.674% |
| pipeline run wall | +0.857% | +0.316% | -0.150% |
| OCR request 합계 | +0.499% | +0.421% | -0.520% |

- 9/9 page snapshot exact
- 페이지 실패 0, HTTP retry 0
- 일본어 60·중국어 42·영어 13 OCR 요청/실행
- 실행 사이 Compose `stop`, 모든 cache 비활성

최소 개선율은 적용하지 않았다. 후보의 작은 명목 이득이 측정 노이즈와 구분되지 않았으므로 `reject_no_proven_speed_gain`으로 확정한다. 과거 작은 표본에서 관측된 +4.178%는 현재 llama.cpp 다언어 반복 실행에서 재현되지 않았다.


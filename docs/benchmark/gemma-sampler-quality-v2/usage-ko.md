# Gemma sampler quality v2 사용법

실제 입력과 결과는 모두 private archive에서만 지정한다. 먼저 reference가 `FROZEN` 상태이고 사용자 24개 표본 검수가 기록되어 있어야 한다.

Windows CUDA 13 장시간 실행 예시는 다음 환경 변수 형태다.

```bat
set SAMPLER_REFERENCE=<private-frozen-reference>
set SAMPLER_PHASE=temperature
call scripts\benchmark_gemma_sampler_quality_v2_cuda13.bat
```

joint phase는 `SAMPLER_SELECTION`에 선택된 두 temperature와 이전 temperature response run을 `SAMPLER_PRIOR_RESPONSE_RUN`으로 지정한다. min-p phase는 선택된 세 tuple과 이전 joint response run을 같은 방식으로 지정한다. BAT가 exit code 75를 받으면 동일 managed run을 자동 resume한다. Windows의 progress checkpoint 임시 파일 교체가 잠시 거부된 것으로 감사 기록이 남은 경우에만 동일 run을 복구할 수 있으며, 그 밖의 failed manifest는 재개하지 않는다. 완료하면 raw 결과를 공개하거나 stage하지 말고, private blind judgment packet으로 다음 gate를 진행한다.

# Gemma sampler quality v2 사용법

실제 입력과 결과는 모두 private archive에서만 지정한다. 먼저 reference가 `FROZEN` 상태이고 사용자 24개 표본 검수가 기록되어 있어야 한다.

Windows CUDA 13 장시간 실행 예시는 다음 환경 변수 형태다.

```bat
set SAMPLER_REFERENCE=<private-frozen-reference>
set SAMPLER_PHASE=temperature
call scripts\benchmark_gemma_sampler_quality_v2_cuda13.bat
```

BAT를 실행하면 별도 `Gemma Sampler Monitor` 창이 자동으로 열린다. 이 창은 Go로
빌드한 read-only 전용 TUI이며, runner·Docker·GPU 작업을 제어하지 않는다.

- 처음 한 번은 private archive의 `gemma-monitor.exe`를 빌드한다. Scoop Go가
  설치되어 있으면 사용자 `Path`가 아직 갱신되지 않은 기존 CMD에서도 Scoop shim을
  직접 찾아 사용하므로 별도 환경 변수 설정이 필요 없다.
- 화면에는 현재 phase/state, 완료·잔여 수, 최근 완료 표본 기반의 rate/ETA, 파일
  freshness, GPU별 VRAM·사용률·온도가 표시된다. ETA는 최근 연속 실행 구간의
  30/90/240개 표본 rate 중앙값을 사용하며, progress가 90초 이상 갱신되지 않으면
  추정을 일시 보류한다.
- `q`, `Esc`, `Ctrl+C`는 모니터 창만 닫는다. 실제 runner·Docker·GPU 작업은 계속
  실행된다. phase가 `WAITING_FOR_JUDGMENT`에 정상 도달하면 모니터는 잠시 결과를
  보여 준 뒤 자동으로 닫힌다.
- runner stdout/stderr는 private archive의 해당 run-id `supervisor-logs` 파일에
  계속 추가 저장된다. raw 결과나 로그는 Git에 stage하지 않는다.
- 의도적인 무인 실행에만 `set SAMPLER_NO_MONITOR=1`을 지정한다. 기본값은 모니터
  표시다. Go 경로를 직접 지정해야 하는 특수 환경에서는 `GEMMA_MONITOR_GO`를,
  private EXE 위치를 바꿔야 할 때만 `GEMMA_MONITOR_OUTPUT`을 지정할 수 있다.

joint phase는 `SAMPLER_SELECTION`에 선택된 두 temperature와 이전 temperature response run을 `SAMPLER_PRIOR_RESPONSE_RUN`으로 지정한다. min-p phase는 선택된 세 tuple과 이전 joint response run을 같은 방식으로 지정한다. BAT가 exit code 75를 받으면 동일 managed run을 자동 resume한다. Windows의 progress checkpoint 임시 파일 교체가 잠시 거부된 것으로 감사 기록이 남은 경우에만 동일 run을 복구할 수 있으며, 그 밖의 failed manifest는 재개하지 않는다. 완료하면 raw 결과를 공개하거나 stage하지 말고, private blind judgment packet으로 다음 gate를 진행한다.

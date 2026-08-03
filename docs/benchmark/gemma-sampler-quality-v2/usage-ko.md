# Gemma sampler quality v2 사용법

이 benchmark는 승인된 frozen reference와 완료된 r6를 private managed archive에서
자동으로 찾아 검증한다. 원문·응답·판정 경로를 BAT나 명령줄에 붙여 넣지 않는다.

## 실행할 것

평소에는 `banchmark_result_log\\tools\\gemma-sampler-launcher.exe`를 더블클릭한다.
이것만 실제 장시간 CUDA13 실행을 시작하는 진입점이다. launcher는 다음을 한 번에 한다.

1. CUDA13 BAT를 백그라운드 supervisor로 시작한다.
2. Bubble Tea 전체화면 monitor 하나만 연다.
3. r6와 frozen reference를 읽기 전용으로 검사한다.
4. 이미 살아 있는 runner가 있으면 새 runner를 만들지 않고 monitor만 다시 붙인다.
5. 이전 Python worker가 끊긴 checkpoint라면 같은 campaign ID로 이어서 실행한다.

launcher는 매번 monitor source가 더 새로우면 `scripts\build_gemma_sampler_monitor.bat
--monitor-only-if-stale`를 먼저 호출한다. Scoop Go가
설치되어 있으면 Scoop shim을 직접 찾으므로, 새 CMD를 열어 PATH를 다시 주입할 필요가
없다. 실행 파일은 ignored private artifact 영역에만 만들어지며 Git에 올리지 않는다.

`--verify`는 GPU·Docker·runner를 시작하지 않는 사전검사다.

```bat
call scripts\benchmark_gemma_sampler_quality_v2_cuda13.bat --verify
```

일반 `scripts\benchmark_gemma_sampler_quality_v2.bat`는 같은 사전검사만 수행한다.
실제 inference를 시작하지 않으므로, 실수로 일반 BAT를 눌러도 campaign은 시작되지 않는다.

## monitor에서 보이는 것

- 새 실행 진행률: `완료 / 124,280`
- r6 재사용: `9,560`, 전체 증거: `완료 + 재사용 / 133,840`
- 현재 단계, temperature, top-p, top-k, min-p, seed, case 위치
- valid·retry·timeout·indeterminate 수
- r6 실측 기반 초기 ETA와 500응답 이후 live 속도 보정 ETA
- 활성 실행 시간·retry backoff 시간, GPU VRAM·사용률·온도, checkpoint freshness
- worker PID, 마지막 완료·마지막 상태, 정확한 private supervisor log 위치

monitor는 read-only다. `q`, `Esc`, `Ctrl+C`는 **monitor 창만** 닫고 runner와 Docker
작업은 계속한다. 다시 EXE를 더블클릭하면 살아 있는 runner에 다시 붙는다.

## 언제 멈추는가

정상 실행은 `WAITING_FOR_FINAL_JUDGMENT`에서 자동으로 멈춘다. Router unload와 GPU 반환이
확인되기 전에는 이 상태로 바뀌지 않는다. contract mismatch, foreign runtime, slot drain
실패, VRAM 반환 실패, 저장공간 부족은 fail-closed로 멈춘다. 이 경우에는 무작정 다시
실행하지 말고 저장된 상태와 log를 확인한 뒤 원인을 해결한다.

PC 재부팅이나 BAT worker 강제 종료 뒤에는 같은 EXE를 다시 실행한다. 이미 first-valid로
저장된 logical slot은 재추론하지 않는다. 정상 완료 전에는 수동으로 결과를 삭제하거나
새 run ID를 만들지 않는다.

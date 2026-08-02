# Gemma CPU-MoE / Host-KV Lab Usage

이 runner는 managed private validation archive에만 raw request, response,
container log, 1초 resource sample을 기록한다.

```powershell
.venv-win\Scripts\python.exe -B scripts\benchmark_gemma_cpu_moe_offload.py --mode plan
```

screen은 private fixed-seed replay를 명시적으로 받아 실행한다.

```powershell
.venv-win\Scripts\python.exe -B scripts\benchmark_gemma_cpu_moe_offload.py `
  --mode screen `
  --translation-replay <private-fixed-seed-replay.json>
```

screen이 반환한 최소 physical-fit level은 별도 structural gate에서 검증한다.

```powershell
.venv-win\Scripts\python.exe -B scripts\benchmark_gemma_cpu_moe_offload.py `
  --mode structural `
  --selected-n-cpu-moe <N> `
  --translation-replay <private-fixed-seed-replay.json>
```

`co-resident-sentinel`은 speed 또는 product promotion을 뜻하지 않는다. 이는
지정한 `N`을 fixed request로 다시 physical-fit 확인한 뒤 Gemma와 Paddle의 실제
동시 load/health 및 cleanup만 확인하는 별도 관측이다. output parity가 맞지 않아도
물리 상주 여부 질문에는 사용할 수 있지만, fit 재확인 실패 시 Paddle을 시작하지 않는다.

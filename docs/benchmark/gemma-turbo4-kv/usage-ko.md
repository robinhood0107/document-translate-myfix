# Gemma Turbo4 KV-V usage

기본 계획과 자원 확인은 다음과 같다.

```bash
.venv-win/Scripts/python.exe -B scripts/benchmark_gemma_turbo4_kv.py --mode plan
.venv-win/Scripts/python.exe -B scripts/benchmark_gemma_turbo4_kv.py --mode preflight
```

실행 데이터는 private validation archive에만 둔다. fixed-seed 입력은 `requests` 배열을 가진 replay JSON 또는 검증된 page snapshot 중 하나를 사용한다. snapshot 입력은 현행 contextual-single 요청 순서를 재구성하고, IQ4_NL model id·seed `20260801`·prompt·schema를 고정한다.

```bash
.venv-win/Scripts/python.exe -B scripts/benchmark_gemma_turbo4_kv.py \
  --mode structural \
  --reuse-image \
  --page-snapshots <private validated snapshot>
```

S1/S6는 passing structural gate를 `--structural-gate`로 명시해 재실행 없이 이어간다. S6는 passing S1 gate도 `--s1-gate`로 요구한다. true series 3+3은 series-queue adapter와 private series fixture가 준비되기 전에는 의도적으로 실행되지 않는다. flat six-page batch는 series 결과가 아니다.

`--output-dir`을 생략하면 managed private artifact harness가 raw request/response, Docker command, 1초 자원 샘플, actual HTTP request ledger, runtime evidence를 기록한다. public Git에는 summary source와 protocol만 남긴다.

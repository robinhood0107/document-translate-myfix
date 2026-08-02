# Gemma Turbo4 KV-V usage

계획과 독립 lab preflight는 다음 명령으로 확인한다.

```bash
.venv-win/Scripts/python.exe -B scripts/benchmark_gemma_turbo4_kv.py --mode plan
.venv-win/Scripts/python.exe -B scripts/benchmark_gemma_turbo4_kv.py --mode preflight
```

fixed-seed 입력은 private replay JSON 또는 검증된 page snapshot에서 재구성한다. IQ4_NL
model id, seed `20260801`, prompt, schema, 요청 순서를 바꾸지 않는다. raw 응답, 텍스트,
명령, 이미지, 자원 샘플은 managed private archive에만 쓴다.

raw response non-exact는 진단으로 기록한다. 모든 hard contract가 통과했을 때만 private
semantic approval이 text-only PASS를 주어 속도 진입을 허용한다. page는 애매한 항목에만
선택적으로 사용한다. 누락 텍스트, `finish_reason` 불일치, JSON 불완전은 approval 대상이
아니며 GPU replay 없이 REJECT로 종료한다.

```bash
.venv-win/Scripts/python.exe -B scripts/benchmark_gemma_turbo4_kv.py \
  --mode structural \
  --reuse-image \
  --translation-replay <private fixed-seed replay>
```

S1/S6는 passing structural gate를 `--structural-gate`로 재사용한다. S6는 passing S1 gate도
요구한다. true series 3+3은 별도 series queue adapter와 private fixture 없이는 실행하지
않는다. `--output-dir`을 생략하면 private artifact harness가 run manifest와 증거를 관리한다.

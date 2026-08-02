# Single llama.cpp Router handoff usage

Private fixture manifest를 준비한 뒤 supported Windows runtime에서 실행한다.

```text
python scripts/benchmark_llamacpp_router_handoff.py --mode preflight --fixture-manifest <private-fixture-manifest>
python scripts/benchmark_llamacpp_router_handoff.py --mode abba --fixture-manifest <private-fixture-manifest> --python <windows-python>
python scripts/benchmark_llamacpp_router_handoff.py --mode review --review-artifact-dir <private-abba-artifacts> --semantic-approval-dir <private-approvals>
```

공개 최종 보고서 갱신은 모든 pair의 review가 끝난 경우에만 별도로 허용한다. 부분 preflight·ABBA·review는 기존 최종 보고서를 덮어쓰지 않는다.

HunyuanOCR volume은 필요할 때만 먼저 준비한다.

```text
python scripts/prepare_hunyuanocr_router_lab_volume.py --mode prepare --source-dir <private-q8-source>
```

모든 raw request/response, text context, source page, image, command, resource sample은 managed private archive에만 남는다. public report에는 pair별 집계만 기록한다.

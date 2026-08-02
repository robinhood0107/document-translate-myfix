# Gemma Turbo4 KV-V architecture

```text
protocol pin + resource preflight
              │
              ├─ build/verify fork image
              ├─ same-seed structural + fork F16/Turbo4 ABBA
              └─ product offscreen ABBA
                     │
                  Turbo4LabRuntimeManager
                     │
        StageBatchedProcessor ensure_server()
        (only after inpainter release gate)
                     │
         exact ct-gemma-turbo4-* container
         loopback port + read-only model volume
```

`Turbo4LabRuntimeManager` is a lab-only subclass of the product local Gemma manager so the existing `isinstance` safety path remains active. Installing it does not start a model. Its first Docker start occurs only when translation needs Gemma after the product inpainter-release gate.

Each candidate fixes context 4096, NGL 23, parallel 1, threads 10, batch 2048, ubatch 512 and F16 K. The only allowed candidate delta is V cache `f16` versus `turbo4`. Shipping F16, fork F16, and Turbo4 all pass through the same adapter, including the product-equivalent inert chat prewarm. The adapter checks model identity, read-only volume mount, image ID, labels, exact command fingerprint, GPU return and exact lab container cleanup. Full-auto raw HTTP requests/responses stay private; only their ordered hashes enter the parity gate.

여기의 parity gate는 raw 재현성 gate다. 문자열 hash가 다르면 현 protocol의 속도
진입은 막지만, 그 사실만으로 번역 의미 회귀라고 결론내리지는 않는다. 번역 텍스트는
[공통 번역 후보 품질 판정 규칙](../translation-quality-evaluation-rule-ko.md)의
source-first 맥락 검수로 별도 판정한다. 반면 request 계약·모델 identity·최종
decoded pixel 계약은 계속 hard gate다.

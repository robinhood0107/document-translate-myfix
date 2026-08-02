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

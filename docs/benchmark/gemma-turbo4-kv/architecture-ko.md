# Gemma Turbo4 KV-V architecture

```text
private fixed-seed text ledger
          │
          ├─ hard contract: identity / request / JSON / finish_reason
          ├─ text-first semantic review for raw deltas
          └─ fork F16 ↔ Turbo4 ABBA
                    │
           shipping F16 ↔ Turbo4 ABBA
                    │
             S1 → S6 → true series 3+3
```

`Turbo4LabRuntimeManager`는 lab 전용 local Gemma manager다. install만으로 모델을
시작하지 않으며, 번역 단계가 필요하고 Arbiter의 inpainter 반환 gate가 확인된 뒤에만
정확한 `ct-gemma-turbo4-*` 컨테이너를 연다. 각 arm은 loopback port와 read-only model
volume을 사용한다.

모든 후보는 context 4096, NGL 23, parallel 1, threads 10, batch 2048, ubatch 512, F16 K를
고정한다. fork F16과 Turbo4의 유일한 차이는 V cache `f16`/`turbo4`다. fork commit, image
ID/digest, 모델 SHA, 명령 fingerprint, health/model identity를 private manifest에 묶는다.
generic official image의 Turbo4 요청과 Turbo4+MTP/draft 조합은 허용하지 않는다.

raw 응답 hash는 재현성 진단값이다. non-exact인 경우 private approval은 원문·전체 대사
맥락·인접 대사·기준/후보 텍스트를 hash/run identity에 묶어 PASS/REVIEW_REQUIRED/REJECT로
남긴다. 페이지는 판단이 필요한 항목에서만 선택적으로 본다. JSON/schema/completion,
요청 계약, runtime identity는 의미 검수로 우회할 수 없다.

Turbo4 E2E에서는 upstream detection/OCR/mask/inpaint snapshot을 exact로 비교하고 render
완료만 요구한다. 각 arm의 실제 HTTP 번역 response ledger는 순서·개수·JSON hard contract를
검사하고 raw delta는 hash-bound text-first 검수로만 허용한다. 의미가 같은 번역의 final decoded
pixel SHA 차이는 진단값이며 품질 실패가 아니다. GPU/RAM/shared/swap은 모든 arm에서 기록하지만
그 관측 자체가 승패를 정하지는 않는다. OOM, runtime/container 불안정, orphan, GPU 반환 미확인은
즉시 REJECT다.

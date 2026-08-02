# Gemma Turbo4 KV-V results history

| Date | Stage | Decision | Notes |
|---|---|---|---|
| 2026-08-02 | fixed-seed structural (shipping F16 control) | INCOMPLETE | 당시 host-memory 자동 중단으로 완결 요청이 없었다. 아래 완결 73-request 기록으로 대체한다. |
| 2026-08-02 | existing 73-request text-ledger hard-contract recheck | REJECT | 새 GPU replay 없이 기존 private ledger를 재검증했다. Turbo4는 73 응답 중 14건이 `finish_reason=length`이고 유효한 translation JSON도 없어 hard gate를 통과하지 못했다. 따라서 의미 승인과 ABBA는 시작하지 않았다. raw mismatch는 shipping↔fork F16 26/73, fork F16↔Turbo4 72/73, shipping F16↔Turbo4 72/73으로 진단값만 보존한다. Peak VRAM 11,925 MiB, OOM 0, orphan 0, 각 arm GPU 반환 확인, R3 추정 111.36%이며 active R3는 실행하지 않았다. |
| 2026-08-02 | mounted GGUF identity follow-up | PASS | read-only model volume의 실제 file SHA-256이 pinned IQ4_NL identity와 일치했다. |

공개 기록에는 PASS/REJECT, 집계 quality 상태, timing confidence interval, peak VRAM,
RAM/shared/swap 관측, OOM/orphan/GPU 반환, R3 추정만 남긴다. 원문·번역문·prompt·image·
local corpus identity와 raw response는 private archive에만 둔다.

# Gemma Turbo4 KV-V latest report

Status: **REJECT — hard contract (기존 73-request 자료 재검증, 새 GPU replay 없음)**

- 번역 품질: semantic PASS/FAIL 판정 대상이 아님. Turbo4 14/73 응답이 `finish_reason=length`이고
  translation JSON이 완결되지 않아 필수 후보 텍스트가 없다. 이 조건은 의미 검수나 raw mismatch
  허용으로 우회할 수 없다.
- raw mismatch 진단: shipping↔fork F16 26/73, fork F16↔Turbo4 72/73,
  shipping F16↔Turbo4 72/73. raw 문자열 차이 자체는 이번 REJECT 사유가 아니다.
- Median / one-sided 95% confidence interval: 없음. fork F16↔Turbo4 ABBA, S1, S6, series 3+3은
  시작하지 않았다.
- 기존 structural 1회 관측: shipping F16 46.927s, fork F16 45.515s, Turbo4 223.554s.
  승격 timing 근거가 아니다.
- Peak VRAM 11,925 MiB; Windows available RAM minimum 3.77 GiB; WSL swap growth 2.44 MiB;
  shared GPU growth 231.1 MiB. 모두 관측값이며 독립 REJECT 사유가 아니다.
- OOM 0, orphan 0, 각 arm 종료 후 GPU 반환 확인. R3 추정은 13,677 MiB / 12,282 MiB = 111.36%로
  90% 초과이며 active Paddle+Gemma 동시 상주는 실행하지 않았다.

이 결과 뒤에는 제품 반영이나 다음 성능 로드맵을 시작하지 않는다. 사용자 승인 전에는
`feature/gemma-turbo4-kv`를 만들지 않는다.

# Gemma llama.cpp 프로필 토너먼트 결과 이력

## 2026-07-29 — protocol v1 준비

- llama.cpp target/no-spec/ngram/MTP 공통 runner 추가
- model/draft SHA-256과 GGUF tokenizer fingerprint 잠금 추가
- 잘못된 MTP pairing과 cache 혼입 거부 추가
- target NGL 23 기준 상·하향 경계 탐색 추가
- runtime-ready, request-only, TTFT, draft/accepted metric, VRAM·swap 분리
- paired 단측 95% bootstrap 판정 추가
- 최소 속도 개선률을 두지 않는 정책 고정
- 같은 로컬 파일명을 가진 MTP의 볼륨 목적지 충돌을 재현하고
  `volume_filename` 고유 계약과 사전 중복 검사를 추가
- HauhauCS·Unsloth QAT·Unsloth base MTP를 계열별 이름으로 분리
- pinned llama.cpp b10133에 맞춰 ngram-mod와 MTP draft 길이 flag를 분리
- target 6개와 MTP 3개의 named volume 전체 SHA-256 9/9 검증 완료

실제 후보 결과와 원문·번역은 Git 밖에 저장한다. 최신 중립 상태 요약은
`generated/latest-report-ko.md`에 기록한다.

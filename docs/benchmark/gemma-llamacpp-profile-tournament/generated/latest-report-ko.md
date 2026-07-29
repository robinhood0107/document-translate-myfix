# Gemma llama.cpp 프로필 토너먼트 최신 상태

- protocol: `v1`
- 상태: runner·named volume 준비·전체 SHA 검증 완료, GPU 실행 대기
- 제품 기준선: `IQ4_NL + contextual-single + chunk 6 + no-spec + F16`
- 최소 속도 향상률: 없음
- raw 결과 위치: Git 밖 validation log

target 6개와 계열별 MTP 3개는 두 named volume에 준비됐고 9/9 전체
SHA-256 검증을 통과했다. pinned llama.cpp는 image b10133이며 ngram과
MTP의 현재 CLI flag 계약을 분리해 검증한다.

실제 GPU 실행은 외부 GPU 점유가 없는 reproducible preflight에서만
진행한다. 우승 프로필은 모든 구조·의미·메모리 게이트와 paired 속도
검증을 통과하기 전까지 확정하지 않는다.

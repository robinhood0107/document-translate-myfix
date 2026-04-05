# 자동번역 벤치 체크리스트

이 문서는 이번 작업과 이후 반복 최적화를 추적하기 위한 체크리스트입니다.

## 1. 저장소 / 브랜치

- [x] 이전 PR 머지 상태 확인
- [x] `develop` 최신화
- [x] 새 작업 브랜치 생성
- [x] `/Sample` gitignore 추가
- [x] 현재 작업 브랜치 첫 push 완료
- [x] 머지 완료된 old remote branch 삭제 마무리

## 2. 기준 번들

- [x] `paddleocr_vl_docker_files/` 저장소 포함
- [x] bundle README 작성
- [x] Gemma current compose drift 기록
- [x] merged baseline vs live-ops baseline 분리 문서화

## 3. 계측

- [x] GPU metrics helper 추가
- [x] memlog에 GPU snapshot 통합
- [x] batch 단계 태그 추가
- [x] webtoon 단계 태그 추가
- [x] background update check env disable 추가

## 4. 벤치 도구

- [x] preset staging 스크립트 추가
- [x] offscreen benchmark 스크립트 추가
- [x] benchmark summarize 스크립트 추가
- [x] translation-only preset 세트 추가

## 5. 문서

- [x] 자원 전략 문서
- [x] 벤치 가이드 문서
- [x] 벤치 결과 이력 문서
- [x] 체크리스트 문서
- [x] README 링크 갱신
- [x] legacy preset 이름 제거 후 translation-only 기준으로 재정리

## 6. 실제 검증

- [x] Python compile / syntax 검증
- [x] `validate_changed_python.py --all`
- [x] `headless_smoke.py`
- [x] `compile_translations.py --check`
- [x] benchmark script smoke 실행
- [x] managed runtime dry-run 확인

## 7. 다음 튜닝 루프

- [x] `translation-baseline` one-page 재확정
- [x] `translation-baseline` batch 30장 재확정
- [x] `translation-ngl20` one-page
- [x] `translation-ngl21` one-page
- [x] `translation-ngl22` one-page
- [x] `translation-ngl23` one-page
- [x] `translation-ngl24` one-page
- [x] one-page 상위 3개 후보 검토 후 representative 승격/pruning 결정
- [x] best `n_gpu_layers` 확정
- [x] `translation-t04` one-page
- [x] `translation-t05` one-page
- [x] `translation-t06` one-page
- [x] `translation-t07` one-page
- [x] 대표 temperature 후보 representative batch 승격
- [ ] 필요 시 `translation-ngl24-ctx3072` rescue
- [x] translated text audit subset 비교
- [x] 최종 승자 preset을 results 문서에 승격

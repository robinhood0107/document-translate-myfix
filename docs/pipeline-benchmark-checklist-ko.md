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
- [x] preset 4종 추가

## 5. 문서

- [x] 자원 전략 문서
- [x] 벤치 가이드 문서
- [x] 벤치 결과 이력 문서
- [x] 체크리스트 문서
- [x] README 링크 갱신
- [x] Gemma 문서 현재 baseline 반영

## 6. 실제 검증

- [x] Python compile / syntax 검증
- [x] `validate_changed_python.py --all`
- [x] `headless_smoke.py`
- [x] `compile_translations.py --check`
- [ ] benchmark script smoke 실행
  현재 `.venv`에 `cv2`가 없어 full pipeline import 단계에서 fail-fast 확인까지만 완료
- [ ] managed runtime dry-run 확인

## 7. 다음 튜닝 루프

- [ ] `repo-default` 수집
- [ ] `live-ops-baseline` 수집
- [ ] `gpu-shift-ocr-front-cpu` 비교
- [ ] `gemma-heavy-offload` 비교
- [ ] 승자 preset을 results 문서에 승격

# 자동번역 벤치 결과 이력

이 문서는 실제 벤치 결과를 커밋 기준으로 누적 기록하는 곳입니다.

## 기록 규칙

- 새 baseline을 승격할 때만 이 문서를 갱신합니다.
- 항상 commit SHA와 preset 이름을 함께 남깁니다.
- representative corpus 기준 결과를 우선 기록합니다.

## 템플릿

```text
### <commit-sha> <preset-name>

- representative elapsed_sec:
- page_done_count:
- page_failed_count:
- gpu_peak_used_mb:
- gpu_floor_free_mb:
- 판단:
```

## 현재 상태

- 벤치 인프라와 preset/문서가 먼저 들어간 상태입니다.
- 다음 단계부터 `/Sample` 30장 코퍼스로 실제 수치를 누적합니다.

# Repository Rules

이 문서는 이 저장소의 Git 규칙, 브랜치 정책, CI/CD 흐름, 번역 반영 규칙의 단일 기준 문서다.
사람과 에이전트 모두 이 문서를 먼저 읽고 작업해야 하며, 이 문서와 다른 문서가 충돌하면 이 문서를 우선한다.

## 1. 기본 원칙

- `main`과 `develop`은 보호 브랜치다.
- 기능 개발은 반드시 별도 작업 브랜치에서만 한다.
- 트래킹되는 코드 변경 전에 먼저 브랜치를 만든다.
- 기능 단위 작업은 `커밋`과 `push`까지 끝나야 완료로 본다.
- 사용자에게 보이는 UI를 바꾸면 문서, 번역, 필요 시 변경 이력까지 함께 갱신한다.
- 가상환경, 캐시, 임시 산출물은 Git에 올리지 않는다.

## 2. 브랜치 모델

이 저장소는 엄격한 `main + develop + tag` 모델을 사용한다.

- `main`
  - 배포 기준 브랜치
  - 직접 커밋, 직접 push 금지
- `develop`
  - 통합 기준 브랜치
  - 일반 기능은 모두 이 브랜치로 PR
  - 직접 커밋, 직접 push 금지
- 작업 브랜치
  - `feature/<slug>`
  - `fix/<slug>`
  - `chore/<slug>`
  - `hotfix/<slug>`
  - `benchmarking/lab`

릴리스는 별도 `release/*` 브랜치가 아니라 `main`에 머지된 커밋에 버전 태그(`vX.Y.Z`)를 달아 발행한다.
`codex/` 접두사는 더 이상 사용하지 않는다.

### 병합 대상

- 일반 기능/수정: `feature/*`, `fix/*`, `chore/*` -> `develop`
- 긴급 수정: `hotfix/<slug>` -> `main`, 이후 `main` -> `develop` 백머지
- 릴리스 발행: `main` 머지 후 버전 태그 생성 -> GitHub Release 작성
- 벤치마크 실험/리포트: `benchmarking/lab`에서만 유지

## 2-1. 벤치마크 자산 규칙

벤치마크는 아래 원칙을 기본으로 한다.

- benchmark harness는 가능하면 실제 offscreen 앱 파이프라인을 기준으로 만든다.
- 공식 점수 범위가 파이프라인 일부일 경우, 실행 범위와 점수 범위를 문서에 분리해 명시한다.
- Windows benchmark family는 가능하면 `pipeline + suite`, `CUDA12 + CUDA13` BAT 쌍을 함께 제공한다.
- raw 결과는 `./banchmark_result_log/<family>/` 아래에 남긴다.
- benchmark family는 최소한 아래 문서 세트를 함께 가진다.
  - workflow
  - usage
  - architecture
  - results history
  - generated/latest report
- benchmark 자산은 `benchmarking/lab`에만 두고, 제품 반영은 별도 `feature/*`, `fix/*`, `chore/*` 작업 브랜치 PR로 승격한다.

## 3. 기능 작업 절차

모든 기능 추가/수정은 아래 순서를 지킨다.

1. `develop` 기준 최신 상태 확인
2. 작업 브랜치 생성
3. 작업
4. 로컬 검증
5. 논리 단위 커밋
6. `git push -u origin <branch>` 또는 기존 업스트림으로 push
7. PR 생성 또는 기존 PR 업데이트

### 표준 명령 예시

```bash
git switch develop
git pull --ff-only origin develop
git switch -c feature/example-task
```

작업 후:

```bash
python scripts/validate_changed_python.py --all
python scripts/headless_smoke.py
python scripts/compile_translations.py --check
git status
git add <intended-files>
git commit -m "feat(workspace): add retry tooltips"
git push -u origin feature/example-task
```

## 4. 커밋 규칙

커밋 제목은 아래 형식을 기본으로 사용한다.

```text
type(scope): summary
```

허용 타입:

- `feat`
- `fix`
- `docs`
- `chore`
- `refactor`
- `test`
- `ci`
- `build`
- `perf`
- `revert`

예시:

- `feat(batch): add one-page auto action`
- `fix(ocr): reuse page-local blk_list in batch cache writes`
- `docs(repo): define git and release rules`

## 5. 완료 조건

아래를 모두 만족해야 작업이 끝난 것으로 본다.

- 올바른 작업 브랜치에서 작업했다.
- `git status`에 의도한 변경만 남아 있다.
- 빠른 검증이 통과했다.
- UI 변경 시 번역 파일과 컴파일된 `.qm`을 갱신했다.
- 기능 단위 커밋을 만들었다.
- 브랜치를 원격에 push했다.
- 병합 대상 브랜치가 맞는 PR을 열었거나 갱신했다.

## 6. 금지 사항

- `main`, `develop`에서 직접 작업
- 커밋 없이 장기간 변경 누적
- push 없이 기능 완료로 간주
- 서로 무관한 변경을 한 커밋/한 PR에 혼합
- 트래킹된 `.venv*`, `__pycache__`, 임시 DB, 캐시 파일 추가
- 번역이 필요한 UI 텍스트를 소스만 바꾸고 `.ts`/`.qm` 갱신 생략

## 7. 번역 규칙

사용자에게 보이는 텍스트를 바꾸면 아래를 반드시 수행한다.

1. 코드의 소스 문자열을 안정적인 `self.tr(...)` 또는 `QCoreApplication.translate(...)`로 유지
2. `resources/translations/ct_*.ts` 업데이트
3. `resources/translations/compiled/*.qm` 재생성
4. 최소한 다음 언어 세트 반영 확인
   - `ko`
   - `fr`
   - `zh-CN`
   - `ru`
   - `ja`
   - `de`
   - `es`
   - `it`
   - `tr`

번역 반영 명령:

```bash
python scripts/compile_translations.py
```

검증 전용:

```bash
python scripts/compile_translations.py --check
```

## 8. 로컬 Git Hooks

이 저장소는 `.githooks/`를 사용한다.

- `pre-commit`
  - 보호 브랜치 커밋 차단
  - 금지된 트래킹 경로 차단
  - 변경된 Python 파일 구문 검증
  - staged/unstaged 혼합 커밋 차단
- `commit-msg`
  - 커밋 제목 형식 검사
- `pre-push`
  - 브랜치 이름 검사
  - 잘못된 원격/업스트림 검사
  - 빠른 검증 실행

초기 설정:

```bash
bash scripts/bootstrap_git_hooks.sh
```

이 설정은 로컬 Git 설정에 `core.hooksPath=.githooks`를 기록한다.

## 9. CI / CD 규칙

### CI

CI는 필수다. 다음 항목이 통과해야 병합 가능하다.

- 브랜치 이름 규칙 검사
- PR 대상 브랜치 흐름 검사
- `main` 문서 승격 allowlist 검사
- 저장소 위생 검사
- Python 구문/컴파일 검사
- 헤드리스 스모크 검사
- 번역 자산 검사

public/free 저장소의 ruleset은 보호 브랜치, PR 강제, 상태 체크, 태그 보호를 담당한다.
브랜치 이름 강제, 브랜치 계열별 base 브랜치 적합성, 금지된 tracked 경로, benchmark 전용 자산 분리, `main` 문서 승격 allowlist 검사는 로컬 훅과 CI 정책 스크립트가 계속 담당한다.
실제 import용 ruleset JSON은 `.github/rulesets/` 아래 파일을 기준으로 관리한다.

### CD

현재 저장소에는 설치형 패키징 파이프라인이 커밋되어 있지 않으므로, CD v1은 `태그 기반 릴리스 거버넌스`에 집중한다.

- `develop`에서 충분히 검증된 변경만 `main`으로 승격
- `main` 머지 커밋에 버전 태그 생성
- GitHub Release 작성
- 필요 시 `pre-release` 표기
- `hotfix/*`는 `main` 기준으로 처리 후 `develop`에 백머지

## 10. 브랜치 보호 설정 가이드

GitHub 저장소 설정에서 아래를 권장한다.

- `main`, `develop` 직접 push 금지
- PR 필수
- CI 체크 통과 필수
- force push 금지
- 버전 태그 보호
- 관리자 예외는 `hotfix` 절차에 한정
- public/free ruleset import는 `docs/repo/github-rulesets-public-free-ko.md`와 `.github/rulesets/*.json`을 기준으로 적용한다.

## 10-1. Main 문서 승격 정책

- `main`에는 운영 필수 문서만 허용한다.
  - 루트: `README.md`, `README_ko.md`, `rules.md`
  - 변경 이력/감사: `docs/history/*.md`
  - 운영 문서: `docs/gemma/*.md`, `docs/hunyuan/*.md`, `docs/repo/github-rulesets-public-free-ko.md`, `hunyuanocr_docker_files/README.md`, `paddleocr_vl_docker_files/README.md`
- `develop`에는 개발/감사/정책 문서를 허용한다.
- `benchmarking/lab`에는 benchmark 전용 문서를 허용한다.
- 아래 문서는 `main`에 올리지 않는다.
  - `docs/i18n/*`
  - `docs/rendering/*`
  - `docs/repo/benchmark-branch-policy-ko.md`
  - benchmark/manual-review/dev-note 성격의 markdown
- 이 정책은 문서 설명만으로 두지 않고, `main` 대상 PR에서 changed markdown/doc path allowlist 검사로 강제한다.

## 11. 세션 종료 체크리스트

작업을 끝내기 전에 아래를 확인한다.

- 현재 브랜치가 유효한 작업 브랜치인가
- 의도하지 않은 변경이 없는가
- 커밋 메시지가 규칙에 맞는가
- 첫 push면 `git push -u origin <branch>`를 했는가
- 이후 push가 최신 상태인가
- PR이 열려 있거나 최신 커밋이 반영되었는가

사용자가 명시적으로 `로컬만` 원한 경우에만 push 요구를 예외로 둔다.

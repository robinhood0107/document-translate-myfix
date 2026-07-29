# Repository Rules

이 문서는 이 저장소의 Git 규칙, 브랜치 정책, CI/CD 흐름, 번역 반영 규칙의 단일 기준 문서다.
사람과 에이전트 모두 이 문서를 먼저 읽고 작업해야 하며, 이 문서와 다른 문서가 충돌하면 이 문서를 우선한다.

## 1. 기본 원칙

- `main`과 `develop`은 보호 브랜치다.
- `main`, `develop`, `benchmarking/lab`은 장기 유지 브랜치이며 삭제 대상으로 취급하지 않는다.
- 기능 개발은 반드시 별도 작업 브랜치에서만 한다.
- 트래킹되는 코드 변경 전에 먼저 브랜치를 만든다.
- 이 저장소에서는 Git worktree를 새로 만들지 않는다. 항상 `C:\path\to\comic-translate` 폴더 안에서 브랜치만 전환하며 작업한다.
- 기능 단위 작업은 `커밋`과 `push`까지 끝나야 완료로 본다.
- 커밋 author, committer, `Co-authored-by`, `Signed-off-by` 및 기타 contributor trailer에 Codex를 포함한 AI 도구를 기록하지 않는다. 커밋에는 의도한 사람 기여자 신원만 남긴다.
- 사용자에게 보이는 UI를 바꾸면 문서, 번역, 필요 시 변경 이력까지 함께 갱신한다.
- 애매한 점이 있거나 트레이드오프가 발생하면, 추천안과 근거를 함께 사용자에게 질문하고 결정한 뒤 진행한다.
- 가상환경, 캐시, 임시 산출물은 Git에 올리지 않는다.
- 폰트 바이너리(`*.ttf`, `*.otf`, `*.woff`, `*.woff2`, `*.ttc`, `*.fon`)와 루트 `fonts/` 디렉터리는 Git에 올리지 않는다.
- 테스트 원본, 실제 작품명, 사용자 로컬 절대경로, OCR/번역/인페인트/렌더 결과물, benchmark raw output은 파일 경로와 문서/테스트 내용 양쪽 모두에서 Git에 올리지 않는다.
- 로컬 작업용 가상환경은 `.venv-win`, `.venv-win-cuda13`만 공식 사용한다. `.venv`는 repo workflow 기준 환경으로 사용하지 않는다.
- 현재 공식 Windows 개발 PC에서는 공통 Python 검사와 빠른 단위 테스트를 가능한 한 `.venv-win`, `.venv-win-cuda13` 양쪽에서 실행한다. CUDA 버전에 종속된 실행·패키징 검사는 해당 환경에서 따로 수행하고 결과를 구분해 기록한다.
- 같은 checkout에서 두 Windows 환경의 Python 검사를 동시에 실행하지 않는다. `__pycache__` 파일 잠금 충돌을 피하도록 `.venv-win` 검사 후 `.venv-win-cuda13` 검사를 순차 실행하고, 필요하면 Python `-B` 옵션을 사용한다.

## 2. 브랜치 모델

이 저장소는 엄격한 `main + develop + tag` 모델을 사용한다.

- `main`
  - 배포 기준 브랜치
  - 직접 커밋, 직접 push 금지
- `develop`
  - 통합 기준 브랜치
  - 일반 기능은 모두 이 브랜치로 PR
  - 직접 커밋, 직접 push 금지
- `benchmarking/lab`
  - benchmark 전용 장기 유지 브랜치
  - benchmark preset, runner, 보고서, 차트, generated asset은 이 브랜치에만 유지
  - 직접 제품 승격 브랜치로 간주하지 않으며, 검증된 제품 변경만 별도 작업 브랜치로 다시 승격
- 작업 브랜치
  - `feature/<slug>`
  - `fix/<slug>`
  - `chore/<slug>`
  - `hotfix/<slug>`

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
- raw 결과는 repo 밖 local validation log에 남긴다. `banchmark_result_log/`, `docs/assets/benchmarking/`, 이미지/아카이브/로그 산출물은 Git에 올리지 않는다.
- benchmark family는 최소한 아래 문서 세트를 함께 가진다.
  - workflow
  - usage
  - architecture
  - results history
  - generated/latest report
- benchmark 자산은 `benchmarking/lab`에만 두고, 제품 반영은 별도 `feature/*`, `fix/*`, `chore/*` 작업 브랜치 PR로 승격한다.
- `benchmarking/lab`도 실제 샘플 이미지, 테스트 결과 이미지, OCR/번역 로그, 작품명, 사용자 로컬 경로를 보관하는 장소로 쓰지 않는다.

## 2-1-1. 민감 산출물 / 원본명 금지 규칙

아래 항목은 경로뿐 아니라 문서, 테스트 fixture, PR 설명에 내용으로도 넣지 않는다.

- 실제 작품명, 원본 파일명, 시리즈명, 사용자가 제공한 민감한 title/slug
- `C:\Users\...`, `/mnt/c/Users/...` 같은 사용자 로컬 절대경로
- `Sample/`, `testmodel/`, `build/`, `banchmark_result_log/`, `docs/assets/benchmarking/`
- `result_*`, `log_*` 형태의 자동번역/검증 출력 폴더
- `.zip`, `.cbz`, `.rar`, `.7z`, `.log` 산출물
- 앱 static/icon으로 명확히 허용된 자산을 제외한 `.png`, `.jpg`, `.jpeg`, `.webp`, `.gif`, `.bmp`, `.tiff` 미디어

테스트에는 `example_source_chapter`, `example source chapter v01 c01 (E)`, `C:\ExampleWorkspace\...`, `<validation-log-root>`, `<benchmark-log-root>` 같은 중립 fixture만 사용한다.
이 규칙은 `scripts/validate_repo_policy.py`가 커밋, push, CI에서 검사한다.

## 2-2. 성능개선/버그수정 감사 문서 규칙

성능개선, 안정화, 버그 헌팅처럼 제품 동작을 장기간 추적하며 여러 문서가 생기는 작업은 `docs/repo/`에 섞지 않는다.

- 프로젝트별 문서 루트는 `docs/performance-and-bugfix-audits/<project-slug>/`를 사용한다.
- 성능개선 프로젝트는 `00-truth-specification-ko.md`를 최상단 진실명세서로 둔다.
- 진실명세서에는 현재 확정 사실, 불변 조건, 금지선, 검증 로그 위치, 후속 문서 링크를 적는다.
- 감사, 계산, 구현 명세, PR별 설계 문서는 `01-...`, `02-...`처럼 번호 prefix를 붙여 정렬한다.
- 서로 다른 주제는 같은 폴더에 합치지 않는다. 예를 들어 자동번역 성능개선과 Runtime/UI 버그 헌팅은 별도 project slug를 사용한다.
- 이미지 결과물, OCR/번역 품질, inpaint/render/export 산출물이 바뀌는 성능개선은 테스트 이미지와 결과물을 repo 밖 validation log에 저장하고, 병합 전 사용자 검토를 요청한다.
- benchmark raw output, preset, runner, chart asset은 여전히 `benchmarking/lab` 정책을 따른다.

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

Windows launcher/runtime 변경이 포함되면 아래 검증도 추가한다.

```bash
python scripts/verify_windows_launchers.py
```

공식 Windows 릴리스는 EXE가 아니라 첫 실행 때 지원 환경을 설치하는
deterministic launcher-source ZIP이다. 릴리스 workflow, release asset,
pinned Windows 런타임, launcher 또는 `main` 승격 후보를 다루는 변경은
CI 전에 Windows 로컬에서 아래 계약을 확인한다.

- `python scripts/compile_translations.py --check`
- `python scripts/build_windows_launcher_source_bundle.py build --version "<version>" --commit HEAD --output-dir build/release`
- builder가 생성한 ZIP과 `SHA256SUMS.txt`를 `verify` subcommand로 재검증
- ZIP을 새 폴더에 추출한 뒤 `COMIC_VERIFY_ONLY=1`로 `run_comic.bat`와
  `run_comic_cuda13.bat`를 모두 실행
- `python scripts/verify_windows_launchers.py`로 기존 설치 환경의
  CUDA12/CUDA13 launcher/runtime 계약 확인

공식 asset은
`comic-translate-v<version>-windows-launcher-source.zip`과
`SHA256SUMS.txt`뿐이다. venv, 모델, 캐시, benchmark runner/raw 결과,
사용자 로컬 절대경로, secret을 포함하지 않는다. PR 본문에는 실제
Windows 명령, ZIP SHA-256, launcher-source 무설치 검사 결과를 적는다.

`scripts/build_windows_gpu_portable.ps1`과
`scripts/build_windows_gpu_onefile.ps1`은 비공식 수동 Nuitka 도구로만
남긴다. 성공 여부와 무관하게 공식 릴리스 gate나 GitHub Release asset을
구성하지 않는다.

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
- Windows launcher/runtime 또는 pinned CUDA 의존성을 건드렸다면 `.venv-win`, `.venv-win-cuda13`, `run_comic.bat`, `run_comic_cuda13.bat`를 모두 검증했다.

## 6. 금지 사항

- `main`, `develop`에서 직접 작업
- 새 Git worktree를 만들어 작업
- 커밋 없이 장기간 변경 누적
- push 없이 기능 완료로 간주
- 서로 무관한 변경을 한 커밋/한 PR에 혼합
- 트래킹된 `.venv*`, `__pycache__`, 임시 DB, 캐시 파일 추가
- 트래킹된 폰트 바이너리 또는 루트 `fonts/` 디렉터리 추가
- 테스트 원본/결과 이미지, 로그, 압축 결과물, 실제 작품명/원본명, 사용자 로컬 절대경로 추가
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

CD는 `main`에 포함된 `vX.Y.Z` 태그 기반 릴리스만 공식 경로로 인정한다.

- `develop`에서 충분히 검증된 변경만 `main`으로 승격
- release 후보는 `main` 승격 전에 Windows 로컬에서 deterministic
  launcher-source bundle과 두 launcher의 `COMIC_VERIFY_ONLY=1` 계약 확인
- `main`에 포함된 커밋에만 버전 태그(`vX.Y.Z`) 생성
- `main` 반영 후 Windows release-preflight CI로 같은 source bundle을 재현
- 해당 태그에서 launcher-source ZIP과 `SHA256SUMS.txt`를 GitHub Release
  자산으로 생성
- `develop`이나 feature 브랜치에 달린 태그는 공식 릴리스로 취급하지 않음
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
  - 루트: `AGENTS.md`, `README.md`, `README_ko.md`, `rules.md`
  - GitHub 운영: `.github/PULL_REQUEST_TEMPLATE.md`
  - 설치/운영: `docs/setup/quickstart*.md`
  - 운영 문서: `docs/gemma/*.md`, `docs/hunyuan/*.md`, `docs/repo/github-rulesets-public-free-ko.md`, `hunyuanocr_docker_files/README.md`, `paddleocr_vl_docker_files/README.md`
- `develop`에는 개발/감사/정책 문서를 허용한다.
- `benchmarking/lab`에는 benchmark 전용 문서를 허용한다.
- 아래 문서는 `main`에 올리지 않는다.
  - `docs/history/*`
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
- Windows launcher/runtime 관련 변경이면 `python scripts/verify_windows_launchers.py`를 돌렸는가
- release/main 승격 후보이면 Windows 로컬에서 deterministic
  launcher-source ZIP·SHA·추출 후 두 launcher 무설치 검사를 통과했는가
- PR이 열려 있거나 최신 커밋이 반영되었는가

사용자가 명시적으로 `로컬만` 원한 경우에만 push 요구를 예외로 둔다.

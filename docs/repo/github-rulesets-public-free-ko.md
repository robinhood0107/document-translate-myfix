# GitHub Rulesets Guide (public/free)

이 문서는 현재 `rules.md` 기준으로 public 저장소 + GitHub Free 환경에서 적용할 ruleset과 운영 기준을 정리한다.

## 1. 전제

- 저장소는 `public`이어야 한다.
- 브랜치 모델은 `main + develop + tag`다.
- 릴리스는 `main`에 머지된 커밋에 버전 태그를 달아 GitHub Release로 발행한다.
- `release/*` 브랜치는 사용하지 않는다.

## 2. ruleset이 담당하는 것

- `main` 보호
- `develop` 보호
- 버전 태그 보호
- PR 필수
- required status checks
- force-push / delete 방지

## 3. ruleset이 담당하지 않는 것

아래 항목은 ruleset이 아니라 CI/훅이 담당한다.

- 브랜치 이름 형식 강제
- `feature/*`, `fix/*`, `chore/*` -> `develop` 흐름 강제
- `hotfix/*` -> `main` 흐름 강제
- `benchmarking/lab` -> 제품 브랜치 PR 금지
- 금지된 tracked 경로 차단
- `main` 문서 승격 allowlist 검사

관련 스크립트:

- `scripts/validate_repo_policy.py`
- `scripts/validate_pr_flow.py`
- `scripts/validate_main_docs_policy.py`

## 4. import할 JSON 파일

- `.github/rulesets/01-protect-main.json`
- `.github/rulesets/02-protect-develop.json`
- `.github/rulesets/03-protect-version-tags.json`

## 5. GitHub에서 실제로 하는 순서

1. 저장소를 `public`으로 둔다.
2. `Settings -> Actions`에서 GitHub Actions를 허용한다.
3. PR을 한 번 돌려 아래 체크 이름이 실제로 생성되게 한다.
   - `branch-name`
   - `validate-pr-flow`
   - `validate-main-docs`
   - `validate`
4. `Settings -> Rules -> Rulesets -> New ruleset -> Import a ruleset`에서 위 JSON 3개를 각각 import한다.
5. import 후 아래를 확인한다.
   - `Protect main`: target `main`
   - `Protect develop`: target `develop`
   - `Protect version tags`: target `v*`
6. `Protect main`의 required status checks는 아래 4개여야 한다.
   - `branch-name`
   - `validate-pr-flow`
   - `validate-main-docs`
   - `validate`
7. `Protect develop`의 required status checks는 아래 3개여야 한다.
   - `branch-name`
   - `validate-pr-flow`
   - `validate`

## 6. 브랜치 이름 정책

브랜치 이름은 ruleset이 아니라 CI와 훅이 강제한다.

- `feature/<slug>`
- `fix/<slug>`
- `chore/<slug>`
- `hotfix/<slug>`
- `benchmarking/lab`

## 7. `main` 문서 승격 정책

`main`에는 운영 필수 문서만 허용한다.

- 허용
  - `README.md`
  - `README_ko.md`
  - `rules.md`
  - `docs/history/*.md`
  - `docs/gemma/*.md`
  - `docs/hunyuan/*.md`
  - `docs/repo/github-rulesets-public-free-ko.md`
  - `hunyuanocr_docker_files/README.md`
  - `paddleocr_vl_docker_files/README.md`
- 비허용
  - `docs/i18n/*`
  - `docs/rendering/*`
  - `docs/repo/benchmark-branch-policy-ko.md`
  - benchmark/manual-review/dev-note 성격의 markdown

## 8. 공식 기준

- Rulesets are available in public repositories with GitHub Free.
- Import a ruleset is an official repository rulesets feature.
- Require status checks to pass is an official branch ruleset rule.
- GitHub Releases are tag-based, so a `release/*` branch is not required.

문서:

- https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-rulesets/about-rulesets
- https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-rulesets/managing-rulesets-for-a-repository
- https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-rulesets/available-rules-for-rulesets
- https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases

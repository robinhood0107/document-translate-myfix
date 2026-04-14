# 문제 해결 명세서 03 - Measurement Execution Protocol

핵심 문제 해결 방향은 사용자가 착안했다.

## 문제

실측 자체가 흔들리면, 시간이 빨라졌는지 느려졌는지보다 “무엇을 같은 조건으로 재봤는지”가 먼저 무너지게 된다.

## 사용자 착안 요약

사용자는 Docker 재기동 시간, healthcheck 대기 시간, VRAM 여유, `ngl` 증가 가능성, timeout, 실제 stage 처리 시간을 모두 체크포인트 단위로 쪼개서 근거를 남기라고 요구했다.

## 실행 프로토콜

### 1. 환경

- 공식 우선 환경: `.venv-win-cuda13`
- 동반 환경: `.venv-win`
- source language: `Japanese`
- target language: `Korean`
- corpus root: `Sample/japan`

### 2. 입력 세트

- smoke: `094.png`, `p_016.jpg`
- 공식 13장:
  - `094.png`
  - `097.png`
  - `101.png`
  - `i_099.jpg`
  - `i_100.jpg`
  - `i_102.jpg`
  - `i_105.jpg`
  - `p_016.jpg`
  - `p_017.jpg`
  - `p_018.jpg`
  - `p_019.jpg`
  - `p_020.jpg`
  - `p_021.jpg`

### 3. 공식 시나리오

1. `baseline_legacy`
2. `candidate_stage_batched_single_ocr`
3. `candidate_stage_batched_dual_resident`

### 4. 필수 기록 파일

- `benchmark_request.json`
- `events.jsonl`
- `timing_summary.json`
- `quality_summary.json`
- `vram_snapshots.jsonl`
- `docker_timeline.json`
- `report.md`

### 5. 필수 비교 항목

- 총 시간
- 순수 처리 시간
- compose up 시간
- health wait 시간
- reuse hit
- timeout/retry
- detect box count
- OCR non-empty / empty / single-char-like
- `p_016.jpg` 난페이지 상태
- page failure 수

## 현재 구현 수준

- baseline은 실제 offscreen app pipeline으로 실행 가능하다.
- runtime progress가 `metrics.jsonl`에 남도록 memlog surface를 연결했다.
- candidate 두 시나리오는 아직 blocked 계약 상태다.

## 해석

이 프로토콜은 “실측 가능한 부분은 즉시 시작하고, 아직 없는 stage-batched runner는 거짓 없이 blocked로 기록한다”는 원칙을 따른다. 이렇게 해야 근거가 없는 속도 주장이나 품질 주장을 피할 수 있다.

## 기대 효과

- Requirement 1 실측 과정이 체크리스트와 같은 언어로 굴러간다.
- Docker / VRAM / 품질 근거가 raw 파일과 narrative 문서 양쪽에 동시에 남는다.
- 나중에 Requirement 2 사용자 검수 패키지가 같은 기록 surface를 재사용할 수 있다.

## 다음 액션

1. smoke baseline 실행
2. baseline full run 실행
3. stage-batched runner 구현 후 candidate 두 시나리오 실측
4. 결과 이력과 generated report 갱신

## 저자 및 기여

- Idea Origin: User
- Planning / Measurement Design / Implementation Detailing / Validation: Collaborative
- Execution Support: Codex

# COO SFX shadow 결과 이력

## 공식 자산 고정

- source commit: d8028f015b8ce99a4dd798427342f97087529357
- ABCNetv2 checkpoint SHA-256:
  25d33d9dc033a65c888e99ef25dbdfadd5b2ae7bf8d3b18e8e85a093956ea6e2
- 역사적 runtime: PyTorch 1.9.0, CUDA 11.1, Detectron2 0.5
- 제품 상태: benchmark-only

공식 COO annotation과 저장소 자체 코드는 각각 CC BY 4.0과 MIT 안내가 있지만,
실제 ABCNetv2 구현에는 학술 목적 조건이 있어 제품 번들링·자동 다운로드는 라이선스
확인 전 금지한다.

## 중립 개발 6페이지

| threshold | polygon 수 | 해석 |
|---:|---:|---|
| 0.2 | 166 | 작은 SFX recall은 늘지만 UI·비SFX 추가 신호도 크게 증가 |
| 0.3 | 112 | 공식 detection 기준, 현재 개발 기본값 |
| 0.5 | 59 | 작은 SFX가 다수 사라져 보수적이지만 recall 손실이 큼 |

threshold 0.3의 같은 checkpoint·입력에 대한 장치 비교:

| 항목 | CPU 참고 | CUDA 공식 |
|---|---:|---:|
| model load | 0.516초 | 4.825초 |
| 6페이지 inference | 6.440초 | 1.404초 |
| process total | 7.953초 | 7.359초 |
| peak reserved VRAM | - | 714.0 MiB |

- inference 단축: 78.203%
- process total 단축: 7.467%
- 페이지별 region 수: 전부 동일
- matched bbox 최저 IoU: 0.999 이상
- 최대 좌표 차이: 0.035px 미만
- 페이지 경계 clip: 양쪽 동일 1개 region
- 실행 전후 WSL swap 증가: 없음

짧은 세트에서는 CUDA 초기화 때문에 총시간 이득이 작지만 페이지가 늘수록 inference
차이가 누적된다. 따라서 이후 COO 실험은 CUDA를 기본으로 하고 CPU는 geometry
검산에만 사용한다.

## Japan 22페이지 source-first truth

Codex가 후보 출력을 보기 전에 원본 22페이지 317영역을 잠근 뒤, 같은 공식 CUDA
checkpoint를 threshold 0.2·0.3·0.5·0.60·0.65·0.70·0.75로 실행했다. 각 실행은
22/22페이지를 완료했고 peak reserved VRAM은 714.0 MiB였다.

| threshold | COO region | SFX/장식 11건 중 신호 | 의미 텍스트 신호 | 판정 |
|---:|---:|---:|---:|---|
| 0.20 | 571 | 5 | 9 | 의미 텍스트 오경고 과다 |
| 0.30 | 394 | 5 | 4 | 의미 텍스트 오경고 |
| 0.50 | 193 | 5 | 1 | 안전 분리 실패 |
| 0.60 | 103 | 1 | 1 | 안전 분리 실패 |
| 0.65 | 52 | 0 | 1 | SFX 보호 이득 없음 |
| 0.70 | 22 | 0 | 0 | SFX 보호 이득 없음 |
| 0.75 | 2 | 0 | 0 | SFX 보호 이득 없음 |

threshold 0.5의 오경고는 `096`의 말풍선 밖 사람 발성 `あ～ん`이었다. COO score는
0.696으로 실제 SFX score 범위와 겹쳤고, 크기·비율·말풍선 밖 여부 같은 geometry
조건으로도 안전하게 분리되지 않았다. polygon overlap 임계값을 0.05~0.50으로
바꿔도 결과의 본질은 같았다.

Paddle crop과 Paddle Spotting은 기준선에서 위험한 SFX/장식 자동 편집이 11건이었고
COO가 최대 5건을 review로 낮췄다. MangaLMM은 기준선 위험 편집이 0건이므로 COO가
줄일 오류가 없었다. COO는 어느 실험에서도 자동 preserve나 자동 삭제 권한을 갖지
않았으며 의미 텍스트 auto-hidden은 0건이었다.

최종 판정은 `reject_no_safe_operating_point`다. benchmark-only 재현 자산은
보존하지만 세 OCR 제품 경로에는 적용하지 않는다.

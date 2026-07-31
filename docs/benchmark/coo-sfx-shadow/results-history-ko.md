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

아직 source-first truth가 잠기지 않았으므로 SFX recall, 의미 대사 오경고,
실제 파괴 방지 수는 판정하지 않았다.

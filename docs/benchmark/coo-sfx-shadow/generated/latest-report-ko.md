# COO SFX shadow latest

## 현재 판정

- 공식 ABCNetv2 checkpoint와 source commit을 고정했다.
- checkpoint는 전용 named volume에 SHA 검증·atomic rename·ready manifest 순서로
  준비했다.
- RTX 4070 SUPER CUDA 추론은 실제로 성공했다.
- 중립 6페이지에서 CPU/CUDA region 수가 같고 좌표는 0.035px 미만 차이였다.
- 페이지 밖으로 조금 확장된 1개 region은 양쪽 모두 같은 페이지 경계로 clip됐다.
- CUDA inference는 CPU 참고 경로보다 78.203% 빨랐지만 초기화를 포함한 total은
  7.467% 빨랐다.
- CUDA peak reserved VRAM은 714.0 MiB였다.
- 공식 detection threshold 0.3을 개발 기본값으로 유지한다.

## 승격 상태

COO는 제품에 승격하지 않는다. 현재는 다음 이유로 benchmark-only다.

1. 세 OCR source-first truth가 아직 잠기지 않아 사람 기준 SFX 보호 이득이 없다.
2. 의미 있는 text_free를 review로 보내는 비용과 실제 파괴 방지 수를 비교하지 못했다.
3. ABCNetv2 제품 사용 라이선스 확인이 끝나지 않았다.

향후 평가에서도 COO는 자동 preserve나 OCR 교정에 쓰지 않고, 말풍선 밖 후보를
review로 낮추는 shadow 신호로만 비교한다.

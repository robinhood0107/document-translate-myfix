# COO SFX shadow 워크플로

이 벤치마크는 일본어 의성어·의태어를 별도 OCR로 번역하기 위한 것이 아니다.
COO ABCNetv2 polygon을 세 OCR 경로의 결과 위에 겹쳐, 잘못 번역·인페인트될 SFX를
사람 검토로 돌릴 가치가 있는지만 측정한다.

## 고정 순서

1. 공식 COO source commit과 ABCNetv2 checkpoint SHA-256을 고정한다.
2. checkpoint는 전용 named volume에 atomic copy하고 ready manifest를 마지막에 쓴다.
3. 같은 source SHA·threshold·image digest로 CPU와 CUDA 결과를 Git 밖에 만든다.
4. compare-devices로 region 수, bbox IoU, 좌표 차이, 시간, CUDA peak memory를
   검증한다.
5. 세 OCR source-first truth가 잠긴 뒤에만 evaluate-shadow를 실행한다.
6. COO 신호는 말풍선 밖 detector block을 review로 낮추는 데만 사용한다.
7. 자동 preserve, 자동 삭제, OCR text 교체는 하지 않는다.
8. SFX 파괴 방지 이득이 없거나 의미 대사를 과도하게 review로 보내면 제품 경로에서
   제거한다.

현재 제품 적용은 금지되어 있다. 이유는 사람 정답 평가가 아직 잠기지 않았고,
ABCNetv2의 제품 사용 라이선스도 별도 확인이 필요하기 때문이다.

공식 자료:

- [COO repository](https://github.com/ku21fan/COO-Comic-Onomatopoeia)
- [ABCNetv2 implementation](https://github.com/aim-uofa/AdelaiDet/tree/master/configs/BAText)

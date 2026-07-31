# COO SFX shadow 아키텍처

## 경계

~~~text
official COO checkpoint + external inference
  → Git-external polygon prediction JSON
  → benchmark_coo_sfx_shadow.py
       ├─ model/source/image/source-page identity 검증
       ├─ CPU/CUDA geometry·속도 비교
       └─ locked truth + normalized OCR run 비교
  → report-only preserve/review 진단
~~~

제품 OCR·semantic routing·인페인트 코드는 이 벤치마크에 의존하지 않는다.
checkpoint, Docker image, polygon overlay, 원본 파일은 저장소에 넣지 않는다.

## 안전 정책

- COO polygon은 OCR text를 제공하거나 교체하지 않는다.
- 말풍선 내부 영역은 COO만으로 처리 상태를 바꾸지 않는다.
- 말풍선 밖 신호도 자동 preserve가 아니라 review로만 낮춘다.
- 의미 있는 text_free가 COO와 겹쳐도 자동 숨김은 0이다.
- 제품 적용은 기존 세 경로에서 발생한 SFX 파괴 후보를 실제로 줄였다는 잠긴 정답
  통계와 별도 라이선스 확인이 모두 있어야 한다.

## 장치 계약

공식 설정은 CUDA SyncBatchNorm이다. CPU 참고 경로는 이 역사적 모델의 GPU 전용
SyncBatchNorm을 eval running-stat 기반 BN으로 바꾼다. 따라서 CPU/CUDA 비교는
geometry 동등성과 성능 손익분기점을 확인하는 용도이며 CPU가 제품 후보는 아니다.

CUDA 통과 조건은 region 수 동일, greedy bbox match 최저 IoU 0.999 이상,
좌표 최대 차이 0.1px 이하이다. 이 조건은 장치 수치 동등성이지 SFX 품질 승격 조건이
아니다.

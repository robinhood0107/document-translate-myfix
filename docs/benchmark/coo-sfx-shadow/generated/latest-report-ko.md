# COO SFX shadow latest

## 최종 판정

COO는 세 OCR 제품 경로 어디에도 적용하지 않는다. 공식 ABCNetv2 CUDA 실행 자체는
정상이고 빠르지만, Japan 22페이지의 잠긴 source-first truth에서 SFX를 안전하게
구분하는 threshold가 존재하지 않았다.

- threshold 0.5에서 기존 Paddle crop/Spotting의 위험한 SFX·장식 편집 11건 중
  5건을 review로 낮췄다.
- 같은 설정이 의미 있는 자유 대사 1건(`096`의 `あ～ん`)도 review로 낮췄다.
- threshold 0.6에서도 SFX 1건과 의미 대사 1건이 함께 걸렸다.
- threshold 0.65 이상에서는 의미 대사 오경고는 사라졌지만 SFX 보호 이득도 0건이었다.
- polygon overlap 임계값 0.05~0.50을 바꿔도 안전한 분리점은 나오지 않았다.
- MangaLMM 경로는 기준선의 위험한 SFX 자동 편집이 이미 0건이어서 COO의 제품 이득이
  없었다.

COO를 자동 preserve에 사용한 적은 없고 모든 실험은 report-only review 신호였다.
그 상태에서도 의미 대사를 불필요하게 review로 보내므로, 제품 복잡도·추가 모델
시작·라이선스 부담을 감수할 가치가 없다.

## CUDA 실행 증거

- 공식 checkpoint/source/image를 고정한 전용 named volume 사용
- RTX 4070 SUPER에서 Japan 22페이지를 threshold 0.2~0.75로 실제 실행
- 실행당 22/22페이지 완료, peak reserved VRAM 714.0 MiB
- 중립 6페이지 CPU/CUDA region 수 동일, 최대 좌표 차이 0.035px 미만
- CUDA inference는 CPU 참고보다 78.203% 빨랐고 초기화 포함 total은 7.467% 빨랐음

이는 CUDA 구현이 정상임을 증명하지만 SFX 분류의 제품 유용성을 증명하지 않는다.
코드·문서·raw 결과는 benchmark 재현 자산으로만 남기고, 제품 runtime·routing·cache
identity에는 COO를 추가하지 않는다.

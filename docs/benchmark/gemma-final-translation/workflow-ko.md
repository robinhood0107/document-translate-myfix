# Gemma 최종 번역 전용 비교 workflow

이 family는 동결된 22개 입력과 페이지 순서를 보존한 292블록 OCR snapshot으로
당시 요청 방식, grouped F16, grouped Q8 세 후보를 비교한 역사적 protocol v3다.
OCR, 인페인트, 렌더는 점수 범위에 포함하지 않았다.

제품의 `contextual-grouped` 실행 경로는 품질 탈락 뒤 퇴역했다. 현재
checkout에서 이 runner를 실행하면 후보 이름과 실제 요청 방식이 달라질 수
있으므로 preflight부터 명시적으로 차단한다. 원본 live suite 재현 대상은
커밋 `76b81c7b903bd9569d116b5eabc966135a13a1f5`이며, 현재 보존 결과를
검수할 때는 새 모델 요청을 만들지 않는 protocol v4 report-only 도구를
사용한다.

## 순서

1. 입력 22개의 이름, 정렬 순서, 크기, SHA-256을 고정한다.
2. OCR snapshot의 22페이지, 292블록, 페이지별 순서를 입력 manifest와
   대조한다. 보관 과정에서 다시 인코딩된 이미지는 lossless decoded pixel
   equality 또는 잠긴 JPEG 오차 한도를 추가로 통과해야 한다.
3. 세 컨테이너의 image ID, model, context, GPU layer, thread, KV, speculative
   설정을 역할별 계약과 비교한 뒤 검증된 container ID로만 정지한다.
4. host 모델과 read-only Docker volume 모델의 전체 SHA-256을 대조한다.
5. Docker engine과 NVIDIA driver identity를 기록한다.
6. Latin-square 순서로 1회차 baseline → F16 → Q8, 2회차 F16 → Q8 →
   baseline을 실행한다.
7. 후보 내부 편차가 5%를 넘거나 F16/Q8 차이가 5% 미만이면 3회차 Q8 →
   baseline → F16을 실행한다. 세 번이면 각 후보가 first/middle/last를 한 번씩
   차지한다.
8. 292/292 순서, 빈 값, 구조 토큰, decoder telemetry, 정상 logical/HTTP 및
   mode별 request 수를 검사한다.
9. 설정명과 속도를 숨긴 A/B/C 품질 검수 파일을 생성한다.
10. 사용자 품질 승인 전에는 22페이지 전체 파이프라인을 실행하지 않는다.

각 후보 컨테이너는 `start`로 시작하고 측정 후 `stop`한다. 정상 흐름에서
컨테이너, volume, 모델 파일을 삭제하지 않는다.

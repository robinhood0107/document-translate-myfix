# 세 OCR 수렴: MangaLMM 현재 HEAD 재생 판정

## 목적

이 문서는 과거 MangaLMM 결과를 현재 제품 구현처럼 오인하지 않도록,
현재 `develop`의 full-page parser와 reconciliation을 실제 Japan 22페이지
응답에 다시 적용한 중간 판정을 고정한다. 원본 이미지, raw 응답, 사람
정답, blind CSV는 Git 밖 검증 폴더에만 보관한다.

## 잠금 정답

- 페이지: Japan 22페이지
- detector 영역: 311개
- 사람이 원본에서 추가한 의미 영역: 6개
- 전체 truth 영역: 317개
- truth contract SHA-256:
  `74478c564e766a15846479871f7ccd49d49a83da7db2d210ca35217f4613af35`
- truth는 후보 결과를 보기 전에 잠갔으며, 공백·줄바꿈·문장부호는
  transcription 정확도에서 제외한다.

## 현재 HEAD live 실행과 오프라인 재생

- 22/22페이지가 구조적으로 종료됐다.
- live detector geometry는 잠금 snapshot의 311개 block과 순서·좌표가
  모두 같았다.
- 선택된 raw 응답 523개를 현재 product reconciliation으로 재생했다.
- detector에 결과가 배정된 block: 234/311
- shadow/review region: 279개
- near-duplicate로 접은 raw region: 10개
- 최종 canonical unit: 513개
- request attempt: 27회, retry 5회
- parser error가 기록된 attempt: 5회
- `finish_reason=length`: 3회

이번 실행은 enrich 전 debug runner로 실행되어 페이지별 elapsed time이
저장되지 않았다. 따라서 이 재생 결과의 `elapsed_seconds=0`은 속도 0초가
아니라 timing evidence unavailable을 뜻한다.

## 자동 텍스트 지표

아래 수치는 사람 의미 판정이 아니라 잠금 transcription과의 자동 비교다.

| 지표 | 현재 HEAD MangaLMM |
|---|---:|
| truth에 text가 배정된 영역 | 240/317 |
| normalized exact | 119/317 (37.54%) |
| normalized character accuracy | 56.11% |
| `translate_inpaint` 영역 text coverage | 153/187 (81.82%) |

과거 보고서의 MangaLMM 사람 의미 정확도 151/187과 위 153/187 coverage는
같은 수치가 아니다. coverage는 글자가 무엇이든 배정되면 올라가므로 품질
승격 근거로 사용하지 않는다.

## 과거 normalized 결과와의 차이

317개 truth 중 29개 출력이 달라졌다.

- 개선 예: 과거 누락이던 `教わったこと早速実践してみよっか`,
  `ズリいズリい`가 현재 detector block에 배정됐다.
- 회귀 예: 의미 대사 `私もですか？`, `舌まで動かせるのか…`,
  `使用済みティッシュを入れるだけで`, `同じおまんこなんだから簡単でしょう？`,
  `ごめんね彼氏君` 등이 안전 판정에서 비워졌다.
- 주된 원인 1: 한 detector block 안의 본문 region과 작은 SFX/noise region을
  함께 compound하지 못하면 본문까지 전부 fail-closed한다.
- 주된 원인 2: MangaLMM region 하나가 detector block 두 개의 연속 문장을
  포함할 때 현재 코드는 텍스트 복제를 피하기 위해 둘 다 review로 남긴다.
- 주된 원인 3: `p_016`처럼 모델 자체가 페이지 의미 텍스트를 거의 내지
  않은 경우 reconciliation만으로 복구할 수 없다.

따라서 PR #215는 안전한 N:1 compound 기반을 만든 유효한 변경이지만,
MangaLMM 품질 수렴의 완료 판정은 아니다.

## COO CUDA 확인

COO는 제품 OCR이 아니라 일본어 SFX의 `preserve/review`용 shadow 신호다.

- CPU와 CUDA 모두 112개 region을 출력했다.
- 최소 bbox IoU: `0.9991657844735014`
- 최대 좌표 차이: `0.03472595170342174 px`
- CUDA inference: CPU보다 78.2041% 단축
- CUDA 전체 프로세스: CPU보다 7.4673% 단축
- CUDA 추가 VRAM: 약 714 MiB
- 실행 중 WSL swap 증가: 0

좌표와 검출 수가 실질적으로 같으므로 향후 COO shadow 실험은 CUDA를
사용할 수 있다. 다만 학술 라이선스와 의미 대사 false-negative 게이트는
그대로 유지한다.

## 새 benchmark 도구

- `scripts/debug_mangalmm_fullpage_ocr.py`
  - 다음 실행부터 source SHA, elapsed, block ID, role/action, raw/shadow
    regions, merge/split 진단을 함께 보존한다.
- `scripts/benchmark_ocr_three_way_convergence.py replay-manga-debug`
  - Docker나 모델 요청 없이 선택된 raw 응답을 현재 parser/reconciliation에
    재생하고 잠금 normalized run을 만든다.
- `scripts/benchmark_ocr_three_way_convergence.py transfer-decisions`
  - 과거 완료 review에서 후보 text와 geometry가 그대로인 판정만 route
    identity 기준으로 새 blind review에 옮긴다. blind label이 바뀌어도
    route를 혼동하지 않는다.

외부 산출물:

- live debug:
  `<validation-root>/ocr-three-way-final/20260731_current_head_manga_japan22_v1`
- current normalized replay:
  `<validation-root>/ocr-three-way-final/20260731_current_head_manga_japan22_replay_v2`
- delta blind review:
  `<validation-root>/ocr-three-way-final/20260731_current_head_three_way_review_v2`

## 다음 제품 수정 경계

1. 한 block 안의 본문+작은 SFX 충돌은 본문을 지우지 않는 dominant-region
   선택 후보로 shadow 검증한다.
2. 한 Manga region이 여러 detector block을 덮는 경우 텍스트 복제나 임의
   분할을 하지 않는다. 같은 bubble·읽기 순서·연속 mask가 증명될 때만
   downstream translation/inpaint/render까지 하나의 compound group으로
   처리한다.
3. 모델 출력 자체가 없는 영역은 reconciliation threshold로 만들지 않는다.
   image profile·prompt/repeat 설정 실험과 분리한다.
4. 현재 delta review의 바뀐 MangaLMM 출력만 원본에서 재판독한다. Paddle
   crop과 Spotting의 동일 evidence 판정은 과거 완료 review에서 승계한다.

Paddle crop 기본값은 변경하지 않는다. 사용자에게 최종 A/B/C 검수를
요청하는 시점은 parser/length 구조 실패가 0이고, 위 회귀가 해결되거나
명시적인 review로 안전하게 남은 뒤다.

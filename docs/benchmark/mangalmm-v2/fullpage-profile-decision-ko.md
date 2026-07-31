# MangaLMM full-page profile 개발 판정

## 결론

MangaLMM 공식 full-page OCR 방향과 공식 prompt는 유효하지만, 비교한 두
profile 모두 필수 의미 영역 coverage 게이트를 통과하지 못했다. MangaLMM
제품 선택지 노출, COO 실험, 잠금 holdout과 최종 46페이지 비교는 이
개발 게이트에서 중단한다.

- 자동 판정: `no_profile_passed_required_coverage`
- development recall 선두: `standard-1728-8192`
- 제품 승격: 없음
- PaddleOCR-VL 기본값: 유지

## 고정 계약

- 입력: full-page PNG
- prompt: MangaLMM 공식 OCR grounding prompt
- completion: 4096
- profile:
  - standard: 1224×1728, context 8192
  - high: 1451×2048, context 12288
- 순서: round 1 standard→high, round 2 high→standard
- detector geometry와 사람 annotation은 후보 실행 전에 SHA-256으로 고정
- GPU 사전검사: 2048 MiB 이하
- cache·Paddle fallback·crop·tile: 사용하지 않음
- 실행 후 managed container는 정상 `stop`

## 검증 결과

- 결과 SHA-256:
  `9e97a10598d0b1229094109449b03b80afed2ea4b4e8d89ef5869b1e1f57ecdb`
- 자동 판정 SHA-256:
  `fef3120c42bc17f6cd8788bc668139ee2a5db65bde86bf79bf4ac2ef7ba312da`
- raw response: 8/8 해시 검증
- overlay: 8/8 해시 검증
- blind overlay: 8/8 해시 검증
- blind review와 unblind key: 해시·상호 참조 검증
- parser 오류: 0
- `finish_reason=length`: 0

| profile | 필수 영역 | 정규화 exact | request 합계 | request 중앙값 |
|---|---:|---:|---:|---:|
| `standard-1728-8192` | 26/34 | 18/34 | 89.835초 | 22.430초 |
| `high-2048-12288` | 22/34 | 16/34 | 83.849초 | 20.967초 |

반투명 화면 난제에서 standard는 매 라운드 5/9, high는 3/9만 찾았다.
다른 개발 페이지에서는 둘 다 8/8을 찾았지만 정규화 exact는 매 라운드
6/8이었다. 이 차이에는 경미한 장음 표기 차이뿐 아니라 단어 의미가
달라지는 OCR 오류도 포함됐다.

high profile은 request 합계가 standard보다 약 6.66% 짧았지만 필수 영역
recall이 더 낮다. 품질 우선 규칙에 따라 속도 우위는 고려하지 않는다.

## 무엇이 확인됐는가

1. 과거 제품 prompt에서 난제 페이지가 EOS 1 token으로 끝난 문제는 공식
   prompt를 사용하자 사라졌다.
2. 단순히 입력 해상도와 context를 높이면 recall이 좋아진다는 가정은
   성립하지 않았다.
3. standard는 화면 밖 의미 text-free를 high보다 더 많이 찾았지만,
   핵심 반투명 대사와 짧은 대사를 계속 놓쳤다.
4. exact duplicate region이 반복해서 출력되므로 후속 canonicalization은
   필요하지만, dedupe만으로 누락된 의미 영역을 복원할 수는 없다.

## 다음 허용 작업

- 이 runner와 탈락 증거는 `benchmarking/lab`에 보존한다.
- 공식 prompt 수정은 별도 제품 PR에서 parser·기존 Manga 동작 회귀가
  없는지 검토할 수 있다.
- MangaLMM을 다시 제품 후보로 열려면 새 모델·공식 upstream 변경처럼
  coverage를 직접 개선할 독립 근거가 필요하다.
- 동일 profile threshold를 이 두 개발 페이지에 맞춰 조정하거나
  crop·tile·Paddle fallback을 추가하는 재시험은 하지 않는다.

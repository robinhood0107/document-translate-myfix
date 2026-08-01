# Optimal++ v1.3.0 완전 실험 등록부

이 등록부는 v1.3.0까지 실제로 비교한 후보를 빠짐없이 기록한다. `채택`은
제품 기본 동작에 반영된 경우, `유지`는 이미 안전한 기존 동작을 보존한 경우,
`탈락`은 품질 또는 반복 측정 조건을 통과하지 못한 경우다. benchmark의 원시
입력·출력·이미지·세부 telemetry는 Git 밖의 `<validation-log-root>`에 있다.

## 파이프라인과 OCR

| 실험 | 핵심 결과 | 판정 | 제품 상태 |
|---|---|---|---|
| Stage-Batched 순차 파이프라인 | legacy 995.846초, 단일 OCR 순차 경로 714.725초, dual-resident 1664.021초 | 동시 상주가 크게 느리고 VRAM 압박도 컸다. | 순차 stage handoff 유지 |
| Paddle detector + crop OCR | 일본어 22페이지 구조 22/22, 의미 텍스트 recall 177/187 | 세 전략 중 최선의 전체 균형 | 기본값 |
| Paddle direct llama.cpp crop | 일본어 22페이지 정규화 OCR 311/311 동일, 직접 요청 23.970초 | 중계 계층 대비 결과 동일·속도 우위 | 채택 |
| Paddle source-aware prewarm | AB/BA 4.004%, 2.961%, 평균/중앙 3.482% 단축 | 현재 폴더 miss가 있을 때만 유효 | 채택 |
| Paddle full-page Spotting | 구조 22/22, 의미 recall 156/187, merge/split 10 | 좌표는 유효하나 line/block 차이와 추가 영역이 많다. | Experimental |
| MangaLMM full-page | 구조 22/22, 의미 recall 151/187, merge/split 30 | full-page 인식은 가능하나 기본 OCR보다 안정성이 낮다. | Experimental |
| MangaLMM 반복 출력 복구 | 반복 창 4096, penalty 1.15에서 긴 출력 잘림이 줄었다. | 구조 안정화에는 유효하나 기본값 승격 근거는 부족 | Experimental 유지 |
| COO SFX shadow | 낮은 threshold는 의미 대사 false review, 높은 threshold는 보호 효과 0 | 안전하게 쓸 수 있는 operating point가 없었다. | 미사용 |
| PP-OCRv6 최고 품질 비교 | 속도는 유리했지만 일본어 대사 누락·오인식이 더 많았다. | 품질 게이트 탈락 | PaddleOCR-VL 유지 |
| Paddle derived image | 반복 시작 개선 1.101초 | 목표 대비 이득이 작고 새 image 유지 비용이 크다. | 탈락 |
| folder-global OCR queue w4 | 결과 동일, 관측 2.8904% 개선 | 반복 측정 신뢰구간 하한 −0.276427% | 동률·미승격 |
| OCR worker 6 | 0.026% 관측 이득 | 노이즈 수준 | 미승격 |
| completion token 768 | 9/9 구조 동일, 평균 0.057%, 중앙 −0.010%, 하한 −1.674% | 실질 이득 미입증 | 1024 유지 |
| 과거 vLLM 전용 batch/sequence/prefix 후보 | 각각 0.970%, 0.585%, 0.865% 관측 이득 | 관리형 vLLM을 퇴역했으므로 직접 llama.cpp 후보가 아니다. | 재시험 안 함 |
| detector batch 2/4 | detector box가 변했다. | geometry 계약 위반 | 탈락 |
| direct transport 이전 HTTP Session 재사용 | Paddle 3.653% 느림 | 결과 유지여도 속도 악화 | 탈락 |

## Gemma 번역

| 실험 | 핵심 결과 | 판정 | 제품 상태 |
|---|---|---|---|
| contextual-grouped + group 7 + no-spec + F16 | 번역 전용 약 40.057% 빠름, 292행 blind에서 21행·36출력 의미 회귀 | 속도가 품질 회귀를 상쇄하지 못한다. | 퇴역 |
| Q8 KV | F16보다 4.730% 느리고 fallback·invalid 발생 | 구조/속도 모두 불리 | 탈락 |
| IQ4_XS | 초기 명목 0.9354% 빠름, 최종 request 평균 1.527%지만 하한 −2.026%, candidate-only 의미 회귀 2행·5출력 | 우위 미입증 및 품질 게이트 탈락 | IQ4_NL 유지 |
| QAT 모델군 | no-spec probe 50.93% 느림, 54 block 비교에서 54.90% 느림, 품질 우위 없음 | 기본 모델을 바꿀 근거 없음 | 탈락 |
| MTP/Speculative 모델군 | 대표 후보가 각각 12.2985%, 41.1746% 느리고 GPU draft 호환 오류도 발생 | target offload·TTFT 부담이 더 컸다. | 탈락 |
| ngram speculative decoding | 반복 측정상 총시간 우위 없음 | 작은 수치도 검증되지 않음 | 탈락 |
| batch 1024 및 추가 batch/ubatch | 구조 통과 후보가 있었으나 최종 의미 검수 또는 누적 이득을 통과하지 못했다. | 안전한 현행 `2048/512` 유지 | 미승격 |
| batch 4096 | 속도 후보였으나 candidate-only 의미 회귀 | 구조만으로 승격하지 않는다. | 탈락 |
| chunk 9/12 | 각각 0.731%, 1.357% 관측 이득 | 최종 의미 게이트를 통과한 누적 우승 후보가 없다. | chunk 6 유지 |
| cache RAM 256 MiB | E2E 1.915%, request 2.105% 단축이었으나 candidate-only 의미 회귀 1건 | 속도가 좋아도 의미 게이트를 통과하지 못함 | 0 MiB 유지 |
| `np=2` | 실질적인 총시간 이득 없음 | 병렬 slot 비용만 늘었다. | 탈락 |
| 명시적 GGUF read-ahead | 하한 −0.201%, swap 증가 | 자연 page cache를 방해할 위험 | 탈락 |
| 자연 OS page cache | healthy 66.811초 → 즉시 재시작 16.865초 | 모델을 상주시킬 필요 없이 확실한 재시작 이득 | 채택 |

## 캐시·인페인트·GPU handoff

| 실험 | 핵심 결과 | 판정 | 제품 상태 |
|---|---|---|---|
| SQLite 번역 결과 cache | all-hit 60.301초 → 1.978초, HTTP/Gemma 0 | 동일 request 재실행에 큰 이득 | 채택 |
| 승인형 Exact TM | 승인 항목만 모델을 생략, result cache와 분리 | 사용자 통제와 정확성을 유지 | 채택 |
| global exact OCR cache | all-hit 19.763초 → 0.110초, OCR runtime/HTTP 0 | raw OCR 재사용에 유효 | 채택 |
| 프로젝트 checkpoint | all-hit 139.961초 → 8.236초, 94.115% 단축 | 단계별 fingerprint가 같은 재실행에 유효 | 채택 |
| checkpoint cold + all-hit 시나리오 | 누적 44.261% 순이득 | cold와 재실행을 함께 보면 순이득 | 채택 |
| checkpoint cold + 한 페이지 수정 | 0.931% 순이득 | 변경 stage와 downstream만 재계산 | 채택 |
| OCR stage-aware sleep | 고정 idle 5초 대비 반환 대기 5.470초 → 1.297초, 출력 동일 | crop마다 sleep/wake 없이 OCR stage 끝에서만 반환 | 채택 |
| GPU 75% overlap | 약 5% 빠름 | VRAM·shared memory·swap 압력 증가 | 탈락 |
| inpaint microbatch 2/4 | 40~44% 빠름 | 시각 결과가 달라졌다. | 탈락 |
| channels-last | 픽셀 변경 | lossless 계약 위반 | 탈락 |
| fixed-shape compile | 고정 shape에서 19.539% 개선 | 가변 입력에서 실패 | shape-bucket 후보만 보류 |
| FP32 LaMa/MPE/AOT/ZITS 후보 | 반투명 구조 배경을 깨끗이 복원하지 못함 | 원본 보존보다 나쁜 사례가 발생 | 미승격 |

## 운영·검증 환경

| 항목 | 결과 | 판정 |
|---|---|---|
| CUDA12/13 제품 suite | 두 환경 모두 876 passed, 6 skipped | 통과 |
| CUDA inpaint contract | 두 환경 모두 13/13 | 통과 |
| Python/헤드리스/번역/Windows launcher 검사 | 모두 통과 | 통과 |
| WSL 메모리 프로필 | 20GB memory, 8GB swap, gradual reclaim | page cache와 Windows 여유의 출발점으로 채택 |
| 정상 컨테이너 종료 | `stop` 유지, 광범위 `down`/prune 금지 | 데이터·page cache·재현성 보호 |

세부 품질 수치와 경로 선택은 [OCR 품질 및 라우팅 판정](02-ocr-quality-and-routing-decision-ko.md),
번역 후보 판정은 [Gemma 번역 및 모델 판정](03-gemma-translation-and-model-decision-ko.md),
재실행 효과는 [캐시·checkpoint·cold 경로 판정](04-cache-checkpoint-and-cold-path-decision-ko.md)을 참고한다.

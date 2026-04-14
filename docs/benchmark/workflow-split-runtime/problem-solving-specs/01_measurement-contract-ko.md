# 문제 해결 명세서 01 - Measurement Contract

핵심 문제 해결 방향은 사용자가 착안했다.

## 문제

시간 이득과 품질 동등성을 주장하려면, 무엇을 어떻게 재는지 계약이 먼저 잠겨 있어야 한다.

## 사용자 착안 요약

사용자는 Docker 재기동, healthcheck 대기, VRAM 여유, `ngl` 증가, timeout, 실제 stage 처리시간을 모두 분리해서 총 시간을 계산해야 근거가 된다고 요구했다.

## 왜 중요한가

- 기동시간만 빠르거나 느린 결과로 전체 결론을 내리면 왜곡된다.
- Gemma와 OCR runtime의 idle/resident 비용을 분리해야 한다.
- Requirement 2 selector는 Requirement 1의 측정 surface를 재사용해야 한다.

## 측정 계약

`총 시간 = 순수 처리 시간 + compose up 시간 + health wait 시간 + timeout/retry 패널티 + warm-up 비용 + 단계 전환 비용`

## 체크포인트

1. batch start
2. detect start/end
3. OCR compose up start/end
4. OCR health wait start/end
5. OCR actual start/end
6. OCR stage end
7. Gemma compose up start/end
8. Gemma health wait start/end
9. translation actual start/end
10. translation stage end
11. inpaint start/end
12. render/export start/end
13. batch done
14. timeout/retry/restart/reuse hit

## 현재 구현 후보

- `report_runtime_progress()`
- `mark_processing_stage()`
- benchmark event emission in `batch_processor.py`
- runtime manager 내부 health / compose lifecycle

## 기대 효과

- stage-batched pipeline과 legacy pipeline을 같은 프레임으로 비교할 수 있다.
- Docker 대기 비용과 실제 처리 비용을 따로 해석할 수 있다.
- Requirement 2에서 dual-resident 비용을 동일 표준으로 재활용할 수 있다.

## 현재 상태

- measurement contract 문서화 완료
- 실제 로그 필드 추가는 아직 미구현

## 다음 액션

- 현재 제품 이벤트와 measurement contract의 필드 대응표를 만든다.
- family runner가 이 필드를 summary로 변환하도록 설계한다.

## 저자 및 기여

- Idea Origin: User
- Planning / Measurement Design / Implementation Detailing / Validation: Collaborative
- Execution Support: Codex

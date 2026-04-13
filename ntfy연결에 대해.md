# ntfy 연동에 대해

## 현재 단계

이번 단계에서는 `ntfy` 실제 전송 기능을 넣지 않는다.

- 완료 알림은 로컬 system sound 또는 `music/*.wav`만 사용
- 코드에는 향후 notifier를 붙일 수 있는 stub hook과 주석만 남김

## 이후 연결 포인트

- 자동 파이프라인 정상 완료 직후
- 자동 파이프라인 실패 직후
- 필요 시 retry 완료 직후

## 권장 인터페이스

Notifier는 UI 코드에서 직접 네트워크 호출하지 않고, 얇은 서비스 함수 하나를 통해 연결한다.

```python
def notify_pipeline_event(event: dict) -> None:
    ...
```

권장 payload:

- `event_type`
- `run_type`
- `success`
- `image_count`
- `output_root`
- `message`

## 주의

- 네트워크 실패가 메인 파이프라인을 망치면 안 된다.
- notifier는 항상 best-effort 비차단 방식이어야 한다.
- 사용자 설정이 꺼져 있으면 완전히 no-op여야 한다.

"""Router 종료 정리는 취소 신호를 받아도 반드시 완료되어야 한다.

실행 로그에서 한 번의 취소가 앱을 영구히 막았다. 인과는 이렇다.

1. 배치 정리가 취소 검사기를 컨테이너 정지까지 내려보냈다.
2. 정지가 스스로 취소되어 `OperationCancelledError` 가 났다.
3. 코디네이터는 그것을 `RELEASE_FAILED` 로 표시했다.
4. `RELEASE_FAILED` 는 소유권을 계속 붙든다. 그래서 이후 **모든** 실행이 startup
   preflight 에서 "Router is in RELEASE_FAILED; it retains ownership until terminal
   cleanup succeeds" 로 실패했다.

따라서 정지 경로는 호출자가 무엇을 넘기든 취소되지 않아야 한다. 호출자 규율에만
의존하면 같은 고착이 다시 생긴다.
"""

from __future__ import annotations

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from modules.utils.local_llama_router import (  # noqa: E402
    LocalLlamaRouterCoordinator,
)


class _RecordingAdapter:
    """정지 요청과 함께 넘어온 취소 검사기를 기록한다."""

    def __init__(self) -> None:
        self.stop_calls: list[object] = []

    def stop_pair(self, contract, *, cancel_checker=None) -> None:
        self.stop_calls.append(cancel_checker)
        if cancel_checker is not None and cancel_checker():
            raise AssertionError(
                "종료 정리에 취소를 보고하는 검사기가 넘어갔습니다."
            )

    def gpu_snapshot(self, contract):
        return {}

    def owned_gpu_process_ids(self, contract):
        return frozenset()


def _always_cancelled() -> bool:
    return True


class TerminalCleanupIgnoresCancelTests(unittest.TestCase):
    def test_stop_container_drops_the_cancel_checker(self) -> None:
        """`finish(stop_container=True)` 는 정의상 정리다. 취소를 받지 않는다."""

        adapter = _RecordingAdapter()
        coordinator = LocalLlamaRouterCoordinator(adapter=adapter)

        # 아직 아무 컨테이너도 소유하지 않은 상태에서는 할 일이 없다. 여기서
        # 확인하려는 것은 취소 검사기가 걸러진다는 사실 자체다.
        coordinator.finish(
            service="paddleocr_vl",
            stop_container=True,
            cancel_checker=_always_cancelled,
        )
        for checker in adapter.stop_calls:
            self.assertFalse(checker is _always_cancelled)

    def test_stop_delegates_with_the_same_protection(self) -> None:
        adapter = _RecordingAdapter()
        coordinator = LocalLlamaRouterCoordinator(adapter=adapter)
        coordinator.stop(service="gemma", cancel_checker=_always_cancelled)
        for checker in adapter.stop_calls:
            self.assertFalse(checker is _always_cancelled)

    def test_a_mid_run_release_still_honours_cancellation(self) -> None:
        """실행 중 핸드오프는 사용자가 취소하면 멈춰야 한다. 그 경로는 그대로다."""

        import inspect

        source = inspect.getsource(LocalLlamaRouterCoordinator.finish)
        # 보호는 stop_container 인 경우로만 한정된다.
        self.assertIn("if stop_container:", source)
        self.assertIn("cancel_checker = None", source)
        guard = source.split("if stop_container:", 1)[1]
        self.assertTrue(
            guard.lstrip().startswith("cancel_checker = None"),
            "취소 무시가 stop_container 조건 밖으로 새어나갔습니다.",
        )


if __name__ == "__main__":
    unittest.main()

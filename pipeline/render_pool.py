"""렌더 워커 전용 Qt 스레드 풀 (Phase 3a).

렌더는 `QGraphicsScene`/`QPainter` 를 쓴다. Qt 는 이들을 **이벤트 디스패처가
있는 스레드**에서 돌 것을 요구한다. `concurrent.futures.ThreadPoolExecutor` 가
만드는 plain Python 스레드에는 디스패처가 없어서, `scene.addItem()` 의 지연
처리가 끝나지 않은 채 `scene.render()` 가 **아무것도 그리지 않고 조용히
성공**한다 — 결과물이 통째로 검은 이미지가 된다(예외도 경고도 없다).

실측(2000x1430 실제 페이지, 렌더 결과 픽셀 합):

    메인 스레드                              1179966789  정상
    plain Python 스레드(ThreadPoolExecutor)           0  전부 검정
    QThreadPool / QRunnable                 1179966789  정상

그래서 렌더 워커는 반드시 Qt 스레드 위에서 돈다. 앱 공유 `QThreadPool`
(`main_page.threadpool`)이 아니라 **전용 인스턴스**를 쓰는 이유는 그대로다:
공유 풀을 잠식하면 자동저장·UI 작업이 렌더 뒤에 밀린다.

호출부가 기존 `concurrent.futures` 관용구(`wait`, `Future.result()`)를 그대로
쓸 수 있도록 `Future` 를 돌려준다.
"""

from __future__ import annotations

import logging
from concurrent.futures import Future
from typing import Any, Callable

from PySide6.QtCore import QRunnable, QThreadPool

logger = logging.getLogger(__name__)


class _RenderRunnable(QRunnable):
    """작업 하나를 Qt 스레드에서 실행하고 그 결과로 `Future` 를 채운다."""

    def __init__(self, future: Future, fn: Callable[..., Any], *args: Any) -> None:
        super().__init__()
        self._future = future
        self._fn = fn
        self._args = args

    def run(self) -> None:  # noqa: D102 - QRunnable 계약
        if not self._future.set_running_or_notify_cancel():
            return
        try:
            self._future.set_result(self._fn(*self._args))
        except BaseException as exc:  # noqa: BLE001 - 페이지 단위 실패로 강등해 전달
            self._future.set_exception(exc)


class QtRenderPool:
    """워커 1개짜리 전용 Qt 스레드 풀. `submit` 은 `Future` 를 돌려준다."""

    def __init__(self, *, max_workers: int = 1, expiry_timeout_ms: int = -1) -> None:
        self._pool = QThreadPool()
        self._pool.setMaxThreadCount(max_workers)
        # 스윕 중 스레드가 만료돼 재생성되지 않게 한다(재생성 자체는 안전하지만
        # 폰트/스레드 최초 접근 비용을 페이지마다 다시 내게 된다).
        self._pool.setExpiryTimeout(expiry_timeout_ms)

    def submit(self, fn: Callable[..., Any], *args: Any) -> Future:
        future: Future = Future()
        self._pool.start(_RenderRunnable(future, fn, *args))
        return future

    def shutdown(self, *, wait: bool = True, timeout_ms: int = 30_000) -> None:
        """아직 시작하지 않은 작업을 버리고, 실행 중인 작업이 끝나기를 기다린다.

        정리는 어떤 상황에서도 실패하면 안 되므로 예외를 밖으로 내보내지 않는다.
        """
        try:
            self._pool.clear()
        except Exception:
            logger.warning("렌더 풀 대기열 정리에 실패했지만 무시합니다.", exc_info=True)
        if not wait:
            return
        try:
            if not self._pool.waitForDone(timeout_ms):
                logger.warning(
                    "렌더 풀이 %d ms 안에 종료되지 않았습니다. 계속 진행합니다.",
                    timeout_ms,
                )
        except Exception:
            logger.warning("렌더 풀 종료 대기에 실패했지만 무시합니다.", exc_info=True)

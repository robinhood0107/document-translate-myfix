from __future__ import annotations

import logging
from collections import deque
from typing import TYPE_CHECKING, Callable

from PySide6 import QtCore
from PySide6.QtCore import QCoreApplication

from app.thread_worker import GenericWorker

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from controller import ComicTranslate


class TaskRunnerController:
    def __init__(self, main: ComicTranslate):
        self.main = main
        self.operation_queue = deque()
        self.is_processing_queue = False

    def run_threaded(
        self,
        callback: Callable,
        result_callback: Callable = None,
        error_callback: Callable = None,
        finished_callback: Callable = None,
        *args,
        **kwargs,
    ):
        return self._queue_operation(
            callback,
            result_callback,
            error_callback,
            finished_callback,
            *args,
            **kwargs,
        )

    def run_threaded_with_progress(
        self,
        callback: Callable,
        progress_callback: Callable,
        result_callback: Callable = None,
        error_callback: Callable = None,
        finished_callback: Callable = None,
        *args,
        **kwargs,
    ):
        operation = {
            "callback": callback,
            "progress_callback": progress_callback,
            "inject_progress": True,
            "result_callback": result_callback,
            "error_callback": error_callback,
            "finished_callback": finished_callback,
            "args": args,
            "kwargs": kwargs,
        }
        self.operation_queue.append(operation)
        if not self.is_processing_queue:
            self._process_next_operation()

    def _queue_operation(
        self,
        callback: Callable,
        result_callback: Callable = None,
        error_callback: Callable = None,
        finished_callback: Callable = None,
        *args,
        **kwargs,
    ):
        operation = {
            "callback": callback,
            "progress_callback": None,
            "inject_progress": False,
            "result_callback": result_callback,
            "error_callback": error_callback,
            "finished_callback": finished_callback,
            "args": args,
            "kwargs": kwargs,
        }

        self.operation_queue.append(operation)
        if not self.is_processing_queue:
            self._process_next_operation()

    def _process_next_operation(self):
        if not self.operation_queue:
            self.is_processing_queue = False
            return

        self.is_processing_queue = True
        operation = self.operation_queue.popleft()

        def enhanced_finished_callback():
            # 호출자가 준 콜백이 예외를 던져도 큐는 계속 진행해야 한다. 그렇지
            # 않으면 이후 모든 작업이 이미 끝난 작업 뒤에서 대기한다. 예외는 Qt
            # 이벤트 루프로 빠져나가면 유실되고 프로세스를 죽일 수 있으므로 로그로
            # 만 남긴다.
            try:
                if operation["finished_callback"]:
                    operation["finished_callback"]()
            except Exception:
                logger.exception("Finished callback failed for a queued operation.")
            finally:
                QtCore.QTimer.singleShot(0, self.main, self._process_next_operation)

        def enhanced_error_callback(error_tuple):
            try:
                if operation["error_callback"]:
                    operation["error_callback"](error_tuple)
            except Exception:
                logger.exception("Error callback failed for a queued operation.")
            finally:
                QtCore.QTimer.singleShot(0, self.main, self._process_next_operation)

        def enhanced_result_callback(result):
            if operation["result_callback"]:
                operation["result_callback"](result)

        self._execute_single_operation(
            operation["callback"],
            enhanced_result_callback,
            enhanced_error_callback,
            enhanced_finished_callback,
            *operation["args"],
            _progress_callback=operation.get("progress_callback"),
            _inject_progress=bool(operation.get("inject_progress")),
            **operation["kwargs"],
        )

    def _execute_single_operation(
        self,
        callback: Callable,
        result_callback: Callable = None,
        error_callback: Callable = None,
        finished_callback: Callable = None,
        *args,
        _progress_callback: Callable = None,
        _inject_progress: bool = False,
        **kwargs,
    ):
        if _inject_progress:
            worker = GenericWorker(
                lambda: callback(worker.signals.progress.emit, *args, **kwargs)
            )
        else:
            worker = GenericWorker(callback, *args, **kwargs)

        def _clear_current_worker() -> None:
            if getattr(self.main, "current_worker", None) is worker:
                self.main.current_worker = None

        if result_callback:
            worker.signals.result.connect(
                lambda result: QtCore.QTimer.singleShot(
                    0, self.main, lambda: result_callback(result)
                )
            )
        if error_callback:
            worker.signals.error.connect(
                lambda error: QtCore.QTimer.singleShot(
                    0, self.main, lambda: error_callback(error)
                )
            )
        if _progress_callback:
            worker.signals.progress.connect(
                lambda event: QtCore.QTimer.singleShot(
                    0, self.main, lambda: _progress_callback(dict(event or {}))
                )
            )
        if finished_callback:
            worker.signals.finished.connect(
                lambda: QtCore.QTimer.singleShot(
                    0,
                    self.main,
                    lambda: (_clear_current_worker(), finished_callback())[1],
                )
            )
        else:
            worker.signals.finished.connect(
                lambda: QtCore.QTimer.singleShot(0, self.main, _clear_current_worker)
            )

        self.main.current_worker = worker
        self.main.threadpool.start(worker)

    def run_threaded_immediate(
        self,
        callback: Callable,
        result_callback: Callable = None,
        error_callback: Callable = None,
        finished_callback: Callable = None,
        *args,
        **kwargs,
    ):
        return self._execute_single_operation(
            callback,
            result_callback,
            error_callback,
            finished_callback,
            *args,
            **kwargs,
        )

    def clear_operation_queue(self):
        self.operation_queue.clear()
        # 비운 큐에는 처리할 것이 남아 있지 않으므로 busy 플래그가 살아남으면 안
        # 된다. 남은 플래그는 이후 모든 작업을 큐에 가둔다.
        self.is_processing_queue = False

    def cancel_current_task(self):
        if self.main.current_worker:
            self.main.current_worker.cancel()

        if self.main._batch_active:
            self.main._batch_cancel_requested = True
            self.main.cancel_button.setEnabled(False)
            self.main.progress_bar.setFormat(
                QCoreApplication.translate("Messages", "Cancelling... %p%")
            )

        self.clear_operation_queue()
        self.is_processing_queue = False

    def run_finish_only(
        self, finished_callback: Callable, error_callback: Callable = None
    ):
        def _noop():
            pass

        self._queue_operation(
            callback=_noop,
            result_callback=None,
            error_callback=error_callback,
            finished_callback=finished_callback,
        )

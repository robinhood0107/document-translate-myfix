from __future__ import annotations

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6 import QtCore, QtWidgets  # noqa: E402

from app.controllers.task_runner import TaskRunnerController  # noqa: E402


def _app() -> QtWidgets.QApplication:
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


class _FakeMain(QtCore.QObject):
    """Minimal stand-in for the controller surface the task runner touches."""

    def __init__(self) -> None:
        super().__init__()
        self.threadpool = QtCore.QThreadPool()
        self.threadpool.setMaxThreadCount(2)
        self.current_worker = None
        self._batch_active = False
        self._batch_cancel_requested = False


class TaskRunnerQueueRecoveryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.app = _app()
        self.main = _FakeMain()
        self.runner = TaskRunnerController(self.main)

    def _drain(self, predicate, timeout_ms: int = 4000) -> bool:
        deadline = QtCore.QElapsedTimer()
        deadline.start()
        while deadline.elapsed() < timeout_ms:
            self.app.processEvents()
            self.main.threadpool.waitForDone(20)
            self.app.processEvents()
            if predicate():
                return True
        return False

    def test_queued_operation_runs_after_a_raising_finished_callback(self) -> None:
        """A failing finished callback must not wedge the queue for every later task."""

        done: list[str] = []

        def boom() -> None:
            raise RuntimeError("finished callback failed")

        self.runner.run_threaded(lambda: done.append("first"), None, None, boom)
        self.assertTrue(self._drain(lambda: "first" in done), "first task never ran")

        self.runner.run_threaded(lambda: done.append("second"), None, None, None)
        self.assertTrue(
            self._drain(lambda: "second" in done),
            "queue stayed blocked after a finished callback raised",
        )
        self.assertTrue(
            self._drain(lambda: not self.runner.is_processing_queue),
            "runner never returned to idle after draining the queue",
        )

    def test_clearing_the_queue_releases_the_processing_flag(self) -> None:
        """clear_operation_queue must not leave the runner permanently busy."""

        self.runner.is_processing_queue = True
        self.runner.operation_queue.append({"callback": lambda: None})
        self.runner.clear_operation_queue()
        self.assertEqual(len(self.runner.operation_queue), 0)
        self.assertFalse(self.runner.is_processing_queue)

    def test_operation_queued_after_a_cleared_queue_still_runs(self) -> None:
        """The rerender path queues work; a stale busy flag must not strand it."""

        done: list[str] = []
        self.runner.is_processing_queue = True
        self.runner.clear_operation_queue()
        self.runner.run_threaded(lambda: done.append("rerender"), None, None, None)
        self.assertTrue(
            self._drain(lambda: "rerender" in done),
            "operation queued after clear_operation_queue never started",
        )

    def test_error_callback_failure_still_advances_the_queue(self) -> None:
        done: list[str] = []

        def failing_task() -> None:
            raise ValueError("task failed")

        def bad_error_callback(_error) -> None:
            raise RuntimeError("error callback failed")

        self.runner.run_threaded(failing_task, None, bad_error_callback, None)
        self.assertTrue(self._drain(lambda: not self.runner.is_processing_queue))

        self.runner.run_threaded(lambda: done.append("after"), None, None, None)
        self.assertTrue(
            self._drain(lambda: "after" in done),
            "queue stayed blocked after an error callback raised",
        )


if __name__ == "__main__":
    unittest.main()

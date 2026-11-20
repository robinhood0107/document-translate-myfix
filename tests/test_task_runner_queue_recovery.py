from __future__ import annotations

import os
import time
import unittest
from threading import Event

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

    def test_cancelled_operation_cannot_advance_the_replacement_queue(self) -> None:
        old_started = Event()
        release_old = Event()
        old_finished = Event()
        new_started = Event()
        release_new = Event()
        third_started = Event()

        def old_task() -> None:
            old_started.set()
            release_old.wait()

        def new_task() -> None:
            new_started.set()
            release_new.wait()

        try:
            self.runner.run_threaded(old_task, finished_callback=old_finished.set)
            self.assertTrue(self._drain(old_started.is_set), "old task never started")

            self.runner.cancel_current_task()
            self.runner.run_threaded(new_task)
            self.runner.run_threaded(lambda: third_started.set())
            self.assertTrue(self._drain(new_started.is_set), "new task never started")
            self.assertEqual(len(self.runner.operation_queue), 1)
            new_worker = self.main.current_worker

            release_old.set()
            self.assertTrue(self._drain(old_finished.is_set), "old finish callback was lost")
            self.assertFalse(
                self._drain(third_started.is_set, timeout_ms=150),
                "old task advanced the replacement queue while its owner was running",
            )
            self.assertIs(self.main.current_worker, new_worker)
        finally:
            release_old.set()
            release_new.set()

        self.assertTrue(
            self._drain(lambda: third_started.is_set() and not self.runner.is_processing_queue),
            "replacement queue did not drain after its owner finished",
        )

    def test_posted_terminal_callback_cannot_advance_a_replacement_queue(self) -> None:
        old_finished = Event()
        new_started = Event()
        release_new = Event()
        third_started = Event()

        self.runner.run_threaded(lambda: None, finished_callback=old_finished.set)
        self.assertTrue(
            self.main.threadpool.waitForDone(4000), "old task did not finish in the pool"
        )
        self.assertFalse(old_finished.is_set())

        try:
            self.runner.cancel_current_task()
            self.runner.run_threaded(lambda: (new_started.set(), release_new.wait()))
            self.runner.run_threaded(lambda: third_started.set())
            self.assertTrue(self._drain(new_started.is_set), "new task never started")
            new_worker = self.main.current_worker

            self.assertTrue(self._drain(old_finished.is_set), "posted old finish callback was lost")
            self.assertFalse(
                self._drain(third_started.is_set, timeout_ms=150),
                "a terminal callback posted before cancellation advanced the replacement queue",
            )
            self.assertIs(self.main.current_worker, new_worker)
        finally:
            release_new.set()

        self.assertTrue(
            self._drain(lambda: third_started.is_set() and not self.runner.is_processing_queue),
            "replacement queue did not drain after its owner finished",
        )

    def test_posted_dequeue_callback_cannot_change_replacement_owner_state(self) -> None:
        new_started = Event()
        release_new = Event()
        third_started = Event()
        replacement_started = Event()

        def replace_queue() -> None:
            self.runner.cancel_current_task()
            self.runner.run_threaded(lambda: (new_started.set(), release_new.wait()))
            self.runner.run_threaded(lambda: third_started.set())
            replacement_started.set()

        def old_finished() -> None:
            QtCore.QTimer.singleShot(0, self.main, replace_queue)

        self.runner.run_threaded(lambda: None, finished_callback=old_finished)

        try:
            self.assertTrue(
                self._drain(replacement_started.is_set), "replacement queue never started"
            )
            self.assertTrue(self._drain(new_started.is_set), "new task never started")
            self.assertFalse(
                self._drain(third_started.is_set, timeout_ms=150),
                "an already-posted dequeue callback advanced the replacement queue",
            )
            self.assertTrue(self.runner.is_processing_queue)
        finally:
            release_new.set()

        self.assertTrue(
            self._drain(lambda: third_started.is_set() and not self.runner.is_processing_queue),
            "replacement queue did not drain after its owner finished",
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

    def test_error_handoff_waits_for_the_successor_to_finish(self) -> None:
        error_calls: list[tuple] = []
        finished_calls: list[str] = []
        started: list[str] = []
        successor_started = Event()
        release_successor = Event()
        third_started = Event()

        def failing_task() -> None:
            raise ValueError("task failed")

        def blocking_successor() -> None:
            started.append("successor")
            successor_started.set()
            release_successor.wait()

        def third_task() -> None:
            started.append("third")
            third_started.set()

        try:
            self.runner.run_threaded(
                failing_task,
                None,
                error_calls.append,
                lambda: finished_calls.append("finished"),
            )
            self.runner.run_threaded(blocking_successor)
            self.runner.run_threaded(third_task)

            callbacks_delivered = self._drain(
                lambda: successor_started.is_set()
                and error_calls
                and finished_calls
            )
            third_started_before_release = third_started.is_set()
        finally:
            release_successor.set()

        self.assertTrue(callbacks_delivered, "error or finished callback was not delivered")
        self.assertTrue(
            self._drain(
                lambda: third_started.is_set() and not self.runner.is_processing_queue
            ),
            "queue did not finish after the successor was released",
        )
        self.assertEqual(started, ["successor", "third"])
        self.assertFalse(third_started_before_release)

    def test_progress_worker_delivers_events_and_result_on_ui_thread(self) -> None:
        progress: list[dict] = []
        results: list[str] = []

        def worker(report_progress):
            report_progress({"stage": "index", "current": 1, "total": 2})
            report_progress({"stage": "decode", "current": 2, "total": 2})
            return "ready"

        self.runner.run_threaded_with_progress(
            worker,
            progress.append,
            results.append,
        )
        self.assertTrue(self._drain(lambda: results == ["ready"]))
        self.assertEqual([event["stage"] for event in progress], ["index", "decode"])

    def test_progress_worker_does_not_block_the_qt_event_loop(self) -> None:
        ticks: list[str] = []
        results: list[str] = []

        def worker(report_progress):
            report_progress({"stage": "index"})
            time.sleep(0.15)
            return "ready"

        timer = QtCore.QTimer(self.main)
        timer.setInterval(10)
        timer.timeout.connect(lambda: ticks.append("tick"))
        timer.start()
        self.runner.run_threaded_with_progress(worker, lambda _event: None, results.append)
        self.assertTrue(self._drain(lambda: results == ["ready"]))
        timer.stop()
        self.assertGreater(len(ticks), 0)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import threading
import unittest

from modules.utils.exceptions import OperationCancelledError
from pipeline.runtime_resource_arbiter import (
    RuntimeLeaseConflictError,
    RuntimeModelState,
    RuntimeResourceArbiter,
)


class RuntimeResourceArbiterTests(unittest.TestCase):
    def test_cancelled_generation_never_starts_model(self) -> None:
        arbiter = RuntimeResourceArbiter()
        token = arbiter.token("ocr")
        arbiter.cancel_generation()
        started = False

        with self.assertRaises(OperationCancelledError):
            with arbiter.model_start(token):
                started = True

        self.assertFalse(started)
        self.assertIsNone(arbiter.snapshot().active_model)

    def test_active_model_blocks_another_model_until_release(self) -> None:
        arbiter = RuntimeResourceArbiter()
        with arbiter.model_start(arbiter.token("ocr")):
            pass

        with self.assertRaisesRegex(
            RuntimeLeaseConflictError,
            "GPU model lease is held by ocr",
        ):
            with arbiter.model_start(arbiter.token("gemma")):
                pass

        with arbiter.model_release(
            "ocr",
            target_state=RuntimeModelState.SLEEPING,
        ):
            pass
        with arbiter.model_start(arbiter.token("gemma")):
            pass

        snapshot = arbiter.snapshot()
        self.assertEqual(snapshot.active_model, "gemma")
        self.assertEqual(snapshot.states["ocr"], "sleeping")
        self.assertEqual(snapshot.states["gemma"], "model_ready")

    def test_cancellation_during_start_cleans_stale_model_once(self) -> None:
        arbiter = RuntimeResourceArbiter()
        token = arbiter.token("ocr")
        started = threading.Event()
        finish_start = threading.Event()
        cleanup_calls: list[str] = []
        errors: list[BaseException] = []

        def run_start() -> None:
            try:
                with arbiter.model_start(
                    token,
                    stale_cleanup=lambda: cleanup_calls.append("cleanup"),
                ):
                    started.set()
                    finish_start.wait(timeout=5.0)
            except BaseException as exc:
                errors.append(exc)

        worker = threading.Thread(target=run_start)
        worker.start()
        self.assertTrue(started.wait(timeout=2.0))
        arbiter.cancel_generation()
        finish_start.set()
        worker.join(timeout=2.0)

        self.assertFalse(worker.is_alive())
        self.assertEqual(cleanup_calls, ["cleanup"])
        self.assertEqual(len(errors), 1)
        self.assertIsInstance(errors[0], OperationCancelledError)
        snapshot = arbiter.snapshot()
        self.assertIsNone(snapshot.active_model)
        self.assertEqual(snapshot.states["ocr"], "stopped")

    def test_external_inpainter_lease_blocks_managed_runtime(self) -> None:
        arbiter = RuntimeResourceArbiter()
        arbiter.acquire_external_model("inpainter")
        self.assertEqual(
            arbiter.snapshot().states["inpainter"],
            "model_loading",
        )

        with self.assertRaisesRegex(
            RuntimeLeaseConflictError,
            "GPU model lease is held by inpainter",
        ):
            with arbiter.model_start(arbiter.token("gemma")):
                pass

        arbiter.mark_external_model_ready("inpainter")
        self.assertEqual(
            arbiter.snapshot().states["inpainter"],
            "model_ready",
        )
        arbiter.release_external_model(
            "inpainter",
            release_succeeded=True,
        )
        with arbiter.model_start(arbiter.token("gemma")):
            pass

        self.assertEqual(arbiter.snapshot().active_model, "gemma")

    def test_release_failure_preserves_lease_and_failed_state(self) -> None:
        arbiter = RuntimeResourceArbiter()
        with arbiter.model_start(arbiter.token("gemma")):
            pass

        with self.assertRaisesRegex(RuntimeError, "release failed"):
            with arbiter.model_release("gemma"):
                raise RuntimeError("release failed")

        snapshot = arbiter.snapshot()
        self.assertEqual(snapshot.active_model, "gemma")
        self.assertEqual(snapshot.states["gemma"], "release_failed")
        with self.assertRaisesRegex(
            RuntimeLeaseConflictError,
            "previous release failed",
        ):
            with arbiter.model_start(arbiter.token("gemma")):
                pass

    def test_unverified_preexisting_release_fails_closed(self) -> None:
        arbiter = RuntimeResourceArbiter()

        with self.assertRaisesRegex(RuntimeError, "release failed"):
            with arbiter.model_release("ocr"):
                raise RuntimeError("release failed")

        snapshot = arbiter.snapshot()
        self.assertEqual(snapshot.active_model, "ocr")
        self.assertEqual(snapshot.states["ocr"], "release_failed")
        with self.assertRaisesRegex(
            RuntimeLeaseConflictError,
            "GPU model lease is held by ocr",
        ):
            with arbiter.model_start(arbiter.token("gemma")):
                pass

    def test_terminal_foreign_teardown_preserves_failed_owner(self) -> None:
        arbiter = RuntimeResourceArbiter()
        with arbiter.model_start(arbiter.token("ocr")):
            pass
        with self.assertRaisesRegex(RuntimeError, "release failed"):
            with arbiter.model_release("ocr"):
                raise RuntimeError("release failed")

        with arbiter.model_release(
            "gemma",
            allow_foreign_owner_teardown=True,
        ):
            pass

        snapshot = arbiter.snapshot()
        self.assertEqual(snapshot.active_model, "ocr")
        self.assertEqual(snapshot.states["ocr"], "release_failed")
        self.assertEqual(snapshot.states["gemma"], "stopped")

    def test_failed_start_cleanup_error_blocks_synchronous_retry(self) -> None:
        arbiter = RuntimeResourceArbiter()

        def failed_cleanup() -> None:
            raise RuntimeError("cleanup failed")

        with self.assertRaisesRegex(RuntimeError, "cleanup failed"):
            with arbiter.model_start(
                arbiter.token("ocr"),
                stale_cleanup=failed_cleanup,
            ):
                raise RuntimeError("startup failed")

        snapshot = arbiter.snapshot()
        self.assertEqual(snapshot.active_model, "ocr")
        self.assertEqual(snapshot.states["ocr"], "release_failed")
        with self.assertRaises(RuntimeLeaseConflictError):
            with arbiter.model_start(arbiter.token("ocr")):
                pass

    def test_observer_failure_cannot_break_runtime_ownership(self) -> None:
        calls: list[str] = []

        def broken_observer(
            service: str,
            state: RuntimeModelState,
            _previous: RuntimeModelState | None,
            _outcome: str,
        ) -> None:
            calls.append(f"{service}:{state.value}")
            raise RuntimeError("telemetry failure")

        arbiter = RuntimeResourceArbiter(
            transition_callback=broken_observer,
        )
        with arbiter.model_start(arbiter.token("ocr")):
            pass
        with arbiter.model_release("ocr"):
            pass

        snapshot = arbiter.snapshot()
        self.assertIsNone(snapshot.active_model)
        self.assertEqual(snapshot.states["ocr"], "stopped")
        self.assertEqual(
            calls,
            [
                "ocr:model_loading",
                "ocr:model_ready",
                "ocr:releasing",
                "ocr:stopped",
            ],
        )


if __name__ == "__main__":
    unittest.main()

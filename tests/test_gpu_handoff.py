from __future__ import annotations

import unittest

from modules.utils.gpu_handoff import wait_for_vram_release


def _snapshot(
    *,
    process_available: bool,
    allocated_mb: float = 0.0,
    reserved_mb: float = 0.0,
    driver_available: bool = False,
    driver_used_mb: float = 0.0,
) -> dict:
    return {
        "process": {
            "available": process_available,
            "allocated_mb": allocated_mb,
            "reserved_mb": reserved_mb,
        },
        "driver": {
            "available": driver_available,
            "primary": (
                {"memory_used_mb": driver_used_mb}
                if driver_available
                else None
            ),
        },
    }


class GPUHandoffTests(unittest.TestCase):
    def test_process_allocator_drop_satisfies_release_gate(self) -> None:
        before = _snapshot(
            process_available=True,
            allocated_mb=1024.0,
            reserved_mb=1280.0,
        )
        after = _snapshot(
            process_available=True,
            allocated_mb=32.0,
            reserved_mb=64.0,
        )

        report = wait_for_vram_release(
            before,
            gpu_release_expected=True,
            timeout_sec=0.0,
            sampler=lambda: after,
        )

        self.assertTrue(report["required"])
        self.assertTrue(report["observed"])
        self.assertEqual(report["status"], "observed")
        self.assertEqual(report["deltas"]["process_allocated_drop_mb"], 992.0)

    def test_expected_release_times_out_when_no_metric_drops(self) -> None:
        before = _snapshot(
            process_available=True,
            allocated_mb=1024.0,
            reserved_mb=1280.0,
        )

        report = wait_for_vram_release(
            before,
            gpu_release_expected=True,
            timeout_sec=0.0,
            sampler=lambda: before,
        )

        self.assertTrue(report["required"])
        self.assertFalse(report["observed"])
        self.assertEqual(report["status"], "timeout")

    def test_missing_metrics_is_reported_without_false_failure(self) -> None:
        unavailable = _snapshot(process_available=False)

        report = wait_for_vram_release(
            unavailable,
            gpu_release_expected=True,
            timeout_sec=0.0,
            sampler=lambda: unavailable,
        )

        self.assertFalse(report["required"])
        self.assertIsNone(report["observed"])
        self.assertEqual(report["status"], "unavailable")


if __name__ == "__main__":
    unittest.main()

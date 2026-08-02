from __future__ import annotations

import unittest

from modules.utils.gpu_handoff import (
    estimate_torch_cuda_storage_mb,
    router_gpu_process_set,
    wait_for_router_container_stop_release,
    wait_for_global_vram_release,
    wait_for_vram_release,
)


def _snapshot(
    *,
    process_available: bool,
    allocated_mb: float = 0.0,
    reserved_mb: float = 0.0,
    gpu_uuid: str = "GPU-target",
    process_driver_mb: float | None = None,
    driver_query_available: bool = True,
    global_driver_used_mb: float = 0.0,
) -> dict:
    process_rows = (
        [
            {
                "pid": 123,
                "gpu_uuid": gpu_uuid,
                "memory_used_mb": process_driver_mb,
            }
        ]
        if process_driver_mb is not None
        else []
    )
    return {
        "process": {
            "available": process_available,
            "device_uuid": gpu_uuid if process_available else "",
            "allocated_mb": allocated_mb,
            "reserved_mb": reserved_mb,
        },
        "driver": {
            "available": True,
            "primary": {
                "uuid": "GPU-primary",
                "memory_used_mb": global_driver_used_mb,
            },
        },
        "driver_process": {
            "query_available": driver_query_available,
            "available": bool(process_rows),
            "pid": 123,
            "rows": process_rows,
            "selected": process_rows[0] if process_rows else None,
        },
    }


def _router_snapshot(*, used_mb: float, process_ids: set[int]) -> dict:
    gpu_uuid = "GPU-router"
    return {
        "driver": {
            "available": True,
            "primary": {
                "uuid": gpu_uuid,
                "memory_used_mb": used_mb,
            },
        },
        "driver_processes": {
            "query_available": True,
            "rows": [
                {
                    "pid": pid,
                    "gpu_uuid": gpu_uuid,
                    "memory_used_mb": None,
                }
                for pid in sorted(process_ids)
            ],
        },
    }


class _FakeStorage:
    def __init__(self, pointer: int, byte_count: int) -> None:
        self._pointer = pointer
        self._byte_count = byte_count

    def data_ptr(self) -> int:
        return self._pointer

    def nbytes(self) -> int:
        return self._byte_count


class _FakeTensor:
    device = "cuda:0"

    def __init__(self, storage: _FakeStorage) -> None:
        self._storage = storage

    def untyped_storage(self) -> _FakeStorage:
        return self._storage


class _FakeModel:
    def __init__(self, tensors: list[_FakeTensor]) -> None:
        self._tensors = tensors

    def parameters(self):
        return iter(self._tensors)

    def buffers(self):
        return iter(())


class _FakeModelWrapper:
    def __init__(self, model: _FakeModel) -> None:
        self.generator = model


class GPUHandoffTests(unittest.TestCase):
    def test_cuda_storage_estimate_deduplicates_shared_storage(self) -> None:
        shared = _FakeStorage(100, 64 * 1024 * 1024)
        unique = _FakeStorage(200, 32 * 1024 * 1024)

        report = estimate_torch_cuda_storage_mb(
            _FakeModel([_FakeTensor(shared), _FakeTensor(shared), _FakeTensor(unique)])
        )

        self.assertTrue(report["available"])
        self.assertEqual(report["storage_count"], 2)
        self.assertEqual(report["total_mb"], 96.0)

    def test_cuda_storage_estimate_discovers_wrapped_torch_module(self) -> None:
        storage = _FakeStorage(300, 48 * 1024 * 1024)

        report = estimate_torch_cuda_storage_mb(
            _FakeModelWrapper(_FakeModel([_FakeTensor(storage)]))
        )

        self.assertTrue(report["available"])
        self.assertEqual(report["storage_count"], 1)
        self.assertEqual(report["total_mb"], 48.0)

    def test_zero_model_router_stop_allows_small_baseline_sampling_jitter(self) -> None:
        baseline = _router_snapshot(used_mb=390.0, process_ids=set())
        before = _router_snapshot(used_mb=388.0, process_ids=set())
        after = _router_snapshot(used_mb=392.0, process_ids=set())

        report = wait_for_router_container_stop_release(
            before,
            container_baseline=baseline,
            owned_router_process_ids=frozenset({1}),
            timeout_sec=0.0,
            sampler=lambda: after,
        )

        self.assertTrue(report["observed"])
        self.assertEqual(report["status"], "observed")

    def test_router_worker_probe_failure_is_not_an_empty_pid_set(self) -> None:
        snapshot = _router_snapshot(used_mb=1200.0, process_ids=set())
        snapshot["router_worker_processes"] = {
            "query_available": False,
            "rows": [],
            "reason": "router-worker-device-fd-inspection-failed",
        }

        process_ids, source = router_gpu_process_set(
            snapshot,
            gpu_uuid="GPU-router",
        )

        self.assertIsNone(process_ids)
        self.assertEqual(source, "router-worker-dxg")

    def test_model_sized_process_allocator_drop_satisfies_release_gate(self) -> None:
        before = _snapshot(
            process_available=True,
            allocated_mb=1024.0,
            reserved_mb=1280.0,
        )
        after = _snapshot(
            process_available=True,
            allocated_mb=100.0,
            reserved_mb=128.0,
        )

        report = wait_for_vram_release(
            before,
            gpu_release_expected=True,
            expected_process_drop_mb=900.0,
            timeout_sec=0.0,
            sampler=lambda: after,
        )

        self.assertTrue(report["required"])
        self.assertTrue(report["observed"])
        self.assertEqual(report["status"], "observed")
        self.assertEqual(report["evidence_source"], "process-allocated")
        self.assertEqual(report["process_threshold_mb"], 810.0)

    def test_incidental_allocator_drop_cannot_satisfy_model_release(self) -> None:
        before = _snapshot(
            process_available=True,
            allocated_mb=1024.0,
            reserved_mb=1280.0,
            global_driver_used_mb=4096.0,
        )
        incidental_drop = _snapshot(
            process_available=True,
            allocated_mb=992.0,
            reserved_mb=1280.0,
            global_driver_used_mb=2048.0,
        )

        report = wait_for_vram_release(
            before,
            gpu_release_expected=True,
            expected_process_drop_mb=900.0,
            timeout_sec=0.0,
            sampler=lambda: incidental_drop,
        )

        self.assertFalse(report["observed"])
        self.assertEqual(report["status"], "timeout")
        self.assertEqual(report["evidence_source"], "process-allocated")

    def test_non_torch_release_uses_pid_and_device_baseline(self) -> None:
        baseline = _snapshot(
            process_available=True,
            allocated_mb=512.0,
            process_driver_mb=512.0,
        )
        before = _snapshot(
            process_available=True,
            allocated_mb=512.0,
            process_driver_mb=2560.0,
        )
        after = _snapshot(
            process_available=True,
            allocated_mb=512.0,
            process_driver_mb=512.0,
        )

        report = wait_for_vram_release(
            before,
            gpu_release_expected=True,
            untracked_gpu_resource_count=1,
            driver_baseline=baseline,
            timeout_sec=0.0,
            sampler=lambda: after,
        )

        self.assertTrue(report["observed"])
        self.assertEqual(report["evidence_source"], "pid-device-driver-baseline")
        self.assertEqual(report["driver_gpu_uuid"], "GPU-target")
        self.assertEqual(report["deltas"]["process_driver_used_drop_mb"], 2048.0)

    def test_global_or_other_process_drop_cannot_satisfy_driver_gate(self) -> None:
        baseline = _snapshot(
            process_available=False,
            process_driver_mb=128.0,
            global_driver_used_mb=1024.0,
        )
        before = _snapshot(
            process_available=False,
            process_driver_mb=2048.0,
            global_driver_used_mb=4096.0,
        )
        other_process_only_drop = _snapshot(
            process_available=False,
            process_driver_mb=2048.0,
            global_driver_used_mb=1024.0,
        )

        report = wait_for_vram_release(
            before,
            gpu_release_expected=True,
            untracked_gpu_resource_count=1,
            driver_baseline=baseline,
            timeout_sec=0.0,
            sampler=lambda: other_process_only_drop,
        )

        self.assertFalse(report["observed"])
        self.assertEqual(report["status"], "timeout")

    def test_process_driver_gate_uses_selected_non_primary_gpu_uuid(self) -> None:
        baseline = _snapshot(
            process_available=False,
            gpu_uuid="GPU-remapped",
            process_driver_mb=0.0,
            global_driver_used_mb=9000.0,
        )
        before = _snapshot(
            process_available=False,
            gpu_uuid="GPU-remapped",
            process_driver_mb=2048.0,
            global_driver_used_mb=9000.0,
        )
        after = _snapshot(
            process_available=False,
            gpu_uuid="GPU-remapped",
            process_driver_mb=0.0,
            global_driver_used_mb=9000.0,
        )

        report = wait_for_vram_release(
            before,
            gpu_release_expected=True,
            untracked_gpu_resource_count=1,
            driver_baseline=baseline,
            timeout_sec=0.0,
            sampler=lambda: after,
        )

        self.assertTrue(report["observed"])
        self.assertEqual(report["driver_gpu_uuid"], "GPU-remapped")

    def test_expected_release_without_attributed_metrics_fails_closed(self) -> None:
        unavailable = _snapshot(
            process_available=False,
            driver_query_available=False,
        )

        report = wait_for_vram_release(
            unavailable,
            gpu_release_expected=True,
            untracked_gpu_resource_count=1,
            driver_baseline=unavailable,
            timeout_sec=0.0,
            sampler=lambda: unavailable,
        )

        self.assertTrue(report["required"])
        self.assertFalse(report["observed"])
        self.assertEqual(report["status"], "unavailable")
        self.assertEqual(report["evidence_source"], "unavailable")

    def test_release_without_provenance_fails_closed(self) -> None:
        available = _snapshot(process_available=True, allocated_mb=1024.0)

        report = wait_for_vram_release(
            available,
            gpu_release_expected=True,
            timeout_sec=0.0,
            sampler=lambda: available,
        )

        self.assertFalse(report["observed"])
        self.assertEqual(report["status"], "unavailable")

    def test_no_expected_gpu_release_is_not_required(self) -> None:
        unavailable = _snapshot(
            process_available=False,
            driver_query_available=False,
        )

        report = wait_for_vram_release(
            unavailable,
            gpu_release_expected=False,
            timeout_sec=0.0,
            sampler=lambda: unavailable,
        )

        self.assertFalse(report["required"])
        self.assertTrue(report["observed"])
        self.assertEqual(report["status"], "not-required")

    def test_global_driver_release_returns_to_model_start_baseline(self) -> None:
        baseline = _snapshot(
            process_available=False,
            global_driver_used_mb=512.0,
        )
        before = _snapshot(
            process_available=False,
            global_driver_used_mb=3072.0,
        )
        after = _snapshot(
            process_available=False,
            global_driver_used_mb=520.0,
        )

        report = wait_for_global_vram_release(
            before,
            gpu_release_expected=True,
            driver_baseline=baseline,
            timeout_sec=0.0,
            sampler=lambda: after,
        )

        self.assertTrue(report["observed"])
        self.assertEqual(report["status"], "observed")
        self.assertEqual(report["evidence_source"], "driver-global-memory")
        self.assertEqual(report["drop_mb"], 2552.0)

    def test_global_driver_release_fails_without_baseline_return(self) -> None:
        baseline = _snapshot(
            process_available=False,
            global_driver_used_mb=512.0,
        )
        before = _snapshot(
            process_available=False,
            global_driver_used_mb=3072.0,
        )
        after = _snapshot(
            process_available=False,
            global_driver_used_mb=1536.0,
        )

        report = wait_for_global_vram_release(
            before,
            gpu_release_expected=True,
            driver_baseline=baseline,
            timeout_sec=0.0,
            sampler=lambda: after,
        )

        self.assertFalse(report["observed"])
        self.assertEqual(report["status"], "timeout")

    def test_sleeping_managed_runtime_allows_only_bounded_process_residue(self) -> None:
        baseline = _snapshot(
            process_available=False,
            global_driver_used_mb=512.0,
        )
        before = _snapshot(
            process_available=False,
            global_driver_used_mb=3072.0,
        )
        after = _snapshot(
            process_available=False,
            global_driver_used_mb=768.0,
        )

        report = wait_for_global_vram_release(
            before,
            gpu_release_expected=True,
            driver_baseline=baseline,
            expected_drop_ratio=0.85,
            residual_allowance_mb=512.0,
            timeout_sec=0.0,
            sampler=lambda: after,
        )

        self.assertTrue(report["observed"])
        self.assertEqual(report["expected_drop_mb"], 2176.0)
        self.assertEqual(report["residual_allowance_mb"], 512.0)

    def test_sleeping_managed_runtime_rejects_partial_model_residency(self) -> None:
        baseline = _snapshot(
            process_available=False,
            global_driver_used_mb=512.0,
        )
        before = _snapshot(
            process_available=False,
            global_driver_used_mb=3072.0,
        )
        after = _snapshot(
            process_available=False,
            global_driver_used_mb=900.0,
        )

        report = wait_for_global_vram_release(
            before,
            gpu_release_expected=True,
            driver_baseline=baseline,
            expected_drop_ratio=0.85,
            residual_allowance_mb=512.0,
            timeout_sec=0.0,
            sampler=lambda: after,
        )

        self.assertFalse(report["observed"])
        self.assertEqual(report["status"], "timeout")

    def test_global_driver_release_fails_closed_when_measurement_missing(self) -> None:
        unavailable = {
            "driver": {"available": False},
        }

        report = wait_for_global_vram_release(
            unavailable,
            gpu_release_expected=True,
            timeout_sec=0.0,
            sampler=lambda: unavailable,
        )

        self.assertFalse(report["observed"])
        self.assertEqual(report["status"], "unavailable")


if __name__ == "__main__":
    unittest.main()

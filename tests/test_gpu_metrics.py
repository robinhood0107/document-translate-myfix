from __future__ import annotations

import os
import sys
import unittest
from types import SimpleNamespace
from unittest import mock

from modules.utils import gpu_metrics
from modules.utils.gpu_handoff import cleanup_python_cuda_memory


class GPUMetricsTests(unittest.TestCase):
    def test_parse_linux_swap_meminfo_reports_used_swap(self) -> None:
        report = gpu_metrics._parse_linux_swap_meminfo(
            "SwapTotal:       2097152 kB\n"
            "SwapFree:        1572864 kB\n"
            "SwapCached:       131072 kB\n",
            source="test-wsl",
        )

        self.assertTrue(report["available"])
        self.assertEqual(report["source"], "test-wsl")
        self.assertEqual(report["swap_total_mb"], 2048.0)
        self.assertEqual(report["swap_free_mb"], 1536.0)
        self.assertEqual(report["swap_used_mb"], 512.0)

    def test_windows_wsl_swap_query_is_fail_open(self) -> None:
        with mock.patch(
            "modules.utils.gpu_metrics._current_process_is_wsl",
            return_value=False,
        ), mock.patch(
            "modules.utils.gpu_metrics.os.name",
            "nt",
        ), mock.patch(
            "modules.utils.gpu_metrics._run_capture_status",
            return_value=(False, ""),
        ):
            report = gpu_metrics.query_wsl_swap_metrics()

        self.assertFalse(report["available"])
        self.assertEqual(report["reason"], "docker-desktop-wsl-query-failed")

    def test_process_driver_metrics_select_current_pid_and_exact_uuid(self) -> None:
        output = "\n".join(
            [
                f"{os.getpid()}, GPU-target, 2048",
                f"{os.getpid()}, GPU-other, 512",
                "99999, GPU-target, 4096",
            ]
        )
        with mock.patch(
            "modules.utils.gpu_metrics._run_capture_status",
            return_value=(True, output),
        ):
            report = gpu_metrics.query_process_driver_gpu_metrics(
                preferred_gpu_uuid="GPU-target"
            )

        self.assertTrue(report["query_available"])
        self.assertTrue(report["available"])
        self.assertEqual(report["selected"]["gpu_uuid"], "GPU-target")
        self.assertEqual(report["selected"]["memory_used_mb"], 2048.0)
        self.assertEqual(len(report["rows"]), 2)

    def test_process_driver_metrics_reject_ambiguous_gpu_without_identity(self) -> None:
        output = "\n".join(
            [
                f"{os.getpid()}, GPU-a, 2048",
                f"{os.getpid()}, GPU-b, 512",
            ]
        )
        with mock.patch(
            "modules.utils.gpu_metrics._run_capture_status",
            return_value=(True, output),
        ):
            report = gpu_metrics.query_process_driver_gpu_metrics()

        self.assertFalse(report["available"])
        self.assertEqual(report["reason"], "ambiguous-process-gpu")

    def test_process_cuda_metrics_include_device_uuid_and_allocator_values(self) -> None:
        cuda = SimpleNamespace(
            is_available=lambda: True,
            current_device=lambda: 1,
            get_device_properties=lambda _index: SimpleNamespace(uuid="GPU-target"),
            memory_allocated=lambda _index: 512 * 1024 * 1024,
            memory_reserved=lambda _index: 768 * 1024 * 1024,
            max_memory_allocated=lambda _index: 1024 * 1024 * 1024,
            max_memory_reserved=lambda _index: 1280 * 1024 * 1024,
        )
        with mock.patch.dict(
            sys.modules,
            {"torch": SimpleNamespace(cuda=cuda)},
        ):
            report = gpu_metrics.query_process_cuda_metrics()

        self.assertTrue(report["available"])
        self.assertEqual(report["device_index"], 1)
        self.assertEqual(report["device_uuid"], "GPU-target")
        self.assertEqual(report["allocated_mb"], 512.0)
        self.assertEqual(report["reserved_mb"], 768.0)

    def test_cuda_handoff_snapshot_passes_torch_uuid_to_driver_query(self) -> None:
        process = {
            "available": True,
            "device_index": 0,
            "device_uuid": "GPU-target",
        }
        with mock.patch(
            "modules.utils.gpu_metrics.query_process_cuda_metrics",
            return_value=process,
        ), mock.patch(
            "modules.utils.gpu_metrics.query_gpu_metrics",
            return_value={"available": True},
        ), mock.patch(
            "modules.utils.gpu_metrics.query_process_driver_gpu_metrics",
            return_value={"available": True},
        ) as driver_process:
            report = gpu_metrics.query_cuda_handoff_metrics()

        self.assertEqual(report["process"], process)
        driver_process.assert_called_once_with(preferred_gpu_uuid="GPU-target")


class GPUCleanupTests(unittest.TestCase):
    def test_cleanup_disabled_does_not_touch_cuda(self) -> None:
        cuda = mock.MagicMock()
        with mock.patch.dict(
            sys.modules,
            {"torch": SimpleNamespace(cuda=cuda)},
        ), mock.patch(
            "modules.utils.gpu_handoff.gc.collect",
            return_value=3,
        ):
            report = cleanup_python_cuda_memory(release_cuda_allocator=False)

        self.assertEqual(report["gc_collected"], 3)
        cuda.is_available.assert_not_called()

    def test_cleanup_synchronizes_and_releases_cuda_allocator(self) -> None:
        cuda = SimpleNamespace(
            is_available=mock.Mock(return_value=True),
            synchronize=mock.Mock(),
            empty_cache=mock.Mock(),
            ipc_collect=mock.Mock(),
        )
        with mock.patch.dict(
            sys.modules,
            {"torch": SimpleNamespace(cuda=cuda)},
        ), mock.patch(
            "modules.utils.gpu_handoff.gc.collect",
            side_effect=[2, 4],
        ):
            report = cleanup_python_cuda_memory()

        self.assertEqual(report["gc_collected"], 6)
        self.assertTrue(report["cuda_synchronized"])
        self.assertTrue(report["cuda_cache_emptied"])
        self.assertTrue(report["cuda_ipc_collected"])
        cuda.synchronize.assert_called_once_with()
        cuda.empty_cache.assert_called_once_with()
        cuda.ipc_collect.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()

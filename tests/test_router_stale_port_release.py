from __future__ import annotations

import os
import unittest
from typing import Any

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from modules.utils.local_llama_router import (  # noqa: E402
    LocalLlamaRouterCoordinator,
    RouterOwnershipError,
    RouterPairKind,
    RouterStateError,
)
from modules.utils.local_llama_router.contracts import (  # noqa: E402
    router_pair_for_engine_key,
)


class _PortAdapter:
    """Adapter double that only models host-port ownership for a Router pair."""

    def __init__(self, *, occupants: dict[int, str], foreign_ports: frozenset[int] = frozenset()) -> None:
        self.occupants = dict(occupants)
        self.foreign_ports = frozenset(foreign_ports)
        self.stopped: list[str] = []

    def stop_owned_pair_ports(
        self,
        pair,
        *,
        cancel_checker=None,
        reject_foreign: bool = True,
        require_ports_free: bool = True,
    ) -> tuple[str, ...]:
        ports = (pair.ocr_port, pair.gemma_port)
        self.last_flags = (reject_foreign, require_ports_free)
        for port in ports:
            if port in self.foreign_ports:
                if reject_foreign:
                    raise RouterOwnershipError(
                        "Router port is held by a foreign container; it will not be stopped."
                    )
                return ()
        released: list[str] = []
        for port in ports:
            name = self.occupants.pop(port, "")
            if name and name not in released:
                released.append(name)
                self.stopped.append(name)
        return tuple(released)


class RouterPairLookupTests(unittest.TestCase):
    def test_pair_is_resolvable_from_the_engine_key_alone(self) -> None:
        crop = router_pair_for_engine_key("PaddleOCR VL")
        spotting = router_pair_for_engine_key("PaddleOCR VL Spotting")
        self.assertIsNotNone(crop)
        self.assertIsNotNone(spotting)
        self.assertEqual(crop.kind, RouterPairKind.CROP)
        self.assertEqual(crop.ocr_port, 18000)
        self.assertEqual(spotting.kind, RouterPairKind.SPOTTING)
        self.assertEqual(spotting.ocr_port, 18002)
        self.assertEqual(crop.gemma_port, 18080)

    def test_unknown_engine_key_has_no_pair(self) -> None:
        self.assertIsNone(router_pair_for_engine_key("HunyuanOCR"))
        self.assertIsNone(router_pair_for_engine_key(""))
        self.assertIsNone(router_pair_for_engine_key(None))


class ReleaseOwnedPairPortsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.pair = router_pair_for_engine_key("PaddleOCR VL")

    def test_leftover_owned_router_is_released_for_the_separate_server_path(self) -> None:
        """A Router container from a previous run must not keep 18000/18080."""

        adapter = _PortAdapter(
            occupants={18000: "comic-translate-router-crop-v2", 18080: "comic-translate-router-crop-v2"}
        )
        coordinator = LocalLlamaRouterCoordinator(adapter=adapter)
        released = coordinator.release_owned_pair_ports(self.pair)
        self.assertEqual(tuple(released), ("comic-translate-router-crop-v2",))
        self.assertEqual(adapter.stopped, ["comic-translate-router-crop-v2"])
        self.assertEqual(adapter.occupants, {})

    def test_no_leftover_container_is_a_no_op(self) -> None:
        adapter = _PortAdapter(occupants={})
        coordinator = LocalLlamaRouterCoordinator(adapter=adapter)
        self.assertEqual(tuple(coordinator.release_owned_pair_ports(self.pair)), ())
        self.assertEqual(adapter.stopped, [])

    def test_non_router_port_holder_is_left_alone_without_an_error(self) -> None:
        """The separate-server container about to be reused must survive."""

        adapter = _PortAdapter(
            occupants={18000: "paddleocr-llamacpp"}, foreign_ports=frozenset({18000})
        )
        coordinator = LocalLlamaRouterCoordinator(adapter=adapter)
        self.assertEqual(tuple(coordinator.release_owned_pair_ports(self.pair)), ())
        self.assertEqual(adapter.stopped, [])
        self.assertEqual(adapter.last_flags, (False, False))

    def test_release_is_refused_while_this_coordinator_owns_a_container(self) -> None:
        """Reclaiming ports behind the live state machine would corrupt it."""

        adapter = _PortAdapter(occupants={18000: "comic-translate-router-crop-v2"})
        coordinator = LocalLlamaRouterCoordinator(adapter=adapter)
        coordinator._pair = self.pair
        coordinator._contract = object()
        with self.assertRaises(RouterStateError):
            coordinator.release_owned_pair_ports(self.pair)
        self.assertEqual(adapter.stopped, [])


class _StubCoordinator:
    def __init__(self) -> None:
        self.released: list[Any] = []

    def release_owned_pair_ports(self, pair, *, cancel_checker=None) -> tuple[str, ...]:
        self.released.append(pair)
        return ("comic-translate-router-crop-v2",)


class OcrRuntimeStalePortReleaseTests(unittest.TestCase):
    """The separate-server OCR path must reclaim its own leftover Router port."""

    def _manager(self):
        from modules.ocr.local_runtime import LocalOCRRuntimeManager

        coordinator = _StubCoordinator()
        manager = LocalOCRRuntimeManager(router_coordinator=coordinator)
        return manager, coordinator

    def test_default_paddle_url_releases_the_leftover_router_port(self) -> None:
        manager, coordinator = self._manager()
        manager._release_stale_router_ports_for_engine(
            "PaddleOCR VL",
            "http://127.0.0.1:18000/v1/chat/completions",
            cancel_checker=None,
        )
        self.assertEqual(len(coordinator.released), 1)
        self.assertEqual(coordinator.released[0].ocr_port, 18000)

    def test_custom_port_never_touches_the_router_container(self) -> None:
        manager, coordinator = self._manager()
        manager._release_stale_router_ports_for_engine(
            "PaddleOCR VL",
            "http://127.0.0.1:19999/v1/chat/completions",
            cancel_checker=None,
        )
        self.assertEqual(coordinator.released, [])

    def test_non_router_engine_never_touches_the_router_container(self) -> None:
        manager, coordinator = self._manager()
        manager._release_stale_router_ports_for_engine(
            "HunyuanOCR",
            "http://127.0.0.1:18000/v1/chat/completions",
            cancel_checker=None,
        )
        self.assertEqual(coordinator.released, [])


if __name__ == "__main__":
    unittest.main()

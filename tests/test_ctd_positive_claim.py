from __future__ import annotations

from threading import Event, Thread
from types import SimpleNamespace

import numpy as np
try:
    import pytest
except ModuleNotFoundError as exc:  # unittest discovery in product venvs
    import unittest

    raise unittest.SkipTest("pytest-only test module") from exc

from modules.masking import ctd_positive_claim
from modules.utils.download import ModelDownloader, ModelID


class _FakeSession:
    def __init__(self, providers=("CUDAExecutionProvider", "CPUExecutionProvider")):
        self._providers = list(providers)

    def get_providers(self):
        return list(self._providers)

    def get_inputs(self):
        return [SimpleNamespace(name="image", shape=[1, 3, 1280, 1280])]

    def get_outputs(self):
        return [SimpleNamespace(name="det"), SimpleNamespace(name="seg")]

    def run(self, _names, _inputs):
        return (
            np.zeros((1, 2, 8, 8), dtype=np.float32),
            np.zeros((1, 1, 8, 8), dtype=np.float32),
        )


def test_positive_claim_provider_fails_closed_on_provider_fallback(monkeypatch) -> None:
    monkeypatch.setattr(
        ctd_positive_claim,
        "make_session",
        lambda *_args, **_kwargs: _FakeSession(("CPUExecutionProvider",)),
    )
    ctd_positive_claim._SESSION_CACHE.clear()

    with pytest.raises(RuntimeError, match="primary_provider_not_honored"):
        ctd_positive_claim._cached_session(
            "model.onnx",
            ["CUDAExecutionProvider", "CPUExecutionProvider"],
            1280,
        )


def test_release_positive_claim_cache_reports_cpu_and_cuda_sessions() -> None:
    ctd_positive_claim._SESSION_CACHE.clear()
    cuda_session = _FakeSession(
        ("CUDAExecutionProvider", "CPUExecutionProvider")
    )
    cpu_session = _FakeSession(("CPUExecutionProvider",))
    ctd_positive_claim._SESSION_CACHE.update(
        {
            ("cuda.onnx", ("CUDAExecutionProvider",), 1280): cuda_session,
            ("cpu.onnx", ("CPUExecutionProvider",), 1280): cpu_session,
        }
    )

    report = ctd_positive_claim.release_ctd_positive_claim_cache()

    assert report == {
        "cache_entry_count": 2,
        "cuda_session_count": 1,
        "cpu_session_count": 1,
        "unknown_session_count": 0,
        "expected_process_reclaim_mb": 0.0,
        "untracked_gpu_resource_count": 1,
        "gpu_release_expected": True,
    }
    assert ctd_positive_claim._SESSION_CACHE == {}


def test_release_positive_claim_cache_fails_closed_for_unknown_session() -> None:
    class _UnknownSession:
        def get_providers(self):
            raise RuntimeError("provider query failed")

    ctd_positive_claim._SESSION_CACHE.clear()
    ctd_positive_claim._SESSION_CACHE[
        ("unknown.onnx", ("CUDAExecutionProvider",), 1280)
    ] = _UnknownSession()

    report = ctd_positive_claim.release_ctd_positive_claim_cache()

    assert report["unknown_session_count"] == 1
    assert report["untracked_gpu_resource_count"] == 1
    assert report["gpu_release_expected"] is True
    assert ctd_positive_claim._SESSION_CACHE == {}


def test_release_positive_claim_cache_serializes_with_session_creation(
    monkeypatch,
) -> None:
    ctd_positive_claim._SESSION_CACHE.clear()
    creation_started = Event()
    allow_creation = Event()
    release_started = Event()
    release_finished = Event()
    session = _FakeSession()
    result: dict[str, object] = {}

    def make_blocked_session(*_args, **_kwargs):
        creation_started.set()
        assert allow_creation.wait(timeout=5.0)
        return session

    def release_cache() -> None:
        release_started.set()
        result["release"] = (
            ctd_positive_claim.release_ctd_positive_claim_cache()
        )
        release_finished.set()

    monkeypatch.setattr(ctd_positive_claim, "make_session", make_blocked_session)

    creator = Thread(
        target=lambda: result.setdefault(
            "session",
            ctd_positive_claim._cached_session(
                "model.onnx",
                ["CUDAExecutionProvider", "CPUExecutionProvider"],
                1280,
            ),
        )
    )
    releaser = Thread(target=release_cache)

    creator.start()
    assert creation_started.wait(timeout=5.0)
    releaser.start()
    assert release_started.wait(timeout=5.0)
    assert not release_finished.wait(timeout=0.05)
    allow_creation.set()
    creator.join(timeout=5.0)
    releaser.join(timeout=5.0)

    assert not creator.is_alive()
    assert not releaser.is_alive()
    assert result["session"] is session
    assert result["release"]["cache_entry_count"] == 1
    assert ctd_positive_claim._SESSION_CACHE == {}


def test_positive_claim_provider_returns_binary_full_page_mask(monkeypatch) -> None:
    session = _FakeSession()
    monkeypatch.setattr(
        ctd_positive_claim.ModelDownloader,
        "primary_path",
        lambda *_args, **_kwargs: "model.onnx",
    )
    monkeypatch.setattr(
        ctd_positive_claim,
        "_cached_session",
        lambda *_args, **_kwargs: session,
    )
    mask = np.zeros((1, 1, 8, 8), dtype=np.float32)
    mask[:, :, 2:6, 3:7] = 1.0
    monkeypatch.setattr(
        ctd_positive_claim,
        "_det_rearrange_forward",
        lambda *_args, **_kwargs: (np.ones((1,), dtype=np.float32), mask),
    )
    provider = ctd_positive_claim.CTDPositiveClaimProvider(
        device="cuda",
        detect_size=1280,
    )

    result = provider.infer(np.zeros((16, 20, 3), dtype=np.uint8))

    assert result.raw_mask.shape == (16, 20)
    assert set(np.unique(result.raw_mask)).issubset({0, 255})
    assert np.count_nonzero(result.raw_mask) > 0
    assert result.providers[0] == "CUDAExecutionProvider"
    assert result.model_sha256 == ModelDownloader.registry[
        ModelID.CTD_POSITIVE_CLAIM_ONNX
    ].sha256[0]
    assert result.model_opset == 12


def test_positive_claim_model_registry_is_pinned_to_managed_release() -> None:
    spec = ModelDownloader.registry[ModelID.CTD_POSITIVE_CLAIM_ONNX]

    assert spec.url == (
        "https://github.com/robinhood0107/document-translate-myfix/"
        "releases/download/inpaint-ctd-1280-v1/"
    )
    assert spec.files == ["comictextdetector-1280.onnx"]
    assert spec.sha256 == [
        "c954820c56e611a0470bf9cc119c4a5ffa73c1c15bbdb028c8cd1b58cb008277"
    ]

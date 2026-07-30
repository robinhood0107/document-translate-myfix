from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import zitspp_feasibility_client as client  # noqa: E402
import zitspp_feasibility_adapter as adapter  # noqa: E402


def _write_bytes(path: Path, value: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value)
    return hashlib.sha256(value).hexdigest()


def test_external_contract_requires_exact_commit_and_hashes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    (source / ".git").mkdir(parents=True)
    license_sha = _write_bytes(
        source / "LICENSE",
        b"Apache License\\nVersion 2.0\\n",
    )
    model = tmp_path / "model_512" / "models" / "last.ckpt"
    lsm = tmp_path / "lsm.pth"
    config_sha = _write_bytes(
        tmp_path / "model_512" / "config.yml",
        b"fp16: false\\n",
    )
    model_sha = _write_bytes(model, b"model")
    lsm_sha = _write_bytes(lsm, b"lsm")
    monkeypatch.setattr(client, "MODEL_SHA256", model_sha)
    monkeypatch.setattr(client, "LSM_SHA256", lsm_sha)
    monkeypatch.setattr(client, "CONFIG_SHA256", config_sha)
    monkeypatch.setattr(client, "LICENSE_SHA256", license_sha)

    def fake_run(command: list[str], *, cwd: Path | None = None) -> str:
        assert cwd == source.resolve()
        if command[-2:] == ["rev-parse", "HEAD"]:
            return client.SOURCE_COMMIT
        if command[-2:] == ["status", "--porcelain"]:
            return ""
        raise AssertionError(command)

    monkeypatch.setattr(client, "_run_text", fake_run)

    contract = client.validate_external_contract(
        source_root=source,
        model_checkpoint=model,
        lsm_checkpoint=lsm,
    )

    assert contract["source_commit"] == client.SOURCE_COMMIT
    assert contract["model_sha256"] == model_sha
    assert contract["lsm_sha256"] == lsm_sha
    assert contract["license"] == "Apache-2.0"


def test_external_contract_rejects_dirty_source(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    (source / ".git").mkdir(parents=True)
    license_sha = _write_bytes(
        source / "LICENSE",
        b"Apache License\\nVersion 2.0\\n",
    )
    model = tmp_path / "model_512" / "models" / "last.ckpt"
    lsm = tmp_path / "lsm.pth"
    config_sha = _write_bytes(
        tmp_path / "model_512" / "config.yml",
        b"fp16: false\\n",
    )
    monkeypatch.setattr(
        client,
        "MODEL_SHA256",
        _write_bytes(model, b"model"),
    )
    monkeypatch.setattr(
        client,
        "LSM_SHA256",
        _write_bytes(lsm, b"lsm"),
    )
    monkeypatch.setattr(client, "CONFIG_SHA256", config_sha)
    monkeypatch.setattr(client, "LICENSE_SHA256", license_sha)

    def fake_run(command: list[str], *, cwd: Path | None = None) -> str:
        if command[-2:] == ["rev-parse", "HEAD"]:
            return client.SOURCE_COMMIT
        return " M test.py"

    monkeypatch.setattr(client, "_run_text", fake_run)

    with pytest.raises(
        client.ZITSConfigurationError,
        match="checkout is dirty",
    ):
        client.validate_external_contract(
            source_root=source,
            model_checkpoint=model,
            lsm_checkpoint=lsm,
        )


def test_docker_image_must_be_digest_pinned(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    (source / ".git").mkdir(parents=True)
    model = tmp_path / "model.ckpt"
    lsm = tmp_path / "lsm.pth"
    model.write_bytes(b"model")
    lsm.write_bytes(b"lsm")
    monkeypatch.setattr(
        client,
        "validate_external_contract",
        lambda **_kwargs: {},
    )

    with pytest.raises(
        client.ZITSConfigurationError,
        match="digest-pinned",
    ):
        client.ZITSPlusPlusDockerInpainter(
            source_root=source,
            model_checkpoint=model,
            lsm_checkpoint=lsm,
            docker_image="example/zits:latest",
        )


def test_release_calls_adapter_close_before_discard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import benchmark_inpaint_quality_gates as gates

    class FakeInpainter:
        def __init__(self) -> None:
            self.closed = False
            self.model = object()

        def close(self) -> None:
            self.closed = True

    inpainter = FakeInpainter()
    gates._release_inpainter(inpainter)

    assert inpainter.closed is True
    assert inpainter.model is None


def test_adapter_exchange_paths_reject_empty_and_traversal(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="empty"):
        adapter._safe_exchange_path(
            tmp_path,
            "",
            must_exist=False,
        )
    with pytest.raises(ValueError, match="escapes"):
        adapter._safe_exchange_path(
            tmp_path,
            "../escape.png",
            must_exist=False,
        )

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "prepare_hunyuanocr_router_lab_volume.py"
SPEC = importlib.util.spec_from_file_location("prepare_hunyuanocr_router_lab_volume", MODULE_PATH)
assert SPEC and SPEC.loader
volume = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = volume
SPEC.loader.exec_module(volume)


def test_source_manifest_rejects_missing_or_wrong_identity(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        volume,
        "FILES",
        {"model.gguf": {"bytes": 2, "sha256": "d" * 64}},
    )
    with pytest.raises(volume.HunyuanVolumeError):
        volume._source_manifest(tmp_path)

    source = tmp_path / "model.gguf"
    source.write_bytes(b"ok")
    with pytest.raises(volume.HunyuanVolumeError):
        volume._source_manifest(tmp_path)


def test_prepare_never_uses_volume_remove_command() -> None:
    source = MODULE_PATH.read_text(encoding="utf-8")
    assert '"volume", "rm"' not in source
    assert '"rm", "-rf"' not in source

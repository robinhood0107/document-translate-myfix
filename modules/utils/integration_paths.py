from __future__ import annotations

from pathlib import Path


def get_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def get_integration_root() -> Path:
    return get_repo_root() / "이식"


def get_integration_docs_root() -> Path:
    return get_integration_root() / "docs"


def get_integration_models_root() -> Path:
    return get_integration_root() / "models"


def get_ctd_torch_path() -> Path:
    return get_integration_models_root() / "comictextdetector.pt"


def get_ctd_onnx_path() -> Path:
    return get_integration_models_root() / "comictextdetector.pt.onnx"


def get_lama_large_512px_ckpt_path() -> Path:
    return get_integration_models_root() / "lama_large_512px.ckpt"


def get_lama_mpe_ckpt_path() -> Path:
    return get_integration_models_root() / "lama_mpe.ckpt"
